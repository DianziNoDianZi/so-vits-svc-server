"""So-VITS-SVC 独立推理服务"""
import gc
import hmac
import json
import os
import re
import shutil
import sys
import uuid
import secrets
import string
import signal
import queue as queue_module
import multiprocessing as mp_module
import sqlalchemy as sa
import threading
import time
import traceback
from collections import deque
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, session, Response,
)
from flask_login import (
    login_user, logout_user, login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from extensions import db, login_manager, init_sqlite_pragmas
from db_models import User, Model, InferenceConfig, Task, TrainingTask, DEFAULT_PARAMS

# ─── 项目路径 ───
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def generate_random_password(length=12):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def migrate_db():
    import sqlalchemy as sa
    inspector = sa.inspect(db.engine)
    cols = [c['name'] for c in inspector.get_columns('training_task')]
    train_cols = [('params_json', 'TEXT'), ('diff_model_path', 'VARCHAR(500)'),
                  ('config_path', 'VARCHAR(500)'), ('diff_config_path', 'VARCHAR(500)'),
                  ('resume_from_id', 'INTEGER'),
                  ('anomaly_token', 'VARCHAR(64)'), ('anomaly_state', 'VARCHAR(20)')]
    for col, typ in train_cols:
        if col not in cols:
            db.session.execute(sa.text(f'ALTER TABLE training_task ADD COLUMN {col} {typ}'))
    # migration for inference task (覆盖参数)
    tcols = [c['name'] for c in inspector.get_columns('task')]
    if 'params_json' not in tcols:
        db.session.execute(sa.text('ALTER TABLE task ADD COLUMN params_json TEXT'))
    # migration for user table
    ucols = [c['name'] for c in inspector.get_columns('user')]
    user_cols = [('email', 'VARCHAR(200)'), ('email_notify', 'BOOLEAN'),
                 ('smtp_user', 'VARCHAR(200)'), ('smtp_pwd', 'VARCHAR(200)'),
                 ('smtp_host', 'VARCHAR(200)'), ('smtp_port', 'INTEGER'),
                 ('report_interval', 'INTEGER'), ('infer_notify', 'BOOLEAN')]
    for col, typ in user_cols:
        if col not in ucols:
            db.session.execute(sa.text(f'ALTER TABLE user ADD COLUMN {col} {typ}'))
    db.session.commit()


def init_admin():
    """首次启动创建管理员并生成随机密码；已存在账号绝不重置密码。"""
    admin = User.query.filter_by(username='admin').first()
    if admin:
        return False, None
    password = generate_random_password()
    admin = User(
        username='admin',
        password_hash=generate_password_hash(password),
    )
    db.session.add(admin)
    db.session.commit()
    return True, password


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        init_sqlite_pragmas()
        # 确保上传子目录存在（新部署的 zip 不含 uploads/）
        for _sub in ('models', 'configs', 'audio', 'results', 'dataset_zips', 'train_data'):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], _sub), exist_ok=True)
        db.create_all()
        migrate_db()
        created, pwd = init_admin()
        if created:
            print(f'\n{"="*50}')
            print(f'  首次启动！管理员账号已创建')
            print(f'  用户名: admin')
            print(f'  初始密码:   {pwd}')
            print(f'  登录后将强制修改密码')
            print(f'{"="*50}\n')

    # Custom Jinja2 filter for parsing JSON in templates
    @app.template_filter('from_json')
    def from_json_filter(s):
        try:
            return json.loads(s) if s else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    return app


app = create_app()


# ========== CSRF 防护 ==========

def _csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']


@app.before_request
def _protect_csrf():
    if request.method == 'POST':
        token = session.get('_csrf_token')
        form_token = request.form.get('_csrf_token', '')
        if not token or not form_token or not hmac.compare_digest(token, form_token):
            abort(400, description='CSRF 校验失败，请刷新页面后重试')


@app.context_processor
def _inject_csrf():
    return {'csrf_token': _csrf_token}


# ========== 登录限速（内存级，防止暴力破解） ==========

_login_attempts = {}
_login_lock = threading.Lock()
_LOGIN_MAX_FAILS = 5
_LOGIN_WINDOW = 900  # 秒


def _login_blocked(remote_addr):
    with _login_lock:
        info = _login_attempts.get(remote_addr)
        return bool(info and info.get('blocked_until') and time.time() < info['blocked_until'])


def _record_login_failure(remote_addr):
    with _login_lock:
        now = time.time()
        info = _login_attempts.get(remote_addr)
        if not info or now - info.get('ts', 0) > _LOGIN_WINDOW:
            info = {'count': 0, 'ts': now, 'blocked_until': 0}
        info['count'] = info.get('count', 0) + 1
        info['ts'] = now
        if info['count'] >= _LOGIN_MAX_FAILS:
            info['blocked_until'] = now + _LOGIN_WINDOW
            info['count'] = 0
        _login_attempts[remote_addr] = info


def _clear_login_failures(remote_addr):
    with _login_lock:
        _login_attempts.pop(remote_addr, None)


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


def allowed_file(name):
    return name.lower().endswith(('.pth', '.pt', '.json', '.yaml', '.yml'))


def flash_errors(form_errors):
    for e in form_errors:
        flash(e, 'danger')


def save_uploaded(file, subdir):
    """Save uploaded file to uploads/<subdir>/ with unique name"""
    ext = os.path.splitext(file.filename)[1]
    unique = uuid.uuid4().hex[:12]
    filename = f'{unique}_{secure_filename(file.filename)}'
    dest_dir = os.path.join(app.config['UPLOAD_FOLDER'], subdir)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    file.save(path)
    return filename


def safe_join(base, *parts):
    """拼接路径并校验结果必须位于 base 目录内，防止路径穿越。"""
    base_abs = os.path.abspath(base)
    full = os.path.abspath(os.path.join(base_abs, *parts))
    if not full.startswith(base_abs + os.sep):
        raise ValueError('非法路径')
    return full


def _latest_g_checkpoint(directory):
    """返回目录中步数最大的 G_*.pth 文件名，没有则返回 None。"""
    best, best_n = None, -1
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for f in names:
        if f.startswith('G_') and f.endswith('.pth'):
            n = int(''.join(c for c in f if c.isdigit()) or 0)
            if n > best_n:
                best, best_n = f, n
    return best


def _chain_root_id(task):
    """沿 resume_from 链向上找最原始的任务 id（SoVITS/扩散 checkpoint 的归属目录）。"""
    seen = set()
    root = task.id
    cur = task.resume_from_id
    while cur and cur not in seen:
        seen.add(cur)
        root = cur
        prev = db.session.get(TrainingTask, cur)
        cur = prev.resume_from_id if prev else None
    return root


# ========== 登录/注册 ==========

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        remote = request.remote_addr or 'unknown'
        if _login_blocked(remote):
            flash('尝试过于频繁，请 15 分钟后再试', 'danger')
        else:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                _clear_login_failures(remote)
                login_user(user)
                if 'password_changed' not in session or not session.get('password_changed'):
                    return redirect(url_for('change_password'))
                return redirect(url_for('dashboard'))
            _record_login_failure(remote)
            flash('用户名或密码错误', 'danger')
    return render_template('login.html')


@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        old = request.form.get('old_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        new_username = request.form.get('username', '').strip()

        if not check_password_hash(current_user.password_hash, old):
            flash('当前密码错误', 'danger')
            return render_template('change_password.html')

        if new_username and new_username != current_user.username:
            existing = User.query.filter_by(username=new_username).first()
            if existing:
                flash('用户名已存在', 'danger')
                return render_template('change_password.html')
            current_user.username = new_username

        if new:
            if len(new) < 6:
                flash('新密码至少 6 位', 'danger')
                return render_template('change_password.html')
            if new != confirm:
                flash('两次密码不一致', 'danger')
                return render_template('change_password.html')
            current_user.password_hash = generate_password_hash(new)

        db.session.commit()
        session['password_changed'] = True
        flash('设置已保存', 'success')
        return redirect(url_for('dashboard'))

    return render_template('change_password.html')


@app.route('/clean-cache', methods=['POST'])
@login_required
def clean_cache():
    import shutil, glob as _glob
    clean_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deleted = 0
    msgs = []

    if request.form.get('clean_cache_files'):
        for root, dirs, _ in os.walk(clean_base):
            if '__pycache__' in dirs:
                p = os.path.join(root, '__pycache__')
                shutil.rmtree(p, ignore_errors=True)
                deleted += 1
        for f in _glob.glob(os.path.join(clean_base, '**', '*.pyc'), recursive=True):
            try:
                os.remove(f)
                deleted += 1
            except Exception:
                pass
        msgs.append('Python 缓存')

    if request.form.get('clean_error_logs'):
        now = time.time()
        for f in _glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'error_*.log')):
            try:
                age = now - os.path.getmtime(f)
                if age > 86400 * 3:
                    os.remove(f)
                    deleted += 1
            except Exception:
                pass
        msgs.append('错误日志')

    if request.form.get('clean_old_tasks'):
        cutoff = datetime.utcnow().timestamp() - 86400 * 7
        old_tasks = Task.query.filter(
            Task.user_id == current_user.id,
            Task.status.in_(['done', 'failed']),
            Task.done_at.isnot(None),
            Task.done_at < datetime.utcfromtimestamp(cutoff),
        ).all()
        for t in old_tasks:
            if t.result_filename:
                rp = os.path.join(app.config['UPLOAD_FOLDER'], 'results', t.result_filename)
                try:
                    os.remove(rp)
                    deleted += 1
                except Exception:
                    pass
            db.session.delete(t)
            deleted += 1
        if old_tasks:
            db.session.commit()
        msgs.append('旧任务')

    if request.form.get('clean_all_tasks'):
        all_done = Task.query.filter(
            Task.user_id == current_user.id,
            Task.status.in_(['done', 'failed']),
        ).all()
        for t in all_done:
            if t.result_filename:
                rp = os.path.join(app.config['UPLOAD_FOLDER'], 'results', t.result_filename)
                try:
                    os.remove(rp)
                    deleted += 1
                except Exception:
                    pass
            db.session.delete(t)
            deleted += 1
        if all_done:
            db.session.commit()
        msgs.append('所有已完成/失败推理任务')

    if request.form.get('clean_train_data'):
        import shutil
        train_data = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data')
        # 保护仍在排队/运行中的任务：不删除它们的 zip 和数据目录
        busy = TrainingTask.query.filter(TrainingTask.status.in_(['pending', 'running'])).all()
        busy_dirs = {f'task_{t.id}' for t in busy}
        busy_zips = {t.dataset_zip for t in busy if t.dataset_zip}
        if os.path.exists(train_data):
            sizes = []
            for d in os.listdir(train_data):
                dp = os.path.join(train_data, d)
                if d in busy_dirs:
                    continue
                if os.path.isdir(dp):
                    sz = sum(f.stat().st_size for f in os.scandir(dp) if f.is_file()) // 1048576
                    sizes.append(f'{d}({sz}MB)')
                    shutil.rmtree(dp, ignore_errors=True)
                    deleted += 1
            msgs.append(f'训练数据({", ".join(sizes)})')
        # 清理留下的 zip
        dz = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_zips')
        if os.path.exists(dz):
            for f in os.listdir(dz):
                if f in busy_zips:
                    continue
                fp = os.path.join(dz, f)
                try:
                    os.remove(fp)
                    deleted += 1
                except Exception:
                    pass
        # 清理失败训练任务
        failed_tasks = TrainingTask.query.filter(
            TrainingTask.user_id == current_user.id,
            TrainingTask.status.in_(['failed', 'done']),
        ).all()
        for t in failed_tasks:
            db.session.delete(t)
            deleted += 1
        db.session.commit()
        msgs.append('训练缓存+失败任务')

    if not msgs:
        flash('没有选择任何清理项', 'warning')
    else:
        flash(f'清理完成：{", ".join(msgs)}，共处理 {deleted} 项', 'success')
    return redirect(url_for('settings'))


def _settings_context():
    """设置页需要的精确清理数据。"""
    train_tasks = TrainingTask.query.filter_by(user_id=current_user.id).order_by(TrainingTask.id.desc()).all()
    models = Model.query.filter_by(user_id=current_user.id).order_by(Model.id.desc()).all()
    dz = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_zips')
    dataset_zips = sorted(os.listdir(dz)) if os.path.isdir(dz) else []
    return {'train_tasks': train_tasks, 'models': models, 'dataset_zips': dataset_zips}


@app.route('/clean/specific', methods=['POST'])
@login_required
def clean_specific():
    """精确清理：按单个训练任务 / 模型 / 数据集，带防误伤检查。"""
    ttype = request.form.get('target_type', '')
    target_id = request.form.get('target_id', type=int)
    action = request.form.get('action', '')
    uid = current_user.id

    if ttype == 'train_task':
        task = db.session.get(TrainingTask, target_id)
        if not task or task.user_id != uid:
            flash('训练任务不存在', 'danger')
            return redirect(url_for('settings'))
        if task.status in ('pending', 'running'):
            flash(f'任务 #{task.id} 正在排队/运行中，清理会破坏它，已取消', 'danger')
            return redirect(url_for('settings'))

        chain_id = task.resume_from_id or task.id
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
        done = []

        if action in ('data_dir', 'all'):
            if os.path.isdir(td):
                shutil.rmtree(td, ignore_errors=True)
                done.append('训练数据目录（该任务将无法续训）')

        if action in ('zip', 'all'):
            if task.dataset_zip:
                busy = TrainingTask.query.filter(
                    TrainingTask.status.in_(['pending', 'running']),
                    TrainingTask.dataset_zip == task.dataset_zip,
                ).count()
                if busy:
                    flash(f'数据集 {task.dataset_zip} 仍被 {busy} 个排队/运行任务使用，已跳过', 'warning')
                else:
                    zp = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_zips', task.dataset_zip)
                    if os.path.exists(zp):
                        os.remove(zp)
                        done.append(f'数据集 {task.dataset_zip}')

        if action in ('model_files', 'all'):
            for attr, sub in (('model_path', 'models'), ('config_path', 'configs'),
                              ('diff_model_path', 'models'), ('diff_config_path', 'configs')):
                p = getattr(task, attr)
                if not p:
                    continue
                ref = Model.query.filter(sa.or_(
                    Model.model_path == p, Model.config_path == p,
                    Model.diff_model_path == p, Model.diff_config_path == p,
                )).first()
                if ref:
                    flash(f'{os.path.basename(p)} 被模型 #{ref.id} 引用，已跳过', 'warning')
                    continue
                fp = os.path.join(app.config['UPLOAD_FOLDER'], sub, os.path.basename(p))
                if os.path.exists(fp):
                    os.remove(fp)
                    done.append(os.path.basename(p))

        if action == 'all':
            task.model_path = task.config_path = task.diff_model_path = task.diff_config_path = None
        db.session.commit()
        if done:
            flash('已清理: ' + ', '.join(done), 'success')
        else:
            flash('没有可清理的内容', 'warning')

    elif ttype == 'model':
        m = db.session.get(Model, target_id)
        if not m or m.user_id != uid:
            flash('模型不存在', 'danger')
            return redirect(url_for('settings'))
        config_ids = [c.id for c in InferenceConfig.query.filter_by(model_id=m.id).all()]
        if config_ids:
            running = Task.query.filter(
                Task.config_id.in_(config_ids),
                Task.status.in_(['pending', 'running']),
            ).count()
            if running:
                flash(f'模型 #{m.id} 正被 {running} 个排队/运行推理任务使用，已取消删除', 'danger')
                return redirect(url_for('settings'))
        InferenceConfig.query.filter_by(model_id=m.id).delete()
        removed = []
        for p in (m.model_path, m.config_path, m.diff_model_path, m.diff_config_path, m.cluster_path):
            if not p:
                continue
            sub = 'models' if p.lower().endswith(('.pth', '.pt')) else 'configs'
            fp = os.path.join(app.config['UPLOAD_FOLDER'], sub, os.path.basename(p))
            if os.path.exists(fp):
                os.remove(fp)
                removed.append(os.path.basename(p))
        db.session.delete(m)
        db.session.commit()
        flash(f'模型 #{m.id} 已删除（含文件与关联推理配置）', 'success')

    elif ttype == 'dataset_zip':
        name = request.form.get('target_name', '').strip()
        zp = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_zips', os.path.basename(name))
        if not name or not os.path.exists(zp):
            flash('数据集不存在', 'danger')
            return redirect(url_for('settings'))
        busy = TrainingTask.query.filter(
            TrainingTask.status.in_(['pending', 'running']),
            TrainingTask.dataset_zip == name,
        ).count()
        if busy:
            flash(f'该数据集仍被 {busy} 个排队/运行任务使用，已取消删除', 'danger')
        else:
            os.remove(zp)
            flash(f'已删除数据集 {name}', 'success')

    return redirect(url_for('settings'))


@app.route('/save-notify', methods=['POST'])
@login_required
def save_notify():
    current_user.email = request.form.get('email', '').strip() or None
    current_user.email_notify = request.form.get('email_notify') == '1'
    current_user.smtp_user = request.form.get('smtp_user', '').strip() or None
    smtp_pwd = request.form.get('smtp_pwd', '').strip()
    if smtp_pwd:
        current_user.smtp_pwd = smtp_pwd
    current_user.smtp_host = request.form.get('smtp_host', '').strip() or None
    try:
        current_user.smtp_port = int(request.form.get('smtp_port', 465) or 465)
    except (ValueError, TypeError):
        current_user.smtp_port = 465
    try:
        current_user.report_interval = int(request.form.get('report_interval', 0) or 0)
    except (ValueError, TypeError):
        current_user.report_interval = 0
    current_user.infer_notify = request.form.get('infer_notify') == '1'
    db.session.commit()
    flash('通知设置已保存', 'success')
    return redirect(url_for('settings'))


@app.route('/test-notify', methods=['POST'])
@login_required
def test_notify():
    from notifier import send as send_mail
    u = current_user
    ok = send_mail(u.email, u.smtp_user, u.smtp_pwd,
                   '[SoVITS] 测试通知', '这是一封测试邮件，通知配置正常！',
                   host=u.smtp_host or None, port=u.smtp_port or None)
    if ok:
        flash('测试邮件已发送，请检查收件箱', 'success')
    else:
        flash('发送失败，请检查邮箱配置', 'danger')
    return redirect(url_for('settings'))


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        old = request.form.get('old_password', '')
        new = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        new_username = request.form.get('username', '').strip()

        if not check_password_hash(current_user.password_hash, old):
            flash('当前密码错误', 'danger')
            return render_template('settings.html', **_settings_context())

        # 推理默认值
        current_user.device_pref = request.form.get('device_pref', 'auto')
        try:
            current_user.memory_limit = float(request.form.get('memory_limit', 0))
        except (ValueError, TypeError):
            current_user.memory_limit = 0

        # 通知设置
        current_user.email = request.form.get('email', '').strip() or None
        current_user.email_notify = request.form.get('email_notify') == '1'
        current_user.smtp_user = request.form.get('smtp_user', '').strip() or None
        smtp_pwd = request.form.get('smtp_pwd', '').strip()
        if smtp_pwd:
            current_user.smtp_pwd = smtp_pwd
        current_user.smtp_host = request.form.get('smtp_host', '').strip() or None
        try:
            current_user.smtp_port = int(request.form.get('smtp_port', 465) or 465)
        except (ValueError, TypeError):
            current_user.smtp_port = 465
        try:
            current_user.report_interval = int(request.form.get('report_interval', 0) or 0)
        except (ValueError, TypeError):
            current_user.report_interval = 0
        current_user.infer_notify = request.form.get('infer_notify') == '1'

        if new_username and new_username != current_user.username:
            existing = User.query.filter_by(username=new_username).first()
            if existing:
                flash('用户名已存在', 'danger')
                return render_template('settings.html', **_settings_context())
            current_user.username = new_username

        if new:
            if len(new) < 6:
                flash('新密码至少 6 位', 'danger')
                return render_template('settings.html', **_settings_context())
            if new != confirm:
                flash('两次密码不一致', 'danger')
                return render_template('settings.html', **_settings_context())
            current_user.password_hash = generate_password_hash(new)

        db.session.commit()
        flash('设置已保存', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html', **_settings_context())


@app.route('/update', methods=['POST'])
@login_required
def update_server():
    """一键更新：git pull + 优雅等待任务结束后重启服务。"""
    repo = request.form.get('repo_url', '').strip()
    import subprocess as _sp
    output = []
    try:
        if repo:
            r = _sp.run(['git', 'pull', repo, 'master'], cwd=PROJECT_DIR,
                        capture_output=True, text=True, timeout=180)
        else:
            r = _sp.run(['git', 'pull'], cwd=PROJECT_DIR,
                        capture_output=True, text=True, timeout=180)
        output.append(r.stdout[-1500:])
        if r.stderr:
            output.append(r.stderr[-800:])
    except Exception as e:
        output.append(f'git pull 失败: {e}')

    running = (TrainingTask.query.filter_by(status='running').count()
               + Task.query.filter_by(status='running').count())

    def _do_restart():
        time.sleep(5)
        if os.name == 'nt':
            return  # Windows 无 systemctl，提示手动重启
        try:
            _sp.Popen(['systemctl', 'restart', 'ssvc'],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        except Exception:
            pass

    def _wait_and_restart():
        while True:
            with app.app_context():
                n = (TrainingTask.query.filter_by(status='running').count()
                     + Task.query.filter_by(status='running').count())
            if n == 0:
                break
            time.sleep(10)
        _do_restart()

    if running > 0:
        threading.Thread(target=_wait_and_restart, daemon=True).start()
        flash(f'更新完成，等待 {running} 个运行中任务结束后自动重启', 'success')
    else:
        threading.Thread(target=_do_restart, daemon=True).start()
        flash('更新完成，5 秒后自动重启服务', 'success')
    if os.name == 'nt':
        flash('当前为 Windows 环境，无法自动重启，请手动重启服务', 'warning')
    flash('git pull 输出：\n' + '\n'.join(output)[-1200:], 'info')
    return redirect(url_for('settings'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ========== 仪表盘 ==========

@app.route('/')
@login_required
def dashboard():
    models = Model.query.filter_by(user_id=current_user.id).all()
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', models=models, configs=configs)


# ========== 模型管理 ==========

@app.route('/models')
@login_required
def model_list():
    models = Model.query.filter_by(user_id=current_user.id).order_by(Model.created_at.desc()).all()
    # 读取每个模型的架构（用于列表筛选）
    model_items = []
    for m in models:
        arch = ''
        if m.config_path:
            cfg_p = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', m.config_path)
            if os.path.exists(cfg_p):
                try:
                    with open(cfg_p, 'r', encoding='utf-8') as f:
                        arch = (json.load(f).get('model') or {}).get('arch', '')
                except Exception:
                    arch = ''
        model_items.append({'m': m, 'arch': arch or 'sovits-v1'})
    return render_template('models_list.html', model_items=model_items)


@app.route('/models/upload', methods=['GET', 'POST'])
@login_required
def model_upload():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        model_file = request.files.get('model_file')
        config_file = request.files.get('config_file')
        diff_file = request.files.get('diff_file')
        diff_config_file = request.files.get('diff_config_file')
        cluster_file = request.files.get('cluster_file')

        if not name:
            flash('请输入模型名称', 'danger')
            return render_template('model_upload.html')
        if not model_file or not model_file.filename:
            flash('请选择模型文件 (.pth)', 'danger')
            return render_template('model_upload.html')
        if not config_file or not config_file.filename:
            flash('请选择配置文件 (.json)', 'danger')
            return render_template('model_upload.html')
        if not allowed_file(model_file.filename) or not allowed_file(config_file.filename):
            flash('文件类型不允许：仅支持 .pth/.pt/.json/.yaml/.yml', 'danger')
            return render_template('model_upload.html')
        if diff_file and diff_file.filename and not allowed_file(diff_file.filename):
            flash('扩散模型文件类型不允许（.pth/.pt）', 'danger')
            return render_template('model_upload.html')
        if diff_config_file and diff_config_file.filename and not allowed_file(diff_config_file.filename):
            flash('扩散配置文件类型不允许（.yaml/.yml）', 'danger')
            return render_template('model_upload.html')
        if cluster_file and cluster_file.filename and not allowed_file(cluster_file.filename):
            flash('聚类模型文件类型不允许（.pth/.pt）', 'danger')
            return render_template('model_upload.html')

        model_path = save_uploaded(model_file, 'models')
        config_path = save_uploaded(config_file, 'configs')
        diff_path = None
        diff_config_path = None
        cluster_path = None
        if diff_file and diff_file.filename:
            diff_path = save_uploaded(diff_file, 'models')
        if diff_config_file and diff_config_file.filename:
            diff_config_path = save_uploaded(diff_config_file, 'configs')
        if cluster_file and cluster_file.filename:
            cluster_path = save_uploaded(cluster_file, 'models')

        m = Model(
            user_id=current_user.id,
            name=name,
            model_path=model_path,
            config_path=config_path,
            diff_model_path=diff_path,
            diff_config_path=diff_config_path,
            cluster_path=cluster_path,
        )
        db.session.add(m)
        db.session.commit()
        flash('模型上传成功', 'success')
        return redirect(url_for('model_list'))
    return render_template('model_upload.html')


@app.route('/models/<int:model_id>/edit', methods=['GET', 'POST'])
@login_required
def model_edit(model_id):
    m = Model.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        abort(403)

    models_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'models')
    configs_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'configs')
    diff_files, cluster_files, diff_configs = [], [], []
    try:
        for f in sorted(os.listdir(models_dir)):
            low = f.lower()
            if low.endswith('.pt'):
                if 'kmeans' in low or 'cluster' in low:
                    cluster_files.append(f)
                else:
                    diff_files.append(f)
    except OSError:
        pass
    try:
        for f in sorted(os.listdir(configs_dir)):
            if f.lower().endswith(('.yaml', '.yml')):
                diff_configs.append(f)
    except OSError:
        pass

    if request.method == 'POST':
        m.name = request.form.get('name', m.name).strip()
        diff_file = request.files.get('diff_file')
        diff_config_file = request.files.get('diff_config_file')
        cluster_file = request.files.get('cluster_file')
        if diff_file and diff_file.filename and not allowed_file(diff_file.filename):
            flash('扩散模型文件类型不允许（.pth/.pt）', 'danger')
            return render_template('model_edit.html', model=m, diff_files=diff_files,
                                   diff_configs=diff_configs, cluster_files=cluster_files)
        if diff_config_file and diff_config_file.filename and not allowed_file(diff_config_file.filename):
            flash('扩散配置文件类型不允许（.yaml/.yml）', 'danger')
            return render_template('model_edit.html', model=m, diff_files=diff_files,
                                   diff_configs=diff_configs, cluster_files=cluster_files)
        if cluster_file and cluster_file.filename and not allowed_file(cluster_file.filename):
            flash('聚类模型文件类型不允许（.pth/.pt）', 'danger')
            return render_template('model_edit.html', model=m, diff_files=diff_files,
                                   diff_configs=diff_configs, cluster_files=cluster_files)

        # 服务器已有文件选择优先于上传
        sel_diff = request.form.get('diff_model_select', '').strip()
        sel_diff_cfg = request.form.get('diff_config_select', '').strip()
        sel_cluster = request.form.get('cluster_select', '').strip()

        if sel_diff == '__clear__':
            m.diff_model_path = None
        elif sel_diff:
            if os.path.exists(os.path.join(models_dir, sel_diff)):
                m.diff_model_path = sel_diff
            else:
                flash('所选扩散模型文件不存在', 'danger')
        elif diff_file and diff_file.filename:
            m.diff_model_path = save_uploaded(diff_file, 'models')

        if sel_diff_cfg == '__clear__':
            m.diff_config_path = None
        elif sel_diff_cfg:
            if os.path.exists(os.path.join(configs_dir, sel_diff_cfg)):
                m.diff_config_path = sel_diff_cfg
            else:
                flash('所选扩散配置文件不存在', 'danger')
        elif diff_config_file and diff_config_file.filename:
            m.diff_config_path = save_uploaded(diff_config_file, 'configs')

        if sel_cluster == '__clear__':
            m.cluster_path = None
        elif sel_cluster:
            if os.path.exists(os.path.join(models_dir, sel_cluster)):
                m.cluster_path = sel_cluster
            else:
                flash('所选聚类模型文件不存在', 'danger')
        elif cluster_file and cluster_file.filename:
            m.cluster_path = save_uploaded(cluster_file, 'models')

        db.session.commit()
        flash('模型已更新', 'success')
        return redirect(url_for('model_list'))
    return render_template('model_edit.html', model=m, diff_files=diff_files,
                           diff_configs=diff_configs, cluster_files=cluster_files)


@app.route('/models/<int:model_id>/delete', methods=['POST'])
@login_required
def model_delete(model_id):
    m = Model.query.get_or_404(model_id)
    if m.user_id != current_user.id:
        abort(403)
    # Delete associated configs
    InferenceConfig.query.filter_by(model_id=m.id).delete()
    # Delete files
    for attr in ['model_path', 'config_path', 'diff_model_path', 'cluster_path']:
        path = getattr(m, attr)
        if path:
            full = os.path.join(app.config['UPLOAD_FOLDER'], 'models' if path.endswith(('.pth', '.pt')) else 'configs',
                                path)
            if os.path.exists(full):
                os.remove(full)
    db.session.delete(m)
    db.session.commit()
    flash('模型已删除', 'success')
    return redirect(url_for('model_list'))


# ========== 推理配置 ==========

@app.route('/configs')
@login_required
def config_list():
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).order_by(
        InferenceConfig.created_at.desc()).all()
    return render_template('configs_list.html', configs=configs)


@app.route('/configs/create', methods=['GET', 'POST'])
@login_required
def config_create():
    models = Model.query.filter_by(user_id=current_user.id).all()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        model_id = request.form.get('model_id', type=int)
        if not name or not model_id:
            flash('请填写名称并选择模型', 'danger')
            return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())

        params = {}
        for key, default in DEFAULT_PARAMS.items():
            val = request.form.get(key)
            if val is not None:
                if isinstance(default, bool):
                    params[key] = val == 'on' or val == '1'
                elif isinstance(default, int):
                    try:
                        params[key] = int(val)
                    except (ValueError, TypeError):
                        params[key] = default
                elif isinstance(default, float):
                    try:
                        params[key] = float(val)
                    except (ValueError, TypeError):
                        params[key] = default
                else:
                    params[key] = val

        c = InferenceConfig(
            user_id=current_user.id,
            model_id=model_id,
            name=name,
            params_json=json.dumps(params, ensure_ascii=False),
        )
        db.session.add(c)
        db.session.commit()
        flash('推理配置已创建', 'success')
        return redirect(url_for('config_list'))
    return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())


@app.route('/configs/<int:config_id>/edit', methods=['GET', 'POST'])
@login_required
def config_edit(config_id):
    c = InferenceConfig.query.get_or_404(config_id)
    if c.user_id != current_user.id:
        abort(403)
    models = Model.query.filter_by(user_id=current_user.id).all()
    if request.method == 'POST':
        c.name = request.form.get('name', c.name).strip()
        model_id = request.form.get('model_id', type=int)
        if model_id:
            m = db.session.get(Model, model_id)
            if not m or m.user_id != current_user.id:
                flash('无效的模型选择', 'danger')
                return render_template('config_create.html', models=models, config=c,
                                       params=json.loads(c.params_json) if c.params_json else DEFAULT_PARAMS.copy())
            c.model_id = model_id

        params = json.loads(c.params_json) if c.params_json else {}
        for key, default in DEFAULT_PARAMS.items():
            val = request.form.get(key)
            if val is not None:
                if isinstance(default, bool):
                    params[key] = val == 'on' or val == '1'
                elif isinstance(default, int):
                    try:
                        params[key] = int(val)
                    except (ValueError, TypeError):
                        params[key] = params.get(key, default)
                elif isinstance(default, float):
                    try:
                        params[key] = float(val)
                    except (ValueError, TypeError):
                        params[key] = params.get(key, default)
                else:
                    params[key] = val
        c.params_json = json.dumps(params, ensure_ascii=False)
        db.session.commit()
        flash('配置已更新', 'success')
        return redirect(url_for('config_list'))

    params = json.loads(c.params_json) if c.params_json else DEFAULT_PARAMS.copy()
    return render_template('config_create.html', models=models, config=c, params=params)


@app.route('/configs/<int:config_id>/delete', methods=['POST'])
@login_required
def config_delete(config_id):
    c = InferenceConfig.query.get_or_404(config_id)
    if c.user_id != current_user.id:
        abort(403)
    db.session.delete(c)
    db.session.commit()
    flash('配置已删除', 'success')
    return redirect(url_for('config_list'))


# ========== 任务队列 ==========

task_queue = queue_module.Queue()
task_worker_started = False
task_processes = {}  # task_id -> subprocess.Popen
task_processes_lock = threading.Lock()

# 推理常驻 daemon（LRU 模型缓存）
inference_q = None
inference_done_q = None
inference_daemon_proc = None
INFERENCE_MODEL_CACHE = int(os.environ.get('INFERENCE_MODEL_CACHE', '3') or 3)


def _stream_lines(stream):
    """把子进程输出按 \r 或 \n 切分成独立行，兼容 tqdm 进度条输出。"""
    buf = b''
    while True:
        try:
            chunk = stream.read1(65536) if hasattr(stream, 'read1') else stream.read(65536)
        except Exception:
            break
        if not chunk:
            break
        buf += chunk
        while True:
            idx = -1
            for sep in (b'\r', b'\n'):
                i = buf.find(sep)
                if i != -1 and (idx == -1 or i < idx):
                    idx = i
            if idx == -1:
                break
            line = buf[:idx]
            buf = buf[idx + 1:]
            if line.strip():
                yield line
    if buf.strip():
        yield buf


def _update_progress(task, msg, force=False):
    """限流写进度，避免高频 commit 造成 SQLite 锁竞争。"""
    now = time.time()
    if not force and now - getattr(_update_progress, '_last', 0) < 1.0:
        return
    _update_progress._last = now
    task.progress_msg = msg
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()


def task_worker():
    """后台任务处理线程"""
    with app.app_context():
        while True:
            task_id = task_queue.get()
            task = db.session.get(Task, task_id)
            if not task:
                continue
            try:
                # 排队期间被用户停止的任务：跳过不投递
                if task.status == 'stopped':
                    continue
                task.status = 'running'
                task.progress_msg = '正在加载模型...'
                db.session.commit()

                cfg_obj = db.session.get(InferenceConfig, task.config_id)
                model = db.session.get(Model, cfg_obj.model_id)
                if task.params_json:
                    params = json.loads(task.params_json)
                else:
                    params = json.loads(cfg_obj.params_json) if cfg_obj.params_json else DEFAULT_PARAMS.copy()
                # 模型没有检索索引时强制关闭 cluster_ratio，避免推理报错
                if params.get('cluster_ratio', 0) > 0 and not model.cluster_path:
                    params['cluster_ratio'] = 0
                    task.progress_msg = '模型未挂载检索索引，cluster_ratio 已置 0'

                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', task.audio_filename)
                k_step = params.get('k_step', 0)

                result_name = f'task_{task_id}_{uuid.uuid4().hex[:8]}.{params.get("output_format", "wav")}'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_name)

                task.progress_msg = '正在推理中...'
                db.session.commit()

                # 用子进程跑推理，跑完进程退出，内存全部释放
                python = sys.executable
                worker = os.path.join(os.path.dirname(__file__), 'inference_worker.py')

                full_model = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model.model_path)
                full_config = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', model.config_path)
                full_diff = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model.diff_model_path) if model.diff_model_path else 'none'
                full_diff_config = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', model.diff_config_path) if model.diff_config_path else 'none'
                full_cluster = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model.cluster_path) if model.cluster_path else 'none'
                # 投递到常驻推理 daemon（LRU 模型缓存）
                if inference_q is None:
                    raise RuntimeError('推理 daemon 未启动')
                payload = {
                    'model_path': full_model,
                    'config_path': full_config,
                    'audio_path': audio_path,
                    'result_path': result_path,
                    'k_step': k_step,
                    'params': params,
                    'diff_path': full_diff,
                    'diff_config_path': full_diff_config,
                    'cluster_path': full_cluster,
                    'device': task.device_pref or 'auto',
                }
                inference_q.put((task_id, payload))

                # 轮询进度文件 + 等待完成通知
                prog_path = result_path + '.prog'
                import soundfile as sf
                try:
                    audio_info = sf.info(audio_path)
                    estimated_total = max(int(audio_info.duration / 10) + 1, 1)
                except Exception:
                    estimated_total = 5
                tail_lines = deque(maxlen=100)
                started_at = time.time()
                task_timeout = int(os.environ.get('INFERENCE_TASK_TIMEOUT', str(6 * 3600)))
                est_total_fixed = estimated_total  # 固定预估总数，不再随实际段数动态扩
                result_ok = False
                result_err = None
                while True:
                    segments_done = 0  # 每次轮询重置，重新累计 prog 中当前段数（否则会持续累加虚涨）
                    # 超时保护
                    if task_timeout > 0 and time.time() - started_at > task_timeout:
                        result_err = f'推理超过 {task_timeout // 3600}h 仍未完成，已终止'
                        break
                    # 完成标记存在 = daemon 已完整写完结果（不依赖 done_q 通知，双保险）
                    if os.path.exists(result_path + '.done'):
                        result_ok = True
                        break
                    # 读取 daemon 进度文件（段级进度）
                    prog_exists = False
                    try:
                        with open(prog_path, 'r', encoding='utf-8', errors='replace') as pf:
                            prog_text = pf.read()
                        prog_exists = True
                        for line in prog_text.splitlines():
                            text = line.strip()
                            if not text:
                                continue
                            tail_lines.append(text)
                            if '#=====segment start' in text:
                                segments_done += 1
                    except OSError:
                        pass
                    # 每次轮询都刷新进度（即使段数没变），避免页面停在"正在加载模型"
                    if prog_exists:
                        base_pct = min(int(segments_done * 100 / max(est_total_fixed, 1)), 99)
                        _update_progress(task, f'推理中 ({base_pct}%) — 已处理 {segments_done} 段')
                    else:
                        _update_progress(task, '正在加载模型/等待推理...')
                    # 完成通知
                    try:
                        tid, ok, err = inference_done_q.get_nowait()
                    except Exception:
                        tid, ok, err = None, False, None
                    if tid is not None:
                        if tid == task_id:
                            result_ok, result_err = ok, err
                            break
                    time.sleep(1)

                if result_ok:
                    task.result_filename = result_name
                    task.status = 'done'
                    task.progress_msg = '推理完成'
                elif result_err and '用户停止推理' in str(result_err):
                    task.status = 'stopped'
                    task.progress_msg = '已停止'
                else:
                    task.status = 'failed'
                    task.error_msg = (result_err or '\n'.join(tail_lines))[-1000:]
                    task.progress_msg = f'推理失败: {(result_err or "未知错误")[:60]}'

                task.done_at = datetime.utcnow()

                # 推理完成/失败邮件通知
                try:
                    if task.user and task.user.infer_notify:
                        from notifier import notify_inference_complete
                        notify_inference_complete(task, os.environ.get('SSVC_SERVER_URL', 'http://127.0.0.1:5000'))
                except Exception:
                    pass

            except Exception as e:
                traceback.print_exc()
                task.status = 'failed'
                task.error_msg = f'{type(e).__name__}: {e}'[:200]
                task.progress_msg = f'失败: {str(e)[:80]}'
            finally:
                try:
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass


def ensure_worker():
    global task_worker_started, inference_q, inference_done_q, inference_daemon_proc
    if inference_q is None:
        try:
            inference_q = mp_module.Queue()
            inference_done_q = mp_module.Queue()
        except Exception as e:
            print(f'[ensure_worker] 推理队列创建失败: {e}', flush=True)
            inference_q = None
            inference_done_q = None
    if inference_q is not None and (inference_daemon_proc is None or not inference_daemon_proc.is_alive()):
        try:
            from inference_daemon import main as _daemon_main
            inference_daemon_proc = mp_module.Process(
                target=_daemon_main,
                args=(inference_q, inference_done_q, INFERENCE_MODEL_CACHE),
                daemon=True,
                name='inference-daemon',
            )
            inference_daemon_proc.start()
            print(f'[ensure_worker] 推理 daemon 已启动 (pid={inference_daemon_proc.pid}, cache={INFERENCE_MODEL_CACHE})', flush=True)
        except Exception as e:
            print(f'[ensure_worker] 推理 daemon 启动失败: {e}', flush=True)
            import traceback as _tb
            _tb.print_exc()
    if not task_worker_started:
        task_worker_started = True
        t = threading.Thread(target=task_worker, daemon=True)
        t.start()

@app.route('/api/tasks/<int:task_id>/status')
@login_required
def api_task_status(task_id):
    """返回推理任务实时状态（供任务列表页 AJAX 轮询）。"""
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)
    pct = None
    m = re.search(r'\((\d+)%\)', task.progress_msg or '')
    if m:
        pct = int(m.group(1))
    return jsonify({
        'status': task.status,
        'progress_msg': task.progress_msg,
        'pct': pct,
        'has_result': bool(task.result_filename),
        'error_msg': (task.error_msg or '')[:200],
    })


@app.route('/api/tasks/<int:task_id>/stream')
@login_required
def api_task_stream(task_id):
    """推理任务进度 SSE：推送 progress/done 事件，前端 EventSource 实时接收。"""
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)

    def gen():
        sent = 0
        while True:
            with app.app_context():
                t = db.session.get(Task, task_id)
                if not t:
                    yield 'event: done\ndata: {"status":"gone"}\n\n'
                    break
                if t.status == 'running':
                    pct = None
                    m = re.search(r'\((\d+)%\)', t.progress_msg or '')
                    if m:
                        pct = int(m.group(1))
                    payload = json.dumps({
                        'status': 'running',
                        'progress_msg': t.progress_msg or '',
                        'pct': pct,
                    }, ensure_ascii=False)
                    yield f'event: progress\ndata: {payload}\n\n'
                else:
                    payload = json.dumps({
                        'status': t.status,
                        'progress_msg': t.progress_msg or '',
                        'error_msg': t.error_msg or '',
                    }, ensure_ascii=False)
                    yield f'event: done\ndata: {payload}\n\n'
                    break
            sent += 1
            if sent > 3600 * 2:  # 最长推送 2 小时，防止连接泄漏
                break
            time.sleep(1)

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/inference', methods=['GET', 'POST'])
@login_required
def inference():
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).all()

    if request.method == 'POST':
        config_id = request.form.get('config_id', type=int)
        audio_file = request.files.get('audio_file')

        if not config_id or not audio_file or not audio_file.filename:
            flash('请选择配置和音频文件', 'danger')
            return render_template('inference.html', configs=configs)

        cfg_obj = InferenceConfig.query.get_or_404(config_id)
        if cfg_obj.user_id != current_user.id:
            abort(403)

        # Save audio
        audio_filename = save_uploaded(audio_file, 'audio')

        # 合并参数：配置默认值 + 推理页临时覆盖
        try:
            cfg_params = json.loads(cfg_obj.params_json) if cfg_obj.params_json else DEFAULT_PARAMS.copy()
        except Exception:
            cfg_params = DEFAULT_PARAMS.copy()
        override = {}
        for key, default in DEFAULT_PARAMS.items():
            if key in ('device', 'memory_limit'):
                continue
            val = request.form.get(key)
            if val is None:
                continue
            if isinstance(default, bool):
                override[key] = val == 'on' or val == '1'
            elif isinstance(default, int):
                try:
                    override[key] = int(val)
                except (ValueError, TypeError):
                    pass
            elif isinstance(default, float):
                try:
                    override[key] = float(val)
                except (ValueError, TypeError):
                    pass
            else:
                override[key] = val
        cfg_params.update(override)

        # Create task
        ensure_worker()
        task = Task(
            user_id=current_user.id,
            config_id=config_id,
            audio_filename=audio_filename,
            params_json=json.dumps(cfg_params, ensure_ascii=False),
            device_pref=current_user.device_pref or 'auto',
            memory_limit=current_user.memory_limit or 0,
            status='pending',
        )
        db.session.add(task)
        db.session.commit()

        task_queue.put(task.id)
        flash('推理任务已提交，请在任务列表中查看进度', 'success')
        return redirect(url_for('task_list'))

    # 最近 6 次推理结果（用于会话试听历史）
    recent = (Task.query.filter_by(user_id=current_user.id, status='done')
              .order_by(Task.created_at.desc()).limit(6).all())
    recent_items = []
    for t in recent:
        if not t.result_filename:
            continue
        try:
            t_params = json.loads(t.params_json or '{}')
        except Exception:
            t_params = {}
        recent_items.append({
            'id': t.id,
            'result': t.result_filename,
            'audio': t.audio_filename,
            'params': t_params,
            'created': t.done_at or t.created_at,
        })
    config_items = []
    for c in configs:
        try:
            c_params = json.loads(c.params_json or '{}')
        except Exception:
            c_params = {}
        config_items.append({'id': c.id, 'name': c.name, 'model': c.model.name,
                             'params': c_params,
                             'has_cluster': bool(c.model.cluster_path)})
    return render_template('inference.html', configs=config_items, recent=recent_items)


@app.route('/tasks')
@login_required
def task_list():
    tasks = Task.query.filter_by(user_id=current_user.id).order_by(Task.created_at.desc()).all()
    train_tasks = TrainingTask.query.filter_by(user_id=current_user.id).order_by(TrainingTask.created_at.desc()).all()
    # 可续训：任务自身或链上原始任务的目录里有 G_*/D_* checkpoint，附最新步数
    resumable_ids = {}
    for t in train_tasks:
        src_id = t.resume_from_id or t.id
        latest = 0
        if t.model_type == 'diffusion':
            dd = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{src_id}')
            if os.path.isdir(dd):
                for f in os.listdir(dd):
                    if f.startswith('model_') and f.endswith('.pt'):
                        latest = max(latest, int(''.join(c for c in f if c.isdigit()) or 0))
        else:
            td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{src_id}')
            if os.path.isdir(td):
                for f in os.listdir(td):
                    if f.startswith('G_') and f.endswith('.pth'):
                        latest = max(latest, int(''.join(c for c in f if c.isdigit()) or 0))
        if latest > 0:
            resumable_ids[t.id] = latest
    return render_template('tasks.html', tasks=tasks, train_tasks=train_tasks, resumable_ids=resumable_ids)


@app.route('/tasks/<int:task_id>/result')
@login_required
def task_result(task_id):
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    if not task.result_filename:
        abort(404)
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', task.result_filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.route('/tasks/<int:task_id>/delete', methods=['POST'])
@login_required
def task_delete(task_id):
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    # 如果任务正在运行，杀掉子进程
    with task_processes_lock:
        proc = task_processes.get(task_id)
        if task.status == 'running' and proc and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            task_processes.pop(task_id, None)
    # Delete result file
    if task.result_filename:
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', task.result_filename)
        if os.path.exists(path):
            os.remove(path)
    db.session.delete(task)
    db.session.commit()
    flash('任务已删除', 'success')
    return redirect(url_for('task_list'))


@app.route('/tasks/<int:task_id>/stop', methods=['POST'])
@login_required
def task_stop(task_id):
    """停止推理任务：运行中给 daemon 发中断信号（缓存保留），排队中直接标记停止。"""
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)
    if task.status == 'running':
        if inference_daemon_proc and inference_daemon_proc.is_alive():
            try:
                os.kill(inference_daemon_proc.pid, signal.SIGINT)
            except Exception:
                # Windows 或信号不可用时降级：直接结束 daemon（缓存丢失），ensure_worker 会自动重启
                try:
                    inference_daemon_proc.terminate()
                except Exception:
                    pass
        task.progress_msg = '正在停止...'
        db.session.commit()
        flash('已请求停止推理任务（当前片段处理完即停）', 'warning')
    elif task.status == 'pending':
        task.status = 'stopped'
        task.progress_msg = '已停止（未开始）'
        task.done_at = datetime.utcnow()
        db.session.commit()
        flash('已停止排队中的推理任务', 'warning')
    else:
        flash('该任务不在运行中，无法停止', 'info')
    return redirect(url_for('task_list'))


@app.route('/results/<filename>')
@login_required
def download_result(filename):
    name = secure_filename(filename)
    if not name:
        abort(400)
    try:
        path = safe_join(app.config['UPLOAD_FOLDER'], 'results', name)
    except ValueError:
        abort(400)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


# ========== 预训练模型管理（训练底模等） ==========

PRETRAIN_DIR = os.path.join(PROJECT_DIR, 'pretrain')
PRETRAIN_FILES = [
    ('g0', 'G_0.pth', 'SoVITS 生成器底模（训练必备，推荐）'),
    ('d0', 'D_0.pth', 'SoVITS 判别器底模（训练必备，推荐）'),
    ('rmvpe', 'rmvpe.pt', 'RMVPE F0 预测器（训练/推理选 rmvpe 时使用）'),
    ('contentvec', 'checkpoint_best_legacy_500.pt', 'ContentVec 编码器（预处理/推理必需）'),
    ('nsf_model', 'nsf_hifigan/model', 'NSF-HiFiGAN 声码器（推理/扩散必需）'),
    ('nsf_config', 'nsf_hifigan/config.json', 'NSF-HiFiGAN 配置'),
    ('hubertsoft', 'hubert-soft-0d54a1f4.pt', 'HuBERTSoft 编码器（选 hubertsoft 时使用）'),
    ('whisper', 'medium.pt', 'Whisper-PPG 编码器（约 1.5GB）'),
    ('whisper_large', 'large-v2.pt', 'Whisper-PPG-Large 编码器（约 3GB）'),
    ('cnhubert', 'chinese-hubert-large-fairseq-ckpt.pt', 'CN-HuBERT-Large 编码器'),
    ('dphubert', 'DPHuBERT-sp0.75.pth', 'DP-HuBERT 编码器'),
    ('wavlm', 'WavLM-Base+.pt', 'WavLM Base+ 编码器'),
]

# 编码器 → 需要的预训练文件（用于训练提交前校验）
ENCODER_PRETRAIN_FILES = {
    'vec768l12': 'checkpoint_best_legacy_500.pt',
    'vec256l9': 'checkpoint_best_legacy_500.pt',
    'hubertsoft': 'hubert-soft-0d54a1f4.pt',
    'whisper-ppg': 'medium.pt',
    'whisper-ppg-large': 'large-v2.pt',
    'cnhubertlarge': 'chinese-hubert-large-fairseq-ckpt.pt',
    'dphubert': 'DPHuBERT-sp0.75.pth',
    'wavlmbase+': 'WavLM-Base+.pt',
}


def _file_size_mb(path):
    try:
        return f'{os.path.getsize(path) / 1048576:.1f} MB'
    except OSError:
        return None


@app.route('/pretrain')
@login_required
def pretrain_page():
    items = []
    for key, rel, desc in PRETRAIN_FILES:
        full = os.path.join(PRETRAIN_DIR, rel)
        size = _file_size_mb(full)
        items.append({'key': key, 'rel': rel, 'desc': desc, 'exists': size is not None, 'size': size})
    return render_template('pretrain.html', items=items)


@app.route('/pretrain/upload', methods=['POST'])
@login_required
def pretrain_upload():
    """上传训练底模/预训练文件到服务器 pretrain 目录，覆盖已有同名文件。"""
    targets = {
        'g0_file': ('G_0.pth', ('.pth', '.pt')),
        'd0_file': ('D_0.pth', ('.pth', '.pt')),
        'rmvpe_file': ('rmvpe.pt', ('.pt', '.pth')),
        'contentvec_file': ('checkpoint_best_legacy_500.pt', ('.pt', '.pth')),
        'nsf_config_file': ('nsf_hifigan/config.json', ('.json',)),
        'hubertsoft_file': ('hubert-soft-0d54a1f4.pt', ('.pt', '.pth')),
        'whisper_file': ('medium.pt', ('.pt', '.pth')),
        'whisper_large_file': ('large-v2.pt', ('.pt', '.pth')),
        'cnhubert_file': ('chinese-hubert-large-fairseq-ckpt.pt', ('.pt', '.pth')),
        'dphubert_file': ('DPHuBERT-sp0.75.pth', ('.pth', '.pt')),
        'wavlm_file': ('WavLM-Base+.pt', ('.pt', '.pth')),
    }
    saved = []
    for field, (rel, exts) in targets.items():
        f = request.files.get(field)
        if not f or not f.filename:
            continue
        if not f.filename.lower().endswith(exts):
            flash(f'{rel} 文件扩展名不允许', 'danger')
            return redirect(url_for('pretrain_page'))
        dest = os.path.join(PRETRAIN_DIR, rel)
        os.makedirs(os.path.dirname(dest) or PRETRAIN_DIR, exist_ok=True)
        f.save(dest)
        saved.append(rel)

    nsf = request.files.get('nsf_model_file')
    if nsf and nsf.filename:
        dest = os.path.join(PRETRAIN_DIR, 'nsf_hifigan', 'model')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        nsf.save(dest)
        saved.append('nsf_hifigan/model')

    if saved:
        flash('已上传: ' + ', '.join(saved), 'success')
    else:
        flash('未选择任何文件', 'warning')
    return redirect(url_for('pretrain_page'))


@app.route('/pretrain/delete', methods=['POST'])
@login_required
def pretrain_delete():
    key = request.form.get('key', '')
    rel = next((r for k, r, _ in PRETRAIN_FILES if k == key), None)
    if not rel:
        flash('无效的文件', 'danger')
        return redirect(url_for('pretrain_page'))
    full = os.path.join(PRETRAIN_DIR, rel)
    if os.path.exists(full):
        os.remove(full)
        flash(f'已删除 {rel}', 'success')
    else:
        flash('文件不存在', 'warning')
    return redirect(url_for('pretrain_page'))


# ========== 文件管理 ==========

FILE_DIRS = {
    'models': '模型文件 (.pth/.pt)',
    'configs': '配置文件 (.json/.yaml)',
    'results': '推理结果',
    'dataset_zips': '数据集 zip',
}


@app.route('/files')
@login_required
def files_page():
    sections = []
    for sub, title in FILE_DIRS.items():
        d = os.path.join(app.config['UPLOAD_FOLDER'], sub)
        items = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    try:
                        size = f'{os.path.getsize(fp) / 1048576:.1f} MB'
                        modified = datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%m-%d %H:%M')
                    except OSError:
                        size, modified = '', ''
                    items.append({'name': f, 'size': size, 'modified': modified})
        sections.append({'key': sub, 'title': title, 'items': items})
    return render_template('files.html', sections=sections)


@app.route('/files/download')
@login_required
def file_download():
    sub = request.args.get('sub', '')
    name = request.args.get('name', '')
    # 用 basename 校验拒绝路径穿越（secure_filename 会吞掉开头的下划线，破坏 _开头的文件名）
    if sub not in FILE_DIRS or not name or os.path.basename(name) != name:
        abort(400)
    try:
        path = safe_join(app.config['UPLOAD_FOLDER'], sub, name)
    except ValueError:
        abort(400)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.route('/files/delete', methods=['POST'])
@login_required
def file_delete():
    sub = request.form.get('sub', '')
    name = request.form.get('name', '')
    if sub not in FILE_DIRS or not name or os.path.basename(name) != name:
        flash('非法文件', 'danger')
        return redirect(url_for('files_page'))
    try:
        path = safe_join(app.config['UPLOAD_FOLDER'], sub, name)
    except ValueError:
        flash('非法路径', 'danger')
        return redirect(url_for('files_page'))
    if not os.path.exists(path):
        flash('文件不存在', 'warning')
        return redirect(url_for('files_page'))
    # 引用检查：被模型/训练任务引用的文件不允许直接删除
    ref = None
    if sub in ('models', 'configs'):
        ref = Model.query.filter(sa.or_(
            Model.model_path == name, Model.config_path == name,
            Model.diff_model_path == name, Model.diff_config_path == name,
            Model.cluster_path == name,
        )).first()
        if not ref:
            ref = TrainingTask.query.filter(sa.or_(
                TrainingTask.model_path == name, TrainingTask.config_path == name,
                TrainingTask.diff_model_path == name, TrainingTask.diff_config_path == name,
            )).first()
    if ref:
        flash(f'{name} 被模型/任务引用，请用"精确清理"或先解除引用', 'danger')
        return redirect(url_for('files_page'))
    os.remove(path)
    flash(f'已删除 {name}', 'success')
    return redirect(url_for('files_page'))


@app.route('/files/delete_batch', methods=['POST'])
@login_required
def file_delete_batch():
    """批量删除文件（同目录多个文件），引用检查与单删一致。"""
    sub = request.form.get('sub', '')
    names = request.form.getlist('names')
    if sub not in FILE_DIRS or not names:
        flash('参数错误', 'danger')
        return redirect(url_for('files_page'))
    ok, skipped = 0, []
    for name in names:
        if not name or os.path.basename(name) != name:
            skipped.append(name)
            continue
        try:
            path = safe_join(app.config['UPLOAD_FOLDER'], sub, name)
        except ValueError:
            skipped.append(name)
            continue
        if not os.path.exists(path):
            skipped.append(name)
            continue
        ref = None
        if sub in ('models', 'configs'):
            ref = Model.query.filter(sa.or_(
                Model.model_path == name, Model.config_path == name,
                Model.diff_model_path == name, Model.diff_config_path == name,
                Model.cluster_path == name,
            )).first()
            if not ref:
                ref = TrainingTask.query.filter(sa.or_(
                    TrainingTask.model_path == name, TrainingTask.config_path == name,
                    TrainingTask.diff_model_path == name, TrainingTask.diff_config_path == name,
                )).first()
        if ref:
            skipped.append(name)
            continue
        try:
            os.remove(path)
            ok += 1
        except OSError:
            skipped.append(name)
    if ok:
        flash(f'已批量删除 {ok} 个文件', 'success')
    if skipped:
        flash(f'{len(skipped)} 个文件跳过（被引用或不存在）：{", ".join(skipped[:5])}', 'warning')
    return redirect(url_for('files_page'))


# ========== 训练 ==========

train_process = None
train_active_id = None


def train_worker_daemon():
    global train_process, train_active_id
    while True:
        with app.app_context():
            task = None
            try:
                task = TrainingTask.query.filter_by(status='pending').order_by(TrainingTask.created_at).first()
                if task and train_process is None:
                    train_active_id = task.id
                    task.status = 'running'
                    task.progress_msg = '初始化...'
                    db.session.commit()

                    from train_worker import run as tr
                    extra = json.loads(task.params_json or '{}')
                    root_id = _chain_root_id(task)

                    # 训练墙钟超时（TRAIN_TIMEOUT 秒，0=不限制）：
                    # 到点由看门狗杀掉训练子进程，worker 会自动保存最新 checkpoint
                    timeout = int(os.environ.get('TRAIN_TIMEOUT', '0') or 0)
                    if timeout > 0:
                        _task_id = task.id

                        def _timeout_watchdog():
                            time.sleep(timeout)
                            with app.app_context():
                                t = db.session.get(TrainingTask, _task_id)
                                if t and t.status == 'running':
                                    from train_worker import stop as stop_train
                                    stop_train()
                                    t.error_msg = '__TIMEOUT__'
                                    db.session.commit()

                        threading.Thread(target=_timeout_watchdog, daemon=True).start()

                    # 训练进度邮件报告：每 report_interval 步发一封
                    _rep_user = task.user
                    _rep_interval = (_rep_user.report_interval or 0) if _rep_user else 0
                    if _rep_interval > 0 and _rep_user and _rep_user.email_notify:
                        _task_id = task.id
                        _interval = _rep_interval

                        def _progress_reporter():
                            last = 0
                            while True:
                                time.sleep(60)
                                try:
                                    with app.app_context():
                                        t = db.session.get(TrainingTask, _task_id)
                                        if not t or t.status != 'running':
                                            break
                                        info = _parse_training_log(t)
                                        step = info.get('current_step', 0)
                                        if step >= _interval and (step // _interval) > (last // _interval):
                                            last = step
                                            losses = ''
                                            ld = info.get('loss_data', [])
                                            if ld:
                                                g = ld[-1].get('g')
                                                d = ld[-1].get('d')
                                                losses = f'G={g} D={d}'
                                            from notifier import notify_train_progress
                                            notify_train_progress(
                                                t,
                                                os.environ.get('SSVC_SERVER_URL', 'http://127.0.0.1:5000'),
                                                step, t.total_steps or 0,
                                                losses, info.get('stage', ''), info.get('eta', ''),
                                            )
                                except Exception:
                                    pass

                        threading.Thread(target=_progress_reporter, daemon=True).start()

                    # 训练异常检测：判别器压制 / NaN，发邮件附确认链接
                    if _rep_user and _rep_user.email_notify:
                        _task_id2 = task.id

                        def _anomaly_watchdog():
                            while True:
                                time.sleep(60)
                                try:
                                    with app.app_context():
                                        t = db.session.get(TrainingTask, _task_id2)
                                        if not t or t.status != 'running':
                                            break
                                        info = _parse_training_log(t)
                                        ld = info.get('loss_data', [])
                                        if len(ld) < 10:
                                            continue
                                        recent = ld[-20:]
                                        kind, detail = None, ''
                                        if any(('g' in x and x['g'] != x['g']) or ('d' in x and x['d'] != x['d'])
                                               for x in recent):
                                            kind = 'nan_loss'
                                            detail = '检测到 NaN Loss，训练可能已经发散'
                                        else:
                                            gs = [x.get('g') for x in recent if x.get('g') is not None]
                                            ds = [x.get('d') for x in recent if x.get('d') is not None]
                                            if len(gs) >= 10 and len(ds) >= 10:
                                                g_avg = sum(gs) / len(gs)
                                                d_avg = sum(ds) / len(ds)
                                                if d_avg < 0.6 and g_avg > 3.0:
                                                    kind = 'disc_pressed'
                                                    detail = f'最近 {len(recent)} 个点：D 均值 {d_avg:.3f}，G 均值 {g_avg:.3f}'
                                        if kind and t.anomaly_state != 'pending':
                                            token = uuid.uuid4().hex[:32]
                                            t.anomaly_token = token
                                            t.anomaly_state = 'pending'
                                            db.session.commit()
                                            from notifier import notify_train_anomaly
                                            notify_train_anomaly(
                                                t, os.environ.get('SSVC_SERVER_URL', 'http://127.0.0.1:5000'),
                                                token, kind, detail)
                                except Exception:
                                    pass

                        threading.Thread(target=_anomaly_watchdog, daemon=True).start()

                    result = tr(
                        task_id=task.id,
                        speaker=task.speaker,
                        dataset_zip=task.dataset_zip,
                        log_path=task.log_path or '',
                        model_type=task.model_type,
                        batch_size=task.batch_size or 4,
                        total_steps=task.total_steps or 4200,
                        keep_ckpts=task.keep_ckpts or 3,
                        resume_from_id=task.resume_from_id or 0,
                        diff_root_id=root_id,
                        **extra,
                    )
                    # 用户手动停止时（status=stopped），train_stop 已保存 checkpoint 并注册模型，
                    # 这里不能再覆盖任务状态
                    if task.status != 'stopped':
                        if task.error_msg == '__TIMEOUT__':
                            task.status = 'failed'
                            task.progress_msg = '训练超时，已自动停止'
                            task.error_msg = '训练超过设定时间（TRAIN_TIMEOUT），已自动停止并保存 checkpoint'
                        else:
                            task.status = result.get('status', 'failed')
                            task.progress_msg = result.get('progress_msg', '')
                            task.error_msg = result.get('error_msg', '')
                        if result.get('model_path'):
                            task.model_path = result['model_path']
                        if result.get('config_path'):
                            task.config_path = result['config_path']
                        if result.get('diff_model_path'):
                            task.diff_model_path = result['diff_model_path']
                        if result.get('diff_config_path'):
                            task.diff_config_path = result['diff_config_path']
                        task.done_at = datetime.utcnow()
                        db.session.commit()
                        try:
                            from notifier import notify_train_complete
                            ip = os.environ.get('SSVC_SERVER_URL', 'http://172.16.77.28:5000')
                            notify_train_complete(task, ip)
                        except Exception:
                            pass
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                if task is not None:
                    try:
                        if not (task.status == 'stopped' or
                                (task.status == 'failed' and task.error_msg == '用户手动停止')):
                            task.status = 'failed'
                            task.error_msg = str(e)[:200]
                            task.progress_msg = '训练异常'
                        task.done_at = datetime.utcnow()
                        db.session.commit()
                    except Exception:
                        pass
            finally:
                train_active_id = None
                train_process = None
        time.sleep(5)


def ensure_train_worker():
    if not any(t.name == 'train-daemon' and t.is_alive() for t in threading.enumerate()):
        t = threading.Thread(target=train_worker_daemon, name='train-daemon', daemon=True)
        t.start()


STAGES = [
    ('init', '初始化'),
    ('unzip', '解压'),
    ('resample', '重采样'),
    ('config', '生成配置'),
    ('features', '提取特征'),
    ('sovits', 'SoVITS 训练'),
    ('diff_config', '扩散配置'),
    ('diff_train', '扩散训练'),
    ('save', '保存模型'),
]

STAGE_KEYWORDS = {
    '初始化': 'init', '解压': 'unzip', '重采样': 'resample',
    '生成配置': 'config', '配置扩散': 'diff_config',
    '提取特征': 'features', '特征提取': 'features',
    '开始 SoVITS': 'sovits', '开始扩散': 'diff_train',
    '保存模型': 'save', '模型已保存': 'save',
}


def detect_stage(log_text, progress_msg):
    lines = log_text.split('\n')
    for line in reversed(lines):
        for kw, stage in STAGE_KEYWORDS.items():
            if kw in line:
                return stage
    for kw, stage in STAGE_KEYWORDS.items():
        if kw in progress_msg:
            return stage
    return 'init'


def _format_eta(seconds):
    if seconds is None or seconds < 0:
        return '...'
    seconds = int(seconds)
    if seconds > 3600:
        return f'{seconds // 3600}h{(seconds % 3600) // 60}m'
    if seconds > 60:
        return f'{seconds // 60}m{seconds % 60}s'
    return f'{seconds}s'


def _parse_step_times(log_content):
    """从日志提取 [HH:MM:SS] ... step: N 序列（sovits Losses 行与扩散 solver 行都带）。"""
    pairs = []
    for line in log_content.split('\n'):
        m = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\].*step:\s*(\d+)', line)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pairs.append((h * 3600 + mi * 60 + s, int(m.group(4))))
    return pairs


def _parse_training_log(active):
    """从训练日志解析进度，页面与 API 共用。"""
    info = {
        'stage': 'init',
        'pct': 0,
        'current_step': 0,
        'total_steps': active.total_steps or 0,
        'diff_epoch': 0,
        'diff_total_epochs': 0,
        'loss_data': [],
        'elapsed': '',
        'eta': '',
        'log_tail': '',
        'eval_mel': None,
        'eval_step': 0,
    }
    if not active or not active.log_path or not os.path.exists(active.log_path):
        return info
    try:
        with open(active.log_path, 'r', encoding='utf-8') as f:
            # 只读尾部（256KB），避免训练日志变大后每次轮询整读文件
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 262144))
            log_content = f.read()
    except Exception:
        log_content = ''
    info['stage'] = detect_stage(log_content, active.progress_msg or '')
    diff_extra = ''  # 扩散 solver 日志（log_info.txt），供 diff_train 阶段与 loss 解析使用

    elapsed_secs = 0
    if active.created_at:
        elapsed_secs = max(int((datetime.utcnow() - active.created_at).total_seconds()), 0)
    info['elapsed'] = _format_eta(elapsed_secs)

    if info['stage'] == 'diff_train':
        diff_cfg_path = os.path.join(PROJECT_DIR, 'configs', 'diffusion.yaml')
        try:
            import yaml
            with open(diff_cfg_path, 'r', encoding='utf-8') as f:
                dc = yaml.safe_load(f)
            info['diff_total_epochs'] = int(dc.get('train', {}).get('epochs', 0) or 0)
        except Exception:
            info['diff_total_epochs'] = 0
        # 扩散 solver 日志写在 expdir/log_info.txt（train.log 里没有），需要单独读取。
        # 格式：epoch: 5 | 123/456 | expdir | batch/s: 1.20 | lr: 0.0001 | loss: 1.234 | time: 10s | step: 12345
        root_id = active.resume_from_id or active.id
        diff_log = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{root_id}', 'log_info.txt')
        diff_extra = ''
        if os.path.exists(diff_log):
            try:
                with open(diff_log, 'r', encoding='utf-8') as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 262144))
                    diff_extra = f.read()
            except Exception:
                diff_extra = ''
        m = re.search(r'epoch:\s*(\d+)\s*\|\s*(\d+)/(\d+)', diff_extra)
        if m:
            info['diff_epoch'] = int(m.group(1))
            batch_total = max(int(m.group(3)), 1)
            diff_total_steps = max(info['diff_total_epochs'], 1) * batch_total
            info['total_steps'] = diff_total_steps
            sm = re.findall(r'\| step:\s*(\d+)', diff_extra)
            if sm:
                info['current_step'] = min(int(sm[-1]), diff_total_steps)
                info['pct'] = min(int(info['current_step'] / diff_total_steps * 100), 99)
                # log_info.txt 无时间戳：用全程平均，但扣除续训起点（避免恢复步数虚增速度）
                start_step = 0
                if active.resume_from_id:
                    dd = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{root_id}')
                    if os.path.isdir(dd):
                        for f in os.listdir(dd):
                            if f.startswith('model_') and f.endswith('.pt'):
                                start_step = max(start_step, int(''.join(c for c in f if c.isdigit()) or 0))
                progress = info['current_step'] - start_step
                if progress > 0 and elapsed_secs > 0:
                    speed = progress / elapsed_secs
                    info['eta'] = _format_eta(int((diff_total_steps - info['current_step']) / max(speed, 1e-6)))
    else:
        total_steps = active.total_steps or 0
        pairs = _parse_step_times(log_content)
        if pairs:
            info['current_step'] = min(pairs[-1][1], total_steps) if total_steps else pairs[-1][1]
            info['total_steps'] = total_steps
            if total_steps > 0:
                info['pct'] = min(int(info['current_step'] / total_steps * 100), 99)
                if info['current_step'] > 0 and elapsed_secs > 0:
                    # 近期速度：最近两个带时间戳的 step 点，避免全程平均被预处理/排队时间拖低
                    if len(pairs) >= 2:
                        t0, s0 = pairs[-2]
                        t1, s1 = pairs[-1]
                        if t1 < t0:
                            t1 += 86400
                        dt = max(t1 - t0, 1)
                        ds = s1 - s0
                        if ds > 0:
                            speed = ds / dt
                            info['eta'] = _format_eta(int((total_steps - info['current_step']) / max(speed, 1e-6)))
        elif info['stage'] == 'features':
            pcts = re.findall(r'(\d+)%\|', log_content)
            if pcts:
                info['pct'] = min(max(int(x) for x in pcts), 99)
            else:
                info['pct'] = 15
        elif info['stage'] == 'resample':
            info['pct'] = 5
        elif info['stage'] == 'unzip':
            info['pct'] = 2

    for line in log_content.split('\n') + (diff_extra or '').split('\n'):
        m = re.search(r'Losses: \[([^\]]+)\]', line)
        if m:
            vals = [float(x.strip()) for x in m.group(1).split(',')]
            if len(vals) >= 2:
                info['loss_data'].append({'g': round(vals[1], 4), 'd': round(vals[0], 4)})
        m3 = re.search(r'\| loss: ([\d.eE+-]+) .*\| step: (\d+)', line)
        if m3:
            info['loss_data'].append({'diff': round(float(m3.group(1)), 4)})
        m2 = re.search(r'Eval Losses: \[([^\]]+)\], step: (\d+)', line)
        if m2:
            info['eval_mel'] = round(float(m2.group(1)), 4)
            info['eval_step'] = int(m2.group(2))
    info['log_tail'] = log_content[-5000:]
    return info


@app.route('/api/train/<int:tid>/status')
@login_required
def api_train_status(tid):
    """返回训练任务实时状态（供任务列表页 AJAX 轮询）。"""
    task = db.session.get(TrainingTask, tid)
    if not task or task.user_id != current_user.id:
        abort(404)
    if task.status == 'running':
        info = _parse_training_log(task)
    else:
        info = {
            'stage': '', 'pct': 100 if task.status == 'done' else 0,
            'current_step': 0, 'total_steps': task.total_steps or 0,
            'diff_epoch': 0, 'diff_total_epochs': 0,
            'loss_data': [], 'elapsed': '', 'eta': '',
            'eval_mel': None, 'eval_step': 0,
        }
    # log_tail 保留给训练页 fetch 局部更新（每次约 5KB，可接受）
    stage_label = dict(STAGES).get(info.get('stage', ''), '')
    return jsonify({'status': task.status, 'progress_msg': task.progress_msg, 'stage_label': stage_label, **info})


@app.route('/train')
@login_required
def train_page():
    ensure_train_worker()
    active = TrainingTask.query.filter_by(status='running').first()
    info = _parse_training_log(active) if active else {}
    log_content = info.get('log_tail', '')
    current_stage = info.get('stage', '')
    pct = info.get('pct', 0)
    current_step = info.get('current_step', 0)
    total_steps = info.get('total_steps', 0)
    diff_epoch = info.get('diff_epoch', 0)
    diff_total_epochs = info.get('diff_total_epochs', 0)
    loss_data = info.get('loss_data', [])
    elapsed = info.get('elapsed', '')
    eta = info.get('eta', '')
    eval_mel = info.get('eval_mel')
    eval_step = info.get('eval_step', 0)
    running = TrainingTask.query.filter_by(status='running').count()
    queued = TrainingTask.query.filter_by(status='pending').count()
    done_count = TrainingTask.query.filter_by(status='done').count()
    history = TrainingTask.query.order_by(TrainingTask.created_at.desc()).limit(20).all()
    # 可续训的任务：已完成/失败且仍有数据目录的
    history_resumable = []
    for t in TrainingTask.query.filter(TrainingTask.status.in_(['done', 'failed', 'stopped'])).order_by(TrainingTask.created_at.desc()).limit(10).all():
        src_id = t.resume_from_id or t.id
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{src_id}')
        latest_step = 0
        if os.path.isdir(td):
            for f in os.listdir(td):
                if f.startswith('G_') and f.endswith('.pth'):
                    latest_step = max(latest_step, int(''.join(c for c in f if c.isdigit()) or 0))
        if latest_step > 0:
            try:
                _srcp = json.loads(t.params_json or '{}') or {}
                _arch = _srcp.get('arch', 'sovits-v1')
                _flow_mode = _srcp.get('flow_mode', 'a2')
            except Exception:
                _arch = 'sovits-v1'
                _flow_mode = 'a2'
            history_resumable.append({
                'id': t.id, 'speaker': t.speaker, 'model_type': t.model_type, 'step': latest_step,
                'arch': _arch, 'flow_mode': _flow_mode,
            })
    # 快速恢复快照（停止后一键继续上次训练）
    qr_meta = None
    qr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', 'quick_resume', 'meta.json')
    if os.path.exists(qr_path):
        try:
            with open(qr_path, 'r', encoding='utf-8') as f:
                qr_meta = json.load(f)
        except Exception:
            qr_meta = None
    return render_template('train.html',
        active=active, log_content=log_content, pct=pct,
        current_step=current_step, total_steps=total_steps,
        diff_epoch=diff_epoch, diff_total_epochs=diff_total_epochs,
        elapsed=elapsed, eta=eta,
        eval_mel=eval_mel, eval_step=eval_step,
        loss_data=json.dumps(loss_data[-200:]),
        stages=STAGES, current_stage=current_stage,
        stage_index=next((i for i, (k, _) in enumerate(STAGES) if k == current_stage), 0),
        running=running, queued=queued, done_count=done_count,
        history=history, history_resumable=history_resumable,
        quick_resume=qr_meta,
        alive=active is not None)


@app.route('/train/quick-resume', methods=['POST'])
@login_required
def train_quick_resume():
    """从 TEMP 快照一键恢复上次训练（不弹步数，自动续到原目标或 +5000）。"""
    qr = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', 'quick_resume')
    meta_p = os.path.join(qr, 'meta.json')
    if not os.path.exists(meta_p):
        flash('没有可快速恢复的训练快照（需要先训练并停止过一次）', 'danger')
        return redirect(url_for('train_page'))
    try:
        with open(meta_p, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:
        flash('快速恢复快照损坏，无法恢复', 'danger')
        return redirect(url_for('train_page'))
    ckpt_step = meta.get('checkpoint_step') or 0
    total = max(meta.get('total_steps') or 0, ckpt_step + 5000)
    log_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{int(time.time())}')
    os.makedirs(log_dir, exist_ok=True)
    params = {
        'arch': meta.get('arch', 'sovits-v1'),
        'd_lr_scale': 0.5 if meta.get('arch') in ('rvc', 'rvc-flow') else 1.0,
        'flow_mode': meta.get('flow_mode', 'a2'),
        'speech_encoder': meta.get('speech_encoder', 'vec768l12'),
        'f0_predictor': meta.get('f0_predictor', 'dio'),
    }
    task = TrainingTask(
        user_id=current_user.id,
        speaker=meta.get('speaker') or 'speaker',
        dataset_zip=meta.get('dataset_zip') or '',
        model_type=meta.get('model_type', 'sovits'),
        batch_size=meta.get('batch_size') or 4,
        total_steps=total,
        keep_ckpts=meta.get('keep_ckpts') or 3,
        params_json=json.dumps(params),
        log_path=os.path.join(log_dir, 'train.log'),
        resume_from_id=meta.get('resume_from_id'),
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    flash(f'已快速恢复训练 #{task.id}：{meta.get("speaker")} 从 checkpoint {ckpt_step} 步继续，目标 {total} 步', 'success')
    return redirect(url_for('task_list'))


@app.route('/train/submit', methods=['POST'])
@login_required
def train_submit():
    speaker = request.form.get('speaker', '').strip()
    if not speaker:
        flash('请输入说话人名称', 'danger')
        return redirect(url_for('train_page'))

    # 校验所选编码器对应的预训练模型是否已上传
    speech_enc = request.form.get('speech_encoder', 'vec768l12')
    need_file = ENCODER_PRETRAIN_FILES.get(speech_enc)
    if need_file and not os.path.exists(os.path.join(PRETRAIN_DIR, need_file)):
        flash(f'编码器 {speech_enc} 需要先上传 {need_file}（请到"预训练"页上传）', 'danger')
        return redirect(url_for('train_page'))

    dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset_zips')
    os.makedirs(dataset_dir, exist_ok=True)
    f = request.files.get('dataset')

    def _int(k, default):
        try: return int(request.form.get(k, default))
        except: return default
    def _flt(k, default):
        try: return float(request.form.get(k, default))
        except: return default

    resume_from = _int('resume_from', 0)
    quick_chain = _int('quick_resume_chain', 0)
    chain_id = 0
    resume_latest = 0
    if resume_from:
        src = TrainingTask.query.get(resume_from)
        if not src or src.user_id != current_user.id:
            flash('续训源任务不存在', 'danger')
            return redirect(url_for('train_page'))
        # checkpoint 存在原始任务目录，续训链上的任务要指回原始任务
        chain_id = src.resume_from_id or src.id
        dataset_zip = src.dataset_zip
        # 计算链上最新 checkpoint 步数，避免总步数小于当前步数导致"一提交就结束"
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
        if os.path.isdir(td):
            for f in os.listdir(td):
                if f.startswith('G_') and f.endswith('.pth'):
                    resume_latest = max(resume_latest, int(''.join(c for c in f if c.isdigit()) or 0))
        print(f'[train_submit] 续训源: 任务 {resume_from} -> 数据目录 task_{chain_id}', flush=True)
    elif quick_chain and (not f or not f.filename):
        # 快速恢复：链任务记录可能已被清理，直接用数据目录续训
        chain_id = quick_chain
        dataset_zip = request.form.get('quick_dataset_zip', '').strip()
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
        if os.path.isdir(td):
            for ff in os.listdir(td):
                if ff.startswith('G_') and ff.endswith('.pth'):
                    resume_latest = max(resume_latest, int(''.join(c for c in ff if c.isdigit()) or 0))
        print(f'[train_submit] 快速恢复: 数据目录 task_{chain_id}, checkpoint {resume_latest}', flush=True)
    else:
        if not f or not f.filename:
            flash('请上传数据集 zip', 'danger')
            return redirect(url_for('train_page'))
        if not f.filename.lower().endswith('.zip'):
            flash('数据集必须是 zip 文件', 'danger')
            return redirect(url_for('train_page'))
        filename = f'_{current_user.id}_{int(time.time())}.zip'
        path = os.path.join(dataset_dir, filename)
        f.save(path)
        dataset_zip = filename

    params = {
        'arch': request.form.get('arch', 'sovits-v1'),
        'd_lr_scale': _flt('d_lr_scale', 1.0),
        'flow_mode': request.form.get('flow_mode', 'a2'),
        'speech_encoder': request.form.get('speech_encoder', 'vec768l12'),
        'f0_predictor': request.form.get('f0_predictor', 'dio'),
        'learning_rate': _flt('learning_rate', 0.0001),
        'segment_size': _int('segment_size', 10240),
        'lr_decay': _flt('lr_decay', 0.999875),
        'auto_stop': _int('auto_stop', 200),
        'log_interval': _int('log_interval', 200),
        'eval_interval': _int('eval_interval', 800),
        'diff_batch_size': _int('diff_batch_size', 8),
        'diff_epochs': _int('diff_epochs', 100000),
        'diff_timesteps': _int('diff_timesteps', 1000),
        'diff_kstep': _int('diff_kstep', 0),
        'diff_layers': _int('diff_layers', 20),
        'diff_chans': _int('diff_chans', 256),
        'diff_hidden': _int('diff_hidden', 128),
        'diff_lr': _flt('diff_lr', 0.0001),
        'diff_decay_step': _int('diff_decay_step', 100000),
        'diff_gamma': _flt('diff_gamma', 0.5),
        'diff_amp': request.form.get('diff_amp', 'fp32'),
    }

    # 续训时继承原任务的模型架构（避免续训链上架构被表单默认值覆盖）
    if resume_from:
        try:
            src_params = json.loads(src.params_json or '{}')
        except Exception:
            src_params = {}
        if not request.form.get('arch'):
            params['arch'] = src_params.get('arch', 'sovits-v1')
        if not request.form.get('d_lr_scale'):
            params['d_lr_scale'] = src_params.get('d_lr_scale', 1.0)
        if not request.form.get('flow_mode'):
            params['flow_mode'] = src_params.get('flow_mode', 'a2')

    mt = request.form.get('model_type', 'sovits')
    total_steps = _int('total_steps', 4200)
    if chain_id and resume_latest > 0 and total_steps <= resume_latest:
        flash(f'续训目标步数 {total_steps} 必须大于当前 checkpoint 的 {resume_latest} 步，请重新填写', 'danger')
        return redirect(url_for('train_page'))
    log_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{int(time.time())}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')

    task = TrainingTask(
        user_id=current_user.id,
        speaker=speaker,
        dataset_zip=dataset_zip,
        model_type=mt,
        batch_size=_int('batch_size', 4),
        total_steps=total_steps,
        keep_ckpts=_int('keep_ckpts', 3),
        params_json=json.dumps(params),
        log_path=log_path,
        resume_from_id=chain_id or None,
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    flash('训练任务已加入队列', 'success')
    return redirect(url_for('train_page'))


@app.route('/train/resume/<int:tid>', methods=['POST'])
@login_required
def train_resume(tid):
    """一键从 checkpoint 续训：复用链上原始任务的数据目录与配置。"""
    src = db.session.get(TrainingTask, tid)
    if not src or src.user_id != current_user.id:
        abort(404)
    chain_id = src.resume_from_id or src.id
    latest_step = 0
    is_diff = src.model_type == 'diffusion'
    if is_diff:
        dd = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{chain_id}')
        if os.path.isdir(dd):
            for f in os.listdir(dd):
                if f.startswith('model_') and f.endswith('.pt'):
                    latest_step = max(latest_step, int(''.join(c for c in f if c.isdigit()) or 0))
    else:
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
        if os.path.isdir(td):
            for f in os.listdir(td):
                if f.startswith('G_') and f.endswith('.pth'):
                    latest_step = max(latest_step, int(''.join(c for c in f if c.isdigit()) or 0))
    if latest_step <= 0:
        flash(f'任务 #{tid} 没有可用 checkpoint，无法续训', 'danger')
        return redirect(url_for('task_list'))

    try:
        params = json.loads(src.params_json or '{}')
    except (json.JSONDecodeError, TypeError):
        params = {}
    if is_diff:
        # 扩散续训：复用扩散配置（epochs 等），总步数对扩散无意义
        total_steps = src.total_steps or 0
    else:
        base = src.total_steps or 4000
        target_steps = request.form.get('target_steps', type=int) or 0
        if target_steps > 0:
            if target_steps <= latest_step:
                flash(f'续训目标步数 {target_steps} 必须大于当前 checkpoint 的 {latest_step} 步', 'danger')
                return redirect(url_for('task_list'))
            total_steps = target_steps
        else:
            total_steps = max(base, latest_step + base)
    log_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{int(time.time())}')
    os.makedirs(log_dir, exist_ok=True)

    task = TrainingTask(
        user_id=current_user.id,
        speaker=src.speaker,
        dataset_zip=src.dataset_zip,
        model_type=src.model_type,
        batch_size=src.batch_size or 4,
        total_steps=total_steps,
        keep_ckpts=src.keep_ckpts or 3,
        params_json=json.dumps(params),
        log_path=os.path.join(log_dir, 'train.log'),
        resume_from_id=chain_id,
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    if is_diff:
        flash(f'已创建扩散续训任务 #{task.id}：从扩散 checkpoint {latest_step} 步继续', 'success')
    else:
        flash(f'已创建续训任务 #{task.id}：从 checkpoint {latest_step} 步继续，目标 {total_steps} 步', 'success')
    return redirect(url_for('task_list'))


@app.route('/train/diffusion/<int:tid>', methods=['POST'])
@login_required
def train_diffusion(tid):
    """直接用现有数据/特征进入扩散训练，跳过 SoVITS 主模型训练。"""
    src = db.session.get(TrainingTask, tid)
    if not src or src.user_id != current_user.id:
        abort(404)
    chain_id = src.resume_from_id or src.id
    td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
    if not os.path.isdir(td) or not any(f.startswith('G_') and f.endswith('.pth') for f in os.listdir(td)):
        flash('源任务没有 SoVITS checkpoint，请先训练主模型', 'danger')
        return redirect(url_for('task_list'))
    try:
        params = json.loads(src.params_json or '{}')
    except (json.JSONDecodeError, TypeError):
        params = {}
    # 扩散特征提取需要与主模型相同的编码器/F0 配置
    params.setdefault('speech_encoder', 'vec768l12')
    params.setdefault('f0_predictor', 'dio')
    log_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{int(time.time())}')
    os.makedirs(log_dir, exist_ok=True)
    task = TrainingTask(
        user_id=current_user.id,
        speaker=src.speaker,
        dataset_zip=src.dataset_zip,
        model_type='diffusion',
        batch_size=src.batch_size or 4,
        total_steps=src.total_steps or 0,
        keep_ckpts=src.keep_ckpts or 3,
        params_json=json.dumps(params),
        log_path=os.path.join(log_dir, 'train.log'),
        resume_from_id=chain_id,
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    flash(f'已创建扩散训练任务 #{task.id}（复用任务 {chain_id} 的数据与特征，跳过主模型）', 'success')
    return redirect(url_for('task_list'))


@app.route('/train/stop', methods=['POST'])
@login_required
def train_stop():
    from train_worker import stop as stop_train
    stop_train()
    task = TrainingTask.query.filter_by(status='running').first()
    if task:
        root_id = _chain_root_id(task)
        chain_id = task.resume_from_id or task.id
        td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{chain_id}')
        latest = _latest_g_checkpoint(td) if os.path.isdir(td) else None
        task.status = 'stopped'
        task.error_msg = '用户手动停止（checkpoint 已保存）'
        task.progress_msg = '已停止'
        task.done_at = datetime.utcnow()
        # 纯扩散任务停止时只保存扩散模型，不再重复注册 SoVITS 主模型
        if latest and task.model_type != 'diffusion':
            try:
                model_name = f'{uuid.uuid4().hex[:8]}_{latest}'
                shutil.copy2(os.path.join(td, latest),
                             os.path.join(app.config['UPLOAD_FOLDER'], 'models', model_name))
                cfg_name = None
                cfg_src = os.path.join(td, 'config.json')
                if os.path.exists(cfg_src):
                    cfg_name = f'{uuid.uuid4().hex[:8]}_config_{latest.replace(".pth", ".json")}'
                    shutil.copy2(cfg_src, os.path.join(app.config['UPLOAD_FOLDER'], 'configs', cfg_name))
                # 自动挂载训练时生成的特征检索索引（{speaker}_cluster.pth）
                cluster_name = None
                try:
                    _cand = f'{task.speaker}_cluster.pth'
                    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'models', _cand)):
                        cluster_name = _cand
                except Exception:
                    pass
                m = Model(
                    user_id=current_user.id,
                    name=f'{task.speaker}-{latest.replace(".pth", "")}step',
                    model_path=model_name,
                    config_path=cfg_name,
                    cluster_path=cluster_name,
                )
                db.session.add(m)
                task.model_path = model_name
                task.config_path = cfg_name
            except Exception as e:
                flash(f'checkpoint 保存失败: {e}', 'danger')
        # 扩散 checkpoint（扩散任务或 sovits_diff 任务停止时一并保存）
        diff_dir = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{root_id}')
        diff_latest = None
        if os.path.isdir(diff_dir):
            best_n = -1
            for f in os.listdir(diff_dir):
                if f.startswith('model_') and f.endswith('.pt'):
                    n = int(''.join(c for c in f if c.isdigit()) or 0)
                    if n > best_n:
                        diff_latest, best_n = f, n
        if diff_latest:
            try:
                diff_name = f'{uuid.uuid4().hex[:8]}_{diff_latest}'
                shutil.copy2(os.path.join(diff_dir, diff_latest),
                             os.path.join(app.config['UPLOAD_FOLDER'], 'models', diff_name))
                diff_cfg_name = None
                diff_cfg_src = os.path.join(PROJECT_DIR, 'configs', 'diffusion.yaml')
                if os.path.exists(diff_cfg_src):
                    diff_cfg_name = f'{uuid.uuid4().hex[:8]}_diff_{diff_latest.replace(".pt", ".yaml")}'
                    shutil.copy2(diff_cfg_src, os.path.join(app.config['UPLOAD_FOLDER'], 'configs', diff_cfg_name))
                task.diff_model_path = diff_name
                task.diff_config_path = diff_cfg_name
            except Exception as e:
                flash(f'扩散 checkpoint 保存失败: {e}', 'danger')
        db.session.commit()
        step_txt = f'（checkpoint {latest} 步）' if (latest and task.model_type != 'diffusion') else ''
        diff_txt = '，扩散模型已保存' if diff_latest else ''
        model_txt = '，已注册到模型列表' if (latest and task.model_type != 'diffusion') else ''
        flash(f'训练已停止{step_txt}{model_txt}{diff_txt}，可直接推理测试', 'success')
        # 停止即完成（扩散训练正常流程是手动停止）：发完成邮件
        try:
            if task.user and task.user.email_notify:
                from notifier import notify_train_complete
                notify_train_complete(task, os.environ.get('SSVC_SERVER_URL', 'http://127.0.0.1:5000'))
        except Exception:
            pass
    train_process = None
    return redirect(url_for('task_list'))


@app.route('/train/result/<int:tid>')
@login_required
def train_result(tid):
    task = TrainingTask.query.get_or_404(tid)
    if task.user_id != current_user.id:
        abort(403)
    is_config = request.args.get('config', '')
    is_diff = request.args.get('diff', '')
    if is_config:
        cf = task.config_path if not is_diff else task.diff_config_path
        if not cf:
            abort(404)
        try:
            path = safe_join(app.config['UPLOAD_FOLDER'], 'configs', secure_filename(cf))
        except ValueError:
            abort(400)
    else:
        model = task.diff_model_path if is_diff else task.model_path
        if not model:
            abort(404)
        try:
            path = safe_join(app.config['UPLOAD_FOLDER'], 'models', secure_filename(model))
        except ValueError:
            abort(400)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.route('/train/anomaly/<int:tid>/<token>')
def train_anomaly_confirm(tid, token):
    """训练异常邮件中的确认链接：continue=继续训练，stop=停止训练。"""
    task = db.session.get(TrainingTask, tid)
    if not task or not task.anomaly_token or task.anomaly_token != token:
        abort(404)
    action = request.args.get('action', '')
    if action == 'continue':
        task.anomaly_state = 'confirmed'
        db.session.commit()
        flash(f'已确认继续训练 #{tid}（若再次出现异常会重新提醒）', 'success')
    elif action == 'stop':
        from train_worker import stop as stop_train
        stop_train()
        task.anomaly_state = 'stopped'
        db.session.commit()
        flash(f'已请求停止训练 #{tid}，checkpoint 会自动保存', 'warning')
    else:
        abort(400)
    return redirect(url_for('task_list'))


@app.route('/train-tasks/<int:tid>/delete', methods=['POST'])
@login_required
def train_task_delete(tid):
    task = TrainingTask.query.get_or_404(tid)
    if task.user_id != current_user.id:
        abort(403)
    import shutil as _sh
    td = os.path.join(app.config['UPLOAD_FOLDER'], 'train_data', f'task_{tid}')
    _sh.rmtree(td, ignore_errors=True)
    db.session.delete(task)
    db.session.commit()
    flash('训练任务已删除', 'success')
    return redirect(url_for('task_list'))


# ========== 启动 ==========

def _recover_tasks():
    """重启后恢复任务队列：running 重置为 pending，pending 重新入队。"""
    with app.app_context():
        for t in Task.query.filter(Task.status.in_(['pending', 'running'])).all():
            t.status = 'pending'
        for tr in TrainingTask.query.filter_by(status='running').all():
            tr.status = 'pending'
            tr.progress_msg = '服务重启后重新排队'
        db.session.commit()
        for t in Task.query.filter_by(status='pending').all():
            task_queue.put(t.id)


def _stop_inference_daemon():
    """主进程退出时通知推理 daemon 结束，避免孤儿进程。"""
    global inference_q, inference_daemon_proc
    try:
        if inference_q is not None:
            inference_q.put(None)
        if inference_daemon_proc is not None:
            inference_daemon_proc.join(timeout=3)
    except Exception:
        pass


if __name__ == '__main__':
    import atexit
    atexit.register(_stop_inference_daemon)
    ensure_train_worker()
    ensure_worker()
    _recover_tasks()
    port = int(os.environ.get('PORT', 5000))
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        app.run(host='0.0.0.0', port=port, threaded=True)
