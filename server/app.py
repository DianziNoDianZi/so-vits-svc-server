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
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, session, Response,
)
from flask_login import (
    login_user, logout_user, login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ─── 项目路径 ───
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from authorization import can_manage_model, can_use_model, is_active_user, is_admin
from config import Config
from extensions import db, login_manager, init_sqlite_pragmas
from db_models import DEFAULT_PARAMS, InferenceConfig, Model, ServerSetting, StoredFile, Task, User, UserQuota


def generate_random_password(length=12):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def migrate_db():
    import sqlalchemy as sa
    inspector = sa.inspect(db.engine)
    # migration for inference task (覆盖参数)
    tcols = [c['name'] for c in inspector.get_columns('task')]
    if 'params_json' not in tcols:
        db.session.execute(sa.text('ALTER TABLE task ADD COLUMN params_json TEXT'))
    # migration for user table
    ucols = [c['name'] for c in inspector.get_columns('user')]
    user_cols = [('role', 'VARCHAR(20)'), ('is_active', 'BOOLEAN'), ('must_change_password', 'BOOLEAN'),
                 ('email', 'VARCHAR(200)'), ('notify_email', 'VARCHAR(200)'),
                 ('email_notify', 'BOOLEAN'), ('smtp_user', 'VARCHAR(200)'), ('smtp_pwd', 'VARCHAR(200)'),
                 ('smtp_host', 'VARCHAR(200)'), ('smtp_port', 'INTEGER'),
                 ('report_interval', 'INTEGER'), ('infer_notify', 'BOOLEAN')]
    for col, typ in user_cols:
        if col not in ucols:
            db.session.execute(sa.text(f'ALTER TABLE user ADD COLUMN {col} {typ}'))
    # migration for user_quota table
    db.create_all()
    # migration for model table
    mcols = [c['name'] for c in inspector.get_columns('model')]
    model_cols = [('visibility', 'VARCHAR(20)'), ('status', 'VARCHAR(20)'), ('description', 'VARCHAR(500)'),
                  ('version', 'VARCHAR(100)'), ('review_note', 'VARCHAR(500)'), ('reviewed_at', 'DATETIME'),
                  ('tags', 'VARCHAR(500)')]
    for col, typ in model_cols:
        if col not in mcols:
            db.session.execute(sa.text(f'ALTER TABLE model ADD COLUMN {col} {typ}'))
    # migration for task table
    tcols = [c['name'] for c in inspector.get_columns('task')]
    task_cols = [('model_id', 'INTEGER'), ('input_bytes', 'INTEGER'), ('input_duration', 'FLOAT'),
                 ('attempt_count', 'INTEGER'), ('priority_snapshot', 'INTEGER'), ('quota_snapshot_json', 'TEXT'),
                 ('lease_expires_at', 'DATETIME'), ('heartbeat_at', 'DATETIME'), ('claimed_by', 'VARCHAR(100)'),
                 ('cancel_requested_at', 'DATETIME'), ('cancel_reason', 'VARCHAR(500)'),
                 ('result_expires_at', 'DATETIME')]
    for col, typ in task_cols:
        if col not in tcols:
            db.session.execute(sa.text(f'ALTER TABLE task ADD COLUMN {col} {typ}'))
    # 兼容旧库：新列 ALTER 后历史行是 NULL，必须回填，否则登录被 user_loader/login_user 拒绝
    try:
        db.session.execute(sa.text("UPDATE user SET is_active = 1 WHERE is_active IS NULL"))
        db.session.execute(sa.text("UPDATE user SET role = 'user' WHERE role IS NULL"))
        db.session.execute(sa.text("UPDATE user SET role = 'admin' WHERE username = 'admin'"))
        db.session.execute(sa.text("UPDATE user SET must_change_password = 0 WHERE must_change_password IS NULL"))
    except Exception:
        pass
    db.session.commit()


def init_admin():
    """首次启动创建管理员并生成随机密码；已存在账号绝不重置密码。"""
    admin = User.query.filter_by(username='admin').first()
    if admin:
        if getattr(admin, 'role', 'user') != 'admin':
            admin.role = 'admin'
            db.session.commit()
        return False, None
    password = generate_random_password()
    admin = User(
        username='admin',
        password_hash=generate_password_hash(password),
        role='admin',
        is_active=True,
        must_change_password=True,
    )
    db.session.add(admin)
    db.session.commit()
    if not UserQuota.query.filter_by(user_id=admin.id).first():
        db.session.add(UserQuota(user_id=admin.id, priority=10, max_queued_tasks=10, max_running_tasks=1, daily_audio_seconds=10**9, storage_quota_bytes=10**12, max_model_bytes=10**12, max_private_models=100, results_retention_days=7))
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


def _current_quota(user):
    quota = UserQuota.query.filter_by(user_id=user.id).first()
    if quota:
        return quota
    quota = UserQuota(
        user_id=user.id,
        max_queued_tasks=int(_get_setting('default_max_queued', 4)),
        max_running_tasks=int(_get_setting('default_max_running', 1)),
        max_input_seconds=int(_get_setting('default_max_input_seconds', 600)),
        daily_audio_seconds=int(_get_setting('default_daily_audio_seconds', 3600)),
        storage_quota_bytes=int(_get_setting('default_storage_quota_bytes', 10 * 1024 ** 3)),
        max_model_bytes=int(_get_setting('default_max_model_bytes', 4 * 1024 ** 3)),
        max_private_models=int(_get_setting('default_max_private_models', 3)),
        priority=int(_get_setting('default_priority', 1)),
        results_retention_days=int(_get_setting('default_result_retention_days', 7)),
    )
    db.session.add(quota)
    db.session.commit()
    return quota


def _usable_models_for(user):
    models = Model.query.order_by(Model.created_at.desc()).all()
    if is_admin(user):
        return [m for m in models if getattr(m, 'status', 'ready') != 'disabled']
    return [m for m in models if can_use_model(user, m)]


def _model_visible_to_user(user, model):
    return can_use_model(user, model)


def _today_cutoff():
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day)


_SMTP_ENV_MAP = {
    'smtp_host': 'SMTP_HOST',
    'smtp_port': 'SMTP_PORT',
    'smtp_user': 'SMTP_USER',
    'smtp_pass': 'SMTP_PASS',
    'mail_from': 'MAIL_FROM',
}


def _get_setting(key, default=None):
    env = _SMTP_ENV_MAP.get(key)
    if env and os.environ.get(env):
        return os.environ[env]
    row = db.session.get(ServerSetting, key)
    return row.value if row and row.value else default


def _set_setting(key, value):
    row = db.session.get(ServerSetting, key)
    if row:
        row.value = value
    else:
        db.session.add(ServerSetting(key=key, value=value))
    db.session.commit()


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


# ========== 注册限速（内存级，防刷号） ==========

_register_attempts = {}
_register_lock = threading.Lock()
_REGISTER_MAX = 5
_REGISTER_WINDOW = 3600  # 秒


def _register_blocked(remote_addr):
    with _register_lock:
        info = _register_attempts.get(remote_addr)
        if info and info.get('count', 0) >= _REGISTER_MAX and time.time() < info.get('ts', 0) + _REGISTER_WINDOW:
            return True
        return False


def _record_register(remote_addr):
    with _register_lock:
        now = time.time()
        info = _register_attempts.get(remote_addr)
        if not info or now - info.get('ts', 0) > _REGISTER_WINDOW:
            info = {'count': 0, 'ts': now}
        info['count'] += 1
        info['ts'] = now
        _register_attempts[remote_addr] = info


@login_manager.user_loader
def load_user(user_id):
    try:
        user = db.session.get(User, int(user_id))
        if user and not is_active_user(user):
            return None
        return user
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
                if login_user(user):
                    _clear_login_failures(remote)
                    if getattr(user, 'must_change_password', False):
                        return redirect(url_for('change_password'))
                    return redirect(url_for('dashboard'))
            _record_login_failure(remote)
            flash('用户名或密码错误', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if _get_setting('allow_registration', os.environ.get('ALLOW_REGISTRATION', '1')) != '1':
        flash('当前未开放注册', 'danger')
        return redirect(url_for('login'))
    if request.method == 'POST':
        remote = request.remote_addr or 'unknown'
        if _register_blocked(remote):
            flash('注册尝试过于频繁，请稍后再试', 'danger')
            return render_template('register.html')

        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        email = request.form.get('email', '').strip()
        notify_email = request.form.get('notify_email', '').strip()

        errors = []
        if len(username) < 2:
            errors.append('用户名至少 2 个字符')
        if len(password) < 6:
            errors.append('密码至少 6 位')
        if password != confirm:
            errors.append('两次密码不一致')
        if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
            errors.append('邮箱格式不正确')
        if notify_email and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', notify_email):
            errors.append('结果接收邮箱格式不正确')
        if not errors and User.query.filter_by(username=username).first():
            errors.append('用户名已存在')

        if errors:
            _record_register(remote)
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', username=username, email=email, notify_email=notify_email)

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role='user',
            is_active=True,
            email=email or None,
            notify_email=notify_email or None,
            infer_notify=True,
        )
        db.session.add(user)
        db.session.commit()
        _current_quota(user)  # 生成默认配额
        login_user(user)
        try:
            from notifier import notify_welcome
            notify_welcome(user)
        except Exception:
            pass
        flash('注册成功，欢迎使用', 'success')
        return redirect(url_for('dashboard'))
    return render_template('register.html')


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

        current_user.must_change_password = False
        db.session.commit()
        flash('设置已保存', 'success')
        return redirect(url_for('dashboard'))

    return render_template('change_password.html')


@app.route('/save-notify', methods=['POST'])
@login_required
def save_notify():
    current_user.email = request.form.get('email', '').strip() or None
    current_user.notify_email = request.form.get('notify_email', '').strip() or None
    current_user.infer_notify = request.form.get('infer_notify') == '1'
    db.session.commit()
    flash('通知设置已保存', 'success')
    return redirect(url_for('settings'))


@app.route('/test-notify', methods=['POST'])
@login_required
def test_notify():
    from notifier import send_via_server
    u = current_user
    recipient = getattr(u, 'notify_email', None) or u.email
    if not recipient:
        flash('请先填写接收邮箱', 'danger')
        return redirect(url_for('settings'))
    ok = send_via_server(recipient, '[SoVITS] 测试通知', '这是一封测试邮件，通知配置正常！')
    if ok:
        flash('测试邮件已发送，请检查收件箱', 'success')
    else:
        flash('发送失败，请检查服务器 SMTP 配置', 'danger')
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
            return render_template('settings.html')

        # 通知设置
        current_user.email = request.form.get('email', '').strip() or None
        current_user.notify_email = request.form.get('notify_email', '').strip() or None
        current_user.infer_notify = request.form.get('infer_notify') == '1'

        if new_username and new_username != current_user.username:
            existing = User.query.filter_by(username=new_username).first()
            if existing:
                flash('用户名已存在', 'danger')
                return render_template('settings.html')
            current_user.username = new_username

        if new:
            if len(new) < 6:
                flash('新密码至少 6 位', 'danger')
                return render_template('settings.html')
            if new != confirm:
                flash('两次密码不一致', 'danger')
                return render_template('settings.html')
            current_user.password_hash = generate_password_hash(new)
            current_user.must_change_password = False

        db.session.commit()
        flash('设置已保存', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ========== 仪表盘 ==========

def _status_label(status):
    return {
        'pending': '排队中',
        'claimed': '待执行',
        'running': '运行中',
        'cancel_requested': '正在停止',
        'done': '已完成',
        'failed': '失败',
        'stopped': '已停止',
        'expired': '已过期',
    }.get(status, status)


def _queue_position(task):
    """返回同用户排队任务中该任务前面还有多少个。"""
    return Task.query.filter(
        Task.user_id == task.user_id,
        Task.id < task.id,
        Task.status.in_(['pending', 'claimed']),
    ).count()


@app.route('/')
@login_required
def dashboard():
    quota = _current_quota(current_user)
    today = _today_cutoff()
    running_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status == 'running').count()
    queued_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status.in_(['pending', 'claimed'])).count()
    used_today = db.session.query(sa.func.coalesce(sa.func.sum(Task.input_duration), 0)).filter(
        Task.user_id == current_user.id,
        Task.created_at >= today,
        Task.status.in_(['pending', 'claimed', 'running', 'done', 'failed', 'stopped']),
    ).scalar() or 0
    done_today = Task.query.filter(
        Task.user_id == current_user.id, Task.status == 'done', Task.done_at >= today).count()
    fail_today = Task.query.filter(
        Task.user_id == current_user.id, Task.status == 'failed', Task.done_at >= today).count()
    models = _usable_models_for(current_user)
    private_models = [m for m in models if m.visibility == 'private']
    official_models = [m for m in models if m.visibility == 'official']
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).all()
    recent = (Task.query.filter_by(user_id=current_user.id)
              .order_by(Task.created_at.desc()).limit(6).all())
    now = datetime.utcnow()
    recent_items = []
    for t in recent:
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        expired = (t.status == 'done' and t.result_expires_at and t.result_expires_at < now)
        recent_items.append({
            'id': t.id, 'status': t.status,
            'status_label': '已过期' if expired else _status_label(t.status),
            'model': model_name, 'progress_msg': t.progress_msg or '',
            'created': t.created_at, 'has_result': bool(t.result_filename),
            'can_download': t.status == 'done' and bool(t.result_filename) and not expired,
            'can_stop': t.status in ('pending', 'claimed', 'running'),
        })
    return render_template('dashboard.html',
        quota=quota,
        running_cnt=running_cnt, queued_cnt=queued_cnt,
        used_today=int(used_today), daily_limit=quota.daily_audio_seconds or 0,
        done_today=done_today, fail_today=fail_today,
        models=models, private_cnt=len(private_models), official_cnt=len(official_models),
        config_cnt=len(configs),
        recent=recent_items)


# ========== 模型管理 ==========

def _arch_label_from_cfg(cfg):
    """从 config dict 提取架构展示标签。返回 (label, sub)"""
    m = cfg.get('model') or {}
    arch = m.get('arch', '')
    if arch == 'rvc-flow':
        fm = m.get('flow_mode', 'a2')
        uni = m.get('use_unified_flow', False)
        if fm == 'a1':
            return 'A1 特征先验流', 'a1'
        return 'A2 后验流' + (' + 统一流' if uni else ''), 'a2'
    if arch == 'rvc':
        return 'RVC 轻量架构', 'rvc'
    return 'SoVITS v1', 'sovits-v1'


def _read_model_cfg(cfg_path):
    """读取 uploads/configs 下的模型 config，失败返回 {}"""
    if not cfg_path:
        return {}
    cfg_p = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', cfg_path)
    if not os.path.exists(cfg_p):
        return {}
    try:
        with open(cfg_p, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}


@app.route('/models')
@login_required
def model_list():
    models = _usable_models_for(current_user)
    # 读取每个模型的架构（用于列表筛选）
    model_items = []
    for m in models:
        cfg = _read_model_cfg(m.config_path)
        arch = (cfg.get('model') or {}).get('arch', '')
        label, sub = _arch_label_from_cfg(cfg)
        onnx_exists = False
        if m.model_path:
            onnx_exists = os.path.exists(
                os.path.join(app.config['UPLOAD_FOLDER'], 'models', m.model_path + '.onnx'))
        model_items.append({
            'm': m, 'arch': arch or 'sovits-v1', 'arch_label': label, 'flow_mode': sub,
            'status': getattr(m, 'status', 'ready'), 'visibility': getattr(m, 'visibility', 'private'),
            'c_kl': (cfg.get('train') or {}).get('c_kl'),
            'tags': [t.strip() for t in (m.tags or '').split(',') if t.strip()],
            'onnx': onnx_exists,
            'can_manage': can_manage_model(current_user, m),
            'can_export': can_manage_model(current_user, m),
        })
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

        quota = _current_quota(current_user)
        if not quota.enabled:
            flash('当前账号已被禁用，无法上传模型', 'danger')
            return render_template('model_upload.html')
        private_cnt = Model.query.filter(
            Model.user_id == current_user.id,
            Model.visibility == 'private',
            Model.status.in_(['pending_review', 'ready']),
        ).count()
        if private_cnt >= quota.max_private_models:
            flash('已达到私有模型数量上限（被拒绝的模型不占名额）', 'danger')
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

        is_official = is_admin(current_user) and request.form.get('visibility') == 'official'
        m = Model(
            user_id=current_user.id,
            name=name,
            visibility='official' if is_official else 'private',
            status='ready' if is_official else 'pending_review',
            model_path=model_path,
            config_path=config_path,
            diff_model_path=diff_path,
            diff_config_path=diff_config_path,
            cluster_path=cluster_path,
        )
        db.session.add(m)
        db.session.commit()
        flash('模型已上传，等待管理员审核', 'success')
        return redirect(url_for('model_list'))
    return render_template('model_upload.html')


@app.route('/models/<int:model_id>/edit', methods=['GET', 'POST'])
@login_required
def model_edit(model_id):
    m = Model.query.get_or_404(model_id)
    if not can_manage_model(current_user, m):
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
        m.tags = request.form.get('tags', '').strip() or None
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
    if not can_manage_model(current_user, m):
        abort(403)
    # Delete associated configs
    InferenceConfig.query.filter_by(model_id=m.id).delete()
    # Delete files
    for attr in ['model_path', 'config_path', 'diff_model_path', 'diff_config_path', 'cluster_path']:
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


@app.route('/models/<int:model_id>/review', methods=['POST'])
@login_required
def model_review(model_id):
    if not is_admin(current_user):
        abort(403)
    m = Model.query.get_or_404(model_id)
    action = request.form.get('action', '')
    note = request.form.get('note', '').strip() or None
    if action == 'approve':
        m.status = 'ready'
        m.review_note = note
        m.reviewed_at = datetime.utcnow()
        flash(f'模型 #{m.id} 已通过审核', 'success')
    elif action == 'reject':
        m.status = 'rejected'
        m.review_note = note
        m.reviewed_at = datetime.utcnow()
        flash(f'模型 #{m.id} 已拒绝', 'warning')
    elif action == 'disable':
        m.status = 'disabled'
        m.review_note = note or m.review_note
        flash(f'模型 #{m.id} 已下架', 'warning')
    elif action == 'enable':
        m.status = 'ready'
        m.review_note = note or m.review_note
        flash(f'模型 #{m.id} 已上架', 'success')
    elif action == 'official':
        m.status = 'ready'
        m.visibility = 'official'
        m.review_note = note or m.review_note
        m.reviewed_at = datetime.utcnow()
        flash(f'模型 #{m.id} 已发布为官方模型', 'success')
    else:
        abort(400)
    db.session.commit()
    return redirect(url_for('admin_models'))


@app.route('/models/<int:model_id>/export_onnx', methods=['POST'])
@login_required
def model_export_onnx(model_id):
    """导出模型为 ONNX（主生成器），推理时自动用 onnxruntime；失败不影响原推理。"""
    m = db.session.get(Model, model_id)
    if not m or not can_manage_model(current_user, m):
        abort(404)
    if not m.model_path or not m.config_path:
        flash('模型缺少模型文件或配置，无法导出', 'danger')
        return redirect(url_for('model_list'))
    model_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models', m.model_path)
    config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', m.config_path)
    out_file = model_file + '.onnx'
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        flash('模型或配置文件不存在', 'danger')
        return redirect(url_for('model_list'))
    import subprocess as _sp
    script = os.path.join(PROJECT_DIR, 'onnx_export_generator.py')
    try:
        r = _sp.run(
            [sys.executable, '-X', 'utf8', script, model_file, config_file, out_file],
            cwd=PROJECT_DIR, capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and os.path.exists(out_file):
            flash(f'ONNX 导出成功（{os.path.getsize(out_file) / 1048576:.0f}MB），推理将自动使用', 'success')
        else:
            err = (r.stderr or r.stdout or '')[-400:]
            flash(f'ONNX 导出失败：{err}', 'danger')
    except Exception as e:
        flash(f'ONNX 导出异常：{e}', 'danger')
    return redirect(url_for('model_list'))


# ========== 管理员 ==========

def _dir_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total / 1048576


@app.route('/admin')
@login_required
def admin_index():
    if not is_admin(current_user):
        abort(403)
    today = _today_cutoff()
    uploads = app.config['UPLOAD_FOLDER']
    disk = {
        'models': _dir_size_mb(os.path.join(uploads, 'models')),
        'configs': _dir_size_mb(os.path.join(uploads, 'configs')),
        'audio': _dir_size_mb(os.path.join(uploads, 'audio')),
        'results': _dir_size_mb(os.path.join(uploads, 'results')),
        'dataset_zips': _dir_size_mb(os.path.join(uploads, 'dataset_zips')),
        'train_data': _dir_size_mb(os.path.join(uploads, 'train_data')),
    }
    running_tasks = []
    for t in Task.query.filter(Task.status.in_(['claimed', 'running'])).order_by(Task.created_at.asc()).all():
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        running_tasks.append({
            'id': t.id, 'user': t.user.username if t.user else '?', 'model': model_name,
            'status': t.status, 'created': t.created_at,
        })
    stats = {
        'users': User.query.count(),
        'active_users': User.query.filter_by(is_active=True).count(),
        'pending': Task.query.filter(Task.status == 'pending').count(),
        'claimed': Task.query.filter(Task.status == 'claimed').count(),
        'running': Task.query.filter(Task.status == 'running').count(),
        'done_today': Task.query.filter(Task.status == 'done', Task.done_at >= today).count(),
        'failed_today': Task.query.filter(Task.status == 'failed', Task.done_at >= today).count(),
        'pending_models': Model.query.filter(Model.status == 'pending_review').count(),
        'official_models': Model.query.filter(Model.visibility == 'official').count(),
        'total_models': Model.query.count(),
    }
    return render_template('admin_overview.html', stats=stats, running_tasks=running_tasks, disk=disk)


@app.route('/admin/users')
@login_required
def admin_users():
    if not is_admin(current_user):
        abort(403)
    users = User.query.order_by(User.id.asc()).all()
    items = []
    for u in users:
        q = _current_quota(u)
        items.append({
            'user': u,
            'quota': q,
            'queued': Task.query.filter(Task.user_id == u.id, Task.status.in_(['pending', 'claimed', 'running'])).count(),
            'private_models': Model.query.filter(Model.user_id == u.id, Model.visibility == 'private').count(),
        })
    return render_template('admin_users.html', items=items)


@app.route('/admin/users/<int:uid>/quota', methods=['POST'])
@login_required
def admin_update_quota(uid):
    if not is_admin(current_user):
        abort(403)
    u = db.session.get(User, uid)
    if not u:
        abort(404)
    q = _current_quota(u)

    def _i(k, d):
        try:
            return int(request.form.get(k, d) or d)
        except (ValueError, TypeError):
            return d

    q.enabled = request.form.get('enabled') == 'on'
    q.max_queued_tasks = _i('max_queued_tasks', 4)
    q.max_running_tasks = _i('max_running_tasks', 1)
    q.max_input_seconds = _i('max_input_seconds', 600)
    q.daily_audio_seconds = _i('daily_audio_seconds', 3600)
    q.storage_quota_bytes = _i('storage_quota_bytes', 10 * 1024 ** 3)
    q.max_model_bytes = _i('max_model_bytes', 4 * 1024 ** 3)
    q.max_private_models = _i('max_private_models', 3)
    q.priority = _i('priority', 1)
    q.results_retention_days = _i('results_retention_days', 7)
    u.is_active = request.form.get('active') == 'on'
    u.role = 'admin' if request.form.get('is_admin') == 'on' else 'user'
    db.session.commit()
    flash(f'已更新用户 {u.username} 的配额', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/tasks')
@login_required
def admin_tasks():
    if not is_admin(current_user):
        abort(403)
    q = Task.query
    uid = request.args.get('user_id', type=int)
    status = request.args.get('status', '').strip()
    if uid:
        q = q.filter(Task.user_id == uid)
    if status:
        q = q.filter(Task.status == status)
    tasks = q.order_by(Task.created_at.desc()).limit(200).all()
    items = []
    for t in tasks:
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        items.append({
            't': t,
            'user': t.user.username if t.user else '?',
            'model': model_name,
            'status_label': _status_label(t.status),
            'can_stop': t.status in ('pending', 'claimed', 'running'),
        })
    users = User.query.order_by(User.username.asc()).all()
    return render_template('admin_tasks.html', items=items, users=users, cur_uid=uid, cur_status=status)


@app.route('/admin/tasks/<int:task_id>/stop', methods=['POST'])
@login_required
def admin_task_stop(task_id):
    if not is_admin(current_user):
        abort(403)
    t = db.session.get(Task, task_id)
    if not t:
        abort(404)
    if t.status in ('claimed', 'running'):
        t.status = 'cancel_requested'
        t.progress_msg = '管理员请求停止'
        db.session.commit()
        flash(f'已请求停止任务 #{task_id}', 'warning')
    elif t.status == 'pending':
        t.status = 'stopped'
        t.progress_msg = '管理员停止'
        t.done_at = datetime.utcnow()
        db.session.commit()
        flash(f'已停止任务 #{task_id}', 'warning')
    else:
        flash('该任务不在运行/排队中', 'info')
    return redirect(url_for('admin_tasks'))


@app.route('/admin/tasks/<int:task_id>/delete', methods=['POST'])
@login_required
def admin_task_delete(task_id):
    if not is_admin(current_user):
        abort(403)
    t = db.session.get(Task, task_id)
    if not t:
        abort(404)
    if t.result_filename:
        p = os.path.join(app.config['UPLOAD_FOLDER'], 'results', t.result_filename)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    db.session.delete(t)
    db.session.commit()
    flash(f'已删除任务 #{task_id}', 'success')
    return redirect(url_for('admin_tasks'))


@app.route('/admin/models')
@login_required
def admin_models():
    if not is_admin(current_user):
        abort(403)
    models = Model.query.order_by(Model.created_at.desc()).all()
    # 待审核置顶，方便审核用户上传的模型
    models.sort(key=lambda m: (0 if m.status == 'pending_review' else 1))
    items = []
    for m in models:
        cfg = _read_model_cfg(m.config_path)
        arch = (cfg.get('model') or {}).get('arch', '')
        label, sub = _arch_label_from_cfg(cfg)
        items.append({
            'm': m,
            'owner': m.owner.username if m.owner else '—',
            'arch_label': label,
            'flow_mode': sub,
            'status': m.status,
            'visibility': m.visibility,
        })
    return render_template('admin_models.html', items=items)


@app.route('/admin/storage')
@login_required
def admin_storage():
    if not is_admin(current_user):
        abort(403)
    uploads = app.config['UPLOAD_FOLDER']
    disk = {
        'models': _dir_size_mb(os.path.join(uploads, 'models')),
        'configs': _dir_size_mb(os.path.join(uploads, 'configs')),
        'audio': _dir_size_mb(os.path.join(uploads, 'audio')),
        'results': _dir_size_mb(os.path.join(uploads, 'results')),
        'dataset_zips': _dir_size_mb(os.path.join(uploads, 'dataset_zips')),
        'train_data': _dir_size_mb(os.path.join(uploads, 'train_data')),
    }
    total = sum(disk.values())
    refs_models = set()
    refs_configs = set()
    for m in Model.query.all():
        refs_models |= {m.model_path, m.diff_model_path, m.cluster_path}
        refs_configs |= {m.config_path, m.diff_config_path}
    refs_results = {t.result_filename for t in Task.query.all()}
    refs_audio = {t.audio_filename for t in Task.query.all()}

    def orphans(sub, refs):
        d = os.path.join(uploads, sub)
        out = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if os.path.isfile(os.path.join(d, f)) and f not in refs:
                    out.append(f)
        return out

    orphan = {
        'models': orphans('models', refs_models),
        'configs': orphans('configs', refs_configs),
        'results': orphans('results', refs_results),
        'audio': orphans('audio', refs_audio),
    }
    return render_template('admin_storage.html', disk=disk, total=total, orphan=orphan)


@app.route('/admin/storage/delete', methods=['POST'])
@login_required
def admin_storage_delete():
    if not is_admin(current_user):
        abort(403)
    sub = request.form.get('sub', '')
    name = request.form.get('name', '').strip()
    if sub not in ('models', 'configs', 'results', 'audio') or not name or os.path.basename(name) != name:
        flash('非法参数', 'danger')
        return redirect(url_for('admin_storage'))
    # 二次校验仍是孤儿，防止误删
    refs_models = set()
    refs_configs = set()
    for m in Model.query.all():
        refs_models |= {m.model_path, m.diff_model_path, m.cluster_path}
        refs_configs |= {m.config_path, m.diff_config_path}
    refs = {
        'models': refs_models, 'configs': refs_configs,
        'results': {t.result_filename for t in Task.query.all()},
        'audio': {t.audio_filename for t in Task.query.all()},
    }
    if name in refs.get(sub, set()):
        flash(f'{name} 已被引用，无法删除', 'danger')
        return redirect(url_for('admin_storage'))
    p = os.path.join(app.config['UPLOAD_FOLDER'], sub, name)
    if os.path.exists(p):
        try:
            os.remove(p)
            flash(f'已删除孤儿文件 {name}', 'success')
        except OSError as e:
            flash(f'删除失败: {e}', 'danger')
    else:
        flash('文件不存在', 'warning')
    return redirect(url_for('admin_storage'))


@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if not is_admin(current_user):
        abort(403)
    if request.method == 'POST':
        if request.form.get('action') == 'test':
            recipient = current_user.notify_email or current_user.email
            if not recipient:
                flash('请先在设置页填写接收邮箱', 'danger')
                return redirect(url_for('admin_settings'))
            from notifier import send_via_server
            ok = send_via_server(recipient, '[SoVITS] SMTP 测试', '服务器 SMTP 配置正常！')
            flash('测试邮件已发送，请检查收件箱' if ok else '测试发送失败，请检查 SMTP 配置', 'success' if ok else 'danger')
            return redirect(url_for('admin_settings'))
        if request.form.get('action') == 'site':
            _set_setting('allow_registration', '1' if request.form.get('allow_registration') == 'on' else '0')
            _set_setting('default_max_queued', request.form.get('default_max_queued', '4'))
            _set_setting('default_max_running', request.form.get('default_max_running', '1'))
            _set_setting('default_max_input_seconds', request.form.get('default_max_input_seconds', '600'))
            _set_setting('default_daily_audio_seconds', request.form.get('default_daily_audio_seconds', '3600'))
            _set_setting('default_max_private_models', request.form.get('default_max_private_models', '3'))
            _set_setting('default_result_retention_days', request.form.get('default_result_retention_days', '7'))
            flash('站点设置已保存（对新注册用户生效）', 'success')
            return redirect(url_for('admin_settings'))
        _set_setting('smtp_host', request.form.get('smtp_host', '').strip())
        _set_setting('smtp_port', request.form.get('smtp_port', '465').strip() or '465')
        _set_setting('smtp_user', request.form.get('smtp_user', '').strip())
        pwd = request.form.get('smtp_pass', '').strip()
        if pwd:
            _set_setting('smtp_pass', pwd)
        _set_setting('mail_from', request.form.get('mail_from', '').strip())
        flash('SMTP 配置已保存', 'success')
        return redirect(url_for('admin_settings'))
    cfg = {
        'smtp_host': _get_setting('smtp_host', ''),
        'smtp_port': _get_setting('smtp_port', '465'),
        'smtp_user': _get_setting('smtp_user', ''),
        'mail_from': _get_setting('mail_from', ''),
    }
    site = {
        'allow_registration': _get_setting('allow_registration', os.environ.get('ALLOW_REGISTRATION', '1')) == '1',
        'default_max_queued': _get_setting('default_max_queued', 4),
        'default_max_running': _get_setting('default_max_running', 1),
        'default_max_input_seconds': _get_setting('default_max_input_seconds', 600),
        'default_daily_audio_seconds': _get_setting('default_daily_audio_seconds', 3600),
        'default_max_private_models': _get_setting('default_max_private_models', 3),
        'default_result_retention_days': _get_setting('default_result_retention_days', 7),
    }
    return render_template('admin_settings.html', cfg=cfg, site=site)


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
    models = _usable_models_for(current_user)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        model_id = request.form.get('model_id', type=int)
        if not name or not model_id:
            flash('请填写名称并选择模型', 'danger')
            return render_template('config_create.html', models=models, params=DEFAULT_PARAMS.copy())
        m = db.session.get(Model, model_id)
        if not m or not can_use_model(current_user, m):
            flash('无效或不可用的模型', 'danger')
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
    models = _usable_models_for(current_user)
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
task_scheduler_started = False
task_processes = {}  # task_id -> subprocess.Popen
task_processes_lock = threading.Lock()
inference_active_id = None
scheduler_last_user_id = None
scheduler_lock = threading.Lock()
cleanup_started = False

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


_STALE_LEASE_SECONDS = 5 * 60  # 心跳超过 5 分钟未更新视为执行器失联


def _recover_expired_leases():
    """回收执行器失联的任务：claimed/running 且心跳过期 → 重新排队（attempt+1）。仅回收时写库。"""
    now = datetime.utcnow()
    candidates = Task.query.filter(Task.status.in_(['claimed', 'running'])).all()
    changed = False
    for t in candidates:
        hb = t.heartbeat_at
        lease = t.lease_expires_at
        stale_hb = hb is None or (now - hb).total_seconds() > _STALE_LEASE_SECONDS
        stale_lease = lease is not None and lease < now
        if stale_hb or stale_lease:
            t.status = 'pending'
            t.claimed_by = None
            t.lease_expires_at = None
            t.heartbeat_at = None
            t.attempt_count = (t.attempt_count or 0) + 1
            t.progress_msg = '执行器失联，重新排队'
            changed = True
    if changed:
        db.session.commit()


def task_scheduler_daemon():
    """按用户公平轮转从数据库领取待执行推理任务。"""
    global scheduler_last_user_id
    with app.app_context():
        while True:
            try:
                _recover_expired_leases()
                if inference_active_id is not None or not task_queue.empty():
                    time.sleep(1)
                    continue

                pending = (Task.query.filter(Task.status == 'pending')
                           .order_by(Task.created_at.asc(), Task.id.asc()).all())
                if not pending:
                    time.sleep(1)
                    continue

                eligible = []
                for t in pending:
                    if not t.user or not is_active_user(t.user):
                        continue
                    quota = _current_quota(t.user)
                    if not quota.enabled:
                        continue
                    running = Task.query.filter(Task.user_id == t.user_id, Task.status == 'running').count()
                    queued = Task.query.filter(Task.user_id == t.user_id, Task.status.in_(['pending', 'claimed', 'running'])).count()
                    if quota.max_running_tasks and running >= quota.max_running_tasks:
                        continue
                    if quota.max_queued_tasks and queued >= quota.max_queued_tasks:
                        continue
                    if t.model_id:
                        model = db.session.get(Model, t.model_id)
                        if not model or not can_use_model(t.user, model):
                            t.status = 'failed'
                            t.error_msg = '模型不可用'
                            t.progress_msg = '失败: 模型不可用'
                            t.done_at = datetime.utcnow()
                            db.session.commit()
                            continue
                    eligible.append((t, quota))

                if not eligible:
                    time.sleep(1)
                    continue

                picked = None
                if scheduler_last_user_id is not None:
                    for t, quota in eligible:
                        if t.user_id != scheduler_last_user_id:
                            picked = (t, quota)
                            break
                if picked is None:
                    # 按优先级和等待时间选择
                    eligible.sort(key=lambda x: (-int(x[1].priority or 1), x[0].created_at, x[0].id))
                    picked = eligible[0]

                task, quota = picked
                task.status = 'claimed'
                task.progress_msg = '等待执行器...'
                task.lease_expires_at = datetime.utcnow() + timedelta(minutes=10)
                task.claimed_by = 'scheduler'
                task.priority_snapshot = quota.priority
                db.session.commit()
                scheduler_last_user_id = task.user_id
                task_queue.put(task.id)
            except Exception:
                traceback.print_exc()
                time.sleep(2)



def task_worker():
    """后台任务处理线程"""
    global inference_active_id
    with app.app_context():
        while True:
            task_id = task_queue.get()
            task = db.session.get(Task, task_id)
            if not task:
                continue
            try:
                # 排队期间被用户停止的任务：跳过不投递
                if task.status in ('stopped', 'cancel_requested'):
                    if task.status == 'cancel_requested':
                        task.status = 'stopped'
                        task.progress_msg = '已停止（排队中取消）'
                        task.done_at = datetime.utcnow()
                        db.session.commit()
                    continue
                inference_active_id = task.id
                task.status = 'running'
                task.progress_msg = '正在加载模型...'
                task.heartbeat_at = datetime.utcnow()
                task.lease_expires_at = datetime.utcnow() + timedelta(hours=6)
                db.session.commit()

                cfg_obj = db.session.get(InferenceConfig, task.config_id)
                model = db.session.get(Model, cfg_obj.model_id) if cfg_obj else None
                if not cfg_obj or not model or not is_active_user(task.user) or not can_use_model(task.user, model):
                    task.status = 'failed'
                    task.error_msg = '模型不可用'
                    task.progress_msg = '失败: 模型不可用'
                    task.done_at = datetime.utcnow()
                    db.session.commit()
                    continue

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
                hb_count = 0
                while True:
                    hb_count += 1
                    if hb_count % 30 == 0:
                        task.heartbeat_at = datetime.utcnow()
                        try:
                            db.session.commit()
                        except Exception:
                            db.session.rollback()
                    # 用户/管理员取消：每 5 秒重读状态，中断 daemon 并停止当前任务
                    if hb_count % 5 == 0:
                        try:
                            cur = db.session.get(Task, task_id)
                        except Exception:
                            cur = None
                        if cur and cur.status == 'cancel_requested':
                            if inference_daemon_proc and inference_daemon_proc.is_alive():
                                try:
                                    os.kill(inference_daemon_proc.pid, signal.SIGINT)
                                except Exception:
                                    pass
                            result_err = '用户停止推理'
                            break
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
                inference_active_id = None
                try:
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass


def ensure_worker():
    global task_worker_started, task_scheduler_started, inference_q, inference_done_q, inference_daemon_proc, cleanup_started
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
    if not task_scheduler_started:
        task_scheduler_started = True
        t = threading.Thread(target=task_scheduler_daemon, daemon=True)
        t.start()
    if not task_worker_started:
        task_worker_started = True
        t = threading.Thread(target=task_worker, daemon=True)
        t.start()
    if not cleanup_started:
        cleanup_started = True
        t = threading.Thread(target=_cleanup_daemon, daemon=True)
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
        model = db.session.get(Model, cfg_obj.model_id)
        if not model or not can_use_model(current_user, model):
            flash('所选模型当前不可用', 'danger')
            return render_template('inference.html', configs=configs)

        quota = _current_quota(current_user)
        if not quota.enabled:
            flash('当前账号已被禁用，无法提交推理', 'danger')
            return render_template('inference.html', configs=configs)

        # Save audio
        audio_filename = save_uploaded(audio_file, 'audio')
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', audio_filename)
        try:
            import soundfile as sf
            audio_info = sf.info(audio_path)
            audio_duration = float(getattr(audio_info, 'duration', 0) or 0)
            audio_size = os.path.getsize(audio_path)
        except Exception:
            audio_duration = 0
            audio_size = os.path.getsize(audio_path)

        if quota.max_input_seconds and audio_duration > quota.max_input_seconds:
            os.remove(audio_path)
            flash(f'音频时长超过上限（{int(quota.max_input_seconds)} 秒）', 'danger')
            return render_template('inference.html', configs=configs)

        queued_count = Task.query.filter(Task.user_id == current_user.id, Task.status.in_(['pending', 'running'])).count()
        if quota.max_queued_tasks and queued_count >= quota.max_queued_tasks:
            os.remove(audio_path)
            flash('当前排队/运行任务数已达到上限', 'danger')
            return render_template('inference.html', configs=configs)

        used_today = db.session.query(sa.func.coalesce(sa.func.sum(Task.input_duration), 0)).filter(
            Task.user_id == current_user.id,
            Task.created_at >= _today_cutoff(),
            Task.status.in_(['pending', 'running', 'done', 'failed', 'stopped']),
        ).scalar() or 0
        if quota.daily_audio_seconds and used_today + audio_duration > quota.daily_audio_seconds:
            os.remove(audio_path)
            flash('今日可推理音频秒数已用完', 'danger')
            return render_template('inference.html', configs=configs)

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
            model_id=model.id,
            audio_filename=audio_filename,
            params_json=json.dumps(cfg_params, ensure_ascii=False),
            device_pref=current_user.device_pref or 'auto',
            memory_limit=current_user.memory_limit or 0,
            input_bytes=audio_size,
            input_duration=audio_duration,
            priority_snapshot=quota.priority,
            quota_snapshot_json=json.dumps({
                'max_queued_tasks': quota.max_queued_tasks,
                'max_running_tasks': quota.max_running_tasks,
                'max_input_seconds': quota.max_input_seconds,
                'daily_audio_seconds': quota.daily_audio_seconds,
                'results_retention_days': quota.results_retention_days,
            }, ensure_ascii=False),
            result_expires_at=datetime.utcnow() + timedelta(days=max(quota.results_retention_days or 7, 1)),
            status='pending',
        )
        db.session.add(task)
        db.session.commit()

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
    hidden = 0
    for c in configs:
        if not can_use_model(current_user, c.model):
            hidden += 1
            continue
        try:
            c_params = json.loads(c.params_json or '{}')
        except Exception:
            c_params = {}
        # 读取模型 config 详情（架构 / flow / 步数）
        m_info = {'arch': '', 'flow_mode': '', 'use_unified_flow': False, 'step': ''}
        if c.model.config_path:
            cfg_p = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', c.model.config_path)
            if os.path.exists(cfg_p):
                try:
                    with open(cfg_p, 'r', encoding='utf-8') as f:
                        _cfg = json.load(f)
                    _m = _cfg.get('model') or {}
                    m_info['arch'] = _m.get('arch', '') or 'sovits-v1'
                    m_info['flow_mode'] = _m.get('flow_mode', '') or ('a2' if _m.get('use_unified_flow') else '')
                    m_info['use_unified_flow'] = bool(_m.get('use_unified_flow'))
                except Exception:
                    pass
        # 从模型名解析 step 数（如 Aris-统一流-A1-G6000 → 6000）
        import re as _re
        _step_match = _re.search(r'G(\d+)step|G(\d+)', c.model.name)
        if _step_match:
            m_info['step'] = _step_match.group(1) or _step_match.group(2)
        config_items.append({'id': c.id, 'name': c.name, 'model': c.model.name,
                             'model_id': c.model.id,
                             'params': c_params,
                             'has_cluster': bool(c.model.cluster_path),
                             'model_info': m_info})
    if hidden:
        flash(f'{hidden} 个配置因模型不可用已隐藏', 'warning')
    return render_template('inference.html', configs=config_items, recent=recent_items)


@app.route('/tasks')
@login_required
def task_list():
    tasks = Task.query.filter_by(user_id=current_user.id).order_by(Task.created_at.desc()).all()
    now = datetime.utcnow()
    items = []
    for t in tasks:
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        expired = (t.status == 'done' and t.result_expires_at and t.result_expires_at < now)
        items.append({
            't': t,
            'model': model_name,
            'status_label': '已过期' if expired else _status_label(t.status),
            'queue_pos': _queue_position(t) if t.status in ('pending', 'claimed') else 0,
            'result_expires': t.result_expires_at,
            'can_download': t.status == 'done' and bool(t.result_filename) and not expired,
            'can_stop': t.status in ('pending', 'claimed', 'running'),
            'can_delete': True,
        })
    return render_template('tasks.html', tasks=items)


@app.route('/tasks/<int:task_id>/result')
@login_required
def task_result(task_id):
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    if not task.result_filename:
        abort(404)
    if task.result_expires_at and task.result_expires_at < datetime.utcnow():
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
    """停止推理任务：排队中直接停；运行中置 cancel_requested，由 worker 统一中断 daemon。"""
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)
    if task.status == 'running':
        task.status = 'cancel_requested'
        task.progress_msg = '正在停止...'
        db.session.commit()
        flash('已请求停止推理任务（当前片段处理完即停）', 'warning')
    elif task.status in ('pending', 'claimed'):
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
    task = Task.query.filter_by(result_filename=secure_filename(filename), user_id=current_user.id).first()
    if not task:
        abort(404)
    return task_result(task.id)


def _recover_tasks():
    """重启后恢复任务状态：由 scheduler 重新领取，不直接入执行队列。"""
    with app.app_context():
        for t in Task.query.filter(Task.status.in_(['pending', 'claimed', 'running'])).all():
            t.status = 'pending'
            t.claimed_by = None
            t.lease_expires_at = None
            t.progress_msg = '服务重启后重新排队'
        db.session.commit()


def _cleanup_daemon():
    """周期性清理：过期结果文件、历史输入音频。"""
    with app.app_context():
        while True:
            try:
                time.sleep(3600)
                now = datetime.utcnow()
                expired = Task.query.filter(
                    Task.status == 'done',
                    Task.result_expires_at.isnot(None),
                    Task.result_expires_at < now,
                ).all()
                for t in expired:
                    if t.result_filename:
                        p = os.path.join(app.config['UPLOAD_FOLDER'], 'results', t.result_filename)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        t.result_filename = None
                    if t.audio_filename:
                        a = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', t.audio_filename)
                        if os.path.exists(a):
                            try:
                                os.remove(a)
                            except OSError:
                                pass
                    t.result_expires_at = None
                cutoff = now - timedelta(days=1)
                old = Task.query.filter(
                    Task.status.in_(['done', 'failed', 'stopped']),
                    Task.done_at.isnot(None),
                    Task.done_at < cutoff,
                    Task.audio_filename.isnot(None),
                ).all()
                for t in old:
                    a = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', t.audio_filename)
                    if os.path.exists(a):
                        try:
                            os.remove(a)
                        except OSError:
                            pass
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass
                time.sleep(60)


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
    ensure_worker()
    _recover_tasks()
    port = int(os.environ.get('PORT', 5000))
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    except ImportError:
        app.run(host='0.0.0.0', port=port, threaded=True)
