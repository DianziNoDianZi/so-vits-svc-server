"""管理员蓝图：总览/用户配额/全局任务/模型审核/存储/设置/公告。"""
import csv
import io
import os
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, send_file, Response
from flask_login import login_required, current_user

from authorization import is_admin
from extensions import db
from db_models import Announcement, Model, Task, User
from services import scheduler
from services.quota import current_quota, get_setting, set_setting
from services.validation import check_model_config
from apputils import dir_size_mb, today_cutoff, arch_label_from_cfg, read_model_cfg, status_label

bp = Blueprint('admin', __name__)


def _guard():
    if not is_admin(current_user):
        abort(403)


def _upload():
    from flask import current_app
    return current_app.config['UPLOAD_FOLDER']


_MODEL_FILE_ATTRS = {
    'model': 'model_path', 'config': 'config_path', 'diff': 'diff_model_path',
    'diff_config': 'diff_config_path', 'cluster': 'cluster_path',
}


def _user_storage_bytes(user_id):
    upload = _upload()
    total = 0

    def _add(rel, sub):
        nonlocal total
        if rel:
            p = os.path.join(upload, sub, os.path.basename(rel))
            if os.path.exists(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass

    for m in Model.query.filter(Model.user_id == user_id).all():
        _add(m.model_path, 'models'); _add(m.config_path, 'configs')
        _add(m.diff_model_path, 'models'); _add(m.diff_config_path, 'configs'); _add(m.cluster_path, 'models')
    for t in Task.query.filter(Task.user_id == user_id).all():
        _add(t.audio_filename, 'audio'); _add(t.result_filename, 'results')
    return total


@bp.route('/admin', endpoint='admin_index')
@login_required
def admin_index():
    _guard()
    today = today_cutoff()
    uploads = _upload()
    disk = {k: dir_size_mb(os.path.join(uploads, k)) for k in ('models', 'configs', 'audio', 'results')}
    running_tasks = []
    for t in Task.query.filter(Task.status.in_(['claimed', 'running'])).order_by(Task.created_at.asc()).all():
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        running_tasks.append({'id': t.id, 'user': t.user.username if t.user else '?',
                              'model': model_name, 'status': t.status, 'created': t.created_at})
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
    try:
        torch_cuda = __import__('torch').cuda.is_available()
    except Exception:
        torch_cuda = False
    daemon_alive = bool(scheduler.inference_daemon_proc and scheduler.inference_daemon_proc.is_alive())
    health = {
        'daemon_alive': daemon_alive,
        'gpu': '可用' if torch_cuda else '不可用',
        'paused': get_setting('scheduler_paused', '0') == '1',
        'queue_depth': Task.query.filter(Task.status.in_(['pending', 'claimed'])).count(),
    }
    return render_template('admin_overview.html', stats=stats, running_tasks=running_tasks, disk=disk, health=health)


@bp.route('/admin/users', endpoint='admin_users')
@login_required
def admin_users():
    _guard()
    items = []
    for u in User.query.order_by(User.id.asc()).all():
        q = current_quota(u)
        items.append({
            'user': u, 'quota': q,
            'queued': Task.query.filter(Task.user_id == u.id, Task.status.in_(['pending', 'claimed', 'running'])).count(),
            'private_models': Model.query.filter(Model.user_id == u.id, Model.visibility == 'private').count(),
            'storage_bytes': _user_storage_bytes(u.id),
        })
    return render_template('admin_users.html', items=items)


@bp.route('/admin/users/<int:uid>/quota', methods=['POST'], endpoint='admin_update_quota')
@login_required
def admin_update_quota(uid):
    _guard()
    u = db.session.get(User, uid)
    if not u:
        abort(404)
    q = current_quota(u)

    def _i(k, d):
        try:
            return int(request.form.get(k, d) or d)
        except (ValueError, TypeError):
            return d

    q.enabled = request.form.get('enabled') == 'on'
    q.max_queued_tasks = _i('max_queued_tasks', 4)
    q.max_running_tasks = _i('max_running_tasks', 1)
    q.max_input_seconds = _i('max_input_seconds', 600)
    q.max_daily_tasks = _i('max_daily_tasks', 50)
    q.max_cpu_cores = _i('max_cpu_cores', 0)
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


@bp.route('/admin/tasks', endpoint='admin_tasks')
@login_required
def admin_tasks():
    _guard()
    q = Task.query
    uid = request.args.get('user_id', type=int)
    status = request.args.get('status', '').strip()
    if uid:
        q = q.filter(Task.user_id == uid)
    if status:
        q = q.filter(Task.status == status)
    items = []
    for t in q.order_by(Task.created_at.desc()).limit(200).all():
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        items.append({'t': t, 'user': t.user.username if t.user else '?', 'model': model_name,
                      'status_label': status_label(t.status),
                      'can_stop': t.status in ('pending', 'claimed', 'running')})
    users = User.query.order_by(User.username.asc()).all()
    paused = get_setting('scheduler_paused', '0') == '1'
    return render_template('admin_tasks.html', items=items, users=users, cur_uid=uid, cur_status=status,
                           paused=paused)


@bp.route('/admin/tasks/pause', methods=['POST'], endpoint='admin_tasks_pause')
@login_required
def admin_tasks_pause():
    _guard()
    paused = get_setting('scheduler_paused', '0') == '1'
    set_setting('scheduler_paused', '0' if paused else '1')
    flash('已暂停接收新任务（运行中任务不受影响）' if not paused else '已恢复接收新任务', 'success')
    return redirect(url_for('admin_tasks'))


@bp.route('/admin/tasks/export', endpoint='admin_tasks_export')
@login_required
def admin_tasks_export():
    _guard()
    q = Task.query
    uid = request.args.get('user_id', type=int)
    status = request.args.get('status', '').strip()
    if uid:
        q = q.filter(Task.user_id == uid)
    if status:
        q = q.filter(Task.status == status)
    tasks = q.order_by(Task.created_at.desc()).limit(5000).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'user', 'status', 'model', 'created_at', 'done_at', 'error'])
    for t in tasks:
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        writer.writerow([t.id, t.user.username if t.user else '', t.status, model_name,
                         t.created_at.strftime('%Y-%m-%d %H:%M:%S') if t.created_at else '',
                         t.done_at.strftime('%Y-%m-%d %H:%M:%S') if t.done_at else '',
                         (t.error_msg or '')[:200]])
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=tasks.csv'})


@bp.route('/admin/tasks/<int:task_id>/stop', methods=['POST'], endpoint='admin_task_stop')
@login_required
def admin_task_stop(task_id):
    _guard()
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


@bp.route('/admin/tasks/<int:task_id>/delete', methods=['POST'], endpoint='admin_task_delete')
@login_required
def admin_task_delete(task_id):
    _guard()
    t = db.session.get(Task, task_id)
    if not t:
        abort(404)
    if t.result_filename:
        p = os.path.join(_upload(), 'results', t.result_filename)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    db.session.delete(t)
    db.session.commit()
    flash(f'已删除任务 #{task_id}', 'success')
    return redirect(url_for('admin_tasks'))


@bp.route('/admin/models', endpoint='admin_models')
@login_required
def admin_models():
    _guard()
    models = list(Model.query.order_by(Model.created_at.desc()).all())
    models.sort(key=lambda m: 0 if m.status == 'pending_review' else 1)  # 待审核置顶
    items = []
    for m in models:
        cfg = read_model_cfg(m.config_path)
        label, sub = arch_label_from_cfg(cfg)
        ok, issues = check_model_config(m)
        items.append({'m': m, 'owner': m.owner.username if m.owner else '—',
                      'arch_label': label, 'flow_mode': sub,
                      'status': m.status, 'visibility': m.visibility,
                      'public_requested': bool(getattr(m, 'public_requested', False)),
                      'valid': ok, 'issues': issues})
    return render_template('admin_models.html', items=items)


@bp.route('/admin/models/<int:model_id>/file/<attr>', endpoint='admin_model_file')
@login_required
def admin_model_file(model_id, attr):
    _guard()
    if attr not in _MODEL_FILE_ATTRS:
        abort(400)
    m = db.session.get(Model, model_id)
    if not m:
        abort(404)
    path = getattr(m, _MODEL_FILE_ATTRS[attr])
    if not path:
        abort(404)
    sub = 'models' if path.endswith(('.pth', '.pt')) else 'configs'
    full = os.path.join(_upload(), sub, path)
    if not os.path.exists(full):
        abort(404)
    return send_file(full, as_attachment=True, download_name=os.path.basename(path))


@bp.route('/admin/storage', endpoint='admin_storage')
@login_required
def admin_storage():
    _guard()
    uploads = _upload()
    disk = {k: dir_size_mb(os.path.join(uploads, k)) for k in ('models', 'configs', 'audio', 'results')}
    total = sum(disk.values())
    refs_models = set()
    refs_configs = set()
    for m in Model.query.all():
        refs_models |= {m.model_path, m.diff_model_path, m.cluster_path}
        refs_configs |= {m.config_path, m.diff_config_path}
    refs_results = {t.result_filename for t in Task.query.all()}
    refs_audio = {t.audio_filename for t in Task.query.all()}

    def _orphans(sub, refs):
        d = os.path.join(uploads, sub)
        if not os.path.isdir(d):
            return []
        return [f for f in sorted(os.listdir(d))
                if os.path.isfile(os.path.join(d, f)) and f not in refs]

    orphan = {
        'models': _orphans('models', refs_models),
        'configs': _orphans('configs', refs_configs),
        'results': _orphans('results', refs_results),
        'audio': _orphans('audio', refs_audio),
    }
    return render_template('admin_storage.html', disk=disk, total=total, orphan=orphan)


@bp.route('/admin/storage/delete', methods=['POST'], endpoint='admin_storage_delete')
@login_required
def admin_storage_delete():
    _guard()
    sub = request.form.get('sub', '')
    name = request.form.get('name', '').strip()
    if sub not in ('models', 'configs', 'results', 'audio') or not name or os.path.basename(name) != name:
        flash('非法参数', 'danger')
        return redirect(url_for('admin_storage'))
    refs_models = set()
    refs_configs = set()
    for m in Model.query.all():
        refs_models |= {m.model_path, m.diff_model_path, m.cluster_path}
        refs_configs |= {m.config_path, m.diff_config_path}
    refs = {'models': refs_models, 'configs': refs_configs,
            'results': {t.result_filename for t in Task.query.all()},
            'audio': {t.audio_filename for t in Task.query.all()}}
    if name in refs.get(sub, set()):
        flash(f'{name} 已被引用，无法删除', 'danger')
        return redirect(url_for('admin_storage'))
    p = os.path.join(_upload(), sub, name)
    if os.path.exists(p):
        try:
            os.remove(p)
            flash(f'已删除孤儿文件 {name}', 'success')
        except OSError as e:
            flash(f'删除失败: {e}', 'danger')
    else:
        flash('文件不存在', 'warning')
    return redirect(url_for('admin_storage'))


@bp.route('/admin/settings', methods=['GET', 'POST'], endpoint='admin_settings')
@login_required
def admin_settings():
    _guard()
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
            set_setting('allow_registration', '1' if request.form.get('allow_registration') == 'on' else '0')
            set_setting('default_max_queued', request.form.get('default_max_queued', '4'))
            set_setting('default_max_running', request.form.get('default_max_running', '1'))
            set_setting('default_max_input_seconds', request.form.get('default_max_input_seconds', '600'))
            set_setting('default_max_daily_tasks', request.form.get('default_max_daily_tasks', '50'))
            set_setting('default_max_cpu_cores', request.form.get('default_max_cpu_cores', '0'))
            set_setting('default_max_private_models', request.form.get('default_max_private_models', '3'))
            set_setting('default_result_retention_days', request.form.get('default_result_retention_days', '7'))
            flash('站点设置已保存（对新注册用户生效）', 'success')
            return redirect(url_for('admin_settings'))
        if request.form.get('action') == 'train':
            from services.training import set_training_enabled
            set_training_enabled(request.form.get('training_enabled') == 'on')
            set_setting('train_cpu_cores', request.form.get('train_cpu_cores', '0'))
            flash('训练功能设置已保存', 'success')
            return redirect(url_for('admin_settings'))
        set_setting('smtp_host', request.form.get('smtp_host', '').strip())
        set_setting('smtp_port', request.form.get('smtp_port', '465').strip() or '465')
        set_setting('smtp_user', request.form.get('smtp_user', '').strip())
        pwd = request.form.get('smtp_pass', '').strip()
        if pwd:
            set_setting('smtp_pass', pwd)
        set_setting('mail_from', request.form.get('mail_from', '').strip())
        flash('SMTP 配置已保存', 'success')
        return redirect(url_for('admin_settings'))
    cfg = {'smtp_host': get_setting('smtp_host', ''), 'smtp_port': get_setting('smtp_port', '465'),
           'smtp_user': get_setting('smtp_user', ''), 'mail_from': get_setting('mail_from', '')}
    site = {
        'allow_registration': get_setting('allow_registration', __import__('os').environ.get('ALLOW_REGISTRATION', '1')) == '1',
        'default_max_queued': get_setting('default_max_queued', 4),
        'default_max_running': get_setting('default_max_running', 1),
        'default_max_input_seconds': get_setting('default_max_input_seconds', 600),
        'default_max_daily_tasks': get_setting('default_max_daily_tasks', 50),
        'default_max_cpu_cores': get_setting('default_max_cpu_cores', 0),
        'default_max_private_models': get_setting('default_max_private_models', 3),
        'default_result_retention_days': get_setting('default_result_retention_days', 7),
    }
    from services.training import training_enabled
    train = {'enabled': training_enabled(), 'cpu_cores': get_setting('train_cpu_cores', '0')}
    return render_template('admin_settings.html', cfg=cfg, site=site, train=train)


@bp.route('/admin/announcements', endpoint='admin_announcements')
@login_required
def admin_announcements():
    _guard()
    items = Announcement.query.order_by(Announcement.created_at.desc()).all()
    return render_template('admin_announcements.html', items=items)


@bp.route('/admin/announcements/create', methods=['POST'], endpoint='admin_announcements_create')
@login_required
def admin_announcements_create():
    _guard()
    title = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    if not title or not content:
        flash('标题和内容不能为空', 'danger')
        return redirect(url_for('admin_announcements'))
    db.session.add(Announcement(title=title, content=content, is_pinned=request.form.get('is_pinned') == 'on',
                                is_active=True, created_by_user_id=current_user.id))
    db.session.commit()
    if request.form.get('email_all') == 'on':
        from notifier import send_via_server
        sent = 0
        body = f'{title}\n\n{content}\n\n--- So-VITS-SVC 推理服务 ---'
        for u in User.query.all():
            rec = getattr(u, 'notify_email', None) or u.email
            if not rec:
                continue
            try:
                if send_via_server(rec, f'[SoVITS] {title}', body):
                    sent += 1
            except Exception:
                continue
        flash(f'公告已发布，邮件已发送给 {sent} 位用户', 'success')
    else:
        flash('公告已发布', 'success')
    return redirect(url_for('admin_announcements'))


@bp.route('/admin/announcements/<int:aid>/toggle', methods=['POST'], endpoint='admin_announcements_toggle')
@login_required
def admin_announcements_toggle(aid):
    _guard()
    a = db.session.get(Announcement, aid)
    if not a:
        abort(404)
    a.is_active = not a.is_active
    db.session.commit()
    flash('公告已上/下架', 'success')
    return redirect(url_for('admin_announcements'))


@bp.route('/admin/announcements/<int:aid>/delete', methods=['POST'], endpoint='admin_announcements_delete')
@login_required
def admin_announcements_delete(aid):
    _guard()
    a = db.session.get(Announcement, aid)
    if not a:
        abort(404)
    db.session.delete(a)
    db.session.commit()
    flash('公告已删除', 'success')
    return redirect(url_for('admin_announcements'))
