"""仪表盘蓝图。"""
import os
from datetime import datetime

from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user

from db_models import Announcement, InferenceConfig, Model, Task
from services.quota import current_quota, today_task_count, usable_models_for
from apputils import status_label, today_cutoff

bp = Blueprint('dashboard', __name__)


def _user_storage_bytes(user_id):
    """统计用户模型文件 + 任务输入/结果文件的磁盘字节。"""
    upload = current_app.config['UPLOAD_FOLDER']
    total = 0

    def _sum_files(rel_paths):
        n = 0
        for rel in rel_paths:
            if not rel:
                continue
            sub = 'models' if rel.lower().endswith(('.pth', '.pt')) else 'configs'
            p = os.path.join(upload, sub, os.path.basename(rel))
            if os.path.exists(p):
                try:
                    n += os.path.getsize(p)
                except OSError:
                    pass
        return n

    for m in Model.query.filter(Model.user_id == user_id).all():
        total += _sum_files([m.model_path, m.config_path, m.diff_model_path, m.diff_config_path, m.cluster_path])
    for t in Task.query.filter(Task.user_id == user_id).all():
        if t.audio_filename:
            p = os.path.join(upload, 'audio', t.audio_filename)
            if os.path.exists(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
        if t.result_filename:
            p = os.path.join(upload, 'results', t.result_filename)
            if os.path.exists(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    return total


@bp.route('/', endpoint='dashboard')
@login_required
def dashboard():
    quota = current_quota(current_user)
    today = today_cutoff()
    running_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status == 'running').count()
    queued_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status.in_(['pending', 'claimed'])).count()
    used_today = today_task_count(current_user)
    done_today = Task.query.filter(Task.user_id == current_user.id, Task.status == 'done', Task.done_at >= today).count()
    fail_today = Task.query.filter(Task.user_id == current_user.id, Task.status == 'failed', Task.done_at >= today).count()

    # 修复口径：私有模型数 = 当前用户自己的；官方数独立统计
    private_cnt = Model.query.filter(
        Model.user_id == current_user.id,
        Model.visibility == 'private',
        Model.status.in_(['pending_review', 'ready']),
    ).count()
    official_cnt = Model.query.filter(Model.visibility == 'official', Model.status == 'ready').count()

    models = usable_models_for(current_user)
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).all()

    now = datetime.utcnow()
    recent_items = []
    for t in (Task.query.filter_by(user_id=current_user.id)
              .order_by(Task.created_at.desc()).limit(6).all()):
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        expired = (t.status == 'done' and t.result_expires_at and t.result_expires_at < now)
        recent_items.append({
            'id': t.id, 'status_label': '已过期' if expired else status_label(t.status),
            'model': model_name, 'progress_msg': t.progress_msg or '',
            'created': t.created_at,
            'can_download': t.status == 'done' and bool(t.result_filename) and not expired,
        })

    announcements = (Announcement.query.filter_by(is_active=True)
                     .order_by(Announcement.is_pinned.desc(), Announcement.created_at.desc()).limit(3).all())
    return render_template('dashboard.html',
        quota=quota, running_cnt=running_cnt, queued_cnt=queued_cnt,
        used_today=used_today, daily_limit=quota.max_daily_tasks or 0,
        done_today=done_today, fail_today=fail_today,
        models=models, private_cnt=private_cnt, official_cnt=official_cnt,
        config_cnt=len(configs), recent=recent_items, announcements=announcements)


@bp.route('/usage', endpoint='usage')
@login_required
def usage():
    quota = current_quota(current_user)
    today = today_cutoff()
    running_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status == 'running').count()
    queued_cnt = Task.query.filter(Task.user_id == current_user.id, Task.status.in_(['pending', 'claimed'])).count()
    used_today = today_task_count(current_user)
    private_cnt = Model.query.filter(
        Model.user_id == current_user.id, Model.visibility == 'private',
        Model.status.in_(['pending_review', 'ready'])).count()
    storage_bytes = _user_storage_bytes(current_user.id)
    return render_template('usage.html', quota=quota, running_cnt=running_cnt, queued_cnt=queued_cnt,
                           used_today=used_today, private_cnt=private_cnt, storage_bytes=storage_bytes,
                           storage_limit=quota.storage_quota_bytes or 0)
