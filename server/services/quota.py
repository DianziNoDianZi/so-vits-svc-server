"""配额与站点设置服务。"""
import os

from extensions import db
from db_models import ServerSetting, UserQuota, Task
from apputils import today_cutoff


_SMTP_ENV_MAP = {
    'smtp_host': 'SMTP_HOST', 'smtp_port': 'SMTP_PORT',
    'smtp_user': 'SMTP_USER', 'smtp_pass': 'SMTP_PASS', 'mail_from': 'MAIL_FROM',
}


def get_setting(key, default=None):
    env = _SMTP_ENV_MAP.get(key)
    if env and os.environ.get(env):
        return os.environ[env]
    row = db.session.get(ServerSetting, key)
    return row.value if row and row.value else default


def set_setting(key, value):
    row = db.session.get(ServerSetting, key)
    if row:
        row.value = value
    else:
        db.session.add(ServerSetting(key=key, value=value))
    db.session.commit()


def current_quota(user):
    q = UserQuota.query.filter_by(user_id=user.id).first()
    if q:
        return q
    q = UserQuota(
        user_id=user.id,
        max_queued_tasks=int(get_setting('default_max_queued', 4)),
        max_running_tasks=int(get_setting('default_max_running', 1)),
        max_input_seconds=int(get_setting('default_max_input_seconds', 600)),
        max_daily_tasks=int(get_setting('default_max_daily_tasks', 50)),
        max_cpu_cores=int(get_setting('default_max_cpu_cores', 0)),
        storage_quota_bytes=int(get_setting('default_storage_quota_bytes', 10 * 1024 ** 3)),
        max_model_bytes=int(get_setting('default_max_model_bytes', 4 * 1024 ** 3)),
        max_private_models=int(get_setting('default_max_private_models', 3)),
        priority=int(get_setting('default_priority', 1)),
        results_retention_days=int(get_setting('default_result_retention_days', 7)),
    )
    db.session.add(q)
    db.session.commit()
    return q


def today_task_count(user):
    return Task.query.filter(
        Task.user_id == user.id,
        Task.created_at >= today_cutoff(),
        Task.status.in_(['pending', 'claimed', 'running', 'done', 'failed', 'stopped']),
    ).count()


def usable_models_for(user):
    from db_models import Model
    from authorization import can_use_model, is_admin
    models = Model.query.order_by(Model.created_at.desc()).all()
    if is_admin(user):
        return [m for m in models if getattr(m, 'status', 'ready') != 'disabled']
    return [m for m in models if can_use_model(user, m)]
