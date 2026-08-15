"""健康检查端点：给外部探活/告警用。"""
import os
import shutil

from flask import Blueprint, jsonify, request

from extensions import db
from services import scheduler

bp = Blueprint('health', __name__)


@bp.route('/healthz', endpoint='healthz')
def healthz():
    db_ok = False
    try:
        db.session.execute(db.text('SELECT 1'))
        db_ok = True
    except Exception:
        pass

    daemon_alive = bool(scheduler.inference_daemon_proc and scheduler.inference_daemon_proc.is_alive())

    queue_depth = 0
    try:
        from db_models import Task
        queue_depth = Task.query.filter(Task.status.in_(['pending', 'claimed'])).count()
    except Exception:
        pass

    disk_free = disk_total = 0
    try:
        from flask import current_app
        du = shutil.disk_usage(current_app.config['UPLOAD_FOLDER'])
        disk_free, disk_total = du.free, du.total
    except Exception:
        pass

    from services.quota import get_setting
    paused = get_setting('scheduler_paused', '0') == '1'

    ok = db_ok and daemon_alive and not paused
    payload = {
        'status': 'ok' if ok else 'degraded',
        'db': db_ok,
        'daemon_alive': daemon_alive,
        'queue_depth': queue_depth,
        'scheduler_paused': paused,
        'disk_free_gb': round(disk_free / (1024 ** 3), 2),
        'disk_total_gb': round(disk_total / (1024 ** 3), 2),
    }
    # 简单防护：GET 返回 JSON 供探活，带 ?check=1 时用于探针区分
    if request.args.get('check') == '1':
        payload['probe'] = True
    return jsonify(payload), (200 if ok else 503)
