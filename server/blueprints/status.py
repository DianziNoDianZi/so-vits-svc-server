"""系统状态页：服务健康、资源占用、队列/任务统计。"""
import os
import shutil
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, current_app
from flask_login import login_required

from extensions import db
from db_models import Model, Task, User
from services import scheduler
from services.quota import get_setting

bp = Blueprint('status', __name__)


@bp.route('/status', endpoint='status')
@login_required
def status():
    now = datetime.utcnow()

    # daemon / 调度
    daemon_alive = bool(scheduler.inference_daemon_proc and scheduler.inference_daemon_proc.is_alive())
    paused = get_setting('scheduler_paused', '0') == '1'

    # 队列
    queue_depth = Task.query.filter(Task.status.in_(['pending', 'claimed'])).count()
    running = Task.query.filter(Task.status == 'running').count()

    # 任务统计（今天）
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    done_today = Task.query.filter(Task.status == 'done', Task.done_at >= today).count()
    failed_today = Task.query.filter(Task.status == 'failed', Task.done_at >= today).count()

    # 最近失败（帮助定位问题）
    recent_failures = (Task.query.filter(Task.status == 'failed', Task.error_msg.isnot(None))
                       .order_by(Task.done_at.desc()).limit(8).all())
    fail_items = []
    for t in recent_failures:
        fail_items.append({
            'id': t.id, 'user': t.user.username if t.user else '?',
            'error': (t.error_msg or '')[:120],
            'done_at': t.done_at,
        })

    # 磁盘
    upload = current_app.config['UPLOAD_FOLDER']
    disk_free = disk_total = 0
    disk_pct = 0
    try:
        du = shutil.disk_usage(upload)
        disk_free, disk_total = du.free, du.total
        disk_pct = int(du.used / du.total * 100) if du.total else 0
    except Exception:
        pass

    # 资源（CPU/内存）
    cpu = mem = None
    try:
        import psutil as _psutil
        # cpu_percent(interval=None) 首次调用返回 0（需要两次采样差值），
        # 这里阻塞采样 0.2s 才能拿到真实占用；内存是即时值不用等。
        cpu = _psutil.cpu_percent(interval=0.2)
        mem = _psutil.virtual_memory().percent
    except Exception:
        import traceback
        current_app.logger.error('status: psutil 读取失败\n' + traceback.format_exc())

    # GPU
    gpu = None
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            free, total = _torch.cuda.mem_get_info()
            gpu = (total - free) / total * 100 if total else 0
    except Exception:
        pass

    # 备份信息
    from services.backup import backup_dir
    backups = []
    if os.path.isdir(backup_dir()):
        for f in sorted(os.listdir(backup_dir()), reverse=True)[:5]:
            if f.startswith('backup_') and f.endswith('.db'):
                p = os.path.join(backup_dir(), f)
                try:
                    backups.append({'name': f, 'mtime': datetime.fromtimestamp(os.path.getmtime(p))})
                except OSError:
                    pass

    stats = {
        'users': User.query.count(),
        'total_models': Model.query.count(),
        'official_models': Model.query.filter(Model.visibility == 'official').count(),
        'total_tasks': Task.query.count(),
    }
    return render_template('status.html',
        daemon_alive=daemon_alive, paused=paused,
        queue_depth=queue_depth, running=running,
        done_today=done_today, failed_today=failed_today,
        fail_items=fail_items,
        disk_free=disk_free, disk_total=disk_total, disk_pct=disk_pct,
        cpu=cpu, mem=mem, gpu=gpu,
        backups=backups, stats=stats,
        resource_warning=scheduler.resource_warning,
    )
