"""任务蓝图：列表/结果下载/删除/停止/详情。"""
import os
import re
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from db_models import Task
from apputils import status_label, global_queue_position

bp = Blueprint('tasks', __name__)


def _current_upload():
    from flask import current_app
    return current_app.config['UPLOAD_FOLDER']


def _parse_pct(msg):
    """从 progress_msg 解析百分比；None 表示没有可用的进度信息。"""
    if not msg or '(' not in msg or '%' not in msg:
        return None
    try:
        return msg.split('(')[1].split('%')[0].strip()
    except Exception:
        return None


@bp.route('/tasks', endpoint='task_list')
@login_required
def task_list():
    from extensions import db as _db
    from db_models import Model as _Model
    status = request.args.get('status', '').strip()
    q = request.args.get('q', '').strip()
    page = max(request.args.get('page', 1, type=int), 1)
    per_page = 20

    query = Task.query.filter_by(user_id=current_user.id)
    if status:
        query = query.filter(Task.status == status)
    if q:
        from sqlalchemy import or_ as _or
        ids = [int(part) for part in re.findall(r'\d+', q)]
        model_ids = [mid for (mid,) in _db.session.query(_Model.id)
                     .filter(_Model.name.ilike(f'%{q}%')).all()]
        conds = []
        if ids:
            conds.append(Task.id.in_(ids))
        if model_ids:
            conds.append(Task.model_id.in_(model_ids))
        query = query.filter(_or(*conds)) if conds else query.filter(_db.false())

    total = query.count()
    pages = max((total + per_page - 1) // per_page, 1)
    page = min(page, pages)
    tasks = query.order_by(Task.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    now = datetime.utcnow()
    items = []
    for t in tasks:
        try:
            model_name = t.config.model.name if t.config and t.config.model else '—'
        except Exception:
            model_name = '—'
        expired = (t.status == 'done' and t.result_expires_at and t.result_expires_at < now)
        queue_pos = global_queue_position(t) if t.status in ('pending', 'claimed') else 0
        pct = _parse_pct(t.progress_msg)
        # 有的任务 status 被调度器重置回 pending 但推理进程还在跑（progress_msg 一直更新），
        # 列表页不能只信 status，否则会把"正在推理"显示成"排队中"。能解析出进度就当运行中显示。
        inferring = t.status == 'running' or pct is not None
        display_status = 'running' if inferring else t.status
        items.append({
            't': t, 'model': model_name,
            'status_label': '已过期' if expired else status_label(display_status),
            'queue_pos': queue_pos,
            'result_expires': t.result_expires_at,
            'pct': pct,
            'inferring': inferring,
            'display_status': display_status,
            'can_download': t.status == 'done' and bool(t.result_filename) and not expired,
            'can_stop': t.status in ('pending', 'claimed', 'running'),
        })
    return render_template('tasks.html', tasks=items, cur_status=status, cur_q=q,
                           page=page, pages=pages, total=total)


@bp.route('/tasks/<int:task_id>/detail', endpoint='task_detail')
@login_required
def task_detail(task_id):
    import json as _json
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    try:
        params = _json.loads(task.params_json or '{}')
    except Exception:
        params = {}
    try:
        model_name = task.config.model.name if task.config and task.config.model else '—'
        config_name = task.config.name if task.config else '—'
    except Exception:
        model_name, config_name = '—', '—'
    expired = (task.status == 'done' and task.result_expires_at and task.result_expires_at < datetime.utcnow())
    return render_template('task_detail.html', t=task, params=params,
                           model=model_name, config=config_name,
                           status_label='已过期' if expired else status_label(task.status),
                           can_download=task.status == 'done' and bool(task.result_filename) and not expired)


@bp.route('/tasks/<int:task_id>/result', endpoint='task_result')
@login_required
def task_result(task_id):
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    if not task.result_filename:
        abort(404)
    if task.result_expires_at and task.result_expires_at < datetime.utcnow():
        abort(404)
    path = os.path.join(_current_upload(), 'results', task.result_filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@bp.route('/tasks/<int:task_id>/delete', methods=['POST'], endpoint='task_delete')
@login_required
def task_delete(task_id):
    from extensions import db
    task = Task.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        abort(403)
    if task.result_filename:
        path = os.path.join(_current_upload(), 'results', task.result_filename)
        if os.path.exists(path):
            os.remove(path)
    db.session.delete(task)
    db.session.commit()
    flash('任务已删除', 'success')
    return redirect(url_for('task_list'))


@bp.route('/tasks/<int:task_id>/stop', methods=['POST'], endpoint='task_stop')
@login_required
def task_stop(task_id):
    from extensions import db
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


@bp.route('/results/<filename>', endpoint='download_result')
@login_required
def download_result(filename):
    task = Task.query.filter_by(result_filename=secure_filename(filename), user_id=current_user.id).first()
    if not task:
        abort(404)
    return task_result(task.id)
