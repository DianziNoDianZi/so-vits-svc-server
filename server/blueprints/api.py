"""对外 REST API（Bearer token 鉴权）+ API token 管理端点。

网页端 token 管理走 /api/token（POST 生成 / DELETE 撤销 / GET 列表），
供 settings.html 的"开发者 API"区块使用；真正对外端点走 /api/v1/*。
"""
import os
import re
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, send_file, render_template
from flask_login import login_required, current_user

from authorization import can_use_model
from extensions import db
from db_models import DEFAULT_PARAMS, InferenceConfig, Model, Task, User
from services.quota import current_quota, today_task_count
from services.scheduler import ensure_worker
from services.api_auth import require_api_token, generate_token, revoke_token
from blueprints.inference import _submit_one, _allowed_audio, _current_upload

bp = Blueprint('api', __name__, url_prefix='/api/v1')


# ========== 网页端 token 管理（供 settings.html 用） ==========

@bp.route('/token', methods=['POST'], endpoint='api_token_create')
@login_required
def api_token_create():
    name = (request.form.get('name') or 'default').strip()[:100]
    token = generate_token(current_user, name or 'default')
    from services.audit import audit_log
    audit_log('api_token_create', f'生成 API token: {name}')
    return jsonify({'token': token, 'note': 'token 仅显示一次，请立即保存'})


@bp.route('/token/<int:token_id>', methods=['DELETE'], endpoint='api_token_revoke')
@login_required
def api_token_revoke(token_id):
    if revoke_token(current_user, token_id):
        from services.audit import audit_log
        audit_log('api_token_revoke', f'撤销 API token #{token_id}')
        return jsonify({'ok': True})
    return jsonify({'error': '未找到'}), 404


@bp.route('/token', methods=['GET'], endpoint='api_token_list')
@login_required
def api_token_list():
    from db_models import ApiToken
    tokens = ApiToken.query.filter_by(user_id=current_user.id, revoked_at=None).all()
    return jsonify({'tokens': [{'id': t.id, 'name': t.name,
                                'created_at': t.created_at.strftime('%Y-%m-%d %H:%M')} for t in tokens]})


# ========== 对外 REST 端点 ==========

@bp.route('/inference', methods=['POST'], endpoint='api_inference')
@require_api_token
def api_inference():
    user = g.api_user
    from apputils import rate_limit_allowed
    if not rate_limit_allowed('infer', f'{user.id}'):
        return jsonify({'error': '提交过于频繁，请稍后再试'}), 429
    quota = current_quota(user)
    if not quota.enabled:
        return jsonify({'error': '账号已被禁用'}), 403

    config_id = request.form.get('config_id', type=int)
    files = [f for f in request.files.getlist('audio') if f and f.filename]
    if not config_id or not files:
        return jsonify({'error': '缺少 config_id 或 audio 文件'}), 400
    cfg_obj = InferenceConfig.query.get_or_404(config_id)
    if cfg_obj.user_id != user.id:
        return jsonify({'error': 'config_id 不属于当前用户'}), 403
    model = db.session.get(Model, cfg_obj.model_id)
    if not model or not can_use_model(user, model):
        return jsonify({'error': '所选模型不可用'}), 403

    queued = Task.query.filter(Task.user_id == user.id, Task.status.in_(['pending', 'running'])).count()
    daily = today_task_count(user)
    limit = min(len(files), 10)
    if quota.max_queued_tasks:
        limit = min(limit, max(quota.max_queued_tasks - queued, 0))
    if quota.max_daily_tasks:
        limit = min(limit, max(quota.max_daily_tasks - daily, 0))
    if limit <= 0:
        return jsonify({'error': '已达到任务提交上限（排队数或每日任务数已满）'}), 429

    ensure_worker()
    form_get = request.form.get
    created, skipped = 0, []
    for f in files[:limit]:
        status, reason = _submit_one(cfg_obj, model, quota, f, user, form_get=form_get)
        if status == 'ok':
            created += 1
        else:
            skipped.append(f'{f.filename}: {reason}')
    return jsonify({'created': created, 'skipped': skipped}), 200


@bp.route('/tasks/<int:task_id>', methods=['GET'], endpoint='api_task_get')
@require_api_token
def api_task_get(task_id):
    user = g.api_user
    task = db.session.get(Task, task_id)
    if not task or task.user_id != user.id:
        return jsonify({'error': '未找到'}), 404
    pct = None
    m = re.search(r'\((\d+)%\)', task.progress_msg or '')
    if m:
        pct = int(m.group(1))
    return jsonify({
        'id': task.id, 'status': task.status,
        'progress_msg': task.progress_msg or '', 'pct': pct,
        'has_result': bool(task.result_filename),
        'error_msg': (task.error_msg or '')[:200],
        'created_at': task.created_at.strftime('%Y-%m-%d %H:%M:%S') if task.created_at else None,
    })


@bp.route('/tasks/<int:task_id>/result', methods=['GET'], endpoint='api_task_result')
@require_api_token
def api_task_result(task_id):
    user = g.api_user
    from apputils import rate_limit_allowed
    if not rate_limit_allowed('download', f'{user.id}'):
        return jsonify({'error': '下载过于频繁，请稍后再试'}), 429
    task = db.session.get(Task, task_id)
    if not task or task.user_id != user.id:
        return jsonify({'error': '未找到'}), 404
    if not task.result_filename:
        return jsonify({'error': '结果尚未生成'}), 404
    path = os.path.join(_current_upload(), 'results', task.result_filename)
    if not os.path.exists(path):
        return jsonify({'error': '结果文件不存在'}), 404
    return send_file(path, as_attachment=True, download_name=task.result_filename)


@bp.route('/models', methods=['GET'], endpoint='api_models')
@require_api_token
def api_models():
    user = g.api_user
    models = [m for m in Model.query.order_by(Model.created_at.desc()).all() if can_use_model(user, m)]
    return jsonify({'models': [{'id': m.id, 'name': m.name, 'visibility': m.visibility,
                                'description': m.description or ''} for m in models]})


@bp.route('/configs', methods=['GET'], endpoint='api_configs')
@require_api_token
def api_configs():
    user = g.api_user
    items = []
    for c in InferenceConfig.query.filter_by(user_id=user.id).order_by(InferenceConfig.created_at.desc()).all():
        if not can_use_model(user, c.model):
            continue
        try:
            params = __import__('json').loads(c.params_json or '{}')
        except Exception:
            params = {}
        items.append({'id': c.id, 'name': c.name, 'model_id': c.model_id,
                      'model_name': c.model.name if c.model else '', 'params': params})
    return jsonify({'configs': items})


@bp.route('/system', methods=['GET'], endpoint='api_system')
@require_api_token
def api_system():
    """系统状态：daemon 存活、队列深度、调度是否暂停、模型/任务统计。"""
    from services import scheduler
    from services.quota import get_setting
    daemon_alive = bool(scheduler.inference_daemon_proc and scheduler.inference_daemon_proc.is_alive())
    queue_depth = Task.query.filter(Task.status.in_(['pending', 'claimed'])).count()
    running = Task.query.filter(Task.status == 'running').count()
    from services.sysinfo import cpu_percent as _cpu, mem_percent as _mem
    cpu = _cpu()
    mem = _mem()
    disk_free = disk_total = None
    try:
        import shutil as _shutil
        from flask import current_app as _app
        du = _shutil.disk_usage(_app.config['UPLOAD_FOLDER'])
        disk_free, disk_total = du.free, du.total
    except Exception:
        pass
    return jsonify({
        'daemon_alive': daemon_alive,
        'scheduler_paused': get_setting('scheduler_paused', '0') == '1',
        'queue_depth': queue_depth,
        'running_tasks': running,
        'cpu_percent': cpu,
        'mem_percent': mem,
        'disk_free_bytes': disk_free,
        'disk_total_bytes': disk_total,
        'total_models': Model.query.count(),
        'official_models': Model.query.filter(Model.visibility == 'official').count(),
        'total_tasks': Task.query.count(),
        'total_users': User.query.count(),
    })


@bp.route('/me', methods=['GET'], endpoint='api_me')
@require_api_token
def api_me():
    """当前用户信息 + 配额 + 今日用量。"""
    user = g.api_user
    quota = current_quota(user)
    queued = Task.query.filter(Task.user_id == user.id, Task.status.in_(['pending', 'claimed', 'running'])).count()
    daily = today_task_count(user)
    return jsonify({
        'user': {'id': user.id, 'username': user.username, 'role': user.role,
                 'is_active': bool(user.is_active)},
        'quota': {
            'enabled': bool(quota.enabled),
            'max_queued_tasks': quota.max_queued_tasks,
            'max_running_tasks': quota.max_running_tasks,
            'max_input_seconds': quota.max_input_seconds,
            'max_daily_tasks': quota.max_daily_tasks,
            'max_private_models': quota.max_private_models,
            'results_retention_days': quota.results_retention_days,
        },
        'usage': {
            'queued_tasks': queued,
            'daily_tasks_used': daily,
            'daily_tasks_remaining': max((quota.max_daily_tasks or 0) - daily, 0),
        },
    })


@bp.route('/tasks', methods=['GET'], endpoint='api_task_list')
@require_api_token
def api_task_list():
    """当前用户的任务列表（按提交时间倒序）。"""
    user = g.api_user
    status = request.args.get('status', '').strip()
    page = max(request.args.get('page', 1, type=int), 1)
    per_page = 50
    q = Task.query.filter(Task.user_id == user.id)
    if status:
        q = q.filter(Task.status == status)
    total = q.count()
    pages = max((total + per_page - 1) // per_page, 1)
    page = min(page, pages)
    rows = q.order_by(Task.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()
    items = []
    for t in rows:
        pct = None
        m = re.search(r'\((\d+)%\)', t.progress_msg or '')
        if m:
            pct = int(m.group(1))
        items.append({
            'id': t.id, 'status': t.status, 'pct': pct,
            'progress_msg': t.progress_msg or '',
            'has_result': bool(t.result_filename),
            'created_at': t.created_at.strftime('%Y-%m-%d %H:%M:%S') if t.created_at else None,
        })
    return jsonify({'total': total, 'page': page, 'pages': pages, 'tasks': items})


@bp.route('/docs', methods=['GET'], endpoint='api_docs')
def api_docs():
    """接口文档页（公开，无需登录）。"""
    return render_template('api_docs.html')
