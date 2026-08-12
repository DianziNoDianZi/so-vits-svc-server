"""推理蓝图：提交推理、进度轮询/SSE。"""
import json
import os
import re
import time
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, jsonify, Response
from flask_login import login_required, current_user

from authorization import can_use_model
from extensions import db
from db_models import DEFAULT_PARAMS, InferenceConfig, Model, Task
from services.quota import current_quota, today_task_count
from services.scheduler import ensure_worker
from apputils import save_uploaded

bp = Blueprint('inference', __name__)


def _current_upload():
    from flask import current_app
    return current_app.config['UPLOAD_FOLDER']


def _build_inference_config_items(configs, user):
    items = []
    hidden = 0
    for c in configs:
        if not can_use_model(user, c.model):
            hidden += 1
            continue
        try:
            c_params = json.loads(c.params_json or '{}')
        except Exception:
            c_params = {}
        m_info = {'arch': '', 'flow_mode': '', 'use_unified_flow': False, 'step': ''}
        if c.model.config_path:
            from apputils import read_model_cfg
            _cfg = read_model_cfg(c.model.config_path)
            _m = _cfg.get('model') or {}
            m_info['arch'] = _m.get('arch', '') or 'sovits-v1'
            m_info['flow_mode'] = _m.get('flow_mode', '') or ('a2' if _m.get('use_unified_flow') else '')
            m_info['use_unified_flow'] = bool(_m.get('use_unified_flow'))
        _sm = re.search(r'G(\d+)step|G(\d+)', c.model.name)
        if _sm:
            m_info['step'] = _sm.group(1) or _sm.group(2)
        items.append({'id': c.id, 'name': c.name, 'model': c.model.name, 'model_id': c.model.id,
                      'params': c_params, 'has_cluster': bool(c.model.cluster_path), 'model_info': m_info})
    return items, hidden


_AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus')
MAX_BATCH = 10


def _allowed_audio(name):
    return name.lower().endswith(_AUDIO_EXTS)


def _submit_one(cfg_obj, model, quota, audio_file, user):
    """提交单个音频：返回 ('ok', None) 或 ('skip', reason)。

    一次传 10 个音频就循环调它十遍，各自独立落库成任务。
    谁家的规则？每天的配额按"任务个数"算，不是按音频秒数——之前按秒数算，
    长音频一提交就把额度吃光，短音频却随便灌，总觉得哪里不对劲。
    """
    if not audio_file or not audio_file.filename:
        return 'skip', '空文件'
    if not _allowed_audio(audio_file.filename):
        return 'skip', f'不支持的类型'
    audio_filename = save_uploaded(audio_file, 'audio')
    audio_path = os.path.join(_current_upload(), 'audio', audio_filename)
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
        return 'skip', f'超 {int(quota.max_input_seconds)} 秒'

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

    task = Task(
        user_id=user.id, config_id=cfg_obj.id, model_id=model.id,
        audio_filename=audio_filename,
        params_json=json.dumps(cfg_params, ensure_ascii=False),
        device_pref=user.device_pref or 'auto',
        memory_limit=user.memory_limit or 0,
        input_bytes=audio_size, input_duration=audio_duration,
        priority_snapshot=quota.priority,
        quota_snapshot_json=json.dumps({
            'max_queued_tasks': quota.max_queued_tasks, 'max_running_tasks': quota.max_running_tasks,
            'max_input_seconds': quota.max_input_seconds, 'max_daily_tasks': quota.max_daily_tasks,
            'max_cpu_cores': quota.max_cpu_cores, 'results_retention_days': quota.results_retention_days,
        }, ensure_ascii=False),
        result_expires_at=datetime.utcnow() + timedelta(days=max(quota.results_retention_days or 7, 1)),
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    return 'ok', None


@bp.route('/inference', methods=['GET', 'POST'], endpoint='inference')
@login_required
def inference():
    configs = InferenceConfig.query.filter_by(user_id=current_user.id).all()
    config_items, hidden = _build_inference_config_items(configs, current_user)

    if request.method == 'POST':
        config_id = request.form.get('config_id', type=int)
        files = [f for f in request.files.getlist('audio_file') if f and f.filename]
        if not config_id or not files:
            flash('请选择配置和音频文件', 'danger')
            return render_template('inference.html', configs=config_items)
        cfg_obj = InferenceConfig.query.get_or_404(config_id)
        if cfg_obj.user_id != current_user.id:
            abort(403)
        model = db.session.get(Model, cfg_obj.model_id)
        if not model or not can_use_model(current_user, model):
            flash('所选模型当前不可用', 'danger')
            return render_template('inference.html', configs=config_items)

        quota = current_quota(current_user)
        if not quota.enabled:
            flash('当前账号已被禁用，无法提交推理', 'danger')
            return render_template('inference.html', configs=config_items)

        queued_count = Task.query.filter(Task.user_id == current_user.id, Task.status.in_(['pending', 'running'])).count()
        daily_used = today_task_count(current_user)
        limit = len(files)
        limit = min(limit, MAX_BATCH)
        if quota.max_queued_tasks:
            limit = min(limit, max(quota.max_queued_tasks - queued_count, 0))
        if quota.max_daily_tasks:
            limit = min(limit, max(quota.max_daily_tasks - daily_used, 0))
        if limit <= 0:
            flash('已达到任务提交上限（排队数或每日任务数已满）', 'danger')
            return render_template('inference.html', configs=config_items)

        ensure_worker()
        created = 0
        skipped = []
        for f in files[:limit]:
            status, reason = _submit_one(cfg_obj, model, quota, f, current_user)
            if status == 'ok':
                created += 1
            else:
                skipped.append(f'{f.filename}: {reason}')
        extra = len(files) - limit
        if extra > 0:
            skipped.append(f'其余 {extra} 个文件超出配额未提交')

        if created:
            msg = f'已提交 {created} 个推理任务'
            if skipped:
                msg += f'，跳过 {len(skipped)} 个（{"；".join(skipped[:2])}）'
            flash(msg, 'success')
        else:
            flash('没有任务被提交' + ('：' + '；'.join(skipped[:3]) if skipped else ''), 'danger')
        return redirect(url_for('task_list'))

    if hidden:
        flash(f'{hidden} 个配置因模型不可用已隐藏', 'warning')
    global_queue = Task.query.filter(Task.status.in_(['pending', 'claimed'])).count()
    return render_template('inference.html', configs=config_items, global_queue=global_queue)


@bp.route('/api/tasks/<int:task_id>/status', endpoint='api_task_status')
@login_required
def api_task_status(task_id):
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)
    pct = None
    m = re.search(r'\((\d+)%\)', task.progress_msg or '')
    if m:
        pct = int(m.group(1))
    return jsonify({
        'status': task.status, 'progress_msg': task.progress_msg, 'pct': pct,
        'has_result': bool(task.result_filename), 'error_msg': (task.error_msg or '')[:200],
    })


@bp.route('/api/tasks/<int:task_id>/stream', endpoint='api_task_stream')
@login_required
def api_task_stream(task_id):
    task = db.session.get(Task, task_id)
    if not task or task.user_id != current_user.id:
        abort(404)

    def gen():
        sent = 0
        while True:
            from flask import current_app
            with current_app.app_context():
                t = db.session.get(Task, task_id)
                if not t:
                    yield 'event: done\ndata: {"status":"gone"}\n\n'
                    break
                if t.status == 'running':
                    pct = None
                    m = re.search(r'\((\d+)%\)', t.progress_msg or '')
                    if m:
                        pct = int(m.group(1))
                    payload = json.dumps({'status': 'running', 'progress_msg': t.progress_msg or '', 'pct': pct}, ensure_ascii=False)
                    yield f'event: progress\ndata: {payload}\n\n'
                else:
                    payload = json.dumps({'status': t.status, 'progress_msg': t.progress_msg or '', 'error_msg': t.error_msg or ''}, ensure_ascii=False)
                    yield f'event: done\ndata: {payload}\n\n'
                    break
            sent += 1
            if sent > 3600 * 2:
                break
            time.sleep(1)

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
