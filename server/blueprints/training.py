"""训练蓝图（可选模块，仅管理员）：提交/续训/停止/结果/日志/checkpoint 注册/进度。"""
import json
import os
import shutil
import uuid
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, send_file, jsonify, Response
from flask_login import login_required, current_user

from authorization import is_admin
from extensions import db
from db_models import Model, TrainingTask
from services.training import (
    training_enabled, set_training_enabled, ensure_training_worker, stop_training,
    chain_root_id, parse_training_log, detect_stage,
)
from services.quota import get_setting
from apputils import save_uploaded

bp = Blueprint('training', __name__)


def _guard():
    if not is_admin(current_user):
        abort(403)


def _upload():
    from flask import current_app
    return current_app.config['UPLOAD_FOLDER']


@bp.route('/train', endpoint='train_page')
@login_required
def train_page():
    _guard()
    if not training_enabled():
        return render_template('train_disabled.html')
    ensure_training_worker()
    active = TrainingTask.query.filter_by(status='running').first()
    info = parse_training_log(active) if active else {}
    history = (TrainingTask.query.order_by(TrainingTask.created_at.desc()).limit(20).all())
    resumable = []
    for t in (TrainingTask.query.filter(TrainingTask.status.in_(['done', 'failed', 'stopped']))
              .order_by(TrainingTask.created_at.desc()).limit(10).all()):
        resumable.append({'id': t.id, 'speaker': t.speaker, 'model_type': t.model_type})
    return render_template('train.html',
        active=active, history=history, resumable=resumable,
        log_content=info.get('log_tail', ''), pct=info.get('pct', 0),
        current_step=info.get('current_step', 0), total_steps=info.get('total_steps', 0),
        loss_data=json.dumps(info.get('loss_data', [])),
        current_stage=info.get('stage', ''),
        stage_label=dict([(k, l) for k, l in [('resample', '重采样'), ('config', '生成配置'),
                                              ('feature', '提取特征'), ('sovits', '训练 SoVITS'),
                                              ('diff', '训练扩散'), ('done', '完成')]]).get(info.get('stage', ''), ''),
        running=TrainingTask.query.filter_by(status='running').count(),
        queued=TrainingTask.query.filter_by(status='pending').count(),
        done_count=TrainingTask.query.filter_by(status='done').count(),
        quick_resume=get_setting('train_cpu_cores', '0'))


@bp.route('/train/enable', methods=['POST'], endpoint='train_enable')
@login_required
def train_enable():
    _guard()
    set_training_enabled(True)
    ensure_training_worker()
    flash('训练功能已开启', 'success')
    return redirect(url_for('train_page'))


@bp.route('/train/submit', methods=['POST'], endpoint='train_submit')
@login_required
def train_submit():
    _guard()
    if not training_enabled():
        abort(404)
    dataset = request.files.get('dataset')
    speaker = request.form.get('speaker', '').strip()
    if not dataset or not dataset.filename or not speaker:
        flash('请选择数据集并填写说话人名称', 'danger')
        return redirect(url_for('train_page'))

    def _i(k, d=0):
        try:
            return int(request.form.get(k, d))
        except (ValueError, TypeError):
            return d

    def _f(k, d=0.0):
        try:
            return float(request.form.get(k, d))
        except (ValueError, TypeError):
            return d

    dataset_zip = save_uploaded(dataset, 'dataset_zips')
    model_type = request.form.get('model_type', 'sovits')
    params = {
        'speech_encoder': request.form.get('speech_encoder', 'vec768l12'),
        'f0_predictor': request.form.get('f0_predictor', 'dio'),
        'arch': request.form.get('arch', 'sovits-v1'),
        'flow_mode': request.form.get('flow_mode', 'a2'),
        'use_unified_flow': request.form.get('use_unified_flow') == 'on',
        'c_fm': _f('c_fm', 0.3), 'c_mel': _f('c_mel', 45), 'c_kl': _f('c_kl', 1.0),
        'learning_rate': _f('learning_rate', 0.0001),
        'segment_size': _i('segment_size', 10240), 'lr_decay': _f('lr_decay', 0.999875),
        'auto_stop': _i('auto_stop', 200), 'log_interval': _i('log_interval', 200),
        'eval_interval': _i('eval_interval', 800), 'ema_decay': _f('ema_decay', 0.999),
        'ema_interval': _i('ema_interval', 100), 'max_speclen': _i('max_speclen', 512),
        'seed': _i('seed', 1234), 'n_layers_q': _i('n_layers_q', 3),
        'hybrid_steps': _i('hybrid_steps', 4), 'enc_q_hidden': _i('enc_q_hidden', 96),
        'd_lr_scale': _f('d_lr_scale', 1.0),
        'vol_aug': request.form.get('vol_aug') == 'on', 'warmup_epochs': _i('warmup_epochs', 0),
        'fp16_run': request.form.get('fp16_run') if request.form.get('fp16_run') else None,
        'diff_batch_size': _i('diff_batch_size', 48), 'diff_epochs': _i('diff_epochs', 100000),
        'diff_timesteps': _i('diff_timesteps', 1000), 'diff_kstep': _i('diff_kstep', 0),
        'diff_layers': _i('diff_layers', 20), 'diff_chans': _i('diff_chans', 512),
        'diff_hidden': _i('diff_hidden', 256), 'diff_lr': _f('diff_lr', 0.0001),
        'diff_decay_step': _i('diff_decay_step', 100000), 'diff_gamma': _f('diff_gamma', 0.5),
        'diff_amp': request.form.get('diff_amp', 'fp32'),
        'diff_interval_val': _i('diff_interval_val', 200), 'diff_max_steps': _i('diff_max_steps', 0),
        'report_interval': _i('report_interval', getattr(current_user, 'report_interval', 0) or 0),
    }
    task = TrainingTask(
        user_id=current_user.id, dataset_zip=dataset_zip, speaker=speaker,
        model_type=model_type, batch_size=_i('batch_size', 4), total_steps=_i('total_steps', 4200),
        keep_ckpts=_i('keep_ckpts', 3), params_json=json.dumps(params, ensure_ascii=False),
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    ensure_training_worker()
    flash(f'训练任务 #{task.id} 已提交，将在后台执行', 'success')
    return redirect(url_for('train_page'))


@bp.route('/train/quick-resume', methods=['POST'], endpoint='train_quick_resume')
@login_required
def train_quick_resume():
    _guard()
    if not training_enabled():
        abort(404)
    tid = request.form.get('quick_resume_chain', type=int) or request.form.get('resume_from', type=int)
    if not tid:
        flash('未选择要续训的任务', 'danger')
        return redirect(url_for('train_page'))
    src = db.session.get(TrainingTask, tid)
    if not src:
        flash('续训源任务不存在', 'danger')
        return redirect(url_for('train_page'))
    params = json.loads(src.params_json or '{}') if src.params_json else {}
    task = TrainingTask(
        user_id=current_user.id, dataset_zip=src.dataset_zip, speaker=src.speaker,
        model_type=src.model_type, batch_size=src.batch_size, total_steps=src.total_steps,
        keep_ckpts=src.keep_ckpts, params_json=json.dumps(params, ensure_ascii=False),
        resume_from_id=src.id, status='pending',
    )
    db.session.add(task)
    db.session.commit()
    ensure_training_worker()
    flash(f'已创建续训任务 #{task.id}（源 #{src.id}）', 'success')
    return redirect(url_for('train_page'))


@bp.route('/train/stop', methods=['POST'], endpoint='train_stop')
@login_required
def train_stop():
    _guard()
    running = TrainingTask.query.filter_by(status='running').first()
    if running:
        stop_training()
        running.status = 'stopped'
        running.progress_msg = '管理员停止'
        running.done_at = datetime.utcnow()
        db.session.commit()
        flash(f'已停止训练任务 #{running.id}', 'warning')
    else:
        flash('当前没有运行中的训练任务', 'info')
    return redirect(url_for('train_page'))


@bp.route('/train/force-restart', methods=['POST'], endpoint='train_force_restart')
@login_required
def train_force_restart():
    """强制重启整个服务器：终止残留训练/推理子进程，systemctl 重启 ssvc。
    用于训练停止后仍有进程残留、CPU 占用不降的情况。"""
    _guard()
    from services.audit import audit_log
    audit_log('train_force_restart', '管理员强制重启服务器（清残留训练进程）')
    # 先尝试优雅停掉当前训练子进程组
    try:
        stop_training()
    except Exception:
        pass
    import subprocess as _sp
    if os.name == 'nt':
        return Response(
            '<html><body style="background:#0d1117;color:#c9d1d9;font-family:sans-serif;'
            'text-align:center;padding-top:15vh"><h2>Windows 无法自动重启服务，请手动重启</h2>'
            '<p>关闭当前窗口后重新运行 start.bat</p>'
            '</body></html>', status=200)
    import threading

    def _restart():
        import time as _t
        _t.sleep(2)
        try:
            _sp.run(['systemctl', 'restart', 'ssvc'], capture_output=True, timeout=60)
        except Exception:
            pass
    threading.Thread(target=_restart, daemon=True).start()
    return Response(
        '<html><body style="background:#0d1117;color:#c9d1d9;font-family:sans-serif;'
        'text-align:center;padding-top:15vh"><h2>正在强制重启服务器…</h2>'
        '<p>残留训练进程将被清掉，服务约 5 秒后恢复。</p>'
        '<script>setTimeout(function(){location.href="/train"},6000)</script>'
        '</body></html>', status=200)


@bp.route('/train/result/<int:tid>', endpoint='train_result')
@login_required
def train_result(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t:
        abort(404)
    kind = request.args.get('diff', '') or 'model'
    want_config = request.args.get('config') == '1'
    if kind == 'diff':
        rel = t.diff_config_path if want_config else t.diff_model_path
    else:
        rel = t.config_path if want_config else t.model_path
    if not rel:
        abort(404)
    sub = 'models' if rel.lower().endswith(('.pth', '.pt')) else 'configs'
    full = os.path.join(_upload(), sub, os.path.basename(rel))
    if not os.path.exists(full):
        abort(404)
    return send_file(full, as_attachment=True, download_name=os.path.basename(rel))


@bp.route('/train/<int:tid>/log', endpoint='train_log')
@login_required
def train_log(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t or not t.log_path:
        abort(404)
    log_p = os.path.join(_upload(), 'train_logs', os.path.basename(t.log_path))
    if not os.path.exists(log_p):
        abort(404)
    with open(log_p, 'r', encoding='utf-8', errors='replace') as f:
        return '<pre style="white-space:pre-wrap;color:#ddd;background:#111;padding:12px">' + f.read()[-20000:] + '</pre>'


@bp.route('/train-tasks/<int:tid>/delete', methods=['POST'], endpoint='train_task_delete')
@login_required
def train_task_delete(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t:
        abort(404)
    if t.status == 'running':
        flash('运行中的任务不能删除', 'warning')
        return redirect(url_for('train_page'))
    db.session.delete(t)
    db.session.commit()
    flash(f'已删除训练任务 #{tid}', 'success')
    return redirect(url_for('train_page'))


@bp.route('/api/train/<int:tid>/status', endpoint='api_train_status')
@login_required
def api_train_status(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t:
        abort(404)
    info = parse_training_log(t)
    return jsonify({'status': t.status, 'progress_msg': t.progress_msg,
                    'stage_label': info['stage'], 'pct': info['pct'],
                    'current_step': info['current_step'], 'total_steps': info['total_steps'],
                    'log_tail': info['log_tail']})


@bp.route('/api/train/<int:tid>/loss_chart.json', endpoint='api_train_loss_chart')
@login_required
def api_train_loss_chart(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t:
        abort(404)
    info = parse_training_log(t)
    return jsonify({'task_id': tid, 'status': t.status, 'total_steps': t.total_steps or 0,
                    'loss_data': info['loss_data'], 'eval_mel': info['eval_mel'], 'eval_step': info['eval_step']})


@bp.route('/api/train/<int:tid>/register_checkpoints', methods=['POST'], endpoint='api_train_register_checkpoints')
@login_required
def api_train_register_checkpoints(tid):
    _guard()
    t = db.session.get(TrainingTask, tid)
    if not t:
        abort(404)
    root_id = chain_root_id(t)
    td = os.path.join(_upload(), 'train_data', f'task_{root_id}')
    if not os.path.isdir(td):
        return jsonify({'ok': False, 'msg': '任务目录不存在', 'registered': [], 'skipped': 0}), 404
    ckpts = []
    for f in sorted(os.listdir(td)):
        if f.startswith('G_') and f.endswith('.pth') and f != 'G_0.pth':
            n = int(''.join(c for c in f if c.isdigit()) or 0)
            ckpts.append((f, n))
    if not ckpts:
        return jsonify({'ok': False, 'msg': '未找到 checkpoint', 'registered': [], 'skipped': 0})
    models_dir = os.path.join(_upload(), 'models')
    configs_dir = os.path.join(_upload(), 'configs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    cluster_name = None
    cand = f'{t.speaker}_cluster.pth'
    if os.path.exists(os.path.join(models_dir, cand)):
        cluster_name = cand
    cfg_src = os.path.join(td, 'config.json')
    registered = []
    skipped = 0
    for fname, step in ckpts:
        name = f'{t.speaker}-G{step}step-task{tid}'
        if Model.query.filter_by(user_id=current_user.id, name=name).first():
            skipped += 1
            continue
        try:
            model_name = f'{uuid.uuid4().hex[:8]}_{fname}'
            dst = os.path.join(models_dir, model_name)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(os.path.join(td, fname), dst)
            except OSError:
                shutil.copy2(os.path.join(td, fname), dst)
            cfg_name = None
            if os.path.exists(cfg_src):
                cfg_name = f'{uuid.uuid4().hex[:8]}_config_{fname.replace(".pth", ".json")}'
                shutil.copy2(cfg_src, os.path.join(configs_dir, cfg_name))
            db.session.add(Model(user_id=current_user.id, name=name, model_path=model_name,
                                 config_path=cfg_name or '', cluster_path=cluster_name))
            registered.append(name)
        except Exception:
            continue
    db.session.commit()
    return jsonify({'ok': True, 'msg': f'已注册 {len(registered)} 个 checkpoint' + (f'，跳过 {skipped} 个' if skipped else ''),
                    'registered': registered, 'skipped': skipped})
