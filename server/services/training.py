"""训练可选模块服务：开关、训练 worker daemon。"""
import json
import os
import threading
import time
import traceback
from datetime import datetime

from extensions import db
from db_models import TrainingTask
from services.quota import get_setting, set_setting

_app = None


def init_app(app):
    global _app
    _app = app


def training_enabled():
    # 训练是"可选功能"，默认关着，只有管理员能开。
    # 原因很简单：训练又吃显存又占机器，普通人开着只会让大家的推理一起排队。
    # 谁想用，谁找管理员开，别默认就怼脸上。
    return get_setting('training_enabled', '0') == '1'


def set_training_enabled(on):
    set_setting('training_enabled', '1' if on else '0')


_train_started = False


def ensure_training_worker():
    global _train_started
    if _train_started:
        return
    _train_started = True
    threading.Thread(target=_train_worker_daemon, daemon=True).start()


def _train_worker_daemon():
    with _app.app_context():
        while True:
            try:
                time.sleep(3)
                if not training_enabled():
                    continue
                task = (TrainingTask.query.filter_by(status='pending')
                        .order_by(TrainingTask.created_at.asc()).first())
                if task:
                    _run_training(task)
            except Exception:
                traceback.print_exc()
                time.sleep(5)


def _run_training(task):
    task.status = 'running'
    task.progress_msg = '开始训练...'
    db.session.commit()
    params = json.loads(task.params_json or '{}')
    log_dir = os.path.join(_app.config['UPLOAD_FOLDER'], 'train_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'task_{task.id}.log')
    task.log_path = f'task_{task.id}.log'
    db.session.commit()

    from train_worker import run as train_run

    old_omp = os.environ.get('OMP_NUM_THREADS')
    old_mkl = os.environ.get('MKL_NUM_THREADS')
    cores = int(get_setting('train_cpu_cores', '0') or 0)
    if cores > 0:
        os.environ['OMP_NUM_THREADS'] = str(cores)
        os.environ['MKL_NUM_THREADS'] = str(cores)
    try:
        result = train_run(
            task_id=task.id,
            speaker=task.speaker,
            dataset_zip=task.dataset_zip,
            log_path=log_path,
            model_type=task.model_type or 'sovits',
            batch_size=task.batch_size or 4,
            total_steps=task.total_steps or 4200,
            keep_ckpts=task.keep_ckpts or 3,
            speech_encoder=params.get('speech_encoder', 'vec768l12'),
            f0_predictor=params.get('f0_predictor', 'dio'),
            learning_rate=params.get('learning_rate', 0.0001),
            segment_size=params.get('segment_size', 10240),
            lr_decay=params.get('lr_decay', 0.999875),
            auto_stop=params.get('auto_stop', 200),
            log_interval=params.get('log_interval', 200),
            eval_interval=params.get('eval_interval', 800),
            arch=params.get('arch', 'sovits-v1'),
            d_lr_scale=params.get('d_lr_scale', 1.0),
            flow_mode=params.get('flow_mode', 'a2'),
            use_unified_flow=bool(params.get('use_unified_flow', False)),
            c_fm=params.get('c_fm', 0.3),
            c_mel=params.get('c_mel', 45),
            c_kl=params.get('c_kl', 1.0),
            ema_decay=params.get('ema_decay', 0.999),
            ema_interval=params.get('ema_interval', 100),
            max_speclen=params.get('max_speclen', 512),
            fp16_run=params.get('fp16_run'),
            vol_aug=bool(params.get('vol_aug', False)),
            warmup_epochs=params.get('warmup_epochs', 0),
            seed=params.get('seed', 1234),
            n_layers_q=params.get('n_layers_q', 3),
            hybrid_steps=params.get('hybrid_steps', 4),
            enc_q_hidden=params.get('enc_q_hidden', 96),
            diff_batch_size=params.get('diff_batch_size', 48),
            diff_epochs=params.get('diff_epochs', 100000),
            diff_timesteps=params.get('diff_timesteps', 1000),
            diff_kstep=params.get('diff_kstep', 0),
            diff_layers=params.get('diff_layers', 20),
            diff_chans=params.get('diff_chans', 512),
            diff_hidden=params.get('diff_hidden', 256),
            diff_lr=params.get('diff_lr', 0.0001),
            diff_decay_step=params.get('diff_decay_step', 100000),
            diff_gamma=params.get('diff_gamma', 0.5),
            diff_amp=params.get('diff_amp', 'fp32'),
            diff_interval_val=params.get('diff_interval_val', 200),
            diff_max_steps=params.get('diff_max_steps', 0),
            resume_from_id=task.resume_from_id or 0,
            diff_root_id=task.resume_from_id or 0,
        )
    except Exception as e:
        result = {'status': 'failed', 'error_msg': f'{type(e).__name__}: {e}',
                  'progress_msg': '训练异常', 'model_path': None, 'config_path': None,
                  'diff_model_path': None, 'diff_config_path': None, 'log': traceback.format_exc()}
    finally:
        if old_omp is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = old_omp
        if old_mkl is None:
            os.environ.pop('MKL_NUM_THREADS', None)
        else:
            os.environ['MKL_NUM_THREADS'] = old_mkl

    task.status = result.get('status', 'failed')
    task.progress_msg = (result.get('progress_msg') or '')[:500]
    task.error_msg = (result.get('error_msg') or '')[:1000]
    task.model_path = result.get('model_path') or None
    task.config_path = result.get('config_path') or None
    task.diff_model_path = result.get('diff_model_path') or None
    task.diff_config_path = result.get('diff_config_path') or None
    task.done_at = datetime.utcnow()
    db.session.commit()


def stop_training():
    from train_worker import stop as _stop
    return _stop()


def chain_root_id(task):
    """沿 resume_from 链向上找最原始的训练任务 id（checkpoint 归属目录）。"""
    seen = set()
    root = task.id
    cur = task.resume_from_id
    while cur and cur not in seen:
        seen.add(cur)
        root = cur
        prev = db.session.get(TrainingTask, cur)
        cur = prev.resume_from_id if prev else None
    return root


_STAGE_LABELS = [
    ('resample', '重采样'), ('config', '生成配置'), ('feature', '提取特征'),
    ('sovits', '训练 SoVITS'), ('diff', '训练扩散'), ('done', '完成'),
]


def detect_stage(progress_msg):
    text = progress_msg or ''
    for key, label in _STAGE_LABELS:
        if key in text:
            return key
    return ''


def parse_training_log(task):
    """读取训练日志，返回 {stage, pct, current_step, total_steps, loss_data, eval_mel, eval_step, log_tail}。"""
    import re as _re
    total_steps = task.total_steps or 0
    info = {'stage': '', 'pct': 0, 'current_step': 0, 'total_steps': total_steps,
            'loss_data': [], 'eval_mel': None, 'eval_step': 0, 'log_tail': ''}
    if not task.log_path:
        return info
    log_p = os.path.join(_app.config['UPLOAD_FOLDER'], 'train_logs', os.path.basename(task.log_path))
    if not os.path.exists(log_p):
        return info
    try:
        with open(log_p, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return info
    tail = ''.join(lines[-40:])
    info['log_tail'] = tail
    info['stage'] = detect_stage(tail[-2000:])
    # 解析损失行，如 "step 100 | G: 5.2 | D: 1.3 | mel: 0.41"
    for line in lines:
        m = _re.search(r'step\s+(\d+)', line)
        g = _re.search(r'G:\s*([\d.]+)', line)
        d = _re.search(r'D:\s*([\d.]+)', line)
        mel = _re.search(r'mel:\s*([\d.]+)', line)
        if m and (g or d):
            point = {'step': int(m.group(1))}
            if g:
                point['g'] = float(g.group(1))
            if d:
                point['d'] = float(d.group(1))
            if mel:
                point['mel'] = float(mel.group(1))
            info['loss_data'].append(point)
    info['current_step'] = info['loss_data'][-1]['step'] if info['loss_data'] else 0
    info['pct'] = min(int(info['current_step'] * 100 / max(total_steps, 1)), 99) if total_steps else 0
    return info
