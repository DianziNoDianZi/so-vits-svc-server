"""后台调度服务：推理调度、worker、清理、资源监控，以及推理 daemon 生命周期。"""
import gc
import json
import os
import sys
import uuid
import signal
import queue as queue_module
import multiprocessing as mp_module
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta

from extensions import db
from db_models import DEFAULT_PARAMS, InferenceConfig, Model, Task, User
from authorization import can_use_model, is_active_user
from services.quota import current_quota, get_setting, set_setting

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_app = None


def init_app(app):
    global _app
    _app = app


# ========== 全局状态 ==========
task_queue = queue_module.Queue()
task_worker_started = False
task_scheduler_started = False
task_processes = {}
task_processes_lock = threading.Lock()
inference_active_id = None
scheduler_last_user_id = None
scheduler_lock = threading.Lock()
cleanup_started = False
resource_monitor_started = False
watchdog_started = False
resource_warning = {'active': False, 'message': ''}

inference_q = None
inference_done_q = None
inference_daemon_proc = None
INFERENCE_MODEL_CACHE = int(os.environ.get('INFERENCE_MODEL_CACHE', '1') or 1)
# 长段推理内存会爆（3.8G 小鸡上 70s 一段直接 OOM），强制切成 clip_seconds 秒的小块推理。
INFERENCE_CLIP_SECONDS = float(os.environ.get('INFERENCE_CLIP_SECONDS', '15') or 15)


def _stream_lines(stream):
    buf = b''
    while True:
        try:
            chunk = stream.read1(65536) if hasattr(stream, 'read1') else stream.read(65536)
        except Exception:
            break
        if not chunk:
            break
        buf += chunk
        while True:
            idx = -1
            for sep in (b'\r', b'\n'):
                i = buf.find(sep)
                if i != -1 and (idx == -1 or i < idx):
                    idx = i
            if idx == -1:
                break
            line = buf[:idx]
            buf = buf[idx + 1:]
            if line.strip():
                yield line
    if buf.strip():
        yield buf


def _update_progress(task, msg, force=False):
    now = time.time()
    if not force and now - getattr(_update_progress, '_last', 0) < 1.0:
        return
    _update_progress._last = now
    task.progress_msg = msg
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()


_STALE_LEASE_SECONDS = 5 * 60


def _recover_expired_leases():
    now = datetime.utcnow()
    candidates = Task.query.filter(Task.status.in_(['claimed', 'running'])).all()
    changed = False
    for t in candidates:
        hb = t.heartbeat_at
        lease = t.lease_expires_at
        stale_hb = hb is None or (now - hb).total_seconds() > _STALE_LEASE_SECONDS
        stale_lease = lease is not None and lease < now
        if stale_hb or stale_lease:
            t.status = 'pending'
            t.claimed_by = None
            t.lease_expires_at = None
            t.heartbeat_at = None
            t.attempt_count = (t.attempt_count or 0) + 1
            t.progress_msg = '执行器失联，重新排队'
            changed = True
    if changed:
        db.session.commit()


def task_scheduler_daemon():
    global scheduler_last_user_id
    with _app.app_context():
        while True:
            try:
                _recover_expired_leases()
                if get_setting('scheduler_paused', '0') == '1':
                    time.sleep(2)  # 管理员暂停新任务调度（运行中任务不受影响）
                    continue
                if inference_active_id is not None or not task_queue.empty():
                    time.sleep(1)
                    continue
                pending = (Task.query.filter(Task.status == 'pending')
                           .order_by(Task.created_at.asc(), Task.id.asc()).all())
                if not pending:
                    time.sleep(1)
                    continue
                eligible = []
                for t in pending:
                    if not t.user or not is_active_user(t.user):
                        continue
                    quota = current_quota(t.user)
                    if not quota.enabled:
                        continue
                    running = Task.query.filter(Task.user_id == t.user_id, Task.status == 'running').count()
                    # 这里当初是个差点让用户崩溃的坑：queued 把"候选任务自己"也算进去了，
                    # 用的是 >=。于是谁把 max_queued_tasks 设成 1，谁的唯一一个任务就永远排不上——
                    # 用户："我上传了个推理人物，怎么一直排队？" 改回严格大于才恢复正常。
                    # 想当然写 >= 的我，给用户道个歉。
                    queued = Task.query.filter(Task.user_id == t.user_id, Task.status.in_(['pending', 'claimed', 'running'])).count()
                    if quota.max_running_tasks and running >= quota.max_running_tasks:
                        continue
                    if quota.max_queued_tasks and queued > quota.max_queued_tasks:
                        continue
                    if t.model_id:
                        model = db.session.get(Model, t.model_id)
                        if not model or not can_use_model(t.user, model):
                            t.status = 'failed'
                            t.error_msg = '模型不可用'
                            t.progress_msg = '失败: 模型不可用'
                            t.done_at = datetime.utcnow()
                            db.session.commit()
                            continue
                    eligible.append((t, quota))
                if not eligible:
                    time.sleep(1)
                    continue
                picked = None
                if scheduler_last_user_id is not None:
                    for t, quota in eligible:
                        if t.user_id != scheduler_last_user_id:
                            picked = (t, quota)
                            break
                if picked is None:
                    eligible.sort(key=lambda x: (-int(x[1].priority or 1), x[0].created_at, x[0].id))
                    picked = eligible[0]
                task, quota = picked
                task.status = 'claimed'
                task.progress_msg = '等待执行器...'
                task.lease_expires_at = datetime.utcnow() + timedelta(minutes=10)
                # 认领时就得把心跳写上去，不然 _recover_expired_leases 一看 heartbeat 是空，
                # 直接当"执行器失联"把刚领的任务踹回排队，来回抖到天荒地老。
                task.heartbeat_at = datetime.utcnow()
                task.claimed_by = 'scheduler'
                task.priority_snapshot = quota.priority
                db.session.commit()
                scheduler_last_user_id = task.user_id
                task_queue.put(task.id)
            except Exception:
                traceback.print_exc()
                time.sleep(2)


def task_worker():
    global inference_active_id
    with _app.app_context():
        while True:
            task_id = task_queue.get()
            task = db.session.get(Task, task_id)
            if not task:
                continue
            try:
                if task.status in ('stopped', 'cancel_requested'):
                    if task.status == 'cancel_requested':
                        task.status = 'stopped'
                        task.progress_msg = '已停止（排队中取消）'
                        task.done_at = datetime.utcnow()
                        db.session.commit()
                    continue
                inference_active_id = task.id
                task.status = 'running'
                task.progress_msg = '正在加载模型...'
                task.heartbeat_at = datetime.utcnow()
                task.lease_expires_at = datetime.utcnow() + timedelta(hours=6)
                db.session.commit()

                cfg_obj = db.session.get(InferenceConfig, task.config_id)
                model = db.session.get(Model, cfg_obj.model_id) if cfg_obj else None
                if not cfg_obj or not model or not is_active_user(task.user) or not can_use_model(task.user, model):
                    task.status = 'failed'
                    task.error_msg = '模型不可用'
                    task.progress_msg = '失败: 模型不可用'
                    task.done_at = datetime.utcnow()
                    db.session.commit()
                    continue

                if task.params_json:
                    params = json.loads(task.params_json)
                else:
                    params = json.loads(cfg_obj.params_json) if cfg_obj.params_json else DEFAULT_PARAMS.copy()
                if params.get('cluster_ratio', 0) > 0 and not model.cluster_path:
                    params['cluster_ratio'] = 0
                    task.progress_msg = '模型未挂载检索索引，cluster_ratio 已置 0'

                upload = _app.config['UPLOAD_FOLDER']
                audio_path = os.path.join(upload, 'audio', task.audio_filename)
                k_step = params.get('k_step', 0)
                result_name = f'task_{task_id}_{uuid.uuid4().hex[:8]}.{params.get("output_format", "wav")}'
                result_path = os.path.join(upload, 'results', result_name)
                task.progress_msg = '正在推理中...'
                db.session.commit()

                worker = os.path.join(SERVER_DIR, 'inference_worker.py')
                full_model = os.path.join(upload, 'models', model.model_path)
                full_config = os.path.join(upload, 'configs', model.config_path)
                full_diff = os.path.join(upload, 'models', model.diff_model_path) if model.diff_model_path else 'none'
                full_diff_config = os.path.join(upload, 'configs', model.diff_config_path) if model.diff_config_path else 'none'
                full_cluster = os.path.join(upload, 'models', model.cluster_path) if model.cluster_path else 'none'
                if inference_q is None:
                    raise RuntimeError('推理 daemon 未启动')
                payload = {
                    'model_path': full_model, 'config_path': full_config,
                    'audio_path': audio_path, 'result_path': result_path,
                    'k_step': k_step, 'params': params,
                    'diff_path': full_diff, 'diff_config_path': full_diff_config,
                    'cluster_path': full_cluster,
                    'device': task.device_pref or 'auto',
                    'max_cpu_cores': int(current_quota(task.user).max_cpu_cores or 0),
                    'clip_seconds': INFERENCE_CLIP_SECONDS,
                }
                inference_q.put((task_id, payload))

                prog_path = result_path + '.prog'
                import soundfile as sf
                try:
                    audio_info = sf.info(audio_path)
                    if INFERENCE_CLIP_SECONDS > 0:
                        # 切块是固定 clip_seconds 一刀，总段数 = ceil(时长/刀宽)，直接精确算
                        import math as _math
                        estimated_total = max(_math.ceil(audio_info.duration / INFERENCE_CLIP_SECONDS), 1)
                    else:
                        estimated_total = max(int(audio_info.duration / 10) + 1, 1)
                except Exception:
                    estimated_total = 5
                tail_lines = deque(maxlen=100)
                started_at = time.time()
                task_timeout = int(os.environ.get('INFERENCE_TASK_TIMEOUT', str(6 * 3600)))
                est_total_fixed = estimated_total
                result_ok = False
                result_err = None
                hb_count = 0
                while True:
                    hb_count += 1
                    if hb_count % 30 == 0:
                        task.heartbeat_at = datetime.utcnow()
                        try:
                            db.session.commit()
                        except Exception:
                            db.session.rollback()
                    if hb_count % 5 == 0:
                        try:
                            cur = db.session.get(Task, task_id)
                        except Exception:
                            cur = None
                        if cur and cur.status == 'cancel_requested':
                            if inference_daemon_proc and inference_daemon_proc.is_alive():
                                try:
                                    os.kill(inference_daemon_proc.pid, signal.SIGINT)
                                except Exception:
                                    pass
                            result_err = '用户停止推理'
                            break
                    segments_done = 0
                    if task_timeout > 0 and time.time() - started_at > task_timeout:
                        result_err = f'推理超过 {task_timeout // 3600}h 仍未完成，已终止'
                        break
                    if os.path.exists(result_path + '.done'):
                        result_ok = True
                        break
                    prog_exists = False
                    try:
                        with open(prog_path, 'r', encoding='utf-8', errors='replace') as pf:
                            prog_text = pf.read()
                        prog_exists = True
                        for line in prog_text.splitlines():
                            text = line.strip()
                            if not text:
                                continue
                            tail_lines.append(text)
                            # 切块模式下每小块打 #=====segment clip start，只数小块，
                            # 与上方按 ceil(时长/刀宽) 算出的总段数严格对齐；非切块才数大块行
                            if INFERENCE_CLIP_SECONDS > 0:
                                if '=====segment clip start' in text:
                                    segments_done += 1
                            elif '#=====segment start' in text:
                                segments_done += 1
                    except OSError:
                        pass
                    if prog_exists:
                        base_pct = min(int(segments_done * 100 / max(est_total_fixed, 1)), 99)
                        _update_progress(task, f'推理中 ({base_pct}%) — 已处理 {segments_done} 段')
                    else:
                        _update_progress(task, '正在加载模型/等待推理...')
                    try:
                        tid, ok, err = inference_done_q.get_nowait()
                    except Exception:
                        tid, ok, err = None, False, None
                    if tid is not None:
                        if tid == task_id:
                            result_ok, result_err = ok, err
                            break
                    time.sleep(1)

                if result_ok:
                    task.result_filename = result_name
                    task.status = 'done'
                    task.progress_msg = '推理完成'
                elif result_err and '用户停止推理' in str(result_err):
                    task.status = 'stopped'
                    task.progress_msg = '已停止'
                else:
                    task.status = 'failed'
                    task.error_msg = (result_err or '\n'.join(tail_lines))[-1000:]
                    task.progress_msg = f'推理失败: {(result_err or "未知错误")[:60]}'
                task.done_at = datetime.utcnow()

                try:
                    if task.user and task.user.infer_notify:
                        from notifier import notify_inference_complete
                        notify_inference_complete(task, os.environ.get('SSVC_SERVER_URL', 'http://127.0.0.1:5000'))
                except Exception:
                    pass
            except Exception as e:
                traceback.print_exc()
                task.status = 'failed'
                task.error_msg = f'{type(e).__name__}: {e}'[:200]
                task.progress_msg = f'失败: {str(e)[:80]}'
            finally:
                inference_active_id = None
                try:
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass


def ensure_worker():
    global task_worker_started, task_scheduler_started, inference_q, inference_done_q, inference_daemon_proc, cleanup_started, resource_monitor_started, watchdog_started
    if inference_q is None:
        try:
            inference_q = mp_module.Queue()
            inference_done_q = mp_module.Queue()
        except Exception as e:
            print(f'[ensure_worker] 推理队列创建失败: {e}', flush=True)
            inference_q = None
            inference_done_q = None
    if inference_q is not None and (inference_daemon_proc is None or not inference_daemon_proc.is_alive()):
        try:
            from inference_daemon import main as _daemon_main
            inference_daemon_proc = mp_module.Process(
                target=_daemon_main, args=(inference_q, inference_done_q, INFERENCE_MODEL_CACHE),
                daemon=True, name='inference-daemon')
            inference_daemon_proc.start()
            print(f'[ensure_worker] 推理 daemon 已启动 (pid={inference_daemon_proc.pid}, cache={INFERENCE_MODEL_CACHE})', flush=True)
        except Exception as e:
            print(f'[ensure_worker] 推理 daemon 启动失败: {e}', flush=True)
            traceback.print_exc()
    if not task_scheduler_started:
        task_scheduler_started = True
        threading.Thread(target=task_scheduler_daemon, daemon=True).start()
    if not task_worker_started:
        task_worker_started = True
        threading.Thread(target=task_worker, daemon=True).start()
    if not cleanup_started:
        cleanup_started = True
        threading.Thread(target=_cleanup_daemon, daemon=True).start()
    if not resource_monitor_started:
        resource_monitor_started = True
        threading.Thread(target=_resource_monitor_daemon, daemon=True).start()
    if not watchdog_started:
        watchdog_started = True
        threading.Thread(target=_watchdog_daemon, daemon=True).start()


def _recover_tasks():
    with _app.app_context():
        for t in Task.query.filter(Task.status.in_(['pending', 'claimed', 'running'])).all():
            t.status = 'pending'
            t.claimed_by = None
            t.lease_expires_at = None
            t.progress_msg = '服务重启后重新排队'
        db.session.commit()


def _watchdog_daemon():
    """监控推理 daemon 进程。daemon 被 OOM/段错误杀死（Python except 拦不住）时，
    把它正在跑的任务标记为失败并自动重启 daemon，避免后续任务永远卡在"推理中"。
    """
    global inference_daemon_proc, inference_active_id
    with _app.app_context():
        while True:
            time.sleep(10)
            try:
                if inference_q is None or inference_done_q is None:
                    continue
                if inference_daemon_proc is not None and inference_daemon_proc.is_alive():
                    continue
                # 进程已死（可能 OOM），先回收僵尸，再处理任务和重启
                if inference_daemon_proc is not None:
                    try:
                        inference_daemon_proc.join(timeout=1)
                    except Exception:
                        pass
                tid = inference_active_id
                if tid is not None:
                    try:
                        t = db.session.get(Task, tid)
                        if t and t.status in ('running', 'claimed', 'pending'):
                            t.status = 'failed'
                            t.error_msg = '推理进程崩溃（可能内存不足），任务失败'
                            t.progress_msg = '失败: 推理进程崩溃（可能内存不足）'
                            t.done_at = datetime.utcnow()
                            db.session.commit()
                        # 塞一个失败通知，让 task_worker 从等待 .done 的循环里醒过来继续
                        inference_done_q.put((tid, False, '推理进程崩溃'))
                    except Exception:
                        db.session.rollback()
                try:
                    from inference_daemon import main as _daemon_main
                    inference_daemon_proc = mp_module.Process(
                        target=_daemon_main, args=(inference_q, inference_done_q, INFERENCE_MODEL_CACHE),
                        daemon=True, name='inference-daemon')
                    inference_daemon_proc.start()
                    print(f'[watchdog] 推理 daemon 崩溃后已自动重启 (pid={inference_daemon_proc.pid})', flush=True)
                except Exception as e:
                    print(f'[watchdog] 推理 daemon 重启失败: {e}', flush=True)
                inference_active_id = None
            except Exception:
                traceback.print_exc()


_RESOURCE_THRESHOLD = float(os.environ.get('RESOURCE_THRESHOLD', '90'))
_RESOURCE_EMAIL_INTERVAL = int(os.environ.get('RESOURCE_EMAIL_INTERVAL', '3600'))


def _resource_monitor_daemon():
    global resource_warning
    try:
        import psutil as _psutil
    except Exception:
        _psutil = None
    with _app.app_context():
        last_was_warning = False
        if _psutil is not None:
            try:
                _psutil.cpu_percent(interval=None)
            except Exception:
                pass
        while True:
            try:
                time.sleep(60)
                reasons = []
                if _psutil is not None:
                    try:
                        cpu = _psutil.cpu_percent(interval=None)
                        if cpu >= _RESOURCE_THRESHOLD:
                            reasons.append(f'CPU {cpu:.0f}%')
                    except Exception:
                        pass
                    try:
                        mem = _psutil.virtual_memory().percent
                        if mem >= _RESOURCE_THRESHOLD:
                            reasons.append(f'内存 {mem:.0f}%')
                    except Exception:
                        pass
                    try:
                        disk = _psutil.disk_usage(_app.config['UPLOAD_FOLDER']).percent
                        if disk >= _RESOURCE_THRESHOLD:
                            reasons.append(f'磁盘 {disk:.0f}%')
                    except Exception:
                        pass
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        free, total = _torch.cuda.mem_get_info()
                        gpu = (total - free) / total * 100
                        if gpu >= _RESOURCE_THRESHOLD:
                            reasons.append(f'显存 {gpu:.0f}%')
                except Exception:
                    pass
                warning = len(reasons) > 0
                if warning and not last_was_warning:
                    msg = '机器资源紧张（' + '、'.join(reasons) + '），推理可能变慢'
                    resource_warning = {'active': True, 'message': msg}
                    _email_resource_warning(msg)
                elif not warning and last_was_warning:
                    resource_warning = {'active': False, 'message': ''}
                last_was_warning = warning
            except Exception:
                time.sleep(10)


def _email_resource_warning(message):
    try:
        last = get_setting('resource_email_at', '0')
        if time.time() - int(last or 0) < _RESOURCE_EMAIL_INTERVAL:
            return
        from notifier import send_via_server, render_email
        sent = 0
        for u in User.query.all():
            rec = getattr(u, 'notify_email', None) or u.email
            if not rec:
                continue
            try:
                subject, body = render_email('resource', message=message, username=u.username)
                if send_via_server(rec, subject, body):
                    sent += 1
            except Exception:
                continue
        if sent:
            set_setting('resource_email_at', str(int(time.time())))
    except Exception:
        pass


def _cleanup_daemon():
    with _app.app_context():
        while True:
            try:
                time.sleep(3600)
                now = datetime.utcnow()
                expired = Task.query.filter(
                    Task.status == 'done',
                    Task.result_expires_at.isnot(None),
                    Task.result_expires_at < now,
                ).all()
                upload = _app.config['UPLOAD_FOLDER']
                for t in expired:
                    if t.result_filename:
                        p = os.path.join(upload, 'results', t.result_filename)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        t.result_filename = None
                    if t.audio_filename:
                        a = os.path.join(upload, 'audio', t.audio_filename)
                        if os.path.exists(a):
                            try:
                                os.remove(a)
                            except OSError:
                                pass
                    t.result_expires_at = None
                cutoff = now - timedelta(days=1)
                old = Task.query.filter(
                    Task.status.in_(['done', 'failed', 'stopped']),
                    Task.done_at.isnot(None),
                    Task.done_at < cutoff,
                    Task.audio_filename.isnot(None),
                ).all()
                for t in old:
                    a = os.path.join(upload, 'audio', t.audio_filename)
                    if os.path.exists(a):
                        try:
                            os.remove(a)
                        except OSError:
                            pass
                db.session.commit()
            except Exception:
                db.session.rollback()
                time.sleep(60)


def stop_inference_daemon():
    global inference_q, inference_daemon_proc
    try:
        if inference_q is not None:
            inference_q.put(None)
        if inference_daemon_proc is not None:
            inference_daemon_proc.join(timeout=3)
    except Exception:
        pass
