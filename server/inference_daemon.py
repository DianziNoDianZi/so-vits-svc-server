"""推理常驻进程：LRU 缓存最近 N 个模型，命中后零加载开销。

与 inference_worker.py（一次性子进程）互补：worker 每次推理后退出释放内存，
daemon 常驻并缓存模型，适合连续推理同模型/同配置的场景。

通信：
  q      -> (task_id, payload_dict)  任务队列
  done_q -> (task_id, success, error_msg_or_None)  完成通知
"""
import gc
import json
import os
import sys
import threading
import traceback
from collections import OrderedDict
from multiprocessing.connection import Listener

import soundfile
import torch
import torchaudio

# PyTorch 2.8 compat（与 inference_worker.py 相同）
_load = torch.load


def _safe_load(*a, **kw):
    kw.pop('weights_only', None)
    first = a[0] if a else None

    def _rewind():
        if first is not None and hasattr(first, 'seek') and hasattr(first, 'tell'):
            try:
                first.seek(0)
            except Exception:
                pass

    try:
        return _load(*a, **kw, weights_only=True)
    except TypeError:
        _rewind()
        return _load(*a, **kw)
    except Exception:
        _rewind()
        try:
            return _load(*a, **kw, weights_only=False)
        except TypeError:
            _rewind()
            return _load(*a, **kw)


torch.load = _safe_load


def _patched_load(path, *a, **kw):
    data, sr = soundfile.read(path, dtype='float32', always_2d=True)
    return torch.from_numpy(data.T).contiguous(), sr


torchaudio.load = _patched_load


import tqdm as _tqdm_mod


class _NewlineTqdm(_tqdm_mod.tqdm):
    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)
        try:
            self.fp.write('\n')
            self.fp.flush()
        except Exception:
            pass


_tqdm_mod.tqdm = _NewlineTqdm

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SERVER_DIR)


class OnnxGenerator:
    """用 onnxruntime 跑主生成器，接口与 PyTorch net_g.infer 对齐。
    采样噪声由外部生成传入（导出时已把 randn 移出图）。"""

    def __init__(self, onnx_path, device):
        import onnxruntime as ort
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.device = device
        self.inter_channels = 192
        self.input_names = [i.name for i in self.sess.get_inputs()]
        for inp in self.sess.get_inputs():
            if inp.name == 'noise' and len(inp.shape) >= 2 and inp.shape[1]:
                self.inter_channels = int(inp.shape[1])

    def infer(self, c, f0, uv, g=None, predict_f0=False, noice_scale=0.35, seed=52468, vol=None):
        import numpy as np
        B, _D, T = c.shape
        torch.manual_seed(seed)
        noise = torch.randn(B, self.inter_channels, T)
        if g is None:
            g_np = np.zeros((B, 1), dtype=np.int64)
        else:
            g_np = np.asarray(g.detach().cpu().reshape(B, 1), dtype=np.int64)
        vol_np = vol.detach().cpu().float().numpy() if vol is not None else np.zeros((B, T), dtype=np.float32)
        feed = {}
        for name in self.input_names:
            if name == 'c':
                feed[name] = c.detach().cpu().float().numpy()
            elif name == 'f0':
                feed[name] = f0.detach().cpu().float().numpy()
            elif name == 'uv':
                feed[name] = uv.detach().cpu().float().numpy()
            elif name == 'g':
                feed[name] = g_np
            elif name == 'noise':
                feed[name] = noise.numpy()
            elif name == 'vol':
                feed[name] = vol_np
        inputs = feed
        out = self.sess.run(['audio'], inputs)[0]
        return torch.from_numpy(out), f0


def _model_key(p):
    """模型缓存 key：结构与 Svc 构造相关的最小参数集合。"""
    return (
        p.get('model_path', ''),
        p.get('config_path', ''),
        p.get('diff_path', ''),
        p.get('diff_config_path', ''),
        p.get('cluster_path', ''),
        bool(p.get('k_step', 0) > 0 and p.get('diff_path')),
        p.get('device_resolved', p.get('device', 'auto')),
    )


class ModelCache:
    def __init__(self, maxsize=3):
        self.maxsize = max(1, int(maxsize))
        self.data = OrderedDict()

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def put(self, key, svc):
        if key in self.data:
            self.data.move_to_end(key)
            self.data[key] = svc
            return
        self.data[key] = svc
        while len(self.data) > self.maxsize:
            _k, old = self.data.popitem(last=False)
            try:
                del old
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_svc(p, device):
    from inference.infer_tool import Svc
    _params = p.get('params') or {}
    svc = Svc(
        net_g_path=p['model_path'],
        config_path=p['config_path'],
        device=device,
        cluster_model_path=p.get('cluster_path') or '',
        nsf_hifigan_enhance=False,
        diffusion_model_path=p.get('diff_path') or '',
        diffusion_config_path=p.get('diff_config_path') or '',
        shallow_diffusion=bool(p.get('k_step', 0) > 0 and p.get('diff_path')),
        only_diffusion=False,
        spk_mix_enable=False,
        feature_retrieval=bool(_params.get('cluster_ratio', 0) > 0 and p.get('cluster_path')),
    )
    # ONNX 生成器：模型旁存在 .onnx 时用 onnxruntime 推理，失败回退 PyTorch
    onnx_path = p['model_path'] + '.onnx'
    if os.path.exists(onnx_path):
        try:
            svc.net_g_ms = OnnxGenerator(onnx_path, device)
            print(f'[daemon] 使用 ONNX 生成器: {os.path.basename(onnx_path)}', flush=True)
        except Exception as e:
            print(f'[daemon] ONNX 加载失败，回退 PyTorch: {e}', flush=True)
    return svc


def _run_inference(p, svc, device):
    import contextlib
    import soundfile as sf
    params = p.get('params') or {}
    spk_id = list(svc.spk2id.keys())[0] if svc.spk2id else None
    if not spk_id:
        raise ValueError('没有可用的说话人')

    class _Tee:
        def __init__(self, file):
            self.file = file

        def write(self, s):
            self.file.write(s)
            self.file.flush()
            try:
                sys.__stdout__.write(s)
                sys.__stdout__.flush()
            except Exception:
                pass

        def flush(self):
            try:
                self.file.flush()
            except Exception:
                pass

    prog_path = p['result_path'] + '.prog'
    with open(prog_path, 'w', encoding='utf-8') as pf, \
            contextlib.redirect_stdout(_Tee(pf)):
        audio = svc.slice_inference(
            raw_audio_path=p['audio_path'],
            spk=spk_id,
            tran=params.get('vc_transform', 0),
            slice_db=params.get('slice_db', -40),
            cluster_infer_ratio=params.get('cluster_ratio', 0),
            auto_predict_f0=params.get('auto_f0', False),
            noice_scale=params.get('noise_scale', 0.4),
            pad_seconds=params.get('pad_seconds', 0.5),
            clip_seconds=float(p.get('clip_seconds') or 0),
            lg_num=0,
            lgr_num=0.75,
            f0_predictor=params.get('f0_predictor', 'pm'),
            enhancer_adaptive_key=0,
            cr_threshold=0.05,
            k_step=p.get('k_step', 0),
            use_spk_mix=False,
            second_encoding=params.get('second_encoding', False),
            loudness_envelope_adjustment=params.get('loudness_envelope', 1),
        )
    try:
        os.remove(prog_path)
    except OSError:
        pass
    svc.clear_empty()
    sf.write(p['result_path'], audio, svc.target_sample,
             format=params.get('output_format', 'wav'))
    # 完成标记：结果完整写入后才创建，供 task_worker 双保险检测
    try:
        with open(p['result_path'] + '.done', 'w') as f:
            f.write('ok')
    except Exception:
        pass


def main(q, done_q, cache_size):
    cache = ModelCache(cache_size)
    # 流式变声与文件任务共用的全局推理锁：一次只跑一个推理
    infer_lock = threading.Lock()
    _stream_active = threading.Event()

    def _start_stream_listener():
        try:
            _stream_listener(cache, infer_lock, _stream_active)
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_start_stream_listener, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        task_id, payload = item
        success, err = False, None
        try:
            with infer_lock:
                device_pref = payload.get('device', 'auto')
                cores = int(payload.get('max_cpu_cores') or 0)
                if device_pref == 'cpu':
                    device = 'cpu'
                    torch.set_num_threads(cores if cores > 0 else 2)
                elif device_pref == 'cuda':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cpu':
                    torch.set_num_threads(cores if cores > 0 else 2)

                # 实际设备写回 payload，参与缓存 key（避免 cuda/cpu 混用命中）
                payload = dict(payload)
                payload['device_resolved'] = device
                old_cwd = os.getcwd()
                os.chdir(PROJECT_DIR)
                try:
                    key = _model_key(payload)
                    svc = cache.get(key)
                    if svc is None:
                        svc = _build_svc(payload, device)
                        cache.put(key, svc)
                    _run_inference(payload, svc, device)
                finally:
                    os.chdir(old_cwd)
            success = True
        except KeyboardInterrupt:
            # 用户停止：中断当前推理，模型缓存保留，继续处理下一个任务
            err = '用户停止推理'
            success = False
        except Exception as e:
            import traceback
            err = traceback.format_exc()
        finally:
            try:
                done_q.put((task_id, success, err))
            except Exception:
                pass


def _stream_listener(cache, infer_lock, stream_active):
    """AF_UNIX socket 端点：接收 ws_server 的流式变声会话。

    协议（multiprocessing.connection，消息自带分帧）：
      -> {op:'init', payload:{...model/流式参数...}}
      -> {op:'infer', audio: <bytes float32 44.1k>}
      -> {op:'close'}
      <- {ok:bool, error?} 或 {ok:bool, audio: <bytes float32 44.1k>}
    """
    from services.streaming import DaemonStreamingVC
    sock_path = os.path.join(SERVER_DIR, 'logs', 'ssvc_stream.sock')
    try:
        os.makedirs(os.path.dirname(sock_path), exist_ok=True)
        if os.path.exists(sock_path):
            os.remove(sock_path)
        listener = Listener(sock_path, family='AF_UNIX')
    except Exception:
        traceback.print_exc()
        return

    while True:
        try:
            conn = listener.accept()
        except Exception:
            continue
        # 每次连接独立线程处理，但全局只允许一个活动流式会话
        if stream_active.is_set():
            try:
                conn.send({'ok': False, 'error': 'busy'})
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            continue
        threading.Thread(target=_handle_stream_conn,
                         args=(conn, cache, infer_lock, stream_active),
                         daemon=True).start()


def _handle_stream_conn(conn, cache, infer_lock, stream_active):
    stream_active.set()
    try:
        first = conn.recv()
        if not isinstance(first, dict) or first.get('op') != 'init':
            conn.send({'ok': False, 'error': 'expected init'})
            return
        payload = first.get('payload') or {}
        p = dict(payload)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        p['device_resolved'] = device
        old_cwd = os.getcwd()
        os.chdir(PROJECT_DIR)
        try:
            key = _model_key(p)
            svc = cache.get(key)
            if svc is None:
                svc = _build_svc(p, device)
                cache.put(key, svc)
            speaker = payload.get('speaker') or (list(svc.spk2id.keys())[0] if svc.spk2id else None)
            if not speaker:
                conn.send({'ok': False, 'error': 'no speaker'})
                return
            vc = DaemonStreamingVC(
                svc, speaker,
                tran=int(payload.get('tran') or 0),
                auto_predict_f0=bool(payload.get('auto_predict_f0', False)),
                noice_scale=float(payload.get('noice_scale') or 0.4),
                f0_predictor=payload.get('f0_predictor') or 'pm',
                k_step=int(payload.get('k_step') or 0),
                cluster_infer_ratio=float(payload.get('cluster_ratio') or 0),
                chunk_seconds=float(payload.get('chunk_seconds') or 0.363),
            )
        except Exception:
            traceback.print_exc()
            try:
                conn.send({'ok': False, 'error': 'init failed'})
            except Exception:
                pass
            return
        finally:
            os.chdir(old_cwd)
        conn.send({'ok': True})
        # 推理循环
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(msg, dict):
                break
            op = msg.get('op')
            if op == 'close':
                break
            if op == 'infer':
                audio = msg.get('audio')
                if audio is None:
                    continue
                import numpy as np
                chunk = np.frombuffer(bytes(audio), dtype=np.float32)
                try:
                    with infer_lock:
                        out = vc.feed(chunk)
                    if out is None:
                        # 未攒满一个窗口，暂无输出
                        conn.send({'ok': True, 'pending': True})
                    else:
                        conn.send({'ok': True, 'audio': out.astype('<f4').tobytes()})
                except Exception:
                    traceback.print_exc()
                    try:
                        conn.send({'ok': False, 'error': 'infer failed'})
                    except Exception:
                        pass
                    break
            else:
                break
    except (EOFError, OSError):
        pass
    except Exception:
        traceback.print_exc()
    finally:
        stream_active.clear()
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    # 直接运行调试：inference_daemon.py <cache_size>
    import multiprocessing as mp
    q = mp.Queue()
    dq = mp.Queue()
    p = mp.Process(target=main, args=(q, dq, int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    p.start()
    print('daemon started, pid', p.pid)
    p.join()
