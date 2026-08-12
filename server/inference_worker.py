"""推理子进程——跑完就退出，内存全部释放"""
import gc
import json
import os
import sys
import uuid

import soundfile
import torch
import torchaudio

# PyTorch 2.8 compat
_load = torch.load


def _safe_load(*a, **kw):
    """优先以 weights_only=True 加载，失败时才回退（兼容旧模型），降低 pickle 注入风险。"""
    kw.pop('weights_only', None)
    first = a[0] if a else None

    def _rewind():
        # fairseq 等传入文件对象时，失败重试前必须把指针退回开头
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

# torchaudio 2.x 用 soundfile 替代（避免缺 torchcodec）
def _patched_load(path, *a, **kw):
    data, sr = soundfile.read(path, dtype='float32', always_2d=True)
    return torch.from_numpy(data.T).contiguous(), sr
torchaudio.load = _patched_load

# 让 tqdm 每次刷新都输出换行，父进程才能按行读取到真实的扩散采样进度
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

# 添加项目路径
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)

def main():
    if len(sys.argv) < 8:
        print("Usage: inference_worker.py <model_path> <config_path> <audio_path> <output_path> <k_step> <params_json> <diff_path> <diff_config_path> <cluster_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    config_path = sys.argv[2]
    audio_path = sys.argv[3]
    output_path = sys.argv[4]
    k_step = int(sys.argv[5])
    params = json.loads(sys.argv[6])
    diff_path = sys.argv[7] if sys.argv[7] != 'none' else None
    diff_config_path = sys.argv[8] if sys.argv[8] != 'none' else None
    cluster_path = sys.argv[9] if sys.argv[9] != 'none' else None

    # 长段推理内存会爆，切成小块再推理；与 daemon 路径共用 INFERENCE_CLIP_SECONDS
    clip_seconds = float(os.environ.get('INFERENCE_CLIP_SECONDS', '15') or 15)

    # 选择推理设备（优先使用用户设置，其次配置参数）
    device_pref = os.environ.get('SSVC_DEVICE', params.get('device', 'auto'))
    if device_pref == 'cpu':
        device = 'cpu'
        torch.set_num_threads(1)
    elif device_pref == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:  # auto
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(2)

    # 切到项目目录加载模型
    old_cwd = os.getcwd()
    os.chdir(PROJECT_DIR)

    from inference.infer_tool import Svc

    svc = Svc(
        net_g_path=model_path,
        config_path=config_path,
        device=device,
        cluster_model_path=cluster_path or '',
        nsf_hifigan_enhance=False,
        diffusion_model_path=diff_path or '',
        # 扩散配置必须是 diffusion.yaml；绝不能回退用主模型的 config.json
        diffusion_config_path=diff_config_path or '',
        shallow_diffusion=k_step > 0 and diff_path is not None,
        only_diffusion=False,
        spk_mix_enable=False,
        # cluster_ratio > 0 且挂了检索索引时启用特征检索（faiss）
        feature_retrieval=bool(params.get('cluster_ratio', 0) > 0),
    )

    os.chdir(old_cwd)

    # 获取说话人
    spk_id = list(svc.spk2id.keys())[0] if svc.spk2id else None
    if not spk_id:
        raise ValueError('没有可用的说话人')

    # 推理
    audio = svc.slice_inference(
        raw_audio_path=audio_path,
        spk=spk_id,
        tran=params.get('vc_transform', 0),
        slice_db=params.get('slice_db', -40),
        cluster_infer_ratio=params.get('cluster_ratio', 0),
        auto_predict_f0=params.get('auto_f0', False),
        noice_scale=params.get('noise_scale', 0.4),
        pad_seconds=params.get('pad_seconds', 0.5),
        clip_seconds=clip_seconds,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor=params.get('f0_predictor', 'pm'),
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=k_step,
        use_spk_mix=False,
        second_encoding=params.get('second_encoding', False),
        loudness_envelope_adjustment=params.get('loudness_envelope', 1),
        hybrid_mode=params.get('hybrid_mode', 'auto'),
    )
    svc.clear_empty()

    # 保存
    soundfile.write(output_path, audio, svc.target_sample, format=params.get('output_format', 'wav'))

    # 显式释放
    del svc
    gc.collect()


if __name__ == '__main__':
    main()
