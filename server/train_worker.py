import gc
import os
import signal
import sys
import time
import json
import shutil
import subprocess
import traceback
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_BASE = os.path.join(os.path.dirname(__file__), 'uploads')
PYTHON = sys.executable
QUICK_RESUME_DIR = os.path.join(UPLOAD_BASE, 'train_data', 'quick_resume')


def _find_ffmpeg():
    """查找 ffmpeg 可执行文件（Windows 下不在 PATH 时用 venv 里的）"""
    if os.name == 'nt':
        venv_bases = [os.path.dirname(sys.executable),
                      os.path.dirname(os.path.dirname(sys.executable))]
        for base in venv_bases:
            for cand in [os.path.join(base, 'ffmpeg.exe'),
                         os.path.join(base, 'Scripts', 'ffmpeg.exe'),
                         os.path.join(base, 'bin', 'ffmpeg.exe')]:
                if os.path.exists(cand):
                    return cand
    import shutil as _sh
    p = _sh.which('ffmpeg')
    return p or 'ffmpeg'


FFMPEG = _find_ffmpeg()

AUDIO_EXTS = ('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus')
MAX_ZIP_ENTRIES = 20000
MAX_ZIP_TOTAL = 50 * 1024 * 1024 * 1024  # 解压总大小上限 50GB
MIN_AUDIO_SECONDS = 2.0  # 过滤过短片段（扩散训练要求最短音频 >= duration=2s）

_current_proc = None


def _find_audio_dir(root):
    """递归查找包含音频文件最多的目录（兼容多层嵌套的 zip 结构）。"""
    best, best_count = None, -1
    for dirpath, _dirnames, filenames in os.walk(root):
        count = sum(1 for f in filenames if f.lower().endswith(AUDIO_EXTS))
        if count > best_count:
            best, best_count = dirpath, count
    return best, best_count


def _latest_checkpoint(directory, prefix, suffix):
    """按文件名中的数字取最新 checkpoint（G_1200.pth -> 1200）。"""
    best, best_n = None, -1
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for f in names:
        if f.startswith(prefix) and f.endswith(suffix):
            n = int(''.join(ch for ch in f if ch.isdigit()) or 0)
            if n > best_n:
                best, best_n = f, n
    return best


def build_retrieval_index(speaker_dir, speaker, out_dir):
    """从训练集 ContentVec 特征（*.soft.pt）建 faiss 检索索引。
    保存为 {speaker}_cluster.pth（pickle: {spk_id: faiss.IndexFlatIP}），
    与 infer_tool.py 的 feature retrieval 加载格式一致。无特征时返回 None。
    """
    try:
        import faiss
        import pickle
    except Exception as e:
        log_line = f'[build_retrieval_index] faiss 不可用: {e}'
        print(log_line)
        return None
    import numpy as np
    import torch as _torch

    feats = []
    if not os.path.isdir(speaker_dir):
        return None
    for f in sorted(os.listdir(speaker_dir)):
        if f.endswith('.soft.pt'):
            try:
                c = _torch.load(os.path.join(speaker_dir, f), map_location='cpu')
                c = c.squeeze(0).transpose(0, 1).numpy().astype('float32')
                feats.append(c)
            except Exception:
                continue
    if not feats:
        return None
    all_feats = np.concatenate(feats, axis=0)
    index = faiss.IndexFlatIP(all_feats.shape[1])
    index.add(all_feats)
    out_path = os.path.join(out_dir, f'{speaker}_cluster.pth')
    with open(out_path, 'wb') as f:
        pickle.dump({0: index}, f)
    # 同时 dump 全量特征 npy，推理时 mmap 零拷贝（替代 faiss reconstruct_n）
    try:
        np.save(out_path.replace('.pth', '.npy'), all_feats)
    except Exception as e:
        print(f'[build_retrieval_index] npy dump 失败（可忽略）: {e}')
    return out_path


def write_quick_resume(task_id, speaker, dataset_zip, model_type, batch_size, total_steps,
                       keep_ckpts, speech_encoder, f0_predictor, arch, flow_mode,
                       use_unified_flow, c_fm, c_kl=1.0, c_mel=45,
                       ema_decay=0.999, ema_interval=100, max_speclen=512, seed=1234,
                       n_layers_q=3, hybrid_steps=4, enc_q_hidden=96, d_lr_scale=1.0,
                       vol_aug=False, warmup_epochs=0, fp16_run=None,
                       learning_rate=0.0001, segment_size=10240, lr_decay=0.999875,
                       auto_stop=200, log_interval=200, eval_interval=800,
                       resume_from_id=None, checkpoint_path=None, config_path=None):
    """把最新 checkpoint + 配置打成"快速恢复"快照（只保留最近 1 份，模型少）。
    停止后训练页/任务列表一键继续上次训练。"""
    try:
        os.makedirs(QUICK_RESUME_DIR, exist_ok=True)
        # 清掉旧快照（只留当前这一份）
        for f in os.listdir(QUICK_RESUME_DIR):
            try:
                os.remove(os.path.join(QUICK_RESUME_DIR, f))
            except OSError:
                pass
        shutil.copy2(checkpoint_path, os.path.join(QUICK_RESUME_DIR, 'G_latest.pth'))
        shutil.copy2(config_path, os.path.join(QUICK_RESUME_DIR, 'config.json'))
        meta = {
            'task_id': task_id,
            'speaker': speaker,
            'dataset_zip': dataset_zip,
            'model_type': model_type,
            'batch_size': batch_size,
            'total_steps': total_steps,
            'keep_ckpts': keep_ckpts,
            'speech_encoder': speech_encoder,
            'f0_predictor': f0_predictor,
            'arch': arch,
            'flow_mode': flow_mode,
            'use_unified_flow': use_unified_flow,
            'c_fm': c_fm,
            'c_kl': c_kl,
            'c_mel': c_mel,
            'ema_decay': ema_decay,
            'ema_interval': ema_interval,
            'max_speclen': max_speclen,
            'seed': seed,
            'n_layers_q': n_layers_q,
            'hybrid_steps': hybrid_steps,
            'enc_q_hidden': enc_q_hidden,
            'd_lr_scale': d_lr_scale,
            'vol_aug': vol_aug,
            'warmup_epochs': warmup_epochs,
            'fp16_run': fp16_run,
            'learning_rate': learning_rate,
            'segment_size': segment_size,
            'lr_decay': lr_decay,
            'auto_stop': auto_stop,
            'log_interval': log_interval,
            'eval_interval': eval_interval,
            'resume_from_id': resume_from_id or task_id,
            'checkpoint_step': int(''.join(c for c in os.path.basename(checkpoint_path) if c.isdigit()) or 0),
        }
        with open(os.path.join(QUICK_RESUME_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f'[write_quick_resume] 快照写入失败: {e}')
        return False


def _apply_encoder_dims(cfg, speech_encoder):
    """按编码器调整模型维度，与 preprocess_flist_config 保持一致。"""
    cfg.setdefault('model', {})['speech_encoder'] = speech_encoder
    m = cfg['model']
    if speech_encoder in ('vec768l12', 'dphubert', 'wavlmbase+'):
        m['ssl_dim'] = m['filter_channels'] = m['gin_channels'] = 768
    elif speech_encoder in ('vec256l9', 'hubertsoft'):
        m['ssl_dim'] = m['gin_channels'] = 256
    elif speech_encoder in ('whisper-ppg', 'cnhubertlarge'):
        m['ssl_dim'] = m['filter_channels'] = m['gin_channels'] = 1024
    elif speech_encoder == 'whisper-ppg-large':
        m['ssl_dim'] = m['filter_channels'] = m['gin_channels'] = 1280
    return cfg


def stop():
    global _current_proc
    if _current_proc and _current_proc.poll() is None:
        # 杀整个会话组，连带 preprocess 等 fork 出的孙进程一起清干净
        try:
            os.killpg(os.getpgid(_current_proc.pid), signal.SIGKILL)
        except Exception:
            _current_proc.kill()
        try:
            _current_proc.wait(timeout=10)
        except Exception:
            pass
        return True
    return False


def run(task_id, speaker, dataset_zip, log_path='', model_type='sovits', batch_size=4, total_steps=4200, keep_ckpts=3,
        speech_encoder='vec768l12', f0_predictor='dio', learning_rate=0.0001, segment_size=10240,
        lr_decay=0.999875, auto_stop=200, log_interval=200, eval_interval=800,
        arch='sovits-v1', d_lr_scale=1.0, flow_mode='a2',
        use_unified_flow=False, c_fm=0.3,
        c_mel=45, c_kl=1.0, ema_decay=0.999, ema_interval=100,
        max_speclen=512, fp16_run=None, vol_aug=False, warmup_epochs=0, seed=1234,
        n_layers_q=3, hybrid_steps=4, enc_q_hidden=96,
        diff_batch_size=48, diff_epochs=100000, diff_timesteps=1000, diff_kstep=0,
        diff_layers=20, diff_chans=512, diff_hidden=256, diff_lr=0.0001,
        diff_decay_step=100000, diff_gamma=0.5, diff_amp='fp32',
        diff_interval_val=200,
        diff_max_steps=0,
        resume_from_id=0, diff_root_id=0):
    log_lines = []
    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        log_lines.append(line)
        print(line)
        if log_path:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
            except Exception:
                pass

    is_resume = bool(resume_from_id)
    if is_resume:
        data_dir = os.path.join(UPLOAD_BASE, 'train_data', f'task_{resume_from_id}')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'续训源任务目录不存在: {data_dir}')
        log(f'续训模式: 复用任务 {resume_from_id} 的数据目录')
    else:
        data_dir = os.path.join(UPLOAD_BASE, 'train_data', f'task_{task_id}')
    os.makedirs(data_dir, exist_ok=True)

    def _exec(cmd, timeout=None):
        global _current_proc
        log(f'运行: {cmd[0]}')
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            has_gpu = False
        if not has_gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        p = subprocess.Popen(
            [PYTHON, '-X', 'utf8'] + cmd,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding='utf-8', errors='replace',
            env=env,
            start_new_session=True,  # 独立会话组，停止时可连孙进程一起杀干净
        )
        _current_proc = p
        # 按 \r 和 \n 切分输出，tqdm 的进度条（\r 刷新）也能实时进日志
        buf = ''
        while True:
            try:
                raw = p.stdout.buffer.read1(65536)
            except Exception:
                break
            if not raw:
                break
            buf += raw.decode('utf-8', errors='replace')
            while True:
                idx = -1
                for sep in ('\r', '\n'):
                    i = buf.find(sep)
                    if i != -1 and (idx == -1 or i < idx):
                        idx = i
                if idx == -1:
                    break
                line = buf[:idx].strip()
                buf = buf[idx + 1:]
                if line:
                    log(line)
        if buf.strip():
            log(buf.strip())
        p.wait(timeout=timeout)
        if p.returncode != 0:
            last = log_lines[-3:] if len(log_lines) >= 3 else log_lines
            detail = ' | '.join(l.strip() for l in last)
            raise RuntimeError(f'{cmd[0]} 退出码 {p.returncode}: {detail}')
        return p.returncode

    try:
        log(f'开始训练: {speaker} ({model_type})')
        # A1（特征先验流）无 enc_q 提供 FM 目标，unified_flow 强制关闭（与 models.py 保护一致），
        # 避免 config.json / 快照 meta 中残留 use_unified_flow=True 污染后续续训
        if flow_mode == 'a1':
            use_unified_flow = False
        model_name = ''
        diff_name = ''
        saved_model = ''
        saved_config = ''
        saved_cluster = ''
        saved_diff_model = ''
        saved_diff_config = ''
        need_diff = model_type in ('sovits_diff', 'diffusion')
        dataset_dir = os.path.join(PROJECT_DIR, 'dataset', '44k')
        sr = 44100
        try:
            import torch as _torch
            _has_gpu = _torch.cuda.is_available()
        except Exception:
            _has_gpu = False
        # GPU 用单进程避免显存爆炸；CPU 并行加速特征提取
        feature_procs = 1 if _has_gpu else min(4, os.cpu_count() or 1)

        def resample_into(raw_dir, speaker_dir):
            """把 raw_dir 里的音频统一重采样到 44.1kHz 单声道，跳过过短片段。"""
            os.makedirs(speaker_dir, exist_ok=True)
            count = 0
            skipped_short = 0
            skipped_err = 0
            for f in sorted(os.listdir(raw_dir)):
                if not f.lower().endswith(AUDIO_EXTS):
                    continue
                src = os.path.join(raw_dir, f)
                try:
                    import soundfile as _sf
                    dur = _sf.info(src).duration
                except Exception:
                    dur = None
                if dur is not None and dur < MIN_AUDIO_SECONDS:
                    skipped_short += 1
                    continue
                count += 1
                dst = os.path.join(speaker_dir, Path(f).stem + '.wav')
                subprocess.run([FFMPEG, '-y', '-i', src,
                                '-ar', str(sr), '-ac', '1', dst],
                               capture_output=True, encoding='utf-8', errors='replace')
            log(f'重采样: 保留 {count} 个, 跳过 <{MIN_AUDIO_SECONDS}s 片段 {skipped_short} 个')
            return count

        if is_resume:
            # ===== 续训：优先复用旧数据；数据被清理时从旧任务 raw 重建 =====
            speaker_dir = os.path.join(dataset_dir, speaker)
            if not os.path.isdir(speaker_dir) or not any(
                    f.lower().endswith(AUDIO_EXTS) for f in os.listdir(speaker_dir)):
                src_raw = os.path.join(UPLOAD_BASE, 'train_data', f'task_{resume_from_id}', 'raw')
                raw_dir, raw_count = _find_audio_dir(src_raw) if os.path.isdir(src_raw) else (None, 0)
                if not raw_dir or raw_count == 0:
                    raise FileNotFoundError(f'续训失败: 旧任务没有 {speaker} 的音频数据')
                os.makedirs(dataset_dir, exist_ok=True)
                if os.path.isdir(speaker_dir):
                    shutil.rmtree(speaker_dir, ignore_errors=True)
                af_count = resample_into(raw_dir, speaker_dir)
                log(f'续训: 从旧任务重建数据，重采样 {af_count} 个文件')
            else:
                log(f'续训: 复用数据 {speaker_dir}')

            # 续训尽可能复用：数据/特征/配置齐全时跳过预处理，直接进入训练
            for d in [os.path.join(PROJECT_DIR, 'configs'), os.path.join(PROJECT_DIR, 'filelists')]:
                os.makedirs(d, exist_ok=True)

            wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')] if os.path.isdir(speaker_dir) else []
            soft_count = sum(1 for f in wav_files if os.path.exists(os.path.join(speaker_dir, f + '.soft.pt')))
            f0_count = sum(1 for f in wav_files if os.path.exists(os.path.join(speaker_dir, f + '.f0.npy')))
            spec_count = sum(1 for f in wav_files if os.path.exists(os.path.join(speaker_dir, f.replace('.wav', '.spec.pt'))))
            feat_ok = bool(wav_files) and soft_count >= len(wav_files) and f0_count >= len(wav_files) and spec_count >= len(wav_files)
            flist_ok = os.path.exists(os.path.join(PROJECT_DIR, 'filelists', 'train.txt')) and os.path.exists(
                os.path.join(PROJECT_DIR, 'filelists', 'val.txt'))

            if not flist_ok:
                _exec(['preprocess_flist_config.py', '--source_dir', dataset_dir, '--speech_encoder', speech_encoder])
            else:
                log(f'续训: filelists 已存在，跳过生成配置')
            config_path = os.path.join(data_dir, 'config.json')
            # 续训以「任务目录独立 config.json」为权威（train.py 保存的副本），
            # 避免所有任务共享全局 configs/config.json 导致架构参数串扰
            saved_cfg = config_path
            if os.path.exists(saved_cfg):
                with open(saved_cfg, 'r', encoding='utf-8') as f:
                    old_cfg = json.load(f)
                if old_cfg.get('model', {}).get('speech_encoder') != speech_encoder:
                    old_cfg = _apply_encoder_dims(old_cfg, speech_encoder)
                    log(f'续训: 编码器更新为 {speech_encoder}')
                # 说话人变化时同步 spk 映射，避免与旧配置错位
                if list((old_cfg.get('spk') or {}).keys()) != [speaker]:
                    old_cfg['spk'] = {speaker: 0}
                    old_cfg['model']['n_speakers'] = 1
                old_cfg['model']['arch'] = arch
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(old_cfg, f, indent=2, ensure_ascii=False)
                log(f'续训: 复用任务独立配置 {saved_cfg}')
            else:
                # 任务目录无配置副本（极端情况），退回全局配置
                config_path = os.path.join(PROJECT_DIR, 'configs', 'config.json')
            if feat_ok:
                log(f'续训: 特征齐全（soft {soft_count}/{len(wav_files)}，f0 {f0_count}/{len(wav_files)}，spec {spec_count}/{len(wav_files)}），跳过特征提取')
            else:
                log(f'续训: 特征不完整（soft {soft_count}/{len(wav_files)}，f0 {f0_count}/{len(wav_files)}，spec {spec_count}/{len(wav_files)}），重新提取')
                diff_flag = ['--use_diff'] if need_diff else []
                _exec(['preprocess_hubert_f0.py', '--in_dir', dataset_dir, '--f0_predictor', f0_predictor,
                       '--num_processes', str(feature_procs)] + diff_flag)
        else:
            # ===== 全新训练：解压 → 重采样 → 配置 → 特征 =====
            zip_path = os.path.join(UPLOAD_BASE, 'dataset_zips', dataset_zip)
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f'数据集不存在: {zip_path}')
            log(f'解压数据集...')
            extract_dir = os.path.join(data_dir, 'raw')
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ZIP_ENTRIES:
                    raise RuntimeError(f'数据集文件过多({len(infos)} 个)，超过上限 {MAX_ZIP_ENTRIES}')
                if sum(i.file_size for i in infos) > MAX_ZIP_TOTAL:
                    raise RuntimeError('数据集解压后总大小超过 50GB，已拒绝')
                zf.extractall(extract_dir)

            raw_dir, raw_count = _find_audio_dir(extract_dir)
            if not raw_dir or raw_count == 0:
                raise RuntimeError('数据集 zip 中没有找到音频文件（支持 mp3/wav/flac/m4a/ogg/opus）')

            os.makedirs(dataset_dir, exist_ok=True)
            # 清空所有旧说话人数据（含当前 speaker，避免旧 wav/特征残留导致特征错配）
            for d in os.listdir(dataset_dir):
                dp = os.path.join(dataset_dir, d)
                if os.path.isdir(dp):
                    shutil.rmtree(dp, ignore_errors=True)
            speaker_dir = os.path.join(dataset_dir, speaker)
            af_count = resample_into(raw_dir, speaker_dir)
            if af_count == 0:
                raise RuntimeError(f'重采样后没有可用音频（全部短于 {MIN_AUDIO_SECONDS}s 被过滤），训练已中止')
            if af_count < 3:
                raise RuntimeError(f'过滤后音频过少({af_count} 个)，至少需要 3 个，训练已中止')

            log(f'生成配置...')
            for d in [os.path.join(PROJECT_DIR, 'configs'), os.path.join(PROJECT_DIR, 'filelists')]:
                os.makedirs(d, exist_ok=True)
            _exec(['preprocess_flist_config.py', '--source_dir', dataset_dir, '--speech_encoder', speech_encoder])
            config_path = os.path.join(PROJECT_DIR, 'configs', 'config.json')
            log(f'配置: {config_path}')

            log(f'提取特征...')
            diff_flag = ['--use_diff'] if need_diff else []
            _exec(['preprocess_hubert_f0.py', '--in_dir', dataset_dir, '--f0_predictor', f0_predictor,
                   '--num_processes', str(feature_procs)] + diff_flag)

        if model_type in ('sovits', 'sovits_diff'):
            log(f'写入训练参数到 config...')
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            cfg['train']['batch_size'] = min(batch_size, 8)
            cfg['train']['learning_rate'] = learning_rate
            cfg['train']['lr_decay'] = lr_decay
            cfg['train']['segment_size'] = segment_size
            cfg['train']['keep_ckpts'] = keep_ckpts
            cfg['train']['log_interval'] = log_interval
            cfg['train']['eval_interval'] = eval_interval
            cfg['train']['all_in_mem'] = False
            try:
                import psutil as _psutil
                _ram_gb = _psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                _ram_gb = 0
            # 内存充足用 4 进程，否则退到 2；极低内存才关多进程（避免 Windows 1455）
            cfg['train']['num_workers'] = 4 if _ram_gb >= 16 else (2 if _ram_gb >= 8 else 0)
            cfg['train']['max_steps'] = total_steps
            cfg['train']['auto_stop'] = auto_stop
            cfg['train']['seed'] = int(seed)
            # Loss 权重
            cfg['train']['c_mel'] = float(c_mel)
            cfg['train']['c_kl'] = float(c_kl)
            # EMA
            cfg['train']['ema_decay'] = float(ema_decay)
            cfg['train']['ema_interval'] = int(ema_interval)
            # 频谱与数据增强
            cfg['train']['max_speclen'] = int(max_speclen)
            cfg['train']['vol_aug'] = bool(vol_aug)
            cfg['train']['warmup_epochs'] = int(warmup_epochs)
            cfg['model']['arch'] = arch
            cfg['model']['n_layers_q'] = int(n_layers_q)
            if arch == 'rvc-flow':
                cfg['model']['flow_mode'] = flow_mode
                cfg['model']['n_flow_layer'] = 2
                cfg['model']['n_layers_trans_flow'] = 2
                cfg['model']['enc_q_hidden'] = int(enc_q_hidden)
                cfg['model']['hybrid_steps'] = int(hybrid_steps)
                # 统一流：同一骨干承载 NF+FM，开启后推理可选 nf/hybrid/fm
                cfg['model']['use_unified_flow'] = bool(use_unified_flow)
                if use_unified_flow:
                    cfg['train']['c_fm'] = float(c_fm)
            cfg['train']['d_lr_scale'] = d_lr_scale
            try:
                import torch as _torch
                _has_gpu = _torch.cuda.is_available()
                _gpu_mem = _torch.cuda.get_device_properties(0).total_memory if _has_gpu else 0
            except Exception:
                _has_gpu, _gpu_mem = False, 0
            # fp16_run：用户显式传 True/False 则尊重，None 则按 GPU 自动判断
            if fp16_run is None:
                cfg['train']['fp16_run'] = bool(_has_gpu and _gpu_mem >= 6 * 1024 ** 3)
            else:
                cfg['train']['fp16_run'] = bool(fp16_run)
            log(f'fp16_run={cfg["train"]["fp16_run"]}, max_steps={total_steps}, auto_stop={auto_stop}')

            # 使用预训练底模（仅当任务目录中没有已有 checkpoint 时）
            if not _latest_checkpoint(data_dir, 'G_', '.pth'):
                for base in ('G_0.pth', 'D_0.pth'):
                    src = os.path.join(PROJECT_DIR, 'pretrain', base)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(data_dir, base))
                        log(f'使用预训练底模: {base}')
            else:
                log('检测到已有 checkpoint，跳过底模')

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            log(f'开始 SoVITS 训练...')
            sovits_error = None
            try:
                _exec(['train.py', '-c', config_path, '-m', data_dir])
            except Exception as e:
                sovits_error = e
                log(f'SoVITS 训练异常退出，尝试保存已有 checkpoint: {e}')

            log(f'保存 SoVITS 模型...')
            # 复制最近 keep_ckpts 个 checkpoint 到模型目录（不只最新一个）
            ckpt_names = []
            if os.path.isdir(data_dir):
                for f in os.listdir(data_dir):
                    if f.startswith('G_') and f.endswith('.pth') and f != 'G_0.pth':
                        try:
                            n = int(''.join(c for c in f if c.isdigit()) or 0)
                        except Exception:
                            n = 0
                        if n > 0:
                            ckpt_names.append((n, f))
            ckpt_names.sort()
            ckpt_names = [f for _n, f in ckpt_names][-max(int(keep_ckpts), 1):]
            model_name = ckpt_names[-1] if ckpt_names else ''
            if not model_name:
                log('⚠ 未产生训练 checkpoint（只有底模 G_0.pth 或为空）')
            for ck in ckpt_names:
                try:
                    shutil.copy2(os.path.join(data_dir, ck), os.path.join(UPLOAD_BASE, 'models', ck))
                except OSError as e:
                    log(f'checkpoint {ck} 复制失败: {e}')
                # 每个 checkpoint 配一份独立配置（内容相同，文件名对应，方便单独下载使用）
                try:
                    ck_cfg = ck.replace('.pth', '.json')
                    shutil.copy2(config_path, os.path.join(UPLOAD_BASE, 'configs', f'config_{ck_cfg}'))
                except OSError as e:
                    log(f'checkpoint {ck} 配置复制失败: {e}')
            if model_name:
                cfg_name = model_name.replace('.pth', '.json')
                saved_model = model_name
                saved_config = f'config_{cfg_name}'
                log(f'SoVITS 模型已保存: {model_name}（共 {len(ckpt_names)} 个 checkpoint + 配置复制到模型目录）')
                # 自动建特征检索索引（失败不阻塞训练结果）
                try:
                    spk_dir = os.path.join(dataset_dir, speaker)
                    cluster_path = build_retrieval_index(
                        spk_dir, speaker, os.path.join(UPLOAD_BASE, 'models'))
                    if cluster_path:
                        saved_cluster = os.path.basename(cluster_path)
                        log(f'特征检索索引已生成: {saved_cluster}')
                    else:
                        log('特征检索索引生成跳过（无特征文件）')
                except Exception as e:
                    log(f'特征检索索引生成失败: {e}')
                    saved_cluster = ''
                # 快速恢复快照（最新 checkpoint + 配置，只保留 1 份）
                try:
                    if write_quick_resume(
                            task_id, speaker, dataset_zip, model_type, batch_size, total_steps,
                            keep_ckpts, speech_encoder, f0_predictor, arch, flow_mode,
                            use_unified_flow, c_fm, c_kl, c_mel,
                            ema_decay, ema_interval, max_speclen, seed,
                            n_layers_q, hybrid_steps, enc_q_hidden, d_lr_scale,
                            vol_aug, warmup_epochs, fp16_run,
                            learning_rate, segment_size, lr_decay, auto_stop,
                            log_interval, eval_interval,
                            resume_from_id, os.path.join(data_dir, model_name), config_path):
                        log(f'快速恢复快照已更新（{os.path.basename(model_name)}）')
                except Exception as e:
                    log(f'快速恢复快照写入失败: {e}')
            else:
                raise FileNotFoundError('训练未产生有效 checkpoint（G_*.pth）')
            if sovits_error:
                raise RuntimeError('SoVITS 训练失败（未产生有效 checkpoint）')

        if need_diff:
            diff_config = os.path.join(PROJECT_DIR, 'configs', 'diffusion.yaml')
            if not os.path.exists(diff_config):
                raise FileNotFoundError(f'扩散配置不存在: {diff_config}')

            log(f'配置扩散参数...')
            import yaml
            with open(diff_config, 'r', encoding='utf-8') as f:
                dc = yaml.safe_load(f)

            try:
                import torch as _torch
                _has_gpu = _torch.cuda.is_available()
            except Exception:
                _has_gpu = False
            dc['device'] = 'cuda' if _has_gpu else 'cpu'
            dc['env']['gpu_id'] = 0
            # 扩散 checkpoint 按"续训链"归属：整条链共享同一 expdir，可续训；
            # 不同链（不同说话人/数据集）之间仍然隔离
            diff_root = diff_root_id or task_id
            dc['env']['expdir'] = os.path.join('logs', '44k', 'diffusion', f'task_{diff_root}')
            dc['model']['n_layers'] = diff_layers
            dc['model']['n_chans'] = diff_chans
            dc['model']['n_hidden'] = diff_hidden
            dc['model']['timesteps'] = diff_timesteps
            dc['model']['k_step_max'] = diff_kstep
            dc['train']['batch_size'] = diff_batch_size
            dc['train']['epochs'] = diff_epochs
            dc['train']['lr'] = diff_lr
            dc['train']['decay_step'] = diff_decay_step
            dc['train']['gamma'] = diff_gamma
            dc['train']['amp_dtype'] = diff_amp
            dc['train']['interval_val'] = max(int(diff_interval_val), 1)
            dc['train']['max_steps'] = max(int(diff_max_steps), 0)
            dc['train']['cache_all_data'] = False
            dc['train']['num_workers'] = 0
            with open(diff_config, 'w', encoding='utf-8') as f:
                yaml.dump(dc, f, allow_unicode=True)

            log(f'开始扩散训练...')
            diff_error = None
            try:
                _exec(['train_diff.py', '-c', diff_config])
            except Exception as e:
                diff_error = e
                log(f'扩散训练异常退出，尝试保存已有 checkpoint: {e}')

            log(f'保存扩散模型...')
            diff_dir = os.path.join(PROJECT_DIR, 'logs', '44k', 'diffusion', f'task_{diff_root}')
            diff_name = _latest_checkpoint(diff_dir, 'model_', '.pt') or ''
            if diff_name:
                shutil.copy2(os.path.join(diff_dir, diff_name), os.path.join(UPLOAD_BASE, 'models', diff_name))
                diff_cfg_name = diff_name.replace('.pt', '.yaml')
                shutil.copy2(diff_config, os.path.join(UPLOAD_BASE, 'configs', f'diff_{diff_cfg_name}'))
                saved_diff_model = diff_name
                saved_diff_config = f'diff_{diff_cfg_name}'
                log(f'扩散模型已保存: {diff_name}')
            else:
                log('⚠ 未找到扩散模型文件')
            if diff_error:
                raise RuntimeError(f'扩散训练失败（已保存最新 checkpoint: {diff_name or "无"}）')

        cfg_name = model_name.replace('.pth', '.json')
        diff_cfg_name = diff_name.replace('.pt', '.yaml') if need_diff and diff_name else ''
        result = {
            'status': 'done',
            'model_path': model_name if model_type != 'diffusion' else '',
            'config_path': f'config_{cfg_name}' if model_type != 'diffusion' and cfg_name else '',
            'cluster_path': saved_cluster,
            'diff_model_path': diff_name if need_diff and diff_name else '',
            'diff_config_path': f'diff_{diff_cfg_name}' if need_diff and diff_name else '',
            'progress_msg': '训练完成', 'error_msg': '', 'log': '\n'.join(log_lines),
        }
    except Exception as e:
        result = {
            'status': 'failed',
            'model_path': saved_model or '',
            'config_path': saved_config or '',
            'cluster_path': saved_cluster or '',
            'diff_model_path': saved_diff_model or '',
            'diff_config_path': saved_diff_config or '',
            'error_msg': f'{type(e).__name__}: {e}',
            'progress_msg': '训练失败',
            'log': '\n'.join(log_lines) + '\n' + traceback.format_exc(),
        }
    finally:
        gc.collect()
        torch = sys.modules.get('torch')
        if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result
