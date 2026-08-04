"""Modified from upstream so-vits-svc (AGPL-3.0, svc-develop-team).
See NOTICE. 2026-08.
"""
import argparse
import logging
import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore', '.*pynvml.*')

import torch

# PyTorch 2.6+ 默认 weights_only=True，老模型加载需要改回 False
_load = torch.load
torch.load = lambda *a, **kw: _load(*a, **{**kw, 'weights_only': False})
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from scipy.signal import medfilt
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def _postprocess_f0_uv(f0, uv, wav, medfilt_kernel=5, uv_energy_ratio=0.05):
    """对提取的 F0/uv 做后处理：
    1. 中值滤波平滑 F0（抹平呼吸/气声段的基频抖动）
    2. 低能量帧（呼吸、气声）强化为清音（uv=0, f0=0），避免模型学到抖动的 F0
    """
    f0 = np.asarray(f0, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32).copy()

    # 1) F0 中值滤波（仅浊音帧；清音帧保持原值，避免 0 混入中值窗口）
    if medfilt_kernel >= 3 and medfilt_kernel % 2 == 1:
        voiced = uv > 0.5
        if np.any(voiced):
            f0_smooth = medfilt(np.where(voiced, f0, np.nan), medfilt_kernel)
            f0 = np.where(voiced & ~np.isnan(f0_smooth), f0_smooth, f0)

    # 2) 低能量帧 -> 清音
    if uv_energy_ratio > 0:
        frame_len = hop_length
        n_frames = len(f0)
        rms = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            seg = wav[i * frame_len:(i + 1) * frame_len]
            if len(seg):
                rms[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        pos = rms[rms > 0]
        if pos.size:
            thr = float(np.median(pos)) * uv_energy_ratio
            low_energy = rms < thr
            f0 = np.where(low_energy, 0.0, f0)
            uv = np.where(low_energy, 0.0, uv)

    return f0, uv


def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None,
                f0_smooth_kernel=5, uv_energy_ratio=0.05):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        f0, uv = _postprocess_f0_uv(f0, uv, wav,
                                    medfilt_kernel=f0_smooth_kernel,
                                    uv_energy_ratio=uv_energy_ratio)
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized

        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )

        #audio_norm = audio / hps.data.max_wav_value

        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu",
                  f0_smooth_kernel=5, uv_energy_ratio=0.05):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor,
                    f0_smooth_kernel, uv_energy_ratio)

def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device,
                     f0_smooth_kernel=5, uv_energy_ratio=0.05):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device,
                                         f0_smooth_kernel=f0_smooth_kernel,
                                         uv_energy_ratio=uv_energy_ratio))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    parser.add_argument(
        '--f0_smooth_kernel', type=int, default=5,
        help='F0 中值滤波窗口（奇数，0=关闭）。平滑呼吸/气声段的基频抖动，默认 5'
    )
    parser.add_argument(
        '--uv_energy_ratio', type=float, default=0.05,
        help='低能量帧判清音的阈值（相对 RMS 中位数的比例，0=关闭）。强化呼吸/气声段的 uv，默认 0.05'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))

    if args.use_diff:
        print("use_diff")
        print("Loading Mel Extractor...")
        mel_extractor = Vocoder(dconfig.vocoder.type, dconfig.vocoder.ckpt, device=device)
        print("Loaded Mel Extractor.")
    else:
        mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device,
                     f0_smooth_kernel=args.f0_smooth_kernel,
                     uv_energy_ratio=args.uv_energy_ratio)
