# So-VITS-SVC Server

A standalone **inference + training** web server for so-vits-svc, with multi-user login, model management, inference presets, a task queue, and one-click deployment.

**English · [中文](README.md)**

## License

This project is a derivative of [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (**AGPL-3.0**) and is released under **AGPL-3.0** (see LICENSE and NOTICE).

- Upstream license and copyright are preserved; modified upstream files are listed in NOTICE
- Full corresponding source is public in this repository (AGPL-3.0 network-use requirement)
- Redistribution and self-hosting must keep copyright/modification notices and stay under the same license

## Features

- **Inference**: upload model + audio → SVC conversion; shallow diffusion (k_step); tune every parameter (F0 / enhancer / retrieval / transposition) right on the inference page
- **Feature retrieval**: faiss index built automatically after training; `cluster_ratio` mixing at inference for more stable timbre
- **Training**: dataset zip → queued training; **resume to a target step** (main model and diffusion, one-click from the task list)
- **Multiple architectures**: SoVITS v1 / RVC direct / RVC-Flow (lightweight TransformerFlow, A1/A2 switchable), selectable on the training page
- **Stop & audition**: stop training anytime; the current checkpoint is saved as a model for immediate inference
- **Monitoring**: live progress, G/D loss curves (separate diffusion curve), **validation loss** (overfitting check), recent-speed ETA
- **Anomaly email confirmation**: when the discriminator crushes the generator or loss goes NaN, an email with continue/stop confirmation links is sent
- **Precise cleanup**: per-task / per-model / per-dataset cleanup with mis-deletion warnings
- **File manager**: paginated browsing, batch delete, download models / configs / results / datasets
- **Pretrained assets page**: upload ContentVec, NSF-HiFiGAN, base models G_0/D_0 etc.
- **Email notifications**: training complete / progress / anomaly and inference complete (configurable SMTP)
- **Performance**: inference model LRU cache (zero load cost on repeated inference), GPU torch.compile
- **Task control**: stop training/inference anytime (inference keeps model cache, checkpoints auto-saved)
- **Quick resume**: one-click continue the last training after stopping (TEMP snapshot, all params editable)
- **Ops**: one-click git pull + graceful restart in settings; training parameter presets
- Security: CSRF protection, login rate limiting, path traversal checks, persistent session key, checkpoint architecture validation

## Model Architectures

On top of the original so-vits-svc (VITS structure), this project adds two self-developed lightweight architectures, selectable on the training page:

| Architecture | Structure | Params | Notes |
|--------------|-----------|--------|-------|
| `sovits-v1` | TextEncoder + Flow + enc_q (original VITS) | ~52M | Best compatibility, interchangeable with community models |
| `rvc` | feature-direct decoder (no TextEncoder / Flow) | ~15.5M | More stable and faster training, good timbre preservation, suits small datasets |
| `rvc-flow` | lightweight TransformerFlow (A1 / A2) | ~16M | Higher quality ceiling, needs more data |

**RVC direct (`arch: "rvc"`)**

TextEncoder and Flow are removed; ContentVec features go through a single projection straight into the NSF-HiFiGAN decoder, with f0 injected by the decoder's harmonic source. About one-third the params of v1, training is more stable and converges faster, and it avoids KL instability of flow on small datasets.

**RVC-Flow (`arch: "rvc-flow"`)**

Adds a lightweight TransformerFlow on top of the direct path to enhance feature representation. Two modes:

- `A1 feature-prior flow` (`flow_mode: "a1"`): flow transforms features forward, KL constrained to a standard-normal prior (for architecture comparison)
- `A2 posterior flow` (`flow_mode: "a2"`, default): a tiny enc_q (1-layer WN) provides the posterior from the spectrogram, flow aligns it to the prior; more stable training

**Checkpoint architecture validation**

Checkpoints record an architecture tag on save; on load the tag is verified and mismatches fail loudly, preventing rvc weights from silently loading as rvc-flow (or vice versa) and corrupting the model. v1 base models (G_0/D_0) remain usable as initialization for the rvc family (shared decoder).

## Quick Start

### Windows (one-click)

```bat
install.bat      :: create venv + install CPU/CUDA torch + deps (1=CPU / 2=CUDA)
start.bat        :: start server, open http://localhost:5000
```

### Linux (all platforms, Chinese mirrors)

One integrated deploy script covering **NVIDIA GPU, AMD GPU and CPU-only**, auto-selecting the PyTorch build.

```bash
chmod +x deploy_linux.sh
sudo bash deploy_linux.sh
# Optional: skip model downloads if the network is slow; upload them via the web UI afterwards
# sudo bash deploy_linux.sh --skip-models
```

> **Before running on Linux:** make sure your GPU driver is installed (NVIDIA: `nvidia-smi` works; AMD: ROCm driver installed and `rocm-smi` works).

The script installs ffmpeg / libsndfile / cmake and other system deps, and sets up Miniconda + Python 3.9 from Tsinghua/Aliyun mirrors. You only need to prepare the GPU driver.

> **Windows users:** the `onnxsim` dependency is built from source and requires CMake. Install CMake from the official site, check *Add CMake to the system PATH for all users*, then reopen the terminal and run the installer.

> **AMD users:** AMD ROCm PyTorch is only supported on Linux (RX 5000/6000/7000/9000 series). On Windows, use the CPU build (fine for inference) or WSL. `install.bat` falls back to CPU automatically when AMD is selected on Windows.

## Configuration

| Item | Env var | Default |
|------|---------|---------|
| Session key | SECRET_KEY | auto-generated, persisted to `server/secret_key.txt` |
| Database | DATABASE_URL | server/data.db |
| Port | PORT | 5000 |
| Inference timeout | INFERENCE_TASK_TIMEOUT | 21600 seconds (6h) |

## Pretrained Models

**Required** for inference: ContentVec + NSF-HiFiGAN. Base models G_0/D_0 are optional but recommended for training.

| File | Size | How to get |
|------|------|------------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | Auto-downloaded by deploy script, or upload via "Pretrained" page |
| `pretrain/nsf_hifigan/model` + `config.json` | ~54MB | Same |
| `pretrain/G_0.pth` + `pretrain/D_0.pth` | ~400MB | Manual upload (recommended) |
| `pretrain/rmvpe.pt` etc. | optional | Upload via "Pretrained" page |

> Large files: scp them directly into `pretrain/`; the web UI detects them automatically.

## Usage

**Inference:**
1. Model management → upload `G_*.pth` + config.json (optionally diffusion model + diffusion.yaml)
2. Create an inference preset (f0 predictor, noise_scale, k_step, etc.)
3. Pick a preset + upload audio on the inference page (params overridable inline) → watch progress / stop / download in the task list

**Training:**
1. Training page → upload dataset zip (clips < 2s are filtered automatically)
2. Set parameters (total steps, auto_stop, encoder; base models load automatically)
3. Watch train/validation loss → "Stop" when satisfied (checkpoint auto-saved)
4. Resume from the task list, or test the current model directly

**Diffusion:** after the main model is trained, click "Train diffusion" in the task list (reuses data/features); attach the diffusion model to the main model and infer with k_step 100~300.

## Directory Layout

```
server/
├── server/              ← Flask app (app.py, templates, workers)
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/
├── configs_template/    ← training config templates
├── pretrain/            ← pretrained models (not in git)
├── train.py train_diff.py preprocess_*.py
├── deploy_linux.sh      ← Linux one-click deploy (CUDA / ROCm / CPU, conda)
├── install.bat / start.bat / start.ps1  ← Windows install/start
└── requirements.txt LICENSE
```

## Notes

- Model weights, database and keys are **not** in git (see .gitignore)
- CPU servers can run inference; training is possible but very slow — prefer GPU training then upload models
- CPU inference tips: use `pm`/`harvest` F0 predictor (`crepe` is several times slower), lower `k_step` (30~100) as needed
- Deploying behind the GFW: apt/conda/pip all use Tsinghua/Aliyun mirrors; torch uses `--no-deps` to skip nvidia packages
- fairseq needs pip 24.0 (handled by the script); librosa 0.10.1 for newer numpy/torch compatibility
- Training DataLoader defaults to `num_workers=2` (auto-upgrades to 4 with enough RAM, drops to 0 below 8GB)
