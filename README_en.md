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
- **Unified flow (NF+FM hybrid inference)**: rvc-flow A2 can enable a unified flow sharing one FFT backbone across NF and FM; inference can switch between NF / FM / Hybrid modes — Hybrid uses NF for fast positioning + a few FM refinement steps, approaching FM quality at near-NF speed (see [docs/unified_flow.md](docs/unified_flow.md))
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
| `rvc-flow` | lightweight TransformerFlow (A1 / A2), unified flow optional | ~16M | Higher quality ceiling, needs more data |

All three share the ContentVec feature extractor and NSF-HiFiGAN decoder; the only difference is the transform path between "feature → decoder", so the decoder part of a pretrained base model (G_0/D_0) is reusable across architectures.

**RVC direct (`arch: "rvc"`)**

TextEncoder and Flow are removed; ContentVec features go through a single projection straight into the NSF-HiFiGAN decoder, with f0 injected by the decoder's harmonic source. About one-third the params of v1, training is more stable and converges faster, and it avoids KL instability of flow on small datasets. Suits small datasets (<30 min) or when you want a model fast.

**RVC-Flow (`arch: "rvc-flow"`)**

Adds a lightweight TransformerFlow on top of the direct path to enhance feature representation. Two flow modes:

- `A1 feature-prior flow` (`flow_mode: "a1"`): flow transforms features forward, KL constrained to a standard-normal prior (for architecture comparison)
- `A2 posterior flow` (`flow_mode: "a2"`, default): a tiny enc_q (1-layer WN) provides the posterior from the spectrogram, flow aligns it to the prior; more stable training

A2 uses a very lightweight enc_q (1-layer WN) to extract a posterior latent `z_q` from the input spectrogram, and the flow aligns the standard-normal prior to the distribution of `z_q`. Compared to A1, which hard-constrains the whole feature space to a normal distribution, A2 only has the flow learn a "prior ↔ posterior" mapping, so the KL term is more stable and small datasets are less likely to collapse.

### Unified Flow

A2 mode can optionally enable the **unified flow** (`use_unified_flow: true`) — the flagship feature of this project: a single shared FFT backbone carries both Normalizing Flow (NF, reversible) and Flow Matching (FM, velocity field) via dual output heads sharing backbone parameters. Full usage guide: [docs/unified_flow.md](docs/unified_flow.md).

**Motivation**

- Pure NF is fast (a single inverse pass), but one reversible step has limited expressiveness and high-frequency detail is often weak.
- Pure FM (multi-step Euler integration from noise to data) has a high quality ceiling, but 32-step integration is slow.
- Both use the same family of flow backbone (FFT/CouplingLayer); training them separately wastes capacity with no parameter reuse.
- Unified flow lets them **share the backbone and each play to its strength**: NF gives a fast start point, FM refines with a few steps — near-FM quality at near-NF speed.

**Structure**

```
                ┌─────────────────────────────┐
   prior z_p ──▶│                             │── head_nf ──▶ reversible s/t  (NF path, channel-split coupling)
                │   shared FFT backbone        │
   x_t (interp)─▶│   (n_layers CouplingLayers)  │── head_fm ──▶ velocity field v (FM path, predicts v≈x_1-x_0)
                └─────────────────────────────┘
                       ▲ shared params
```

Inside each `GeneralizedCouplingLayer`:

- **Shared backbone**: multi-layer FFT (WN + attention) feature extraction, shared by NF and FM — no parameter doubling.
- **NF head (`head_nf`)**: outputs `s` (scale) / `t` (shift) of the channel-split coupling, guaranteeing reversibility for exact prior↔posterior transforms.
- **FM head (`head_fm`)**: outputs the velocity field `v` guiding the `x_0` (noise/start) → `x_1` (data) trajectory; not reversible, but multi-step Euler integration approaches high-quality samples.

**Training**

`forward` computes both the NF loss (KL + reconstruction, same as original A2) and the FM loss:

- `x_1 = z_q.detach()` (true posterior, gradient detached from enc_q to avoid FM backprop interfering with NF)
- `x_0 = NF_inverse(prior_sample).detach()` (**matches the inference start point** — a key consistency guarantee)
- Linear interpolation `x_t = (1-t)·x_0 + t·x_1` between `x_0` and `x_1`; FM head predicts velocity `v`, MSE fits the true velocity `u_t = x_1 - x_0`
- `loss_flow_match` is weighted by `c_fm` (default 0.5) and merged into `loss_gen_all` for joint backprop

Two critical engineering fixes (missing in early versions; without them Hybrid inference is unintelligible):

1. **Training/inference start-point consistency**: FM training must use "NF inverse output" as `x_0`, not pure noise. Otherwise, at inference FM starts from the NF output, which doesn't match the training distribution, and the velocity field is wrong.
2. **`head_fm` zero initialization**: FM head weights and bias are zeroed, so early on `v=0` (identity map) and Hybrid ≈ NF; the velocity field is learned gradually, preventing a randomly-initialized large velocity from destroying the NF output.

Training logs print an extra `FlowMatch Loss` line, and the web training page draws a third yellow FM loss curve.

**Inference (three modes share one set of weights)**

| Mode | Flow | Steps | Speed | Quality | Use case |
|------|------|-------|-------|---------|----------|
| `nf` | prior sampling → NF inverse → decoder | 1 | Fastest | Slightly weak HF | Speed / real-time |
| `fm` | pure noise → FM 32-step Euler → decoder | 32 | Slowest | Highest ceiling | Offline refinement |
| `hybrid` (recommended) | NF inverse start → FM 4-step refine → decoder | 1+4 | Near NF | Near FM | **Default** |

**Performance** (G_6000.pth, see [docs/unified_flow.md](docs/unified_flow.md) sections 5–6)

| Mode | Time (s) | HF ratio | Notes |
|------|----------|----------|-------|
| NF | 2.14 | 0.260 | Slightly weak HF |
| Hybrid(4) | 0.15 | 0.272 | Near-FM quality, 4.2× faster |
| FM(32) | 0.63 | 0.275 | Quality ceiling |
| Original audio | — | 0.287~0.323 | baseline |

Step-sweep conclusion: `hybrid_steps=4` is the quality/speed sweet spot; 2 steps is faster but measurably lower quality; 8+ has diminishing returns. `c_fm=0.5` is the empirical best (0.1 → FM gets almost no gradient).

**ONNX export**: unified-flow models support export; the Euler integration loop is unrolled per config's `hybrid_steps`, and output matches PyTorch (SNR ~30dB).

**Checkpoint architecture validation**

Checkpoints record an architecture tag (`arch` + `flow_mode` + `use_unified_flow`) on save; on load the tag is verified and mismatches fail loudly, preventing rvc weights from silently loading as rvc-flow (or vice versa) and corrupting the model. v1 base models (G_0/D_0) remain usable as initialization for the rvc family (shared decoder).

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

## Training Parameters

| Parameter | Meaning | Suggestion |
|-----------|---------|------------|
| `total_steps` | total training steps (target total when resuming) | 3000~10000 for small datasets |
| `batch_size` | samples per batch | 4~8 GPU; 1~4 CPU |
| `keep_ckpts` | number of checkpoints kept | 3 |
| `speech_encoder` | feature encoder | `vec768l12` (default) |
| `f0_predictor` | F0 extractor for training | `harvest` (most stable); avoid `dio` |
| `learning_rate` / `lr_decay` | LR and decay | 0.0001 / 0.999~0.999875 |
| `segment_size` | training slice length | 10240 (lower if memory tight) |
| `auto_stop` | stop if loss stalls for N steps | 200 (0=off) |
| `arch` | model architecture | v1 / rvc / rvc-flow (see Architecture section) |
| `d_lr_scale` | discriminator LR scale (first 1000 steps) | 0.5 for rvc family, 1.0 for v1 |
| `flow_mode` | rvc-flow mode | `a2` (recommended) |
| `use_unified_flow` | enable unified flow (shared NF+FM backbone, A2 only) | false (A2 users set true as needed) |
| `c_fm` | FM loss weight (unified flow only) | 0.5 (too small e.g. 0.1 → FM can't learn) |
| `hybrid_steps` | FM refinement steps for hybrid inference | 4 (2 faster but lower quality; 8+ diminishing) |

Diffusion training params: `diff_epochs`, `diff_timesteps` (1000), `diff_kstep` (0=full), `diff_layers/chans/hidden`, `diff_lr`, `diff_decay_step`, `diff_gamma`, `diff_amp`.

## Inference Parameters

| Parameter | Meaning | Suggestion |
|-----------|---------|------------|
| `f0_predictor` | F0 extractor | `pm`/`harvest` on CPU; `crepe` accurate but slow |
| `k_step` | shallow diffusion steps | 100~300 with diffusion model; 0 without |
| `cluster_ratio` | feature retrieval mix | 0.2~0.5 (needs retrieval index) |
| `vc_transform` | transposition (semitones) | 0 |
| `slice_db` | slicing threshold (dB) | -40; raise to -30~-35 if over-sliced |
| `noise_scale` | generation noise | 0.4; 0.25 if harsh |
| `pad_seconds` | segment padding | 0.5 |
| `enhancer` / `second_encoding` / `loudness_envelope` | optional post-processing | usually off |
| `hybrid_mode` | unified-flow inference mode (unified-flow models only) | `auto` (unified→hybrid, else→nf); options `nf`/`fm`/`hybrid` |
| `output_format` | output format | wav / mp3 / flac |

## Deployment & Ops

```bash
systemctl status ssvc          # status
systemctl restart ssvc         # restart after code updates
journalctl -u ssvc -n 100      # logs (inference/training errors)
```

Update: use the "System Update" button in settings (git pull + graceful restart), or manually `git pull gitee master && systemctl restart ssvc`.

Env vars: `PORT` (5000), `INFERENCE_TASK_TIMEOUT` (21600s), `TRAIN_TIMEOUT` (0=unlimited), `INFERENCE_MODEL_CACHE` (3; lower if memory tight), `SSVC_COMPILE` (GPU torch.compile), `SSVC_SERVER_URL` (for email links).

Data safety: weights, database, keys and logs never enter git (see .gitignore). Updates never overwrite `uploads/`, `pretrain/` or `data.db`.

## FAQ

- **Robotic/metallic/harsh output?** Use `harvest` F0 for training (not `dio`); watch for overfitting on small datasets; train a diffusion model and use k_step 100~300; lower noise_scale to 0.25.
- **Inference too slow?** Avoid `crepe` on CPU; lower k_step; slice long audio into 1~2 min clips.
- **"Model architecture mismatch" error?** Checkpoint and config architectures differ (v1/rvc/rvc-flow mixed); use matching config or retrain.
- **Task stuck at running?** Model loading on CPU takes 1~2 min with no progress; you can stop and resubmit; queued tasks auto-recover after restart.
- **OOM on small servers?** Set `INFERENCE_MODEL_CACHE=1`, `cluster_ratio=0`, lower training batch_size.
- **Cluster-related errors?** Auto-generated `*_cluster.pth` is a faiss retrieval index; the loader auto-detects it and also accepts manually uploaded kmeans models.
- **Unified-flow Hybrid inference unintelligible / all noise?** Make sure the checkpoint is from after the "FM consistency fix + head_fm zero-init" — early checkpoints trained FM from a pure-noise start point that doesn't match inference, which corrupts speech. See [docs/unified_flow.md](docs/unified_flow.md) FAQ.

## Directory Layout

```
server/
├── server/              ← Flask app (app.py, templates, workers)
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/
├── configs_template/    ← training config templates
├── docs/                ← design docs (unified flow, etc.)
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
