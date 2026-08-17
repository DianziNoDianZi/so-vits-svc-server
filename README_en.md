# So-VITS-SVC Multi-User Inference Server

An independently deployed **multi-user inference web server** for so-vits-svc, with an optional **training** module (admin only).
Users can register to use official models published by the platform or upload private models, submit audio for asynchronous inference, and download results.
Administrators are responsible for model review, resource quota allocation, queue management, and announcements.

**[中文](README.md) · English**

## License

This project is a derivative work of [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
(**AGPL-3.0**), released in its entirety under the **AGPL-3.0** license (see LICENSE and NOTICE).

- The upstream LICENSE and copyright attributions are retained; modified upstream files are declared in NOTICE
- The complete corresponding source code is publicly available in this repository, satisfying AGPL-3.0's requirement to provide source code for network services
- When using/redistributing this repository, you must comply with AGPL-3.0: retain copyright and modification notices, and distribute under the same license

## Features

- **Multi-user**: open registration (can be disabled in admin panel), roles (admin/user), login rate limiting, CSRF
- **Model management**: private models uploaded by users require admin review; **apply for public release** (set as official shared model after review approval); admins can list/delist official models, download model files for verification, and automatically validate configuration legitimacy; rejected models are deleted directly
- **Resource quotas**: admins set per-user queued/running task limits, single audio duration limits, daily task submission limits, CPU core counts, private model limits, result retention days, and priority
- **Fair scheduling**: tasks are queued with per-user round-robin + priority, single GPU serial execution; executor disconnection auto-recovery; admins can pause/resume new task scheduling
- **Batch inference**: upload multiple audio files at once to create multiple tasks; task list supports filtering/search/pagination, task detail pages, and visible queue positions
- **Server-wide email**: admins configure SMTP once; users set **result delivery email addresses**; automatic email notifications on inference completion, resource strain, and announcement broadcasts (rate-limited to prevent abuse)
- **Resource monitoring**: banner alerts at the top of all pages + email alerts when CPU/memory/disk/GPU memory are under strain
- **Announcements**: admins publish/pin/broadcast via email; users view on homepage and announcements page
- **Object-level result downloads**: access-controlled downloads based on task ownership; results retained for 7 days by default (configurable) before automatic cleanup
- **Inference performance**: resident daemon + LRU model cache, ONNX acceleration (automatic fallback to PyTorch on failure)
- **Real-time voice conversion (WebSocket)**: client microphone → `ws://:5001` → server real-time voice conversion → playback echo; 44.1kHz, `chunk_seconds` configurable (0.1~2.0s), API Key authentication; near-real-time (CPU approximately 1~2s)
>**About real-time voice conversion**
>
>The developer spent a long time testing real-time voice conversion, but it just wouldn't work. If you have any suggestions, please raise them in the issues — I would be very grateful!
- **Training (optional module, admin only)**: backend toggle control; SoVITS / SoVITS+diffusion, continued training, stopping, registering checkpoints as inference models, progress polling
- **Admin backend**: overview (health/pause/pending review/disk), user quotas, model review, global task queue (can stop/delete/export CSV), storage and orphaned files, announcements, site settings, training toggle
- **Long-term operations**:
  - **Automatic database backups**: SQLite online backups (WAL-safe), scheduled + retention count, admin manual trigger/download
  - **Structured logging**: `server/logs/app.log` rotated by size, with timestamps/levels/request information
  - **Health checks**: `/healthz` returns JSON with db/daemon/queue/disk/pause status; `/status` system status page (CPU/memory/GPU/task statistics/recent failures/recent backups)
  - **Audit logs**: all critical admin operations (quotas/reviews/tasks/settings/updates/announcements/backups/API Keys) are fully logged and viewable in the backend
  - **Invitation code registration**: registration mode can be set to closed/optional/required; admins batch-generate/revoke codes to control the pace of open registration
  - **External REST API**: available to users/admins alike, authenticated via `X-API-Key`, supporting inference submission/status queries/result downloads/system status; `/api/v1/docs` online documentation
  - **General rate limiting**: inference/download/upload rate-limited per user (requests/minute), thresholds adjustable in the backend
- Security: CSRF, login/registration rate limiting, general rate limiting, path traversal protection, session key persistence, checkpoint architecture validation, `weights_only` priority loading for untrusted weights

## Model Architectures

Building on the original so-vits-svc (VITS structure), this project adds two custom lightweight architectures that can be selected directly on the training page (if enabled):

| Architecture | Structure | Parameters | Characteristics |
|------|------|--------|------|
| `sovits-v1` | TextEncoder + Flow + enc_q (original VITS) | ~52M | Best compatibility, interoperable with community models |
| `rvc` | Feature direct-to-decoder (no TextEncoder / Flow) | ~15.5M | More stable and faster training, good timbre preservation, suitable for small datasets |
| `rvc-flow` | Lightweight TransformerFlow (A1 / A2), optional unified flow | ~16M | Higher audio quality ceiling, requires more data support |

All three architectures share the ContentVec feature extractor and NSF-HiFiGAN decoder; the differences lie only in the transformation path between "features → decoder," so the decoder portion of the same pretrained base models (G_0/D_0) can be reused across architectures.

### Recommended Dataset Sizes and Training Steps per Architecture

| Architecture | Recommended Dataset Size | Recommended Total Steps (`total_steps`) | Notes |
|------|-------------|--------------------------|------|
| `sovits-v1` | 30 minutes ~ 2 hours | 10000 ~ 50000 | More data is better; below 30 minutes tends to underfit with unstable timbre |
| `rvc` | 5 ~ 30 minutes | 5000 ~ 20000 | Fast convergence on small data, stable timbre, the top choice for short datasets; diminishing returns beyond 1 hour |
| `rvc-flow` | 1 ~ 3 hours | 20000 ~ 80000 | Needs sufficient data to support the flow; too little data causes instability or unintelligibility; more data yields a higher quality ceiling |

- **A2 posterior flow (default)** is more stable than A1 on small datasets; recommended for datasets under 30 minutes.
- **A1 feature prior flow** achieves more thorough timbre disentanglement, but requires a dataset of ≥1 hour and `c_kl` tuned to `0.1`.
- **Unified flow (A2 + `use_unified_flow`)** has greater capacity; not recommended for datasets under 1 hour.
- The `total_steps` values above are references for "training from scratch"; for **continued training**, enter the target total step count (including already-trained steps). Observe the loss on the page: once mel loss reaches a stable plateau and subjective listening is satisfactory, early stopping is fine — you don't need to run the full count.

**RVC Lightweight Direct Connection (`arch: "rvc"`)**

Removes TextEncoder and Flow; ContentVec features pass through a single-layer projection directly into the NSF-HiFiGAN decoder, with f0 injected via the decoder's harmonic source. Parameter count is about one-third of v1, training is more stable and converges faster, and it avoids the KL instability issues of flow on small datasets. Suitable for small datasets (<30 minutes) or scenarios requiring rapid model production.

**RVC-Flow (`arch: "rvc-flow"`)**

Adds a lightweight TransformerFlow on top of the direct connection to enhance feature representation, with two switchable flow modes:

- `A1 feature prior flow` (`flow_mode: "a1"`): the flow forward-transforms content features `c`, with KL constrained to a **fixed** standard normal prior N(0,1); training and inference paths are identical (both use the forward flow), with no posterior encoder and no prior sampling
- `A2 posterior flow` (`flow_mode: "a2"`, default): a minimal enc_q (1-layer WN) provides posterior `z_q` from the spectrogram; the flow performs prior↔posterior alignment; training is more stable

**A1 Feature Prior Flow (Detailed)**

The design goal is to **disentangle timbre from pronunciation**: A1 makes `z` fully determined by the source audio's content features `c` (ContentVec output); the flow only performs forward encoding `c → z_p`; the decoder then reconstructs the spectrogram from `z_p` — pronunciation information comes from the source, timbre information from the speaker embedding, with no cross-contamination.

- **Training path**: `x = pre(c) + emb_uv + vol` → `z_p = flow(x, x_mask, g=g)` → `dec(z_p_slice, g=g, f0=pitch_slice)`, with **no enc_q and no prior sampling** throughout — training is inference.
- **Inference path**: completely identical to training, `z_p = flow(x)` → `dec(z_p)`, deterministic forward transformation (`noise_scale` does not affect A1 because there is no prior sampling step).
- **Fixed prior N(0,1)**: `m_p=0, logs_p=0`, the KL term simplifies to `KL = -0.5 + 0.5 * ||z_p||²`, constraining only the flow output variance ≈1 to prevent drift. Early versions used `prior_proj(x)` to learn prior mean/variance, but because flow and prior_proj shared the input `x`, they would **collude** (jointly pushing KL toward -∞, measured at -262); this was deprecated and removed.
- **`c_kl` adjustment**: A1's KL term has a different numerical range than A2 (A2 KL ≈ positive, A1 KL can be negative); the default `c_kl=1.0` would let the KL loss dominate and suppress the mel reconstruction loss. Empirically, A1 recommends `c_kl=0.1`, letting mel loss dominate with KL as light regularization.
- **Unified flow/Hybrid not supported**: the FM training of unified flow requires `enc_q` to provide `z_q` as the FM target; A1 has no `enc_q` and cannot provide it. At the code level, A1 forces the use of `TransformerCouplingBlock` (not `GeneralizedFlow`), and the `infer_hybrid` entry point has assertions preventing misuse.

### Unified Flow

In A2 mode, you can enable the "**Unified Flow**" (`use_unified_flow: true`): use **the same FFT backbone** to simultaneously support two paths — Normalizing Flow (NF, invertible) and Flow Matching (FM, velocity field) — with dual output heads sharing backbone parameters. Full documentation in [docs/unified_flow.md](docs/unified_flow.md).

**Design Motivation**

- Pure NF inference is fast (single inverse transform), but the expressiveness of a single invertible transform is limited; high-frequency details are often weak.
- Pure FM (multi-step Euler integration from noise to data) has a high quality ceiling, but 32-step integration makes inference slow.
- Both use the same class of flow backbone (FFT/CouplingLayer), but when trained separately their parameters are completely non-reusable, wasting capacity.
- Unified flow lets both **share the backbone and leverage their respective strengths**: NF provides a fast starting point, FM refines with a small number of steps, achieving quality close to pure FM and speed close to pure NF.

**Structure**

```
                ┌─────────────────────────────┐
   prior z_p ──▶│                             │── head_nf ──▶ invertible transform s/t  (NF path, channel-split coupling)
                │    shared FFT backbone      │
   x_t (interp)▶│   (n_layers CouplingLayers) │── head_fm ──▶ velocity field v      (FM path, predicts v≈x_1-x_0)
                └─────────────────────────────┘
                       ▲ shared parameters
```

Inside each `GeneralizedCouplingLayer`:

- **Shared backbone**: multi-layer FFT (WN + attention) extracts features, shared by NF and FM without doubling parameters.
- **NF head (`head_nf`)**: outputs the `s` (scale) / `t` (translate) of a channel-split coupling, guaranteeing invertibility, for precise prior↔posterior transformation.
- **FM head (`head_fm`)**: outputs the velocity field `v`, guiding the trajectory of `x_0` (noise/start) → `x_1` (data); not invertible, but can approach high-quality samples via multi-step Euler integration.

**Training**

`forward` simultaneously computes NF loss (KL + reconstruction) and FM loss:

- `x_1 = z_q.detach()` (true posterior; gradients to enc_q are cut to prevent FM backprop from interfering with NF)
- `x_0 = NF_inverse_transform(prior_sample).detach()` (**consistent with the inference starting point** — a critical consistency guarantee)
- Linear interpolation between `[x_0, x_1]`: `x_t = (1-t)·x_0 + t·x_1`; the FM head predicts velocity `v`; MSE fits the true velocity `u_t = x_1 - x_0`
- `loss_flow_match` is weighted by `c_fm` (default 0.5) and merged into `loss_gen_all` for unified backpropagation

Two critical engineering fixes (absent in early versions, causing Hybrid inference to produce unintelligible speech):

1. **Training/inference starting point consistency**: the `x_0` for FM training must use the "NF inverse transform output" rather than pure noise; otherwise, at inference time FM starting from the NF output would be out-of-distribution relative to training, and the velocity field would be computed incorrectly.
2. **`head_fm` zero initialization**: FM head weights and biases are zeroed out; initially `v=0` (identity mapping), Hybrid ≈ NF; the velocity field is learned progressively during training, avoiding the large random velocity field from destroying the NF output.

**Inference (three modes sharing the same weights)**

| Mode | Process | Steps | Speed | Quality | Use case |
|------|------|------|------|------|------|
| `nf` | prior sample → NF inverse transform → decoder | 1 | Fastest | High frequencies slightly weak | Speed-focused / real-time |
| `fm` | pure noise → FM 32-step Euler integration → decoder | 32 | Slowest | Highest ceiling | Offline refinement |
| `hybrid` (recommended) | NF inverse transform as starting point → FM 4-step refinement → decoder | 1+4 | Close to NF | Close to FM | **Default** |

**Checkpoint Architecture Validation**

Checkpoints record architecture tags (`arch` + `flow_mode` + `use_unified_flow`) at save time and are validated at load time: architecture mismatches raise errors immediately, preventing silent loading of rvc weights as rvc-flow (or vice versa) that would corrupt the model. v1 base models (G_0/D_0) can still be reused as initialization weights for the rvc series (the decoder portion is universal).

## Quick Start

### Windows Local (One-Click Install)

```bat
install.bat      :: create venv + install CPU/CUDA torch + dependencies (choose 1=CPU / 2=CUDA)
start.bat        :: start the service, visit http://localhost:5000
```

On first startup, the console prints the initial admin password (account `admin`); you are forced to change it after login.

### Linux Deployment

```bash
sudo bash deploy_linux.sh   # NVIDIA GPU / AMD ROCm / CPU auto-adaptation, all via domestic mirrors
```

- The script automatically installs system dependencies such as ffmpeg, libsndfile, and cmake, and configures Miniconda/Python 3.9.
- GPU drivers must be installed in advance (verify with `nvidia-smi` for NVIDIA; install ROCm and add the user to the `video`/`render` groups for AMD).
- If the network is slow, add `--skip-models` to skip model downloads, then manually place models in `pretrain/` after deployment.

## Configuration (Environment Variables)

| Item | Environment Variable | Default |
|----|---------|------|
| Session key | SECRET_KEY | Auto-generated and persisted to `server/secret_key.txt` |
| Database | DATABASE_URL | server/data.db |
| Service port | PORT | 5000 |
| Inference timeout | INFERENCE_TASK_TIMEOUT | 21600 (seconds, 6 hours) |
| Inference model cache | INFERENCE_MODEL_CACHE | 1 (set to 1 for low memory) |
| Inference chunk seconds | INFERENCE_CLIP_SECONDS | 15 (long audio split into chunks to prevent OOM, 0=no splitting) |
| Resource alert threshold/rate limit | RESOURCE_THRESHOLD / RESOURCE_EMAIL_INTERVAL | 90 / 3600 |
| Server SMTP | SMTP_HOST/PORT/USER/PASS/MAIL_FROM | Empty (can also be configured in admin panel) |
| Access URL in emails | SSVC_SERVER_URL | Auto-detected |
| Registration toggle | ALLOW_REGISTRATION | 1 (can also be configured in admin panel) |

## Pretrained Models

ContentVec + NSF-HiFiGAN are **required** for inference; training base models G_0/D_0 are optional but recommended.

| File | Size | Acquisition |
|------|------|------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | Auto-downloaded by deployment script, or place manually |
| `pretrain/nsf_hifigan/model` + `config.json` | ~54MB | Same as above |
| `pretrain/G_0.pth` + `pretrain/D_0.pth` | ~400MB | Place manually (recommended) |
| `pretrain/rmvpe.pt` and other encoders | Optional | Place manually |

> For large files, it's recommended to scp them directly into the `pretrain/` directory; the service auto-detects them on startup.

## Usage Flow

**Users:**
1. Register an account (can fill in delivery email and result delivery email)
2. Upload private models on the Models page → wait for admin review; or directly use platform **official models**
3. Create inference configurations (can be based on official/own models)
4. On the Inference page, select a configuration + upload audio (can select multiple at once for batch submission) → view progress/queue position/stop/download in the task list
5. View full parameters and errors on the task detail page; results are automatically cleaned up when retention expires

**Admins:**
1. Backend "Admin ▾" → Overview: pending review models, health status (daemon/GPU/queue), pause/resume scheduling, **database backups** (immediate backup/download)
2. **Models**: review user-uploaded models (approve / reject = delete), list/delist official models, download for verification, automatic validation
3. **Users**: enable/disable, set per-user quotas (daily task count, CPU cores, queued/running, private models, result retention)
4. **Invitation codes**: generate/revoke invitation codes, combined with the Settings page "invitation code mode" (closed/optional/required) to control registration pace
5. **Tasks**: global queue, can stop/delete/export CSV
6. **Announcements**: publish, pin, email broadcast
7. **Settings**: registration toggle, invitation code mode, site default quotas, **rate limit thresholds** (inference/download/upload), **backup interval and retention count**, SMTP, training feature toggle and CPU cores
8. **Audit logs**: full traceability of critical admin operations, filterable by operation type
9. **Training** (if enabled): upload datasets to train SoVITS, register as inference models upon completion

**Developers (users/admins)**: generate an API Key in Settings → "Developer API"; see [REST API](#rest-api) below.

## Optional Training Module (Admin Only)

Disabled by default; admins enable it in `Settings → Training Feature`, or visit `/train` for one-click enablement.

- **Submit training**: upload dataset zip (auto-filters clips shorter than 2 seconds) + speaker + parameters (total steps/encoder/F0/architecture/flow/unified flow/diffusion)
- **Continued training**: one-click "continue training" on historical tasks to a specified step count
- **Stop**: can stop during running, automatically saving the current checkpoint
- **Register model**: upon training completion, register `G_*.pth` as an inference model (automatically attaches cluster index)
- **Progress**: page polling shows stage/percentage/step

## Training Parameter Reference (within the Training Module)

| Parameter | Meaning | Recommendation |
|------|------|------|
| `total_steps` | Total training steps (target total steps when continuing training) | See "Recommended Dataset Sizes and Training Steps per Architecture" above |
| `batch_size` | Samples per batch | GPU 4~8; CPU 1~4 |
| `keep_ckpts` | Number of recent checkpoints to retain | 3 |
| `speech_encoder` | Feature encoder | `vec768l12` (recommended) |
| `f0_predictor` | F0 extractor (for training) | `harvest` most stable; `dio` faster but worse |
| `arch` / `flow_mode` | Architecture / rvc-flow mode | v1 / rvc / rvc-flow; A2 default |
| `use_unified_flow` | Unified flow (A2 only) | false (enable as needed) |
| `c_fm` | FM loss weight (unified flow) | 0.5 |
| `c_mel` / `c_kl` | Mel / KL weights | 45; A1 use 0.1, A2 use 1.0 |

**Diffusion parameters** (when selecting SoVITS+diffusion): `diff_epochs`, `diff_timesteps`, `diff_kstep` (shallow diffusion max steps, 0=full), `diff_layers/chans/hidden` (capacity), `diff_lr`, `diff_amp` (fp32/fp16/bf16).

## Inference Parameter Reference

The Inference page can temporarily override configuration defaults (parameters collapsed by default):

| Parameter | Meaning | Recommendation |
|------|------|------|
| `f0_predictor` | F0 extractor | `pm`/`harvest` (fast on CPU); `crepe` most accurate but several times slower |
| `k_step` | Shallow diffusion steps | 100~300 when a diffusion model is attached; 0 when none |
| `cluster_ratio` | Feature retrieval mixing ratio | 0.2~0.5 (requires model with retrieval index attached; automatically disabled without one) |
| `vc_transform` | Pitch shift (semitones) | 0 |
| `slice_db` | Slicing threshold (dB) | -40; adjust to -30~-35 if audio is sliced too finely |
| `noise_scale` | Generation noise | 0.4; lower to 0.25 if the voice sounds rough |
| `pad_seconds` | Inter-segment padding | 0.5 |
| `auto_f0` / `enhancer` / `second_encoding` | Auto F0 / NSF enhancement / second encoding | Generally off |
| `loudness_envelope` | Loudness envelope | 0~1 |
| `hybrid_mode` | Unified flow inference mode (only effective for unified flow models) | `auto`; options: `nf`/`fm`/`hybrid` |
| `output_format` | Output format | wav / mp3 / flac |

**ONNX export** (optional, improves inference speed): the model edit page can export the main generator to ONNX; after export, the same directory automatically uses onnxruntime for inference, with automatic fallback to PyTorch on failure. Command line alternative:
```bash
python onnx_export_generator.py <model.pth> <config.json> <output.onnx>
```

## REST API

External REST interface, available to users and admins alike (permissions consistent with account roles). Generate a Key in Settings → "Developer API"; send requests with the `X-API-Key: <key>` header (compatible with `Authorization: Bearer <key>`). Full online documentation at **`/api/v1/docs`**.

Quick examples:

```bash
# Submit inference (multipart audio upload)
curl -X POST https://your-domain/api/v1/inference \
  -H "X-API-Key: $KEY" \
  -F "config_id=1" \
  -F "audio=@song.wav"

# Query task status / download result
curl https://your-domain/api/v1/tasks/42 -H "X-API-Key: $KEY"
curl -o result.wav https://your-domain/api/v1/tasks/42/result -H "X-API-Key: $KEY"

# System status / my quota / available models / my configs
curl https://your-domain/api/v1/system -H "X-API-Key: $KEY"
curl https://your-domain/api/v1/me -H "X-API-Key: $KEY"
curl https://your-domain/api/v1/models -H "X-API-Key: $KEY"
curl https://your-domain/api/v1/configs -H "X-API-Key: $KEY"
```

| Method | Path | Description |
|------|------|------|
| POST | `/api/v1/inference` | Submit inference (`config_id` + `audio` file(s), multiple files supported) |
| GET | `/api/v1/tasks/<id>` | Single task status |
| GET | `/api/v1/tasks` | My task list (`?status=done` filter) |
| GET | `/api/v1/tasks/<id>/result` | Download result file |
| GET | `/api/v1/models` | Models available to me |
| GET | `/api/v1/configs` | My inference configurations (including parameters) |
| GET | `/api/v1/me` | My information + quota + today's usage |
| GET | `/api/v1/system` | System status (daemon/queue/scheduling/CPU/memory/disk/statistics) |
| WS | `ws://host:5001/api/v1/ws/stream` | Real-time voice conversion (see [docs/realtime_vc.md](docs/realtime_vc.md)) |

Error codes: `401` invalid Key · `403` account disabled or resource not owned by you · `404` not found · `429` rate-limited or quota exhausted · `400` parameter error.

## Deployment and Operations

**Linux service management (systemd)**

```bash
systemctl status ssvc          # view status (HTTP service :5000)
systemctl restart ssvc         # restart (required after code updates)
systemctl status ssvc-ws       # view status (real-time voice conversion WebSocket :5001)
systemctl restart ssvc-ws      # restart WebSocket service
journalctl -u ssvc -n 100      # view service logs (stdout captured by systemd)
tail -f server/logs/app.log    # view structured application logs (rotated)
```

**Updating code**

```bash
cd ~/server/so-vits-svc-inference && git pull origin master && systemctl restart ssvc ssvc-ws
```

**Real-time voice conversion**: depends on `gevent`/`gevent-websocket` (included in requirements; re-run the deployment script or manually `pip install gevent gevent-websocket`). Client interface protocol in [docs/realtime_vc.md](docs/realtime_vc.md).

**Data safety**: model weights, database, keys, uploaded files, **backups** (`server/backups/`), and **logs** (`server/logs/`) are all excluded from git (see .gitignore). Code updates will not overwrite `uploads/`, `pretrain/`, `data.db`, or `backups/`.

## FAQ

**Why can't I use my uploaded model for inference immediately?**
Private models must be approved by an admin on the "Models" page before use; alternatively, use official shared models directly.

**How do I receive results after inference completes?**
Fill in "result delivery email" on the Settings page and enable notifications; server SMTP must be configured by an admin in "Settings → SMTP".

**Tasks stay queued?**
Check whether the backend has "paused new tasks"; single-GPU round-robin scheduling means long tasks cause subsequent tasks to queue; the Overview/Tasks pages show global queue counts.

**Inference tasks stay running / progress not moving?**
No progress during the model loading phase (1~2 minutes on CPU) is normal; click "Stop" and resubmit.

**Trained voice sounds electronic/metallic/hoarse?**
F0 predictor issues (`dio` during training tends to cause hoarseness; switch to `harvest`); undertraining or overfitting; or train a diffusion model and attach it with `k_step` 100~300 for repair; lower inference `noise_scale` to 0.25.

**"Model architecture mismatch" error?**
The checkpoint and configuration architectures are inconsistent (v1/rvc/rvc-flow mixed). Use a matching configuration, or retrain.

**Server low on memory (OOM)?**
Lower `INFERENCE_MODEL_CACHE=1`, `INFERENCE_CLIP_SECONDS=15` (long audio auto-chunking reduces peak memory), set inference config `cluster_ratio=0`, and limit per-user daily/queue quotas.

**How to restore a database backup?**
Backend "Overview → Database Backup" downloads `backup_*.db`. To restore: stop the service → replace `server/data.db` with the backup file (delete the same-named `-wal`/`-shm`) → start the service.

**How to submit inference via API?**
Generate a Key in Settings → "Developer API", then call `/api/v1/inference` (multipart) with the `X-API-Key` header. See examples in [REST API](#rest-api) above and `/api/v1/docs`.

**How to restrict registration (prevent spam accounts)?**
Set backend "Settings → Invitation Code Mode" to "Required", and generate invitation codes on the "Invitation Codes" page to distribute to users.

**Unified flow Hybrid inference produces unintelligible speech / all noise?**
Confirm the checkpoint was trained with a version after the "FM consistency fix + head_fm zero initialization"; early checkpoints (before the fix) trained FM from a pure noise starting point, which is inconsistent with the inference starting point and will corrupt speech.

## Directory Structure

```
server/
├── server/              ← Flask service (app.py entry point, blueprints/ routes, services/ service layer, templates, worker)
│   ├── blueprints/      ← auth/dashboard/models/configs/inference/tasks/announcements/admin/training/health checks/status/REST API
│   ├── services/        ← quotas/scheduling/training/model validation/backups/audit/logging/system resources/REST authentication
│   ├── templates/       ← page templates
│   ├── inference_daemon.py  inference_worker.py  ← inference execution layer
│   ├── backups/         ← automatic database backups (not in git)
│   ├── logs/            ← structured rotated logs (not in git)
│   └── app.py           ← application factory and entry point
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/   ← inference algorithms
├── docs/                ← design documents such as unified flow
├── pretrain/            ← pretrained models (not in git)
├── train.py train_diff.py preprocess_*.py   ← training scripts (optional module execution layer)
├── deploy_linux.sh      ← Linux one-click deployment
├── install.bat / start.bat / start.ps1      ← Windows install/start
└── requirements.txt LICENSE NOTICE
```

## Notes

- Model weights, database, keys, etc. are **not included in the git repository** (see .gitignore)
- CPU servers can perform inference; training is theoretically possible but very slow — GPU training followed by model upload is recommended
- CPU inference: use `pm`/`harvest` for the F0 predictor (`crepe` is several times slower); lower shallow diffusion `k_step` as needed (30~100)
- Training is an **optional module**, disabled by default and admin-only; enable as needed
- fairseq compatibility requires pip 24.0; librosa 0.10.1 is used for compatibility with newer numpy/torch