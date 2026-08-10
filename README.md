# So-VITS-SVC 服务器独立部署项目

独立部署的 so-vits-svc **推理 + 训练** Web 服务器，支持多用户登录、模型管理、
推理配置、任务队列、一键部署。

**中文 · [English](README_en.md)**

##  许可证

本项目是 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
（**AGPL-3.0**）的派生作品，整体以 **AGPL-3.0** 协议发布（见 LICENSE 与 NOTICE）。

- 保留上游 LICENSE 与版权归属，修改过的上游文件已在 NOTICE 中声明
- 完整对应源码在本仓库公开，满足 AGPL-3.0 对网络服务提供源码的要求
- 使用/二次分发本仓库时须遵守 AGPL-3.0：保留版权与修改声明、以相同协议分发

## 功能

-  **推理服务**：上传模型 + 音频 → SVC 换声；浅扩散（k_step）修音；推理页直接调全部参数（F0/增强/检索/变调），无需跳配置页
-  **特征检索**：训练完自动建 faiss 索引，推理时 `cluster_ratio` 检索混合，音色更稳
-  **训练服务**：数据集 zip → 队列训练；**续训到指定步数**（主模型/扩散均可，任务列表一键续训）
-  **多架构支持**：SoVITS v1 / RVC 轻量直连 / RVC-Flow（轻量 TransformerFlow，A1/A2 可切换），训练页选择
-  **统一流（NF+FM 混合推理）**：rvc-flow A2 可开启统一流，共享 FFT 骨干同时承载 NF 与 FM，推理可选 NF / FM / Hybrid 三模式；Hybrid 用 NF 快速定位 + FM 少步精修，质量接近纯 FM、速度接近纯 NF（详见 [docs/unified_flow.md](docs/unified_flow.md)）
-  **中途停止测试**：训练中随时停止，自动保存当前 checkpoint 并注册为模型，马上推理试听
-  **训练监控**：实时进度、G/D Loss 曲线（扩散单独曲线）、**验证集 loss**（判断过拟合）、近期速度 ETA
-  **异常邮件确认**：判别器压制生成器 / Loss NaN 时发邮件，附"继续/停止"确认链接
-  **精确清理**：按单个训练任务/模型/数据集清理，防误伤检查
-  **文件管理**：分页浏览、批量删除、自由下载模型/配置/结果/数据集
-  **预训练页**：网页上传/管理 ContentVec、NSF-HiFiGAN、训练底模 G_0/D_0 等
-  **邮件通知**：训练完成/进度/异常、推理完成自动发邮件（SMTP 可配置）
-  **性能**：推理模型 LRU 缓存（同模型连续推理零加载开销）、GPU torch.compile
-  **ONNX 加速**：主生成器可导出 ONNX，推理自动走 onnxruntime（动态 shape），失败自动回退 PyTorch
-  **任务控制**：训练/推理任务均可随时停止（推理保留模型缓存，checkpoint 自动保存）
-  **快速恢复**：训练停止后一键继续上次训练（TEMP 快照，可载入全参数修改）
-  **运维**：设置页一键 git pull + 优雅重启；训练页参数预设
-  安全：CSRF 防护、登录限速、路径穿越防护、会话密钥持久化、checkpoint 架构校验

## 模型架构

在原版 so-vits-svc（VITS 结构）基础上，本项目新增了两套自研轻量架构，训练页可直接选择：

| 架构 | 结构 | 参数量 | 特点 |
|------|------|--------|------|
| `sovits-v1` | TextEncoder + Flow + enc_q（原版 VITS） | ~52M | 兼容性最好，与社区模型互通 |
| `rvc` | 特征直连解码器（无 TextEncoder / Flow） | ~15.5M | 训练更稳更快，音色保持好，适合小数据 |
| `rvc-flow` | 轻量 TransformerFlow（A1 / A2），可开统一流 | ~16M | 音质上限更高，需要更多数据支撑 |

三套架构共享 ContentVec 特征提取器与 NSF-HiFiGAN 解码器，差异只在「特征→解码器」之间的变换路径，因此同一份预训练底模（G_0/D_0）的解码器部分可跨架构复用。

**RVC 轻量直连（`arch: "rvc"`）**

去掉 TextEncoder 和 Flow，ContentVec 特征经单层投影直接进入 NSF-HiFiGAN 解码器，f0 由解码器谐波源注入。参数量约为 v1 的三分之一，训练更稳定、收敛更快，也避免了 flow 在小数据集上 KL 不稳定导致的问题。适合数据量小（<30 分钟）或追求快速出模型的场景。

**RVC-Flow（`arch: "rvc-flow"`）**

在直连基础上加入轻量 TransformerFlow 增强特征表达，两种 flow 模式可切换：

- `A1 特征先验流`（`flow_mode: "a1"`）：flow 正向变换内容特征 `c`，KL 约束到**固定**标准正态先验 N(0,1)；训练与推理路径一致（都走 flow 正向），无后验编码器、无先验采样
- `A2 后验流`（`flow_mode: "a2"`，默认）：极小 enc_q（1 层 WN）从频谱提供后验 `z_q`，flow 做先验↔后验对齐，训练更稳

A2 用一个极轻量的 enc_q（1 层 WN）从输入频谱提取后验隐变量 `z_q`，flow 负责把标准正态先验对齐到 `z_q` 的分布。相比 A1 把整个特征空间硬约束到正态，A2 只让 flow 学一个「先验↔后验」的映射，KL 项更稳，小数据集也不易崩。

**A1 特征先验流（详细）**

设计目标是**解耦音色与发音**：A2 的 `z_q` 来自目标说话人频谱，同时耦合了音色和发音习惯，换音色时容易把目标说话人的咬字习惯一起带过来。A1 让 `z` 完全由源音频的内容特征 `c`（ContentVec 输出）决定，flow 只做 `c → z_p` 的正向编码，decoder 再从 `z_p` 还原频谱——发音信息来自源，音色信息来自 speaker embedding，互不污染。

- **训练路径**：`x = pre(c) + emb_uv + vol` → `z_p = flow(x, x_mask, g=g)` → `dec(z_p_slice, g=g, f0=pitch_slice)`，全程**无 enc_q、无先验采样**，训练即推理。
- **推理路径**：与训练完全一致，`z_p = flow(x)` → `dec(z_p)`，确定性正向变换（`noise_scale` 不影响 A1，因为无先验采样步骤）。
- **固定先验 N(0,1)**：`m_p=0, logs_p=0`，KL 项简化为 `KL = -0.5 + 0.5 * ||z_p||²`，仅约束 flow 输出方差≈1 防止漂移。早期版本曾用 `prior_proj(x)` 学习先验均值/方差，但因 flow 和 prior_proj 共享输入 `x` 会**串谋**（两者一起把 KL 推向 -∞，实测 -262），已废弃并删除 `prior_proj`。
- **`c_kl` 调整**：A1 的 KL 项数值范围与 A2 不同（A2 KL≈正值，A1 KL 可为负），默认 `c_kl=1.0` 会让 KL 损失主导并压制 mel 重建损失。实测 A1 推荐 `c_kl=0.1`，让 mel loss 成为主导项，KL 仅作轻度正则。
- **不支持统一流/Hybrid**：统一流的 FM 训练需要 `enc_q` 提供 `z_q` 作为 FM 目标（`x_1 = z_q.detach()`），A1 无 `enc_q` 无法提供 FM 目标。代码层面 A1 强制使用 `TransformerCouplingBlock`（非 `GeneralizedFlow`），`infer_hybrid` 入口有断言阻止误调用。A1 推理只能走纯 NF 正向（`infer()`），但因其正向变换本身就是确定性的，无需 Hybrid 精修。

### 统一流（Unified Flow）

A2 模式下可开启「**统一流**」（`use_unified_flow: true`），这是本项目的主打特性：用**同一组 FFT 骨干**同时承载 Normalizing Flow（NF，可逆）与 Flow Matching（FM，速度场）两条路径，双输出头共享骨干参数。完整使用说明见 [docs/unified_flow.md](docs/unified_flow.md)。

**设计动机**

- 纯 NF 推理快（一次逆变换），但单步可逆变换的表达能力有限，高频细节常偏弱。
- 纯 FM（多步欧拉积分从噪声到数据）质量上限高，但 32 步积分推理慢。
- 二者用同一类 flow 骨干（FFT/CouplingLayer），分别训练时参数完全不复用，浪费容量。
- 统一流让两者**共享骨干、各取所长**：NF 给出快速起点，FM 用少量步数精修，质量接近纯 FM、速度接近纯 NF。

**结构**

```
                ┌─────────────────────────────┐
   先验 z_p ───▶│                             │── head_nf ──▶ 可逆变换 s/t  (NF 路径，channel-split coupling)
                │   共享 FFT 骨干              │
   x_t (插值) ─▶│   (n_layers 个 CouplingLayer)│── head_fm ──▶ 速度场 v      (FM 路径，预测 v≈x_1-x_0)
                └─────────────────────────────┘
                       ▲ 共享参数
```

每个 `GeneralizedCouplingLayer` 内部：

- **共享骨干**：多层 FFT（WN + attention）提取特征，NF 与 FM 共用，参数量不翻倍。
- **NF 头（`head_nf`）**：输出 channel-split coupling 的 `s`（缩放）/ `t`（平移），保证可逆，用于先验↔后验的精确变换。
- **FM 头（`head_fm`）**：输出速度场 `v`，引导 `x_0`（噪声/起点）→ `x_1`（数据）的轨迹；不可逆，但可多步欧拉积分逼近高质量样本。

**训练**

`forward` 同时计算 NF loss（KL + 重构，与原 A2 一致）和 FM loss：

- `x_1 = z_q.detach()`（真实后验，切断到 enc_q 的梯度，避免 FM 反传干扰 NF）
- `x_0 = NF逆变换(先验采样).detach()`（**与推理起点一致**，关键一致性保证）
- 在 `[x_0, x_1]` 间线性插值 `x_t = (1-t)·x_0 + t·x_1`，FM 头预测速度 `v`，MSE 拟合真实速度 `u_t = x_1 - x_0`
- `loss_flow_match` 按 `c_fm`（默认 0.5）加权并入 `loss_gen_all` 统一反传

两个关键工程修复（早期版本未做，会导致 Hybrid 推理听不清文字）：

1. **训练/推理起点一致**：FM 训练的 `x_0` 必须用「NF 逆变换输出」而非纯噪声，否则推理时 FM 从 NF 输出起步会与训练分布不符，速度场算错。
2. **`head_fm` 零初始化**：FM 头权重和偏置置零，初期 `v=0`（恒等映射），Hybrid ≈ NF；随训练渐进学习速度场，不会因随机初始化的大速度场破坏 NF 输出。

训练日志会额外打印 `FlowMatch Loss`，前端训练页会画出第三条黄色 FM loss 曲线。

**推理（三种模式共享同一份权重）**

| 模式 | 流程 | 步数 | 速度 | 质量 | 适用 |
|------|------|------|------|------|------|
| `nf` | 先验采样 → NF 逆变换 → decoder | 1 | 最快 | 高频略弱 | 追求速度 / 实时 |
| `fm` | 纯噪声 → FM 32 步欧拉积分 → decoder | 32 | 最慢 | 上限最高 | 离线精修 |
| `hybrid`（推荐） | NF 逆变换给起点 → FM 4 步精修 → decoder | 1+4 | 接近 NF | 接近 FM | **默认** |

**性能数据**（G_6000.pth，详见 [docs/unified_flow.md](docs/unified_flow.md) 第 5、6 节）

| 模式 | 耗时(s) | HF ratio | 说明 |
|------|---------|----------|------|
| NF | 2.14 | 0.260 | 高频略不足 |
| Hybrid(4) | 0.15 | 0.272 | 质量接近 FM，速度快 4.2× |
| FM(32) | 0.63 | 0.275 | 质量上限 |
| 原始音频 | — | 0.287~0.323 | baseline |

步数扫描结论：`hybrid_steps=4` 是质量/速度平衡点；2 步略快但质量有可测下降，8 步以上收益递减。`c_fm=0.5` 是经验最佳（0.1 时 FM 几乎无梯度）。

**ONNX 导出**：统一流模型支持导出，按 config 的 `hybrid_steps` 展开欧拉积分循环，输出与 PyTorch 一致（SNR ~30dB）。

**checkpoint 架构校验**

checkpoint 保存时记录架构标签（`arch` + `flow_mode` + `use_unified_flow`），加载时校验：架构不匹配直接报错，避免把 rvc 的权重静默加载成 rvc-flow（或反之）导致模型损坏。v1 底模（G_0/D_0）作为初始化权重仍可复用到 rvc 系列（解码器部分通用）。

## 快速开始

### Windows 本机（一键安装）

```bat
install.bat      :: 建 venv + 装 CPU/CUDA torch + 依赖（选 1=CPU / 2=CUDA）
start.bat        :: 启动服务，访问 http://localhost:5000
```

### Linux 部署（全平台支持，全走国内镜像）

本项目提供了一个整合的一键部署脚本，支持 **NVIDIA GPU、AMD GPU 和纯 CPU** 环境，自动切换 PyTorch 版本。

```bash
# 下载脚本后赋予执行权限
chmod +x deploy_linux.sh

# 运行部署脚本（需 root 权限）
sudo bash deploy_linux.sh

# （可选）如果你的网络下载模型慢，可以加上 --skip-models 跳过模型下载，
# 部署完成后再到网页后台手动上传预训练模型。
# sudo bash deploy_linux.sh --skip-models
```
#### LinuxGPU 部署提醒：

运行 sudo bash deploy_linux.sh 之前，请确保你的显卡驱动已经安装好。

>NVIDIA 用户：请确保能通过 nvidia-smi 看到显卡信息。

>AMD 用户：请确保已安装 AMD ROCm 驱动，并将当前用户加入 video 和 render 组，运行 rocm-smi 能正常看到显卡信息。

脚本会自动完成的工作：脚本会**自动安装 ffmpeg、libsndfile、cmake 等系统环境依赖**，并对接国内清华镜像源安装 Miniconda 和 Python 3.9 环境，用户**只需要提前搞定显卡驱动**即可，**无需手动配置其他复杂环境**。

> **Windows用户请特别注意：**
> 
> 项目依赖中的 onnxsim 在安装时需要通过源码编译，强制依赖系统级构建工具。
>
> 请前往 CMake 官网 下载并安装 Windows 版本的 CMake（下载 .msi 安装包）。
> 
> 在安装向导中，务必勾选 Add CMake to the system PATH for all users（将 CMake 添加到系统环境变量）。
> 
> 安装完成后，重启命令行终端（CMD / PowerShell），输入 cmake --version 确认安装成功，最后再执行安装命令。

> **AMD 显卡用户请注意：**
> 1. **Linux 用户**：AMD 显卡使用了 AMD ROCm 版本的 PyTorch 进行加速，目前只有 RX 5000、6000、7000 系列以及最新的 9000 系列才能正常使用 ROCm 的 GPU 加速，请确认你的显卡型号和驱动。
> 2. **Windows 用户**：极其抱歉，AMD ROCm 版的 PyTorch 官方目前**仅在 Linux 系统下提供支持**。如果你是 Windows 用户，建议直接使用 **CPU 版本**（推理完全够用），或者使用 WSL 部署 Linux 子系统来使用 ROCm 加速。
> 3. **一键脚本适配**：如果你使用本项目的 `install.bat` 进行安装，当你在 Windows 下选择 AMD 选项时，脚本会自动为你回退切换到 CPU 版本，无需担心报错。
> 
> **对AMD用户的提示：**
> 开发者所用设备为 NVIDIA 显卡，AMD 显卡目前没有条件进行测试。如果你在使用中发现项目在 AMD 上会出现问题，欢迎提交 Issue，我会尽量协助解决。

## 配置

| 项 | 环境变量 | 默认 |
|----|---------|------|
| 会话密钥 | SECRET_KEY | 自动生成并持久化到 `server/secret_key.txt` |
| 数据库 | DATABASE_URL | server/data.db |
| 服务端口 | PORT | 5000 |
| 推理超时 | INFERENCE_TASK_TIMEOUT | 21600（秒，6 小时） |

## 预训练模型

推理**必需** ContentVec + NSF-HiFiGAN；训练底模 G_0/D_0 可选但推荐。

| 文件 | 大小 | 获取 |
|------|------|------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | 部署脚本自动下载，或网页"预训练"页上传 |
| `pretrain/nsf_hifigan/model` + `config.json` | ~54MB | 同上 |
| `pretrain/G_0.pth` + `pretrain/D_0.pth` | ~400MB | 手动上传（推荐） |
| `pretrain/rmvpe.pt` 等编码器 | 可选 | 网页"预训练"页上传 |

> 大文件建议 scp 直接放到 `pretrain/` 目录，网页会自动识别。

## 使用流程

**推理：**
1. 模型管理 → 上传 G_*.pth + config.json（可附带扩散模型 + diffusion.yaml）
2. 创建推理配置（f0 预测器、noise_scale、k_step 等）
3. 推理页选配置 + 传音频 → 提交（可在推理页临时覆盖参数）→ 任务列表看进度/停止/下载结果

**训练：**
1. 训练页 → 上传数据集 zip（自动过滤 <2 秒短片段）
2. 设置参数（总步数、auto_stop、编码器、底模自动加载）
3. 训练页看 train/验证 loss → 满意就"停止"（自动保存 checkpoint）
4. 任务列表"续训"可继续，或直接测试当前模型

**扩散：** 主模型训好后，任务列表点"训扩散"直接进入扩散训练（复用数据/特征），
完成后把扩散模型挂到主模型上，k_step 100~300 推理。

## 训练参数说明

训练页可配置的参数及其含义、建议值：

| 参数 | 含义 | 建议 |
|------|------|------|
| `total_steps` | 总训练步数（续训时为目标总步数） | 小数据集 3000~10000；大数据集按需加 |
| `batch_size` | 每批样本数 | GPU 4~8；CPU 1~4（显存/内存小就调低） |
| `keep_ckpts` | 保留最近几个 checkpoint | 3（磁盘紧张可 1~2） |
| `speech_encoder` | 特征编码器 | `vec768l12`（推荐）；数据量大可试 WavLM/Whisper |
| `f0_predictor` | F0 提取器 | `harvest` 最稳（训练用）；`dio` 快但质量差 |
| `learning_rate` | 学习率 | 0.0001（一般不用动） |
| `lr_decay` | 学习率衰减 | 0.999~0.999875 |
| `segment_size` | 训练切片长度（帧） | 10240（内存小调低） |
| `auto_stop` | loss 连续 N 步无改善自动停 | 200（0=关闭） |
| `arch` | 模型架构 | v1 / rvc / rvc-flow（见"模型架构"章节） |
| `d_lr_scale` | 判别器 lr 缩放（前 1000 步） | rvc 系列建议 0.5，v1 用 1.0 |
| `flow_mode` | rvc-flow 模式 | `a2`（推荐） |
| `use_unified_flow` | 启用统一流（NF+FM 共享骨干，A2 专用） | false（A2 用户按需开 true） |
| `c_fm` | FM loss 权重（仅统一流） | 0.5（过小如 0.1 会让 FM 学不动） |
| `c_mel` | Mel 重建 loss 权重 | 45（一般不用动） |
| `c_kl` | KL loss 权重 | A2 用 1.0；**A1 推荐 0.1**（A1 的 KL 可为负，1.0 会主导并压制 mel） |
| `hybrid_steps` | Hybrid 推理 FM 精修步数 | 4（2 快但质量略降，8+ 收益递减） |
| `ema_decay` / `ema_interval` | EMA 权重衰减 / 更新间隔（推理用 EMA 更稳） | 0.999 / 100 |

**扩散训练参数**（训扩散时）：`diff_epochs`（默认 100000，靠 loss/停止控制）、`diff_timesteps`（1000）、`diff_kstep`（浅扩散最大步，0=全量）、`diff_layers`/`diff_chans`/`diff_hidden`（网络容量，内存小就调小）、`diff_lr`、`diff_decay_step`、`diff_gamma`、`diff_amp`（fp32/fp16/bf16）。

## 推理参数说明

推理页可直接覆盖配置默认值（参数默认折叠）：

| 参数 | 含义 | 建议 |
|------|------|------|
| `f0_predictor` | F0 提取器 | `pm`/`harvest`（CPU 快）；`crepe` 最准但慢数倍 |
| `k_step` | 浅扩散步数 | 挂了扩散模型时 100~300；无扩散填 0 |
| `cluster_ratio` | 特征检索混合比例 | 0.2~0.5（需模型挂检索索引，无索引自动禁用） |
| `vc_transform` | 变调（半音） | 0 |
| `slice_db` | 切片阈值（dB） | -40；音频切太碎时调 -30~-35 |
| `noise_scale` | 生成噪声 | 0.4；声音毛糙可调 0.25 |
| `pad_seconds` | 段间填充 | 0.5 |
| `auto_f0` | 自动预测 F0 | 一般关闭 |
| `enhancer` | NSF 增强 | 可选 |
| `second_encoding` | 二次编码 | 一般关闭 |
| `loudness_envelope` | 响度包络 | 0~1 |
| `hybrid_mode` | 统一流推理模式（仅统一流模型生效） | `auto`（统一流→hybrid，否则→nf）；可选 `nf`/`fm`/`hybrid` |
| `output_format` | 输出格式 | wav / mp3 / flac |

**ONNX 导出**（可选，提升推理速度）：

```bash
python onnx_export_generator.py <模型.pth> <config.json> <输出.onnx>
```

导出后把 `.onnx` 放到模型同目录（`模型名.pth.onnx`），推理时自动用 onnxruntime 跑生成器；文件缺失或加载失败时自动回退 PyTorch，不影响原有推理。统一流模型同样支持导出（按 config 的 `hybrid_steps` 展开 FM 欧拉积分循环），验证输出与 PyTorch 一致。

## 部署与运维

**Linux 服务管理（systemd）**

```bash
systemctl status ssvc          # 查看状态
systemctl restart ssvc         # 重启（更新代码后必须）
journalctl -u ssvc -n 100      # 查看日志（推理/训练报错在这里）
```

**更新代码**

- 设置页"系统更新"按钮：git pull + 等任务结束自动重启
- 手动：`cd /opt/so-vits-svc && git pull gitee master && systemctl restart ssvc`

**环境变量**

| 变量 | 作用 | 默认 |
|------|------|------|
| `PORT` | 服务端口 | 5000 |
| `SECRET_KEY` | 会话密钥（自动持久化） | 自动生成 |
| `DATABASE_URL` | 数据库路径 | server/data.db |
| `INFERENCE_TASK_TIMEOUT` | 推理超时（秒） | 21600 |
| `TRAIN_TIMEOUT` | 训练墙钟超时（秒，0=不限） | 0 |
| `INFERENCE_MODEL_CACHE` | 推理模型缓存数量 | 3（内存小调 1） |
| `SSVC_COMPILE` | GPU torch.compile（0 关闭） | 1 |
| `SSVC_SERVER_URL` | 邮件/页面链接地址 | 自动探测 |

**数据安全**：模型权重、数据库、密钥、日志均不入 git（见 .gitignore）。更新代码不会覆盖 `uploads/`、`pretrain/`、`data.db`。

## 常见问题（FAQ）

**训练出的声音有电音/金属音/沙哑？**
- F0 预测器问题：训练用 `dio` 容易导致沙哑，改用 `harvest` 重训
- 训练不足或过拟合：小数据集训过头反而变差，找到合适步数（看验证 mel）
- 浅扩散修音：训扩散模型挂上，推理 `k_step` 100~300，能压掉大部分高频毛刺
- 推理 `noise_scale` 调低到 0.25 也能缓解

**推理很慢？**
- CPU 上 `crepe` F0 极慢，换 `pm`/`harvest`
- `k_step` 越高越慢，按需调低
- 长音频切成 1~2 分钟片段分别推理

**报"模型架构不匹配"？**
checkpoint 和配置的架构不一致（v1/rvc/rvc-flow 混用）。用匹配的配置，或重新训练。

**推理任务一直 running / 进度不动？**
- 模型加载阶段（CPU 上 1~2 分钟）无进度是正常的
- 任务可点"停止"后重新提交
- 服务重启后排队任务会自动恢复

**服务器内存小（OOM）？**
调低 `INFERENCE_MODEL_CACHE=1`，推理配置 `cluster_ratio=0`，训练 `batch_size` 调低。

**cluster 相关报错？**
训练自动生成的 `*_cluster.pth` 是 faiss 检索索引，加载器会自动识别；手动上传 kmeans 模型也兼容。

**统一流 Hybrid 推理听不清文字 / 全是噪声？**
- 确认 checkpoint 是用「FM 一致性修复 + head_fm 零初始化」之后的版本
- 早期 checkpoint（修复前）FM 从纯噪声起点训练，与推理起点不一致，会破坏语音
- 详见 [docs/unified_flow.md](docs/unified_flow.md) FAQ 章节

## 目录结构

```
server/
├── server/              ← Flask 服务（app.py、模板、worker）
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/
├── configs_template/    ← 训练配置模板
├── docs/                ← 统一流等设计文档
├── pretrain/            ← 预训练模型（不在 git 中）
├── train.py train_diff.py preprocess_*.py
├── deploy_linux.sh      ← Linux 一键部署（CUDA / ROCm / CPU 三选一，conda）
├── install.bat / start.bat / start.ps1  ← Windows 安装/启动
└── requirements.txt LICENSE
```

## 注意事项

- 模型权重、数据库、密钥等**不进入 git 仓库**（见 .gitignore）
- CPU 服务器可推理；训练理论上可行但非常慢，建议 GPU 训练后上传模型
- CPU 推理性能提示：F0 预测器建议用 `pm`/`harvest`（`crepe` 慢数倍），浅扩散 `k_step` 按需调低（30~100）
- 国内网络部署：apt/conda/pip 全部走清华/阿里云镜像，torch 用 `--no-deps` 跳过 nvidia 包
- fairseq 兼容需要 pip 24.0（脚本自动处理）；librosa 用 0.10.1 兼容新版 numpy/torch
- 训练 DataLoader 默认 `num_workers=2`（内存充足自动升到 4，<8GB 降为 0）
