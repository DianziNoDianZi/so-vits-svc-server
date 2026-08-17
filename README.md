# So-VITS-SVC 多用户推理服务器

独立部署的 so-vits-svc **多用户推理 Web 服务器**，可选启用**训练**模块（仅管理员）。
用户注册后可使用平台发布的官方模型或上传私有模型，提交音频异步推理并下载结果；
管理员负责审核模型、分配资源配额、管理队列与公告。

**中文 · [English](README_en.md)**

## 许可证

本项目是 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
（**AGPL-3.0**）的派生作品，整体以 **AGPL-3.0** 协议发布（见 LICENSE 与 NOTICE）。

- 保留上游 LICENSE 与版权归属，修改过的上游文件已在 NOTICE 中声明
- 完整对应源码在本仓库公开，满足 AGPL-3.0 对网络服务提供源码的要求
- 使用/二次分发本仓库时须遵守 AGPL-3.0：保留版权与修改声明、以相同协议分发

## 功能

- **多用户**：开放注册（后台可关）、角色（admin/user）、登录限速、CSRF
- **模型管理**：用户上传私有模型需管理员审核；**申请公开**（审核通过设为官方共享）；管理员可上/下架官方模型、下载模型文件核对、自动校验配置合法性；拒绝的模型直接删除
- **资源配额**：管理员为每个用户设置排队/运行任务数、单音频时长、每日可提交任务数、CPU 核心数、私有模型数、结果保留天数、优先级
- **公平调度**：任务按用户轮转 + 优先级排队，单 GPU 串行；执行器失联自动回收；管理员可暂停/恢复新任务调度
- **批量推理**：一次上传多个音频生成多个任务；任务列表支持筛选/搜索/分页、任务详情页、排队位置可见
- **服务器统一邮件**：管理员配置一次 SMTP，用户设置**结果接收邮箱**；推理完成、资源紧张、公告群发自动发邮件（限频防刷）
- **资源监控**：CPU/内存/磁盘/显存紧张时所有页面顶部横幅提示 + 邮件告警
- **公告**：管理员发布/置顶/群发邮件，用户在首页与公告页查看
- **结果对象级下载**：按任务归属鉴权下载，结果默认保留 7 天（可配置）后自动清理
- **推理性能**：常驻 daemon + LRU 模型缓存、ONNX 加速（失败回退 PyTorch）
- **实时变声（WebSocket）**：客户端麦克风 → `ws://:5001` → 服务器实时变声 → 回传播放；44.1kHz，`chunk_seconds` 可配（0.1~2.0s），API Key 鉴权；准实时（CPU 约 1~2s）
- **训练（可选模块，仅管理员）**：后台开关控制；SoVITS / SoVITS+扩散、续训、停止、注册 checkpoint 为推理模型、进度轮询
- **管理员后台**：总览（健康/暂停/待审核/磁盘）、用户配额、模型审核、全局任务队列（可停止/删除/导出 CSV）、存储与孤儿文件、公告、站点设置、训练开关
- **长线运营**：
  - **数据库自动备份**：SQLite 在线备份（WAL 安全），定时 + 保留份数，管理员手动触发/下载
  - **结构化日志**：`server/logs/app.log` 按大小轮转，带时间/级别/请求信息
  - **健康检查**：`/healthz` 返回 db/daemon/队列/磁盘/暂停状态 JSON；`/status` 系统状态页（CPU/内存/GPU/任务统计/最近失败/最近备份）
  - **审计日志**：管理员关键操作（配额/审核/任务/设置/更新/公告/备份/API Key）全程留痕，后台可查
  - **邀请码注册**：注册模式可设关闭/选填/必填，管理员批量生成/撤销，控制开放节奏
  - **对外 REST API**：用户/管理员通用，`X-API-Key` 鉴权，可提交推理/查状态/下载结果/查系统状态；`/api/v1/docs` 在线文档
  - **通用限流**：推理/下载/上传按用户限次（次/分钟），阈值后台可调
- 安全：CSRF、登录/注册限速、通用限流、路径穿越防护、会话密钥持久化、checkpoint 架构校验、不可信权重 `weights_only` 优先加载

## 模型架构

在原版 so-vits-svc（VITS 结构）基础上，本项目新增两套自研轻量架构，训练（若启用）页可直接选择：

| 架构 | 结构 | 参数量 | 特点 |
|------|------|--------|------|
| `sovits-v1` | TextEncoder + Flow + enc_q（原版 VITS） | ~52M | 兼容性最好，与社区模型互通 |
| `rvc` | 特征直连解码器（无 TextEncoder / Flow） | ~15.5M | 训练更稳更快，音色保持好，适合小数据 |
| `rvc-flow` | 轻量 TransformerFlow（A1 / A2），可开统一流 | ~16M | 音质上限更高，需要更多数据支撑 |

三套架构共享 ContentVec 特征提取器与 NSF-HiFiGAN 解码器，差异只在「特征→解码器」之间的变换路径，因此同一份预训练底模（G_0/D_0）的解码器部分可跨架构复用。

### 各架构数据集量与训练轮数建议

| 架构 | 建议数据集量 | 建议总步数（`total_steps`） | 说明 |
|------|-------------|--------------------------|------|
| `sovits-v1` | 30 分钟 ~ 2 小时 | 10000 ~ 50000 | 数据越多越好；低于 30 分钟易欠拟合，音色发飘 |
| `rvc` | 5 ~ 30 分钟 | 5000 ~ 20000 | 小数据收敛快、音色稳定，是短数据的首选；超过 1 小时收益递减 |
| `rvc-flow` | 1 ~ 3 小时 | 20000 ~ 80000 | 需要足够数据撑起 flow，数据太少会不稳定或听不清；数据越多音质上限越高 |

- **A2 后验流（默认）** 比 A1 在小数据上更稳，建议 <30 分钟数据用 A2。
- **A1 特征先验流** 音色解耦更彻底，但建议数据 ≥1 小时，且把 `c_kl` 调到 `0.1`。
- **统一流（A2 + `use_unified_flow`）** 容量更大，数据 <1 小时不建议开。
- 上述 `total_steps` 是"从零训练"的参考；**续训**时填目标总步数（含已训步数）。观测页面 loss：mel 降到稳定平台、主观试听满意即可早停，不必跑满。

**RVC 轻量直连（`arch: "rvc"`）**

去掉 TextEncoder 和 Flow，ContentVec 特征经单层投影直接进入 NSF-HiFiGAN 解码器，f0 由解码器谐波源注入。参数量约为 v1 的三分之一，训练更稳定、收敛更快，也避免了 flow 在小数据集上 KL 不稳定导致的问题。适合数据量小（<30 分钟）或追求快速出模型的场景。

**RVC-Flow（`arch: "rvc-flow"`）**

在直连基础上加入轻量 TransformerFlow 增强特征表达，两种 flow 模式可切换：

- `A1 特征先验流`（`flow_mode: "a1"`）：flow 正向变换内容特征 `c`，KL 约束到**固定**标准正态先验 N(0,1)；训练与推理路径一致（都走 flow 正向），无后验编码器、无先验采样
- `A2 后验流`（`flow_mode: "a2"`，默认）：极小 enc_q（1 层 WN）从频谱提供后验 `z_q`，flow 做先验↔后验对齐，训练更稳

**A1 特征先验流（详细）**

设计目标是**解耦音色与发音**：A1 让 `z` 完全由源音频的内容特征 `c`（ContentVec 输出）决定，flow 只做 `c → z_p` 的正向编码，decoder 再从 `z_p` 还原频谱——发音信息来自源，音色信息来自 speaker embedding，互不污染。

- **训练路径**：`x = pre(c) + emb_uv + vol` → `z_p = flow(x, x_mask, g=g)` → `dec(z_p_slice, g=g, f0=pitch_slice)`，全程**无 enc_q、无先验采样**，训练即推理。
- **推理路径**：与训练完全一致，`z_p = flow(x)` → `dec(z_p)`，确定性正向变换（`noise_scale` 不影响 A1，因为无先验采样步骤）。
- **固定先验 N(0,1)**：`m_p=0, logs_p=0`，KL 项简化为 `KL = -0.5 + 0.5 * ||z_p||²`，仅约束 flow 输出方差≈1 防止漂移。早期版本曾用 `prior_proj(x)` 学习先验均值/方差，但因 flow 和 prior_proj 共享输入 `x` 会**串谋**（两者一起把 KL 推向 -∞，实测 -262），已废弃并删除。
- **`c_kl` 调整**：A1 的 KL 项数值范围与 A2 不同（A2 KL≈正值，A1 KL 可为负），默认 `c_kl=1.0` 会让 KL 损失主导并压制 mel 重建损失。实测 A1 推荐 `c_kl=0.1`，让 mel loss 成为主导项，KL 仅作轻度正则。
- **不支持统一流/Hybrid**：统一流的 FM 训练需要 `enc_q` 提供 `z_q` 作为 FM 目标，A1 无 `enc_q` 无法提供。代码层面 A1 强制使用 `TransformerCouplingBlock`（非 `GeneralizedFlow`），`infer_hybrid` 入口有断言阻止误调用。

### 统一流（Unified Flow）

A2 模式下可开启「**统一流**」（`use_unified_flow: true`）：用**同一组 FFT 骨干**同时承载 Normalizing Flow（NF，可逆）与 Flow Matching（FM，速度场）两条路径，双输出头共享骨干参数。完整说明见 [docs/unified_flow.md](docs/unified_flow.md)。

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

`forward` 同时计算 NF loss（KL + 重构）和 FM loss：

- `x_1 = z_q.detach()`（真实后验，切断到 enc_q 的梯度，避免 FM 反传干扰 NF）
- `x_0 = NF逆变换(先验采样).detach()`（**与推理起点一致**，关键一致性保证）
- 在 `[x_0, x_1]` 间线性插值 `x_t = (1-t)·x_0 + t·x_1`，FM 头预测速度 `v`，MSE 拟合真实速度 `u_t = x_1 - x_0`
- `loss_flow_match` 按 `c_fm`（默认 0.5）加权并入 `loss_gen_all` 统一反传

两个关键工程修复（早期版本未做，会导致 Hybrid 推理听不清文字）：

1. **训练/推理起点一致**：FM 训练的 `x_0` 必须用「NF 逆变换输出」而非纯噪声，否则推理时 FM 从 NF 输出起步会与训练分布不符，速度场算错。
2. **`head_fm` 零初始化**：FM 头权重和偏置置零，初期 `v=0`（恒等映射），Hybrid ≈ NF；随训练渐进学习速度场，不会因随机初始化的大速度场破坏 NF 输出。

**推理（三种模式共享同一份权重）**

| 模式 | 流程 | 步数 | 速度 | 质量 | 适用 |
|------|------|------|------|------|------|
| `nf` | 先验采样 → NF 逆变换 → decoder | 1 | 最快 | 高频略弱 | 追求速度 / 实时 |
| `fm` | 纯噪声 → FM 32 步欧拉积分 → decoder | 32 | 最慢 | 上限最高 | 离线精修 |
| `hybrid`（推荐） | NF 逆变换给起点 → FM 4 步精修 → decoder | 1+4 | 接近 NF | 接近 FM | **默认** |

**checkpoint 架构校验**

checkpoint 保存时记录架构标签（`arch` + `flow_mode` + `use_unified_flow`），加载时校验：架构不匹配直接报错，避免把 rvc 的权重静默加载成 rvc-flow（或反之）导致模型损坏。v1 底模（G_0/D_0）作为初始化权重仍可复用到 rvc 系列（解码器部分通用）。

## 快速开始

### Windows 本机（一键安装）

```bat
install.bat      :: 建 venv + 装 CPU/CUDA torch + 依赖（选 1=CPU / 2=CUDA）
start.bat        :: 启动服务，访问 http://localhost:5000
```

首次启动会在控制台打印管理员初始密码（`admin` 账号），登录后强制修改。

### Linux 部署

```bash
sudo bash deploy_linux.sh   # NVIDIA GPU / AMD ROCm / CPU 自动适配，全走国内镜像
```

- 脚本自动安装 ffmpeg、libsndfile、cmake 等系统依赖，并配置 Miniconda/Python 3.9。
- 显卡驱动需提前装好（NVIDIA 用 `nvidia-smi` 确认；AMD 装 ROCm 并把用户加入 `video`/`render` 组）。
- 网络慢可加 `--skip-models` 跳过模型下载，部署后手动放 `pretrain/`。

## 配置（环境变量）

| 项 | 环境变量 | 默认 |
|----|---------|------|
| 会话密钥 | SECRET_KEY | 自动生成并持久化到 `server/secret_key.txt` |
| 数据库 | DATABASE_URL | server/data.db |
| 服务端口 | PORT | 5000 |
| 推理超时 | INFERENCE_TASK_TIMEOUT | 21600（秒，6 小时） |
| 推理模型缓存 | INFERENCE_MODEL_CACHE | 1（内存小调 1） |
| 推理切块秒数 | INFERENCE_CLIP_SECONDS | 15（长音频切成小块防 OOM，0=不切） |
| 资源告警阈值/限频 | RESOURCE_THRESHOLD / RESOURCE_EMAIL_INTERVAL | 90 / 3600 |
| 服务器 SMTP | SMTP_HOST/PORT/USER/PASS/MAIL_FROM | 空（也可在后台配置） |
| 邮件内访问地址 | SSVC_SERVER_URL | 自动探测 |
| 注册开关 | ALLOW_REGISTRATION | 1（也可在后台配置） |

## 预训练模型

推理**必需** ContentVec + NSF-HiFiGAN；训练底模 G_0/D_0 可选但推荐。

| 文件 | 大小 | 获取 |
|------|------|------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | 部署脚本自动下载，或手动放入 |
| `pretrain/nsf_hifigan/model` + `config.json` | ~54MB | 同上 |
| `pretrain/G_0.pth` + `pretrain/D_0.pth` | ~400MB | 手动放入（推荐） |
| `pretrain/rmvpe.pt` 等编码器 | 可选 | 手动放入 |

> 大文件建议 scp 直接放到 `pretrain/` 目录，服务启动时自动识别。

## 使用流程

**用户：**
1. 注册账号（可填接收邮箱与结果接收邮箱）
2. 模型页上传私有模型 → 等待管理员审核；或直接使用平台**官方模型**
3. 创建推理配置（可基于官方/自己的模型）
4. 推理页选配置 + 上传音频（可一次多选批量提交）→ 任务列表看进度/排队位置/停止/下载
5. 任务详情页看完整参数与错误；结果保留到期自动清理

**管理员：**
1. 后台“管理 ▾”→ 总览：待审核模型、健康状态（daemon/GPU/排队）、暂停/恢复调度、**数据库备份**（立即备份/下载）
2. **模型**：审核用户上传模型（通过 / 拒绝即删除），官方模型上/下架、下载核对、自动校验
3. **用户**：启用/禁用、设置每人配额（每日任务数、CPU 核心、排队/运行、私有模型、结果保留）
4. **邀请码**：生成/撤销邀请码，配合设置页“邀请码模式”（关闭/选填/必填）控制注册节奏
5. **任务**：全局队列，可停止/删除/导出 CSV
6. **公告**：发布、置顶、邮件群发
7. **设置**：注册开关、邀请码模式、站点默认配额、**限流阈值**（推理/下载/上传）、**备份间隔与保留份数**、SMTP、训练功能开关与 CPU 核心
8. **审计日志**：管理员关键操作全程留痕，按操作类型筛选
9. **训练**（若开启）：上传数据集训练 SoVITS，完成后注册为推理模型

**开发者（用户/管理员）**：设置页“开发者 API”生成 API Key，见下方 [REST API](#rest-api)。

## 训练可选模块（仅管理员）

默认关闭，管理员在 `设置 → 训练功能` 开启，或访问 `/train` 一键开启。

- **提交训练**：上传数据集 zip（自动过滤 <2 秒短片段）+ 说话人 + 参数（总步数/编码器/F0/架构/flow/统一流/扩散）
- **续训**：历史任务一键“续训”到指定步数
- **停止**：运行中可停止，自动保存当前 checkpoint
- **注册模型**：训练完成把 `G_*.pth` 注册为推理模型（自动挂聚类索引）
- **进度**：页面轮询显示阶段/百分比/step

## 训练参数说明（训练模块内）

| 参数 | 含义 | 建议 |
|------|------|------|
| `total_steps` | 总训练步数（续训时为目标总步数） | 见上文「各架构数据集量与训练轮数建议」 |
| `batch_size` | 每批样本数 | GPU 4~8；CPU 1~4 |
| `keep_ckpts` | 保留最近几个 checkpoint | 3 |
| `speech_encoder` | 特征编码器 | `vec768l12`（推荐） |
| `f0_predictor` | F0 提取器（训练用） | `harvest` 最稳；`dio` 快但差 |
| `arch` / `flow_mode` | 架构 / rvc-flow 模式 | v1 / rvc / rvc-flow；A2 默认 |
| `use_unified_flow` | 统一流（A2 专用） | false（按需开） |
| `c_fm` | FM loss 权重（统一流） | 0.5 |
| `c_mel` / `c_kl` | Mel / KL 权重 | 45；A1 用 0.1，A2 用 1.0 |

**扩散参数**（选 SoVITS+扩散 时）：`diff_epochs`、`diff_timesteps`、`diff_kstep`（浅扩散最大步，0=全量）、`diff_layers/chans/hidden`（容量）、`diff_lr`、`diff_amp`（fp32/fp16/bf16）。

## 推理参数说明

推理页可临时覆盖配置默认值（参数默认折叠）：

| 参数 | 含义 | 建议 |
|------|------|------|
| `f0_predictor` | F0 提取器 | `pm`/`harvest`（CPU 快）；`crepe` 最准但慢数倍 |
| `k_step` | 浅扩散步数 | 挂了扩散模型时 100~300；无扩散填 0 |
| `cluster_ratio` | 特征检索混合比例 | 0.2~0.5（需模型挂检索索引，无索引自动禁用） |
| `vc_transform` | 变调（半音） | 0 |
| `slice_db` | 切片阈值（dB） | -40；音频切太碎时调 -30~-35 |
| `noise_scale` | 生成噪声 | 0.4；声音毛糙可调 0.25 |
| `pad_seconds` | 段间填充 | 0.5 |
| `auto_f0` / `enhancer` / `second_encoding` | 自动 F0 / NSF 增强 / 二次编码 | 一般关闭 |
| `loudness_envelope` | 响度包络 | 0~1 |
| `hybrid_mode` | 统一流推理模式（仅统一流模型生效） | `auto`；可选 `nf`/`fm`/`hybrid` |
| `output_format` | 输出格式 | wav / mp3 / flac |

**ONNX 导出**（可选，提升推理速度）：模型编辑页可导出主生成器为 ONNX，导出后同目录自动用 onnxruntime 推理，失败自动回退 PyTorch。也可命令行：
```bash
python onnx_export_generator.py <模型.pth> <config.json> <输出.onnx>
```

## REST API

对外 REST 接口，用户和管理员通用（权限与账号一致）。在设置页“开发者 API”生成 Key，请求头带 `X-API-Key: <key>`（兼容 `Authorization: Bearer <key>`）。完整在线文档见 **`/api/v1/docs`**。

快速示例：

```bash
# 提交推理（multipart 上传音频）
curl -X POST https://你的域名/api/v1/inference \
  -H "X-API-Key: $KEY" \
  -F "config_id=1" \
  -F "audio=@song.wav"

# 查询任务状态 / 下载结果
curl https://你的域名/api/v1/tasks/42 -H "X-API-Key: $KEY"
curl -o result.wav https://你的域名/api/v1/tasks/42/result -H "X-API-Key: $KEY"

# 系统状态 / 我的配额 / 可用模型 / 我的配置
curl https://你的域名/api/v1/system -H "X-API-Key: $KEY"
curl https://你的域名/api/v1/me -H "X-API-Key: $KEY"
curl https://你的域名/api/v1/models -H "X-API-Key: $KEY"
curl https://你的域名/api/v1/configs -H "X-API-Key: $KEY"
```

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/inference` | 提交推理（`config_id` + `audio` 文件，可多文件） |
| GET | `/api/v1/tasks/<id>` | 单个任务状态 |
| GET | `/api/v1/tasks` | 我的任务列表（`?status=done` 筛选） |
| GET | `/api/v1/tasks/<id>/result` | 下载结果文件 |
| GET | `/api/v1/models` | 我可用的模型 |
| GET | `/api/v1/configs` | 我的推理配置（含参数） |
| GET | `/api/v1/me` | 我的信息 + 配额 + 今日用量 |
| GET | `/api/v1/system` | 系统状态（daemon/队列/调度/CPU/内存/磁盘/统计） |
| WS | `ws://host:5001/api/v1/ws/stream` | 实时变声（见 [docs/realtime_vc.md](docs/realtime_vc.md)） |

错误码：`401` Key 无效 · `403` 账号禁用或资源不属于你 · `404` 不存在 · `429` 限流或配额满 · `400` 参数错误。

## 部署与运维

**Linux 服务管理（systemd）**

```bash
systemctl status ssvc          # 查看状态（HTTP 服务 :5000）
systemctl restart ssvc         # 重启（更新代码后必须）
systemctl status ssvc-ws       # 查看状态（实时变声 WebSocket :5001）
systemctl restart ssvc-ws      # 重启 WebSocket 服务
journalctl -u ssvc -n 100      # 查看服务日志（systemd 捕获的 stdout）
tail -f server/logs/app.log    # 查看结构化应用日志（轮转）
```

**更新代码**

```bash
cd ~/server/so-vits-svc-inference && git pull origin master && systemctl restart ssvc ssvc-ws
```

**实时变声**：依赖 `gevent`/`gevent-websocket`（requirements 已含，重跑部署脚本或手动 `pip install gevent gevent-websocket`）。客户端接口协议见 [docs/realtime_vc.md](docs/realtime_vc.md)。

**数据安全**：模型权重、数据库、密钥、上传文件、**备份**（`server/backups/`）、**日志**（`server/logs/`）均不入 git（见 .gitignore）。更新代码不会覆盖 `uploads/`、`pretrain/`、`data.db`、`backups/`。

## 常见问题（FAQ）

**上传的模型为什么不能立刻推理？**
私有模型需管理员在“模型”页审核通过后才能用；也可直接使用官方共享模型。

**推理完成后怎么收到结果？**
设置页填“结果接收邮箱”并开启通知；服务器 SMTP 需管理员在“设置→SMTP”配置。

**任务一直排队？**
检查后台是否“暂停新任务”；单 GPU 按用户轮转，长任务会让后续任务排队；总览/任务页可看全局排队数。

**推理任务一直 running / 进度不动？**
模型加载阶段（CPU 上 1~2 分钟）无进度正常；可点“停止”后重新提交。

**训练出的声音有电音/金属音/沙哑？**
F0 预测器问题（训练用 `dio` 易沙哑，改 `harvest`）；训练不足或过拟合；或训扩散模型挂上 `k_step` 100~300 修音；推理 `noise_scale` 调低到 0.25。

**报“模型架构不匹配”？**
checkpoint 和配置的架构不一致（v1/rvc/rvc-flow 混用）。用匹配的配置，或重新训练。

**服务器内存小（OOM）？**
调低 `INFERENCE_MODEL_CACHE=1`，`INFERENCE_CLIP_SECONDS=15`（长音频自动切块降峰值内存），推理配置 `cluster_ratio=0`，限制用户每日/排队配额。

**怎么恢复数据库备份？**
后台“总览 → 数据库备份”下载 `backup_*.db`。恢复：停服务 → 用备份文件替换 `server/data.db`（删掉同名 `-wal`/`-shm`）→ 启动服务。

**如何用 API 提交推理？**
设置页“开发者 API”生成 Key，`X-API-Key` 请求头调用 `/api/v1/inference`（multipart）。示例见上文 [REST API](#rest-api) 与 `/api/v1/docs`。

**怎么限制注册（防垃圾号）？**
后台“设置 → 邀请码模式”设为“必填”，配合“邀请码”页生成邀请码分发给用户。

**统一流 Hybrid 推理听不清文字 / 全是噪声？**
确认 checkpoint 是用「FM 一致性修复 + head_fm 零初始化」之后的版本；早期 checkpoint（修复前）FM 从纯噪声起点训练，与推理起点不一致，会破坏语音。

## 目录结构

```
server/
├── server/              ← Flask 服务（app.py 入口、blueprints/ 路由、services/ 服务层、模板、worker）
│   ├── blueprints/      ← 认证/仪表盘/模型/配置/推理/任务/公告/管理/训练/健康检查/状态/REST API
│   ├── services/        ← 配额/调度/训练/模型校验/备份/审计/日志/系统资源/REST鉴权
│   ├── templates/       ← 页面模板
│   ├── inference_daemon.py  inference_worker.py  ← 推理执行层
│   ├── backups/         ← 数据库自动备份（不在 git）
│   ├── logs/            ← 结构化轮转日志（不在 git）
│   └── app.py           ← 应用工厂与入口
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/   ← 推理算法
├── docs/                ← 统一流等设计文档
├── pretrain/            ← 预训练模型（不在 git 中）
├── train.py train_diff.py preprocess_*.py   ← 训练脚本（可选模块执行层）
├── deploy_linux.sh      ← Linux 一键部署
├── install.bat / start.bat / start.ps1      ← Windows 安装/启动
└── requirements.txt LICENSE NOTICE
```

## 注意事项

- 模型权重、数据库、密钥等**不进入 git 仓库**（见 .gitignore）
- CPU 服务器可推理；训练理论上可行但非常慢，建议 GPU 训练后上传模型
- CPU 推理：F0 预测器用 `pm`/`harvest`（`crepe` 慢数倍），浅扩散 `k_step` 按需调低（30~100）
- 训练为**可选模块**，默认关闭、仅管理员可用；按需开启
- fairseq 兼容需要 pip 24.0；librosa 用 0.10.1 兼容新版 numpy/torch
