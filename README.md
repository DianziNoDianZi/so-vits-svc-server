# So-VITS-SVC 服务器独立部署项目

独立部署的 so-vits-svc **推理 + 训练** Web 服务器，支持多用户登录、模型管理、
推理配置、任务队列、一键部署。

## ⚠️ 许可证

本仓库使用 **AGPL-3.0** 许可证（见 LICENSE）。
包含 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的派生代码
（inference/, modules/, diffusion/, utils.py 等），遵守上游开源协议。

## 功能

- 🎤 **推理服务**：上传模型 + 音频 → SVC 换声；浅扩散（k_step）修音
- 🎓 **训练服务**：数据集 zip → 队列训练；**续训**（从 checkpoint 继续，主模型/扩散均可）
- 🧪 **中途停止测试**：训练中随时停止，自动保存当前 checkpoint 并注册为模型，马上推理试听
- 📊 **训练监控**：实时日志 / Loss 曲线 / **验证集 loss**（判断过拟合）/ ETA
- 🧹 **精确清理**：按单个训练任务/模型/数据集清理，防误伤检查
- 📁 **文件管理**：自由下载/删除服务器上的模型、配置、结果、数据集
- 🧬 **预训练页**：网页上传/管理 ContentVec、NSF-HiFiGAN、训练底模 G_0/D_0 等
- 📧 **邮件通知**：训练完成自动发邮件（SMTP 可配置）
- 🔒 安全：CSRF 防护、登录限速、路径穿越防护、会话密钥持久化

## 快速开始

### Windows 本机（一键安装）

```bat
install.bat      :: 建 venv + 装 CPU/CUDA torch + 依赖（选 1=CPU / 2=CUDA）
start.bat        :: 启动服务，访问 http://localhost:5000
```

### Linux CPU 服务器（conda 版，推荐，全走国内镜像）

```bash
cd /opt/so-vits-svc
bash deploy_cpu_conda.sh --skip-models   # 跳过模型下载，部署后网页上传
# 或不带 --skip-models 自动下载预训练模型
```

脚本自动完成：系统库 → Miniconda（清华源）→ Python 3.9 环境 → CPU torch（无 nvidia 包）
→ 全部依赖 → swap → **systemd 后台服务**（ssh 断开不掉、开机自启）。

### Linux GPU 服务器

```bash
bash deploy_ubuntu.sh
```

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
3. 上传音频 → 提交推理 → 任务列表看进度/下载结果

**训练：**
1. 训练页 → 上传数据集 zip（自动过滤 <2 秒短片段）
2. 设置参数（总步数、auto_stop、编码器、底模自动加载）
3. 训练页看 train/验证 loss → 满意就"停止"（自动保存 checkpoint）
4. 任务列表"续训"可继续，或直接测试当前模型

**扩散：** 主模型训好后，任务列表点"训扩散"直接进入扩散训练（复用数据/特征），
完成后把扩散模型挂到主模型上，k_step 100~300 推理。

## 目录结构

```
server/
├── server/              ← Flask 服务（app.py、模板、worker）
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/
├── configs_template/    ← 训练配置模板
├── pretrain/            ← 预训练模型（不在 git 中）
├── train.py train_diff.py preprocess_*.py
├── deploy_cpu.sh        ← Linux CPU 一键部署（apt 版）
├── deploy_cpu_conda.sh  ← Linux CPU 一键部署（conda 版，推荐）
├── deploy_ubuntu.sh     ← Linux GPU 部署
├── install.bat / start.bat / start.ps1  ← Windows 安装/启动
└── requirements.txt LICENSE
```

## 注意事项

- 模型权重、数据库、密钥等**不进入 git 仓库**（见 .gitignore）
- CPU 服务器可推理；训练理论上可行但非常慢，建议 GPU 训练后上传模型
- 国内网络部署：apt/conda/pip 全部走清华/阿里云镜像，torch 用 `--no-deps` 跳过 nvidia 包
- fairseq 兼容需要 pip 24.0（脚本自动处理）；librosa 用 0.10.1 兼容新版 numpy/torch
