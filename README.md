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

-  **推理服务**：上传模型 + 音频 → SVC 换声；浅扩散（k_step）修音
-  **训练服务**：数据集 zip → 队列训练；**续训**（从 checkpoint 继续，主模型/扩散均可）
-  **中途停止测试**：训练中随时停止，自动保存当前 checkpoint 并注册为模型，马上推理试听
-  **训练监控**：实时日志 / Loss 曲线 / **验证集 loss**（判断过拟合）/ ETA
-  **精确清理**：按单个训练任务/模型/数据集清理，防误伤检查
-  **文件管理**：自由下载/删除服务器上的模型、配置、结果、数据集
-  **预训练页**：网页上传/管理 ContentVec、NSF-HiFiGAN、训练底模 G_0/D_0 等
-  **邮件通知**：训练完成自动发邮件（SMTP 可配置）
-  安全：CSRF 防护、登录限速、路径穿越防护、会话密钥持久化

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
├── deploy_linux.sh      ← Linux 一键部署（CUDA / ROCm / CPU 三选一，conda）
├── install.bat / start.bat / start.ps1  ← Windows 安装/启动
└── requirements.txt LICENSE
```

## 注意事项

- 模型权重、数据库、密钥等**不进入 git 仓库**（见 .gitignore）
- CPU 服务器可推理；训练理论上可行但非常慢，建议 GPU 训练后上传模型
- 国内网络部署：apt/conda/pip 全部走清华/阿里云镜像，torch 用 `--no-deps` 跳过 nvidia 包
- fairseq 兼容需要 pip 24.0（脚本自动处理）；librosa 用 0.10.1 兼容新版 numpy/torch
- 训练 DataLoader 默认 `num_workers=2`（内存充足自动升到 4，<8GB 降为 0）
