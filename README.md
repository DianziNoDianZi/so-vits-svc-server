# So-VITS-SVC Server

so-vits-svc 推理 + 训练 Web 服务器：数据集上传、队列训练、续训、Loss 监控、随时停止试听、推理换声，全部在浏览器里完成。多用户、任务队列、邮件通知，支持 NVIDIA / AMD / 纯 CPU 一键部署。

## 效果试听

实际训练出的模型推理结果：

- [示例 1：短句](docs/demo/demo-1.wav)
- [示例 2：完整句子](docs/demo/demo-2.wav)
- [示例 3：另一片段](docs/demo/demo-3.wav)

## 快速开始

**Windows**

```bat
install.bat   :: 建 venv + 装依赖（选 1=CPU / 2=CUDA）
start.bat     :: 启动，访问 http://localhost:5000
```

**Linux**（CUDA / ROCm / CPU 三选一，自动走国内镜像）

```bash
chmod +x deploy_linux.sh
sudo bash deploy_linux.sh
sudo bash deploy_linux.sh --skip-models   # 模型下载慢时可跳过，之后网页上传
```

> Windows 需先装 CMake（onnxsim 编译依赖）；AMD Linux 用户需装好 ROCm 驱动。

## 主要功能

- 推理：上传模型 + 音频换声，支持浅扩散（k_step）修音
- 训练：数据集 zip 队列训练，主模型 / 扩散均可从 checkpoint 续训
- 中途停止：随时停止，自动保存 checkpoint 为模型，立即试听
- 监控：实时日志、G/D Loss 曲线、验证集 loss、ETA
- 清理：按任务 / 模型 / 数据集精确清理，防误伤提示
- 文件管理：网页下载 / 删除模型、配置、结果、数据集
- 预训练页：上传管理 ContentVec、NSF-HiFiGAN、G_0/D_0、RMVPE
- 邮件：训练完成 / 每 N 步 / 推理完成自动通知（SMTP 自配）
- 安全：CSRF、登录限速、路径穿越防护、持久化会话密钥

## 使用流程

**推理**：模型管理上传 G_*.pth + config.json（可附扩散模型）→ 建推理配置 → 传音频提交任务 → 任务列表下载结果

**训练**：训练页传数据集 zip（自动过滤 <2 秒片段）→ 设步数等参数 → 看 Loss 判断收敛 / 过拟合 → 满意就停止（checkpoint 自动存为模型）→ 主模型好了可一键训扩散

## 预训练模型

| 文件 | 大小 | 获取 |
|------|------|------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | 脚本自动下载 / 网页上传 |
| `pretrain/nsf_hifigan/model` + config | ~54MB | 同上 |
| `pretrain/G_0.pth` + `D_0.pth` | ~400MB | 网页上传（推荐） |
| `pretrain/rmvpe.pt` 等 | 可选 | 网页上传 |

大文件可 scp 到 `pretrain/`，网页自动识别。

## 配置

| 项 | 环境变量 | 默认 |
|----|---------|------|
| 会话密钥 | SECRET_KEY | 自动生成并持久化 |
| 数据库 | DATABASE_URL | server/data.db |
| 端口 | PORT | 5000 |
| 推理超时 | INFERENCE_TASK_TIMEOUT | 21600 秒 |

## 协议

本项目是 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)（AGPL-3.0）的派生作品，整体以 AGPL-3.0 发布。使用 / 二次分发须保留版权与修改声明并以相同协议分发，修改文件清单见 NOTICE。
