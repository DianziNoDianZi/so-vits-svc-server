# So-VITS-SVC 服务器独立部署项目

独立部署的 so-vits-svc 推理 + 训练服务器。

## ⚠️ 许可证

本仓库使用 **AGPL-3.0** 许可证（见 LICENSE）。
包含 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的派生代码
（inference/, modules/, utils.py 等），遵守上游开源协议。

## 功能

- 🎤 推理服务：上传模型 + 音频 → SVC 换声
- 🎓 训练服务：上传数据集 → 队列训练 → 模型下载
- 📧 邮件通知：训练完成自动发邮件
- 📊 训练监控：实时日志 / Loss 曲线 / 阶段指示
- 🧹 清理功能：缓存/旧任务一键清理

## 快速部署（Ubuntu/Debian 22.04）

```bash
git clone https://github.com/DianziNoDianZi/so-vits-svc-server.git /opt/so-vits-svc
cd /opt/so-vits-svc
bash deploy_ubuntu.sh
```

## 配置

| 项 | 环境变量 | 默认 |
|----|---------|------|
| 会话密钥 | SECRET_KEY | change-this-to-a-random-secret（**必须改**） |
| 数据库 | DATABASE_URL | server/data.db |
| 服务端口 | PORT | 5000 |

## 预训练模型

部署时需要以下文件（不包含在 git 仓库中）：

| 文件 | 大小 | 获取方式 |
|------|------|---------|
| `pretrain/checkpoint_best_legacy_500.pt` | ~180MB | 部署脚本自动下载（hf-mirror） |
| `pretrain/nsf_hifigan/model` + `config.json` | ~54MB | 部署脚本自动下载（官方 release，含国内加速镜像） |
| `pretrain/G_0.pth` + `pretrain/D_0.pth` | ~1.2GB | 需手动上传（可选，SoVITS 训练底模，推荐） |

**手动上传位置（仅底模需要）：**
```
/opt/so-vits-svc/pretrain/G_0.pth
/opt/so-vits-svc/pretrain/D_0.pth
```

运行部署脚本时会自动检查并下载/提示缺失的文件。

## 目录结构

```
server/
├── server/              ← Flask 服务
├── inference/ modules/ diffusion/ vencoder/ vdecoder/ cluster/
├── configs_template/    ← 训练配置模板
├── pretrain/            ← 预训练模型（部署时下载）
├── train.py train_diff.py preprocess_*.py
└── requirements.txt deploy_ubuntu.sh LICENSE
```

## 使用流程

**推理：**
1. 模型管理 → 上传 G_*.pth + config.json
2. 创建推理配置
3. 上传音频 → 提交推理 → 下载结果

**训练：**
1. 训练页 → 上传数据集 zip
2. 设置参数 → 提交任务
3. 等邮件通知 → 下载模型
