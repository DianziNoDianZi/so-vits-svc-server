#!/bin/bash
# So-VITS-SVC 推理服务部署脚本 (Ubuntu/Debian + CPU + 国内镜像)
set -e

echo "=== So-VITS-SVC 推理服务部署 ==="

DEPLOY_DIR="/opt/so-vits-svc"
INFER_DIR="$DEPLOY_DIR/server"
PYPI_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# 1. 系统依赖（国内镜像源）
echo "[1/5] 安装系统依赖..."
if grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
elif grep -qi "debian" /etc/os-release 2>/dev/null; then
    sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list 2>/dev/null || true
    sed -i 's|http://security.debian.org|http://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list 2>/dev/null || true
fi
apt-get update -qq
apt-get install -y -qq libsndfile1 ffmpeg wget gnupg

# 强制 Python 3.9（Debian 12 默认 3.11 与 fairseq 不兼容）
if grep -qi "debian" /etc/os-release 2>/dev/null && grep -q "12" /etc/debian_version 2>/dev/null; then
    echo "deb https://mirrors.aliyun.com/debian bullseye main" >> /etc/apt/sources.list.d/bullseye.list
    apt-get update -qq
    apt-get install -y -qq -t bullseye python3.9 python3.9-venv python3.9-dev
    PYTHON="python3.9"
else
    PY_VER=""
    for v in 3.11 3.10 3.9; do
        if apt-cache show "python$v" &>/dev/null; then
            PY_VER="$v" && break
        fi
    done
    if [ -z "$PY_VER" ]; then
        apt-get install -y -qq python3 python3-venv python3-dev
        PY_VER="3"
    else
        apt-get install -y -qq "python$PY_VER" "python$PY_VER-venv" "python$PY_VER-dev"
    fi
    PYTHON="python$PY_VER"
fi
echo "使用 Python: $PYTHON"

# 2. 下载预训练模型
echo "[2/5] 检查预训练模型..."

# 2a. ContentVec 编码器（自动下载）
PT_PATH="$DEPLOY_DIR/pretrain/checkpoint_best_legacy_500.pt"
if [ ! -f "$PT_PATH" ]; then
    echo "  下载 ContentVec 编码器..."
    wget -q --show-progress "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/checkpoint_best_legacy_500.pt" -O "$PT_PATH" 2>/dev/null || {
        echo "  下载失败，请手动上传:"
        echo "  scp -P 25163 ./checkpoint_best_legacy_500.pt root@服务器IP:$PT_PATH"
    }
else
    echo "  ContentVec 已存在"
fi

# 2b. nsf_hifigan 声码器（自动下载）
NSF_PATH="$DEPLOY_DIR/pretrain/nsf_hifigan/model"
NSF_ZIP_URL="https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip"
NSF_ZIP_URL_MIRROR="https://ghfast.top/$NSF_ZIP_URL"
if [ ! -f "$NSF_PATH" ]; then
    echo "  下载 nsf_hifigan 声码器..."
    mkdir -p "$DEPLOY_DIR/pretrain/nsf_hifigan"
    wget -q --show-progress "$NSF_ZIP_URL_MIRROR" -O /tmp/nsf_hifigan.zip 2>/dev/null || \
    wget -q --show-progress "$NSF_ZIP_URL" -O /tmp/nsf_hifigan.zip 2>/dev/null || true
    if [ -f /tmp/nsf_hifigan.zip ] && [ $(stat -c%s /tmp/nsf_hifigan.zip 2>/dev/null || echo 0) -gt 10000000 ]; then
        mkdir -p /tmp/nsf_hifigan_extract
        unzip -o -q /tmp/nsf_hifigan.zip -d /tmp/nsf_hifigan_extract
        cp -f /tmp/nsf_hifigan_extract/nsf_hifigan/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null || \
        cp -f /tmp/nsf_hifigan_extract/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null
        rm -rf /tmp/nsf_hifigan.zip /tmp/nsf_hifigan_extract
        echo "  nsf_hifigan 下载完成"
    else
        rm -f /tmp/nsf_hifigan.zip
        echo "  下载失败，请手动上传:"
        echo "  位置: $DEPLOY_DIR/pretrain/nsf_hifigan/"
        echo "  需要: model + config.json 两个文件"
        echo "  来源: so-vits-svc 发行包 / GPT-SoVITS 项目内同路径文件"
    fi
else
    echo "  nsf_hifigan 已存在"
fi

# 2c. SoVITS 训练底模（可选，推荐上传）
BASE_G="$DEPLOY_DIR/pretrain/G_0.pth"
BASE_D="$DEPLOY_DIR/pretrain/D_0.pth"
if [ ! -f "$BASE_G" ] || [ ! -f "$BASE_D" ]; then
    echo ""
    echo "  💡 可选: 上传 SoVITS 训练底模 (G_0.pth + D_0.pth)"
    echo "    作用: 从预训练权重开始训练，收敛更快、音质更好"
    echo "    上传到: $DEPLOY_DIR/pretrain/"
    echo "    不传也能训练，只是从零开始"
    echo ""
fi

# 3. Python 环境
echo "[3/5] 配置 Python 环境..."
cd $INFER_DIR
$PYTHON -m venv venv
source venv/bin/activate
pip install -q --upgrade pip setuptools wheel $PYPI_MIRROR
pip install -q torch torchaudio $PYPI_MIRROR
pip install -q flask flask-login flask-sqlalchemy waitress psutil soundfile librosa numpy $PYPI_MIRROR

# 4. 创建目录和 swap
echo "[4/5] 创建目录和 swap..."
mkdir -p uploads/models uploads/configs uploads/audio uploads/results

if ! swapon --show | grep -q /swapfile; then
    echo "创建 2GB swap..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# 5. 注册服务
echo "[5/5] 注册系统服务..."
cat > /etc/systemd/system/ssvc.service << EOF
[Unit]
Description=So-VITS-SVC Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INFER_DIR
ExecStart=$INFER_DIR/venv/bin/python app.py
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=$DEPLOY_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ssvc
systemctl start ssvc

echo ""
echo "=== 部署完成！ ==="
IP=$(hostname -I | awk '{print $1}')
echo "地址: http://$IP:5000"
echo ""
echo "首次登录："
echo "  用户名: admin"
echo "  密码:   journalctl -u ssvc -n 50 | grep '初始密码'"
echo ""
echo "常用命令："
echo "  systemctl status ssvc     # 状态"
echo "  journalctl -u ssvc -f     # 日志"
echo "  systemctl restart ssvc    # 重启"
