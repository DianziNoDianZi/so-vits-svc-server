#!/bin/bash
# So-VITS-SVC Linux Conda Multi-Engine Deployment Script (Fix goto bug)
# Supports: 1. NVIDIA CUDA, 2. AMD ROCm, 3. CPU only
set -e

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$DEPLOY_DIR/server"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="ssvc"

# Check for root privileges
if [ "$(id -u)" != "0" ]; then
    echo "[ERROR] Please run as root:  sudo bash deploy_linux.sh"
    exit 1
fi

SKIP_MODELS="${SKIP_MODELS:-0}"
if [ "$1" = "--skip-models" ]; then
    SKIP_MODELS=1
fi

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "============================================"
echo "  So-VITS-SVC Linux Conda Deployment"
echo "============================================"
echo "Select PyTorch version for your hardware:"
echo "  1 = NVIDIA CUDA 12.1 GPU"
echo "  2 = AMD ROCm GPU (Linux only)"
echo "  3 = CPU (Inference only)"

# 改用 while 循环代替 goto，完美兼容所有 Bash 版本
while true; do
    read -p "Enter choice [1/2/3]: " TORCH_CHOICE
    case $TORCH_CHOICE in
        1) echo ">>> Selected: NVIDIA CUDA GPU"; break ;;
        2) echo ">>> Selected: AMD ROCm GPU"; break ;;
        3) echo ">>> Selected: CPU"; break ;;
        *) echo "[ERROR] Invalid input. Please enter 1, 2, or 3." ;;
    esac
done
echo ""

# === 2. Install System Dependencies ===
echo "[1/5] Installing system libraries..."
if command -v apt-get >/dev/null 2>&1; then
    PKG_CMD="apt-get -o DPkg::Lock::Timeout=300 -o APT::Get::Assume-Yes=true"
    $PKG_CMD update
    $PKG_CMD install ffmpeg libsndfile1 wget unzip build-essential cmake
elif command -v yum >/dev/null 2>&1; then
    PKG_CMD="yum install -y"
    $PKG_CMD epel-release
    $PKG_CMD ffmpeg libsndfile wget unzip gcc gcc-c++ make cmake
else
    echo "[ERROR] Unsupported package manager"
    exit 1
fi

# === 3. Install Miniconda ===
echo "[2/5] Installing Miniconda..."
if [ ! -d "$CONDA_DIR" ]; then
    # 这台机器连不了海外，只用国内镜像（官方 repo.anaconda.com 会连不上）
    MINICONDA_URLS=(
        "https://mirrors.ustc.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        "https://mirrors.nju.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    )
    downloaded=0
    for u in "${MINICONDA_URLS[@]}"; do
        echo "  Downloading Miniconda from: $u"
        # 加超时防止无响应无限挂；加浏览器 UA 绕开部分镜像（如 USTC）对 wget 默认 UA 的 403
        if wget --timeout=25 --tries=2 --user-agent="Mozilla/5.0" --show-progress "$u" -O /tmp/miniconda.sh && [ -s /tmp/miniconda.sh ]; then
            downloaded=1
            break
        fi
    done
    if [ "$downloaded" != "1" ]; then
        echo "[ERROR] Miniconda download failed from all mirrors."
        echo "        Please download it manually:"
        echo "        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "        then run:  bash <下载的脚本> -b -p $CONDA_DIR"
        exit 1
    fi
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

cat > "$HOME/.condarc" <<EOF
# 这台机器连不了海外，conda 只用 USTC（阿里云 anaconda 的 pkgs/main 是 404，清华又连不上，别加了）
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/r/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
default_channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/r/
custom_channels:
  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud
show_channel_urls: true
EOF

# === 4. Create Python Environment ===
echo "[3/5] Creating Python 3.9 Conda environment..."
if ! conda env list | awk -v n="$ENV_NAME" '$1==n' | grep -q .; then
    conda create -y -n "$ENV_NAME" python=3.9 \
        --retries 6 \
        --override-channels \
        -c https://mirrors.ustc.edu.cn/anaconda/pkgs/main/ \
        -c https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
fi
conda activate "$ENV_NAME"

# === 5. Install Python Dependencies ===
echo "[4/5] Installing Python dependencies..."
PIP_ARGS="--root-user-action=ignore --timeout 60 --retries 5"
pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip 'setuptools<81' wheel

echo "  Installing PyTorch..."
# 连不了海外，PyTorch 走国内镜像。单个镜像可能没同步/404（阿里云 cpu/ 就是空的），
# 所以多个轮着试；都失败再回退清华 PyPI（torch 是 CUDA 版 wheel，CPU 上也能跑，只是下载大）。
if [ "$TORCH_CHOICE" = "1" ]; then
    MIRRORS="https://mirrors.aliyun.com/pytorch-wheels/cu121/ https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/"
elif [ "$TORCH_CHOICE" = "2" ]; then
    MIRRORS="https://mirrors.aliyun.com/pytorch-wheels/rocm/ https://mirror.sjtu.edu.cn/pytorch-wheels/rocm/"
else
    MIRRORS="https://mirrors.aliyun.com/pytorch-wheels/cpu/ https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/"
fi
TORCH_OK=0
for IDX in $MIRRORS; do
    echo "  Trying torch from: $IDX"
    if pip install $PIP_ARGS torch torchaudio --index-url "$IDX"; then
        TORCH_OK=1
        break
    fi
done
if [ "$TORCH_OK" != "1" ]; then
    echo "  国内 pytorch 镜像都失败，回退清华 PyPI"
    pip install $PIP_ARGS torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo "  Downgrading pip to bypass omegaconf dependency lock..."
python -m pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple 'pip<24.1'

echo "  Installing fairseq (compiling C++, this may take a few minutes)..."
pip install $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple fairseq==0.12.2

echo "  Installing remaining requirements (skipping onnxsim)..."
grep -v -E '^(onnx|onnxsim|onnxoptimizer|fairseq)' "$DEPLOY_DIR/requirements.txt" > /tmp/req_server.txt
pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/req_server.txt

# === 6. Download Pretrained Models & Enable Swap ===
echo "[5/5] Downloading pretrained models and configuring swap..."
if [ "$SKIP_MODELS" = "1" ]; then
    echo "  [skip] Skipping pretrained models download."
else
    mkdir -p "$DEPLOY_DIR/pretrain/nsf_hifigan"
    if [ ! -f "$DEPLOY_DIR/pretrain/checkpoint_best_legacy_500.pt" ]; then
        echo "  Downloading ContentVec encoder..."
        wget -q "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/checkpoint_best_legacy_500.pt" \
            -O "$DEPLOY_DIR/pretrain/checkpoint_best_legacy_500.pt" || echo "  Download failed!"
    fi
    if [ ! -f "$DEPLOY_DIR/pretrain/nsf_hifigan/model" ]; then
        echo "  Downloading NSF-HiFiGAN vocoder..."
        wget -q "https://ghfast.top/https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip" \
            -O /tmp/nsf_hifigan.zip || true
        if [ -s /tmp/nsf_hifigan.zip ]; then
            mkdir -p /tmp/nsf_hifigan_x
            unzip -o -q /tmp/nsf_hifigan.zip -d /tmp/nsf_hifigan_x
            cp -f /tmp/nsf_hifigan_x/nsf_hifigan/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null || \
            cp -f /tmp/nsf_hifigan_x/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null || true
            rm -rf /tmp/nsf_hifigan.zip /tmp/nsf_hifigan_x
        else
            echo "  Download failed. Please upload manually."
            rm -f /tmp/nsf_hifigan.zip
        fi
    fi
fi

# Setup Swap
if ! swapon --show | grep -q /swapfile; then
    echo "  Creating 4GB swap file..."
    fallocate -l 4G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=4096 status=none
    chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

pip cache purge >/dev/null 2>&1 || true
conda clean -a -y >/dev/null 2>&1 || true
rm -rf /tmp/pip-* /tmp/apt_out.log /tmp/nsf_hifigan* /tmp/miniconda.sh 2>/dev/null || true

# === 7. Register System Service ===
PY_BIN="$CONDA_DIR/envs/$ENV_NAME/bin/python"
if command -v systemctl >/dev/null 2>&1; then
    echo "  Registering systemd service..."
    cat > /etc/systemd/system/ssvc.service <<EOF
[Unit]
Description=So-VITS-SVC Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$SERVER_DIR
ExecStart=$PY_BIN app.py
Restart=on-failure
RestartSec=5
Environment=PORT=5000
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable ssvc
    systemctl restart ssvc
else
    cd "$SERVER_DIR"
    nohup "$PY_BIN" app.py > server.out 2>&1 &
    echo $! > server.pid
fi

echo ""
echo "============================================"
echo "Deployment complete!"
echo "Service is running in the background."
echo "Access URL: http://$(hostname -I | awk '{print $1}'):5000"
echo "============================================"