#!/bin/bash
# So-VITS-SVC CPU-only deployment via Miniconda (no system python involved)
# 优点: 完全不改系统 Python，apt 只装系统库，版本隔离，国内镜像可装
set -e

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$DEPLOY_DIR/server"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="ssvc"
APT="apt-get -o DPkg::Lock::Timeout=300 -o APT::Get::Assume-Yes=true"

if [ "$(id -u)" != "0" ]; then
    echo "[ERROR] Please run as root:  sudo bash deploy_cpu_conda.sh"
    exit 1
fi

SKIP_MODELS="${SKIP_MODELS:-0}"
if [ "$1" = "--skip-models" ]; then
    SKIP_MODELS=1
fi

retry_apt() {
    local n=0
    until "$@" >/tmp/apt_out.log 2>&1; do
        n=$((n + 1))
        if [ "$n" -ge 3 ]; then
            echo "[ERROR] apt failed 3 times:"
            tail -n 15 /tmp/apt_out.log
            exit 1
        fi
        echo "  apt failed (lock? network?), retrying in 10s... ($n/3)"
        sleep 10
        dpkg --configure -a 2>/dev/null || true
    done
}

echo "=== So-VITS-SVC CPU deployment (conda) ==="

# 1. 系统库（与 Python 无关，不会冲突）
echo "[1/5] Installing system libraries..."
retry_apt $APT update
retry_apt $APT install ffmpeg libsndfile1 wget unzip build-essential cmake

# 2. 安装 Miniconda（清华镜像，国内可达）
echo "[2/5] Installing Miniconda..."
if [ ! -d "$CONDA_DIR" ]; then
    wget -q "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh || {
        echo "[ERROR] Miniconda 下载失败，请手动下载后重试"
        exit 1
    }
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 直接写入只含清华镜像的 .condarc（完全替换默认 channels，避免访问 repo.anaconda.com）
[ -f "$HOME/.condarc" ] && cp "$HOME/.condarc" "$HOME/.condarc.bak" 2>/dev/null || true
cat > "$HOME/.condarc" <<EOF
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
show_channel_urls: true
EOF

# 3. 创建 python 3.9 环境（隔离，不影响系统）
echo "[3/5] Creating python 3.9 conda env..."
if ! conda env list | awk -v n="$ENV_NAME" '$1==n' | grep -q .; then
    conda create -y -n "$ENV_NAME" python=3.9 \
        --override-channels \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
fi
conda activate "$ENV_NAME"

# 4. pip 依赖（清华 PyPI）
echo "[4/5] Installing python dependencies..."
PIP_ARGS="--root-user-action=ignore --timeout 60 --retries 5"
pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip 'setuptools<81' wheel
# CPU 版 torch：依次尝试国内镜像，最后回退清华 PyPI
torch_ok=0
for m in "https://mirrors.aliyun.com/pytorch-wheels/cpu/" \
         "https://mirrors.huaweicloud.com/pytorch/wheels/cpu/" \
         "https://mirrors.bfsu.edu.cn/pytorch/wheels/cpu/"; do
    echo "  trying CPU torch mirror: $m"
    if pip install $PIP_ARGS torch torchaudio --index-url "$m"; then
        torch_ok=1
        break
    fi
done
if [ "$torch_ok" = "0" ]; then
    echo "  CPU mirrors failed. Installing torch with --no-deps (skip nvidia-* packages, works on CPU)"
    pip install $PIP_ARGS --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchaudio
    pip install $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple filelock typing-extensions sympy networkx jinja2 fsspec
fi
# fairseq 依赖的 omegaconf 旧元数据不兼容新版 pip(>=24.1)，降到 24.0
python -m pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple 'pip<24.1'
python -m pip --version
# 服务器运行不需要 ONNX 导出依赖（onnx/onnxsim/onnxoptimizer），跳过以省去 cmake 编译
echo "  Installing fairseq (Cython 编译，需要几分钟，请耐心等待)..."
pip install $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple fairseq==0.12.2
echo "  Installing remaining dependencies..."
grep -v -E '^(onnx|onnxsim|onnxoptimizer|fairseq)' "$DEPLOY_DIR/requirements.txt" > /tmp/req_server.txt
pip install -q $PIP_ARGS -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/req_server.txt

# 5. 预训练模型 + swap + 后台服务
echo "[5/5] Pretrained models, swap and service..."
if [ "$SKIP_MODELS" = "1" ]; then
    echo "  [skip] 跳过预训练模型下载。部署后请到网页 预训练 页上传 ContentVec + NSF-HiFiGAN"
else
    mkdir -p "$DEPLOY_DIR/pretrain/nsf_hifigan"
    if [ ! -f "$DEPLOY_DIR/pretrain/checkpoint_best_legacy_500.pt" ]; then
        echo "Downloading ContentVec encoder..."
        wget -q "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/checkpoint_best_legacy_500.pt" \
            -O "$DEPLOY_DIR/pretrain/checkpoint_best_legacy_500.pt" \
            || echo "  Download failed - upload it manually to $DEPLOY_DIR/pretrain/"
    fi
    if [ ! -f "$DEPLOY_DIR/pretrain/nsf_hifigan/model" ]; then
        echo "Downloading nsf_hifigan vocoder..."
        wget -q "https://ghfast.top/https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip" \
            -O /tmp/nsf_hifigan.zip || true
        if [ -s /tmp/nsf_hifigan.zip ]; then
            mkdir -p /tmp/nsf_hifigan_x
            unzip -o -q /tmp/nsf_hifigan.zip -d /tmp/nsf_hifigan_x
            cp -f /tmp/nsf_hifigan_x/nsf_hifigan/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null || \
            cp -f /tmp/nsf_hifigan_x/* "$DEPLOY_DIR/pretrain/nsf_hifigan/" 2>/dev/null || true
            rm -rf /tmp/nsf_hifigan.zip /tmp/nsf_hifigan_x
        else
            echo "  Download failed - upload model + config.json to $DEPLOY_DIR/pretrain/nsf_hifigan/"
            rm -f /tmp/nsf_hifigan.zip
        fi
    fi
fi

# 6. 清理缓存，减小磁盘占用
echo "Cleaning caches..."
pip cache purge >/dev/null 2>&1 || true
conda clean -a -y >/dev/null 2>&1 || true
rm -rf /tmp/pip-* /tmp/apt_out.log /tmp/nsf_hifigan* /tmp/miniconda.sh 2>/dev/null || true

if ! swapon --show | grep -q /swapfile; then
    echo "Creating 4GB swap..."
    fallocate -l 4G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=4096 status=none
    chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

PY_BIN="$CONDA_DIR/envs/$ENV_NAME/bin/python"
if command -v systemctl >/dev/null 2>&1; then
    echo "Registering systemd service..."
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
echo "Deployment done (conda env: $ENV_NAME)"
echo "URL: http://$(hostname -I | awk '{print $1}'):5000"
