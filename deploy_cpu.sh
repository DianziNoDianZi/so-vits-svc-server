#!/bin/bash
# So-VITS-SVC CPU-only deployment (Ubuntu/Debian)
# Inference works; training is possible but very slow.
set -e

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$DEPLOY_DIR/server"
APT="apt-get -o DPkg::Lock::Timeout=300 -o APT::Get::Assume-Yes=true"

if [ "$(id -u)" != "0" ]; then
    echo "[ERROR] Please run as root:  sudo bash deploy_cpu.sh"
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

has_pkgs() {
    for p in "$@"; do
        apt-cache show "$p" >/dev/null 2>&1 || return 1
    done
    return 0
}

echo "=== So-VITS-SVC CPU deployment ==="

# 1. system dependencies (wait for unattended-upgrades lock, retry on failure)
echo "[1/5] Installing system dependencies..."
if pgrep -x unattended-upgr >/dev/null 2>&1; then
    echo "  unattended-upgrades is running, waiting for it to release the lock..."
fi
retry_apt $APT update
retry_apt $APT install ffmpeg libsndfile1 wget unzip build-essential cmake

# 2. Python 3.9/3.10 (fairseq 0.12.2 不兼容 3.11+)
echo "[2/5] Preparing Python..."
. /etc/os-release 2>/dev/null || true
echo "  OS: ${PRETTY_NAME:-unknown}"

PYTHON=""
for cand in python3.9 python3.10; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import ensurepip" >/dev/null 2>&1; then
        PYTHON="$cand"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    # Ubuntu: 先启用 universe 组件（python3.9 在 universe 里）
    if grep -qi ubuntu /etc/os-release 2>/dev/null; then
        echo "  enabling universe component..."
        if command -v add-apt-repository >/dev/null 2>&1; then
            add-apt-repository -y universe >/dev/null 2>&1 || true
        else
            for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do
                [ -f "$f" ] || continue
                sed -i -E 's/^(Components:[[:space:]]*)main(.*)$/\1main universe\2/' "$f" 2>/dev/null || true
                sed -i -E 's/^(deb[^ ]+ [^ ]+ [^ ]+) main( .*)?$/\1 main universe\2/' "$f" 2>/dev/null || true
            done
        fi
        retry_apt $APT update
    fi
    if has_pkgs python3.9 python3.9-venv python3.9-dev; then
        retry_apt $APT install python3.9 python3.9-venv python3.9-dev
        PYTHON=python3.9
    elif has_pkgs python3.10 python3.10-venv python3.10-dev; then
        retry_apt $APT install python3.10 python3.10-venv python3.10-dev
        PYTHON=python3.10
    else
        echo "  fallback: bullseye 源 (Debian 12 / 特殊环境)"
        apt-get install -y debian-archive-keyring 2>/dev/null || true
        if [ -f /usr/share/keyrings/debian-archive-keyring.gpg ]; then
            SIGNED="signed-by=/usr/share/keyrings/debian-archive-keyring.gpg"
        else
            SIGNED="trusted=yes"
            echo "  [warn] 使用 trusted=yes 添加 bullseye 镜像（仅部署用）"
        fi
        echo "deb [${SIGNED}] https://mirrors.aliyun.com/debian bullseye main" > /etc/apt/sources.list.d/bullseye.list
        retry_apt $APT update
        retry_apt $APT install -t bullseye python3.9 python3.9-venv python3.9-dev
        PYTHON=python3.9
    fi
fi
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.9/3.10 不可用，请手动安装后重试"
    exit 1
fi
echo "  Using Python: $PYTHON"

# 3. venv + CPU torch (key difference: CPU wheel, no CUDA runtime)
echo "[3/5] Creating venv and installing CPU torch..."
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
cd "$SERVER_DIR"
if [ ! -d venv ]; then
    "$PYTHON" -m venv venv
fi
source venv/bin/activate
pip install -q -i "$PIP_MIRROR" --upgrade pip 'setuptools<81' wheel
# CPU 版 torch：依次尝试国内镜像，最后回退清华 PyPI
torch_ok=0
for m in "https://mirrors.aliyun.com/pytorch-wheels/cpu/" \
         "https://mirrors.huaweicloud.com/pytorch/wheels/cpu/" \
         "https://mirrors.bfsu.edu.cn/pytorch/wheels/cpu/"; do
    echo "  trying CPU torch mirror: $m"
    if pip install -q torch torchaudio --index-url "$m"; then
        torch_ok=1
        break
    fi
done
if [ "$torch_ok" = "0" ]; then
    echo "  CPU mirrors failed. Installing torch with --no-deps (skip nvidia-* packages, works on CPU)"
    pip install -q --no-deps -i "$PIP_MIRROR" torch torchaudio
    pip install -q -i "$PIP_MIRROR" filelock typing-extensions sympy networkx jinja2 fsspec
fi
# fairseq 依赖的 omegaconf 旧元数据不兼容新版 pip(>=24.1)，降到 24.0
python -m pip install -q -i "$PIP_MIRROR" 'pip<24.1'
python -m pip --version
# 服务器运行不需要 ONNX 导出依赖（onnx/onnxsim/onnxoptimizer），跳过以省去 cmake 编译
echo "  Installing fairseq (Cython 编译，需要几分钟，请耐心等待)..."
pip install -i "$PIP_MIRROR" fairseq==0.12.2
echo "  Installing remaining dependencies..."
grep -v -E '^(onnx|onnxsim|onnxoptimizer|fairseq)' "$DEPLOY_DIR/requirements.txt" > /tmp/req_server.txt
pip install -q -i "$PIP_MIRROR" -r /tmp/req_server.txt

# 4. pretrained models (needed for inference; skip if already present)
echo "[4/5] Checking pretrained models..."
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

# 清理 pip 缓存，减小磁盘占用
echo "Cleaning pip cache..."
pip cache purge >/dev/null 2>&1 || true

# 5. optional swap (CPU inference/training can need a few GB)
echo "[5/5] Preparing swap..."
if ! swapon --show | grep -q /swapfile; then
    echo "Creating 4GB swap..."
    fallocate -l 4G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=4096 status=none
    chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo ""
echo "Deployment done."

# 5. run as a background service (survives SSH disconnect and reboots)
if command -v systemctl >/dev/null 2>&1; then
    echo "Registering systemd service..."
    cat > /etc/systemd/system/ssvc.service <<EOF
[Unit]
Description=So-VITS-SVC Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$SERVER_DIR
ExecStart=$SERVER_DIR/venv/bin/python app.py
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
    echo ""
    echo "Service started in background (systemd)."
    echo "  status:  systemctl status ssvc"
    echo "  logs:    journalctl -u ssvc -f"
    echo "  stop:    systemctl stop ssvc"
    echo "  restart: systemctl restart ssvc"
else
    echo "systemd not available, starting with nohup..."
    cd "$SERVER_DIR"
    nohup venv/bin/python app.py > server.out 2>&1 &
    echo $! > server.pid
    echo "Started with nohup (pid $(cat server.pid)). Logs: $SERVER_DIR/server.out"
    echo "Stop: kill \$(cat $SERVER_DIR/server.pid)"
fi

echo ""
echo "URL: http://$(hostname -I | awk '{print $1}'):5000"
