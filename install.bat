@echo off
:: 设置CMD编码为UTF-8，防止乱码
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   So-VITS-SVC Server - One-click install
echo ============================================
echo.

set "PY=python"
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ first.
    pause
    exit /b 1
)

echo [1/4] Creating venv...
if not exist "venv\Scripts\python.exe" (
    %PY% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)
set "VPY=venv\Scripts\python.exe"

:: Configure domestic mirror for China users
echo Configuring pip mirror (Tsinghua)...
"%VPY%" -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
if errorlevel 1 (
    echo [WARNING] Failed to set pip mirror, using default.
)

echo Upgrading pip/setuptools/wheel...
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] pip upgrade failed. Check network.
    pause
    exit /b 1
)

echo [2/4] Choose torch build:
echo   1 = CPU (smaller, inference only, slow training)
echo   2 = NVIDIA CUDA GPU (needed for real training, ~2.5GB download)
echo   3 = AMD ROCm GPU (for Radeon RX 5000/6000/7000 series)
:torch_choice
set "TORCH_CHOICE="
set /p TORCH_CHOICE="Choice (1/2/3): "
if "%TORCH_CHOICE%"=="1" goto torch_cpu
if "%TORCH_CHOICE%"=="2" goto torch_cuda
if "%TORCH_CHOICE%"=="3" goto torch_amd
echo Invalid choice, please enter 1, 2 or 3.
goto torch_choice

:torch_cpu
echo Installing CPU torch...
"%VPY%" -m pip install torch torchaudio
if errorlevel 1 (
    echo [ERROR] torch install failed.
    pause
    exit /b 1
)
goto install_reqs

:torch_cuda
echo Installing NVIDIA CUDA torch (cu121)...
"%VPY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] torch install failed.
    pause
    exit /b 1
)
goto install_reqs

:torch_amd
echo.
echo [WARNING] 官方 PyTorch 暂不支持在 Windows 原生环境安装 AMD ROCm 版本！
echo.
echo 给你的建议：
echo (1) 如果你只是想跑推理，请直接重选 1 (CPU 版)。
echo (2) 如果你需要用 AMD 显卡进行训练，请使用微软 WSL2 安装 Linux 子系统，并在其中运行 Linux 版脚本。
echo.
echo 自动为您切换回 CPU 安装，请稍候...
:: Windows 下使用 ping 实现几秒钟延时，比 timeout 更通用
ping 127.0.0.1 -n 4 >nul
goto torch_cpu

:install_reqs
echo [3/4] Installing requirements...
chcp 65001 >nul

echo   - [Step 1] Fix requirements.txt encoding...
"%VPY%" -c "import os; open('requirements.txt', 'w', encoding='utf-8').write(open('requirements.txt', 'r', encoding='utf-8', errors='ignore').read())"

echo   - [Step 2] Pre-install pyyaml (using newer pip to avoid compile errors)...
"%VPY%" -m pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple

echo   - [Step 3] Downgrade pip to 19.2 (Bypass omegaconf metadata check)...
"%VPY%" -m pip install pip==19.2

echo   - [Step 4] Install all requirements...
"%VPY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo   - [Step 5] Upgrade pip back to 24.1 for speed...
"%VPY%" -m pip install pip==24.1

echo   - [Step 6] Install python cmake...
"%VPY%" -m pip install cmake

if errorlevel 1 (
    echo.
    echo [ERROR] requirements install failed.
    echo.
    echo 【Windows用户请注意】：
    echo 如果报错 "Could not find 'cmake' executable!"，说明你系统没有 CMake。
    echo 解决方法：去官网 https://cmake.org/download/ 下载 .msi
    echo 安装时【务必勾选】 "Add CMake to the system PATH for all users"
    echo 装完后【关闭】CMD窗口，重新运行脚本。
    echo.
    pause
    exit /b 1
)

echo [4/4] Checking pretrained models...
if exist "pretrain\checkpoint_best_legacy_500.pt" (
    echo   ContentVec: OK
) else (
    echo   ContentVec: missing - upload later via web UI (Pretrain page)
)
if exist "pretrain\nsf_hifigan\model" (
    echo   NSF-HiFiGAN: OK
) else (
    echo   NSF-HiFiGAN: missing - upload later via web UI (Pretrain page)
)

echo.
echo ============================================
echo  Install done. Start the server:  start.bat
echo ============================================
pause