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
echo   2 = CUDA GPU (needed for real training, ~2.5GB download)
:torch_choice
set "TORCH_CHOICE="
set /p TORCH_CHOICE="Choice [1/2]: "
if "%TORCH_CHOICE%"=="1" goto torch_cpu
if "%TORCH_CHOICE%"=="2" goto torch_cuda
echo Invalid choice, please enter 1 or 2.
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
echo Installing CUDA torch (cu121)...
"%VPY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] torch install failed.
    pause
    exit /b 1
)
goto install_reqs

:install_reqs
echo [3/4] Installing requirements...

:: 1. 降级 pip 解决 omegaconf 元数据死锁问题
echo   - Step 1: 降级 pip 至 24.1 版本...
"%VPY%" -m pip install pip==24.1

:: 2. 提前安装 omegaconf (阿里云镜像缺少2.0.6，强制切清华源解决)
echo   - Step 2: 预装 omegaconf 2.0.6 (使用清华源)...
"%VPY%" -m pip install omegaconf==2.0.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 3. 预装 Python 版 cmake
echo   - Step 3: 预装 Python cmake...
"%VPY%" -m pip install cmake

:: 4. 正式安装项目依赖
echo   - Step 4: 安装项目全部依赖...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] requirements install failed.
    echo.
    echo 【提示：如果你遇到 "Could not find 'cmake' executable!"】
    echo 说明你电脑里缺少 Windows 系统级别的 CMake 软件。
    echo 解决方法：
    echo 1. 立即访问官网：https://cmake.org/download/
    echo 2. 下载 Windows 64位版本 (.msi 安装包)。
    echo 3. 安装时【务必勾选】 "Add CMake to the system PATH for all users"！
    echo 4. 安装完成后，关掉当前 CMD 窗口，重新运行 install.bat 即可。
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