@echo off
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

:: 2. 提前安装一个合法的 omegaconf，避免依赖冲突
echo   - Step 2: 预装 omegaconf 2.0.6...
"%VPY%" -m pip install omegaconf==2.0.6

:: 3. 预装 Python 版 cmake
echo   - Step 3: 预装 Python cmake...
"%VPY%" -m pip install cmake

:: 4. 正式安装项目依赖（此时不会再有冲突）
echo   - Step 4: 安装项目全部依赖...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] requirements install failed.
    echo.
    echo 【特别提示：如果你是 Windows 系统】：
    echo 如果上方报错是 "Could not find 'cmake' executable!"，
    echo 说明 pip 安装的 cmake 没有生效，你还需要手动安装【系统 CMake】。
    echo.
    echo 解决方法：
    echo 1. 访问 https://cmake.org/download/ 下载 Windows 版 CMake (.msi)
    echo 2. 安装时【务必勾选】 "Add CMake to the system PATH for all users"
    echo 3. 安装完毕后，关掉这个 CMD 窗口，重新运行本脚本即可。
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