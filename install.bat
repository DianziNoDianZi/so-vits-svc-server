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
%VPY% -m pip install -q --upgrade pip setuptools wheel

echo [2/4] Choose torch build:
echo   1 = CPU (smaller, inference only, slow training)
echo   2 = CUDA GPU (needed for real training, ~2.5GB download)
set /p TORCH_CHOICE="Choice [1/2]: "
if "%TORCH_CHOICE%"=="2" (
    echo Installing CUDA torch (cu121)...
    %VPY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing CPU torch...
    %VPY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    echo [ERROR] torch install failed. Check network and try again.
    pause
    exit /b 1
)

echo [3/4] Installing requirements...
%VPY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] requirements install failed.
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
