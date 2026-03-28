@echo off
title MSTDN-A Setup
echo ============================================
echo   MSTDN-A Emotion Detection - CPU Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or 3.11 from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found
echo.

:: Install dependencies
echo [1/3] Installing Python dependencies (this may take a few minutes)...
echo.
pip install -r requirements-cpu.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages may have failed. Trying individually...
    pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    pip install librosa sounddevice fastapi "uvicorn[standard]" pydantic
    pip install opencv-python deepface tf-keras tensorflow-cpu
    pip install numpy scipy pandas yt-dlp imageio-ffmpeg transformers
)
echo.

:: Verify torch
echo [2/3] Verifying PyTorch installation...
python -c "import torch; print(f'  PyTorch {torch.__version__} | Device: cpu')"
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo.

:: Verify checkpoint
echo [3/3] Checking model checkpoint...
if exist "checkpoints\online_tuned.pt" (
    echo [OK] Model checkpoint found: checkpoints\online_tuned.pt
) else (
    echo [ERROR] Model checkpoint missing! Place online_tuned.pt in checkpoints\
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo   To start the server, run:
echo     run.bat
echo.
echo   Then open in your browser:
echo     Dashboard:  http://localhost:8000
echo     YT Test:    http://localhost:8000/yt_test
echo.
pause
