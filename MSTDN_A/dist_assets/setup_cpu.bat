@echo off
title MSTDN-A CPU Setup
color 0E
echo.
echo   =============================================
echo     MSTDN-A Emotion Detection - CPU Setup
echo   =============================================
echo.

cd /d "%~dp0"

:: ── Check Python ────────────────────────────────
echo   [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR] Python not found!
    echo   Install Python 3.10 or 3.11 from https://www.python.org
    echo   IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   [OK] Python %PYVER%
echo.

:: ── Install PyTorch CPU ────────────────────────
echo   [2/5] Installing PyTorch (CPU-only)...
echo   (This may take several minutes on first install)
echo.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1
echo.

:: ── Install other dependencies ─────────────────
echo   [3/5] Installing dependencies...
echo.
pip install -r requirements-cpu.txt 2>&1
echo.

:: ── Setup model caches ─────────────────────────
echo   [4/5] Setting up model caches...

:: Copy InsightFace models
if exist "insightface_models\buffalo_l" (
    if not exist "%USERPROFILE%\.insightface\models\buffalo_l" (
        echo   Copying InsightFace models...
        mkdir "%USERPROFILE%\.insightface\models\buffalo_l" 2>nul
        xcopy /s /y "insightface_models\buffalo_l\*" "%USERPROFILE%\.insightface\models\buffalo_l\" >nul
    ) else (
        echo   [OK] InsightFace models already present
    )
)

:: Copy wav2vec2 cache
if exist "hf_cache\models--facebook--wav2vec2-base" (
    if not exist "%USERPROFILE%\.cache\huggingface\hub\models--facebook--wav2vec2-base" (
        echo   Copying wav2vec2-base model cache...
        mkdir "%USERPROFILE%\.cache\huggingface\hub" 2>nul
        xcopy /s /y "hf_cache\models--facebook--wav2vec2-base\*" "%USERPROFILE%\.cache\huggingface\hub\models--facebook--wav2vec2-base\" >nul
    ) else (
        echo   [OK] wav2vec2-base cache already present
    )
)
echo.

:: ── Verify installation ────────────────────────
echo   [5/5] Verifying installation...
python -c "import torch; print(f'  PyTorch {torch.__version__} (CPU)')" 2>&1
if errorlevel 1 (
    echo   [ERROR] PyTorch verification failed!
    pause
    exit /b 1
)

:: Check checkpoint
if exist "checkpoints\online_tuned.pt" (
    echo   [OK] Trained checkpoint found
) else if exist "checkpoints\english_finetune_r2.pt" (
    echo   [OK] Base checkpoint found
) else (
    echo   [ERROR] No model checkpoint found in checkpoints\
    pause
    exit /b 1
)

:: Mark setup complete
echo done > .setup_done

echo.
echo   =============================================
echo     Setup Complete!
echo   =============================================
echo.
echo   Double-click MSTDN-A.bat to start.
echo.
pause
