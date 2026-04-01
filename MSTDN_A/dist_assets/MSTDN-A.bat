@echo off
title MSTDN-A Emotion Detection
color 0B
echo.
echo   =============================================
echo     MSTDN-A  Emotion Detection System
echo     Starting server...
echo   =============================================
echo.

cd /d "%~dp0"

:: Check if setup was done
if not exist ".setup_done" (
    echo   [!] First time? Run setup.bat first!
    echo.
    pause
    exit /b 1
)

echo   Dashboard:    http://localhost:8000
echo   YT Test:      http://localhost:8000/yt_test
echo   Multi Mode:   http://localhost:8000/multi
echo   YT Multi:     http://localhost:8000/yt_test/multi
echo.
echo   Press Ctrl+C to stop the server.
echo.

:: Open browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

:: Start server
python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
pause
