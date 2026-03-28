@echo off
title MSTDN-A Server
echo.
echo   MSTDN-A Emotion Detection Server
echo   ---------------------------------
echo   Dashboard:  http://localhost:8000
echo   YT Test:    http://localhost:8000/yt_test
echo   Press Ctrl+C to stop
echo.
python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
pause
