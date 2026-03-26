@echo off
cd /d "%~dp0"
echo Starting MSTDN-A dashboard on http://127.0.0.1:8000
echo Keep this window open while using the dashboard.
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000 --log-level info
echo.
echo Dashboard stopped.
pause
