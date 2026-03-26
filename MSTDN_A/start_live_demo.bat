@echo off
cd /d "%~dp0"
echo Starting MSTDN-A live demo...
echo Keep this window open while using the demo.
python live_demo.py
echo.
echo Live demo stopped.
pause
