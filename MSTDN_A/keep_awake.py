"""
Prevents Windows from sleeping or turning off the screen during training.
Automatically exits when training completes.
Run: py -3.11 keep_awake.py
"""
import ctypes
import time
from pathlib import Path

LOG_FILE = Path(r"C:\Users\ADMIN\AppData\Local\Temp\claude\c--Users-ADMIN-Desktop-Bineetha---emoDet\tasks\bttoe2cga.output")

# Windows SetThreadExecutionState flags
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

def allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def training_done():
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
        return "Stage 5 complete" in text or "Epoch 20/20" in text
    except Exception:
        return False

prevent_sleep()
print("System keep-awake ACTIVE — screen and sleep are blocked.")
print("Monitoring training... will release automatically when done.\n")

try:
    while True:
        if training_done():
            allow_sleep()
            print("Training complete — keep-awake released. System can sleep normally now.")
            break
        prevent_sleep()   # renew every 30s to be safe
        time.sleep(30)
except KeyboardInterrupt:
    allow_sleep()
    print("Keep-awake manually stopped — system can sleep normally now.")
