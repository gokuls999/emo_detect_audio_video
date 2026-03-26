"""
One-click restart for MSTDN-A Stage 3 after power cut.
Double-click this file or: py -3.11 restart_training.py
"""
import subprocess, sys, time, re, torch
from pathlib import Path

BASE     = Path(__file__).parent
CKPT     = BASE / "checkpoints" / "stage3_resume.pt"
LOG_FILE = BASE / "logs" / "stage3_current.log"
LOG_FILE.parent.mkdir(exist_ok=True)

print("=" * 50)
print("  MSTDN-A — Power Recovery Restart")
print("=" * 50)

if not CKPT.exists():
    print("ERROR: No checkpoint found!")
    input("Press Enter to exit.")
    sys.exit(1)

ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
epoch = ckpt.get("epoch", "?")
print(f"Resuming from epoch {epoch}/60\n")

# Clear old log
LOG_FILE.write_text("")

# Update LOG_FILE path in training_ui.py and keep_awake.py
for fname in ["training_ui.py", "keep_awake.py"]:
    fpath = BASE / fname
    txt   = fpath.read_text(encoding="utf-8")
    txt   = re.sub(r'LOG_FILE\s*=\s*Path\(r?"[^"]*"\)',
                   f'LOG_FILE      = Path(r"{str(LOG_FILE)}")', txt)
    fpath.write_text(txt, encoding="utf-8")

# Start all 3 processes
train = subprocess.Popen(
    [sys.executable, "-m", "training.stage3_distill"],
    cwd=str(BASE),
    stdout=open(LOG_FILE, "w"),
    stderr=subprocess.STDOUT,
)
time.sleep(3)
ui    = subprocess.Popen([sys.executable, "training_ui.py"],  cwd=str(BASE))
ka    = subprocess.Popen([sys.executable, "keep_awake.py"],   cwd=str(BASE))

print(f"Training PID  : {train.pid}")
print(f"UI PID        : {ui.pid}")
print(f"Keep-awake PID: {ka.pid}")
print(f"Log           : {LOG_FILE}")
print("\nAll running! This window can be closed.")
input("Press Enter to exit.")
