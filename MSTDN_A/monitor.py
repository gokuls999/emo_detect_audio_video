"""
Live training monitor — shows animated progress of Stage 2 training.
Run from MSTDN_A/: py -3.11 monitor.py
"""
import re
import sys
import time
from pathlib import Path

LOG_FILE = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude\c--Users-ADMIN-Desktop-Bineetha---emoDet\tasks\blmik2m75.output"
)

TOTAL_EPOCHS  = 50
TOTAL_BATCHES = 459
BAR_WIDTH     = 35
SPINNERS      = ["|", "/", "-", "\\"]

# Regex patterns
RE_EPOCH  = re.compile(r"stage2 epoch\s+(\d+).*?(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s+([\d.]+)it/s\]")
RE_LOSS   = re.compile(r"Epoch\s+(\d+)/\d+\s+(.*)")
RE_RESUME = re.compile(r"Resumed at epoch\s+(\d+)")

def bar(current, total, width=BAR_WIDTH):
    filled = int(width * current / max(total, 1))
    return "#" * filled + "-" * (width - filled)

def clear_lines(n):
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")

def format_time(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m}m"

def read_tail(path, n=300):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return []

def parse_state(lines):
    state = {
        "epoch": 0, "batch": 0, "speed": 0.0,
        "elapsed_s": 0, "remaining_s": 0,
        "losses": {}, "last_completed_epoch": 0,
    }
    for line in lines:
        m = RE_RESUME.search(line)
        if m:
            state["last_completed_epoch"] = int(m.group(1))

        m = RE_EPOCH.search(line)
        if m:
            state["epoch"]       = int(m.group(1))
            state["batch"]       = int(m.group(2))
            em, es               = int(m.group(4)), int(m.group(5))
            rm, rs               = int(m.group(6)), int(m.group(7))
            state["elapsed_s"]   = em * 60 + es
            state["remaining_s"] = rm * 60 + rs
            state["speed"]       = float(m.group(8))

        m = RE_LOSS.search(line)
        if m:
            state["last_completed_epoch"] = int(m.group(1))
            pairs = re.findall(r"(\w+)=([\d.]+)", m.group(2))
            state["losses"] = {k: float(v) for k, v in pairs}

    return state

def main():
    print("\033[?25l", end="")   # hide cursor
    try:
        spin_i   = 0
        printed  = 0

        print()
        print("  \033[1;36mMSTDN-A  Stage 2 Teacher — Live Training Monitor\033[0m")
        print("  Press  Ctrl+C  to exit\n")
        printed = 3

        while True:
            lines = read_tail(LOG_FILE)
            s     = parse_state(lines)
            spin  = SPINNERS[spin_i % len(SPINNERS)]
            spin_i += 1

            epoch      = s["epoch"] or (s["last_completed_epoch"] + 1)
            batch      = s["batch"]
            speed      = s["speed"]
            ep_done    = s["last_completed_epoch"]
            losses     = s["losses"]

            # ── overall progress ──────────────────────────────────────────
            overall_batches_done = ep_done * TOTAL_BATCHES + batch
            overall_total        = TOTAL_EPOCHS * TOTAL_BATCHES
            overall_pct          = overall_batches_done / max(overall_total, 1) * 100

            # ── time estimates ────────────────────────────────────────────
            remaining_this_epoch = s["remaining_s"]
            remaining_other      = (TOTAL_EPOCHS - epoch) * (TOTAL_BATCHES / max(speed, 0.1))
            total_remaining_min  = (remaining_this_epoch + remaining_other) / 60

            # ── batch progress this epoch ─────────────────────────────────
            ep_pct = batch / TOTAL_BATCHES * 100

            # ── build display ─────────────────────────────────────────────
            if printed > 3:
                clear_lines(printed - 3)

            lines_out = []

            # Overall
            lines_out.append(
                f"  {spin} \033[1mOverall\033[0m  [{bar(overall_batches_done, overall_total)}]"
                f"  {overall_pct:5.1f}%   ETA {format_time(total_remaining_min)}"
            )

            # Epoch
            lines_out.append(
                f"    \033[33mEpoch {epoch:>2}/{TOTAL_EPOCHS}\033[0m"
                f"  [{bar(batch, TOTAL_BATCHES)}]"
                f"  {ep_pct:5.1f}%   {batch}/{TOTAL_BATCHES} batches   {speed:.1f} it/s"
            )

            # Losses
            if losses:
                loss_str = "   ".join(
                    f"\033[36m{k}\033[0m={v:.4f}"
                    for k, v in losses.items()
                    if k in ("loss", "ce", "bce", "stress", "valence", "arousal")
                )
                lines_out.append(f"    \033[90mLast epoch losses:\033[0m  {loss_str}")
            else:
                lines_out.append(f"    \033[90mWaiting for first epoch to complete...\033[0m")

            # Completed epochs
            lines_out.append(
                f"    \033[90mCompleted epochs: {ep_done}/{TOTAL_EPOCHS}"
                f"   Checkpoint saved after every epoch\033[0m"
            )

            # GPU bar (static — confirmed 301 MB / 11264 MB)
            gpu_pct = 301 / 11264 * 100
            lines_out.append(
                f"    \033[35mGPU VRAM\033[0m  [{bar(301, 11264, 25)}]"
                f"  {gpu_pct:.1f}%  (~301 MB / 11 GB)"
            )

            lines_out.append("")

            for ln in lines_out:
                print(ln)

            printed = 3 + len(lines_out)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\033[?25h", end="")   # restore cursor
        print("\n  Monitor stopped. Training continues in background.\n")

if __name__ == "__main__":
    main()
