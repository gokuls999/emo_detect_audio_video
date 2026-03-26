"""MELD Download Progress Monitor — auto-closes when done."""
import os, time, tkinter as tk
from tkinter import ttk

FILE   = r"C:\Users\ADMIN\Desktop\Bineetha - emoDet\MELD-zip\MELD.Raw.tar.gz"
TOTAL  = 10_878_146_150   # bytes (confirmed via Content-Length)

BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
TEXT    = "#c9d1d9"
MUTED   = "#8b949e"

root = tk.Tk()
root.title("MELD Download")
root.configure(bg=BG)
root.resizable(False, False)

W, H = 520, 220
sw   = root.winfo_screenwidth()
sh   = root.winfo_screenheight()
root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

# ── widgets ──────────────────────────────────────────────────────────────────
tk.Label(root, text="MELD Dataset Download", font=("Helvetica", 13, "bold"),
         bg=BG, fg=TEXT).pack(pady=(18, 2))
tk.Label(root, text="MELD.Raw.tar.gz  ·  10.1 GB  ·  Friends TV emotion dataset",
         font=("Helvetica", 8), bg=BG, fg=MUTED).pack()

frame = tk.Frame(root, bg=CARD, bd=0, highlightthickness=1,
                 highlightbackground=BORDER)
frame.pack(fill="x", padx=24, pady=14)

style = ttk.Style()
style.theme_use("default")
style.configure("dl.Horizontal.TProgressbar",
                troughcolor=BORDER, background=ACCENT,
                thickness=18, borderwidth=0)

bar = ttk.Progressbar(frame, style="dl.Horizontal.TProgressbar",
                      orient="horizontal", length=470, maximum=100)
bar.pack(padx=16, pady=(16, 8))

lbl_pct  = tk.Label(frame, text="0.0%", font=("Helvetica", 20, "bold"),
                    bg=CARD, fg=ACCENT)
lbl_pct.pack()

info_row = tk.Frame(frame, bg=CARD)
info_row.pack(pady=(4, 14))

lbl_size  = tk.Label(info_row, text="0 MB / 10,134 MB",
                     font=("Helvetica", 9), bg=CARD, fg=TEXT)
lbl_size.pack(side="left", padx=16)

lbl_speed = tk.Label(info_row, text="-- MB/s",
                     font=("Helvetica", 9), bg=CARD, fg=MUTED)
lbl_speed.pack(side="left", padx=16)

lbl_eta   = tk.Label(info_row, text="ETA: calculating…",
                     font=("Helvetica", 9), bg=CARD, fg=MUTED)
lbl_eta.pack(side="left", padx=16)

lbl_done  = tk.Label(root, text="", font=("Helvetica", 10, "bold"),
                     bg=BG, fg=GREEN)
lbl_done.pack()

# ── polling ───────────────────────────────────────────────────────────────────
_prev_size = 0
_prev_time = time.time()
_history   = []   # (timestamp, bytes) for smoothed speed

def _poll():
    global _prev_size, _prev_time

    try:
        current = os.path.getsize(FILE)
    except FileNotFoundError:
        current = 0

    now   = time.time()
    delta_bytes = current - _prev_size
    delta_time  = now - _prev_time

    # keep a 10-sample rolling window for speed smoothing
    _history.append((now, current))
    if len(_history) > 10:
        _history.pop(0)

    if len(_history) >= 2:
        span_bytes = _history[-1][1] - _history[0][1]
        span_time  = _history[-1][0] - _history[0][0]
        speed_bps  = span_bytes / max(span_time, 0.001)
    else:
        speed_bps = 0

    pct       = min(current / TOTAL * 100, 100)
    mb_done   = current / 1_048_576
    mb_total  = TOTAL   / 1_048_576
    speed_mb  = speed_bps / 1_048_576
    remaining = (TOTAL - current) / max(speed_bps, 1)

    # colour bar by progress
    if pct >= 100:
        col = GREEN
    elif pct > 60:
        col = ACCENT
    elif pct > 30:
        col = YELLOW
    else:
        col = ACCENT

    style.configure("dl.Horizontal.TProgressbar", background=col)
    bar["value"] = pct
    lbl_pct.config(text=f"{pct:.1f}%", fg=col)
    lbl_size.config(text=f"{mb_done:,.0f} MB / {mb_total:,.0f} MB")
    lbl_speed.config(text=f"{speed_mb:.2f} MB/s" if speed_bps > 0 else "-- MB/s")

    if speed_bps > 0 and pct < 100:
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        eta  = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        lbl_eta.config(text=f"ETA: {eta}")
    elif pct >= 100:
        lbl_eta.config(text="")

    _prev_size = current
    _prev_time = now

    if pct >= 100:
        bar["value"] = 100
        lbl_pct.config(text="100%  ✓", fg=GREEN)
        lbl_size.config(text=f"{mb_total:,.0f} MB / {mb_total:,.0f} MB")
        lbl_speed.config(text="Done")
        lbl_done.config(text="Download complete — closing in 3 seconds…")
        root.after(3000, root.destroy)
    else:
        root.after(1000, _poll)

root.after(500, _poll)
root.mainloop()
