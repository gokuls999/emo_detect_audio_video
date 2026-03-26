"""
training_monitor.py  —  compact terminal-style monitor
Run: py -3 training_monitor.py
"""
import ctypes, json, time, tkinter as tk
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "finetune_status.json"
POLL_MS     = 2000

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def _keep_awake():
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    root.after(30_000, _keep_awake)

def _release_awake():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# -- terminal colour palette ---------------------------------------------------
BG      = "#0c0c0c"
BG2     = "#111111"
BORDER  = "#1e8c3a"
GREEN   = "#00ff41"
CYAN    = "#00d4ff"
YELLOW  = "#f9f940"
RED     = "#ff4444"
DIM     = "#1e5c2a"
DIMTXT  = "#3a8a50"
WHITE   = "#e0e0e0"
FONT    = "Consolas"

# -- window --------------------------------------------------------------------
root = tk.Tk()
root.title("MSTDN-A Monitor")
root.configure(bg=BG)
root.resizable(False, False)

W, H = 530, 218
sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

# -- title bar -----------------------------------------------------------------
title_bar = tk.Frame(root, bg=BG2, highlightthickness=1, highlightbackground=BORDER)
title_bar.pack(fill="x", padx=2, pady=(2, 0))

tk.Label(title_bar, text="[ MSTDN-A  Fine-tuning ]",
         font=(FONT, 9, "bold"), bg=BG2, fg=GREEN).pack(side="left", padx=8, pady=4)

live_lbl = tk.Label(title_bar, text="[LIVE]",
                     font=(FONT, 9, "bold"), bg=BG2, fg=GREEN)
live_lbl.pack(side="right", padx=8, pady=4)

# -- progress canvas -----------------------------------------------------------
cv_frame = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground=BORDER)
cv_frame.pack(fill="x", padx=2, pady=(1, 0))

BAR_W  = 300   # filled bar pixel width
BAR_H  = 12
PAD_X  = 10

def make_bar_row(parent, label_text, bar_color):
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", padx=PAD_X, pady=(5, 1))

    tk.Label(row, text=label_text, font=(FONT, 8), bg=BG,
             fg=DIMTXT, width=7, anchor="w").pack(side="left")

    cv = tk.Canvas(row, width=BAR_W, height=BAR_H, bg=DIM,
                   highlightthickness=1, highlightbackground=BORDER)
    cv.pack(side="left", padx=(2, 6))

    pct_lbl = tk.Label(row, text=" 0.0%", font=(FONT, 8, "bold"),
                        bg=BG, fg=bar_color, width=6, anchor="w")
    pct_lbl.pack(side="left")

    info_lbl = tk.Label(row, text="", font=(FONT, 8),
                         bg=BG, fg=DIMTXT, anchor="w")
    info_lbl.pack(side="left", padx=(4, 0))

    return cv, pct_lbl, info_lbl

epoch_cv, epoch_pct, epoch_info = make_bar_row(cv_frame, "Epoch ", GREEN)
batch_cv, batch_pct, batch_info = make_bar_row(cv_frame, "Batch ", CYAN)

def draw_bar(cv, pct, color):
    cv.delete("all")
    filled = int(BAR_W * pct / 100)
    if filled > 0:
        cv.create_rectangle(0, 0, filled, BAR_H, fill=color, outline="")
    # tick marks every 25%
    for x in [BAR_W//4, BAR_W//2, BAR_W*3//4]:
        cv.create_line(x, 0, x, BAR_H, fill=DIM, width=1)

# -- stats row -----------------------------------------------------------------
stats_frame = tk.Frame(root, bg=BG2, highlightthickness=1, highlightbackground=BORDER)
stats_frame.pack(fill="x", padx=2, pady=(1, 0))

stats_lbl = tk.Label(stats_frame,
    text="  Loss --      Train --%   Val --%   Best --%",
    font=(FONT, 8), bg=BG2, fg=WHITE, anchor="w")
stats_lbl.pack(fill="x", padx=8, pady=(4, 1))

time_lbl = tk.Label(stats_frame,
    text="  ETA  --      Finish --:--   LR --   Device --",
    font=(FONT, 8), bg=BG2, fg=DIMTXT, anchor="w")
time_lbl.pack(fill="x", padx=8, pady=(0, 4))

# -- status bar ----------------------------------------------------------------
status_frame = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground=BORDER)
status_frame.pack(fill="x", padx=2, pady=(1, 2))

status_row = tk.Frame(status_frame, bg=BG)
status_row.pack(fill="x", padx=8, pady=(4, 1))

# rotating arc spinner
_arc_angle  = 0
_spinner_on = True

spin_cv = tk.Canvas(status_row, width=14, height=14, bg=BG, highlightthickness=0)
spin_cv.pack(side="left", padx=(0, 5))

status_lbl = tk.Label(status_row, text="Waiting for training...",
                       font=(FONT, 8), bg=BG, fg=DIMTXT, anchor="w")
status_lbl.pack(side="left")

update_lbl = tk.Label(status_frame, text="",
                       font=(FONT, 7), bg=BG, fg="#1a3a22", anchor="w")
update_lbl.pack(fill="x", padx=8, pady=(0, 4))

def _spin():
    global _arc_angle
    if _spinner_on:
        spin_cv.delete("all")
        spin_cv.create_oval(1, 1, 13, 13, outline="#1e5c2a", width=2)
        spin_cv.create_arc(1, 1, 13, 13, start=_arc_angle, extent=120,
                            outline=GREEN, width=2, style="arc")
        _arc_angle = (_arc_angle + 14) % 360
    root.after(40, _spin)

_blink_on = True
def _blink():
    global _blink_on
    _blink_on = not _blink_on
    live_lbl.config(fg=GREEN if _blink_on else BG2)
    root.after(800, _blink)

root.after(40, _spin)
root.after(800, _blink)

# -- poll ----------------------------------------------------------------------
def _poll():
    try:
        if not STATUS_FILE.exists():
            root.after(POLL_MS, _poll)
            return

        with open(STATUS_FILE, encoding="utf-8") as f:
            s = json.load(f)

        phase        = s.get("phase", "?")
        epoch        = s.get("epoch", 0)
        total_epochs = s.get("total_epochs", 20)
        batch        = s.get("batch", 0)
        total_bat    = s.get("total_batches", 1)
        loss         = s.get("loss")
        train_acc    = s.get("train_acc_pct", 0)
        val_acc      = s.get("val_acc_pct", 0)
        best_acc     = s.get("best_val_acc_pct", 0)
        lr           = s.get("lr", "--")
        eta          = s.get("eta", "--")
        device       = s.get("device", "--")
        updated_at   = s.get("updated_at", "")
        elapsed      = s.get("elapsed", "--")
        done         = (phase == "done")

        # epoch bar
        ep_pct = min((epoch - 1 + batch / max(total_bat, 1)) / total_epochs * 100, 100)
        if done: ep_pct = 100
        ep_color = GREEN if done else (YELLOW if ep_pct > 60 else GREEN)
        draw_bar(epoch_cv, ep_pct, ep_color)
        epoch_pct.config(text=f"{ep_pct:5.1f}%", fg=ep_color)
        epoch_info.config(text=f"{epoch}/{total_epochs}")

        # batch bar
        bat_pct = min(batch / max(total_bat, 1) * 100, 100)
        if phase in ("validating", "done"): bat_pct = 100
        draw_bar(batch_cv, bat_pct, CYAN)
        batch_pct.config(text=f"{bat_pct:5.1f}%")
        batch_info.config(text=f"{batch}/{total_bat}")

        # stats
        loss_s = f"{loss:.4f}" if loss is not None else "  --  "
        stats_lbl.config(
            text=f"  Loss {loss_s}   Train {train_acc:4.1f}%   "
                 f"Val {val_acc:4.1f}%   Best {best_acc:4.1f}%"
        )

        # estimated finish time
        finish_str = "--:--"
        eta_secs = s.get("eta_seconds")
        if eta_secs is not None:
            finish_ts = time.localtime(time.time() + eta_secs)
            finish_str = time.strftime("%H:%M", finish_ts)

        time_lbl.config(
            text=f"  ETA  {eta:<10}  Finish ~{finish_str}   "
                 f"LR {lr}   {device.upper()}"
        )

        # status
        if done:
            global _spinner_on
            _spinner_on = False
            spin_cv.delete("all")
            spin_cv.create_oval(1, 1, 13, 13, fill=GREEN, outline=GREEN)
            spin_cv.create_text(7, 7, text="OK", fill=BG,
                                 font=(FONT, 6, "bold"))
            live_lbl.config(text="[DONE]", fg=GREEN)
            status_lbl.config(
                text=f"Training complete!  Best val: {best_acc:.1f}%  Elapsed: {elapsed}",
                fg=GREEN
            )
            _release_awake()
        elif phase == "validating":
            status_lbl.config(
                text=f"Epoch {epoch}/{total_epochs} -- Validating...",
                fg=YELLOW
            )
        else:
            status_lbl.config(
                text=f"Epoch {epoch}/{total_epochs} -- Batch {batch}/{total_bat}  ETA {eta}",
                fg=WHITE
            )

        update_lbl.config(text=f"  Last update: {updated_at}")

    except Exception as e:
        status_lbl.config(text=f"Error: {str(e)[:60]}", fg=RED)

    root.after(POLL_MS, _poll)

root.after(500, _poll)
root.after(100, _keep_awake)
root.mainloop()
_release_awake()
