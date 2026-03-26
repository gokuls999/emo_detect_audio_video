"""
MSTDN-A Training Monitor UI
Run: py -3.11 training_ui.py
"""
import re
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import threading
import time

LOG_FILE      = Path(r"C:\Users\ADMIN\AppData\Local\Temp\claude\c--Users-ADMIN-Desktop-Bineetha---emoDet\tasks\bttoe2cga.output")
TOTAL_EPOCHS  = 20
TOTAL_BATCHES = 402
STAGE_LABEL   = "Stage 5 — Acoustic Domain Augmentation"

RE_EPOCH  = re.compile(r"stage5 epoch\s+(\d+).*?(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s+([\d.]+)it/s\]")
RE_LOSS   = re.compile(r"Epoch\s+(\d+)/\d+\s+(.*)")
RE_RESUME = re.compile(r"Resumed at epoch\s+(\d+)")

BG        = "#0d1117"
CARD      = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"
BAR_BG    = "#21262d"
BAR_FILL  = "#1f6feb"
BAR_EP    = "#238636"


def parse_log():
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
    except Exception:
        return {}
    epoch = 0; batch = 0; speed = 0.0; ep_done = 0; losses = {}; remaining_s = 0
    for line in lines:
        m = RE_RESUME.search(line)
        if m:
            ep_done = int(m.group(1))
        m = RE_EPOCH.search(line)
        if m:
            epoch = int(m.group(1)); batch = int(m.group(2))
            remaining_s = int(m.group(6)) * 60 + int(m.group(7))
            speed = float(m.group(8))
        m = RE_LOSS.search(line)
        if m:
            ep_done = int(m.group(1))
            losses = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(2))}
    return {
        "epoch":       epoch or (ep_done + 1),
        "batch":       batch,
        "speed":       speed,
        "ep_done":     ep_done,
        "losses":      losses,
        "remaining_s": remaining_s,
    }


class TrainingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MSTDN-A  —  Training Monitor")
        self.root.configure(bg=BG)
        self.root.geometry("680x520")
        self.root.resizable(False, False)

        self.spin_chars = ["◐", "◓", "◑", "◒"]
        self.spin_idx   = 0
        self.pulse      = 0
        self.running    = True

        self._build_ui()
        self._start_update_thread()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=CARD, pady=14)
        header.pack(fill="x")

        self.spin_lbl = tk.Label(header, text="◐", font=("Segoe UI", 22),
                                 bg=CARD, fg=ACCENT)
        self.spin_lbl.pack(side="left", padx=(20, 8))

        tk.Label(header, text=f"MSTDN-A  {STAGE_LABEL}",
                 font=("Segoe UI", 15, "bold"), bg=CARD, fg=TEXT).pack(side="left")

        self.status_dot = tk.Label(header, text="● RUNNING",
                                   font=("Segoe UI", 9, "bold"), bg=CARD, fg=GREEN)
        self.status_dot.pack(side="right", padx=20)

        # Separator
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG, padx=24, pady=18)
        body.pack(fill="both", expand=True)

        # ── Overall Progress ─────────────────────────────────────────────────
        self._section(body, "OVERALL PROGRESS")

        self.overall_pct_lbl = tk.Label(body, text="0.0%", font=("Segoe UI", 28, "bold"),
                                        bg=BG, fg=ACCENT)
        self.overall_pct_lbl.pack(anchor="w")

        overall_bar_frame = tk.Frame(body, bg=BAR_BG, height=18, bd=0,
                                     highlightbackground=BORDER, highlightthickness=1)
        overall_bar_frame.pack(fill="x", pady=(4, 2))
        overall_bar_frame.pack_propagate(False)
        self.overall_bar = tk.Frame(overall_bar_frame, bg=BAR_FILL, height=18)
        self.overall_bar.place(x=0, y=0, relheight=1, width=0)
        self._overall_bar_frame = overall_bar_frame

        self.overall_info = tk.Label(body, text="", font=("Segoe UI", 9),
                                     bg=BG, fg=SUBTEXT)
        self.overall_info.pack(anchor="w", pady=(2, 14))

        # ── Epoch Progress ───────────────────────────────────────────────────
        self._section(body, "CURRENT EPOCH")

        ep_row = tk.Frame(body, bg=BG)
        ep_row.pack(fill="x")
        self.epoch_lbl = tk.Label(ep_row, text="Epoch  0 / 50",
                                  font=("Segoe UI", 13, "bold"), bg=BG, fg=TEXT)
        self.epoch_lbl.pack(side="left")
        self.ep_pct_lbl = tk.Label(ep_row, text="0.0%",
                                   font=("Segoe UI", 13, "bold"), bg=BG, fg=BAR_EP)
        self.ep_pct_lbl.pack(side="right")

        ep_bar_frame = tk.Frame(body, bg=BAR_BG, height=14,
                                highlightbackground=BORDER, highlightthickness=1)
        ep_bar_frame.pack(fill="x", pady=(4, 2))
        ep_bar_frame.pack_propagate(False)
        self.ep_bar = tk.Frame(ep_bar_frame, bg=BAR_EP, height=14)
        self.ep_bar.place(x=0, y=0, relheight=1, width=0)
        self._ep_bar_frame = ep_bar_frame

        self.ep_info = tk.Label(body, text="", font=("Segoe UI", 9),
                                bg=BG, fg=SUBTEXT)
        self.ep_info.pack(anchor="w", pady=(2, 14))

        # ── Stats row ────────────────────────────────────────────────────────
        stats_frame = tk.Frame(body, bg=BG)
        stats_frame.pack(fill="x", pady=(0, 14))

        self.card_speed    = self._stat_card(stats_frame, "SPEED",        "-- it/s")
        self.card_eta      = self._stat_card(stats_frame, "TIME LEFT",    "--")
        self.card_finishes = self._stat_card(stats_frame, "FINISHES AT",  "--:--")
        self.card_vram     = self._stat_card(stats_frame, "GPU VRAM",     "301 MB / 11 GB")
        self.card_ckpt     = self._stat_card(stats_frame, "CHECKPOINT",   "saved / epoch")

        # ── Loss row ─────────────────────────────────────────────────────────
        self._section(body, "LAST EPOCH LOSSES")
        loss_frame = tk.Frame(body, bg=BG)
        loss_frame.pack(fill="x")
        self.loss_labels = {}
        for name in ("loss", "ce", "bce", "stress", "valence", "arousal"):
            col = tk.Frame(loss_frame, bg=CARD, padx=10, pady=6,
                           highlightbackground=BORDER, highlightthickness=1)
            col.pack(side="left", padx=(0, 6))
            tk.Label(col, text=name.upper(), font=("Segoe UI", 7, "bold"),
                     bg=CARD, fg=SUBTEXT).pack()
            lbl = tk.Label(col, text="--", font=("Segoe UI", 11, "bold"),
                           bg=CARD, fg=YELLOW)
            lbl.pack()
            self.loss_labels[name] = lbl

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(0, 4))
        tk.Label(f, text=title, font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=SUBTEXT).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                               expand=True, padx=(8, 0), pady=6)

    def _stat_card(self, parent, label, default):
        card = tk.Frame(parent, bg=CARD, padx=12, pady=8,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(side="left", padx=(0, 8), fill="x", expand=True)
        tk.Label(card, text=label, font=("Segoe UI", 7, "bold"),
                 bg=CARD, fg=SUBTEXT).pack(anchor="w")
        val = tk.Label(card, text=default, font=("Segoe UI", 12, "bold"),
                       bg=CARD, fg=TEXT)
        val.pack(anchor="w")
        return val

    # ── Update logic ──────────────────────────────────────────────────────────

    def _start_update_thread(self):
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while self.running:
            data = parse_log()
            self.root.after(0, self._refresh_ui, data)
            time.sleep(1)

    def _refresh_ui(self, d):
        if not d:
            return

        epoch      = d["epoch"]
        batch      = d["batch"]
        speed      = d["speed"]
        ep_done    = d["ep_done"]
        losses     = d["losses"]
        rem_s      = d["remaining_s"]

        overall    = ep_done * TOTAL_BATCHES + batch
        total      = TOTAL_EPOCHS * TOTAL_BATCHES
        ov_pct     = overall / total * 100
        ep_pct     = batch / TOTAL_BATCHES * 100

        eta_min    = (rem_s + (TOTAL_EPOCHS - epoch) * (TOTAL_BATCHES / max(speed, 0.1))) / 60
        h, m       = divmod(int(eta_min), 60)
        eta_str    = f"{h}h {m}m" if h else f"{m}m"

        from datetime import datetime, timedelta
        finish_dt  = datetime.now() + timedelta(minutes=eta_min)
        finish_str = finish_dt.strftime("%I:%M %p")  # e.g. 03:45 PM

        # Spinner
        self.spin_lbl.config(text=self.spin_chars[self.spin_idx % 4])
        self.spin_idx += 1

        # Overall bar
        self.overall_pct_lbl.config(text=f"{ov_pct:.1f}%")
        w = self._overall_bar_frame.winfo_width()
        self.overall_bar.place(width=int(w * ov_pct / 100))
        self.overall_info.config(
            text=f"{overall:,} / {total:,} batches   •   {ep_done} epochs completed"
        )

        # Epoch bar
        self.epoch_lbl.config(text=f"Epoch  {epoch} / {TOTAL_EPOCHS}")
        self.ep_pct_lbl.config(text=f"{ep_pct:.1f}%")
        w2 = self._ep_bar_frame.winfo_width()
        self.ep_bar.place(width=int(w2 * ep_pct / 100))
        self.ep_info.config(
            text=f"{batch} / {TOTAL_BATCHES} batches this epoch"
        )

        # Stat cards
        self.card_speed.config(text=f"{speed:.1f} it/s")
        self.card_eta.config(text=eta_str)
        self.card_finishes.config(text=finish_str)

        # Losses
        for name, lbl in self.loss_labels.items():
            val = losses.get(name)
            lbl.config(text=f"{val:.4f}" if val is not None else "--")

    def on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = TrainingUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
