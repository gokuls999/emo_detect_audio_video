import re, time, sys
from pathlib import Path

LOG_FILE = Path(r'C:\Users\ADMIN\AppData\Local\Temp\claude\c--Users-ADMIN-Desktop-Bineetha---emoDet\tasks\blmik2m75.output')
TOTAL_EPOCHS  = 50
TOTAL_BATCHES = 459
W = 38
RE_EPOCH  = re.compile(r'stage2 epoch\s+(\d+).*?(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s+([\d.]+)it/s\]')
RE_LOSS   = re.compile(r'Epoch\s+(\d+)/\d+\s+(.*)')
RE_RESUME = re.compile(r'Resumed at epoch\s+(\d+)')
SPIN = ['|', '/', '-', 'x']

def bar(c, t, w=W):
    f = int(w * c / max(t, 1))
    return '#' * f + '-' * (w - f)

def parse():
    lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()[-300:]
    epoch = 0; batch = 0; speed = 0.0; ep_done = 0; losses = {}; remaining_s = 0
    for line in lines:
        m = RE_RESUME.search(line)
        if m: ep_done = int(m.group(1))
        m = RE_EPOCH.search(line)
        if m:
            epoch = int(m.group(1)); batch = int(m.group(2))
            remaining_s = int(m.group(6)) * 60 + int(m.group(7))
            speed = float(m.group(8))
        m = RE_LOSS.search(line)
        if m:
            ep_done = int(m.group(1))
            losses = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', m.group(2))}
    return epoch or (ep_done + 1), batch, speed, ep_done, losses, remaining_s

print()
print(' +---------------------------------------------------------+')
print(' |   MSTDN-A  Stage 2 Teacher Training  --  Live Status   |')
print(' +---------------------------------------------------------+')
print()

for i in range(12):
    epoch, batch, speed, ep_done, losses, rem_s = parse()
    overall = ep_done * TOTAL_BATCHES + batch
    total   = TOTAL_EPOCHS * TOTAL_BATCHES
    pct     = overall / total * 100
    ep_pct  = batch / TOTAL_BATCHES * 100
    eta_min = (rem_s + (TOTAL_EPOCHS - epoch) * (TOTAL_BATCHES / max(speed, 0.1))) / 60
    h, m    = divmod(int(eta_min), 60)
    eta     = f'{h}h {m}m' if h else f'{m}m'
    sp      = SPIN[i % 4]

    print(f' {sp}  Overall  [{bar(overall, total)}]  {pct:5.1f}%   ETA {eta}')
    print(f'    Epoch {epoch:>2}/{TOTAL_EPOCHS}   [{bar(batch, TOTAL_BATCHES)}]  {ep_pct:5.1f}%')
    print(f'    Batch {batch}/{TOTAL_BATCHES}   Speed: {speed:.1f} it/s   Done: {ep_done}/{TOTAL_EPOCHS} epochs')
    if losses:
        loss_str = '   '.join(f'{k}={v:.4f}' for k, v in losses.items() if k in ('loss', 'ce', 'stress', 'bce'))
        print(f'    Losses >> {loss_str}')
    print()
    sys.stdout.flush()
    time.sleep(1.5)

print(' Training continues in background.')
