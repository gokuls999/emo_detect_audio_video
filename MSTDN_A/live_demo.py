"""
MSTDN-A Live Demo
-----------------
Real-time emotion detection from webcam + microphone.
Run: py -3.11 live_demo.py

Controls:
  Q / ESC  — quit
  S        — save screenshot
"""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import librosa
import torch

# ── paths ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
CHECKPOINT = BASE / "checkpoints" / "stage5_final.pt"

# ── constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SEC     = 1          # audio chunk size fed to model
INFER_EVERY   = 1.0        # seconds between inference runs
CAMERA_ID     = 0          # change if wrong camera opens

EMOTIONS = [
    "Anger", "Disgust", "Fear", "Happiness", "Neutral",
    "Sadness", "Surprise", "Contempt", "Anxiety", "Helplessness", "Disappointment"
]

EMOTION_COLORS = {          # BGR
    "Anger":        (0,   0,   220),
    "Disgust":      (0,   128, 0),
    "Fear":         (128, 0,   128),
    "Happiness":    (0,   220, 220),
    "Neutral":      (200, 200, 200),
    "Sadness":      (220, 100, 0),
    "Surprise":     (0,   200, 255),
    "Contempt":     (0,   80,  160),
    "Anxiety":      (0,   140, 255),
    "Helplessness": (80,  80,  200),
    "Disappointment":(60, 60,  180),
}

# ── load model ─────────────────────────────────────────────────────────────
print("Loading model...")
from utils.runtime import choose_torch_device, force_transformers_offline
force_transformers_offline()
from models.student.student_model import MSTDNAStudent

device = choose_torch_device("cuda")
model  = MSTDNAStudent().to(device)
ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
state  = ckpt.get("model", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
print(f"Model loaded on {device}")

# ── shared state ───────────────────────────────────────────────────────────
audio_buffer = deque(maxlen=SAMPLE_RATE * 5)   # 5s rolling buffer
result_lock  = threading.Lock()
latest       = {
    "emotion":    "Neutral",
    "confidence": 0.0,
    "stress":     0.0,
    "valence":    0.0,
    "arousal":    0.0,
    "probs":      [1/11] * 11,
}

# ── audio capture thread ───────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    audio_buffer.extend(mono.tolist())

# ── inference thread ───────────────────────────────────────────────────────
def inference_loop():
    while True:
        time.sleep(INFER_EVERY)
        raw = np.array(list(audio_buffer), dtype=np.float32)
        if raw.size < SAMPLE_RATE:
            continue
        audio = raw[-SAMPLE_RATE * 3:]             # last 3 s
        if len(audio) < SAMPLE_RATE * 3:
            audio = np.pad(audio, (0, SAMPLE_RATE * 3 - len(audio)))
        spec  = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=80)
        wav_t = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
        spc_t = torch.tensor(spec,  dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(wav_t, spc_t)
        probs   = torch.softmax(out["primary_logits"], dim=-1).cpu().numpy()[0]
        idx     = int(probs.argmax())
        emotion = EMOTIONS[idx]
        conf    = float(probs[idx])
        stress  = float(torch.sigmoid(out["stress_score"]).item())
        valence = float(torch.tanh(out["valence"]).item())
        arousal = float(torch.tanh(out["arousal"]).item())
        with result_lock:
            latest.update({
                "emotion":    emotion,
                "confidence": conf,
                "stress":     stress,
                "valence":    valence,
                "arousal":    arousal,
                "probs":      probs.tolist(),
            })

# ── drawing helpers ─────────────────────────────────────────────────────────
def bar(img, x, y, w, h, value, color, label=""):
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,60,60), -1)
    fill = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (120,120,120), 1)
    if label:
        cv2.putText(img, f"{label}: {value:.2f}", (x+4, y+h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230,230,230), 1, cv2.LINE_AA)

def overlay(frame, data):
    h, w = frame.shape[:2]
    panel_w = 260
    panel   = np.zeros((h, panel_w, 3), dtype=np.uint8)

    color = EMOTION_COLORS.get(data["emotion"], (200,200,200))

    # ── main emotion label ──
    cv2.putText(panel, data["emotion"], (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"conf {data['confidence']*100:.0f}%", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)

    # ── metric bars ──
    bar(panel, 10, 85,  240, 18, (data["stress"]+1)/2,   (0,80,220),  "Stress")
    bar(panel, 10, 112, 240, 18, (data["valence"]+1)/2,  (0,200,120), "Valence")
    bar(panel, 10, 139, 240, 18, (data["arousal"]+1)/2,  (0,180,255), "Arousal")

    # ── emotion probability bars ──
    cv2.putText(panel, "Emotion distribution", (10, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)
    for i, (emo, prob) in enumerate(zip(EMOTIONS, data["probs"])):
        y0 = 182 + i * 22
        ecol = EMOTION_COLORS.get(emo, (160,160,160))
        bar(panel, 10, y0, 200, 16, prob, ecol)
        cv2.putText(panel, emo[:10], (215, y0+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)

    # ── stress alert ──
    if data["stress"] > 0.7:
        cv2.putText(panel, "! HIGH STRESS", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)

    return np.hstack([frame, panel])


# ── main ───────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Try changing CAMERA_ID at top of file.")
        return

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()

    # Start inference thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print("\nLive demo running. Press Q or ESC to quit, S to save screenshot.\n")
    screenshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with result_lock:
            data = dict(latest)

        vis = overlay(frame, data)

        # title bar
        cv2.putText(vis, "MSTDN-A  Emotion Detection  [Q=quit  S=screenshot]",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1, cv2.LINE_AA)

        cv2.imshow("MSTDN-A Live Demo", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('s'), ord('S')):
            path = BASE / f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(str(path), vis)
            screenshot_n += 1
            print(f"Screenshot saved: {path}")

    stream.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")

if __name__ == "__main__":
    main()
