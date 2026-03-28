# -*- coding: utf-8 -*-
"""
MSTDN-A Admin Dashboard Server
Run: py -3.11 dashboard/server.py
Then open: http://localhost:8000
"""
from __future__ import annotations
import asyncio, json, os, sys, subprocess, threading, time
from queue import Queue, Empty

# Force UTF-8 stdout/stderr so Unicode printed by DeepFace/face_id doesn't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dashboard.database import (
    init_db, create_session, end_session, list_sessions, get_session,
    add_participant, remove_participant, get_participants,
    log_reading, get_readings, session_summary, log_alert, get_alerts,
    export_session_excel, store_face_embedding, get_face_embeddings, has_face,
)
from dashboard.capture import VideoCapture, AudioCapture
from dashboard import face_id
from utils.runtime import choose_torch_device, force_transformers_offline

# ── constants ───────────────────────────────────────────────────────────────
CHECKPOINT  = BASE / "checkpoints" / "english_finetune_r2.pt"
SAMPLE_RATE = 16000
MAX_SAMPLES = SAMPLE_RATE * 4    # 4 seconds — must match training
INFER_EVERY = 1.0   # seconds

EMOTIONS = [
    "Anger","Disgust","Fear","Happiness","Neutral",
    "Sadness","Surprise","Contempt","Anxiety","Helplessness","Disappointment"
]
EMOJIS = ["😠","🤢","😨","😊","😐","😢","😲","😒","😰","😞","😔"]
ONLINE_CKPT = BASE / "checkpoints" / "online_tuned.pt"

# ── model ────────────────────────────────────────────────────────────────────
print("Loading MSTDN-A model…")
force_transformers_offline()
from models.student.student_model import MSTDNAStudent
device = choose_torch_device("cuda")
_model = MSTDNAStudent().to(device)
_ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
_state = _ckpt.get("model", _ckpt)
# Allow checkpoint to load even if wav2vec branch is disabled (shape mismatches)
current = _model.state_dict()
filtered = {k: v for k, v in _state.items() if k in current and getattr(v, "shape", None) == current[k].shape}
_model.load_state_dict(filtered, strict=False)
_model.eval()
print(f"Model ready on {device}")


def preprocess_audio(raw: np.ndarray) -> tuple:
    """Match EXACT training preprocessing from finetune_english_r2.py.
    Returns (wav_tensor, spec_tensor) ready for model input."""
    # 1. Pad or truncate to 4 seconds (64000 samples)
    if len(raw) < MAX_SAMPLES:
        raw = np.pad(raw, (0, MAX_SAMPLES - len(raw)))
    else:
        raw = raw[:MAX_SAMPLES]

    # 2. RMS normalization (NOT peak) — matches training exactly
    rms = np.sqrt(np.mean(raw ** 2)) + 1e-8
    raw = np.clip(raw / max(rms, 0.01), -1.0, 1.0)

    # 3. Mel spectrogram with log1p — matches training exactly
    spec = np.log1p(
        librosa.feature.melspectrogram(y=raw, sr=SAMPLE_RATE, n_mels=80)
    ).astype(np.float32)

    # 4. Convert to tensors
    wav_t = torch.tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
    spc_t = torch.tensor(spec, dtype=torch.float32, device=device).unsqueeze(0)
    return wav_t, spc_t


# ── hardware ─────────────────────────────────────────────────────────────────
video  = VideoCapture(camera_id=0)
audio1 = AudioCapture()   # mic 1
audio2 = AudioCapture()   # mic 2
audio  = audio1   # legacy reference for /api/audio/level

# ── app state ─────────────────────────────────────────────────────────────────
SILENCE_THRESH  = 0.015   # Above typical ambient noise (~0.011); only actual speech triggers audio model.
AUDIO_TEMP      = 1.0     # No temperature scaling — proper preprocessing fixes bias
NEUTRAL_PENALTY = 0.0     # No logit manipulation — clean model output
STRESS_ALPHA    = 0.25    # EMA smoothing for stress (lower = smoother, less spiking)
STRESS_ALERT_TH = 0.80    # Alert threshold (raised from 0.72 to reduce false positives)
_FACE_EMO_MAP  = {       # DeepFace emotion name → MAFW class index
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'sad': 5, 'surprise': 6,
}

class AppState:
    session_id:        int | None = None
    participants:      list[dict] = []
    active_speaker:    int | None = None   # participant id (auto or manual)
    face_auto:         bool       = True   # auto-identify from camera
    registered_faces:  list[dict] = []    # [{participant_id, name, embedding}]
    detected_faces:    list[dict] = []    # latest face detections with names
    face_emotion_probs: "np.ndarray | None" = None          # scene-level (fallback)
    face_emotion_per_pid: "dict" = {}                       # {pid: np.ndarray}
    # ── dual-mic ──
    mic1_device:  "int | None" = None     # sounddevice index for mic 1
    mic2_device:  "int | None" = None     # sounddevice index for mic 2
    mic2_enabled: bool         = False    # whether mic 2 is active
    mic1_pid:     "int | None" = None     # participant assigned to mic 1
    mic2_pid:     "int | None" = None     # participant assigned to mic 2
    latest_result:     dict       = {
        "emotion":"Neutral","emoji":"😐","confidence":0.0,
        "stress":0.0,"valence":0.0,"arousal":0.0,"probs":[1/11]*11,
        "rms":0.0,"alert":False,
    }
    running: bool = False
    _loop:   asyncio.AbstractEventLoop | None = None
    _clients: set[WebSocket]                  = set()
    _history: dict[str, list]                 = {}   # participant_id -> last 60 stress vals
    # Online learning buffer
    audio_buffer: list                       = []   # circular buffer of (wav_t, spc_t)
    BUFFER_SIZE:  int                        = 8
    teach_count:  int                        = 0

state = AppState()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MSTDN-A Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.on_event("startup")
async def startup():
    state._loop = asyncio.get_event_loop()
    init_db()

# ── broadcast helper ──────────────────────────────────────────────────────────
async def _broadcast(msg: dict):
    dead = set()
    for ws in list(state._clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    state._clients -= dead

def broadcast_from_thread(msg: dict):
    if state._loop:
        asyncio.run_coroutine_threadsafe(_broadcast(msg), state._loop)

# ── face recognition loop (runs every 2s in background) ──────────────────────
def _face_loop():
    while True:
        time.sleep(2.0)
        if not state.running:
            state.detected_faces = []
            continue
        try:
            frame = video.get_frame()
            if frame is None:
                continue

            # ── face ID (only when registered faces exist) ────────────────
            detections = []
            if state.registered_faces:
                detections = face_id.identify_all(frame, state.registered_faces)
                state.detected_faces = detections
                annotated = face_id.annotate_frame(frame, detections)
                video._lock.acquire()
                video._frame = annotated
                video._lock.release()
                if state.face_auto and detections:
                    identified = [d for d in detections if d["participant_id"] is not None]
                    if identified:
                        pid = identified[0]["participant_id"]
                        if pid != state.active_speaker:
                            state.active_speaker = pid
                            broadcast_from_thread({"type":"speaker_changed",
                                                   "participant_id": pid,
                                                   "auto": True,
                                                   "name": identified[0]["name"]})
                broadcast_from_thread({
                    "type": "faces_detected",
                    "faces": [{"participant_id": d["participant_id"],
                                "name": d["name"],
                                "distance": round(d["distance"], 3)} for d in detections]
                })

            # ── face emotion analysis — always runs regardless of registration ──
            try:
                from deepface import DeepFace
                fa = DeepFace.analyze(frame, actions=["emotion"],
                                      enforce_detection=False, silent=True,
                                      detector_backend="opencv")

                def _fe_to_probs(fa_result: dict) -> "np.ndarray":
                    fe  = fa_result["emotion"]
                    dom = fa_result.get("dominant_emotion", "neutral")
                    tot = sum(fe.values()) or 1.0

                    # Step 1 — map raw DeepFace scores to MAFW 11-class
                    fp = np.zeros(11, dtype=np.float32)
                    for k, v in fe.items():
                        if k in _FACE_EMO_MAP:
                            fp[_FACE_EMO_MAP[k]] = float(v) / tot

                    # Step 2 — temperature T=2.5 to flatten Neutral-heavy distribution
                    known = [_FACE_EMO_MAP[k] for k in fe if k in _FACE_EMO_MAP]
                    sub   = fp[known]
                    log_p = np.log(np.clip(sub, 1e-10, 1.0)) / 2.5
                    log_p -= log_p.max()
                    sub   = np.exp(log_p); sub /= sub.sum()
                    for j, idx in enumerate(known):
                        fp[idx] = sub[j]

                    # Step 3 — anchor: ensure DeepFace's own dominant_emotion
                    # actually leads the distribution (fixes anger/fear being
                    # numerically outscored by neutral in raw scores)
                    if dom in _FACE_EMO_MAP:
                        d_idx = _FACE_EMO_MAP[dom]
                        cur_max = fp.max()
                        if fp[d_idx] < cur_max:
                            # give dominant emotion a clear lead
                            fp[d_idx] = cur_max + 0.08
                            fp /= fp.sum()

                    return fp

                # Only update if DeepFace found a face (confidence > 0.5)
                face_conf = fa[0].get("face_confidence", 1.0)
                if face_conf >= 0.5:
                    state.face_emotion_probs = _fe_to_probs(fa[0])

                new_per_pid: dict = {}
                for det in detections:
                    pid2 = det.get("participant_id")
                    if pid2 is None:
                        continue
                    bbox = det.get("bbox", (0, 0, 0, 0))
                    x, y, w2, h2 = bbox
                    if w2 > 20 and h2 > 20:
                        crop = frame[max(0,y):y+h2, max(0,x):x+w2]
                        try:
                            fa2 = DeepFace.analyze(crop, actions=["emotion"],
                                                   enforce_detection=False,
                                                   silent=True,
                                                   detector_backend="opencv")
                            conf2 = fa2[0].get("face_confidence", 1.0)
                            if conf2 >= 0.5:
                                new_per_pid[pid2] = _fe_to_probs(fa2[0])
                            else:
                                new_per_pid[pid2] = state.face_emotion_probs
                        except Exception:
                            new_per_pid[pid2] = state.face_emotion_probs
                    else:
                        new_per_pid[pid2] = state.face_emotion_probs
                state.face_emotion_per_pid = new_per_pid
            except Exception as _dfe:
                print(f"[deepface] {str(_dfe)[:120]}")
                # keep last known face_emotion_probs — don't reset to None
        except Exception as e:
            print(f"[face_loop] {e}")

threading.Thread(target=_face_loop, daemon=True).start()

# ── inference loop (background thread) ───────────────────────────────────────
def _inference_loop():
    hidden        = None
    stress_streak = 0
    stress_ema    = 0.2   # start at calm baseline
    while True:
        time.sleep(INFER_EVERY)
        if not state.running:
            hidden = None
            stress_streak = 0
            continue
        try:
            # ── pick active mic: use louder of the two ───────────────────
            rms1 = audio1.rms()
            rms2 = audio2.rms() if state.mic2_enabled else 0.0
            if rms2 > rms1:
                active_audio = audio2
                rms = rms2
                # override active speaker to mic2 participant if assigned
                if state.mic2_pid and state.mic2_pid != state.active_speaker:
                    state.active_speaker = state.mic2_pid
                    broadcast_from_thread({"type": "speaker_changed",
                                           "participant_id": state.mic2_pid,
                                           "auto": True, "name": next(
                        (p["name"] for p in state.participants if p["id"]==state.mic2_pid), "Mic 2")})
            else:
                active_audio = audio1
                rms = rms1
                if state.mic1_pid and rms1 > SILENCE_THRESH and state.mic1_pid != state.active_speaker:
                    state.active_speaker = state.mic1_pid
                    broadcast_from_thread({"type": "speaker_changed",
                                           "participant_id": state.mic1_pid,
                                           "auto": True, "name": next(
                        (p["name"] for p in state.participants if p["id"]==state.mic1_pid), "Mic 1")})

            raw = active_audio.get_audio(seconds=4.0)  # 4s to match training

            # RMS for silence detection only
            peak = float(np.abs(raw).max())

            stress  = state.latest_result.get("stress",  0.2)
            valence = state.latest_result.get("valence", 0.0)
            arousal = state.latest_result.get("arousal", 0.3)

            pid_now = state.active_speaker
            _pid_fp = state.face_emotion_per_pid.get(pid_now) if pid_now else None
            fp = _pid_fp if _pid_fp is not None else state.face_emotion_probs

            if rms < SILENCE_THRESH:
                # ── silence: 100% face emotion ────────────────────────────────
                if fp is not None:
                    probs = fp / (fp.sum() + 1e-8)
                else:
                    # No face & no audio — keep last result, don't force Neutral
                    probs = None
            else:
                # ── speech: 100% audio ───────────────────────────────────────
                wav_t, spc_t = preprocess_audio(raw)
                # Buffer for online learning
                if len(state.audio_buffer) >= state.BUFFER_SIZE:
                    state.audio_buffer.pop(0)
                state.audio_buffer.append((wav_t.detach().clone(), spc_t.detach().clone()))
                with torch.no_grad():
                    out = _model(wav_t, spc_t, hidden=hidden)
                hidden = out.get("hidden")

                audio_p = torch.softmax(out["primary_logits"] / AUDIO_TEMP, dim=-1).cpu().numpy()[0]

                raw_stress = float(torch.sigmoid(out["stress_score"]).item())
                valence    = float(torch.tanh(out["valence"]).item())
                arousal    = float(torch.tanh(out["arousal"]).item())

                # EMA smooth stress — prevents spiking from single loud frame
                stress_ema = STRESS_ALPHA * raw_stress + (1 - STRESS_ALPHA) * stress_ema
                stress     = stress_ema

                probs  = audio_p   # 100% audio — no face mixing

            if probs is None:
                time.sleep(INFER_EVERY)
                continue

            idx     = int(probs.argmax())
            emotion = EMOTIONS[idx]
            emoji   = EMOJIS[idx]
            conf    = float(probs[idx])

            # stress alert — raised threshold + streak still required
            stress_streak = stress_streak + 1 if stress > STRESS_ALERT_TH else max(0, stress_streak - 1)
            alert = stress_streak >= 3

            pid = state.active_speaker
            pname = next((p["name"] for p in state.participants if p["id"]==pid), "Group")

            result = dict(
                emotion=emotion, emoji=emoji, confidence=round(conf,3),
                stress=round(stress,3), valence=round(valence,3),
                arousal=round(arousal,3), probs=probs.tolist(),
                rms=round(rms,4), alert=alert,
                active_speaker=pid,
                active_speaker_name=pname,
            )
            state.latest_result = result

            # DB logging
            if state.session_id:
                log_reading(state.session_id, pid,
                            emotion, conf, stress, valence, arousal, probs.tolist())
                if alert:
                    msg = f"High stress detected for {pname} ({stress:.0%})"
                    log_alert(state.session_id, pid, "HIGH_STRESS", msg)

            # participant history
            key = str(pid or "group")
            if key not in state._history:
                state._history[key] = []
            state._history[key].append(round(stress, 3))
            if len(state._history[key]) > 60:
                state._history[key].pop(0)

            broadcast_from_thread({"type": "emotion_update", **result,
                                   "history": state._history,
                                   "rms1": round(rms1, 4),
                                   "rms2": round(rms2, 4)})
        except Exception as e:
            print(f"[inference] {e}")

threading.Thread(target=_inference_loop, daemon=True).start()

# ── video stream ─────────────────────────────────────────────────────────────
@app.get("/video")
async def video_stream():
    async def gen():
        while True:
            jpg = video.get_jpeg(quality=80)
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state._clients.add(ws)
    # send current state immediately
    await ws.send_json({"type": "init",
                        "session_id": state.session_id,
                        "participants": state.participants,
                        "running": state.running,
                        "latest": state.latest_result})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state._clients.discard(ws)

# ── REST: root ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(str(STATIC / "index.html"))

# ── REST: audio devices ───────────────────────────────────────────────────────
import sounddevice as _sd

@app.get("/api/audio/devices")
def list_audio_devices():
    """Return deduplicated input devices — one entry per physical microphone."""
    devs     = _sd.query_devices()
    apis     = _sd.query_hostapis()
    # prefer WASAPI (highest quality), fall back to MME
    wasapi_idx = next((i for i,a in enumerate(apis) if "WASAPI" in a["name"]), None)
    seen, result = set(), []
    default_dev  = _sd.default.device[0]
    # first pass: WASAPI only
    for i, d in enumerate(devs):
        if d["max_input_channels"] < 1:
            continue
        if wasapi_idx is not None and d["hostapi"] != wasapi_idx:
            continue
        base = d["name"].strip()
        if base not in seen:
            seen.add(base)
            result.append({"id": i, "name": base, "default": i == default_dev})
    # second pass: add any remaining input devices not yet listed (non-WASAPI fallback)
    for i, d in enumerate(devs):
        if d["max_input_channels"] < 1:
            continue
        base = d["name"].strip()
        if base not in seen:
            seen.add(base)
            result.append({"id": i, "name": base, "default": i == default_dev})
    return result

class AudioSetupIn(BaseModel):
    mic1_device: int | None = None
    mic2_device: int | None = None
    mic1_pid:    int | None = None
    mic2_pid:    int | None = None

@app.post("/api/audio/setup")
def audio_setup(body: AudioSetupIn):
    state.mic1_device = body.mic1_device
    state.mic2_device = body.mic2_device
    state.mic1_pid    = body.mic1_pid
    state.mic2_pid    = body.mic2_pid
    return {"ok": True}

# ── REST: sessions ────────────────────────────────────────────────────────────
def _auto_detect_mics() -> list[int]:
    """Return up to two MME input device indices (most compatible on Windows).
    Skips generic Sound Mapper. Falls back to any input if MME not available."""
    devs  = _sd.query_devices()
    apis  = _sd.query_hostapis()
    mme_idx = next((i for i, a in enumerate(apis) if a["name"] == "MME"), None)
    seen, result = set(), []
    # prefer MME — works with more sample rates than WASAPI on most Windows setups
    for i, d in enumerate(devs):
        if d["max_input_channels"] < 1:
            continue
        if mme_idx is not None and d["hostapi"] != mme_idx:
            continue
        base = d["name"].strip()
        if "sound mapper" in base.lower():  # skip generic fallback device
            continue
        if base not in seen:
            seen.add(base)
            result.append(i)
        if len(result) >= 2:
            return result
    # fallback: any input not already listed
    for i, d in enumerate(devs):
        if d["max_input_channels"] >= 1 and i not in result:
            if "sound mapper" in d["name"].lower():
                continue
            result.append(i)
        if len(result) >= 2:
            return result
    return result


class SessionIn(BaseModel):
    name: str
    location: str = ""

@app.post("/api/session/start")
def session_start(body: SessionIn):
    if state.running:
        raise HTTPException(400, "Session already running")
    sid = create_session(body.name, body.location)
    state.session_id      = sid
    state.participants    = []
    state.active_speaker  = None
    state.running         = True
    state._history        = {}
    state.detected_faces  = []
    state.registered_faces = get_face_embeddings()
    # ── auto-detect both microphones ──
    mics = _auto_detect_mics()
    mic1_dev = mics[0] if mics else None
    mic2_dev = mics[1] if len(mics) > 1 else None
    print(f"[mics] auto-detected devices: {mics}")
    audio1._device = mic1_dev
    try:
        audio1.start()
        print(f"[mics] mic1 started (device={mic1_dev})")
    except Exception as e:
        print(f"[mics] mic1 FAILED: {e}")
    if mic2_dev is not None:
        audio2._device = mic2_dev
        try:
            audio2.start()
            state.mic2_enabled = True
            print(f"[mics] mic2 started (device={mic2_dev})")
        except Exception as e:
            print(f"[mics] mic2 FAILED: {e}")
            state.mic2_enabled = False
    else:
        state.mic2_enabled = False
        print("[mics] only one input device found")
    video.start()
    broadcast_from_thread({"type":"session_started","session_id":sid,
                           "name":body.name,"location":body.location})
    return {"session_id": sid}

@app.post("/api/session/stop")
def session_stop():
    if not state.running:
        raise HTTPException(400, "No active session")
    state.running = False
    video.stop()
    audio1.stop()
    if state.mic2_enabled:
        audio2.stop()
        state.mic2_enabled = False
    end_session(state.session_id)
    summary = session_summary(state.session_id)
    broadcast_from_thread({"type":"session_stopped","summary":summary})
    sid = state.session_id
    state.session_id = None
    return {"summary": summary}

@app.get("/api/sessions")
def sessions_list():
    return list_sessions()

@app.get("/api/session/{sid}/summary")
def session_sum(sid: int):
    return session_summary(sid)

@app.get("/api/session/{sid}/readings")
def session_readings(sid: int, pid: int | None = None):
    return get_readings(sid, pid)

@app.get("/api/session/{sid}/alerts")
def session_alerts_api(sid: int):
    return get_alerts(sid)

@app.get("/api/session/{sid}/export")
def session_export(sid: int):
    from dashboard.pdf_report import export_pdf
    path = str(BASE / f"session_{sid}_report.pdf")
    export_pdf(sid, path)
    return FileResponse(path, media_type="application/pdf",
                        filename=f"session_{sid}_report.pdf")

# ── REST: participants ────────────────────────────────────────────────────────
class ParticipantIn(BaseModel):
    name: str
    role: str = ""
    department: str = ""

@app.post("/api/participants")
def participant_add(body: ParticipantIn):
    if not state.session_id:
        raise HTTPException(400, "No active session")
    pid = add_participant(state.session_id, body.name, body.role, body.department)
    p   = {"id": pid, "name": body.name, "role": body.role,
           "department": body.department, "session_id": state.session_id}
    state.participants.append(p)
    broadcast_from_thread({"type":"participant_added","participant":p})
    return p

@app.delete("/api/participants/{pid}")
def participant_remove(pid: int):
    remove_participant(pid)
    state.participants = [p for p in state.participants if p["id"] != pid]
    if state.active_speaker == pid:
        state.active_speaker = None
    broadcast_from_thread({"type":"participant_removed","participant_id":pid})
    return {"ok": True}

@app.post("/api/participants/{pid}/focus")
def participant_focus(pid: int):
    state.active_speaker = pid
    broadcast_from_thread({"type":"speaker_changed","participant_id":pid})
    return {"active_speaker": pid}

@app.post("/api/participants/focus/clear")
def focus_clear():
    state.active_speaker = None
    broadcast_from_thread({"type":"speaker_changed","participant_id":None})
    return {"active_speaker": None}

# ── REST: face registration ───────────────────────────────────────────────────
@app.post("/api/participants/{pid}/register_face")
def register_face(pid: int):
    """Capture current camera frame and register this participant's face."""
    frame = video.get_frame()
    if frame is None:
        raise HTTPException(400, "Camera not active. Start a session first.")
    emb = face_id.extract_embedding(frame)
    if emb is None:
        raise HTTPException(400, "No face detected in camera. Make sure face is clearly visible.")
    store_face_embedding(pid, emb)
    # update in-memory registered faces
    pname = next((p["name"] for p in state.participants if p["id"]==pid), "Unknown")
    state.registered_faces = [r for r in state.registered_faces if r["participant_id"] != pid]
    state.registered_faces.append({"participant_id": pid, "name": pname, "embedding": emb})
    broadcast_from_thread({"type":"face_registered","participant_id":pid,"name":pname})
    return {"ok": True, "participant_id": pid, "name": pname}

@app.get("/api/participants/{pid}/face_status")
def face_status(pid: int):
    return {"registered": has_face(pid)}

@app.delete("/api/participants/{pid}/face")
def delete_face(pid: int):
    from dashboard.database import _conn
    with _conn() as c:
        c.execute("DELETE FROM face_embeddings WHERE participant_id=?", (pid,))
    state.registered_faces = [r for r in state.registered_faces if r["participant_id"] != pid]
    return {"ok": True}

@app.post("/api/face/auto/{enabled}")
def face_auto(enabled: str):
    state.face_auto = enabled.lower() == "true"
    return {"face_auto": state.face_auto}

@app.get("/api/faces/detected")
def faces_detected():
    return state.detected_faces

# ── REST: current state ───────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "running":        state.running,
        "session_id":     state.session_id,
        "participants":   state.participants,
        "active_speaker": state.active_speaker,
        "latest":         state.latest_result,
        "device":         str(device),
    }

@app.get("/api/audio/level")
def audio_level():
    return {"rms": round(audio.rms(), 4)}

@app.get("/api/audio/debug")
def audio_debug():
    """Show all detected input devices and current RMS levels."""
    devs = _sd.query_devices()
    inputs = [{"id": i, "name": d["name"], "channels": d["max_input_channels"],
               "hostapi": _sd.query_hostapis()[d["hostapi"]]["name"]}
              for i, d in enumerate(devs) if d["max_input_channels"] >= 1]
    return {
        "all_inputs": inputs,
        "auto_detect": _auto_detect_mics(),
        "mic1_running": audio1._stream is not None,
        "mic2_running": audio2._stream is not None,
        "rms1": round(audio1.rms(), 4),
        "rms2": round(audio2.rms(), 4),
        "mic2_enabled": state.mic2_enabled,
    }

@app.get("/api/audio/levels")
def audio_levels():
    """Return live RMS + running status for both mics (polled by frontend)."""
    return {
        "rms1":    round(audio1.rms(), 4),
        "rms2":    round(audio2.rms(), 4) if state.mic2_enabled else 0.0,
        "mic1_on": audio1._stream is not None,
        "mic2_on": state.mic2_enabled and audio2._stream is not None,
    }

# ── Dashboard teach endpoints ─────────────────────────────────────────────────
class DashTeachIn(BaseModel):
    emotion_idx: int

@app.post("/api/teach")
def dash_teach(body: DashTeachIn):
    if body.emotion_idx < 0 or body.emotion_idx > 10:
        raise HTTPException(400, "emotion_idx must be 0-10")
    result = do_teach_step(body.emotion_idx, source="dash")
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result

@app.post("/api/teach/save")
def dash_teach_save():
    torch.save(_model.state_dict(), ONLINE_CKPT)
    return {"saved": str(ONLINE_CKPT), "teach_count": state.teach_count}

@app.get("/api/teach/status")
def dash_teach_status():
    return {
        "teach_count": state.teach_count,
        "buffer_size": len(state.audio_buffer),
        "configured": _online_configured,
        "checkpoint": str(ONLINE_CKPT),
    }

# ── YouTube test ──────────────────────────────────────────────────────────────
try:
    import yt_dlp as _ytdlp
    _ytdlp_ok = True
except ImportError:
    _ytdlp_ok = False

_FFMPEG = r"C:\Users\ADMIN\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

class YTState:
    running:     bool              = False
    muted:       bool              = False
    _audio_proc                    = None   # ffmpeg PCM pipe
    _video_proc                    = None   # ffmpeg frame pipe
    _yt_audio                      = None   # yt-dlp audio downloader process
    _yt_video                      = None   # yt-dlp video downloader process
    _clients:    set               = set()
    face_probs: "np.ndarray|None"  = None
    stress_ema:  float             = 0.2
    latest:      dict              = {
        "type":"emotion_update","emotion":"Neutral","emoji":"😐",
        "confidence":0.0,"stress":0.0,"valence":0.0,"arousal":0.0,
        "probs":[1/11]*11,"mode":"idle"
    }
    # Online learning state
    audio_buffer: list             = []     # circular buffer of (wav_t, spc_t)
    BUFFER_SIZE:  int              = 8      # keep last 8 chunks (~16s audio)
    teach_count:  int              = 0      # total teach steps this session
    teach_lock:   "threading.Lock" = None   # prevent concurrent training
    # Auto-learn
    auto_learn:        bool        = False  # auto teach from own predictions
    last_auto_teach:   float       = 0.0   # throttle timer

yt_state  = YTState()
yt_state.teach_lock = threading.Lock()
_yt_q: "Queue[bytes]" = Queue(maxsize=8)

async def _yt_bcast(msg: dict):
    dead = set()
    for ws in list(yt_state._clients):
        try:    await ws.send_json(msg)
        except: dead.add(ws)
    yt_state._clients -= dead

def _yt_bcast_sync(msg: dict):
    if state._loop:
        asyncio.run_coroutine_threadsafe(_yt_bcast(msg), state._loop)

def _yt_cleanup():
    yt_state.running = False
    for attr in ("_audio_proc", "_video_proc", "_yt_audio", "_yt_video"):
        proc = getattr(yt_state, attr, None)
        if proc:
            try: proc.kill()
            except: pass
            setattr(yt_state, attr, None)
    while not _yt_q.empty():
        try: _yt_q.get_nowait()
        except: break
    yt_state.audio_buffer = []  # clear buffer on new video, keep teach_count

# -- online learning engine ---------------------------------------------------
import torch.nn.functional as F

_online_optimizer = None
_online_scaler    = None
_online_configured = False
ONLINE_LR          = 5e-6
ONLINE_GRAD_CLIP   = 1.0
ONLINE_LABEL_SMOOTH = 0.05
ONLINE_SAVE_EVERY  = 10

# Class weights: boost under-performing classes (anger=3x, fear=2.5x, disgust=2x)
_ONLINE_CLASS_WEIGHTS = None

def _init_online_training():
    """Lazy init: freeze layers, create optimizer+scaler. Called once on first teach."""
    global _online_optimizer, _online_scaler, _online_configured, _ONLINE_CLASS_WEIGHTS
    if _online_configured:
        return
    # Freeze strategy — same as finetune_english_r2.configure_trainable
    for p in _model.parameters():
        p.requires_grad = False
    dab = _model.deep_audio_branch
    if dab.enabled:
        for layer in dab.model.encoder.layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in dab.proj.parameters():
            p.requires_grad = True
    for p in _model.fusion.parameters():
        p.requires_grad = True
    for p in _model.heads.parameters():
        p.requires_grad = True

    trainable = [p for p in _model.parameters() if p.requires_grad]
    _online_optimizer = torch.optim.AdamW(trainable, lr=ONLINE_LR, weight_decay=1e-4)
    _online_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    weights = [3.0, 2.0, 2.5, 0.8, 0.5, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0]
    _ONLINE_CLASS_WEIGHTS = torch.tensor(weights, dtype=torch.float32, device=device)

    _online_configured = True
    n = sum(p.numel() for p in trainable)
    print(f"[online] Training initialized. Trainable: {n:,} params")

def do_teach_step(emotion_idx: int, source: str = "yt") -> dict:
    """Single training step using last buffered audio + user label."""
    buf = yt_state.audio_buffer if source == "yt" else state.audio_buffer
    with yt_state.teach_lock:
        if not buf:
            return {"success": False, "error": "No audio buffered yet"}
        _init_online_training()

        wav_t, spc_t = buf[-1]
        label = torch.tensor([emotion_idx], dtype=torch.long, device=device)

        _model.train()
        try:
            _online_optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = _model(wav_t, spc_t)
                loss = F.cross_entropy(
                    out["primary_logits"], label,
                    weight=_ONLINE_CLASS_WEIGHTS,
                    label_smoothing=ONLINE_LABEL_SMOOTH,
                )
            if not torch.isfinite(loss):
                return {"success": False, "error": "NaN loss"}

            _online_scaler.scale(loss).backward()
            _online_scaler.unscale_(_online_optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in _model.parameters() if p.requires_grad],
                ONLINE_GRAD_CLIP,
            )
            _online_scaler.step(_online_optimizer)
            _online_scaler.update()

            yt_state.teach_count += 1
            state.teach_count = yt_state.teach_count  # keep in sync
            loss_val = loss.item()

            if yt_state.teach_count % ONLINE_SAVE_EVERY == 0:
                torch.save(_model.state_dict(), ONLINE_CKPT)
                print(f"[online] Checkpoint saved ({yt_state.teach_count} steps)")

            with torch.no_grad():
                out2 = _model(wav_t, spc_t)
                new_pred = int(out2["primary_logits"].argmax(dim=-1).item())

            print(f"[online] teach #{yt_state.teach_count}: "
                  f"label={EMOTIONS[emotion_idx]} loss={loss_val:.4f} "
                  f"now_pred={EMOTIONS[new_pred]}")

            return {
                "success": True,
                "loss": round(loss_val, 4),
                "teach_count": yt_state.teach_count,
                "new_prediction": EMOTIONS[new_pred],
                "taught_emotion": EMOTIONS[emotion_idx],
            }
        finally:
            _model.eval()

# Auto-learn constants
AUTO_LEARN_CONF_THRESHOLD = 0.62   # only teach if model is ≥62% confident
AUTO_LEARN_COOLDOWN       = 12.0   # seconds between auto-teach steps

def _auto_teach(idx: int):
    """Called from _yt_infer thread — throttled pseudo-label self-training."""
    now = time.time()
    if now - yt_state.last_auto_teach < AUTO_LEARN_COOLDOWN:
        return
    yt_state.last_auto_teach = now
    result = do_teach_step(idx, source="yt")
    if result.get("success"):
        _yt_bcast_sync({
            "type":          "auto_teach_ack",
            "taught_emotion": result["taught_emotion"],
            "loss":           result["loss"],
            "teach_count":    result["teach_count"],
        })

# -- audio reader: ffmpeg pipe → queue ----------------------------------------
def _yt_audio_reader():
    CHUNK = 16000 * 2 * 2   # 2s @ 16kHz int16 — padded to 4s by preprocess_audio
    _chunks_read = 0
    while True:
        proc = yt_state._audio_proc
        if proc is None or not yt_state.running:
            time.sleep(0.1); continue
        try:
            data = proc.stdout.read(CHUNK)
            if not data:
                # Only kill pipeline if NOT muted — when muted, video keeps running
                if yt_state.running and not yt_state.muted:
                    print(f"[yt_audio] stream ended after {_chunks_read} chunks")
                    yt_state.running = False
                    _yt_bcast_sync({"type": "yt_ended"})
                _chunks_read = 0
                time.sleep(0.5)
                continue
            _chunks_read += 1
            if _chunks_read <= 3 or _chunks_read % 20 == 0:
                print(f"[yt_audio] chunk #{_chunks_read}  {len(data)} bytes  q={_yt_q.qsize()}")
            try: _yt_q.put_nowait(data)
            except: pass
        except Exception as e:
            print(f"[yt_audio] exception: {e}")
            time.sleep(0.1)

# -- face reader: ffmpeg video pipe → DeepFace → live broadcast ---------------
def _yt_face_reader():
    W, H = 320, 240
    FBYTES = W * H * 3
    while True:
        proc = yt_state._video_proc
        if proc is None or (proc.poll() is not None):
            time.sleep(0.3); continue
        if not yt_state.running:
            time.sleep(0.3); continue
        try:
            data = proc.stdout.read(FBYTES)
            if len(data) < FBYTES:
                time.sleep(0.1); continue
            frame = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)
            from deepface import DeepFace
            fa  = DeepFace.analyze(frame, actions=["emotion"],
                                    enforce_detection=False, silent=True,
                                    detector_backend="opencv")
            fc  = fa[0].get("face_confidence", 1.0)
            if fc >= 0.3:
                fe  = fa[0]["emotion"]
                dom = fa[0].get("dominant_emotion", "neutral")
                fp  = np.zeros(11, dtype=np.float32)
                tot = sum(fe.values()) or 1.0
                for k, v in fe.items():
                    if k in _FACE_EMO_MAP:
                        fp[_FACE_EMO_MAP[k]] = float(v) / tot
                known = [_FACE_EMO_MAP[k] for k in fe if k in _FACE_EMO_MAP]
                sub   = fp[known]
                lp    = np.log(np.clip(sub, 1e-10, 1.0)) / 2.5
                lp   -= lp.max(); sub = np.exp(lp); sub /= sub.sum()
                for j, idx2 in enumerate(known): fp[idx2] = sub[j]
                if dom in _FACE_EMO_MAP:
                    di = _FACE_EMO_MAP[dom]
                    if fp[di] < fp.max():
                        fp[di] = fp.max() + 0.08; fp /= fp.sum()
                yt_state.face_probs = fp
                # Face broadcast disabled — audio-only mode for diagnosis
        except Exception:
            time.sleep(0.3)

# -- inference thread: queue → model → broadcast ------------------------------
def _yt_infer():
    _last_face_push = 0.0
    while True:
        try:
            data = _yt_q.get(timeout=1.0)
        except Empty:
            # When muted and running, push face-only update on a 1s timer
            if yt_state.running and yt_state.muted:
                now = time.time()
                if now - _last_face_push >= 1.0 and yt_state.face_probs is not None:
                    _last_face_push = now
                    fp    = yt_state.face_probs
                    probs = fp / (fp.sum() + 1e-8)
                    idx   = int(probs.argmax())
                    result = {
                        "type":       "emotion_update",
                        "emotion":    EMOTIONS[idx],
                        "emoji":      EMOJIS[idx],
                        "confidence": round(float(probs[idx]), 3),
                        "stress":     round(float(yt_state.stress_ema), 3),
                        "valence":    0.0,
                        "arousal":    0.0,
                        "probs":      probs.tolist(),
                        "mode":       "face_only",
                    }
                    yt_state.latest = result
                    _yt_bcast_sync(result)
            continue
        if not yt_state.running:
            continue
        try:
            raw   = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms   = float(np.sqrt(np.mean(raw ** 2)))

            # ── audio model ──────────────────────────────────────────────
            wav_t, spc_t = preprocess_audio(raw)

            # Buffer for online learning
            if len(yt_state.audio_buffer) >= yt_state.BUFFER_SIZE:
                yt_state.audio_buffer.pop(0)
            yt_state.audio_buffer.append((wav_t.detach().clone(), spc_t.detach().clone()))

            with torch.no_grad():
                out = _model(wav_t, spc_t)
            raw_logits = out["primary_logits"].cpu().numpy()[0]
            ap  = torch.softmax(out["primary_logits"] / AUDIO_TEMP, dim=-1).cpu().numpy()[0]
            rs  = float(torch.sigmoid(out["stress_score"]).item())
            valence = float(torch.tanh(out["valence"]).item())
            arousal = float(torch.tanh(out["arousal"]).item())
            yt_state.stress_ema = STRESS_ALPHA * rs + (1 - STRESS_ALPHA) * yt_state.stress_ema
            stress = yt_state.stress_ema

            # Blend: 50% audio + 50% face when face available
            fp = yt_state.face_probs
            if fp is not None:
                face_p = fp / (fp.sum() + 1e-8)
                probs  = 0.5 * ap + 0.5 * face_p
                probs  = probs / (probs.sum() + 1e-8)
                mode   = "audio+face"
            else:
                probs  = ap
                mode   = "audio"

            idx    = int(probs.argmax())
            names = ['Ang','Dis','Fear','Hap','Neu','Sad','Sur','Con','Anx','Hlp','Dsp']
            lg_str = ' '.join(f'{names[i]}:{raw_logits[i]:+.2f}' for i in range(7))
            pr_str = ' '.join(f'{names[i]}:{probs[i]*100:.0f}%' for i in range(7))
            print(f"[yt] logits=[{lg_str}] probs=[{pr_str}] rms={rms:.4f} → {EMOTIONS[idx]}")
            result = {
                "type":       "emotion_update",
                "emotion":    EMOTIONS[idx],
                "emoji":      EMOJIS[idx],
                "confidence": round(float(probs[idx]), 3),
                "stress":     round(float(stress), 3),
                "valence":    round(float(valence), 3),
                "arousal":    round(float(arousal), 3),
                "probs":      probs.tolist(),
                "mode":       mode,
                "audio_rms":  round(rms, 4),
            }
            yt_state.latest = result
            _yt_bcast_sync(result)

            # Auto-learn: always runs when video is playing and confidence is high enough
            if float(probs[idx]) >= AUTO_LEARN_CONF_THRESHOLD:
                threading.Thread(target=_auto_teach, args=(idx,), daemon=True).start()
        except Exception as e:
            print(f"[yt_infer] {e}")

threading.Thread(target=_yt_audio_reader, daemon=True).start()
threading.Thread(target=_yt_face_reader,  daemon=True).start()
threading.Thread(target=_yt_infer,        daemon=True).start()

@app.get("/yt_test")
def yt_test_page():
    return FileResponse(str(STATIC / "yt_test.html"))

class YTLoadIn(BaseModel):
    url: str

@app.post("/api/yt/load")
def yt_load(body: YTLoadIn):
    if not _ytdlp_ok:
        raise HTTPException(400, "yt-dlp not installed")
    _yt_cleanup()
    try:
        # extract all format info including stream URLs + HTTP headers
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with _ytdlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(body.url, download=False)

        vid_id = info.get("id", "")
        title  = info.get("title", "")
        fmts   = info.get("formats", [])
        print(f"[yt_load] {len(fmts)} formats found for: {title[:60]}")

        # ── pick best audio-only stream ────────────────────────────────────
        audio_fmts = [
            f for f in fmts
            if f.get("url")
            and f.get("acodec", "none") not in ("none", None)
            and f.get("vcodec", "none") in ("none", None)
        ]
        if not audio_fmts:  # fallback: any format that has audio
            audio_fmts = [f for f in fmts if f.get("url") and f.get("acodec", "none") not in ("none", None)]
        if not audio_fmts:
            raise ValueError("No audio stream found in formats")

        best_audio    = max(audio_fmts, key=lambda f: f.get("abr") or f.get("tbr") or 0)
        audio_url     = best_audio["url"]
        hdrs_audio    = best_audio.get("http_headers", {})
        hdr_audio_str = "".join(f"{k}: {v}\r\n" for k, v in hdrs_audio.items())
        print(f"[yt_load] audio url prefix: {audio_url[:60]}")
        print(f"[yt_load] audio headers: {list(hdrs_audio.keys())}")

        # ── pick ~360p video stream (good for face detection) ──────────────
        video_fmts = [
            f for f in fmts
            if f.get("url")
            and f.get("vcodec", "none") not in ("none", None)
        ]
        video_url = None
        if video_fmts:
            best_video    = min(video_fmts, key=lambda f: abs((f.get("height") or 0) - 360))
            video_url     = best_video["url"]
            hdrs_video    = best_video.get("http_headers", {})
            hdr_video_str = "".join(f"{k}: {v}\r\n" for k, v in hdrs_video.items())
            print(f"[yt_load] video {best_video.get('height')}p url prefix: {video_url[:60]}")

        # ── audio pipe: ffmpeg streams directly from CDN → 16kHz PCM ─────
        audio_proc = subprocess.Popen(
            [_FFMPEG,
             "-re",                          # real-time mode: match playback speed
             "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5",
             "-headers", hdr_audio_str,
             "-i", audio_url,
             "-vn", "-f", "s16le", "-ar", "16000", "-ac", "1", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        yt_state._audio_proc = audio_proc
        print("[yt_load] audio ffmpeg started, pid:", audio_proc.pid)

        # ── video pipe: ffmpeg streams video frames for face detection ──
        if video_url:
            video_proc = subprocess.Popen(
                [_FFMPEG,
                 "-re",
                 "-reconnect", "1", "-reconnect_streamed", "1",
                 "-reconnect_delay_max", "5",
                 "-headers", hdr_video_str,
                 "-i", video_url,
                 "-an", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", "320x240", "-r", "2", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            yt_state._video_proc = video_proc
            print(f"[yt_load] video ffmpeg started, pid: {video_proc.pid}")
        else:
            print("[yt_load] no video stream found — audio only")

        yt_state.running    = True
        yt_state.muted      = False
        yt_state.face_probs = None
        yt_state.stress_ema = 0.2  # reset stress on every new load
        return {"ok": True, "video_id": vid_id, "title": title}
    except Exception as e:
        print(f"[yt_load] ERROR: {e}")
        raise HTTPException(400, str(e))

@app.get("/api/yt/debug")
def yt_debug():
    """Full latest result with all probs."""
    return yt_state.latest

@app.get("/api/yt/status")
def yt_status():
    """Debug: current YT pipeline state."""
    ap = yt_state._audio_proc
    vp = yt_state._video_proc
    return {
        "running":     yt_state.running,
        "muted":       yt_state.muted,
        "audio_pid":   ap.pid if ap else None,
        "audio_alive": (ap.poll() is None) if ap else False,
        "video_pid":   vp.pid if vp else None,
        "video_alive": (vp.poll() is None) if vp else False,
        "face_probs":  yt_state.face_probs.tolist() if yt_state.face_probs is not None else None,
        "stress_ema":  round(yt_state.stress_ema, 3),
        "queue_size":  _yt_q.qsize(),
        "latest_emo":  yt_state.latest.get("emotion"),
        "clients":     len(yt_state._clients),
    }

@app.post("/api/yt/stop")
def yt_stop():
    _yt_cleanup()
    return {"ok": True}

class YTMuteIn(BaseModel):
    muted: bool

@app.post("/api/yt/mute")
def yt_mute_ep(body: YTMuteIn):
    yt_state.muted = body.muted
    # If video is still alive, revive running so face-only mode keeps working
    if body.muted:
        vp = yt_state._video_proc
        if vp is not None and vp.poll() is None:
            yt_state.running = True
    return {"muted": body.muted}

class YTTeachIn(BaseModel):
    emotion_idx: int

@app.post("/api/yt/teach")
def yt_teach(body: YTTeachIn):
    if body.emotion_idx < 0 or body.emotion_idx > 10:
        raise HTTPException(400, "emotion_idx must be 0-10")
    result = do_teach_step(body.emotion_idx)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    # Manual correction overrides auto-learn — pause auto for a full cooldown cycle
    yt_state.last_auto_teach = time.time()
    _yt_bcast_sync({"type": "teach_ack", **result})
    return result

@app.post("/api/yt/teach/save")
def yt_teach_save():
    torch.save(_model.state_dict(), ONLINE_CKPT)
    return {"saved": str(ONLINE_CKPT), "teach_count": yt_state.teach_count}

@app.get("/api/yt/teach/status")
def yt_teach_status():
    return {
        "teach_count": yt_state.teach_count,
        "buffer_size": len(yt_state.audio_buffer),
        "configured": _online_configured,
        "checkpoint": str(ONLINE_CKPT),
        "auto_learn": yt_state.auto_learn,
    }

class AutoLearnIn(BaseModel):
    enabled: bool

@app.post("/api/yt/auto_learn")
def yt_auto_learn(body: AutoLearnIn):
    yt_state.auto_learn = body.enabled
    yt_state.last_auto_teach = 0.0  # reset cooldown on toggle
    print(f"[online] auto_learn {'ON' if body.enabled else 'OFF'}")
    return {"auto_learn": yt_state.auto_learn}

@app.websocket("/ws/yt")
async def ws_yt_ep(ws: WebSocket):
    await ws.accept()
    yt_state._clients.add(ws)
    await ws.send_json({"type": "init", "latest": yt_state.latest})
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        yt_state._clients.discard(ws)

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n  MSTDN-A Dashboard >> http://localhost:8000")
    print(f"  YT Test           >> http://localhost:8000/yt_test\n")
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
