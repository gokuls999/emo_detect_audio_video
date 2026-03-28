"""
Face emotion recognition using ONNX FERPlus model.
No tensorflow/deepface dependency — runs on onnxruntime.
Input: BGR frame or face crop → Output: 11-class MAFW emotion probabilities.
"""
from __future__ import annotations
import threading
import numpy as np
import cv2
from pathlib import Path

_lock = threading.Lock()
_session = None
_cascade = None
_ready = False

# FERPlus 8 classes → MAFW 11-class mapping
# FERPlus: 0=neutral, 1=happiness, 2=surprise, 3=sadness, 4=anger, 5=disgust, 6=fear, 7=contempt
_FERPLUS_TO_MAFW = {
    0: 4,   # neutral   → neutral (idx 4)
    1: 3,   # happiness → happiness (idx 3)
    2: 6,   # surprise  → surprise (idx 6)
    3: 5,   # sadness   → sadness (idx 5)
    4: 0,   # anger     → anger (idx 0)
    5: 1,   # disgust   → disgust (idx 1)
    6: 2,   # fear      → fear (idx 2)
    7: 7,   # contempt  → contempt (idx 7)
}

_FERPLUS_NAMES = ["neutral", "happiness", "surprise", "sadness",
                  "anger", "disgust", "fear", "contempt"]


def _ensure():
    global _session, _cascade, _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        try:
            import onnxruntime as ort
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "emotion-ferplus-8.onnx"
            if not model_path.exists():
                print("[face_emotion] Downloading FERPlus ONNX model...")
                import urllib.request
                url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
                urllib.request.urlretrieve(url, str(model_path))
                print("[face_emotion] Download complete.")
            _session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            _cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            _ready = True
            print("[face_emotion] FERPlus ONNX model ready.")
        except Exception as e:
            print(f"[face_emotion] Init failed: {e}")


def _preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Convert a face crop to model input: 1x1x64x64 float32."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    # normalize to 0-1 range
    img = resized.astype(np.float32) / 255.0
    return img.reshape(1, 1, 64, 64)


def _raw_to_mafw(logits: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert FERPlus 8-class logits to MAFW 11-class probs with temperature scaling."""
    # softmax on raw logits
    exp = np.exp(logits - logits.max())
    ferplus_probs = exp / exp.sum()

    # dominant emotion from FERPlus
    dom_idx = int(ferplus_probs.argmax())
    dom_name = _FERPLUS_NAMES[dom_idx]

    # map to 11-class MAFW
    fp = np.zeros(11, dtype=np.float32)
    for fer_idx, mafw_idx in _FERPLUS_TO_MAFW.items():
        fp[mafw_idx] = ferplus_probs[fer_idx]

    # temperature scaling T=2.5 to flatten neutral-heavy distributions
    known = list(_FERPLUS_TO_MAFW.values())
    sub = fp[known]
    log_p = np.log(np.clip(sub, 1e-10, 1.0)) / 2.5
    log_p -= log_p.max()
    sub = np.exp(log_p)
    sub /= sub.sum()
    for j, idx in enumerate(known):
        fp[idx] = sub[j]

    # anchor: ensure dominant emotion leads the distribution
    mafw_dom_idx = _FERPLUS_TO_MAFW.get(dom_idx)
    if mafw_dom_idx is not None:
        cur_max = fp.max()
        if fp[mafw_dom_idx] < cur_max:
            fp[mafw_dom_idx] = cur_max + 0.08
            fp /= fp.sum()

    return fp, dom_name


def analyze_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Detect face in frame and return (mafw_11_probs, confidence).
    Returns (None, 0.0) if no face detected.
    """
    _ensure()
    if _session is None or _cascade is None:
        return None, 0.0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    if len(faces) == 0:
        return None, 0.0

    # pick largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]

    # reject whole-frame false positives
    fh, fw = frame_bgr.shape[:2]
    if (w * h) > (fw * fh * 0.85):
        return None, 0.0

    crop = frame_bgr[max(0, y):y+h, max(0, x):x+w]
    if crop.size == 0:
        return None, 0.0

    inp = _preprocess_face(crop)
    logits = _session.run(None, {"Input3": inp})[0][0]
    probs, _ = _raw_to_mafw(logits)
    conf = float(probs.max())
    return probs, conf


def analyze_crop(face_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Analyze a pre-cropped face image (no detection needed).
    Returns (mafw_11_probs, confidence).
    """
    _ensure()
    if _session is None:
        return None, 0.0

    if face_bgr.size == 0:
        return None, 0.0

    inp = _preprocess_face(face_bgr)
    logits = _session.run(None, {"Input3": inp})[0][0]
    probs, _ = _raw_to_mafw(logits)
    conf = float(probs.max())
    return probs, conf


# warm up in background
threading.Thread(target=_ensure, daemon=True).start()
