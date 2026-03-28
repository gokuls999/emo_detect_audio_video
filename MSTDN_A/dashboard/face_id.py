"""
Face detection + recognition for MSTDN-A Dashboard.
Uses OpenCV Haar Cascade for detection and histogram-based embeddings
for lightweight face identification (no deepface/tensorflow required).
"""
from __future__ import annotations
import threading
import numpy as np
import cv2

_lock = threading.Lock()
_model_ready = False
_face_cascade: cv2.CascadeClassifier | None = None

THRESHOLD = 0.40   # cosine distance threshold (lower = more strict)
EMBED_SIZE = 64     # resize face crop to this square before embedding


def _ensure_model():
    global _model_ready, _face_cascade
    if _model_ready:
        return
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
        _model_ready = True
        print("[face_id] OpenCV Haar cascade ready.")
    except Exception as e:
        print(f"[face_id] Model warm-up failed: {e}")


def _compute_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """Convert a face crop into a normalized embedding vector."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE))
    # histogram equalize for lighting invariance
    equalized = cv2.equalizeHist(resized)
    emb = equalized.flatten().astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def extract_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Extract face embedding from frame. Returns None if no face found."""
    _ensure_model()
    if _face_cascade is None:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    # pick largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]

    # reject if face region covers >85% of frame (likely no real face)
    fh, fw = frame_bgr.shape[:2]
    if (w * h) > (fw * fh * 0.85):
        return None

    crop = frame_bgr[max(0, y):y+h, max(0, x):x+w]
    if crop.size == 0:
        return None
    return _compute_embedding(crop)


def detect_faces(frame_bgr: np.ndarray) -> list[dict]:
    """
    Detect all faces in frame. Returns list of:
      {"bbox": (x,y,w,h), "embedding": np.ndarray | None, "crop": np.ndarray}
    """
    _ensure_model()
    if _face_cascade is None:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detections = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(detections) == 0:
        return []

    fh, fw = frame_bgr.shape[:2]
    faces = []
    for (x, y, w, h) in detections:
        # skip if region is too large (whole-frame false positive)
        if (w * h) > (fw * fh * 0.85):
            continue
        crop = frame_bgr[max(0, y):y+h, max(0, x):x+w]
        emb = _compute_embedding(crop) if crop.size > 0 else None
        faces.append({"bbox": (int(x), int(y), int(w), int(h)),
                       "embedding": emb, "crop": crop})
    return faces


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def identify(
    face_embedding: np.ndarray,
    registered: list[dict],   # [{"participant_id": int, "name": str, "embedding": np.ndarray}]
) -> tuple[int | None, str, float]:
    """
    Match face_embedding against registered embeddings.
    Returns (participant_id, name, distance). participant_id=None if no match.
    """
    if not registered or face_embedding is None:
        return None, "Unknown", 1.0

    best_id, best_name, best_dist = None, "Unknown", THRESHOLD
    for r in registered:
        if r["embedding"] is None:
            continue
        dist = cosine_distance(face_embedding, r["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_id   = r["participant_id"]
            best_name = r["name"]
    return best_id, best_name, best_dist


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],   # from identify_all()
) -> np.ndarray:
    """Draw bounding boxes + names on frame."""
    out = frame.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        pid   = d["participant_id"]
        name  = d["name"]
        dist  = d["distance"]
        color = (0, 220, 100) if pid is not None else (100, 100, 100)
        label = f"{name} ({(1-dist)*100:.0f}%)" if pid is not None else "Unknown"
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y-th-10), (x+tw+8, y), color, -1)
        cv2.putText(out, label, (x+4, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    return out


def identify_all(
    frame_bgr: np.ndarray,
    registered: list[dict],
) -> list[dict]:
    """
    Full pipeline: detect all faces, identify each one.
    Returns list of {participant_id, name, distance, bbox, embedding}
    """
    faces = detect_faces(frame_bgr)
    result = []
    for f in faces:
        pid, name, dist = identify(f["embedding"], registered)
        result.append({
            "participant_id": pid,
            "name":           name,
            "distance":       dist,
            "bbox":           f["bbox"],
            "embedding":      f["embedding"],
        })
    return result


# warm up model in background so first session start is fast
threading.Thread(target=_ensure_model, daemon=True).start()
