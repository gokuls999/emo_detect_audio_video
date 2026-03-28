"""
Face detection + recognition for MSTDN-A Dashboard.
Uses InsightFace (ArcFace/buffalo_l) for accurate face detection & embeddings.
Runs on ONNX Runtime — no tensorflow required.
"""
from __future__ import annotations
import threading
import numpy as np
import cv2

_lock = threading.Lock()
_model_ready = False
_face_app = None

THRESHOLD = 0.40   # cosine distance threshold (lower = more strict)


def _ensure_model():
    global _model_ready, _face_app
    if _model_ready:
        return
    with _lock:
        if _model_ready:
            return
        try:
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))
            _model_ready = True
            print("[face_id] InsightFace ArcFace model ready.")
        except Exception as e:
            print(f"[face_id] Model warm-up failed: {str(e).encode('ascii', 'replace').decode()}")


def extract_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Extract ArcFace embedding from the largest face in frame."""
    _ensure_model()
    if _face_app is None:
        return None

    faces = _face_app.get(frame_bgr)
    if not faces:
        return None

    # pick largest face by bounding box area
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # reject if face region covers >85% of frame (likely false positive)
    h_img, w_img = frame_bgr.shape[:2]
    fw = best.bbox[2] - best.bbox[0]
    fh = best.bbox[3] - best.bbox[1]
    if (fw * fh) > (w_img * h_img * 0.85):
        return None

    emb = best.embedding.astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def detect_faces(frame_bgr: np.ndarray) -> list[dict]:
    """
    Detect all faces in frame. Returns list of:
      {"bbox": (x,y,w,h), "embedding": np.ndarray | None, "crop": np.ndarray}
    """
    _ensure_model()
    if _face_app is None:
        return []

    faces = _face_app.get(frame_bgr)
    if not faces:
        return []

    h_img, w_img = frame_bgr.shape[:2]
    result = []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        w = x2 - x1
        h = y2 - y1
        # skip if region is too large (whole-frame false positive)
        if (w * h) > (w_img * h_img * 0.85):
            continue
        emb = f.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        crop = frame_bgr[max(0, y1):y2, max(0, x1):x2] if w > 0 else None
        result.append({
            "bbox": (x1, y1, w, h),
            "embedding": emb,
            "crop": crop,
        })
    return result


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
