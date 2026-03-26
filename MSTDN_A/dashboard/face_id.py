"""
Face detection + recognition for MSTDN-A Dashboard.
Uses DeepFace (ArcFace model) for accurate face embeddings.
"""
from __future__ import annotations
import threading
import numpy as np
import cv2

_lock = threading.Lock()
_model_ready = False

THRESHOLD = 0.40   # cosine distance threshold (lower = more strict)


def _ensure_model():
    global _model_ready
    if _model_ready:
        return
    try:
        from deepface import DeepFace
        # warm-up: force model download on first call
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.represent(dummy, model_name="ArcFace",
                           enforce_detection=False, detector_backend="skip")
        _model_ready = True
        print("[face_id] ArcFace model ready.")
    except Exception as e:
        print(f"[face_id] Model warm-up failed: {str(e).encode('ascii', 'replace').decode()}")


def extract_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Extract ArcFace embedding. Tries strict detection first, then relaxed."""
    _ensure_model()
    from deepface import DeepFace
    h_img, w_img = frame_bgr.shape[:2]

    # Pass 1: strict — opencv must find a face
    try:
        result = DeepFace.represent(
            frame_bgr, model_name="ArcFace",
            enforce_detection=True, detector_backend="opencv",
        )
        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)
    except Exception as e:
        print(f"[face_id] strict detection failed ({str(e)[:70]}), trying relaxed…")

    # Pass 2: relaxed — accept result only if detected region < 85% of frame
    try:
        result = DeepFace.represent(
            frame_bgr, model_name="ArcFace",
            enforce_detection=False, detector_backend="opencv",
        )
        if result:
            region = result[0].get("facial_area", {})
            rw = region.get("w", 0)
            rh = region.get("h", 0)
            if rw > 10 and rh > 10 and (rw * rh) < (w_img * h_img * 0.85):
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                return emb / (np.linalg.norm(emb) + 1e-8)
            print("[face_id] relaxed: whole-frame fallback rejected — move closer/face camera")
    except Exception as e:
        print(f"[face_id] extract_embedding failed: {str(e)[:80]}")
    return None


def detect_faces(frame_bgr: np.ndarray) -> list[dict]:
    """
    Detect all faces in frame. Returns list of:
      {"bbox": (x,y,w,h), "embedding": np.ndarray | None, "crop": np.ndarray}
    """
    _ensure_model()
    try:
        from deepface import DeepFace
        results = DeepFace.represent(
            frame_bgr,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv",
        )
        faces = []
        for r in results:
            region = r.get("facial_area", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)
            emb = np.array(r["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            crop = frame_bgr[max(0,y):y+h, max(0,x):x+w] if w > 0 else None
            faces.append({"bbox": (x, y, w, h), "embedding": emb, "crop": crop})
        return faces
    except Exception as e:
        print(f"[face_id] detect_faces: {e}")
        return []


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
