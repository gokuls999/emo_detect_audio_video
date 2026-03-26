"""Webcam + microphone capture threads for MSTDN-A Dashboard."""
from __future__ import annotations
import threading
import time
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

TARGET_SR = 16000   # sample rate the model expects

class VideoCapture:
    def __init__(self, camera_id: int = 0):
        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None
        self._cam_id = camera_id

    def start(self):
        # Wait for any previous thread to exit cleanly first
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=3.0)

        self._stop.clear()
        self._frame = None
        cap = cv2.VideoCapture(self._cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._thread = threading.Thread(target=self._run, args=(cap,), daemon=True)
        self._thread.start()

    def _run(self, cap):
        # Use cap passed in to avoid race when start() replaces self._cap
        while not self._stop.is_set():
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
            else:
                time.sleep(0.05)
        cap.release()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_jpeg(self, quality: int = 75) -> bytes | None:
        frame = self.get_frame()
        if frame is None:
            # dark placeholder so MJPEG stream stays alive while camera warms up
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (22, 27, 34)
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ret else None

    def stop(self):
        self._stop.set()


class AudioCapture:
    """Microphone capture that auto-selects a supported sample rate."""

    def __init__(self, device=None):
        self._device  = device
        self._cap_sr  = TARGET_SR   # actual capture rate (may differ from TARGET_SR)
        self._buf: deque = deque(maxlen=TARGET_SR * 10)
        self._lock    = threading.Lock()
        self._stream  = None

    def start(self):
        # Clean up previous stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Query device's native sample rate as preferred fallback
        native_sr = TARGET_SR
        if self._device is not None:
            try:
                info = sd.query_devices(self._device)
                native_sr = int(info["default_samplerate"])
            except Exception:
                pass

        # Try rates in priority order
        tried = []
        for sr in dict.fromkeys([TARGET_SR, native_sr, 48000, 44100, 22050]):
            try:
                stream = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=1024,
                    device=self._device,
                    callback=self._cb,
                )
                stream.start()
                self._stream = stream
                self._cap_sr = sr
                # Resize buffer for capture rate
                with self._lock:
                    self._buf = deque(maxlen=sr * 10)
                print(f"[audio] device={self._device} opened at {sr} Hz")
                return
            except Exception as e:
                tried.append((sr, str(e)))

        raise RuntimeError(f"Could not open mic device={self._device}. Tried: {tried}")

    def _cb(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._lock:
            self._buf.extend(mono.tolist())

    def get_audio(self, seconds: float = 3.0) -> np.ndarray:
        """Return `seconds` of audio resampled to TARGET_SR (16 kHz)."""
        n_cap = int(self._cap_sr * seconds)
        with self._lock:
            raw = list(self._buf)
        arr = np.array(raw, dtype=np.float32)
        if len(arr) >= n_cap:
            arr = arr[-n_cap:]
        else:
            arr = np.pad(arr, (n_cap - len(arr), 0))
        # Resample to 16 kHz if the device ran at a different rate
        if self._cap_sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=self._cap_sr, target_sr=TARGET_SR)
        n_out = int(TARGET_SR * seconds)
        if len(arr) >= n_out:
            return arr[-n_out:]
        return np.pad(arr, (n_out - len(arr), 0))

    def rms(self) -> float:
        n = self._cap_sr  # last 1 second at capture rate
        with self._lock:
            raw = list(self._buf)[-n:]
        if not raw:
            return 0.0
        return float(np.sqrt(np.mean(np.square(raw))))

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
