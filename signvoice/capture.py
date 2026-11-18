"""Camera / video input, live landmark overlay, a live feature stream and clip quality checks."""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import FeatureSpec, POSE_SKELETON
from .landmarks import FrameLandmarks, HolisticExtractor, RawSequence
from .normalize import NormalizationError, StreamingNormalizer, estimate_reference


# --------------------------------------------------------------------------
# Video source (webcam index or video file)
# --------------------------------------------------------------------------
class VideoSource:
    """Uniform wrapper over a webcam (``0``, ``1`` ...) or a video file path.

    Yields ``(frame_bgr, t_seconds)``.  Frames are returned UN-mirrored.  For files, ``t`` is
    ``frame_index / file_fps`` (exact); for cameras it is wall-clock time since the first frame,
    because webcams rarely deliver exactly the requested frame-rate."""

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720, fps: int = 30):
        import cv2
        self._cv2 = cv2
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.is_camera = isinstance(source, int)
        if self.is_camera:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY  # DSHOW opens much faster on Windows
            self.cap = cv2.VideoCapture(source, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        else:
            if not Path(source).exists():
                raise FileNotFoundError(source)
            self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {source!r} (is another app using the camera?)")
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or float(fps)
        self._t0: float | None = None
        self._n = 0

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self):
        """-> (frame, t) or (None, None) at end of stream / camera failure."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_camera:
            now = time.monotonic()
            self._t0 = now if self._t0 is None else self._t0
            t = now - self._t0
        else:
            t = self._n / self.nominal_fps
        self._n += 1
        return frame, t

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            frame, t = self.read()
            if frame is None:
                return
            yield frame, t

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------
def draw_landmarks(img: np.ndarray, fl: FrameLandmarks, mirror: bool = True) -> None:
    """Draw detected landmarks onto ``img`` in place.  Pass ``mirror=True`` when ``img`` is the
    selfie-flipped display copy (landmarks always come from the un-mirrored frame)."""
    import cv2
    import mediapipe as mp
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
    h, w = img.shape[:2]

    def px(p):
        x = (1 - p[0]) if mirror else p[0]
        return int(x * w), int(p[1] * h)

    for pts, color in ((fl.lh, (255, 160, 60)), (fl.rh, (60, 200, 255))):
        if np.isnan(pts).all():
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, px(pts[a]), px(pts[b]), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, px(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if not np.isnan(fl.pose).all():
        for a, b in POSE_SKELETON:
            if fl.pose[a, 3] > 0.4 and fl.pose[b, 3] > 0.4:
                cv2.line(img, px(fl.pose[a]), px(fl.pose[b]), (120, 255, 120), 2, cv2.LINE_AA)
    if not np.isnan(fl.face).all():
        for p in fl.face:
            cv2.circle(img, px(p), 1, (200, 200, 200), -1)


# --------------------------------------------------------------------------
# Live feature stream (what a real-time recogniser consumes)
# --------------------------------------------------------------------------
@dataclass
class StreamItem:
    frame: np.ndarray
    landmarks: FrameLandmarks
    features: np.ndarray | None   # (F,) normalised, None until a signer is in view
    valid: np.ndarray | None      # (4,) pose/face/lh/rh detected
    t: float


class FeatureStream:
    """camera/video -> Holistic -> streaming normalisation, one item per frame.

    ``window(n)`` returns the last ``n`` feature vectors as ``(n, F)`` (zero-padded at the start),
    ready to feed an LSTM/GRU, so a live recogniser can be dropped on top of this class."""

    def __init__(self, source: int | str = 0, spec: FeatureSpec | None = None, dominant: str = "right",
                 width: int = 1280, height: int = 720, fps: int = 30, model_complexity: int = 1,
                 buffer: int = 256):
        self.spec = spec or FeatureSpec()
        self.source = VideoSource(source, width, height, fps)
        self.extractor = HolisticExtractor(model_complexity=model_complexity)
        self.normalizer = StreamingNormalizer(self.spec, dominant)
        w, h = self.source.size
        self.aspect = w / h if h else 1.0
        self._buf: deque[np.ndarray] = deque(maxlen=buffer)

    def window(self, n: int) -> np.ndarray:
        out = np.zeros((n, self.spec.dim), np.float32)
        items = list(self._buf)[-n:]
        if items:
            out[-len(items):] = np.stack(items)
        return out

    def __iter__(self) -> Iterator[StreamItem]:
        for frame, t in self.source.frames():
            fl = self.extractor.process(frame)
            res = self.normalizer.update(fl, self.aspect)
            feats, valid = res if res is not None else (None, None)
            if feats is not None:
                self._buf.append(feats)
            yield StreamItem(frame, fl, feats, valid, t)

    def close(self) -> None:
        self.extractor.close()
        self.source.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Clip quality check
# --------------------------------------------------------------------------
def quality_issues(raw: RawSequence, span: tuple[int, int] | None = None, min_frames: int = 12,
                   min_fps: float = 12.0) -> list[str]:
    """Human-readable problems with a recorded clip (empty list = good)."""
    issues: list[str] = []
    r = raw.slice(*span) if span else raw
    if len(r) < min_frames:
        return [f"clip too short ({len(r)} frames)"]
    rates = r.detection_rates()
    if rates["pose"] < 0.9:
        issues.append(f"upper body detected in only {rates['pose']:.0%} of frames - step back / show shoulders")
    if rates["face"] < 0.8:
        issues.append(f"face detected in only {rates['face']:.0%} of frames")
    if max(rates["lh"], rates["rh"]) < 0.5:
        issues.append("hands visible in under half of the frames - keep hands inside the picture")
    if raw.fps and raw.fps < min_fps:
        issues.append(f"low frame-rate ({raw.fps:.0f} fps) - lower resolution or close other apps")
    try:
        estimate_reference(r)
    except NormalizationError as exc:
        issues.append(str(exc))
    return issues


def process_video_file(path: str | Path, model_complexity: int = 1) -> RawSequence:
    """Extract raw landmarks from a recorded video file (fresh tracker state per file)."""
    with VideoSource(str(path)) as src, HolisticExtractor(model_complexity=model_complexity) as ex:
        w, h = src.size
        frames, ts = [], []
        for frame, t in src.frames():
            frames.append(ex.process(frame))
            ts.append(t)
    return RawSequence.from_frames(frames, ts, aspect=w / h if h else 1.0,
                                   meta={"source_video": Path(path).name, "width": w, "height": h})
