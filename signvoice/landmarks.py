"""MediaPipe Holistic wrapper and the raw landmark containers.

Raw landmarks are stored exactly as MediaPipe reports them (x, y in 0..1 image
coordinates, z as relative depth) with NaN for anything that was not detected.
All normalisation happens later (see normalize.py) so it can be re-tuned
without re-recording or re-extracting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import FACE_IDX, MASK_ORDER, N_FACE, N_HAND, N_POSE, POSE_IDX


# --------------------------------------------------------------------------
# One frame
# --------------------------------------------------------------------------
@dataclass
class FrameLandmarks:
    pose: np.ndarray  # (N_POSE, 4)  x, y, z, visibility
    face: np.ndarray  # (N_FACE, 3)
    lh: np.ndarray    # (21, 3)  signer's LEFT hand
    rh: np.ndarray    # (21, 3)  signer's RIGHT hand

    @staticmethod
    def empty() -> "FrameLandmarks":
        return FrameLandmarks(
            np.full((N_POSE, 4), np.nan, np.float32), np.full((N_FACE, 3), np.nan, np.float32),
            np.full((N_HAND, 3), np.nan, np.float32), np.full((N_HAND, 3), np.nan, np.float32))

    def present(self) -> tuple[bool, bool, bool, bool]:
        """(pose, face, lh, rh) detected?"""
        return tuple(bool(~np.isnan(a).all()) for a in (self.pose, self.face, self.lh, self.rh))


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------
class HolisticExtractor:
    """Runs MediaPipe Holistic and returns :class:`FrameLandmarks`.

    Feed it **un-mirrored** BGR frames (as read from the camera / video file).
    With un-mirrored input MediaPipe's "left hand" is the signer's real left
    hand.  Mirror only the *display* copy, never the frame you process.
    """

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, smooth_landmarks: bool = True):
        try:
            import mediapipe as mp
            solutions = mp.solutions
        except (ImportError, AttributeError) as exc:  # newer MediaPipe wheels dropped `solutions`
            raise ImportError(
                "MediaPipe Holistic (legacy solutions API) is required. "
                'Install it with:  pip install "mediapipe==0.10.14"') from exc
        self._holistic = solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks, enable_segmentation=False,
            refine_face_landmarks=False, min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def process(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._holistic.process(rgb)
        out = FrameLandmarks.empty()
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            out.pose = np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in POSE_IDX], np.float32)
        if res.face_landmarks:
            lm = res.face_landmarks.landmark
            out.face = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in FACE_IDX], np.float32)
        if res.left_hand_landmarks:
            out.lh = np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks.landmark], np.float32)
        if res.right_hand_landmarks:
            out.rh = np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks.landmark], np.float32)
        return out

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# A whole clip
# --------------------------------------------------------------------------
@dataclass
class RawSequence:
    """Raw landmarks of one clip.  NaN = not detected."""

    pose: np.ndarray        # (T, N_POSE, 4)
    face: np.ndarray        # (T, N_FACE, 3)
    lh: np.ndarray          # (T, 21, 3)
    rh: np.ndarray          # (T, 21, 3)
    timestamps: np.ndarray  # (T,) seconds from clip start
    aspect: float = 1.0     # frame width / height (MediaPipe x,y are normalised per axis)
    meta: dict = field(default_factory=dict)  # e.g. {"trim": [start, end], "width": 640}

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def fps(self) -> float:
        if len(self) < 2:
            return 0.0
        dt = float(self.timestamps[-1] - self.timestamps[0])
        return (len(self) - 1) / dt if dt > 0 else 0.0

    @property
    def mask(self) -> np.ndarray:
        """(T, 4) bool, columns follow config.MASK_ORDER."""
        return np.stack([~np.isnan(a.reshape(len(self), -1)).all(axis=1)
                         for a in (self.pose, self.face, self.lh, self.rh)], axis=1)

    def detection_rates(self) -> dict[str, float]:
        m = self.mask
        return {k: float(m[:, i].mean()) if len(self) else 0.0 for i, k in enumerate(MASK_ORDER)}

    @classmethod
    def from_frames(cls, frames: list[FrameLandmarks], timestamps, aspect: float = 1.0,
                    meta: dict | None = None) -> "RawSequence":
        if not frames:
            return cls(np.zeros((0, N_POSE, 4), np.float32), np.zeros((0, N_FACE, 3), np.float32),
                       np.zeros((0, N_HAND, 3), np.float32), np.zeros((0, N_HAND, 3), np.float32),
                       np.zeros(0, np.float32), aspect, meta or {})
        ts = np.asarray(timestamps, np.float32)
        return cls(np.stack([f.pose for f in frames]), np.stack([f.face for f in frames]),
                   np.stack([f.lh for f in frames]), np.stack([f.rh for f in frames]),
                   ts - ts[0], float(aspect), dict(meta or {}))

    def slice(self, start: int, end: int) -> "RawSequence":
        return RawSequence(self.pose[start:end], self.face[start:end], self.lh[start:end],
                           self.rh[start:end], self.timestamps[start:end] - self.timestamps[start:end][:1],
                           self.aspect, dict(self.meta))

    def trimmed(self) -> "RawSequence":
        """Apply the non-destructive ``meta['trim']`` = [start, end) if present."""
        trim = self.meta.get("trim")
        return self.slice(int(trim[0]), int(trim[1])) if trim else self

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pose=self.pose, face=self.face, lh=self.lh, rh=self.rh,
                            timestamps=self.timestamps, aspect=np.float32(self.aspect),
                            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "RawSequence":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["pose"], z["face"], z["lh"], z["rh"], z["timestamps"],
                       float(z["aspect"]), json.loads(str(z["meta"])))
