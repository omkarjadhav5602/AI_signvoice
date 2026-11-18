"""Signer-adaptive normalisation of raw Holistic landmarks.

Different signers differ in body size, distance to the camera, camera tilt,
framing, and handedness.  Three layers of adaptation remove those differences:

1. Body frame  (per clip / per frame)
   * origin  = mid-point of the shoulders (follows the body, removes framing/sway)
   * scale   = shoulder width, median over the clip (removes camera distance / body size)
   * rotation = shoulder-line roll, median over the clip (removes camera tilt)
   * x, y, z are put in the same units first (MediaPipe's y is normalised by
     image height, x and z by width), so distances and angles are meaningful.

2. Local frames
   * hand shape = each hand relative to its own wrist, divided by hand size
     (identical hand-shapes give identical features wherever the hand is)
   * face       = relative to the nose tip, divided by face width; no rotation,
     because head tilt/nod are themselves sign-language markers.

3. Handedness + per-signer statistics
   * left-handed signers are mirrored so the *dominant* hand always sits in the
     "right" slot (:func:`mirror_raw`)
   * :class:`SignerStats` z-scores features per signer (like per-speaker CMVN
     in speech recognition) and can be updated online for a new user.

Depth note: in MediaPipe, hand z is relative to the wrist and face z is
relative to the face centre, so only pose z is body-relative.  Hand "body" z is
therefore kept as local relative depth (just scaled).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import (F_EDGE_A, F_EDGE_B, F_NOSE_TIP, FACE_MIRROR, FeatureSpec, H_MIDDLE_MCP, H_WRIST,
                     N_HAND, P_LSH, P_RSH, POSE_MIRROR, VIS_THRESH)
from .landmarks import FrameLandmarks, RawSequence


class NormalizationError(RuntimeError):
    """Raised when no usable signer (shoulders) was detected in the clip."""


# --------------------------------------------------------------------------
# Reference (per-signer geometry estimate)
# --------------------------------------------------------------------------
@dataclass
class Reference:
    scale: float              # shoulder width (isotropic image units)
    roll: float               # shoulder-line angle, radians
    origin: np.ndarray        # (3,) mid-shoulder fallback
    hand_scale: np.ndarray    # (2,) wrist -> middle-MCP distance for L, R (NaN = unknown)
    face_scale: float         # face width (NaN = unknown)

    def blend(self, new: "Reference", alpha: float) -> "Reference":
        def ema(old, cur):
            return cur if np.isnan(old) else old if np.isnan(cur) else (1 - alpha) * old + alpha * cur
        return Reference(
            scale=(1 - alpha) * self.scale + alpha * new.scale,
            roll=(1 - alpha) * self.roll + alpha * new.roll,
            origin=(1 - alpha) * self.origin + alpha * new.origin,
            hand_scale=np.array([ema(a, b) for a, b in zip(self.hand_scale, new.hand_scale)]),
            face_scale=ema(self.face_scale, new.face_scale))


@dataclass
class NormalizedSequence:
    features: np.ndarray   # (T, F) float32
    valid: np.ndarray      # (T, 4) bool  pose/face/lh/rh available (after gap filling)
    reference: Reference
    spec: FeatureSpec


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def mirror_raw(raw: RawSequence) -> RawSequence:
    """Left/right mirror a clip (for left-dominant signers)."""
    pose, face = raw.pose[:, POSE_MIRROR].copy(), raw.face[:, FACE_MIRROR].copy()
    lh, rh = raw.rh.copy(), raw.lh.copy()
    for a in (pose, face, lh, rh):
        a[..., 0] = 1.0 - a[..., 0]
    return replace(raw, pose=pose, face=face, lh=lh, rh=rh)


def _isotropic(raw: RawSequence):
    """Copies with y divided by the aspect ratio so x, y, z share one unit (image width)."""
    out = []
    for a in (raw.pose, raw.face, raw.lh, raw.rh):
        a = a.astype(np.float32, copy=True)
        a[..., 1] /= raw.aspect
        out.append(a)
    return out


def _shoulders_ok(pose: np.ndarray) -> np.ndarray:
    sh = pose[:, [P_LSH, P_RSH]]
    finite = np.isfinite(sh[..., :3]).all(axis=(1, 2))
    return finite & (sh[:, 0, 3] >= VIS_THRESH) & (sh[:, 1, 3] >= VIS_THRESH)


def _nanmedian(x: np.ndarray, axis=None):
    x = np.asarray(x)
    return np.nan if not np.isfinite(x).any() else np.nanmedian(x, axis=axis)


def _hand_size(hand: np.ndarray) -> np.ndarray:
    return np.linalg.norm(hand[:, H_MIDDLE_MCP] - hand[:, H_WRIST], axis=-1)


def estimate_reference(raw: RawSequence) -> Reference:
    """Per-clip geometry of the signer (medians are robust to detection glitches)."""
    pose, face, lh, rh = _isotropic(raw)
    ok = _shoulders_ok(pose)
    if not ok.any():
        raise NormalizationError("no frame with both shoulders visible - is the signer in frame?")
    ls, rs = pose[ok, P_LSH, :3], pose[ok, P_RSH, :3]
    vec = ls - rs
    scale = float(np.median(np.hypot(vec[:, 0], vec[:, 1])))
    if scale < 1e-4:
        raise NormalizationError("shoulder width is ~0 - landmarks are degenerate")
    roll = float(np.median(np.arctan2(vec[:, 1], vec[:, 0])))
    origin = np.median((ls + rs) / 2, axis=0)
    hand_scale = np.array([_nanmedian(_hand_size(lh)), _nanmedian(_hand_size(rh))])
    face_w = np.linalg.norm(face[:, F_EDGE_A, :2] - face[:, F_EDGE_B, :2], axis=-1)
    return Reference(scale, roll, origin, hand_scale, float(_nanmedian(face_w)))


def _fill_rows(arr: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Replace invalid rows by linear interpolation (nearest value at the ends)."""
    if valid.all():
        return arr
    out, t = arr.copy(), np.arange(len(arr))
    if not valid.any():
        out[:] = fallback
        return out
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t, t[valid], arr[valid, d])
    return out


def fill_gaps(arr: np.ndarray, valid: np.ndarray, max_gap: int):
    """Linearly interpolate NaN gaps of at most ``max_gap`` frames that are bounded
    by valid frames on both sides.  Longer gaps stay NaN.  Returns (array, valid)."""
    valid = valid.copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2 or max_gap <= 0:
        return arr, valid
    flat = arr.reshape(len(arr), -1).copy()
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(a + 1, b) - a) / (b - a)
            flat[a + 1:b] = flat[a] * (1 - w[:, None]) + flat[b] * w[:, None]
            valid[a + 1:b] = True
    return flat.reshape(arr.shape), valid


def _to_body(pts: np.ndarray, origin: np.ndarray, ref: Reference, subtract_z: bool) -> np.ndarray:
    d = pts - origin[:, None, :]
    c, s = np.cos(-ref.roll), np.sin(-ref.roll)
    x = d[..., 0] * c - d[..., 1] * s
    y = d[..., 0] * s + d[..., 1] * c
    z = d[..., 2] if subtract_z else pts[..., 2]
    return np.stack([x, y, z], axis=-1) / ref.scale


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def normalize_sequence(raw: RawSequence, spec: FeatureSpec | None = None,
                       reference: Reference | None = None, dominant: str = "right",
                       max_gap: int = 5) -> NormalizedSequence:
    """Raw landmarks -> (T, F) signer-normalised feature matrix.

    reference : use a given geometry (streaming) instead of estimating it from the clip.
    dominant  : "left" mirrors the clip so the dominant hand is always the "right" one.
    max_gap   : longest run of missing frames (per body part) to bridge by interpolation.
    """
    spec = spec or FeatureSpec()
    T = len(raw)
    if T == 0:
        raise NormalizationError("empty clip")
    if dominant == "left":
        raw = mirror_raw(raw)
    ref = reference or estimate_reference(raw)
    pose, face, lh, rh = _isotropic(raw)

    # --- body frame --------------------------------------------------------
    sh_ok = _shoulders_ok(pose)
    mid = (pose[:, P_LSH, :3] + pose[:, P_RSH, :3]) / 2
    origin = _fill_rows(mid, sh_ok, ref.origin)

    v_pose = np.isfinite(pose[:, :, :3]).all(axis=(1, 2))
    v_face = np.isfinite(face).all(axis=(1, 2))
    v_lh = np.isfinite(lh).all(axis=(1, 2))
    v_rh = np.isfinite(rh).all(axis=(1, 2))

    pose_b = _to_body(pose[..., :3], origin, ref, subtract_z=True)
    lh_b = _to_body(lh, origin, ref, subtract_z=False)
    rh_b = _to_body(rh, origin, ref, subtract_z=False)

    # --- hand shape: wrist-relative, hand-size scaled ----------------------
    def hand_shape(h: np.ndarray, ref_size: float) -> np.ndarray:
        size = ref_size if np.isfinite(ref_size) else _hand_size(h)[:, None, None]
        return (h - h[:, H_WRIST:H_WRIST + 1]) / np.maximum(size, 1e-4)

    lh_s, rh_s = hand_shape(lh, ref.hand_scale[0]), hand_shape(rh, ref.hand_scale[1])

    # --- face: nose-centred, face-width scaled, no rotation ----------------
    fw = ref.face_scale
    if not np.isfinite(fw):
        fw = np.linalg.norm(face[:, F_EDGE_A, :2] - face[:, F_EDGE_B, :2], axis=-1)[:, None, None]
    face_l = (face - face[:, F_NOSE_TIP:F_NOSE_TIP + 1]) / np.maximum(fw, 1e-4)

    # --- bridge short detection gaps, then zero-fill the rest --------------
    parts = {}
    for name, arrs, v in (("pose", [pose_b], v_pose), ("face", [face_l], v_face),
                          ("lh", [lh_b, lh_s], v_lh), ("rh", [rh_b, rh_s], v_rh)):
        filled, v_out = [], v
        for a in arrs:
            fa, v_out = fill_gaps(a, v, max_gap)
            filled.append(np.nan_to_num(fa, nan=0.0))
        parts[name] = (filled, v_out)
    valid = np.stack([parts[k][1] for k in ("pose", "face", "lh", "rh")], axis=1)

    blocks = []
    if spec.pose:
        blocks.append(parts["pose"][0][0].reshape(T, -1))
    if spec.hands_body:
        blocks += [parts["lh"][0][0].reshape(T, -1), parts["rh"][0][0].reshape(T, -1)]
    if spec.hands_shape:
        blocks += [parts["lh"][0][1].reshape(T, -1), parts["rh"][0][1].reshape(T, -1)]
    if spec.face:
        blocks.append(parts["face"][0][0].reshape(T, -1))
    if spec.masks:
        blocks.append(valid.astype(np.float32))
    feats = np.concatenate(blocks, axis=1).astype(np.float32)
    assert feats.shape[1] == spec.dim, (feats.shape, spec.dim)
    return NormalizedSequence(feats, valid, ref, spec)


# --------------------------------------------------------------------------
# Online (frame-by-frame) normalisation for live use
# --------------------------------------------------------------------------
class StreamingNormalizer:
    """Causal version of :func:`normalize_sequence` for live camera input.

    The signer geometry (scale / roll / hand size) is tracked with an exponential
    moving average, so the first frames adapt quickly and later ones stay stable.
    Returns ``None`` until a signer's shoulders have been seen.
    """

    def __init__(self, spec: FeatureSpec | None = None, dominant: str = "right", alpha: float = 0.05):
        self.spec, self.dominant, self.alpha = spec or FeatureSpec(), dominant, alpha
        self.ref: Reference | None = None

    def reset(self) -> None:
        self.ref = None

    def update(self, frame: FrameLandmarks, aspect: float = 1.0):
        """-> (features (F,), valid (4,)) or None if no signer yet."""
        raw = RawSequence.from_frames([frame], [0.0], aspect)
        if self.dominant == "left":
            raw = mirror_raw(raw)
        try:
            cur = estimate_reference(raw)
            self.ref = cur if self.ref is None else self.ref.blend(cur, self.alpha)
        except NormalizationError:
            if self.ref is None:
                return None
        ns = normalize_sequence(raw, self.spec, reference=self.ref, dominant="right", max_gap=0)
        return ns.features[0], ns.valid[0]


# --------------------------------------------------------------------------
# Per-signer feature statistics (signer adaptation, layer 3)
# --------------------------------------------------------------------------
class SignerStats:
    """Running mean / std of one signer's features, counting only frames in which
    the relevant body part was actually detected.  ``update`` can be called
    incrementally, so the same class adapts to a new user online."""

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        d = self.spec.dim
        self.n, self.s, self.ss = np.zeros(d), np.zeros(d), np.zeros(d)
        self._cm = self.spec.column_mask_index()

    def _weights(self, valid: np.ndarray) -> np.ndarray:
        return valid[:, np.maximum(self._cm, 0)] & (self._cm >= 0)[None, :]

    def update(self, features: np.ndarray, valid: np.ndarray) -> None:
        w = self._weights(valid)
        x = features.astype(np.float64)
        self.n += w.sum(0)
        self.s += (x * w).sum(0)
        self.ss += (x * x * w).sum(0)

    @property
    def mean(self) -> np.ndarray:
        return np.divide(self.s, self.n, out=np.zeros_like(self.s), where=self.n > 0)

    @property
    def std(self) -> np.ndarray:
        var = np.divide(self.ss, self.n, out=np.ones_like(self.ss), where=self.n > 1) - self.mean ** 2
        std = np.sqrt(np.maximum(var, 0))
        return np.where((self.n > 1) & (std > 1e-3), std, 1.0)

    def transform(self, features: np.ndarray, valid: np.ndarray) -> np.ndarray:
        w = self._weights(valid)
        out = np.where(w, (features - self.mean) / self.std, features)  # mask columns pass through
        out = np.where((self._cm >= 0)[None, :] & ~w, 0.0, out)         # undetected parts stay 0
        return out.astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, n=self.n, s=self.s, ss=self.ss)

    @classmethod
    def load(cls, path, spec: FeatureSpec | None = None) -> "SignerStats":
        obj = cls(spec)
        with np.load(path) as z:
            obj.n, obj.s, obj.ss = z["n"], z["s"], z["ss"]
        return obj


# --------------------------------------------------------------------------
# Sequence utilities
# --------------------------------------------------------------------------
def resample(x: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample (T, F) to (length, F) - gives every clip the same length."""
    T = len(x)
    if T == 0:
        return np.zeros((length,) + x.shape[1:], x.dtype)
    if T == 1:
        return np.repeat(x, length, axis=0)
    src = np.linspace(0, T - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return (x[lo] * (1 - w) + x[hi] * w).astype(x.dtype)


def add_velocity(x: np.ndarray) -> np.ndarray:
    """Append frame-to-frame differences: (T, F) -> (T, 2F)."""
    v = np.vstack([np.zeros((1, x.shape[1]), x.dtype), np.diff(x, axis=0)])
    return np.concatenate([x, v], axis=1)


def active_frames(raw: RawSequence, rest_y: float = 1.0) -> np.ndarray:
    """Frames in which at least one hand is detected and raised above ``rest_y``
    shoulder-widths below the shoulder line (i.e. not hanging at the side)."""
    spec = FeatureSpec(pose=False, hands_body=True, hands_shape=False, face=False, masks=False)
    try:
        ns = normalize_sequence(raw, spec)
    except NormalizationError:
        return np.zeros(len(raw), bool)
    hb = ns.features[:, spec.slices()["hands_body"]].reshape(len(raw), 2, N_HAND, 3)
    wrist_y = hb[:, :, H_WRIST, 1]  # body frame: y grows downwards, shoulders at 0
    return ((wrist_y < rest_y) & ns.valid[:, 2:4]).any(axis=1)


def find_active_span(raw: RawSequence, rest_y: float = 1.0, pad: int = 6, min_active: int = 3):
    """(start, end) frame span of the signing, with ``pad`` frames of context either side.
    Falls back to the whole clip when too few active frames are found."""
    act = np.flatnonzero(active_frames(raw, rest_y))
    if len(act) < min_active:
        return 0, len(raw)
    return max(int(act[0]) - pad, 0), min(int(act[-1]) + pad + 1, len(raw))
