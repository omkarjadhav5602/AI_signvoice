"""Feature layout and constants shared by the whole input module.

Everything that decides *which* landmarks are kept, and *in what order* they
appear in a feature vector, lives here so that recording, extraction,
normalisation and training code can never disagree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Pose: upper-body subset of MediaPipe's 33 pose landmarks
# --------------------------------------------------------------------------
# nose, shoulders, elbows, wrists, pinky / index / thumb knuckles (L then R).
POSE_IDX = (0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
N_POSE = len(POSE_IDX)
# Positions *inside the reduced array* (not MediaPipe indices).
P_NOSE, P_LSH, P_RSH, P_LEL, P_REL, P_LWR, P_RWR = 0, 1, 2, 3, 4, 5, 6
POSE_MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11])
POSE_SKELETON = ((P_LSH, P_RSH), (P_LSH, P_LEL), (P_LEL, P_LWR), (P_RSH, P_REL), (P_REL, P_RWR))

# --------------------------------------------------------------------------
# Hands: all 21 landmarks per hand
# --------------------------------------------------------------------------
N_HAND = 21
H_WRIST, H_MIDDLE_MCP = 0, 9

# --------------------------------------------------------------------------
# Face: a compact subset of the 468 FaceMesh points that carry the
# non-manual markers of sign language (mouth, eyes, eyebrows) plus head pose
# (nose, face oval).  132 points instead of 468 keeps clips small.
# --------------------------------------------------------------------------
_LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409)
_LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415)
_RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
_LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
_RIGHT_BROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
_LEFT_BROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
_NOSE = (1, 4, 6, 168)
_FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
              377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

FACE_IDX = tuple(dict.fromkeys(
    _LIPS_OUTER + _LIPS_INNER + _RIGHT_EYE + _LEFT_EYE + _RIGHT_BROW + _LEFT_BROW + _NOSE + _FACE_OVAL))
N_FACE = len(FACE_IDX)
_FACE_POS = {m: i for i, m in enumerate(FACE_IDX)}
F_NOSE_TIP = _FACE_POS[1]
F_EDGE_A, F_EDGE_B = _FACE_POS[234], _FACE_POS[454]  # left/right face edge -> face width


def _build_face_mirror() -> np.ndarray:
    """Permutation that maps each face point to its left/right mirror point."""
    pairs = list(zip(_RIGHT_EYE, _LEFT_EYE)) + list(zip(_RIGHT_BROW, _LEFT_BROW)) + [
        (61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (185, 409), (40, 270), (39, 269),
        (37, 267), (78, 308), (95, 324), (88, 318), (178, 402), (87, 317), (191, 415), (80, 310),
        (81, 311), (82, 312), (338, 109), (297, 67), (332, 103), (284, 54), (251, 21), (389, 162),
        (356, 127), (454, 234), (323, 93), (361, 132), (288, 58), (397, 172), (365, 136),
        (379, 150), (378, 149), (400, 176), (377, 148)]
    mirror = {}
    for a, b in pairs:
        mirror[a], mirror[b] = b, a
    for self_symmetric in (0, 17, 13, 14, 10, 152, 1, 4, 6, 168):
        mirror[self_symmetric] = self_symmetric
    missing = [m for m in FACE_IDX if m not in mirror]
    if missing:  # fail loudly at import time rather than silently corrupt data
        raise RuntimeError(f"face mirror table incomplete: {missing}")
    return np.array([_FACE_POS[mirror[m]] for m in FACE_IDX])


FACE_MIRROR = _build_face_mirror()

# --------------------------------------------------------------------------
# Presence masks and feature groups
# --------------------------------------------------------------------------
MASK_ORDER = ("pose", "face", "lh", "rh")  # order of the (T, 4) presence array
VIS_THRESH = 0.5                           # min pose visibility for shoulders to be trusted


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature groups go into the normalised vector (order is fixed)."""

    pose: bool = True          # upper-body pose in the signer's body frame       (13*3)
    hands_body: bool = True    # both hands in the body frame (where the hand is) (2*21*3)
    hands_shape: bool = True   # both hands relative to own wrist (hand shape)    (2*21*3)
    face: bool = True          # face subset, nose-centred, face-width scaled     (132*3)
    masks: bool = True         # 4 flags: was pose/face/left/right hand detected  (4)

    def groups(self) -> list[tuple[str, int]]:
        g: list[tuple[str, int]] = []
        if self.pose:
            g.append(("pose", N_POSE * 3))
        if self.hands_body:
            g.append(("hands_body", 2 * N_HAND * 3))
        if self.hands_shape:
            g.append(("hands_shape", 2 * N_HAND * 3))
        if self.face:
            g.append(("face", N_FACE * 3))
        if self.masks:
            g.append(("masks", len(MASK_ORDER)))
        return g

    @property
    def dim(self) -> int:
        return sum(n for _, n in self.groups())

    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for name, n in self.groups():
            out[name] = slice(start, start + n)
            start += n
        return out

    def column_mask_index(self) -> np.ndarray:
        """For every feature column, the index into MASK_ORDER it depends on (-1 = none)."""
        cols = []
        for name, n in self.groups():
            if name == "pose":
                cols.append(np.full(n, 0))
            elif name == "face":
                cols.append(np.full(n, 1))
            elif name in ("hands_body", "hands_shape"):
                half = n // 2
                cols += [np.full(half, 2), np.full(half, 3)]
            else:
                cols.append(np.full(n, -1))
        return np.concatenate(cols)
