"""Synthetic-data tests: no camera, no MediaPipe model needed for the maths."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signvoice import config as C
from signvoice.dataset import ISLDataset, build_arrays, signer_independent_split
from signvoice.landmarks import FrameLandmarks, RawSequence
from signvoice.normalize import (NormalizationError, SignerStats, StreamingNormalizer,
                                 find_active_span, mirror_raw, normalize_sequence, resample)
from signvoice.vocabulary import load_sentences, load_signs, validate

ASPECT = 16 / 9
RNG = np.random.default_rng(0)
HAND_SHAPE = RNG.normal(0, 0.03, (21, 3)).astype(np.float32)
HAND_SHAPE[0] = 0
HAND_SHAPE[9] = [0.0, -0.08, 0.0]       # middle MCP above wrist
FACE_SHAPE = RNG.normal(0, 0.03, (C.N_FACE, 3)).astype(np.float32)
FACE_SHAPE[C.F_NOSE_TIP] = 0
FACE_SHAPE[C.F_EDGE_A] = [-0.07, 0, 0]
FACE_SHAPE[C.F_EDGE_B] = [0.07, 0, 0]


def make_clip(T=40, right_wrist_xy=(0.30, 0.55), left_visible=True):
    """A canonical upright signer in image coords; right hand waves, left hand optional."""
    frames, ts = [], []
    for t in range(T):
        fl = FrameLandmarks.empty()
        pose = np.zeros((C.N_POSE, 4), np.float32)
        pose[:, 3] = 1.0
        pose[C.P_NOSE, :3] = [0.5, 0.25, -0.1]
        pose[C.P_LSH, :3] = [0.62, 0.45, 0.0]
        pose[C.P_RSH, :3] = [0.38, 0.45, 0.0]
        wob = 0.05 * np.sin(t / 5)
        pose[C.P_RWR, :3] = [right_wrist_xy[0] + wob, right_wrist_xy[1] - 0.1 * t / T, -0.2]
        pose[C.P_LWR, :3] = [0.70, 0.75, -0.1]
        fl.pose = pose
        fl.face = FACE_SHAPE + np.array([0.5, 0.27, -0.02], np.float32)
        fl.rh = HAND_SHAPE + np.array([*pose[C.P_RWR, :2], 0.0], np.float32)
        if left_visible:
            fl.lh = HAND_SHAPE * [-1, 1, 1] + np.array([0.70, 0.75, 0.0], np.float32)
        frames.append(fl)
        ts.append(t / 30)
    return RawSequence.from_frames(frames, ts, aspect=ASPECT)


def transform(raw, scale=1.0, angle=0.0, shift=(0.0, 0.0)):
    """Apply a similarity transform in true image geometry (x scaled by aspect)."""
    c, s = np.cos(angle), np.sin(angle)

    def tf(a):
        a = a.copy()
        X, Y = (a[..., 0] - 0.5) * raw.aspect, a[..., 1] - 0.5
        a[..., 0] = (scale * (X * c - Y * s)) / raw.aspect + 0.5 + shift[0]
        a[..., 1] = scale * (X * s + Y * c) + 0.5 + shift[1]
        a[..., 2] = a[..., 2] * scale
        return a
    out = RawSequence(tf(raw.pose[..., :3]), tf(raw.face), tf(raw.lh), tf(raw.rh), raw.timestamps, raw.aspect)
    out.pose = np.concatenate([out.pose, raw.pose[..., 3:]], axis=-1)
    return out


def test_vocabulary_is_100_signs_40_sentences_and_consistent():
    signs, sents = load_signs(), load_sentences()
    assert len(signs) == 100 and len(sents) == 40
    assert validate(signs, sents) == []


def test_face_mirror_is_involution():
    assert (C.FACE_MIRROR[C.FACE_MIRROR] == np.arange(C.N_FACE)).all()
    assert C.FeatureSpec().dim == 691


def test_invariant_to_scale_shift_and_roll():
    base = make_clip()
    ref = normalize_sequence(base).features
    spec = C.FeatureSpec().slices()
    moved = normalize_sequence(transform(base, scale=0.6, shift=(0.1, -0.05))).features
    np.testing.assert_allclose(moved, ref, atol=2e-3)           # everything, incl. face
    tilted = normalize_sequence(transform(base, scale=1.3, angle=np.deg2rad(12))).features
    for g in ("pose", "hands_body"):                            # face is deliberately not rotated
        np.testing.assert_allclose(tilted[:, spec[g]], ref[:, spec[g]], atol=2e-3)


def test_hand_shape_is_position_independent():
    spec = C.FeatureSpec().slices()["hands_shape"]
    a = normalize_sequence(make_clip(right_wrist_xy=(0.30, 0.55))).features[:, spec]
    b = normalize_sequence(make_clip(right_wrist_xy=(0.25, 0.65))).features[:, spec]
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_left_dominant_signer_matches_mirrored_right_dominant():
    right = make_clip()
    left = mirror_raw(right)                    # a left-handed signer doing the same sign
    f_right = normalize_sequence(right, dominant="right").features
    f_left = normalize_sequence(left, dominant="left").features
    np.testing.assert_allclose(f_left, f_right, atol=1e-4)


def test_gap_filling_bridges_short_gaps_only():
    raw = make_clip(T=40)
    raw.rh[10:13] = np.nan          # 3-frame dropout -> bridged
    raw.rh[20:30] = np.nan          # 10-frame dropout -> stays missing
    ns = normalize_sequence(raw, max_gap=5)
    assert ns.valid[10:13, 3].all() and not ns.valid[20:30, 3].any()
    assert np.isfinite(ns.features).all()
    hb = ns.features[:, C.FeatureSpec().slices()["hands_body"]]
    assert np.allclose(hb[20:30, 63:], 0)       # right-hand columns zero where undetected


def test_no_signer_raises():
    empty = RawSequence.from_frames([FrameLandmarks.empty()] * 5, np.arange(5) / 30)
    with pytest.raises(NormalizationError):
        normalize_sequence(empty)


def test_streaming_close_to_clip_normalisation():
    raw = make_clip(T=60)
    clip = normalize_sequence(raw).features
    sn = StreamingNormalizer()
    stream = []
    for t in range(len(raw)):
        fl = FrameLandmarks(raw.pose[t], raw.face[t], raw.lh[t], raw.rh[t])
        feats, valid = sn.update(fl, raw.aspect)
        stream.append(feats)
    np.testing.assert_allclose(np.stack(stream), clip, atol=5e-3)


def test_signer_stats_zscore_and_masked_columns():
    spec = C.FeatureSpec()
    ns = normalize_sequence(make_clip(left_visible=False), spec)
    st = SignerStats(spec)
    st.update(ns.features, ns.valid)
    z = st.transform(ns.features, ns.valid)
    sl = spec.slices()
    rh_cols = np.arange(sl["hands_body"].start + 63, sl["hands_body"].stop)
    assert abs(z[:, rh_cols].mean()) < 0.2                      # detected hand ~ zero mean
    assert np.allclose(z[:, sl["hands_body"].start:sl["hands_body"].start + 63], 0)  # absent left hand stays 0
    assert (z[:, sl["masks"]] == ns.features[:, sl["masks"]]).all()


def test_resample_and_active_span():
    x = np.arange(10, dtype=np.float32)[:, None]
    assert resample(x, 5).shape == (5, 1) and resample(x, 19)[1, 0] == pytest.approx(0.5)
    raw = make_clip(T=40)
    raw.rh[:10] = np.nan; raw.rh[30:] = np.nan; raw.lh[:] = np.nan   # hand only in frames 10..29
    s, e = find_active_span(raw, pad=2)
    assert (s, e) == (8, 32)


def test_dataset_roundtrip_splits_and_arrays(tmp_path):
    signs, sents = load_signs(), load_sentences()
    ds = ISLDataset(tmp_path / "ds")
    for sid, dom in (("S01", "right"), ("S02", "left"), ("S03", "right")):
        ds.register_signer(sid, dom, consent=True)
        orient = mirror_raw if dom == "left" else (lambda r: r)   # left-handed signer = mirror image
        for take in range(2):
            ds.add_clip(orient(make_clip()), "sign", signs[0], sid, "t")
        ds.add_clip(orient(make_clip(T=55)), "sentence", sents[0], sid, "t")
    ds2 = ISLDataset(tmp_path / "ds")                           # reload from disk
    assert len(ds2.records("ok")) == 9 and ds2.count("S01", "sign", signs[0].id) == 2
    bad = ds2.records("ok", "sign", "S01")[0]["clip_id"]
    ds2.set_status(bad, "rejected")
    assert ds2.count("S01", "sign", signs[0].id) == 1

    a = build_arrays(ds2, "sign", signs, fixed_len=16)
    assert a["X"].shape == (5, 16, 691) and set(a["y"]) == {1}
    b = build_arrays(ds2, "sentence", signs, fixed_len=None, sentences=sents)
    assert b["X"].shape == (3, 55, 691) and b["y_len"][0] == 3
    # mirrored left-dominant clip yields the same features as the right-dominant one (before z-scoring)
    f = lambda sid: normalize_sequence(ds2.load_raw(ds2.records("ok", "sentence", sid)[0]),
                                       dominant=ds2.dominant_hand(sid)).features
    np.testing.assert_allclose(f("S01"), f("S02"), atol=1e-4)

    split = signer_independent_split(["S01", "S02", "S03"], seed=1)
    assert sorted(split.values()) == ["test", "train", "val"]
    assert set(signer_independent_split(["S01", "S02"]).values()) == {"train", "test"}


def test_video_file_path_with_no_person_is_handled(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from signvoice.capture import process_video_file, quality_issues
    path = tmp_path / "blank.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(15):
        vw.write(np.zeros((240, 320, 3), np.uint8))
    vw.release()
    raw = process_video_file(path, model_complexity=1)
    assert len(raw) == 15 and raw.aspect == pytest.approx(320 / 240)
    assert quality_issues(raw)                                   # flags the empty clip
    with pytest.raises(NormalizationError):
        normalize_sequence(raw)


def test_recorder_flow_headless_and_resume(tmp_path, monkeypatch):
    """Drive the interactive recorder with scripted key presses over a synthetic video."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from signvoice.recorder import Recorder, build_tasks

    video = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(60):
        vw.write(np.zeros((240, 320, 3), np.uint8))
    vw.release()

    keys = iter([32] + [-1] * 12 + [32, 13, 27])        # start, ...recording..., stop, accept, quit
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: next(keys, 27))

    signs, sents = load_signs(), load_sentences()
    ds = ISLDataset(tmp_path / "ds")
    with pytest.raises(PermissionError):                         # no consent -> refuse to record
        Recorder(ds, "S01", [])
    ds.register_signer("S01", "right", consent=True)
    tasks = build_tasks(ds, "S01", signs, sents, ("sign",), reps_sign=2)
    assert len(tasks) == 200 and tasks[0].rep == 1 and tasks[100].rep == 2   # rounds, not blocks
    saved = Recorder(ds, "S01", tasks[:3], str(video), countdown=0.0, session="test").run()
    assert saved == 1
    rec = ds.records("ok")[0]
    assert rec["signer_id"] == "S01" and int(rec["n_frames"]) >= 10
    assert (ds.root / rec["path"]).exists()
    resumed = build_tasks(ds, "S01", signs, sents, ("sign",), reps_sign=2)
    assert len(resumed) == 199                                    # resumable: finished clip not repeated


def test_include_vocab_matching_logic_offline():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import check_vocab_against_include as chk
    assert chk.norm("55. Thank you") == "thank you" == chk.norm("THANK-YOU")
    include = {chk.norm(l): (c, l) for c, l in [("Greetings", "Thank you"), ("People", "Friend"),
                                                ("Pronouns", "she"), ("Days_and_Time", "Today")]}
    matched, unmatched = chk.match(load_signs(), include)
    assert set(matched) == {"THANK-YOU", "FRIEND", "HE-SHE", "TODAY"} and "EAT" in unmatched
