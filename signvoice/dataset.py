"""On-disk dataset: one .npz of raw landmarks per clip + CSV manifest + signer registry.

dataset/
  manifest.csv          one row per clip (append / status-change only through this class)
  signers.csv           signer registry (consent, handedness, ...)
  clips/<signer>/<sign|sentence>/<label_id>/<clip_id>.npz
  videos/...            optional raw video (only if recorded with --save-video)
  processed/            arrays written by scripts/build_features.py
"""
from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import FeatureSpec
from .landmarks import RawSequence
from .normalize import (NormalizationError, SignerStats, normalize_sequence, resample)
from .vocabulary import Sentence, Sign, gloss_to_index

MANIFEST_FIELDS = [
    "clip_id", "clip_type", "label_id", "label", "gloss", "signer_id", "take", "session", "n_frames",
    "trim_start", "trim_end", "fps", "duration_s", "width", "height", "pose_rate", "face_rate", "lh_rate",
    "rh_rate", "path", "video_path", "status", "recorded_at"]
SIGNER_FIELDS = ["signer_id", "dominant_hand", "experience", "hearing_status", "consent", "consent_date", "notes"]
CLIP_TYPES = ("sign", "sentence")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


class ISLDataset:
    def __init__(self, root: str | Path = "dataset"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.csv"
        self.signers_path = self.root / "signers.csv"
        self.rows: list[dict] = self._read(self.manifest_path)
        self.signer_rows: list[dict] = self._read(self.signers_path)

    # -- csv helpers ---------------------------------------------------------
    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)  # atomic: a crash can never leave a half-written manifest

    # -- signers -------------------------------------------------------------
    def register_signer(self, signer_id: str, dominant_hand: str = "right", experience: str = "",
                        hearing_status: str = "", consent: bool = False, notes: str = "") -> dict:
        """Register (or update) a signer.  ``consent`` must be True before recording."""
        if dominant_hand not in ("left", "right"):
            raise ValueError("dominant_hand must be 'left' or 'right'")
        row = {"signer_id": signer_id, "dominant_hand": dominant_hand, "experience": experience,
               "hearing_status": hearing_status, "consent": "yes" if consent else "no",
               "consent_date": datetime.now().date().isoformat() if consent else "", "notes": notes}
        self.signer_rows = [r for r in self.signer_rows if r["signer_id"] != signer_id] + [row]
        self._write(self.signers_path, SIGNER_FIELDS, self.signer_rows)
        return row

    def signer(self, signer_id: str) -> dict | None:
        return next((r for r in self.signer_rows if r["signer_id"] == signer_id), None)

    def signers(self) -> list[str]:
        return sorted({r["signer_id"] for r in self.rows if r["status"] == "ok"})

    # -- queries -------------------------------------------------------------
    def records(self, status: str | None = "ok", clip_type: str | None = None,
                signer_id: str | None = None, label_id: str | None = None) -> list[dict]:
        return [r for r in self.rows
                if (status is None or r["status"] == status)
                and (clip_type is None or r["clip_type"] == clip_type)
                and (signer_id is None or r["signer_id"] == signer_id)
                and (label_id is None or r["label_id"] == label_id)]

    def count(self, signer_id: str, clip_type: str, label_id: str) -> int:
        return len(self.records("ok", clip_type, signer_id, label_id))

    def _next_take(self, signer_id: str, clip_type: str, label_id: str) -> int:
        takes = [int(r["take"]) for r in self.records(None, clip_type, signer_id, label_id)]
        return max(takes, default=0) + 1

    # -- writing -------------------------------------------------------------
    def add_clip(self, raw: RawSequence, clip_type: str, item: Sign | Sentence, signer_id: str,
                 session: str = "", trim: tuple[int, int] | None = None,
                 video_src: str | Path | None = None) -> dict:
        """Store one clip.  If ``video_src`` is given the video file is *moved* into videos/."""
        if clip_type not in CLIP_TYPES:
            raise ValueError(f"clip_type must be one of {CLIP_TYPES}")
        label = item.gloss if isinstance(item, Sign) else item.text
        gloss = item.gloss
        take = self._next_take(signer_id, clip_type, item.id)
        clip_id = f"{signer_id}_{clip_type}_{_slug(item.id)}_{take:02d}"
        rel = Path("clips") / signer_id / clip_type / item.id / f"{clip_id}.npz"
        trim = trim or (0, len(raw))
        raw.meta.update({"trim": [int(trim[0]), int(trim[1])], "clip_id": clip_id, "signer_id": signer_id,
                         "clip_type": clip_type, "label_id": item.id})
        raw.save(self.root / rel)
        video_path = ""
        if video_src and Path(video_src).exists():
            vrel = Path("videos") / signer_id / clip_type / item.id / f"{clip_id}{Path(video_src).suffix}"
            (self.root / vrel).parent.mkdir(parents=True, exist_ok=True)
            Path(video_src).replace(self.root / vrel)
            video_path = vrel.as_posix()
        rates = raw.slice(*trim).detection_rates()
        row = {"clip_id": clip_id, "clip_type": clip_type, "label_id": item.id, "label": label,
               "gloss": gloss, "signer_id": signer_id, "take": take, "session": session,
               "n_frames": len(raw), "trim_start": trim[0], "trim_end": trim[1],
               "fps": round(raw.fps, 2), "duration_s": round(float(raw.timestamps[-1]) if len(raw) else 0, 2),
               "width": raw.meta.get("width", ""), "height": raw.meta.get("height", ""),
               "pose_rate": round(rates["pose"], 3), "face_rate": round(rates["face"], 3),
               "lh_rate": round(rates["lh"], 3), "rh_rate": round(rates["rh"], 3),
               "path": rel.as_posix(), "video_path": video_path, "status": "ok",
               "recorded_at": datetime.now().isoformat(timespec="seconds")}
        self.rows.append(row)
        self._write(self.manifest_path, MANIFEST_FIELDS, self.rows)
        return row

    def set_status(self, clip_id: str, status: str) -> None:
        """Mark a clip 'ok' / 'rejected' (files are kept; the clip is just excluded)."""
        for r in self.rows:
            if r["clip_id"] == clip_id:
                r["status"] = status
        self._write(self.manifest_path, MANIFEST_FIELDS, self.rows)

    # -- reading -------------------------------------------------------------
    def load_raw(self, record: dict, trimmed: bool = True) -> RawSequence:
        raw = RawSequence.load(self.root / record["path"])
        return raw.trimmed() if trimmed else raw

    def dominant_hand(self, signer_id: str) -> str:
        s = self.signer(signer_id)
        return s["dominant_hand"] if s else "right"

    # -- reporting -----------------------------------------------------------
    def coverage(self, signs: list[Sign], sentences: list[Sentence], reps_sign: int, reps_sentence: int) -> dict:
        """Per-signer progress and which items are still short of the target repetitions."""
        out = {}
        for sid in sorted({r["signer_id"] for r in self.rows} | {r["signer_id"] for r in self.signer_rows}):
            short_signs = [s.gloss for s in signs if self.count(sid, "sign", s.id) < reps_sign]
            short_sents = [s.id for s in sentences if self.count(sid, "sentence", s.id) < reps_sentence]
            out[sid] = {"sign_clips": len(self.records("ok", "sign", sid)),
                        "sentence_clips": len(self.records("ok", "sentence", sid)),
                        "signs_below_target": short_signs, "sentences_below_target": short_sents}
        return out


# --------------------------------------------------------------------------
# Signer-independent splits
# --------------------------------------------------------------------------
def signer_independent_split(signer_ids: list[str], n_val: int = 1, n_test: int = 1, seed: int = 0) -> dict[str, str]:
    """Assign WHOLE signers to train / val / test (never split one signer's clips).

    Evaluating on unseen signers is the honest test of signer-adaptive recognition.
    With fewer than 3 signers val is dropped; with fewer than 2 everything is train
    (and you should not report accuracy from it)."""
    ids = sorted(set(signer_ids))
    random.Random(seed).shuffle(ids)
    if len(ids) < 2:
        return {i: "train" for i in ids}
    if len(ids) < 3:
        n_val = 0
    n_test = min(n_test, len(ids) - 1)
    n_val = min(n_val, len(ids) - n_test - 1)
    split = {}
    for k, sid in enumerate(ids):
        split[sid] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return split


# --------------------------------------------------------------------------
# Feature export for training
# --------------------------------------------------------------------------
def build_arrays(ds: ISLDataset, clip_type: str, signs: list[Sign], spec: FeatureSpec | None = None,
                 fixed_len: int | None = 64, signer_standardise: bool = True, max_gap: int = 5,
                 sentences: list[Sentence] | None = None) -> dict:
    """Normalise every OK clip of ``clip_type`` and stack into arrays.

    fixed_len=N  -> X is (N_clips, N, F): every clip resampled to N frames (isolated signs).
    fixed_len=None -> X is zero-padded to the longest clip, with ``lengths`` (sentences).
    Sentence targets are gloss-index sequences (0 = blank/pad) ready for CTC training.
    """
    spec = spec or FeatureSpec()
    g2i = gloss_to_index(signs)
    sent_by_id = {s.id: s for s in (sentences or [])}
    clips, skipped = [], []
    for rec in ds.records("ok", clip_type):
        try:
            ns = normalize_sequence(ds.load_raw(rec), spec, dominant=ds.dominant_hand(rec["signer_id"]),
                                    max_gap=max_gap)
        except NormalizationError as exc:
            skipped.append((rec["clip_id"], str(exc)))
            continue
        clips.append((rec, ns))

    if signer_standardise:  # per-signer statistics from that signer's own clips
        stats: dict[str, SignerStats] = {}
        for rec, ns in clips:
            stats.setdefault(rec["signer_id"], SignerStats(spec)).update(ns.features, ns.valid)
        feats = [stats[rec["signer_id"]].transform(ns.features, ns.valid) for rec, ns in clips]
    else:
        stats, feats = {}, [ns.features for _, ns in clips]

    lengths = np.array([len(f) for f in feats], np.int32)
    if fixed_len:
        X = np.stack([resample(f, fixed_len) for f in feats]).astype(np.float32) if feats else \
            np.zeros((0, fixed_len, spec.dim), np.float32)
    else:
        L = int(lengths.max()) if len(lengths) else 0
        X = np.zeros((len(feats), L, spec.dim), np.float32)
        for i, f in enumerate(feats):
            X[i, :len(f)] = f

    out = {"X": X, "lengths": lengths,
           "clip_id": np.array([r["clip_id"] for r, _ in clips]),
           "signer_id": np.array([r["signer_id"] for r, _ in clips]),
           "label_id": np.array([r["label_id"] for r, _ in clips]),
           "skipped": skipped, "stats": stats}
    if clip_type == "sign":
        out["y"] = np.array([g2i[r["gloss"]] for r, _ in clips], np.int32)
    else:
        seqs = [[g2i[g] for g in (sent_by_id[r["label_id"]].glosses if r["label_id"] in sent_by_id
                                  else r["gloss"].split())] for r, _ in clips]
        width = max((len(s) for s in seqs), default=0)
        out["y_seq"] = np.array([s + [0] * (width - len(s)) for s in seqs], np.int32).reshape(len(seqs), width)
        out["y_len"] = np.array([len(s) for s in seqs], np.int32)
    return out
