#!/usr/bin/env python
"""Normalise every OK clip and write training arrays + per-signer stats + signer-independent splits.

  python scripts/build_features.py                       # defaults
  python scripts/build_features.py --sign-len 48 --no-face
Outputs in dataset/processed/:  signs.npz, sentences.npz, splits.json, signer_stats/<signer>.npz
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

from signvoice.config import FeatureSpec
from signvoice.dataset import ISLDataset, build_arrays, signer_independent_split
from signvoice.vocabulary import gloss_to_index, load_sentences, load_signs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--sign-len", type=int, default=64, help="frames per isolated sign after resampling")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-hand-shape", action="store_true")
    ap.add_argument("--no-signer-standardise", action="store_true")
    ap.add_argument("--val-signers", type=int, default=1)
    ap.add_argument("--test-signers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = ISLDataset(args.dataset)
    signs, sentences = load_signs(), load_sentences()
    spec = FeatureSpec(face=not args.no_face, hands_shape=not args.no_hand_shape)
    out = ds.root / "processed"
    (out / "signer_stats").mkdir(parents=True, exist_ok=True)

    built = {}
    for clip_type, name, length in (("sign", "signs", args.sign_len), ("sentence", "sentences", None)):
        a = build_arrays(ds, clip_type, signs, spec, fixed_len=length,
                         signer_standardise=not args.no_signer_standardise, sentences=sentences)
        for sid, st in a.pop("stats").items():
            st.save(out / "signer_stats" / f"{sid}.npz")
        built[name] = (a, a.pop("skipped"))

    # split only over signers that actually contribute usable clips
    usable = sorted({str(s) for a, _ in built.values() for s in a["signer_id"]})
    split = signer_independent_split(usable, args.val_signers, args.test_signers, args.seed)
    (out / "splits.json").write_text(json.dumps(split, indent=2))
    (out / "labels.json").write_text(json.dumps(gloss_to_index(signs), indent=2))
    print("signer split:", split)
    if len(split) < 3:
        print("WARNING: fewer than 3 usable signers - there is no meaningful signer-independent test set yet.")

    for name, (a, skipped) in built.items():
        a["split"] = np.array([split.get(s, "train") for s in a["signer_id"]])
        np.savez_compressed(out / f"{name}.npz", **a)
        print(f"{name}: X{a['X'].shape} from {len(a['clip_id'])} clips ({len(skipped)} skipped)")
        for cid, why in skipped:
            print(f"   skipped {cid}: {why}")
    print("feature groups:", {k: (v.start, v.stop) for k, v in spec.slices().items()})


if __name__ == "__main__":
    main()
