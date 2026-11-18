#!/usr/bin/env python
"""Dataset health report: size, per-signer coverage, detection rates, flagged clips."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter

import numpy as np

from signvoice.dataset import ISLDataset
from signvoice.vocabulary import load_sentences, load_signs, validate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--reps-sign", type=int, default=5)
    ap.add_argument("--reps-sentence", type=int, default=3)
    args = ap.parse_args()
    signs, sentences = load_signs(), load_sentences()
    print(f"vocabulary: {len(signs)} signs, {len(sentences)} sentences; problems: {validate(signs, sentences) or 'none'}")
    ds = ISLDataset(args.dataset)
    ok = ds.records("ok")
    print(f"clips: {len(ok)} ok, {len(ds.records('rejected'))} rejected, {len(ds.signers())} signer(s)")
    for t in ("sign", "sentence"):
        rows = ds.records("ok", t)
        if not rows:
            continue
        dur = np.array([float(r["duration_s"]) for r in rows])
        print(f"{t}: {len(rows)} clips, {len(set(r['label_id'] for r in rows))} distinct labels, "
              f"mean {dur.mean():.1f}s (min {dur.min():.1f}, max {dur.max():.1f})")
    for sid, cov in ds.coverage(signs, sentences, args.reps_sign, args.reps_sentence).items():
        print(f"\n{sid}: {cov['sign_clips']} sign clips | {cov['sentence_clips']} sentence clips")
        print(f"   signs short of {args.reps_sign} reps: {len(cov['signs_below_target'])} "
              f"{cov['signs_below_target'][:8]}{' ...' if len(cov['signs_below_target']) > 8 else ''}")
        print(f"   sentences short of {args.reps_sentence} reps: {len(cov['sentences_below_target'])}")
    flagged = [r for r in ok if float(r["pose_rate"]) < 0.9 or max(float(r["lh_rate"]), float(r["rh_rate"])) < 0.3]
    print(f"\nclips with weak detection (pose<90% or hands<30%): {len(flagged)}")
    for r in flagged[:15]:
        print(f"   {r['clip_id']}  pose {r['pose_rate']}  lh {r['lh_rate']}  rh {r['rh_rate']}")
    print("\nclips per signer:", dict(Counter(r["signer_id"] for r in ok)))


if __name__ == "__main__":
    main()
