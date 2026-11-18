#!/usr/bin/env python
"""Turn already-recorded videos (phone / camera files) into dataset clips.

Layout A (folder):   VIDEOS/<signer>/<sign|sentence>/<label_id>/anything.mp4
                     e.g. videos_in/S02/sign/SGN001/take1.mp4,  videos_in/S02/sentence/SEN004/a.mp4
Layout B (index):    --index index.csv with columns  video,clip_type,label_id,signer_id

  python scripts/extract_from_videos.py --videos videos_in
  python scripts/extract_from_videos.py --index index.csv
Signers must already be registered (with consent) via record_dataset.py register.
"""
import _bootstrap  # noqa: F401
import argparse
import csv
from pathlib import Path

from signvoice.capture import process_video_file, quality_issues
from signvoice.dataset import ISLDataset
from signvoice.normalize import find_active_span
from signvoice.vocabulary import load_sentences, load_signs

EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def jobs_from_folder(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXT:
            rel = p.relative_to(root).parts
            if len(rel) < 4:
                print(f"skip (expected signer/type/label_id/file): {p}")
                continue
            yield p, rel[1], rel[2], rel[0]


def jobs_from_index(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield Path(r["video"]), r["clip_type"], r["label_id"], r["signer_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos", type=Path)
    g.add_argument("--index", type=Path)
    ap.add_argument("--session", default="imported")
    ap.add_argument("--keep-video", action="store_true", help="copy the source video into the dataset")
    ap.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    ap.add_argument("--no-trim", action="store_true")
    args = ap.parse_args()

    ds = ISLDataset(args.dataset)
    items = {("sign", s.id): s for s in load_signs()} | {("sentence", s.id): s for s in load_sentences()}
    jobs = list(jobs_from_folder(args.videos) if args.videos else jobs_from_index(args.index))
    ok = bad = 0
    for video, clip_type, label_id, signer_id in jobs:
        signer = ds.signer(signer_id)
        if not signer or signer["consent"] != "yes":
            print(f"skip {video}: signer {signer_id} not registered with consent")
            bad += 1
            continue
        item = items.get((clip_type, label_id))
        if item is None:
            print(f"skip {video}: unknown {clip_type} id {label_id}")
            bad += 1
            continue
        raw = process_video_file(video, args.model_complexity)
        span = (0, len(raw)) if args.no_trim else find_active_span(raw)
        issues = quality_issues(raw, span)
        row = ds.add_clip(raw, clip_type, item, signer_id, args.session, trim=span)
        if args.keep_video:
            import shutil
            dest = ds.root / "videos" / signer_id / clip_type / label_id / f"{row['clip_id']}{video.suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, dest)
        if issues:  # keep the clip but flag it for review rather than silently dropping data
            print(f"{row['clip_id']}: saved with warnings -> " + "; ".join(issues))
        else:
            print(f"{row['clip_id']}: ok ({len(raw)} frames)")
        ok += 1
    print(f"done: {ok} imported, {bad} skipped")


if __name__ == "__main__":
    main()
