#!/usr/bin/env python
"""Register signers and record the pilot ISL dataset.

  python scripts/record_dataset.py register --signer S01 --dominant right --experience native \
         --hearing deaf --consent-confirmed
  python scripts/record_dataset.py record --signer S01 --session 2026-11-20-lab
  python scripts/record_dataset.py record --signer S01 --types sentence --reps-sentence 3
  python scripts/record_dataset.py status
"""
import _bootstrap  # noqa: F401
import argparse

from signvoice.dataset import ISLDataset
from signvoice.recorder import Recorder, build_tasks
from signvoice.vocabulary import load_sentences, load_signs, validate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset", help="dataset folder (default: ./dataset)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="add a signer (consent is mandatory before recording)")
    r.add_argument("--signer", required=True, help="anonymous id such as S01 (never a real name)")
    r.add_argument("--dominant", choices=["left", "right"], default="right")
    r.add_argument("--experience", default="", help="e.g. native / fluent / learner")
    r.add_argument("--hearing", default="", help="e.g. deaf / hard-of-hearing / hearing")
    r.add_argument("--notes", default="")
    r.add_argument("--consent-confirmed", action="store_true",
                   help="the signer has read and signed the consent form (docs/DATA_COLLECTION.md)")

    c = sub.add_parser("record", help="interactive recording session")
    c.add_argument("--signer", required=True)
    c.add_argument("--session", default="", help="free-text tag: date / place / lighting")
    c.add_argument("--types", nargs="+", choices=["sign", "sentence"], default=["sign", "sentence"])
    c.add_argument("--reps-sign", type=int, default=5)
    c.add_argument("--reps-sentence", type=int, default=3)
    c.add_argument("--only", nargs="*", help="only these ids / glosses, e.g. SGN001 HELLO SEN004")
    c.add_argument("--camera", default="0", help="camera index or video file")
    c.add_argument("--width", type=int, default=1280)
    c.add_argument("--height", type=int, default=720)
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--countdown", type=float, default=3.0)
    c.add_argument("--save-video", action="store_true", help="also keep the raw video (needs video consent)")
    c.add_argument("--no-trim", action="store_true", help="do not auto-trim idle frames")
    c.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    c.add_argument("--seed", type=int, default=0)

    sub.add_parser("status", help="progress per signer")
    args = ap.parse_args()

    signs, sentences = load_signs(), load_sentences()
    problems = validate(signs, sentences)
    if problems:
        raise SystemExit("vocabulary problems:\n  " + "\n  ".join(problems))
    ds = ISLDataset(args.dataset)

    if args.cmd == "register":
        if not args.consent_confirmed:
            raise SystemExit("Refusing to register: pass --consent-confirmed once the signer has given consent.")
        row = ds.register_signer(args.signer, args.dominant, args.experience, args.hearing, True, args.notes)
        print("registered:", row)
    elif args.cmd == "record":
        only = set(args.only) if args.only else None
        tasks = build_tasks(ds, args.signer, signs, sentences, tuple(args.types), args.reps_sign,
                            args.reps_sentence, only, seed=args.seed)
        if not tasks:
            raise SystemExit("Nothing left to record for this signer with these settings.")
        n_s = sum(t.clip_type == "sign" for t in tasks)
        print(f"{len(tasks)} clips to record ({n_s} signs, {len(tasks) - n_s} sentences). "
              "Stop any time with Q - progress is saved after every clip.")
        Recorder(ds, args.signer, tasks, args.camera, args.width, args.height, args.fps, args.countdown,
                 save_video=args.save_video, session=args.session, model_complexity=args.model_complexity,
                 auto_trim=not args.no_trim).run()
    elif args.cmd == "status":
        for sid, cov in ds.coverage(signs, sentences, 5, 3).items():
            print(f"{sid}: {cov['sign_clips']} sign clips, {cov['sentence_clips']} sentence clips | "
                  f"signs below 5 reps: {len(cov['signs_below_target'])}, "
                  f"sentences below 3 reps: {len(cov['sentences_below_target'])}")


if __name__ == "__main__":
    main()
