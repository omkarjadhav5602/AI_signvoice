#!/usr/bin/env python
"""Cross-check data/vocab/signs.csv against the published INCLUDE ISL lexicon (AI4Bharat / IIT Madras).

INCLUDE = 263 isolated ISL word signs in 15 categories, recorded with experienced signers
(CC-BY-4.0).  Labels come from Hugging Face (ai4bharat/INCLUDE); videos are on Zenodo record 4010759.

  pip install datasets
  python scripts/check_vocab_against_include.py            # report only
  python scripts/check_vocab_against_include.py --write    # also add "INCLUDE: <category>/<label>" to signs.csv notes

Needs internet.  Signs that match can be compared with INCLUDE's reference videos by your ISL expert;
signs that do NOT match must be verified by the expert some other way (ISLRTC dictionary, a signer).
INCLUDE has no verbs / question words / sentences, so many of your signs are expected to be unmatched.
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import re

from signvoice.vocabulary import VOCAB_DIR, load_signs

SYNONYMS = {"HE-SHE": ["he", "she"], "TOILET": ["toilet", "washroom"]}


def norm(label: str) -> str:
    """'55. Thank you' -> 'thank you';  'THANK-YOU' -> 'thank you'."""
    label = re.sub(r"^\s*\d+\.\s*", "", label)
    return re.sub(r"[-_\s]+", " ", label).strip().lower()


def load_include_labels() -> dict[str, tuple[str, str]]:
    from datasets import load_dataset  # imported lazily so the module can be tested offline
    out: dict[str, tuple[str, str]] = {}
    for split in load_dataset("ai4bharat/INCLUDE").values():
        for parent, label in zip(split["parent_label"], split["label"]):
            out.setdefault(norm(label), (parent, re.sub(r"^\s*\d+\.\s*", "", label)))
    return out


def match(signs, include: dict[str, tuple[str, str]]):
    """-> (matched {gloss: (category, label)}, unmatched [gloss])"""
    matched, unmatched = {}, []
    for s in signs:
        for cand in SYNONYMS.get(s.gloss, [norm(s.gloss)]):
            if norm(cand) in include:
                matched[s.gloss] = include[norm(cand)]
                break
        else:
            unmatched.append(s.gloss)
    return matched, unmatched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="annotate signs.csv notes and save include_labels.csv")
    args = ap.parse_args()

    include = load_include_labels()
    signs = load_signs()
    matched, unmatched = match(signs, include)
    print(f"INCLUDE labels: {len(include)} | your signs: {len(signs)} | "
          f"found in INCLUDE: {len(matched)} | NOT in INCLUDE: {len(unmatched)}")
    for g, (cat, lab) in matched.items():
        print(f"  match   {g:<12} -> INCLUDE {cat}/{lab}")
    print("  not in INCLUDE (expert must verify):", ", ".join(unmatched))

    if args.write:
        out_labels = VOCAB_DIR / "include_labels.csv"
        with open(out_labels, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "label"])
            w.writerows(sorted(include.values()))
        path = VOCAB_DIR / "signs.csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            note = f"INCLUDE: {matched[r['gloss']][0]}/{matched[r['gloss']][1]}" if r["gloss"] in matched else ""
            keep = [p for p in r["notes"].split(" | ") if p and not p.startswith("INCLUDE:")]
            r["notes"] = " | ".join(keep + ([note] if note else []))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "gloss", "category", "notes"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_labels.name} and updated notes in {path.name}")


if __name__ == "__main__":
    main()
