"""Loads and validates the pilot vocabulary (data/vocab/*.csv)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent.parent / "data" / "vocab"


@dataclass(frozen=True)
class Sign:
    id: str
    gloss: str
    category: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    gloss: str          # space-separated sign glosses, in signing order
    notes: str = ""

    @property
    def glosses(self) -> list[str]:
        return self.gloss.split()


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_signs(path: Path | str | None = None) -> list[Sign]:
    rows = _read(Path(path) if path else VOCAB_DIR / "signs.csv")
    return [Sign(r["id"], r["gloss"], r.get("category", ""), r.get("notes", "")) for r in rows]


def load_sentences(path: Path | str | None = None) -> list[Sentence]:
    rows = _read(Path(path) if path else VOCAB_DIR / "sentences.csv")
    return [Sentence(r["id"], r["text"], r["gloss"], r.get("notes", "")) for r in rows]


def validate(signs: list[Sign], sentences: list[Sentence]) -> list[str]:
    """Return a list of problems (empty list = vocabulary is consistent)."""
    problems = []
    ids = [s.id for s in signs] + [s.id for s in sentences]
    problems += [f"duplicate id: {i}" for i in {i for i in ids if ids.count(i) > 1}]
    glosses = [s.gloss for s in signs]
    problems += [f"duplicate gloss: {g}" for g in {g for g in glosses if glosses.count(g) > 1}]
    known = set(glosses)
    for s in sentences:
        problems += [f"{s.id}: gloss '{g}' is not in signs.csv" for g in s.glosses if g not in known]
    return problems


def gloss_to_index(signs: list[Sign]) -> dict[str, int]:
    """0 is reserved for the CTC blank / padding, so signs start at 1."""
    return {s.gloss: i + 1 for i, s in enumerate(signs)}
