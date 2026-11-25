"""Obviously fake notes for tests and smoke runs.

Each note is a numbered "synthetic record" made of invented words in three parts, labelled
S01 to S04. Every label has four marker words, and notes mix them with filler words and the
odd marker from another label. A real encoder could learn the markers, but the tiny random
models used in the tests and the smoke run stay at chance: they only show that the code runs
end to end. Nothing here resembles a patient record, and scores on these notes say nothing
about DRG prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("S01", "S02", "S03", "S04")
MARKERS = {
    "S01": ("amber", "birch", "copper", "dune"),
    "S02": ("ember", "fjord", "garnet", "heath"),
    "S03": ("indigo", "juniper", "kestrel", "lagoon"),
    "S04": ("marble", "nectar", "oriole", "pebble"),
}
FILLER = (
    "quill",
    "tally",
    "vessel",
    "widget",
    "yarrow",
    "zephyr",
    "lantern",
    "meadow",
    "parcel",
    "ripple",
    "saddle",
    "thimble",
)
PARTS = ("part a", "part b", "part c")


def make_note(record: int, label: str, rng: np.random.Generator) -> str:
    """One fake note, for example "synthetic record 12: part a: amber quill birch ..."."""
    others = [other for other in LABELS if other != label]
    sections = [f"synthetic record {record}:"]
    for part in PARTS:
        words = list(rng.choice(MARKERS[label], size=2))
        words += list(rng.choice(FILLER, size=int(rng.integers(4, 9))))
        if rng.random() < 0.2:
            words.append(str(rng.choice(MARKERS[str(rng.choice(others))])))
        rng.shuffle(words)
        sections.append(f"{part}: {' '.join(words)}.")
    return " ".join(sections)


def make_notes(n: int, seed: int = 0, first_record: int = 1) -> pd.DataFrame:
    """``n`` fake notes with balanced labels, as columns ``note_id``, ``text`` and ``label``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    labels = [LABELS[i % len(LABELS)] for i in range(n)]
    rng.shuffle(labels)
    records = range(first_record, first_record + n)
    return pd.DataFrame(
        {
            "note_id": [f"synthetic-{r}" for r in records],
            "text": [make_note(r, label, rng) for r, label in zip(records, labels)],
            "label": labels,
        }
    )


def write_splits(
    out_dir: str | Path,
    n_train: int = 200,
    n_val: int = 60,
    n_test: int = 60,
    seed: int = 0,
) -> dict[str, Path]:
    """Write ``train.csv``, ``val.csv`` and ``test.csv`` with distinct record numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    first = 1
    for offset, (name, n) in enumerate((("train", n_train), ("val", n_val), ("test", n_test))):
        path = out_dir / f"{name}.csv"
        make_notes(n, seed=seed + offset, first_record=first).to_csv(path, index=False)
        paths[name] = path
        first += n
    return paths
