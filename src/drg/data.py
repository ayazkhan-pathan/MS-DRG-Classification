"""Read note CSVs, map DRG codes to class ids and batch note text for a model.

A note CSV has one row per admission, with a column of note text and, for training and
evaluation, a column holding the DRG code. DRG codes are kept as strings, so "003" stays
"003", and any set of codes works: the label map is built from the training labels and
saved next to the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

LABEL_MAP_FILE = "label_map.json"


def read_notes(
    path: str | Path,
    text_column: str = "text",
    label_column: str | None = "label",
    id_column: str | None = None,
) -> pd.DataFrame:
    """Read a note CSV into a frame with columns ``text``, and ``label`` and ``id`` if asked.

    Every column is read as a string. Pass ``label_column=None`` for notes without labels.
    """
    wanted = {"text": text_column}
    if label_column is not None:
        wanted["label"] = label_column
    if id_column is not None:
        wanted["id"] = id_column

    header = pd.read_csv(path, nrows=0).columns
    missing = [col for col in wanted.values() if col not in header]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {', '.join(missing)}; "
            f"the file has {', '.join(map(str, header))}"
        )

    frame = pd.read_csv(path, usecols=list(wanted.values()), dtype=str, keep_default_na=False)
    frame = frame.rename(columns={col: name for name, col in wanted.items()})
    frame = frame[list(wanted)]
    if "label" in frame:
        frame["label"] = frame["label"].str.strip()
        _check_not_empty(frame["label"], path, label_column)
    _check_not_empty(frame["text"].str.strip(), path, text_column)
    return frame.reset_index(drop=True)


def _check_not_empty(values: pd.Series, path, column: str) -> None:
    empty = values.eq("")
    if empty.any():
        rows = ", ".join(str(i + 1) for i in values.index[empty][:5])
        raise ValueError(
            f"{path}: {int(empty.sum())} empty value(s) in {column!r} (data rows {rows})"
        )


def subsample(frame: pd.DataFrame, size: float | None, seed: int = 42) -> pd.DataFrame:
    """Draw a random subset of rows, kept in file order.

    ``size`` below 1 is a fraction of the rows; 1 or more is a number of rows. ``None``
    keeps everything.
    """
    if size is None:
        return frame
    if size <= 0:
        raise ValueError(f"subsample must be positive, got {size}")
    if size < 1:
        n = max(1, round(len(frame) * size))
    elif float(size).is_integer():
        n = min(int(size), len(frame))
    else:
        raise ValueError(
            f"subsample must be a fraction below 1 or a whole number of rows, got {size}"
        )
    return frame.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


@dataclass(frozen=True)
class LabelMap:
    """The DRG codes a model predicts, in class-id order."""

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("a label map needs at least one label")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels in a label map must be unique")

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> LabelMap:
        return cls(tuple(sorted({str(label) for label in labels})))

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        return label in self.label2id

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    def encode(self, labels: Iterable[str]) -> list[int]:
        """Class ids for DRG codes; raises ``KeyError`` for a code the map does not hold."""
        lookup = self.label2id
        ids = []
        for label in labels:
            if label not in lookup:
                raise KeyError(f"DRG {label!r} is not in the label map")
            ids.append(lookup[label])
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.labels[int(i)] for i in ids]

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / LABEL_MAP_FILE
        path.write_text(json.dumps({"labels": list(self.labels)}, indent=1) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> LabelMap:
        path = Path(directory) / LABEL_MAP_FILE
        return cls(tuple(json.loads(path.read_text())["labels"]))


class NoteDataset(torch.utils.data.Dataset):
    """Note text with optional class ids; tokenisation happens per batch in the collator."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None = None):
        if labels is not None and len(labels) != len(texts):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = None if labels is None else [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


@dataclass
class TextCollator:
    """Tokenise a batch of notes, truncate each to ``max_length`` tokens and pad the batch."""

    tokenizer: object
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer(
            [f["text"] for f in features],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = dict(batch)
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return batch
