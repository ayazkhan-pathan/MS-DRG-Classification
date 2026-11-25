"""Score notes with a fine-tuned model: evaluate against known DRGs, or predict the top DRGs.

Column names, the token limit and the truncation side default to the values saved in the
model's ``training_config.json``; each can be overridden.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from drg.data import LabelMap, NoteDataset, TextCollator, read_notes
from drg.metrics import classification_metrics
from drg.train import TRAINING_CONFIG_FILE

log = logging.getLogger(__name__)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DRGModel:
    """A fine-tuned model with its tokenizer, label map and training settings."""

    model: torch.nn.Module
    tokenizer: object
    label_map: LabelMap
    settings: dict
    device: torch.device

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "auto") -> DRGModel:
        model_dir = Path(model_dir)
        settings_path = model_dir / TRAINING_CONFIG_FILE
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        label_map = LabelMap.load(model_dir)
        torch_device = resolve_device(device)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(torch_device)
        model.eval()
        if model.config.num_labels != len(label_map):
            raise ValueError(
                f"{model_dir}: the model has {model.config.num_labels} outputs but the label "
                f"map holds {len(label_map)} DRGs"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return cls(model, tokenizer, label_map, settings, torch_device)

    def scores(
        self,
        texts,
        max_length: int | None = None,
        truncation_side: str | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Logits, one row per note and one column per DRG in label-map order."""
        self.tokenizer.truncation_side = truncation_side or self.settings.get(
            "truncation_side", "right"
        )
        collate = TextCollator(self.tokenizer, max_length or self.settings.get("max_length", 512))
        loader = torch.utils.data.DataLoader(
            NoteDataset(texts), batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        out = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)


def _columns(settings: dict, text_column, label_column) -> tuple[str, str]:
    return (
        text_column or settings.get("text_column", "text"),
        label_column or settings.get("label_column", "label"),
    )


def evaluate(
    model_dir: str | Path,
    csv: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    max_length: int | None = None,
    truncation_side: str | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> dict:
    """Top-1, top-5, top-10 and macro-F1 on a labelled CSV.

    Notes whose DRG is not in the model's label map are kept and count as errors; each such
    DRG is its own class in macro-F1.
    """
    drg = DRGModel.load(model_dir, device)
    text_col, label_col = _columns(drg.settings, text_column, label_column)
    notes = read_notes(csv, text_column=text_col, label_column=label_col)

    known = drg.label_map.label2id
    unseen = sorted(set(notes["label"]) - set(known))
    extra = {label: len(known) + i for i, label in enumerate(unseen)}
    labels = np.array([known.get(label, extra.get(label)) for label in notes["label"]])

    scores = drg.scores(notes["text"], max_length, truncation_side, batch_size)
    metrics = classification_metrics(scores, labels)
    unseen_notes = int(notes["label"].isin(unseen).sum())
    if unseen_notes:
        log.warning("%d notes have a DRG the model never saw in training", unseen_notes)
    return {"notes": len(notes), "unseen_drg_notes": unseen_notes, **metrics}


def predict(
    model_dir: str | Path,
    csv: str | Path,
    top_k: int = 5,
    text_column: str | None = None,
    id_column: str | None = None,
    max_length: int | None = None,
    truncation_side: str | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> pd.DataFrame:
    """The ``top_k`` most likely DRGs for each note, with softmax probabilities."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    drg = DRGModel.load(model_dir, device)
    text_col, _ = _columns(drg.settings, text_column, None)
    notes = read_notes(csv, text_column=text_col, label_column=None, id_column=id_column)

    scores = drg.scores(notes["text"], max_length, truncation_side, batch_size)
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    k = min(top_k, probs.shape[1])
    order = np.argsort(-probs, axis=1, kind="stable")[:, :k]

    table = pd.DataFrame({id_column: notes["id"]}) if id_column else pd.DataFrame(index=notes.index)
    labels = np.array(drg.label_map.labels)
    for rank in range(k):
        table[f"drg_{rank + 1}"] = labels[order[:, rank]]
        table[f"prob_{rank + 1}"] = np.round(probs[np.arange(len(probs)), order[:, rank]], 6)
    return table
