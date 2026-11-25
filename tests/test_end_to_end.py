import json

import numpy as np
import pandas as pd
import pytest
from transformers import AutoConfig

from drg.cli import main
from drg.data import LabelMap
from drg.predict import evaluate, predict
from drg.synthetic import LABELS

METRIC_KEYS = {"top1_accuracy", "top5_accuracy", "top10_accuracy", "macro_f1"}


def test_train_saves_model_label_map_and_settings(trained_model):
    for name in ("config.json", "label_map.json", "training_config.json", "eval_metrics.json"):
        assert (trained_model / name).exists(), name
    assert LabelMap.load(trained_model).labels == LABELS
    config = AutoConfig.from_pretrained(trained_model)
    assert config.id2label == dict(enumerate(LABELS))
    settings = json.loads((trained_model / "training_config.json").read_text())
    assert (settings["max_length"], settings["num_labels"], settings["wandb"]) == (64, 4, False)
    metrics = json.loads((trained_model / "eval_metrics.json").read_text())
    assert 0 <= metrics["eval_macro_f1"] <= 1


def test_checkpoints_were_ranked_by_macro_f1(trained_model):
    checkpoints = sorted(
        trained_model.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    assert checkpoints
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
    scores = [entry["eval_macro_f1"] for entry in state["log_history"] if "eval_macro_f1" in entry]
    assert len(scores) == 2  # one evaluation per epoch
    assert state["best_metric"] == pytest.approx(max(scores))


def test_evaluate_command_writes_metrics(trained_model, synthetic_dir, tmp_path):
    out = tmp_path / "metrics.json"
    args = ["--model-dir", str(trained_model), "--csv", str(synthetic_dir / "test.csv")]
    assert main(["evaluate", *args, "--output", str(out), "--device", "cpu"]) == 0
    metrics = json.loads(out.read_text())
    assert set(metrics) >= METRIC_KEYS
    assert metrics["notes"] == 16 and metrics["unseen_drg_notes"] == 0
    assert all(0 <= metrics[key] <= 1 for key in METRIC_KEYS)


def test_predict_command_writes_ranked_drgs(trained_model, synthetic_dir, tmp_path):
    out = tmp_path / "predictions.csv"
    args = ["--model-dir", str(trained_model), "--csv", str(synthetic_dir / "test.csv")]
    assert main(["predict", *args, "--output", str(out), "--top-k", "3", "--device", "cpu"]) == 0
    table = pd.read_csv(out, dtype=str)
    assert list(table.columns) == ["drg_1", "prob_1", "drg_2", "prob_2", "drg_3", "prob_3"]
    assert len(table) == 16
    assert set(table[["drg_1", "drg_2", "drg_3"]].to_numpy().ravel()) <= set(LABELS)
    probs = table[["prob_1", "prob_2", "prob_3"]].astype(float).to_numpy()
    assert np.all(np.diff(probs, axis=1) <= 0)
    assert np.all(probs.sum(axis=1) <= 1 + 1e-6)


def test_predict_without_labels_keeps_ids(trained_model, synthetic_dir, tmp_path):
    notes = pd.read_csv(synthetic_dir / "test.csv").drop(columns="label")
    notes.to_csv(tmp_path / "unlabelled.csv", index=False)
    table = predict(
        trained_model, tmp_path / "unlabelled.csv", top_k=10, id_column="note_id", device="cpu"
    )
    assert table["note_id"].tolist() == notes["note_id"].tolist()
    assert "drg_4" in table and "drg_5" not in table  # only four DRGs exist


def test_unseen_drg_counts_as_an_error(trained_model, synthetic_dir, tmp_path):
    notes = pd.read_csv(synthetic_dir / "test.csv", dtype=str)
    notes.loc[0, "label"] = "S99"
    notes.to_csv(tmp_path / "unseen.csv", index=False)
    metrics = evaluate(trained_model, tmp_path / "unseen.csv", device="cpu")
    assert metrics["unseen_drg_notes"] == 1
    top1 = predict(trained_model, tmp_path / "unseen.csv", top_k=1, device="cpu")["drg_1"]
    assert metrics["top1_accuracy"] == pytest.approx(np.mean(top1.to_numpy() == notes["label"]))
