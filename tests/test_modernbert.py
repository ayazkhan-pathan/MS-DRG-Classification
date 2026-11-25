"""The long-context option: ModernBERT-base reading up to 8,192 tokens per note."""

import json

import pandas as pd
import pytest
from transformers import AutoConfig, AutoTokenizer, BertConfig, ModernBertConfig

from drg.cli import build_parser, main
from drg.data import LabelMap
from drg.predict import evaluate, predict
from drg.synthetic import LABELS
from drg.train import TrainConfig, build_training_arguments, check_max_length

MAX_LENGTH = 1024  # past BERT's 512 positions, and quick on a CPU
REQUIRED = ["train", "--train-csv", "a.csv", "--eval-csv", "b.csv", "--output-dir", "out"]


def test_cli_takes_the_long_context_setup():
    extra = ["--model", "answerdotai/ModernBERT-base", "--max-length", "8192"]
    args = vars(build_parser().parse_args([*REQUIRED, *extra, "--gradient-checkpointing"]))
    args.pop("command")
    config = TrainConfig(**args)
    assert (config.base_model, config.max_length) == ("answerdotai/ModernBERT-base", 8192)
    assert build_training_arguments(config).gradient_checkpointing


def test_gradient_checkpointing_is_off_by_default(tmp_path):
    args = build_training_arguments(TrainConfig("a.csv", "b.csv", str(tmp_path)))
    assert not args.gradient_checkpointing


def test_position_limit_allows_8192_tokens_for_modernbert_only():
    check_max_length(8192, ModernBertConfig(), "answerdotai/ModernBERT-base")
    with pytest.raises(ValueError, match="8192 positions"):
        check_max_length(8193, ModernBertConfig(), "answerdotai/ModernBERT-base")
    with pytest.raises(ValueError, match="512 positions"):
        check_max_length(8192, BertConfig(), "emilyalsentzer/Bio_ClinicalBERT")


@pytest.fixture(scope="module")
def long_notes(synthetic_dir, tmp_path_factory):
    """The synthetic splits with each note repeated until it runs past MAX_LENGTH tokens."""
    out = tmp_path_factory.mktemp("long-notes")
    for name in ("train", "val", "test"):
        notes = pd.read_csv(synthetic_dir / f"{name}.csv", dtype=str)
        notes["text"] = [" ".join([text] * 30) for text in notes["text"]]
        notes.to_csv(out / f"{name}.csv", index=False)
    return out


@pytest.fixture(scope="module")
def modernbert_run(tiny_modernbert_8k, long_notes, tmp_path_factory):
    """The tiny ModernBERT fine-tuned for one epoch on long notes, on the CPU."""
    out = tmp_path_factory.mktemp("modernbert-run")
    status = main(
        [
            "train",
            "--train-csv", str(long_notes / "train.csv"),
            "--eval-csv", str(long_notes / "val.csv"),
            "--model", str(tiny_modernbert_8k),
            "--max-length", str(MAX_LENGTH),
            "--output-dir", str(out),
            "--epochs", "1",
            "--batch-size", "8",
            "--learning-rate", "1e-3",
            "--logging-steps", "2",
            "--gradient-checkpointing",
            "--cpu",
        ]
    )  # fmt: skip
    assert status == 0
    return out


def test_notes_run_past_the_token_limit(tiny_modernbert_8k, long_notes):
    tokenizer = AutoTokenizer.from_pretrained(tiny_modernbert_8k)
    text = pd.read_csv(long_notes / "train.csv")["text"][0]
    assert len(tokenizer(text)["input_ids"]) > MAX_LENGTH


def test_modernbert_trains_on_1024_token_inputs(modernbert_run):
    assert AutoConfig.from_pretrained(modernbert_run).model_type == "modernbert"
    assert LabelMap.load(modernbert_run).labels == LABELS
    settings = json.loads((modernbert_run / "training_config.json").read_text())
    assert (settings["max_length"], settings["gradient_checkpointing"]) == (MAX_LENGTH, True)
    metrics = json.loads((modernbert_run / "eval_metrics.json").read_text())
    assert 0 <= metrics["eval_macro_f1"] <= 1


def test_modernbert_evaluates_and_predicts_at_the_trained_length(modernbert_run, long_notes):
    metrics = evaluate(modernbert_run, long_notes / "test.csv", device="cpu")
    assert metrics["notes"] == 16 and 0 <= metrics["top1_accuracy"] <= 1
    table = predict(modernbert_run, long_notes / "test.csv", top_k=2, device="cpu")
    assert list(table.columns) == ["drg_1", "prob_1", "drg_2", "prob_2"]
    assert set(table["drg_1"]) <= set(LABELS)
