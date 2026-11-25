from pathlib import Path

import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from drg.cli import main
from drg.synthetic import write_splits

TINY_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"
TINY_MODERNBERT = "hf-internal-testing/tiny-random-ModernBertForSequenceClassification"


@pytest.fixture(scope="session")
def tiny_model() -> str:
    """A random BERT of about 88,000 parameters from the Hugging Face test models."""
    return TINY_MODEL


@pytest.fixture(scope="session")
def tiny_modernbert_8k(tmp_path_factory) -> Path:
    """The tiny random ModernBERT (about 10 MB) with ModernBERT-base's limit of 8,192 positions.

    ModernBERT uses rotary position embeddings, so the limit is a config value and the
    weights are unchanged.
    """
    out = tmp_path_factory.mktemp("tiny-modernbert-8k")
    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODERNBERT)
    model.config.max_position_embeddings = 8192
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(TINY_MODERNBERT).save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    write_splits(out, n_train=48, n_val=16, n_test=16, seed=0)
    return out


@pytest.fixture(scope="session")
def trained_model(synthetic_dir, tmp_path_factory) -> Path:
    """A tiny random BERT fine-tuned for two epochs on the synthetic notes, on the CPU."""
    out = tmp_path_factory.mktemp("run")
    status = main(
        [
            "train",
            "--train-csv", str(synthetic_dir / "train.csv"),
            "--eval-csv", str(synthetic_dir / "val.csv"),
            "--model", TINY_MODEL,
            "--output-dir", str(out),
            "--epochs", "2",
            "--batch-size", "8",
            "--max-length", "64",
            "--learning-rate", "1e-3",
            "--logging-steps", "2",
            "--cpu",
        ]
    )  # fmt: skip
    assert status == 0
    return out
