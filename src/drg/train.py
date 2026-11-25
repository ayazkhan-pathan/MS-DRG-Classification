"""Fine-tune a sequence-classification model to predict DRG codes from note text.

The default model is Bio_ClinicalBERT (512 tokens) with a new linear DRG head; any encoder
with a sequence-classification head works, such as answerdotai/ModernBERT-base, which reads
up to 8,192 tokens. Each note is truncated to ``max_length`` tokens. The Hugging Face Trainer
evaluates on the validation CSV every epoch (or every ``eval_steps``) and keeps the checkpoint
with the best macro-F1. The output directory then holds the model, its tokenizer,
``label_map.json``, ``training_config.json`` and ``eval_metrics.json``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from drg.data import LabelMap, NoteDataset, TextCollator, read_notes, subsample
from drg.metrics import compute_metrics

log = logging.getLogger(__name__)

TRAINING_CONFIG_FILE = "training_config.json"
EVAL_METRICS_FILE = "eval_metrics.json"
BEST_METRIC = "macro_f1"


@dataclass
class TrainConfig:
    train_csv: str
    eval_csv: str
    output_dir: str
    base_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 512
    truncation_side: str = "right"
    subsample: float | None = None
    eval_subsample: float | None = None
    epochs: float = 3.0
    max_steps: int = -1
    batch_size: int = 16
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 50
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    cpu: bool = False
    num_workers: int = 0
    cache_dir: str | None = None
    resume_from_checkpoint: str | None = None
    wandb: bool = False
    wandb_project: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.eval_strategy not in ("epoch", "steps"):
            raise ValueError("eval_strategy must be 'epoch' or 'steps'")
        if self.truncation_side not in ("right", "left"):
            raise ValueError("truncation_side must be 'right' or 'left'")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8 tokens")


def build_training_arguments(config: TrainConfig) -> TrainingArguments:
    """Trainer settings: evaluate and save on the same schedule, keep the best macro-F1."""
    steps = config.eval_steps if config.eval_strategy == "steps" else None
    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size or config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy=config.eval_strategy,
        save_strategy=config.eval_strategy,
        eval_steps=steps,
        save_steps=steps or 500,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=config.logging_steps,
        seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        use_cpu=config.cpu,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["wandb"] if config.wandb else "none",
        run_name=config.run_name,
    )


def check_max_length(max_length: int, model_config, model_name: str) -> None:
    """Refuse a token limit beyond the positions the model was built for (512 for BERT)."""
    positions = getattr(model_config, "max_position_embeddings", None)
    if positions is not None and max_length > positions:
        raise ValueError(
            f"max_length {max_length} is longer than the {positions} positions "
            f"{model_name} can read"
        )


def train(config: TrainConfig) -> dict[str, float]:
    """Fine-tune ``config.base_model`` and return the best checkpoint's validation metrics."""
    model_config = AutoConfig.from_pretrained(config.base_model, cache_dir=config.cache_dir)
    check_max_length(config.max_length, model_config, config.base_model)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = {"text_column": config.text_column, "label_column": config.label_column}
    train_notes = subsample(read_notes(config.train_csv, **columns), config.subsample, config.seed)
    eval_notes = subsample(
        read_notes(config.eval_csv, **columns), config.eval_subsample, config.seed
    )
    label_map = LabelMap.from_labels(train_notes["label"])

    seen = eval_notes["label"].isin(label_map.labels)
    if not seen.all():
        log.warning(
            "%d validation notes have a DRG that is not in the training set; they are left "
            "out of validation during training (drg evaluate counts them as errors)",
            int((~seen).sum()),
        )
        eval_notes = eval_notes[seen].reset_index(drop=True)
    if eval_notes.empty:
        raise ValueError("no validation notes left with a DRG seen in training")

    log.info(
        "fine-tuning %s on %d notes and %d DRGs, validating on %d notes",
        config.base_model,
        len(train_notes),
        len(label_map),
        len(eval_notes),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, cache_dir=config.cache_dir)
    tokenizer.truncation_side = config.truncation_side
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(label_map),
        id2label=label_map.id2label,
        label2id=label_map.label2id,
        ignore_mismatched_sizes=True,
        cache_dir=config.cache_dir,
    )

    if config.wandb and config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=NoteDataset(train_notes["text"], label_map.encode(train_notes["label"])),
        eval_dataset=NoteDataset(eval_notes["text"], label_map.encode(eval_notes["label"])),
        data_collator=TextCollator(tokenizer, config.max_length),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        trainer.save_model(str(output_dir))
        label_map.save(output_dir)
        settings = asdict(config) | {
            "num_labels": len(label_map),
            "train_notes": len(train_notes),
            "eval_notes": len(eval_notes),
        }
        (output_dir / TRAINING_CONFIG_FILE).write_text(json.dumps(settings, indent=1) + "\n")
        (output_dir / EVAL_METRICS_FILE).write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics
