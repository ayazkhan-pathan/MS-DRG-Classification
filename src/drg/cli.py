"""Command line: ``drg train``, ``drg evaluate`` and ``drg predict``.

Examples:
    drg train --train-csv data/mimic/train.csv --eval-csv data/mimic/val.csv \\
        --output-dir runs/bioclinicalbert-512
    drg train --train-csv data/mimic/train.csv --eval-csv data/mimic/val.csv \\
        --model answerdotai/ModernBERT-base --max-length 8192 --output-dir runs/modernbert-8192
    drg evaluate --model-dir runs/bioclinicalbert-512 --csv data/mimic/test.csv
    drg predict --model-dir runs/bioclinicalbert-512 --csv notes.csv --output top5.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from drg import __version__


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show each option's default in --help, except empty ones (None or off)."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    if number >= 1 and not number.is_integer():
        raise argparse.ArgumentTypeError(
            f"use a fraction below 1 or a whole number of rows, got {value}"
        )
    return number


def _add_scoring_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", required=True, help="output directory of drg train")
    parser.add_argument("--csv", required=True, help="CSV of notes to score")
    parser.add_argument("--text-column", help="default: the column used in training")
    parser.add_argument("--max-length", type=int, help="token limit (default: as trained)")
    parser.add_argument("--truncation-side", choices=("right", "left"), help="default: as trained")
    parser.add_argument("--batch-size", type=int, default=32, help="notes per batch")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1 or mps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drg", description="Predict MS-DRG codes from discharge summaries."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    tr = commands.add_parser(
        "train", help="fine-tune a model on a labelled note CSV", formatter_class=_HelpFormatter
    )
    data = tr.add_argument_group("data")
    data.add_argument("--train-csv", required=True, help="training CSV")
    data.add_argument("--eval-csv", required=True, help="validation CSV for model selection")
    data.add_argument("--text-column", default="text", help="column holding the note text")
    data.add_argument("--label-column", default="label", help="column holding the DRG code")
    data.add_argument(
        "--subsample",
        type=_positive_float,
        help="train on a random subset: a fraction below 1 or a number of rows",
    )
    data.add_argument("--eval-subsample", type=_positive_float, help="the same for validation")

    model = tr.add_argument_group("model")
    model.add_argument(
        "--model",
        dest="base_model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="base model on the Hugging Face Hub, or a local directory; "
        "answerdotai/ModernBERT-base reads up to 8,192 tokens",
    )
    model.add_argument(
        "--max-length", type=int, default=512, help="tokens kept per note (up to the model's limit)"
    )
    model.add_argument(
        "--truncation-side",
        choices=("right", "left"),
        default="right",
        help="right keeps the start of each note, left keeps the end",
    )
    model.add_argument("--cache-dir", help="Hugging Face cache directory")

    opt = tr.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=float, default=3.0, help="training epochs")
    opt.add_argument(
        "--max-steps", type=int, default=-1, help="stop after this many steps; -1 runs every epoch"
    )
    opt.add_argument("--batch-size", type=int, default=16, help="training notes per device step")
    opt.add_argument("--eval-batch-size", type=int, help="default: --batch-size")
    opt.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="steps per optimiser update"
    )
    opt.add_argument("--learning-rate", type=float, default=2e-5, help="peak AdamW learning rate")
    opt.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    opt.add_argument(
        "--warmup-ratio", type=float, default=0.0, help="share of steps for linear warm-up"
    )
    opt.add_argument("--seed", type=int, default=42, help="seed for shuffling and subsampling")
    opt.add_argument("--fp16", action="store_true", help="float16 mixed precision (GPU)")
    opt.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
    opt.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute activations in the backward pass, to fit long inputs in memory",
    )
    opt.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is free")
    opt.add_argument("--num-workers", type=int, default=0, help="data loader workers")

    run = tr.add_argument_group("checkpoints and logging")
    run.add_argument("--output-dir", required=True, help="where the model and checkpoints go")
    run.add_argument(
        "--eval-strategy",
        choices=("epoch", "steps"),
        default="epoch",
        help="validate and save every epoch or every --eval-steps",
    )
    run.add_argument("--eval-steps", type=int, default=500, help="with --eval-strategy steps")
    run.add_argument("--save-total-limit", type=int, default=2, help="checkpoints kept on disk")
    run.add_argument("--logging-steps", type=int, default=50, help="steps between loss logs")
    run.add_argument("--resume-from-checkpoint", help="checkpoint directory to resume from")
    run.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    run.add_argument("--wandb-project", help="W&B project name")
    run.add_argument("--run-name", help="W&B run name")

    ev = commands.add_parser(
        "evaluate",
        help="top-1, top-5, top-10 and macro-F1 on a CSV",
        formatter_class=_HelpFormatter,
    )
    _add_scoring_options(ev)
    ev.add_argument("--label-column", help="default: the column used in training")
    ev.add_argument("--output", help="write the metrics to this JSON file")

    pr = commands.add_parser(
        "predict", help="write the most likely DRGs for each note", formatter_class=_HelpFormatter
    )
    _add_scoring_options(pr)
    pr.add_argument("--output", required=True, help="predictions CSV")
    pr.add_argument("--top-k", type=int, default=5, help="DRGs written per note")
    pr.add_argument("--id-column", help="copy this column into the predictions")
    return parser


def _train(args: argparse.Namespace) -> None:
    from drg.train import TrainConfig, train

    options = vars(args).copy()
    options.pop("command")
    metrics = train(TrainConfig(**options))
    print(json.dumps(metrics, indent=1))


def _evaluate(args: argparse.Namespace) -> None:
    from drg.predict import evaluate

    metrics = evaluate(
        args.model_dir,
        args.csv,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        batch_size=args.batch_size,
        device=args.device,
    )
    text = json.dumps(metrics, indent=1)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)


def _predict(args: argparse.Namespace) -> None:
    from drg.predict import predict

    table = predict(
        args.model_dir,
        args.csv,
        top_k=args.top_k,
        text_column=args.text_column,
        id_column=args.id_column,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        batch_size=args.batch_size,
        device=args.device,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"wrote {len(table)} predictions to {args.output}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    {"train": _train, "evaluate": _evaluate, "predict": _predict}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
