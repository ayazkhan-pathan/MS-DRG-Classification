import pytest

from drg.cli import build_parser, main
from drg.train import TrainConfig, build_training_arguments, train

REQUIRED = ["train", "--train-csv", "a.csv", "--eval-csv", "b.csv", "--output-dir", "out"]


@pytest.mark.parametrize("command", ["train", "evaluate", "predict"])
def test_help_works(command, capsys):
    with pytest.raises(SystemExit) as stop:
        main([command, "--help"])
    assert stop.value.code == 0
    assert "usage: drg" in capsys.readouterr().out


def test_help_shows_defaults_but_not_empty_ones(capsys):
    with pytest.raises(SystemExit):
        main(["train", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "(default: emilyalsentzer/Bio_ClinicalBERT)" in text
    assert "(default: 512)" in text and "(default: 2e-05)" in text
    assert "(default: None)" not in text and "(default: False)" not in text


def config_from_cli(*extra: str) -> TrainConfig:
    args = vars(build_parser().parse_args([*REQUIRED, *extra]))
    args.pop("command")
    return TrainConfig(**args)


def test_train_defaults_match_drg_llamas_clinicalbert_settings():
    config = config_from_cli()
    assert config.base_model == "emilyalsentzer/Bio_ClinicalBERT"
    assert (config.max_length, config.batch_size, config.epochs) == (512, 16, 3.0)
    assert (config.learning_rate, config.weight_decay) == (2e-5, 0.01)
    assert config.subsample is None
    assert config.wandb is False
    assert config.gradient_checkpointing is False


def test_trainer_keeps_best_macro_f1_and_logs_nowhere_by_default(tmp_path):
    args = build_training_arguments(TrainConfig("a.csv", "b.csv", str(tmp_path)))
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
    assert args.load_best_model_at_end
    assert args.metric_for_best_model == "macro_f1"
    assert args.greater_is_better
    assert args.report_to == []


def test_wandb_only_when_asked(tmp_path):
    config = TrainConfig("a.csv", "b.csv", str(tmp_path), wandb=True)
    assert build_training_arguments(config).report_to == ["wandb"]


def test_step_schedule(tmp_path):
    config = TrainConfig("a.csv", "b.csv", str(tmp_path), eval_strategy="steps", eval_steps=40)
    args = build_training_arguments(config)
    assert (args.eval_steps, args.save_steps) == (40, 40)


@pytest.mark.parametrize("value", ["0", "-0.5", "2.5"])
def test_subsample_rejects_bad_values(value, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([*REQUIRED, "--subsample", value])
    assert "subsample" in capsys.readouterr().err


def test_max_length_beyond_the_model_positions_is_refused(tiny_model, synthetic_dir, tmp_path):
    config = TrainConfig(
        str(synthetic_dir / "train.csv"),
        str(synthetic_dir / "val.csv"),
        str(tmp_path / "out"),
        base_model=tiny_model,
        max_length=1024,
    )
    with pytest.raises(ValueError, match="512 positions"):
        train(config)
    assert not (tmp_path / "out").exists()
