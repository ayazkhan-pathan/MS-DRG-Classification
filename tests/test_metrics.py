import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from transformers import EvalPrediction

from drg.metrics import classification_metrics, compute_metrics, macro_f1, top_k_accuracy


def random_case(seed, n=400, classes=30):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, classes, size=n)
    scores = rng.normal(size=(n, classes))
    # Make the scores informative, so the metrics are not all near chance.
    scores[np.arange(n), labels] += rng.uniform(0, 3, size=n)
    return scores, labels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accuracy_matches_sklearn(seed):
    scores, labels = random_case(seed)
    ours = classification_metrics(scores, labels)["top1_accuracy"]
    assert ours == pytest.approx(accuracy_score(labels, scores.argmax(axis=1)))


@pytest.mark.parametrize("k", [1, 5, 10])
def test_top_k_matches_sklearn(k):
    scores, labels = random_case(3)
    expected = top_k_accuracy_score(labels, scores, k=k, labels=np.arange(scores.shape[1]))
    assert top_k_accuracy(scores, labels, k) == pytest.approx(expected)


@pytest.mark.parametrize("seed", [4, 5])
def test_macro_f1_matches_drg_llama_definition(seed):
    scores, labels = random_case(seed, n=150, classes=60)  # some classes never occur
    predictions = scores.argmax(axis=1)
    assert not set(predictions) <= set(labels)  # some predictions are of absent classes
    expected = f1_score(
        labels, predictions, average="macro", labels=np.unique(labels), zero_division=0
    )
    assert macro_f1(predictions, labels) == pytest.approx(expected)


def test_predicting_an_absent_class_adds_no_zero_to_the_average():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 2, 1, 1])  # class 2 is predicted but never true
    # Class 0: F1 2/3; class 1: F1 1. Averaging over the union would add a 0 for class 2.
    assert macro_f1(predictions, labels) == pytest.approx(5 / 6)
    assert macro_f1(predictions, labels, classes=[0, 1, 2]) == pytest.approx(5 / 9)


def test_macro_f1_with_fixed_classes_matches_sklearn():
    labels = np.array([0, 0, 1, 2, 2, 2, 5])
    predictions = np.array([0, 1, 1, 2, 0, 2, 2])
    classes = np.arange(8)  # 3, 4, 6 and 7 have no notes and no predictions
    expected = f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
    assert macro_f1(predictions, labels, classes=classes) == pytest.approx(expected)


def test_perfect_scores():
    labels = np.array([2, 0, 1, 1])
    scores = np.eye(3)[labels]
    metrics = classification_metrics(scores, labels)
    assert metrics == {
        "top1_accuracy": 1.0,
        "top5_accuracy": 1.0,
        "top10_accuracy": 1.0,
        "macro_f1": 1.0,
    }


def test_k_at_least_the_number_of_classes_is_always_a_hit():
    scores, labels = random_case(6, n=50, classes=4)
    assert top_k_accuracy(scores, labels, k=4) == 1.0
    assert top_k_accuracy(scores, labels, k=10) == 1.0


def test_label_outside_the_scores_counts_as_a_miss():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 2])  # class 2 was never seen in training
    metrics = classification_metrics(scores, labels)
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top5_accuracy"] == pytest.approx(2 / 3)
    # Class 2 is a miss; class 0 also gets that note as a false positive.
    expected = f1_score(
        labels, scores.argmax(axis=1), average="macro", labels=np.unique(labels), zero_division=0
    )
    assert metrics["macro_f1"] == pytest.approx(expected)


def test_bad_shapes_raise():
    with pytest.raises(ValueError, match="2-D"):
        top_k_accuracy(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="labels"):
        classification_metrics(np.zeros((3, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="k must"):
        top_k_accuracy(np.zeros((3, 2)), np.zeros(3), k=0)
    with pytest.raises(ValueError, match="no notes"):
        macro_f1([], [])


def test_trainer_hook_accepts_eval_prediction():
    scores, labels = random_case(7)
    hook = compute_metrics(EvalPrediction(predictions=scores, label_ids=labels))
    assert hook == classification_metrics(scores, labels)
    assert set(hook) == {"top1_accuracy", "top5_accuracy", "top10_accuracy", "macro_f1"}
