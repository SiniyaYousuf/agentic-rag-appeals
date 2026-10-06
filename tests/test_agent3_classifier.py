"""Tests for Agent 3 — label mapping, splitting, metrics and report formatting.

These cover the pure functions only; nothing here downloads LegalBERT or
runs a training loop, so the suite stays offline and fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.agent3_classifier import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    LabelledSplit,
    build_arg_parser,
    compute_class_weights,
    compute_metrics,
    format_confusion_matrix,
    format_per_class_report,
    load_split,
    score_predictions,
    stratified_holdout,
)


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------


def test_label_space_matches_tpafs_ordering() -> None:
    """Class ids must follow the TPAFS ID2LABEL contract exactly."""
    assert ID2LABEL == {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
    assert LABEL2ID == {"Insufficient": 0, "Upheld": 1, "Overturned": 2}
    assert NUM_LABELS == 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a minimal text/label CSV and return its path."""
    frame = pd.DataFrame(rows, columns=["text", "label"])
    frame.to_csv(path, index=False)
    return path


def test_load_split_maps_labels_to_ids(tmp_path: Path) -> None:
    """Label strings are mapped to their TPAFS class ids."""
    csv_path = write_csv(
        tmp_path / "mini.csv",
        [
            ("case a", "Upheld"),
            ("case b", "Overturned"),
            ("case c", "Insufficient"),
        ],
    )
    split = load_split(csv_path, "mini")
    assert split.labels == [1, 2, 0]
    assert split.texts == ["case a", "case b", "case c"]
    assert split.label_counts() == {"Insufficient": 1, "Upheld": 1, "Overturned": 1}


def test_load_split_drops_blank_text(tmp_path: Path) -> None:
    """Rows with empty or whitespace-only text are dropped."""
    csv_path = write_csv(
        tmp_path / "blanks.csv",
        [("real case", "Upheld"), ("   ", "Overturned")],
    )
    split = load_split(csv_path, "blanks")
    assert len(split) == 1
    assert split.texts == ["real case"]


def test_load_split_respects_max_samples(tmp_path: Path) -> None:
    """``max_samples`` truncates the loaded split."""
    csv_path = write_csv(
        tmp_path / "many.csv", [(f"case {i}", "Upheld") for i in range(10)]
    )
    assert len(load_split(csv_path, "many", max_samples=3)) == 3


def test_load_split_missing_file_raises(tmp_path: Path) -> None:
    """A missing CSV raises FileNotFoundError with a recovery hint."""
    with pytest.raises(FileNotFoundError, match="Phase 1 data prep"):
        load_split(tmp_path / "nope.csv", "nope")


def test_load_split_rejects_unknown_label(tmp_path: Path) -> None:
    """An out-of-vocabulary label value is rejected rather than silently mapped."""
    csv_path = write_csv(tmp_path / "bad.csv", [("case", "Partially Overturned")])
    with pytest.raises(ValueError, match="Unexpected label"):
        load_split(csv_path, "bad")


def test_load_split_rejects_missing_column(tmp_path: Path) -> None:
    """A CSV without the label column is rejected."""
    csv_path = tmp_path / "nolabel.csv"
    pd.DataFrame({"text": ["case"]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_split(csv_path, "nolabel")


# ---------------------------------------------------------------------------
# Stratified holdout
# ---------------------------------------------------------------------------


def make_split(counts: dict[str, int]) -> LabelledSplit:
    """Build a synthetic split with the requested per-class counts."""
    texts: list[str] = []
    labels: list[int] = []
    for label_name, count in counts.items():
        for index in range(count):
            texts.append(f"{label_name} case {index}")
            labels.append(LABEL2ID[label_name])
    return LabelledSplit(name="synthetic", texts=texts, labels=labels)


def test_stratified_holdout_preserves_class_proportions() -> None:
    """Every class appears in both parts, in roughly its original proportion."""
    split = make_split({"Insufficient": 20, "Upheld": 500, "Overturned": 480})
    train_split, val_split = stratified_holdout(split, val_size=0.1)

    assert len(train_split) + len(val_split) == len(split)
    assert val_split.label_counts() == {
        "Insufficient": 2,
        "Upheld": 50,
        "Overturned": 48,
    }
    assert train_split.label_counts() == {
        "Insufficient": 18,
        "Upheld": 450,
        "Overturned": 432,
    }


def test_stratified_holdout_is_deterministic() -> None:
    """The same seed yields the same partition (random_state=42 protocol)."""
    split = make_split({"Insufficient": 20, "Upheld": 200, "Overturned": 180})
    first_train, first_val = stratified_holdout(split, 0.1, random_state=42)
    second_train, second_val = stratified_holdout(split, 0.1, random_state=42)
    assert first_val.texts == second_val.texts
    assert first_train.texts == second_train.texts


def test_stratified_holdout_train_and_val_are_disjoint() -> None:
    """No example leaks from train into validation."""
    split = make_split({"Insufficient": 10, "Upheld": 100, "Overturned": 90})
    train_split, val_split = stratified_holdout(split, 0.2)
    assert set(train_split.texts).isdisjoint(set(val_split.texts))


@pytest.mark.parametrize("val_size", [0.0, 1.0, -0.1, 1.5])
def test_stratified_holdout_rejects_out_of_range_size(val_size: float) -> None:
    """``val_size`` must be a proper fraction."""
    split = make_split({"Insufficient": 10, "Upheld": 10, "Overturned": 10})
    with pytest.raises(ValueError, match="val_size must be in"):
        stratified_holdout(split, val_size)


def test_stratified_holdout_rejects_singleton_class() -> None:
    """A class with a single example cannot be stratified."""
    split = make_split({"Insufficient": 1, "Upheld": 50, "Overturned": 50})
    with pytest.raises(ValueError, match="fewer than 2 examples"):
        stratified_holdout(split, 0.1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_score_predictions_perfect() -> None:
    """Perfect predictions give macro-F1 and accuracy of 1.0."""
    gold = np.array([0, 1, 2, 1, 2])
    metrics = score_predictions(gold, gold.copy())
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    for label_name in ID2LABEL.values():
        assert metrics[f"f1_{label_name}"] == pytest.approx(1.0)


def test_score_predictions_handles_never_predicted_class() -> None:
    """A class the model never predicts scores 0.0, not NaN.

    This is the realistic failure mode for Insufficient (~1-2% of the corpus).
    """
    gold = np.array([0, 1, 1, 2, 2])
    predicted = np.array([1, 1, 1, 2, 2])
    metrics = score_predictions(gold, predicted)
    assert metrics["f1_Insufficient"] == pytest.approx(0.0)
    assert metrics["recall_Insufficient"] == pytest.approx(0.0)
    assert not np.isnan(metrics["macro_f1"])
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_score_predictions_reports_gold_support() -> None:
    """Support counts come from the gold labels, not the predictions."""
    gold = np.array([0, 0, 1, 1, 1, 2])
    predicted = np.array([1, 1, 1, 1, 1, 1])
    metrics = score_predictions(gold, predicted)
    assert metrics["support_Insufficient"] == 2
    assert metrics["support_Upheld"] == 3
    assert metrics["support_Overturned"] == 1


def test_score_predictions_macro_f1_is_unweighted_mean() -> None:
    """macro_f1 is the plain mean of the three per-class F1 scores."""
    gold = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    metrics = score_predictions(gold, predicted)
    per_class = [metrics[f"f1_{name}"] for name in ID2LABEL.values()]
    assert metrics["macro_f1"] == pytest.approx(sum(per_class) / NUM_LABELS)


class StubEvalPrediction:
    """Minimal stand-in for ``EvalPrediction`` (duck-typed by compute_metrics)."""

    def __init__(self, predictions: np.ndarray, label_ids: np.ndarray) -> None:
        """Store raw logits and gold label ids."""
        self.predictions = predictions
        self.label_ids = label_ids


def test_compute_metrics_argmaxes_logits() -> None:
    """compute_metrics turns logits into class ids before scoring."""
    logits = np.array(
        [
            [9.0, 0.0, 0.0],  # -> Insufficient
            [0.0, 9.0, 0.0],  # -> Upheld
            [0.0, 0.0, 9.0],  # -> Overturned
        ]
    )
    metrics = compute_metrics(StubEvalPrediction(logits, np.array([0, 1, 2])))
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_compute_metrics_unwraps_tuple_predictions() -> None:
    """Some model heads return a tuple; the first element is the logits."""
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    stub = StubEvalPrediction((logits, np.zeros((2, 4))), np.array([0, 1]))
    assert compute_metrics(stub)["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_exposes_the_model_selection_key() -> None:
    """``macro_f1`` must exist — Trainer selects the best checkpoint on it."""
    logits = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    metrics = compute_metrics(StubEvalPrediction(logits, np.array([0, 1])))
    assert "macro_f1" in metrics


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def test_compute_class_weights_favours_the_rare_class() -> None:
    """Inverse-frequency weighting ranks Insufficient highest."""
    split = make_split({"Insufficient": 20, "Upheld": 500, "Overturned": 480})
    weights = compute_class_weights(split)
    assert weights.shape == (NUM_LABELS,)
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)


def test_compute_class_weights_uniform_for_balanced_data() -> None:
    """A balanced split yields equal weights."""
    split = make_split({"Insufficient": 30, "Upheld": 30, "Overturned": 30})
    weights = compute_class_weights(split)
    assert float(weights[0]) == pytest.approx(1.0)
    assert float(weights[1]) == pytest.approx(1.0)
    assert float(weights[2]) == pytest.approx(1.0)


def test_compute_class_weights_survives_absent_class() -> None:
    """A class with zero examples must not produce inf or NaN weights."""
    split = make_split({"Upheld": 50, "Overturned": 50})
    weights = compute_class_weights(split)
    assert bool(np.isfinite(weights.numpy()).all())


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def test_format_confusion_matrix_places_counts_on_the_diagonal() -> None:
    """A perfect classifier puts all mass on the diagonal."""
    gold = np.array([0, 0, 1, 1, 1, 2])
    rendered = format_confusion_matrix(gold, gold.copy())
    lines = rendered.splitlines()
    assert "rows = gold" in lines[0]
    assert all(name in lines[1] for name in ID2LABEL.values())
    assert lines[2].startswith("gold Insufficient")
    assert lines[2].split()[2:] == ["2", "0", "0"]
    assert lines[3].split()[2:] == ["0", "3", "0"]
    assert lines[4].split()[2:] == ["0", "0", "1"]


def test_format_confusion_matrix_includes_absent_classes() -> None:
    """Rows and columns exist for classes with no examples at all."""
    gold = np.array([1, 1, 2])
    rendered = format_confusion_matrix(gold, np.array([1, 2, 2]))
    assert "gold Insufficient" in rendered
    assert len(rendered.splitlines()) == 2 + NUM_LABELS


def test_format_per_class_report_lists_every_class_and_macro_avg() -> None:
    """The per-class table covers all three classes plus the macro average."""
    gold = np.array([0, 1, 1, 2, 2])
    metrics = score_predictions(gold, np.array([0, 1, 2, 2, 2]))
    rendered = format_per_class_report(metrics)
    for label_name in ID2LABEL.values():
        assert label_name in rendered
    assert "macro avg" in rendered
    assert f"{metrics['macro_f1']:.4f}" in rendered


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------


def test_cli_defaults_match_the_phase3_protocol() -> None:
    """Defaults encode the training protocol specified in CLAUDE.md."""
    args = build_arg_parser().parse_args([])
    assert args.seed == 42
    assert args.batch_size == 16
    assert args.epochs == 3
    assert args.max_length == 512
    assert args.model_name == "nlpaueb/legal-bert-base-uncased"
    assert Path(args.checkpoint_dir).name == "legalbert_a3only"
    assert args.class_weights is False


def test_cli_accepts_smoke_test_overrides() -> None:
    """Sample caps and the class-weight flag are wired through."""
    args = build_arg_parser().parse_args(
        ["--max-train-samples", "50", "--max-eval-samples", "20", "--class-weights"]
    )
    assert args.max_train_samples == 50
    assert args.max_eval_samples == 20
    assert args.class_weights is True
