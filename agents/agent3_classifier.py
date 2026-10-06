"""Agent 3 — Decision Classifier (A3-only baseline).

Pipeline stage 3 of the three-agent RAG system. This module is the
**closed-book baseline**: it fine-tunes ``nlpaueb/legal-bert-base-uncased``
on the raw appeal case text alone, with no query formulation (Agent 1) and
no retrieved evidence (Agent 2). It deliberately imports nothing from the
other agents so the baseline stays independent of retrieval quality.

The macro-F1 produced here is the reference point that the RAG-augmented
conditions (A2+A3, full A1+A2+A3) are compared against in the ablation table.

Labels follow the TPAFS 3-class construction::

    0 = Insufficient   (sufficiency_id == 0)
    1 = Upheld
    2 = Overturned

Model selection
---------------
``metric_for_best_model`` is evaluated each epoch on a stratified validation
split carved out of the training file (default 10%, ``random_state=42``).
The test split is held out entirely and scored exactly once, after the best
checkpoint has been restored. Pass ``--val-size 0`` to evaluate directly on
the test split instead (leaks test into model selection — not recommended
for reported results).

CLI usage::

    # full run (GPU strongly recommended — see --help for CPU smoke testing)
    python agents/agent3_classifier.py

    # quick CPU smoke test on a few hundred rows
    python agents/agent3_classifier.py --max-train-samples 200 \
        --max-eval-samples 100 --epochs 1 --max-length 128
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# Label space — order fixed by the TPAFS baseline, do not reorder.
ID2LABEL: dict[int, str] = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
LABEL2ID: dict[str, int] = {name: idx for idx, name in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "imr_train_10k.csv"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "imr_test.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "legalbert_a3only"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Defaults mirror the Phase 3 protocol in CLAUDE.md.
RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_VAL_SIZE = 0.1

# Primary metric. Trainer prefixes eval metrics with "eval_", but
# metric_for_best_model is given as the bare key.
BEST_METRIC = "macro_f1"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class LabelledSplit:
    """A label-mapped split of the benchmark, ready for tokenisation."""

    name: str
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        """Return the number of examples in the split."""
        return len(self.labels)

    def label_counts(self) -> dict[str, int]:
        """Return per-class example counts keyed by human-readable label."""
        counts = {name: 0 for name in ID2LABEL.values()}
        for label_id in self.labels:
            counts[ID2LABEL[label_id]] += 1
        return counts


class AppealsDataset(torch.utils.data.Dataset):
    """Torch dataset of tokenised appeal texts with integer class labels.

    Tokenisation happens up front (the corpus is small enough to hold in
    memory) and padding is left to the collator, so batches pad to their
    longest member rather than always to ``max_length``.
    """

    def __init__(
        self,
        split: LabelledSplit,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Tokenise ``split`` with ``tokenizer``, truncating to ``max_length``."""
        if len(split) == 0:
            raise ValueError(f"Split '{split.name}' is empty - nothing to tokenise.")
        self.split = split
        encoded = tokenizer(
            split.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids: list[list[int]] = encoded["input_ids"]
        self.attention_mask: list[list[int]] = encoded["attention_mask"]
        self.token_type_ids: Optional[list[list[int]]] = encoded.get("token_type_ids")
        self.labels = split.labels

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one tokenised example as a dict of plain Python lists."""
        item: dict[str, Any] = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[index]
        return item


def load_split(
    csv_path: Path,
    name: str,
    max_samples: Optional[int] = None,
) -> LabelledSplit:
    """Load a processed IMR CSV and map its ``label`` column to class ids.

    Args:
        csv_path: Path to a CSV with ``text`` and ``label`` columns.
        name: Human-readable split name, used in logs and error messages.
        max_samples: If given, keep only the first N rows (smoke testing).

    Returns:
        A :class:`LabelledSplit` with texts and integer labels.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If required columns are missing or a label is unknown.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{name} CSV not found at {csv_path}. Run the Phase 1 data prep "
            "step to regenerate data/processed/."
        )

    logger.info("Loading %s split from %s", name, csv_path)
    frame = pd.read_csv(csv_path)

    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required column(s): {sorted(missing_columns)}. "
            f"Found: {sorted(frame.columns)}"
        )

    before = len(frame)
    frame = frame.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    frame = frame[frame[TEXT_COLUMN].astype(str).str.strip().astype(bool)]
    dropped = before - len(frame)
    if dropped:
        logger.warning("Dropped %d rows from %s with empty text/label", dropped, name)

    unknown_labels = set(frame[LABEL_COLUMN].unique()) - set(LABEL2ID)
    if unknown_labels:
        raise ValueError(
            f"Unexpected label value(s) in {csv_path}: {sorted(unknown_labels)}. "
            f"Expected one of {sorted(LABEL2ID)}."
        )

    if max_samples is not None:
        frame = frame.head(max_samples)

    split = LabelledSplit(
        name=name,
        texts=frame[TEXT_COLUMN].astype(str).tolist(),
        labels=[LABEL2ID[value] for value in frame[LABEL_COLUMN]],
    )
    logger.info(
        "%s split: %d rows, distribution %s", name, len(split), split.label_counts()
    )
    return split


def stratified_holdout(
    split: LabelledSplit,
    val_size: float,
    random_state: int = RANDOM_STATE,
) -> tuple[LabelledSplit, LabelledSplit]:
    """Split ``split`` into train/validation parts, stratified by label.

    Args:
        split: The split to subdivide.
        val_size: Fraction held out for validation, in (0, 1).
        random_state: Seed for the split (fixed at 42 across the project).

    Returns:
        ``(train_split, val_split)``.

    Raises:
        ValueError: If ``val_size`` is outside (0, 1), or the rarest class has
            too few members for a stratified split.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    label_array = np.asarray(split.labels)
    rarest_count = int(np.bincount(label_array, minlength=NUM_LABELS).min())
    if rarest_count < 2:
        raise ValueError(
            "Cannot stratify: at least one class has fewer than 2 examples "
            f"(counts: {split.label_counts()}). Increase --max-train-samples."
        )

    train_indices, val_indices = train_test_split(
        np.arange(len(split)),
        test_size=val_size,
        random_state=random_state,
        stratify=label_array,
    )
    train_split = LabelledSplit(
        name=f"{split.name}[train]",
        texts=[split.texts[i] for i in train_indices],
        labels=[split.labels[i] for i in train_indices],
    )
    val_split = LabelledSplit(
        name=f"{split.name}[val]",
        texts=[split.texts[i] for i in val_indices],
        labels=[split.labels[i] for i in val_indices],
    )
    logger.info(
        "Stratified holdout (val_size=%.2f, seed=%d): train=%d %s | val=%d %s",
        val_size,
        random_state,
        len(train_split),
        train_split.label_counts(),
        len(val_split),
        val_split.label_counts(),
    )
    return train_split, val_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score_predictions(
    gold_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Score predictions against gold labels.

    ``zero_division=0`` matters here: the Insufficient class is ~1-2% of the
    corpus, so the model may predict it zero times in an early epoch.

    Args:
        gold_labels: Gold class ids, shape ``(n,)``.
        predicted_labels: Predicted class ids, shape ``(n,)``.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, and per-class
        precision/recall/F1/support.
    """
    all_ids = list(ID2LABEL)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                gold_labels,
                predicted_labels,
                labels=all_ids,
                average="macro",
                zero_division=0,
            )
        ),
    }
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
        labels=all_ids,
        average=None,
        zero_division=0,
    )
    for position, label_id in enumerate(all_ids):
        label_name = ID2LABEL[label_id]
        metrics[f"precision_{label_name}"] = float(precision[position])
        metrics[f"recall_{label_name}"] = float(recall[position])
        metrics[f"f1_{label_name}"] = float(f1_per_class[position])
        metrics[f"support_{label_name}"] = int(support[position])
    return metrics


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """Compute macro-F1, accuracy and per-class metrics for the Trainer.

    Args:
        eval_prediction: Trainer-supplied logits and gold label ids.

    Returns:
        Metric dict; ``macro_f1`` is the model-selection key.
    """
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = np.argmax(logits, axis=-1)
    return score_predictions(np.asarray(eval_prediction.label_ids), predicted_labels)


def format_confusion_matrix(
    gold_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> str:
    """Render the confusion matrix as a fixed-width text table.

    Args:
        gold_labels: Gold class ids.
        predicted_labels: Predicted class ids.

    Returns:
        A multi-line string with gold labels as rows, predictions as columns.
    """
    all_ids = list(ID2LABEL)
    matrix = confusion_matrix(gold_labels, predicted_labels, labels=all_ids)
    label_names = [ID2LABEL[label_id] for label_id in all_ids]
    row_header_width = max(len(name) for name in label_names) + len("gold ") + 2
    cell_width = max(max(len(name) for name in label_names), 6) + 2

    header = " " * row_header_width + "".join(
        f"{name:>{cell_width}}" for name in label_names
    )
    lines = ["(rows = gold, columns = predicted)", header]
    for row_index, label_id in enumerate(all_ids):
        row_label = f"gold {ID2LABEL[label_id]}"
        cells = "".join(
            f"{int(matrix[row_index][column_index]):>{cell_width}}"
            for column_index in range(len(all_ids))
        )
        lines.append(f"{row_label:<{row_header_width}}{cells}")
    return "\n".join(lines)


def format_per_class_report(metrics: dict[str, float]) -> str:
    """Render per-class precision/recall/F1/support as a text table.

    Args:
        metrics: Output of :func:`score_predictions`.

    Returns:
        A multi-line string.
    """
    name_width = max(len(name) for name in ID2LABEL.values()) + 2
    lines = [
        f"{'class':<{name_width}}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}"
    ]
    for label_name in ID2LABEL.values():
        lines.append(
            f"{label_name:<{name_width}}"
            f"{metrics[f'precision_{label_name}']:>11.4f}"
            f"{metrics[f'recall_{label_name}']:>9.4f}"
            f"{metrics[f'f1_{label_name}']:>9.4f}"
            f"{metrics[f'support_{label_name}']:>9d}"
        )
    lines.append(
        f"{'macro avg':<{name_width}}{'':>11}{'':>9}{metrics['macro_f1']:>9.4f}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted cross-entropy.

    Only used when ``--class-weights`` is passed. The A3-only baseline runs
    unweighted by default so its macro-F1 stays directly comparable to the
    TPAFS reference implementation.
    """

    def __init__(self, class_weights: torch.Tensor, **kwargs: Any) -> None:
        """Store per-class loss weights, ordered by class id."""
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Compute weighted cross-entropy over the model's logits."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        inputs["labels"] = labels
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(split: LabelledSplit) -> torch.Tensor:
    """Return inverse-frequency class weights normalised to mean 1.0.

    Args:
        split: The training split whose distribution sets the weights.

    Returns:
        A float tensor of shape ``(NUM_LABELS,)`` ordered by class id.
    """
    counts = np.bincount(np.asarray(split.labels), minlength=NUM_LABELS).astype(float)
    counts[counts == 0] = 1.0  # avoid divide-by-zero on an absent class
    weights = counts.sum() / (NUM_LABELS * counts)
    weights = weights / weights.mean()
    logger.info(
        "Class weights: %s",
        {ID2LABEL[i]: round(float(weights[i]), 3) for i in range(NUM_LABELS)},
    )
    return torch.tensor(weights, dtype=torch.float)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_arguments(
    args: argparse.Namespace, use_gpu: bool
) -> TrainingArguments:
    """Assemble ``TrainingArguments`` from parsed CLI arguments.

    Args:
        args: Parsed CLI namespace.
        use_gpu: Whether a CUDA device is available (enables fp16).

    Returns:
        Configured :class:`TrainingArguments`.
    """
    trainer_output_dir = Path(args.checkpoint_dir) / "trainer_checkpoints"
    return TrainingArguments(
        output_dir=str(trainer_output_dir),
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        fp16=use_gpu,
        dataloader_num_workers=args.dataloader_workers,
        report_to=args.report_to,
        run_name=args.run_name,
    )


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune LegalBERT on case text alone and score the test split.

    Args:
        args: Parsed CLI namespace.

    Returns:
        A dict with the final test metrics, validation metrics and run config.
    """
    set_global_seed(args.seed)

    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    logger.info("Device: %s (fp16=%s)", device_name, use_gpu)
    if not use_gpu:
        logger.warning(
            "No CUDA device found. Fine-tuning LegalBERT on 10k cases at "
            "max_length=%d will take many hours on CPU - run this on Colab "
            "(Phase 3), or pass --max-train-samples for a local smoke test.",
            args.max_length,
        )

    full_train = load_split(Path(args.train_csv), "train", args.max_train_samples)
    test_split = load_split(Path(args.test_csv), "test", args.max_eval_samples)

    if args.val_size > 0:
        train_split, eval_split = stratified_holdout(
            full_train, args.val_size, args.seed
        )
    else:
        logger.warning(
            "--val-size 0: selecting the best checkpoint on the TEST split. "
            "This leaks test into model selection; do not report the result "
            "as a clean held-out number."
        )
        train_split, eval_split = full_train, test_split

    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = AppealsDataset(train_split, tokenizer, args.max_length)
    eval_dataset = AppealsDataset(eval_split, tokenizer, args.max_length)
    test_dataset = AppealsDataset(test_split, tokenizer, args.max_length)

    training_arguments = build_training_arguments(args, use_gpu)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if args.class_weights:
        trainer = WeightedLossTrainer(
            class_weights=compute_class_weights(train_split), **trainer_kwargs
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    logger.info(
        "Starting fine-tuning: %d train / %d eval examples, %s epochs, batch=%d",
        len(train_dataset),
        len(eval_dataset),
        args.epochs,
        args.batch_size,
    )
    trainer.train()

    validation_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info(
        "Best checkpoint validation macro-F1: %.4f",
        validation_metrics.get(f"eval_{BEST_METRIC}", float("nan")),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logger.info("Saved best checkpoint to %s", checkpoint_dir)

    logger.info("Scoring held-out test split (%d examples)", len(test_dataset))
    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = np.argmax(logits, axis=-1)
    gold_labels = np.asarray(test_split.labels)
    test_metrics = score_predictions(gold_labels, predicted_labels)

    report: dict[str, Any] = {
        "run": {
            "condition": "A3-only (closed-book, no retrieval)",
            "model_name": args.model_name,
            "train_csv": str(args.train_csv),
            "test_csv": str(args.test_csv),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "val_size": args.val_size,
            "class_weights": args.class_weights,
            "device": device_name,
            "n_train": len(train_dataset),
            "n_val": len(eval_dataset),
            "n_test": len(test_dataset),
        },
        "validation_metrics": {
            key.removeprefix("eval_"): value
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float))
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": confusion_matrix(
            gold_labels, predicted_labels, labels=list(ID2LABEL)
        ).tolist(),
        "label_order": [ID2LABEL[label_id] for label_id in sorted(ID2LABEL)],
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "agent3_a3only_metrics.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote metrics to %s", results_path)

    print_final_report(report, gold_labels, predicted_labels)
    return report


def print_final_report(
    report: dict[str, Any],
    gold_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Print the headline test-set results to stdout.

    Args:
        report: The assembled results dict.
        gold_labels: Gold class ids for the test split.
        predicted_labels: Predicted class ids for the test split.
    """
    test_metrics = report["test_metrics"]
    print()
    print("=" * 72)
    print("AGENT 3 - A3-ONLY BASELINE (closed-book, no retrieval)")
    print("=" * 72)
    print(f"Model         : {report['run']['model_name']}")
    print(f"Test examples : {report['run']['n_test']}")
    print()
    print(f"MACRO-F1 (test) : {test_metrics['macro_f1']:.4f}")
    print(f"Accuracy (test) : {test_metrics['accuracy']:.4f}")
    print()
    print("Per-class metrics")
    print("-" * 72)
    print(format_per_class_report(test_metrics))
    print()
    print("Confusion matrix")
    print("-" * 72)
    print(format_confusion_matrix(gold_labels, predicted_labels))
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the A3-only training script."""
    parser = argparse.ArgumentParser(
        description=(
            "Agent 3 - fine-tune LegalBERT on appeal case text (A3-only baseline)."
        )
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV_PATH)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help=(
            "Fraction of the train file held out for per-epoch model selection. "
            "0 evaluates on the test split instead (leaks test - not recommended)."
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help=(
            "Use inverse-frequency weighted cross-entropy "
            "(helps the rare Insufficient class)."
        ),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Truncate the train file to N rows (CPU smoke testing).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Truncate the test file to N rows (CPU smoke testing).",
    )
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Trainer reporting integration, e.g. 'wandb' or 'none'.",
    )
    parser.add_argument("--run-name", type=str, default="legalbert_a3only")
    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args()
    try:
        train_and_evaluate(args)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Training aborted: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
