"""Retrieval proxy evaluation — A1 vs A1-null ablation on retrieval quality.

Runs the two Phase 2 retrieval conditions over a stratified sample of the
imr-appeals test split and compares retrieval similarity scores:

- FULL: Agent 1 (zero-shot FLAN-T5) formulates the query and forwards
  metadata (insurance_type) to Agent 2's tag filter.
- NULL: the raw case text is passed directly to Agent 2 with no metadata
  (the thesis's "A1-null" ablation: FLAN-T5 skipped entirely).

Cosine similarity against the KB is a *proxy* for retrieval quality (the KB
is unlabelled, so there is no gold relevance); the decisive comparison is the
downstream macro-F1 delta in Phase 4.

Usage::

    python evaluation/retrieval_proxy_eval.py [--sample-size 200]

Outputs:

- ``evaluation/retrieval_proxy_results.csv``  — one row per case
- ``evaluation/retrieval_proxy_summary.json`` — aggregated metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent1_query_formulator import Agent1QueryFormulator  # noqa: E402
from agents.agent2_retrieval import Agent2Retriever, RetrievedPassage  # noqa: E402

logger = logging.getLogger(__name__)

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "imr_test.csv"
RESULTS_CSV = PROJECT_ROOT / "evaluation" / "retrieval_proxy_results.csv"
SUMMARY_JSON = PROJECT_ROOT / "evaluation" / "retrieval_proxy_summary.json"

TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.3
RANDOM_STATE = 42  # fixed everywhere in this project (CLAUDE.md)
SNIPPET_CHARS = 120


def load_stratified_sample(test_csv: Path, sample_size: int) -> pd.DataFrame:
    """Load a stratified sample preserving class proportions.

    Uses proportional allocation per label with a fixed seed; rounding
    remainders go to the largest classes so the total is exactly
    ``sample_size``. Every class keeps at least one case.
    """
    frame = pd.read_csv(test_csv)
    frame["case_id"] = frame.index
    proportions = frame["label"].value_counts(normalize=True)
    allocation = (proportions * sample_size).apply(np.floor).astype(int).clip(lower=1)
    while allocation.sum() < sample_size:
        allocation[proportions.idxmax()] += 1
    while allocation.sum() > sample_size:
        allocation[proportions.idxmax()] -= 1

    parts = [
        frame[frame["label"] == label].sample(n=count, random_state=RANDOM_STATE)
        for label, count in allocation.items()
    ]
    sample = pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE)
    logger.info(
        "Sampled %d/%d cases: %s",
        len(sample),
        len(frame),
        allocation.to_dict(),
    )
    return sample.reset_index(drop=True)


def summarise_hits(
    hits: Sequence[RetrievedPassage], doc_paths: dict[int, str]
) -> dict[str, Any]:
    """Flatten a top-k hit list into result-row fields."""
    scores = [hit.score for hit in hits]
    return {
        "top5_doc_ids": ";".join(str(hit.doc_id) for hit in hits),
        "top5_doc_paths": ";".join(doc_paths.get(hit.doc_id, "") for hit in hits),
        "top5_scores": ";".join(f"{score:.4f}" for score in scores),
        "top1_score": scores[0] if scores else float("nan"),
        "top5_mean_score": float(np.mean(scores)) if scores else float("nan"),
        "top5_snippets": " || ".join(
            hit.passage_text[:SNIPPET_CHARS].replace("\n", " ") for hit in hits
        ),
    }


def compute_summary(results: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-case results into the summary metrics dict."""

    def unique_docs(column: str) -> int:
        docs: set[str] = set()
        for cell in results[column]:
            docs.update(str(cell).split(";"))
        docs.discard("")
        return len(docs)

    per_class: dict[str, Any] = {}
    for label, group in results.groupby("label"):
        per_class[str(label)] = {
            "n_cases": int(len(group)),
            "full_top1_mean": round(float(group["full_top1_score"].mean()), 4),
            "null_top1_mean": round(float(group["null_top1_score"].mean()), 4),
            "full_top5_mean": round(float(group["full_top5_mean_score"].mean()), 4),
            "null_top5_mean": round(float(group["null_top5_mean_score"].mean()), 4),
        }

    low_confidence = {
        "full": results.loc[
            results["full_top1_score"] < LOW_CONFIDENCE_THRESHOLD, "case_id"
        ].tolist(),
        "null": results.loc[
            results["null_top1_score"] < LOW_CONFIDENCE_THRESHOLD, "case_id"
        ].tolist(),
    }

    full_top1 = float(results["full_top1_score"].mean())
    null_top1 = float(results["null_top1_score"].mean())
    full_top5 = float(results["full_top5_mean_score"].mean())
    null_top5 = float(results["null_top5_mean_score"].mean())

    return {
        "n_cases": int(len(results)),
        "top_k": TOP_K,
        "mean_top1_score": {"full": round(full_top1, 4), "null": round(null_top1, 4)},
        "mean_top5_mean_score": {
            "full": round(full_top5, 4),
            "null": round(null_top5, 4),
        },
        "score_delta_full_minus_null": {
            "top1": round(full_top1 - null_top1, 4),
            "top5_mean": round(full_top5 - null_top5, 4),
        },
        "per_class": per_class,
        "unique_kb_docs_retrieved": {
            "full": unique_docs("full_top5_doc_ids"),
            "null": unique_docs("null_top5_doc_ids"),
        },
        "agent1_fallback_rate": round(float(results["agent1_used_fallback"].mean()), 4),
        "low_confidence_cases_top1_below_0.3": low_confidence,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print the summary as a readable console report."""
    print("\n" + "=" * 72)
    print(f"RETRIEVAL PROXY EVALUATION — {summary['n_cases']} cases, top-{summary['top_k']}")
    print("=" * 72)
    print(f"{'metric':<34}{'FULL (A1+A2)':>14}{'NULL (A2 only)':>16}{'delta':>8}")
    print("-" * 72)
    top1 = summary["mean_top1_score"]
    top5 = summary["mean_top5_mean_score"]
    delta = summary["score_delta_full_minus_null"]
    print(f"{'mean top-1 similarity':<34}{top1['full']:>14.4f}{top1['null']:>16.4f}{delta['top1']:>+8.4f}")
    print(f"{'mean top-5 mean similarity':<34}{top5['full']:>14.4f}{top5['null']:>16.4f}{delta['top5_mean']:>+8.4f}")
    print("-" * 72)
    print("per-class mean top-1 (FULL / NULL):")
    for label, stats in summary["per_class"].items():
        print(
            f"  {label:<14} n={stats['n_cases']:<5} "
            f"{stats['full_top1_mean']:.4f} / {stats['null_top1_mean']:.4f}"
        )
    unique = summary["unique_kb_docs_retrieved"]
    print(f"unique KB docs retrieved: FULL={unique['full']}, NULL={unique['null']}")
    print(f"Agent 1 fallback rate: {summary['agent1_fallback_rate']:.1%}")
    low = summary["low_confidence_cases_top1_below_0.3"]
    print(f"low-confidence cases (top-1 < {LOW_CONFIDENCE_THRESHOLD}): "
          f"FULL={len(low['full'])}, NULL={len(low['null'])}")
    if low["full"] or low["null"]:
        print(f"  FULL case_ids: {low['full']}")
        print(f"  NULL case_ids: {low['null']}")
    print("=" * 72)


def run_evaluation(sample_size: int) -> None:
    """Run both retrieval conditions over the sample and write outputs."""
    from tqdm import tqdm

    sample = load_stratified_sample(TEST_CSV, sample_size)

    retriever = Agent2Retriever()
    formulator = Agent1QueryFormulator()
    doc_paths: dict[int, str] = (
        retriever.passages.drop_duplicates("doc_id")
        .set_index("doc_id")["relative_path"]
        .astype(str)
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="cases"):
        case_text = str(row.text)

        formulated = formulator.formulate(case_text)
        full_hits = retriever.retrieve(
            formulated.query, top_k=TOP_K, insurance_type=formulated.insurance_type
        )
        null_hits = retriever.retrieve(case_text, top_k=TOP_K)

        record: dict[str, Any] = {
            "case_id": int(row.case_id),
            "label": str(row.label),
            "appeal_type": str(row.appeal_type),
            "agent1_query": formulated.query,
            "agent1_insurance_type": formulated.insurance_type,
            "agent1_used_fallback": formulated.used_fallback,
        }
        record.update({f"full_{k}": v for k, v in summarise_hits(full_hits, doc_paths).items()})
        record.update({f"null_{k}": v for k, v in summarise_hits(null_hits, doc_paths).items()})
        rows.append(record)

    results = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    logger.info("Wrote %s (%d rows)", RESULTS_CSV, len(results))

    summary = compute_summary(results)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", SUMMARY_JSON)

    print_summary(summary)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Retrieval proxy evaluation (A1 vs A1-null)")
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    run_evaluation(args.sample_size)


if __name__ == "__main__":
    main()
