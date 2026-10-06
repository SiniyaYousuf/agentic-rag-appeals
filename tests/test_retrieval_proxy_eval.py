"""Tests for the retrieval proxy evaluation's pure functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.retrieval_proxy_eval import compute_summary, load_stratified_sample


@pytest.fixture()
def synthetic_test_csv(tmp_path: Path) -> Path:
    """A small imbalanced 3-class test CSV mirroring imr_test.csv columns."""
    frame = pd.DataFrame(
        {
            "text": [f"case text {i}" for i in range(100)],
            "appeal_type": ["Medical Necessity"] * 100,
            "label": ["Upheld"] * 53 + ["Overturned"] * 45 + ["Insufficient"] * 2,
        }
    )
    path = tmp_path / "test.csv"
    frame.to_csv(path, index=False)
    return path


class TestLoadStratifiedSample:
    def test_exact_sample_size(self, synthetic_test_csv: Path) -> None:
        sample = load_stratified_sample(synthetic_test_csv, 20)
        assert len(sample) == 20

    def test_proportions_roughly_preserved(self, synthetic_test_csv: Path) -> None:
        sample = load_stratified_sample(synthetic_test_csv, 20)
        counts = sample["label"].value_counts()
        assert counts["Upheld"] >= counts["Overturned"] >= counts["Insufficient"]

    def test_minority_class_kept(self, synthetic_test_csv: Path) -> None:
        # 2% of 20 rounds to 0 — every class must still get at least 1 case.
        sample = load_stratified_sample(synthetic_test_csv, 20)
        assert (sample["label"] == "Insufficient").sum() >= 1

    def test_deterministic(self, synthetic_test_csv: Path) -> None:
        first = load_stratified_sample(synthetic_test_csv, 20)
        second = load_stratified_sample(synthetic_test_csv, 20)
        assert first["case_id"].tolist() == second["case_id"].tolist()

    def test_case_id_is_original_row_index(self, synthetic_test_csv: Path) -> None:
        sample = load_stratified_sample(synthetic_test_csv, 20)
        original = pd.read_csv(synthetic_test_csv)
        row = sample.iloc[0]
        assert original.loc[row["case_id"], "text"] == row["text"]


class TestComputeSummary:
    @pytest.fixture()
    def results(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "case_id": [0, 1, 2],
                "label": ["Upheld", "Overturned", "Upheld"],
                "agent1_used_fallback": [False, True, False],
                "full_top1_score": [0.7, 0.5, 0.25],
                "full_top5_mean_score": [0.6, 0.45, 0.2],
                "full_top5_doc_ids": ["1;2;3;4;5", "1;6;7;8;9", "10;11;12;13;14"],
                "null_top1_score": [0.6, 0.55, 0.4],
                "null_top5_mean_score": [0.5, 0.5, 0.35],
                "null_top5_doc_ids": ["1;2;3;4;5", "1;2;3;4;5", "1;2;3;4;5"],
            }
        )

    def test_means_and_delta(self, results: pd.DataFrame) -> None:
        summary = compute_summary(results)
        assert summary["mean_top1_score"]["full"] == pytest.approx(0.4833, abs=1e-4)
        assert summary["mean_top1_score"]["null"] == pytest.approx(0.5167, abs=1e-4)
        assert summary["score_delta_full_minus_null"]["top1"] == pytest.approx(
            -0.0333, abs=1e-3
        )

    def test_unique_doc_counts(self, results: pd.DataFrame) -> None:
        summary = compute_summary(results)
        assert summary["unique_kb_docs_retrieved"]["full"] == 14
        assert summary["unique_kb_docs_retrieved"]["null"] == 5

    def test_low_confidence_cases(self, results: pd.DataFrame) -> None:
        summary = compute_summary(results)
        assert summary["low_confidence_cases_top1_below_0.3"]["full"] == [2]
        assert summary["low_confidence_cases_top1_below_0.3"]["null"] == []

    def test_per_class_breakdown(self, results: pd.DataFrame) -> None:
        summary = compute_summary(results)
        assert summary["per_class"]["Upheld"]["n_cases"] == 2
        assert summary["per_class"]["Overturned"]["full_top1_mean"] == 0.5

    def test_fallback_rate(self, results: pd.DataFrame) -> None:
        summary = compute_summary(results)
        assert summary["agent1_fallback_rate"] == pytest.approx(1 / 3, abs=1e-4)
