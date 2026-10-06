"""Tests for Agent 2 — chunking, tag handling, and FAISS retrieval logic.

The retrieval tests use a stub embedding model over a tiny synthetic index,
so they run offline without downloading all-MiniLM-L6-v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pandas as pd
import pytest

from agents.agent2_retrieval import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    EMBEDDING_DIM,
    Agent2Retriever,
    chunk_text,
    parse_tags,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_and_blank_input(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []
        assert chunk_text(None) == []  # type: ignore[arg-type]

    def test_short_text_single_passage(self) -> None:
        text = "prior authorization denied for MRI"
        assert chunk_text(text) == [text]

    def test_long_text_windows_and_overlap(self) -> None:
        words = [f"word{i}" for i in range(500)]
        passages = chunk_text(" ".join(words))
        assert len(passages) > 1
        # Every passage respects the word cap.
        assert all(len(p.split()) <= CHUNK_WORDS for p in passages)
        # Consecutive passages share exactly the overlap window.
        first_words = passages[0].split()
        second_words = passages[1].split()
        assert first_words[-CHUNK_OVERLAP_WORDS:] == second_words[:CHUNK_OVERLAP_WORDS]

    def test_full_coverage_no_words_lost(self) -> None:
        words = [f"w{i}" for i in range(1000)]
        passages = chunk_text(" ".join(words))
        recovered = set()
        for passage in passages:
            recovered.update(passage.split())
        assert recovered == set(words)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_words=10, overlap_words=10)


# ---------------------------------------------------------------------------
# parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_numpy_array(self) -> None:
        assert parse_tags(np.array(["cms", "kb"])) == ["cms", "kb"]

    def test_python_list(self) -> None:
        assert parse_tags(["medicaid", "kb"]) == ["medicaid", "kb"]

    def test_csv_numpy_repr_string(self) -> None:
        # numpy arrays round-trip through CSV without commas.
        assert parse_tags("['cms' 'kb' 'regulatory-guidance']") == [
            "cms",
            "kb",
            "regulatory-guidance",
        ]

    def test_non_sequence(self) -> None:
        assert parse_tags(None) == []
        assert parse_tags(1.5) == []


# ---------------------------------------------------------------------------
# Agent2Retriever on a synthetic index
# ---------------------------------------------------------------------------


class StubModel:
    """Deterministic stand-in for SentenceTransformer.

    Maps each text to a fixed unit vector based on which keyword it contains,
    so nearest-neighbour results are fully predictable.
    """

    KEYWORD_AXIS = {"cardiology": 0, "oncology": 1, "physiotherapy": 2}

    def encode(self, texts: Sequence[str], **_kwargs) -> np.ndarray:
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for keyword, axis in self.KEYWORD_AXIS.items():
                if keyword in text.lower():
                    vectors[row, axis] = 1.0
            if not vectors[row].any():
                vectors[row, 3] = 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


@pytest.fixture()
def synthetic_index_dir(tmp_path: Path) -> Path:
    """Build a small on-disk index: 3 docs x ~2 passages with distinct topics."""
    passages = pd.DataFrame(
        [
            {"passage_id": 0, "doc_id": 0, "passage_num": 0,
             "passage_text": "cardiology stent coverage criteria",
             "tags": ["medicare", "kb"], "config": "regulatory-guidance",
             "source_url": "", "relative_path": ""},
            {"passage_id": 1, "doc_id": 0, "passage_num": 1,
             "passage_text": "cardiology rehabilitation guidance",
             "tags": ["medicare", "kb"], "config": "regulatory-guidance",
             "source_url": "", "relative_path": ""},
            {"passage_id": 2, "doc_id": 1, "passage_num": 0,
             "passage_text": "oncology chemotherapy medical necessity",
             "tags": ["medicaid", "kb"], "config": "legal",
             "source_url": "", "relative_path": ""},
            {"passage_id": 3, "doc_id": 2, "passage_num": 0,
             "passage_text": "physiotherapy session limits",
             "tags": ["aca", "kb"], "config": "clinical-guidelines",
             "source_url": "", "relative_path": ""},
        ]
    )
    embeddings = StubModel().encode(passages["passage_text"].tolist())
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(tmp_path / "kb_index.faiss"))
    passages.to_parquet(tmp_path / "passages.parquet", index=False)
    (tmp_path / "index_config.json").write_text(json.dumps({"num_passages": 4}))
    return tmp_path


class TestAgent2Retriever:
    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Agent2Retriever(index_dir=tmp_path / "nowhere")

    def test_empty_query_raises(self, synthetic_index_dir: Path) -> None:
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        with pytest.raises(ValueError):
            retriever.retrieve("   ")

    def test_top_hit_matches_topic(self, synthetic_index_dir: Path) -> None:
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("oncology treatment denial", top_k=2)
        assert hits[0].doc_id == 1
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_tag_filter_restricts_results(self, synthetic_index_dir: Path, monkeypatch) -> None:
        # Lower the fallback floor so a 1-doc filter is honoured in the test.
        monkeypatch.setattr("agents.agent2_retrieval.MIN_FILTERED_DOCS", 1)
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("cardiology denial", top_k=4, filter_tags=["aca"])
        assert {hit.doc_id for hit in hits} == {2}

    def test_clinical_guidelines_bypass_tag_filter(
        self, synthetic_index_dir: Path, monkeypatch
    ) -> None:
        # Doc 2 (clinical-guidelines) has no medicare tag but must stay in the
        # candidate pool when filtering on medicare.
        monkeypatch.setattr("agents.agent2_retrieval.MIN_FILTERED_DOCS", 1)
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("physiotherapy limits", top_k=1, filter_tags=["medicare"])
        assert hits[0].doc_id == 2
        assert hits[0].config == "clinical-guidelines"

    def test_strict_filter_ablation_excludes_clinical(
        self, synthetic_index_dir: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr("agents.agent2_retrieval.MIN_FILTERED_DOCS", 1)
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve(
            "physiotherapy limits",
            top_k=4,
            filter_tags=["medicare"],
            always_include_configs=(),
        )
        assert all(hit.doc_id == 0 for hit in hits)

    def test_filter_fallback_when_too_few_docs(self, synthetic_index_dir: Path) -> None:
        # Default MIN_FILTERED_DOCS (100) exceeds the synthetic KB, so the
        # filter must fall back to searching the full index.
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("cardiology denial", top_k=1, filter_tags=["aca"])
        assert hits[0].doc_id == 0

    def test_max_passages_per_doc_diversifies(self, synthetic_index_dir: Path) -> None:
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("cardiology denial", top_k=3, max_passages_per_doc=1)
        doc_ids = [hit.doc_id for hit in hits]
        assert len(doc_ids) == len(set(doc_ids))

    def test_evidence_summary_format(self, synthetic_index_dir: Path) -> None:
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        hits = retriever.retrieve("oncology treatment denial", top_k=1)
        summary = retriever.format_evidence_summary(hits)
        assert "[Evidence 1 | doc 1 |" in summary
        assert "oncology" in summary

    def test_resolve_filter_tags_insurance_type(self, synthetic_index_dir: Path) -> None:
        retriever = Agent2Retriever(index_dir=synthetic_index_dir, model=StubModel())
        tags = retriever.resolve_filter_tags(insurance_type="Medicaid")
        assert "medicaid" in tags
        assert retriever.resolve_filter_tags(insurance_type="unknown-type") == []
