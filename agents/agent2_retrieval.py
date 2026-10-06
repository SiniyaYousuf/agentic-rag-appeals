"""Agent 2 — Evidence Retrieval & Summarisation.

Pipeline stage 2 of the three-agent RAG system:

1. (build time)  Chunk the working KB (``data/raw/hicric_kb_working.csv``)
   into overlapping word-window passages, embed them with
   ``sentence-transformers/all-MiniLM-L6-v2`` (unit-normalised), and store
   them in a FAISS ``IndexFlatIP`` (inner product == cosine similarity).
2. (query time)  Tag-filter the KB to a subset of documents (jurisdiction /
   insurance_type tags forwarded by Agent 1), then run dense retrieval over
   the filtered passages and return the top-k passages with doc IDs.

CLI usage::

    python agents/agent2_retrieval.py build
    python agents/agent2_retrieval.py query "Is proton beam therapy medically necessary?" --tags cms aca

Artifacts written to ``indexes/faiss_index/``:

- ``kb_index.faiss``     — FAISS IndexFlatIP over passage embeddings
- ``passages.parquet``   — passage text + doc-level metadata (one row per passage)
- ``index_config.json``  — model name, chunk parameters, corpus statistics
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (see CLAUDE.md — Key Design Decisions)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "hicric_kb_working.csv"
INDEX_DIR = PROJECT_ROOT / "indexes" / "faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# all-MiniLM-L6-v2 truncates input at 256 word-piece tokens, so passages are
# capped at ~180 words (~230 tokens) with a 30-word overlap between windows.
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

DEFAULT_TOP_K = 5
# Tag filtering aims for ~400-800 candidate docs; below this floor the filter
# is considered too aggressive and retrieval falls back to the full KB.
MIN_FILTERED_DOCS = 100

# Configs whose docs always stay in the candidate pool when a tag filter is
# active. Clinical guidelines carry no insurance/jurisdiction tags, but they
# are the primary evidence for medical-necessity appeals (~75% of cases), so
# an insurance-type filter must not exclude them.
ALWAYS_INCLUDE_CONFIGS: tuple[str, ...] = ("clinical-guidelines",)

# Insurance-type metadata (from Agent 1) mapped onto hicric doc tags.
INSURANCE_TYPE_TO_TAGS: dict[str, list[str]] = {
    "medicaid": ["medicaid", "chip"],
    "medicare": ["medicare", "cms"],
    "commercial": ["aca", "erisa", "dol", "cms"],
    "private": ["aca", "erisa", "dol", "cms"],
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Args:
        text: Raw document text.
        chunk_words: Maximum number of whitespace-delimited words per passage.
        overlap_words: Number of words shared between consecutive passages.

    Returns:
        List of passage strings (empty list for blank input).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]

    passages: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        # Skip a trailing runt window that is pure overlap of the previous one.
        if start > 0 and len(window) <= overlap_words:
            break
        passages.append(" ".join(window))
    return passages


def parse_tags(raw_tags: Any) -> list[str]:
    """Normalise a tags cell (numpy array, list, or CSV-string repr) to a list."""
    if isinstance(raw_tags, (list, tuple, np.ndarray)):
        return [str(tag) for tag in raw_tags]
    if isinstance(raw_tags, str):
        # CSV round-trips numpy arrays as e.g. "['cms' 'kb' 'regulatory-guidance']"
        try:
            return [str(tag) for tag in ast.literal_eval(raw_tags.replace("' '", "', '"))]
        except (ValueError, SyntaxError):
            return raw_tags.split()
    return []


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def load_kb(kb_csv_path: Path = KB_CSV_PATH) -> pd.DataFrame:
    """Load the working KB CSV and normalise the tags column to lists."""
    if not kb_csv_path.exists():
        raise FileNotFoundError(
            f"KB file not found: {kb_csv_path}. Run data/download_kb.py first."
        )
    kb_frame = pd.read_csv(kb_csv_path)
    before_dedup = len(kb_frame)
    kb_frame = kb_frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    kb_frame["tags"] = kb_frame["tags"].apply(parse_tags)
    kb_frame["doc_id"] = kb_frame.index.astype(int)
    logger.info(
        "Loaded KB: %d documents from %s (%d exact-duplicate texts dropped)",
        len(kb_frame),
        kb_csv_path,
        before_dedup - len(kb_frame),
    )
    return kb_frame


def build_passage_frame(kb_frame: pd.DataFrame) -> pd.DataFrame:
    """Explode KB documents into one row per passage, preserving doc metadata."""
    records: list[dict[str, Any]] = []
    for row in kb_frame.itertuples(index=False):
        for passage_num, passage in enumerate(chunk_text(row.text)):
            records.append(
                {
                    "passage_id": len(records),
                    "doc_id": int(row.doc_id),
                    "passage_num": passage_num,
                    "passage_text": passage,
                    "tags": list(row.tags),
                    "config": getattr(row, "config", ""),
                    "source_url": getattr(row, "source_url", ""),
                    "relative_path": getattr(row, "relative_path", ""),
                }
            )
    passage_frame = pd.DataFrame.from_records(records)
    logger.info(
        "Chunked %d docs into %d passages (chunk=%d words, overlap=%d)",
        kb_frame["doc_id"].nunique(),
        len(passage_frame),
        CHUNK_WORDS,
        CHUNK_OVERLAP_WORDS,
    )
    return passage_frame


def embed_texts(texts: Sequence[str], model: Any = None, batch_size: int = 64) -> np.ndarray:
    """Embed texts with all-MiniLM-L6-v2, unit-normalised for cosine/IP search."""
    if model is None:
        model = load_embedding_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_embedding_model() -> Any:
    """Load the sentence-transformers embedding model (torch backend).

    Benchmarked 2026-07-15 on the build machine: torch fp32 = 32 passages/s,
    ONNX fp32 = 23/s, ONNX quint8 = 38/s but cos-vs-fp32 only 0.981 — torch
    is the best exact backend, so no ONNX.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_index(
    kb_csv_path: Path = KB_CSV_PATH,
    index_dir: Path = INDEX_DIR,
    shard_size: int = 20_000,
) -> None:
    """Build the FAISS IndexFlatIP over KB passage embeddings and save it.

    The corpus is embedded in shards of ``shard_size`` passages, with each
    shard checkpointed to ``index_dir/shards/`` as a ``.npy`` file. If the
    build is interrupted, re-running resumes from the last completed shard.

    Writes ``kb_index.faiss``, ``passages.parquet``, and ``index_config.json``
    into ``index_dir``.
    """
    import time

    kb_frame = load_kb(kb_csv_path)
    passage_frame = build_passage_frame(kb_frame)

    index_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = index_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    # Persist passage metadata up-front so shards can be validated against it.
    passage_frame.to_parquet(index_dir / "passages.parquet", index=False)

    num_passages = len(passage_frame)
    num_shards = (num_passages + shard_size - 1) // shard_size
    model: Any = None
    build_start = time.perf_counter()
    embedded_now = 0

    for shard_num in range(num_shards):
        shard_path = shard_dir / f"embeddings_{shard_num:05d}.npy"
        start_row = shard_num * shard_size
        end_row = min(start_row + shard_size, num_passages)
        if shard_path.exists():
            logger.info("Shard %d/%d already embedded; skipping", shard_num + 1, num_shards)
            continue
        if model is None:
            model = load_embedding_model()
        shard_texts = passage_frame["passage_text"].iloc[start_row:end_row].tolist()
        shard_embeddings = embed_texts(shard_texts, model=model)
        # Write atomically so an interrupt cannot leave a truncated shard.
        # np.save appends ".npy" to bare paths, so pass an open handle instead.
        temp_path = shard_path.with_suffix(".npy.tmp")
        with open(temp_path, "wb") as temp_file:
            np.save(temp_file, shard_embeddings)
        temp_path.replace(shard_path)

        embedded_now += len(shard_texts)
        rate = embedded_now / (time.perf_counter() - build_start)
        remaining = num_passages - end_row
        logger.info(
            "Shard %d/%d done (%d/%d passages, %.0f passages/s, ~%.0f min left)",
            shard_num + 1,
            num_shards,
            end_row,
            num_passages,
            rate,
            remaining / rate / 60 if rate > 0 else -1,
        )

    # Assemble the FAISS index from all shards.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    for shard_num in range(num_shards):
        shard = np.load(shard_dir / f"embeddings_{shard_num:05d}.npy")
        index.add(np.ascontiguousarray(shard, dtype=np.float32))
    if index.ntotal != num_passages:
        raise RuntimeError(
            f"Index has {index.ntotal} vectors but corpus has {num_passages} passages"
        )
    faiss.write_index(index, str(index_dir / "kb_index.faiss"))

    config = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "index_type": "IndexFlatIP",
        "normalized": True,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "num_documents": int(kb_frame["doc_id"].nunique()),
        "num_passages": num_passages,
        "kb_csv": str(kb_csv_path),
    }
    (index_dir / "index_config.json").write_text(json.dumps(config, indent=2))
    logger.info(
        "Saved FAISS index (%d vectors) and metadata to %s", index.ntotal, index_dir
    )


# ---------------------------------------------------------------------------
# Query-time retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievedPassage:
    """A single retrieval hit returned by Agent 2."""

    passage_id: int
    doc_id: int
    score: float
    passage_text: str
    tags: list[str] = field(default_factory=list)
    config: str = ""
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "passage_text": self.passage_text,
            "tags": self.tags,
            "config": self.config,
            "source_url": self.source_url,
        }


class Agent2Retriever:
    """Evidence retriever: tag-first filtering + FAISS dense retrieval.

    Args:
        index_dir: Directory containing ``kb_index.faiss`` and
            ``passages.parquet`` (created by :func:`build_index`).
        model: Optional pre-loaded SentenceTransformer (loaded lazily if None).
    """

    def __init__(self, index_dir: Path = INDEX_DIR, model: Any = None) -> None:
        index_path = index_dir / "kb_index.faiss"
        passages_path = index_dir / "passages.parquet"
        if not index_path.exists() or not passages_path.exists():
            raise FileNotFoundError(
                f"Index artifacts missing in {index_dir}. "
                "Run: python agents/agent2_retrieval.py build"
            )
        self.index: faiss.IndexFlatIP = faiss.read_index(str(index_path))
        self.passages: pd.DataFrame = pd.read_parquet(passages_path)
        self.passages["tags"] = self.passages["tags"].apply(parse_tags)
        self._model = model
        logger.info(
            "Loaded FAISS index with %d passages over %d docs",
            self.index.ntotal,
            self.passages["doc_id"].nunique(),
        )

    @property
    def model(self) -> Any:
        """Lazily-loaded embedding model."""
        if self._model is None:
            self._model = load_embedding_model()
        return self._model

    # -- tag filtering ------------------------------------------------------

    def resolve_filter_tags(
        self,
        insurance_type: Optional[str] = None,
        extra_tags: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Map Agent 1 metadata (insurance_type + free tags) to hicric doc tags."""
        filter_tags: list[str] = []
        if insurance_type:
            filter_tags.extend(
                INSURANCE_TYPE_TO_TAGS.get(insurance_type.strip().lower(), [])
            )
        if extra_tags:
            filter_tags.extend(tag.strip().lower() for tag in extra_tags)
        return sorted(set(filter_tags))

    def _candidate_passage_ids(
        self,
        filter_tags: Sequence[str],
        always_include_configs: Sequence[str] = ALWAYS_INCLUDE_CONFIGS,
    ) -> Optional[np.ndarray]:
        """Return passage ids whose doc tags intersect ``filter_tags``.

        Passages from ``always_include_configs`` (e.g. clinical guidelines,
        which carry no insurance tags) always stay in the candidate pool.
        Returns None (meaning "search the whole KB") when no tags are given or
        when the tag filter alone matches fewer than ``MIN_FILTERED_DOCS``
        documents.
        """
        if not filter_tags:
            return None
        tag_set = set(filter_tags)
        tag_mask = self.passages["tags"].apply(lambda tags: bool(tag_set & set(tags)))
        num_tag_docs = self.passages.loc[tag_mask, "doc_id"].nunique()
        if num_tag_docs < MIN_FILTERED_DOCS:
            logger.warning(
                "Tag filter %s matched only %d docs (< %d); falling back to full KB",
                sorted(tag_set),
                num_tag_docs,
                MIN_FILTERED_DOCS,
            )
            return None
        config_mask = self.passages["config"].isin(set(always_include_configs))
        matched = self.passages[tag_mask | config_mask]
        logger.info(
            "Tag filter %s: %d tag-matched docs; candidate pool %d docs / %d passages "
            "(always-include configs: %s)",
            sorted(tag_set),
            num_tag_docs,
            matched["doc_id"].nunique(),
            len(matched),
            list(always_include_configs),
        )
        return matched["passage_id"].to_numpy(dtype=np.int64)

    # -- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        insurance_type: Optional[str] = None,
        filter_tags: Optional[Sequence[str]] = None,
        max_passages_per_doc: Optional[int] = None,
        always_include_configs: Sequence[str] = ALWAYS_INCLUDE_CONFIGS,
    ) -> list[RetrievedPassage]:
        """Retrieve the top-k evidence passages for a formulated query.

        Args:
            query: Compact retrieval query produced by Agent 1.
            top_k: Number of passages to return.
            insurance_type: Optional Agent 1 metadata (e.g. "medicaid").
            filter_tags: Optional explicit hicric tags to filter on.
            max_passages_per_doc: If set, cap hits per document to diversify
                evidence across documents.
            always_include_configs: Configs whose passages bypass the tag
                filter (default: clinical guidelines). Pass ``()`` to apply
                the tag filter strictly (ablation).

        Returns:
            Ranked list of :class:`RetrievedPassage` (highest cosine first).
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        resolved_tags = self.resolve_filter_tags(insurance_type, filter_tags)
        candidate_ids = self._candidate_passage_ids(resolved_tags, always_include_configs)

        query_embedding = np.asarray(
            self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )

        # Over-fetch when capping per-doc so we can still fill top_k.
        fetch_k = top_k if max_passages_per_doc is None else top_k * 5
        search_params = None
        if candidate_ids is not None:
            fetch_k = min(fetch_k, len(candidate_ids))
            search_params = faiss.SearchParameters(
                sel=faiss.IDSelectorBatch(candidate_ids)
            )
        scores, passage_ids = self.index.search(
            query_embedding, fetch_k, params=search_params
        )

        results: list[RetrievedPassage] = []
        per_doc_counts: dict[int, int] = {}
        for score, passage_id in zip(scores[0], passage_ids[0]):
            if passage_id < 0:  # FAISS pads missing results with -1
                continue
            row = self.passages.iloc[int(passage_id)]
            doc_id = int(row["doc_id"])
            if max_passages_per_doc is not None:
                if per_doc_counts.get(doc_id, 0) >= max_passages_per_doc:
                    continue
                per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
            results.append(
                RetrievedPassage(
                    passage_id=int(passage_id),
                    doc_id=doc_id,
                    score=float(score),
                    passage_text=str(row["passage_text"]),
                    tags=list(row["tags"]),
                    config=str(row.get("config", "")),
                    source_url=str(row.get("source_url", "")),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def format_evidence_summary(self, results: Sequence[RetrievedPassage]) -> str:
        """Format retrieval hits as the evidence block consumed by Agent 3."""
        lines = []
        for rank, hit in enumerate(results, start=1):
            lines.append(
                f"[Evidence {rank} | doc {hit.doc_id} | score {hit.score:.3f}] "
                f"{hit.passage_text}"
            )
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: ``build`` the index or run a test ``query``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Agent 2 — Evidence Retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and save the FAISS index")
    build_parser.add_argument("--kb-csv", type=Path, default=KB_CSV_PATH)
    build_parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)

    query_parser = subparsers.add_parser("query", help="Run a retrieval query")
    query_parser.add_argument("query", type=str)
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query_parser.add_argument("--insurance-type", type=str, default=None)
    query_parser.add_argument("--tags", nargs="*", default=None)
    query_parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)

    args = parser.parse_args()
    if args.command == "build":
        build_index(kb_csv_path=args.kb_csv, index_dir=args.index_dir)
    elif args.command == "query":
        retriever = Agent2Retriever(index_dir=args.index_dir)
        hits = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            insurance_type=args.insurance_type,
            filter_tags=args.tags,
        )
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))


if __name__ == "__main__":
    main()
