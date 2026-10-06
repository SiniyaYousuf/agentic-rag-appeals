"""Agent 1 — Query Formulator.

Pipeline stage 1 of the three-agent RAG system: given a raw appeal case
description (the ``text`` column of Persius/imr-appeals), produce

1. a compact 1-2 sentence retrieval query for Agent 2's dense retriever, and
2. metadata tags (``jurisdiction``, ``insurance_type``) forwarded to Agent 2
   for tag-first KB filtering.

Uses zero-shot ``google/flan-t5-base`` (no fine-tuning, per the project's
design decision to avoid NER annotation overhead). Generation is deterministic
(beam search, no sampling) so pipeline runs are reproducible.

CLI usage::

    python agents/agent1_query_formulator.py "MRI of the lumbar spine was denied as not medically necessary..."
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-base"

# The labelled benchmark (Persius/imr-appeals) is California Independent
# Medical Review data, so jurisdiction is constant for this corpus.
DEFAULT_JURISDICTION = "california"

VALID_INSURANCE_TYPES = ("medicaid", "medicare", "commercial", "unknown")

QUERY_PROMPT_TEMPLATE = (
    "Summarize this health insurance appeal case as a short search query "
    "(one or two sentences) for finding relevant medical policy and clinical "
    "guidelines. Mention the treatment, the condition, and the reason for "
    "denial.\n\nCase: {case_text}\n\nSearch query:"
)

INSURANCE_PROMPT_TEMPLATE = (
    "What type of health insurance does the patient in this case have? "
    "Answer with exactly one of: medicaid, medicare, commercial, unknown.\n\n"
    "Case: {case_text}\n\nAnswer:"
)

# FLAN-T5 was trained with 512-token inputs; case texts are truncated to fit.
MAX_INPUT_TOKENS = 512
MAX_QUERY_TOKENS = 64
MAX_ANSWER_TOKENS = 4
NUM_BEAMS = 4


@dataclass
class FormulatedQuery:
    """Structured output of Agent 1, consumed by Agent 2."""

    query: str
    insurance_type: str = "unknown"
    jurisdiction: str = DEFAULT_JURISDICTION
    used_fallback: bool = False
    case_text: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "query": self.query,
            "insurance_type": self.insurance_type,
            "jurisdiction": self.jurisdiction,
            "used_fallback": self.used_fallback,
        }


def build_query_prompt(case_text: str) -> str:
    """Render the retrieval-query prompt for a case description."""
    return QUERY_PROMPT_TEMPLATE.format(case_text=case_text.strip())


def build_insurance_prompt(case_text: str) -> str:
    """Render the insurance-type classification prompt for a case description."""
    return INSURANCE_PROMPT_TEMPLATE.format(case_text=case_text.strip())


def parse_insurance_type(generated: str) -> str:
    """Map a raw FLAN-T5 answer onto the closed insurance-type label set.

    Falls back to ``"unknown"`` for anything outside the label set rather
    than propagating free text into Agent 2's tag filter.
    """
    answer = generated.strip().lower().rstrip(".")
    for label in VALID_INSURANCE_TYPES:
        if label in answer:
            return label
    return "unknown"


def is_degenerate_query(query: str, case_text: str) -> bool:
    """Detect generation failure modes observed with zero-shot FLAN-T5-base.

    Prompt iteration on 12 train cases (2026-07-17) showed three failure
    modes worth guarding against (verbatim copying of the case prefix is NOT
    one of them — it is benign for dense retrieval):

    1. Too short to be a usable query (< 5 words).
    2. Repetition loops (the same phrase generated over and over).
    3. Ungrounded output (instruction echo / generic sentence whose content
       words mostly do not occur in the case text).
    """
    words = query.split()
    if len(words) < 5:
        return True

    trigrams = list(zip(words, words[1:], words[2:]))
    if trigrams and len(set(trigrams)) / len(trigrams) < 0.6:
        return True

    def normalise(word: str) -> str:
        return word.lower().strip(".,;:()!?\"'")

    case_words = {normalise(word) for word in case_text.split()}
    content_words = [normalise(word) for word in words if len(normalise(word)) > 3]
    if content_words:
        grounded = sum(word in case_words for word in content_words)
        if grounded / len(content_words) < 0.4:
            return True
    return False


def fallback_query(case_text: str, max_words: int = 60) -> str:
    """Deterministic fallback: the first ``max_words`` words of the case text.

    Used when generation produces an empty or degenerate query so the
    pipeline never forwards an empty string to Agent 2 (this is also the
    A1-null ablation behaviour).
    """
    words = case_text.split()
    return " ".join(words[:max_words])


class Agent1QueryFormulator:
    """Zero-shot FLAN-T5 query formulator.

    Args:
        model: Optional pre-loaded ``transformers`` seq2seq model (loaded
            lazily from ``MODEL_NAME`` when None).
        tokenizer: Optional pre-loaded tokenizer (loaded lazily when None).
        device: Torch device string; auto-detected when None.
        infer_insurance_type: When True, classify insurance type with FLAN-T5.
            Off by default: Persius/imr-appeals is California DMHC IMR data,
            which is commercial managed care by definition, and zero-shot
            FLAN-T5 was observed to guess "medicare" from patient age alone.
            Kept available as an ablation.
        default_insurance_type: Value used when inference is disabled.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: Optional[str] = None,
        infer_insurance_type: bool = False,
        default_insurance_type: str = "commercial",
    ) -> None:
        if default_insurance_type not in VALID_INSURANCE_TYPES:
            raise ValueError(
                f"default_insurance_type must be one of {VALID_INSURANCE_TYPES}"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self.infer_insurance_type = infer_insurance_type
        self.default_insurance_type = default_insurance_type

    def _ensure_loaded(self) -> None:
        """Load the model and tokenizer on first use."""
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s", MODEL_NAME, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self._device)
        self._model.eval()

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Run deterministic beam-search generation for a single prompt."""
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                do_sample=False,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def formulate(self, case_text: str) -> FormulatedQuery:
        """Formulate a retrieval query + metadata for one case description.

        Args:
            case_text: The cleaned background span (``text`` column of
                Persius/imr-appeals).

        Returns:
            A :class:`FormulatedQuery`; never has an empty ``query``.

        Raises:
            ValueError: If ``case_text`` is empty or whitespace.
        """
        if not case_text or not case_text.strip():
            raise ValueError("case_text must be a non-empty string")

        raw_query = self._generate(build_query_prompt(case_text), MAX_QUERY_TOKENS)
        used_fallback = False
        query = raw_query.strip()
        if is_degenerate_query(query, case_text):
            logger.warning(
                "Degenerate generated query %r; falling back to truncated case text",
                query,
            )
            query = fallback_query(case_text)
            used_fallback = True

        if self.infer_insurance_type:
            raw_answer = self._generate(
                build_insurance_prompt(case_text), MAX_ANSWER_TOKENS
            )
            insurance_type = parse_insurance_type(raw_answer)
        else:
            insurance_type = self.default_insurance_type

        return FormulatedQuery(
            query=query,
            insurance_type=insurance_type,
            jurisdiction=DEFAULT_JURISDICTION,
            used_fallback=used_fallback,
            case_text=case_text,
        )

    def formulate_batch(self, case_texts: Sequence[str]) -> list[FormulatedQuery]:
        """Formulate queries for a sequence of cases (simple loop, logged)."""
        results: list[FormulatedQuery] = []
        for case_num, case_text in enumerate(case_texts):
            results.append(self.formulate(case_text))
            if (case_num + 1) % 100 == 0:
                logger.info("Formulated %d/%d queries", case_num + 1, len(case_texts))
        return results


def main() -> None:
    """CLI entry point: formulate a query for one case description."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Agent 1 — Query Formulator")
    parser.add_argument("case_text", type=str, help="Raw case description text")
    args = parser.parse_args()

    formulator = Agent1QueryFormulator()
    result = formulator.formulate(args.case_text)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
