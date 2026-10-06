"""Tests for Agent 1 — prompt construction, output parsing, and fallbacks.

The formulator tests use a stub generator so they run offline without
downloading google/flan-t5-base.
"""

from __future__ import annotations

from typing import Optional

import pytest

from agents.agent1_query_formulator import (
    DEFAULT_JURISDICTION,
    Agent1QueryFormulator,
    FormulatedQuery,
    build_insurance_prompt,
    build_query_prompt,
    fallback_query,
    is_degenerate_query,
    parse_insurance_type,
)


class StubFormulator(Agent1QueryFormulator):
    """Formulator whose generation is scripted per prompt type."""

    def __init__(
        self,
        query_output: str,
        insurance_output: str,
        infer_insurance_type: bool = True,
    ) -> None:
        super().__init__(
            model=object(),
            tokenizer=object(),
            infer_insurance_type=infer_insurance_type,
        )
        self._query_output = query_output
        self._insurance_output = insurance_output

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        if prompt.endswith("Search query:"):
            return self._query_output
        if prompt.endswith("Answer:"):
            return self._insurance_output
        raise AssertionError(f"unexpected prompt ending: {prompt[-40:]!r}")


CASE = (
    "The patient is a 54-year-old male with chronic lumbar pain. His health "
    "plan denied an MRI of the lumbar spine as not medically necessary, "
    "stating that conservative therapy had not been exhausted."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_query_prompt_contains_case_and_instruction(self) -> None:
        prompt = build_query_prompt(CASE)
        assert CASE in prompt
        assert "Search query:" in prompt
        assert "reason for denial" in prompt

    def test_insurance_prompt_lists_closed_label_set(self) -> None:
        prompt = build_insurance_prompt(CASE)
        assert CASE in prompt
        for label in ("medicaid", "medicare", "commercial", "unknown"):
            assert label in prompt

    def test_prompts_strip_whitespace(self) -> None:
        assert build_query_prompt("  text  ").count("  text  ") == 0


# ---------------------------------------------------------------------------
# parse_insurance_type
# ---------------------------------------------------------------------------


class TestParseInsuranceType:
    @pytest.mark.parametrize(
        ("generated", "expected"),
        [
            ("medicaid", "medicaid"),
            ("Medicare.", "medicare"),
            ("  COMMERCIAL ", "commercial"),
            ("unknown", "unknown"),
            ("The patient has medicaid coverage", "medicaid"),
            ("", "unknown"),
            ("private insurance", "unknown"),
            ("HMO", "unknown"),
        ],
    )
    def test_maps_to_closed_label_set(self, generated: str, expected: str) -> None:
        assert parse_insurance_type(generated) == expected


# ---------------------------------------------------------------------------
# is_degenerate_query — cases mirror real FLAN-T5-base failures observed
# during prompt iteration on train cases (2026-07-17)
# ---------------------------------------------------------------------------


class TestIsDegenerateQuery:
    def test_verbatim_copy_is_benign(self) -> None:
        # Parroting the case prefix is fine for dense retrieval.
        assert is_degenerate_query(" ".join(CASE.split()[:30]), CASE) is False

    def test_too_short(self) -> None:
        assert is_degenerate_query("Drug overdose", CASE) is True

    def test_repetition_loop(self) -> None:
        looped = (
            "speech therapy is denied because the patient is experiencing a "
            "developmental delay and therefore, " * 4
        )
        assert is_degenerate_query(looped.strip(), CASE) is True

    def test_ungrounded_generic_sentence(self) -> None:
        assert is_degenerate_query(
            "The case is a health insurance appeal about coverage.", CASE
        ) is True

    def test_grounded_paraphrase_is_fine(self) -> None:
        assert is_degenerate_query(
            "MRI lumbar spine denied medically necessary conservative therapy",
            CASE,
        ) is False


# ---------------------------------------------------------------------------
# fallback_query
# ---------------------------------------------------------------------------


class TestFallbackQuery:
    def test_truncates_to_max_words(self) -> None:
        text = " ".join(f"w{i}" for i in range(200))
        assert fallback_query(text, max_words=60).split() == [f"w{i}" for i in range(60)]

    def test_short_text_unchanged(self) -> None:
        assert fallback_query("short case text") == "short case text"


# ---------------------------------------------------------------------------
# Agent1QueryFormulator.formulate (stubbed generation)
# ---------------------------------------------------------------------------


class TestFormulate:
    def test_happy_path(self) -> None:
        formulator = StubFormulator(
            query_output="MRI lumbar spine denial medical necessity chronic pain",
            insurance_output="commercial",
        )
        result = formulator.formulate(CASE)
        assert isinstance(result, FormulatedQuery)
        assert result.query.startswith("MRI lumbar spine")
        assert result.insurance_type == "commercial"
        assert result.jurisdiction == DEFAULT_JURISDICTION
        assert result.used_fallback is False

    def test_degenerate_query_triggers_fallback(self) -> None:
        formulator = StubFormulator(query_output="  no ", insurance_output="unknown")
        result = formulator.formulate(CASE)
        assert result.used_fallback is True
        assert result.query == fallback_query(CASE)
        assert len(result.query.split()) >= 3

    def test_empty_case_text_raises(self) -> None:
        formulator = StubFormulator(query_output="x", insurance_output="unknown")
        with pytest.raises(ValueError):
            formulator.formulate("   ")

    def test_to_dict_is_json_serialisable(self) -> None:
        formulator = StubFormulator(
            query_output="MRI lumbar spine denied conservative therapy not exhausted",
            insurance_output="medicare",
        )
        payload = formulator.formulate(CASE).to_dict()
        assert payload == {
            "query": "MRI lumbar spine denied conservative therapy not exhausted",
            "insurance_type": "medicare",
            "jurisdiction": DEFAULT_JURISDICTION,
            "used_fallback": False,
        }

    def test_insurance_inference_off_by_default_uses_commercial(self) -> None:
        formulator = StubFormulator(
            query_output="MRI lumbar spine denial",
            insurance_output="SHOULD NEVER BE GENERATED",
            infer_insurance_type=False,
        )
        result = formulator.formulate(CASE)
        assert result.insurance_type == "commercial"

    def test_default_construction_does_not_infer(self) -> None:
        formulator = Agent1QueryFormulator(model=object(), tokenizer=object())
        assert formulator.infer_insurance_type is False
        assert formulator.default_insurance_type == "commercial"

    def test_invalid_default_insurance_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Agent1QueryFormulator(default_insurance_type="hmo")

    def test_formulate_batch_preserves_order(self) -> None:
        formulator = StubFormulator(
            query_output="stable generated query text",
            insurance_output="medicaid",
        )
        results = formulator.formulate_batch([CASE, CASE + " extra"])
        assert len(results) == 2
        assert all(r.insurance_type == "medicaid" for r in results)
        assert results[1].case_text.endswith("extra")
