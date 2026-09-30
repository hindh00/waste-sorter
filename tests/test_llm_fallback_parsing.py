"""
Tests for the pure JSON-parsing part of llm_fallback.py. No network call,
no API key required — _parse_llm_json never touches the `anthropic` package.
"""

from waste_sorter.llm_fallback import _parse_llm_json

KNOWN = ["can", "bottle_plastic", "other_unlabeled"]


def test_parses_clean_json():
    text = '{"category": "can", "confidence": 0.9, "reasoning": "Looks like an aluminum can."}'
    result = _parse_llm_json(text, KNOWN)
    assert result.parse_ok
    assert result.category == "can"
    assert result.confidence == 0.9
    assert "aluminum" in result.reasoning


def test_parses_json_wrapped_in_prose():
    text = 'Sure, here is my answer:\n{"category": "bottle_plastic", "confidence": 0.7, "reasoning": "clear bottle"}\nHope that helps!'
    result = _parse_llm_json(text, KNOWN)
    assert result.parse_ok
    assert result.category == "bottle_plastic"


def test_parses_json_in_code_fence():
    text = '```json\n{"category": "can", "confidence": 0.8, "reasoning": "metallic"}\n```'
    result = _parse_llm_json(text, KNOWN)
    assert result.parse_ok
    assert result.category == "can"


def test_empty_string_degrades_gracefully():
    result = _parse_llm_json("", KNOWN)
    assert not result.parse_ok
    assert result.category == "other_unlabeled"
    assert result.confidence == 0.0


def test_prose_only_degrades_gracefully():
    result = _parse_llm_json("I think this is a can, but I'm not sure how to format that.", KNOWN)
    assert not result.parse_ok
    assert result.category == "other_unlabeled"


def test_truncated_json_degrades_gracefully():
    result = _parse_llm_json('{"category": "can", "confidence": 0.', KNOWN)
    assert not result.parse_ok
    assert result.category == "other_unlabeled"


def test_wrong_types_do_not_crash():
    # confidence="high" fails float() conversion; the whole result should
    # degrade gracefully rather than raise or return partial garbage.
    text = '{"category": 123, "confidence": "high", "reasoning": null}'
    result = _parse_llm_json(text, KNOWN)
    assert not result.parse_ok
    assert result.category == "other_unlabeled"


def test_confidence_out_of_range_is_clamped():
    text = '{"category": "can", "confidence": 5.0, "reasoning": "very sure"}'
    result = _parse_llm_json(text, KNOWN)
    assert result.parse_ok
    assert result.confidence == 1.0


def test_unknown_category_is_preserved_not_forced_to_other():
    """An LLM proposing a brand-new category name should come through as-is
    so the caller/logger can flag it as a candidate new class, rather than
    being silently coerced to 'other_unlabeled'."""
    text = '{"category": "coffee_pod", "confidence": 0.6, "reasoning": "looks like a coffee capsule"}'
    result = _parse_llm_json(text, KNOWN)
    assert result.parse_ok
    assert result.category == "coffee_pod"
