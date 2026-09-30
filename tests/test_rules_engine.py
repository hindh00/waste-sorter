"""
Unit tests for rules_engine.py. No dataset, no trained weights, no network
required — these should pass right after `pip install -r requirements.txt`.
"""

import textwrap

import pytest

from waste_sorter.rules_engine import InvalidRulesError, RulesEngine


def write_rules(tmp_path, content: str):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(content))
    return path


def test_loads_valid_rules(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories:
          can:
            bin: "Metal Recycling"
            steps: ["Rinse it out."]
    """)
    engine = RulesEngine(path)
    assert engine.confidence_threshold == 0.5
    assert engine.known_categories() == ["can"]


def test_lookup_known_category(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories:
          can:
            bin: "Metal Recycling"
            steps: ["Rinse it out.", "Crush it."]
    """)
    engine = RulesEngine(path)
    result = engine.lookup("can")
    assert result is not None
    assert result.bin == "Metal Recycling"
    assert result.steps == ["Rinse it out.", "Crush it."]


def test_lookup_unknown_category_returns_none(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories:
          can:
            bin: "Metal Recycling"
            steps: ["Rinse it out."]
    """)
    engine = RulesEngine(path)
    assert engine.lookup("spaceship_part") is None


def test_missing_confidence_threshold_raises(tmp_path):
    path = write_rules(tmp_path, """
        categories:
          can:
            bin: "Metal Recycling"
            steps: ["Rinse it out."]
    """)
    with pytest.raises(InvalidRulesError):
        RulesEngine(path)


def test_out_of_range_threshold_raises(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 1.5
        categories:
          can:
            bin: "Metal Recycling"
            steps: ["Rinse it out."]
    """)
    with pytest.raises(InvalidRulesError):
        RulesEngine(path)


def test_category_missing_bin_raises(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories:
          can:
            steps: ["Rinse it out."]
    """)
    with pytest.raises(InvalidRulesError):
        RulesEngine(path)


def test_category_empty_steps_raises(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories:
          can:
            bin: "Metal Recycling"
            steps: []
    """)
    with pytest.raises(InvalidRulesError):
        RulesEngine(path)


def test_empty_categories_raises(tmp_path):
    path = write_rules(tmp_path, """
        confidence_threshold: 0.5
        categories: {}
    """)
    with pytest.raises(InvalidRulesError):
        RulesEngine(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RulesEngine(tmp_path / "does_not_exist.yaml")


def test_real_rules_file_loads():
    """Sanity check against the actual project rules.yaml."""
    engine = RulesEngine("rules.yaml")
    assert 0.0 <= engine.confidence_threshold <= 1.0
    assert "other_unlabeled" in engine.known_categories()
    for category in engine.known_categories():
        rule = engine.lookup(category)
        assert rule.bin
        assert rule.steps
