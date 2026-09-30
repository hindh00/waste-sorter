"""
rules_engine.py

Loads rules.yaml and answers "what bin does this category go in, and what
should I do before tossing it." Deliberately boring: pure dictionary
lookups, zero conditionals about specific cities or materials. If this file
ever grows an if/elif chain, that logic wants to move into rules.yaml
instead (e.g. as a per-material sub-rule), not into Python.

Retargeting the whole app to a different city's recycling program should
mean editing rules.yaml, never touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class InvalidRulesError(ValueError):
    """Raised when rules.yaml is missing required fields or malformed."""


@dataclass(frozen=True)
class RuleResult:
    category: str
    bin: str
    steps: list[str]


class RulesEngine:
    def __init__(self, rules_path: str | Path = "rules.yaml"):
        rules_path = Path(rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(f"rules.yaml not found at {rules_path}")

        with rules_path.open("r") as f:
            data = yaml.safe_load(f) or {}

        if "categories" not in data or not isinstance(data["categories"], dict):
            raise InvalidRulesError("rules.yaml must have a top-level 'categories' mapping")
        if not data["categories"]:
            raise InvalidRulesError("rules.yaml 'categories' must not be empty")

        threshold = data.get("confidence_threshold")
        if threshold is None:
            raise InvalidRulesError("rules.yaml must set 'confidence_threshold'")
        threshold = float(threshold)
        if not (0.0 <= threshold <= 1.0):
            raise InvalidRulesError(
                f"confidence_threshold must be between 0 and 1, got {threshold}"
            )

        for name, entry in data["categories"].items():
            if not isinstance(entry, dict) or "bin" not in entry or "steps" not in entry:
                raise InvalidRulesError(
                    f"category '{name}' must define both 'bin' and 'steps'"
                )
            if not isinstance(entry["steps"], list) or not entry["steps"]:
                raise InvalidRulesError(
                    f"category '{name}' must have a non-empty list of 'steps'"
                )

        self.confidence_threshold: float = threshold
        self._categories: dict[str, dict] = data["categories"]

    def lookup(self, category: str) -> RuleResult | None:
        """Returns the rule for a category, or None if it isn't known.

        A None return is the caller's signal to fall back (e.g. to the LLM,
        or to the 'other_unlabeled' rule) rather than crash on a KeyError.
        """
        entry = self._categories.get(category)
        if entry is None:
            return None
        return RuleResult(category=category, bin=entry["bin"], steps=list(entry["steps"]))

    def known_categories(self) -> list[str]:
        return list(self._categories.keys())
