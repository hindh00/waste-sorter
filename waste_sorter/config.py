"""
config.py

Loads config.yaml (runtime/app plumbing: model path, image size, LLM model
id, log path) into one typed settings object. Deliberately separate from
rules_engine.py, which owns rules.yaml (domain rules: bin + prep steps).
Nothing here should ever contain city-specific recycling logic — that
belongs only in rules.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSettings:
    weights_path: Path
    image_size: int


@dataclass(frozen=True)
class LlmSettings:
    enabled: bool
    model: str
    max_tokens: int


@dataclass(frozen=True)
class Settings:
    model: ModelSettings
    llm: LlmSettings
    rules_path: Path
    category_map_path: Path
    fallback_log_path: Path


def load_settings(config_path: str | Path = PROJECT_ROOT / "config.yaml") -> Settings:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return Settings(
        model=ModelSettings(
            weights_path=_resolve(raw["model"]["weights_path"]),
            image_size=int(raw["model"]["image_size"]),
        ),
        llm=LlmSettings(
            enabled=bool(raw["llm"].get("enabled", True)),
            model=raw["llm"]["model"],
            max_tokens=int(raw["llm"]["max_tokens"]),
        ),
        rules_path=_resolve(raw["rules_path"]),
        category_map_path=_resolve(raw["category_map_path"]),
        fallback_log_path=_resolve(raw["fallback_log_path"]),
    )
