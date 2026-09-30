"""
streamlit_app.py

Minimal demo: upload a close-up photo of one item, get told which bin it
goes in and how to prep it. Falls back to Claude when the local model
isn't confident (or doesn't know the category at all), and logs those
cases for future retraining.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow running via `streamlit run app/streamlit_app.py` from the repo root
# without needing the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waste_sorter.classifier import WasteClassifier, WeightsNotFoundError
from waste_sorter.config import Settings, load_settings
from waste_sorter.fallback_logger import log_fallback
from waste_sorter.llm_fallback import classify_with_llm
from waste_sorter.rules_engine import RulesEngine

load_dotenv()

st.set_page_config(page_title="Waste-sorting assistant", page_icon="♻️")


@st.cache_resource
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource
def get_rules(_settings: Settings) -> RulesEngine:
    return RulesEngine(_settings.rules_path)


@st.cache_resource
def get_classifier(_settings: Settings, confidence_threshold: float):
    """Returns (classifier, error) — error is a WeightsNotFoundError instance
    if construction failed, so the caller can render a clear banner instead
    of letting Streamlit show a raw traceback."""
    try:
        return WasteClassifier(_settings.model.weights_path, confidence_threshold), None
    except WeightsNotFoundError as e:
        return None, e


settings = get_settings()
rules = get_rules(settings)
# rules.yaml's confidence_threshold is the single source of truth; the
# classifier just needs the number to compute is_confident.
classifier, weights_error = get_classifier(settings, rules.confidence_threshold)

llm_enabled = settings.llm.enabled
api_key = os.environ.get("ANTHROPIC_API_KEY") if llm_enabled else None

st.title("Waste-sorting assistant")
st.caption("Upload a close-up photo of one item to see which bin it goes in.")

if weights_error is not None:
    st.error(
        "**No trained model yet.**\n\n"
        "This app ships with code only, not a trained classifier. "
        f"{weights_error}"
    )
    st.stop()

if not llm_enabled:
    st.info(
        "AI fallback is disabled (llm.enabled: false in config.yaml). "
        "Low-confidence or unrecognized items will be shown as unsorted "
        "instead of being sent to Claude."
    )
elif not api_key:
    st.warning(
        "ANTHROPIC_API_KEY is not set — the LLM fallback is disabled. "
        "Low-confidence or unrecognized items will not get a fallback answer "
        "until you add a key to .env (see .env.example)."
    )

with st.sidebar:
    st.subheader("Current settings")
    st.write(f"Confidence threshold: **{rules.confidence_threshold:.0%}**")
    st.caption("Edit rules.yaml to change this — it's not editable here on purpose.")
    st.write("Known categories:")
    st.caption(", ".join(rules.known_categories()))

uploaded = st.file_uploader("Item photo", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, width=300)

    temp_path = Path(f"logs/_upload_{uuid.uuid4().hex}.jpg")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded.getvalue())

    try:
        result = classifier.predict(str(temp_path))

        if result.is_confident and rules.lookup(result.category) is not None:
            rule = rules.lookup(result.category)
            st.success(f"**{result.category}** ({result.confidence:.0%} confident) → **{rule.bin}**")
            for step in rule.steps:
                st.write(f"- {step}")

        elif api_key:
            st.info("Not confident locally — asking Claude...")
            llm_result = classify_with_llm(
                str(temp_path),
                known_categories=rules.known_categories(),
                api_key=api_key,
                model=settings.llm.model,
                max_tokens=settings.llm.max_tokens,
            )

            if not llm_result.parse_ok:
                st.warning("Claude's answer couldn't be parsed — treating this as unknown.")

            rule = rules.lookup(llm_result.category) or rules.lookup("other_unlabeled")
            st.success(f"**{llm_result.category}** → **{rule.bin}**")
            for step in rule.steps:
                st.write(f"- {step}")
            if llm_result.reasoning:
                st.caption(f"Claude's reasoning: {llm_result.reasoning}")
            if llm_result.category not in rules.known_categories():
                st.warning("This looked like a new category — logged for review.")

            log_fallback(
                settings.fallback_log_path,
                image_id=uploaded.name,
                predicted_category=llm_result.category,
                confidence=llm_result.confidence,
                bin_name=rule.bin,
                known_categories=rules.known_categories(),
                reasoning=llm_result.reasoning,
            )

        else:
            reason = (
                "AI fallback is disabled"
                if not llm_enabled
                else "no ANTHROPIC_API_KEY is set to ask Claude instead"
            )
            st.error(
                f"Local model wasn't confident about '{result.category}' "
                f"({result.confidence:.0%}), and {reason}."
            )

    finally:
        temp_path.unlink(missing_ok=True)
