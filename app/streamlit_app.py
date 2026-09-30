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

st.set_page_config(page_title="Sort", page_icon="♻️", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    #MainMenu, header, footer { visibility: hidden; }

    .stApp {
        background: linear-gradient(-45deg, #d4fc79, #a8edea, #84fab0, #8fd3f4);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        overflow-x: hidden;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .blob {
        position: fixed;
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.35;
        z-index: 0;
        pointer-events: none;
    }
    .blob1 { width: 260px; height: 260px; background: #ffffff; top: -60px; left: -80px; animation: float1 12s ease-in-out infinite; }
    .blob2 { width: 200px; height: 200px; background: #ffe29f; bottom: -60px; right: -60px; animation: float2 14s ease-in-out infinite; }
    .blob3 { width: 150px; height: 150px; background: #a1c4fd; top: 40%; right: -40px; animation: float1 10s ease-in-out infinite; }

    @keyframes float1 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, 40px); }
    }
    @keyframes float2 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(-30px, -30px); }
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 480px;
        margin-top: 2.5rem;
        padding: 2.5rem 2rem 2rem 2rem;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(14px);
        border-radius: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        color: #222 !important;
    }
    .block-container p, .block-container span, .block-container label,
    .block-container div, .block-container li, .block-container summary,
    .block-container h1, .block-container h2, .block-container h3 {
        color: #222 !important;
    }
    .block-container [data-testid="stCaptionContainer"],
    .block-container [data-testid="stCaptionContainer"] * {
        color: #555 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.6) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #444 !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stBaseButton-secondary"] * {
        color: #222 !important;
    }
    </style>
    <div class="blob blob1"></div>
    <div class="blob blob2"></div>
    <div class="blob blob3"></div>
    """,
    unsafe_allow_html=True,
)


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

st.markdown("<h1 style='margin-bottom:0;'>♻️ Sort it</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;margin-top:0;'>Snap a photo. We'll tell you the bin.</p>", unsafe_allow_html=True)

if weights_error is not None:
    st.error("No trained model found yet.")
    st.stop()

if not llm_enabled:
    st.caption("🤖 AI fallback is off — unrecognized items show as unsorted.")
elif not api_key:
    st.caption("🔑 No API key set — AI fallback is off until you add one.")

with st.expander("⚙️ Details"):
    st.write(f"Confidence threshold: **{rules.confidence_threshold:.0%}**")
    st.caption(", ".join(rules.known_categories()))

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded:
    st.image(uploaded, width=300)

    temp_path = Path(f"logs/_upload_{uuid.uuid4().hex}.jpg")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded.getvalue())

    try:
        result = classifier.predict(str(temp_path))

        if result.is_confident and rules.lookup(result.category) is not None:
            rule = rules.lookup(result.category)
            st.success(f"**{rule.bin}** · {result.category} ({result.confidence:.0%})")
            for step in rule.steps:
                st.write(f"- {step}")

        elif api_key:
            with st.spinner("Thinking..."):
                llm_result = classify_with_llm(
                    str(temp_path),
                    known_categories=rules.known_categories(),
                    api_key=api_key,
                    model=settings.llm.model,
                    max_tokens=settings.llm.max_tokens,
                )

            if not llm_result.parse_ok:
                st.caption("⚠️ Unclear answer — treating as unknown.")

            rule = rules.lookup(llm_result.category) or rules.lookup("other_unlabeled")
            st.success(f"**{rule.bin}** · {llm_result.category}")
            for step in rule.steps:
                st.write(f"- {step}")
            if llm_result.reasoning:
                with st.expander("Why?"):
                    st.caption(llm_result.reasoning)
            if llm_result.category not in rules.known_categories():
                st.caption("🆕 New category — logged.")

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
            st.error(f"Not sure — '{result.category}' ({result.confidence:.0%}).")

    finally:
        temp_path.unlink(missing_ok=True)
