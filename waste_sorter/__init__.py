"""waste_sorter — the app's core logic, kept independent of the Streamlit UI.

Modules:
    config          — loads config.yaml + rules.yaml into typed settings
    classifier      — thin wrapper around the trained YOLO-cls model
    rules_engine    — pure category -> bin/steps lookups from rules.yaml
    llm_fallback    — Claude vision call for low-confidence/unknown items
    fallback_logger — append-only log of every fallback event
"""
