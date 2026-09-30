"""
llm_fallback.py

Called when the local classifier's confidence is below rules.yaml's
confidence_threshold, or when it predicts a category that isn't in
rules.yaml. Sends the photo to Claude along with the app's currently known
categories, so its answer stays consistent with what rules.yaml can
actually handle, and asks for structured JSON back.

Requires the ANTHROPIC_API_KEY environment variable to be set (see
.env.example).

The fragile part — parsing the model's response — is split into a pure
function (_parse_llm_json) with no network dependency, so it can be
exercised offline with hand-written malformed strings. It never raises:
a response that isn't valid JSON (empty, prose-wrapped, code-fenced,
truncated, wrong types) degrades to a safe "other_unlabeled" result with
parse_ok=False, which the caller can surface to the user instead of
crashing the app.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

FALLBACK_PROMPT_TEMPLATE = """You are helping classify a single photographed waste item into one of the following categories: {categories}.

Look closely at the photo of one item and respond with ONLY a JSON object of exactly this shape, no other text:
{{"category": "<one of the listed categories, or \\"other_unlabeled\\" if none fit>", "confidence": <a number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}
"""


@dataclass(frozen=True)
class LlmFallbackResult:
    category: str
    confidence: float
    reasoning: str
    raw_response: str
    parse_ok: bool


def _encode_image(image_path: str | Path) -> tuple[str, bytes]:
    image_path = Path(image_path)
    media_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return media_type, image_path.read_bytes()


def _parse_llm_json(text: str, known_categories: list[str]) -> LlmFallbackResult:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = json.loads(match.group(0)) if match else json.loads(text)

        category = str(payload.get("category", "other_unlabeled")).strip()
        if category not in known_categories:
            # LLM proposed something outside our known vocabulary — keep the
            # raw string (it's logged as a candidate new category) but don't
            # pretend it's a rules.yaml hit; caller decides how to handle it.
            pass

        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(payload.get("reasoning", "")).strip()

        return LlmFallbackResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=text,
            parse_ok=True,
        )
    except Exception:
        return LlmFallbackResult(
            category="other_unlabeled",
            confidence=0.0,
            reasoning="Could not parse a confident answer from the model's response.",
            raw_response=text,
            parse_ok=False,
        )


def classify_with_llm(
    image_path: str | Path,
    known_categories: list[str],
    api_key: str,
    model: str,
    max_tokens: int = 512,
) -> LlmFallbackResult:
    """Returns a structured result from Claude for a single item photo."""
    import anthropic  # imported lazily so offline JSON-parsing tests don't need it installed

    media_type, image_bytes = _encode_image(image_path)
    client = anthropic.Anthropic(api_key=api_key)
    prompt = FALLBACK_PROMPT_TEMPLATE.format(categories=", ".join(known_categories))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw_text = response.content[0].text
    return _parse_llm_json(raw_text, known_categories)
