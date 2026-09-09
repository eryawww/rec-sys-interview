"""Rephrase, with an LLM, evidence the recommender already produced.

The model is never asked to choose, score or order items. It receives the
ranked list and each item's native reason and returns wording. Anything
it says about an item that was not recommended is discarded, and the
order of the input list is preserved regardless of the order it replies
in. This keeps the recommender falsifiable: the two explanation layers
cannot disagree about what was recommended or why.

Every failure mode degrades to the native reason, so a caller always
receives one explanation per recommendation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

from recsys.algorithms.base import Rec

logger = logging.getLogger("recsys.explain")

BASE_URL = "https://api.deepseek.com"
MAX_REASON_CHARS = 160

EXAMPLE = (
    '{"explanations": [{"item_id": "i1", '
    '"reason": "You often watch drama, so this may fit."}]}'
)


class MissingApiKey(RuntimeError):
    """DEEPSEEK_API_KEY is not set."""


@dataclass(frozen=True, slots=True)
class Explanation:
    item_id: str
    reason: str
    source: str  # "llm" | "native"


def build_messages(
    user_name: str,
    genres: list[str],
    recent_titles: list[str],
    recs: list[Rec],
) -> list[dict[str, str]]:
    """Assemble the chat messages.

    DeepSeek's JSON mode requires the word "json" and a worked example of
    the output shape in the prompt, both of which are present below.
    """
    catalogue = "\n".join(
        f"- {rec.item_id} | {rec.title} | {rec.genre} | evidence: {rec.reason}" for rec in recs
    )
    context = [f"Viewer: {user_name}"]
    if genres:
        context.append(f"Watches mostly: {', '.join(genres)}")
    if recent_titles:
        context.append(f"Recently watched: {', '.join(recent_titles)}")

    system = (
        "You write one-sentence explanations for streaming recommendations. "
        "You are given items that have already been chosen and the evidence "
        "for each. Rephrase that evidence warmly and specifically. "
        "Never add, remove, reorder or rank items. "
        "Reply with json only, in exactly this shape: " + EXAMPLE
    )
    user = (
        "\n".join(context)
        + "\n\nRecommended items:\n"
        + catalogue
        + "\n\nReturn json with one entry per item_id above, each reason "
        f"under {MAX_REASON_CHARS} characters."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _native(recs: list[Rec]) -> list[Explanation]:
    return [Explanation(rec.item_id, rec.reason, "native") for rec in recs]


def explain(
    user_name: str,
    genres: list[str],
    recent_titles: list[str],
    recs: list[Rec],
    *,
    client: httpx.Client | None = None,
) -> tuple[list[Explanation], str, bool]:
    """Return (explanations, model, degraded), one entry per rec, in order."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise MissingApiKey(
            "DEEPSEEK_API_KEY is not set. Recommendations still work; "
            "only generated explanations are unavailable."
        )

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if not recs:
        return [], model, False

    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "10"))
    owns_client = client is None
    client = client or httpx.Client(base_url=BASE_URL, timeout=timeout)

    payload = {
        "model": model,
        "messages": build_messages(user_name, genres, recent_titles, recs),
        "response_format": {"type": "json_object"},
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("empty content")
        parsed = json.loads(content)
        by_id = {
            entry["item_id"]: str(entry["reason"])
            for entry in parsed["explanations"]
            if entry.get("item_id") and entry.get("reason")
        }
    except Exception as exc:  # noqa: BLE001 - every failure degrades identically
        logger.warning("Explanation call failed (%s); using native reasons.", type(exc).__name__)
        return _native(recs), model, True
    finally:
        if owns_client:
            client.close()

    out: list[Explanation] = []
    degraded = False
    for rec in recs:
        text = by_id.get(rec.item_id)
        if text:
            out.append(Explanation(rec.item_id, text[:MAX_REASON_CHARS], "llm"))
        else:
            out.append(Explanation(rec.item_id, rec.reason, "native"))
            degraded = True
    return out, model, degraded
