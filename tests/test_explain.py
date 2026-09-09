import json

import httpx
import pytest

from recsys.algorithms.base import Rec
from recsys.explain import Explanation, MissingApiKey, build_messages, explain

RECS = [
    Rec("i1", "Start-Up", "series", "drama", 2.0, "Because you watched Crash Landing on You"),
    Rec("i2", "Vincenzo", "series", "thriller", 1.5, "Because you watched Start-Up"),
]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com")


def reply(payload: dict) -> httpx.Response:
    body = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return httpx.Response(200, json=body)


def test_prompt_mentions_json_and_shows_an_example():
    messages = build_messages("Ada", ["drama"], ["Crash Landing on You"], RECS)
    blob = " ".join(m["content"] for m in messages)
    assert "json" in blob.lower()
    assert "explanations" in blob
    assert "Start-Up" in blob


def test_prompt_carries_the_native_reason_as_evidence():
    messages = build_messages("Ada", ["drama"], [], RECS)
    blob = " ".join(m["content"] for m in messages)
    assert "Because you watched Crash Landing on You" in blob


def test_successful_call_returns_llm_reasons():
    def handler(request):
        return reply(
            {
                "explanations": [
                    {"item_id": "i1", "reason": "You keep coming back to drama series."},
                    {"item_id": "i2", "reason": "A thriller with the same lead."},
                ]
            }
        )

    out, model, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is False
    assert [e.source for e in out] == ["llm", "llm"]
    assert out[0].reason == "You keep coming back to drama series."
    assert model


def test_timeout_degrades_to_native_reasons():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert [e.reason for e in out] == [r.reason for r in RECS]
    assert all(e.source == "native" for e in out)


def test_http_error_degrades():
    out, _, degraded = explain(
        "Ada",
        ["drama"],
        [],
        RECS,
        client=make_client(lambda request: httpx.Response(500, text="boom")),
    )
    assert degraded is True
    assert all(e.source == "native" for e in out)


def test_malformed_json_degrades():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert all(e.source == "native" for e in out)


def test_empty_content_degrades():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True


def test_partial_response_degrades_only_the_missing_item():
    def handler(request):
        return reply({"explanations": [{"item_id": "i1", "reason": "Drama, your usual."}]})

    out, _, degraded = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert degraded is True
    assert out[0].source == "llm"
    assert out[1].source == "native"
    assert out[1].reason == RECS[1].reason


def test_llm_cannot_introduce_or_reorder_items():
    def handler(request):
        return reply(
            {
                "explanations": [
                    {"item_id": "i2", "reason": "second"},
                    {"item_id": "i1", "reason": "first"},
                    {"item_id": "i999", "reason": "not recommended at all"},
                ]
            }
        )

    out, _, _ = explain("Ada", ["drama"], [], RECS, client=make_client(handler))
    assert [e.item_id for e in out] == ["i1", "i2"]


def test_overlong_reason_is_truncated():
    def handler(request):
        return reply({"explanations": [{"item_id": "i1", "reason": "x" * 500}]})

    out, _, _ = explain("Ada", ["drama"], [], RECS[:1], client=make_client(handler))
    assert len(out[0].reason) <= 160


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingApiKey):
        explain("Ada", ["drama"], [], RECS)


def test_explanation_is_immutable():
    e = Explanation("i1", "why", "llm")
    with pytest.raises(AttributeError):
        e.reason = "other"  # type: ignore[misc]
