import json

import httpx
import pytest

from router.backends import BackendError, OpenAICompatibleBackend
from router.classify import classify
from router.router import HybridRouter, Prices, summarize


def completion(text: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def backend(name: str, handler) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(name, "http://backend.test/v1", f"{name}-model", transport=httpx.MockTransport(handler))


def ok_handler(text: str):
    return lambda request: httpx.Response(200, json=completion(text))


def failing_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "down"})


def test_simple_request_routes_small():
    decision = classify("What is the capital of France?")
    assert decision.tier == "small"


def test_complex_request_routes_frontier():
    decision = classify("Refactor this service to fix a race condition and review the security of the auth flow.")
    assert decision.tier == "frontier"
    assert any("complex terms" in r for r in decision.reasons)


def test_code_block_adds_score():
    decision = classify("Why does this fail?\n```python\ndef f():\n    return 1\n```")
    assert any("contains code" in r for r in decision.reasons)


def test_many_tools_push_to_frontier():
    assert classify("Look up the order and the stock", tool_count=3).tier == "frontier"


def test_small_answer_used_when_good():
    router = HybridRouter(
        frontier=backend("frontier", ok_handler("frontier answer")),
        small=backend("small", ok_handler("small answer")),
    )
    record = router.route([{"role": "user", "content": "What is 2+2?"}])
    assert record.tier == "small" and record.answer == "small answer" and record.fallback_used is False


def test_falls_back_to_frontier_when_small_fails():
    router = HybridRouter(
        frontier=backend("frontier", ok_handler("frontier answer")),
        small=backend("small", failing_handler),
    )
    record = router.route([{"role": "user", "content": "What is 2+2?"}])
    assert record.tier == "frontier" and record.fallback_used is True
    assert any("small model failed" in r for r in record.reasons)


def test_escalates_when_small_answer_is_empty():
    router = HybridRouter(
        frontier=backend("frontier", ok_handler("frontier answer")),
        small=backend("small", ok_handler("")),
    )
    record = router.route([{"role": "user", "content": "What is 2+2?"}])
    assert record.fallback_used is True and record.answer == "frontier answer"


def test_no_small_model_configured_uses_frontier():
    router = HybridRouter(frontier=backend("frontier", ok_handler("frontier answer")))
    record = router.route([{"role": "user", "content": "What is 2+2?"}])
    assert record.tier == "frontier" and record.fallback_used is False


def test_cost_uses_per_million_token_prices():
    prices = Prices(input_per_m=1.0, output_per_m=2.0)
    router = HybridRouter(frontier=backend("frontier", lambda r: httpx.Response(200, json=completion("ok", 1_000_000, 500_000))),
                          prices=prices)
    record = router.route([{"role": "user", "content": "hello there"}])
    assert record.estimated_cost_usd == pytest.approx(2.0)


def test_summary_counts_tiers_and_fallbacks():
    router = HybridRouter(
        frontier=backend("frontier", ok_handler("frontier answer")),
        small=backend("small", failing_handler),
    )
    records = [router.route([{"role": "user", "content": "What is 2+2?"}]) for _ in range(3)]
    report = summarize(records)
    assert report["requests"] == 3
    assert report["fallbacks"] == 3
    assert report["small_share"] == 0.0


def test_backend_unexpected_response_raises_backend_error():
    bad = backend("frontier", lambda r: httpx.Response(200, json={"oops": True}))
    with pytest.raises(BackendError):
        bad.chat([{"role": "user", "content": "hi"}])


def test_api_requires_frontier_key(monkeypatch):
    from fastapi.testclient import TestClient
    from router import main

    monkeypatch.delenv("FRONTIER_API_KEY", raising=False)
    response = TestClient(main.app).post("/api/chat", json={"message": "hello"})
    assert response.status_code == 503
