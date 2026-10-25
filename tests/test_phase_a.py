import asyncio
from dataclasses import replace

from app import llm
from app.config import settings
from app.llm import _mock_response, metrics


def test_mock_swarm_finds_grep_style_lines():
    result = _mock_response("swarm", [{"content": "L00001|ordinary line\nL00002|NVRM: Xid 79 GPU has fallen off the bus"}])
    assert len(result["findings"]) == 1
    assert result["findings"][0]["line_start"] == 2
    assert "Xid" in result["findings"][0]["quote"]


def test_metrics_accumulate_by_role():
    metrics.record("unit", 2, 3, 0.5)
    assert metrics.calls["unit"] >= 1
    assert metrics.prompt_tokens["unit"] >= 2
    assert metrics.completion_tokens["unit"] >= 3


def test_client_uses_configured_base_url_and_key(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url):
            captured.update(api_key=api_key, base_url=base_url)

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "settings", replace(
        settings, api_key="provider-secret", base_url="https://provider.example/v1/"
    ))

    llm._client_for_live()

    assert captured == {
        "api_key": "provider-secret",
        "base_url": "https://provider.example/v1/",
    }

