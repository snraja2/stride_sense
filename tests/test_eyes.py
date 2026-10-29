import asyncio
from dataclasses import replace

from app import llm
from app.eyes import analyze_dashboard
from app.ingest import load_bundle
from tools.make_incident import make_incident


def test_mock_eyes_returns_fixed_synthetic_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(llm.settings, mode="mock"))
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))

    result = asyncio.run(analyze_dashboard(bundle))

    assert result["observations"][0]["id"] == "O1"
    assert "Synthetic mock" in result["observations"][0]["claim"]


def test_eyes_failure_emits_warning_and_returns_empty_observations(tmp_path, monkeypatch):
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))

    async def broken_call(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm, "call", broken_call)
    events = []
    result = asyncio.run(analyze_dashboard(bundle, events.append))

    assert result["observations"] == []
    assert events[0]["type"] == "warning"


def test_eyes_disabled_skips_the_provider_call(tmp_path, monkeypatch):
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))

    async def forbidden_call(*args, **kwargs):
        raise AssertionError("provider should not be called when eyes is disabled")

    monkeypatch.setattr(llm, "call", forbidden_call)
    result = asyncio.run(analyze_dashboard(bundle, enabled=False))
    assert result["observations"] == []
