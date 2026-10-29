import asyncio
from dataclasses import replace

from app import llm
from app.config import settings
from app.llm import _mock_reduce_response, _mock_response, metrics


def test_mock_swarm_finds_grep_style_lines():
    result = _mock_response("swarm", [{"content": "L00001|ordinary line\nL00002|NVRM: Xid 79 GPU has fallen off the bus"}])
    assert len(result["findings"]) == 1
    assert result["findings"][0]["line_start"] == 2
    assert "Xid" in result["findings"][0]["quote"]


def test_mock_swarm_ignores_xid_inside_incident_identifier():
    result = _mock_response("swarm", [{"content": "L00001|job_id: synthetic-xid79_demo"}])
    assert result["findings"] == []


def test_mock_eyes_flags_dashboard_only_scenarios_as_fixtures():
    thermal = _mock_response("eyes", [{"content": "Synthetic mock fixture: thermal_throttle"}])
    power = _mock_response("eyes", [{"content": "Synthetic mock fixture: power_cap"}])
    assert "94 C" in thermal["observations"][0]["claim"]
    assert "power stays flat" in power["observations"][0]["claim"]


def test_mock_reducer_uses_dashboard_observation_and_preemption_rank_location():
    visual = _mock_reduce_response([{"content": '{"clusters": [], "visual": {"observations": [{"id": "O1", "claim": "Synthetic GPU temperature is 94 C and SM clock collapsed."}]}}'}])
    assert visual["result"]["root_cause"]["category"] == "thermal_throttle"
    assert visual["result"]["evidence"][0]["observation_id"] == "O1"

    preemption = _mock_reduce_response([{"content": '{"clusters": ['
        '{"kind":"warning","severity":5,"first_ts":"2026-10-01T12:55:00Z","exemplars":[{"file":"logs/scheduler.log","line_start":4,"line_end":4,"quote":"Job preempted by higher priority job"}]},'
        '{"kind":"error","severity":4,"first_ts":"2026-10-01T12:55:00Z","exemplars":[{"file":"logs/rank_013.log","line_start":323,"line_end":323,"quote":"Received signal 15 from scheduler"}]}], "visual": {"observations": []}}'}])
    root = preemption["result"]["root_cause"]
    assert (root["category"], root["node"], root["gpu"], root["rank"]) == ("preemption", "node-01", 5, 13)


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

