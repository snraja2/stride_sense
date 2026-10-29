import asyncio
from dataclasses import replace
from pathlib import Path

from app import llm
from app.events import EventRecorder, read_events, replay_events
from app.ingest import load_bundle
from app.limits import RunLimiter
from app.pipeline import run
from tools.make_incident import make_incident


def test_mock_pipeline_emits_complete_verified_demo_without_reading_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(llm.settings, mode="mock"))
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name.lower() == "truth.json":
            raise AssertionError("pipeline must not read truth.json")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    events = []
    report = asyncio.run(run(bundle, {"run_id": "test-run", "incident_id": "xid79_demo"}, events.append, tmp_path / "run"))

    assert report["diagnosis"]["verdict"] == "diagnosis"
    assert report["diagnosis"]["root_cause"]["rank"] == 13
    assert report["verification"] == {"verified": 2, "total": 2, "failed": 0}
    event_types = [event["type"] for event in events]
    assert event_types[0] == "run_started"
    assert "baseline" in event_types and "omni" in event_types and "clusters" in event_types
    assert "verdict" in event_types and event_types[-1] == "done"
    assert (tmp_path / "run" / "postmortem.md").is_file()
    assert (tmp_path / "run" / "runbook.md").is_file()


def test_event_recorder_persists_and_replays_in_order(tmp_path):
    path = tmp_path / "events.jsonl"

    async def record():
        recorder = EventRecorder(path)
        await recorder.emit({"type": "run_started", "run_id": "r1"})
        await recorder.emit({"type": "done"})
        await recorder.finish()

    asyncio.run(record())
    recorded = read_events(path)

    async def replay():
        return [event async for event in replay_events(path, speed=100)]

    replayed = asyncio.run(replay())
    assert [row["event"]["type"] for row in recorded] == ["run_started", "done"]
    assert replayed == [row["event"] for row in recorded]


def test_live_limiter_counts_per_ip_and_daily_but_mock_is_unlimited():
    limiter = RunLimiter(per_ip_per_hour=2, daily_cap=3)

    assert limiter.allow("a", live=True)
    assert limiter.allow("a", live=True)
    assert not limiter.allow("a", live=True)
    assert limiter.allow("b", live=True)
    assert not limiter.allow("c", live=True)
    assert limiter.allow("mock", live=False)
