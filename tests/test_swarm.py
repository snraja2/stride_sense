import asyncio
from dataclasses import replace

from app import llm
from app.ingest import load_bundle
from app.swarm import run_swarm, scan
from app.verify import verify_quote
from tools.make_incident import make_incident


def test_mock_swarm_finds_grounded_xid_and_emits_a_terminal_tile_per_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(llm.settings, mode="mock"))
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))
    events = []

    findings = asyncio.run(run_swarm(bundle.chunks, bundle, events.append))

    assert any(
        item["file"] == "logs/dmesg_node-01.log" and "Xid" in item["quote"]
        for item in findings
    )
    assert all(verify_quote(bundle, item["file"], item["line_start"], item["line_end"], item["quote"]) for item in findings)
    terminal_tiles = [event for event in events if event["type"] == "tile" and event["state"] in {"clean", "hit", "error"}]
    assert len(terminal_tiles) == len(bundle.chunks)


def test_swarm_drops_unverified_model_findings(tmp_path, monkeypatch):
    (tmp_path / "rank_000.log").write_text("2026-10-01T12:00:00Z INFO healthy\n")
    bundle = load_bundle(tmp_path)
    calls = 0

    async def fake_call(role, messages, **kwargs):
        nonlocal calls
        calls += 1
        return {"findings": [{"line_start": 1, "line_end": 1, "quote": "made up error", "kind": "error", "severity": 5, "why": "not present"}]}, 0.01

    monkeypatch.setattr(llm, "call", fake_call)
    events = []
    findings = asyncio.run(scan(bundle.chunks[0], bundle, events.append))

    assert calls == 1
    assert findings == []
    assert events[-1]["state"] == "clean"
