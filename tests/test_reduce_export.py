import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from app import llm
from app.cluster import cluster_findings
from app.export import render_postmortem, render_runbook, safe_runbook_items, write_exports
from app.ingest import load_bundle, silence_findings
from app.llm import StreamDelta
from app.reduce import _stream_once, diagnose
from app.swarm import run_swarm
from tools.make_incident import make_incident


def test_mock_reducer_verifies_xid79_root_cause_and_rank_location(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(llm.settings, mode="mock"))
    bundle = load_bundle(make_incident("xid79_demo", tmp_path, 7))
    findings = asyncio.run(run_swarm(bundle.chunks, bundle))
    findings.extend(silence_findings(bundle))
    clusters = cluster_findings(findings)

    result = asyncio.run(diagnose(bundle, clusters, []))

    assert result["result"]["verdict"] == "diagnosis"
    root = result["result"]["root_cause"]
    assert (root["category"], root["node"], root["gpu"], root["rank"]) == ("gpu_fell_off_bus", "node-01", 5, 13)
    assert root["first_seen_ts"] == "2026-10-01T12:55:00+00:00"
    assert result["verification"] == {"verified": 2, "total": 2, "failed": 0}
    assert {item["file"] for item in result["result"]["evidence"]} == {
        "logs/dmesg_node-01.log", "logs/rank_013.log"
    }


def test_corrupted_citation_is_retried_then_removed_or_replaced(tmp_path, monkeypatch):
    (tmp_path / "rank_000.log").write_text("2026-10-01T12:00:00Z CUDA out of memory\n")
    bundle = load_bundle(tmp_path)
    calls = 0

    def payload(quote):
        return {
            "verdict": "diagnosis",
            "root_cause": {"summary": "CUDA allocation failed.", "category": "cuda_oom", "node": "node-00", "gpu": 0, "rank": 0, "first_seen_ts": "2026-10-01T12:00:00+00:00"},
            "confidence": 0.8,
            "symptoms": [], "timeline": [],
            "evidence": [{"id": "E1", "file": "rank_000.log", "line_start": 1, "line_end": 1, "quote": quote, "observation_id": None, "supports": "Direct root-cause evidence."}],
            "ruled_out": [], "runbook": [], "missing_data": None,
        }

    async def fake_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        quote = "fabricated quote" if calls == 1 else "CUDA out of memory"
        raw = "Diagnosis [E1]\n---JSON---\n" + json.dumps(payload(quote))
        yield StreamDelta(text=raw)

    monkeypatch.setattr(llm, "call_stream", fake_stream)
    result = asyncio.run(diagnose(bundle, [], []))

    assert calls == 2
    assert result["verification"] == {"verified": 1, "total": 1, "failed": 0}
    assert result["result"]["evidence"][0]["quote"] == "CUDA out of memory"


def test_reducer_stream_strips_think_tags_and_ignores_reasoning_content(monkeypatch):
    response = {"verdict": "abstain", "root_cause": {"category": "other"}}

    async def fake_stream(*args, **kwargs):
        yield SimpleNamespace(text="<think>hidden reason</thi", reasoning_content="private reasoning")
        yield SimpleNamespace(text="nk>Visible narrative.\n---JSO", reasoning_content="private reasoning")
        yield SimpleNamespace(text="N---\n" + json.dumps(response), reasoning_content="private reasoning")

    monkeypatch.setattr(llm, "call_stream", fake_stream)
    events = []
    narrative, _, _ = asyncio.run(_stream_once([], events.append, 1))

    emitted = "".join(event["text"] for event in events)
    assert narrative == "Visible narrative."
    assert "hidden reason" not in emitted
    assert "private reasoning" not in emitted


def test_runbook_filters_destructive_commands_and_exports_markdown(tmp_path):
    result = {
        "verdict": "diagnosis",
        "root_cause": {"summary": "A GPU fault occurred.", "category": "gpu_fell_off_bus"},
        "timeline": [],
        "evidence": [{"id": "E1", "file": "logs/dmesg_node-01.log", "line_start": 3, "line_end": 3, "quote": "Xid 79", "supports": "Direct root-cause evidence."}],
        "ruled_out": [],
        "runbook": [
            {"step": 1, "command": "nvidia-smi -q", "purpose": "Inspect GPU health."},
            {"step": 2, "command": "rm -rf /", "purpose": "unsafe"},
            {"step": 3, "command": "mkfs.ext4 /dev/sda", "purpose": "unsafe"},
            {"step": 4, "command": "dd if=/dev/zero of=/dev/sda", "purpose": "unsafe"},
            {"step": 5, "command": "shutdown now", "purpose": "unsafe"},
            {"step": 6, "command": "reboot now", "purpose": "unsafe"},
            {"step": 7, "command": "kill -9 1", "purpose": "unsafe"},
            {"step": 8, "command": "SUGGESTION: reboot node-01 after approval", "purpose": "Review only."},
        ],
        "verification": {"verified": 1, "total": 1, "failed": 0},
    }

    safe = safe_runbook_items(result)
    markdown = render_runbook(result)
    postmortem = render_postmortem(result, "The GPU failed. [E1]")
    written = write_exports(tmp_path / "run", result, "The GPU failed. [E1]")

    assert [item["command"] for item in safe] == ["nvidia-smi -q", "SUGGESTION: reboot node-01 after approval"]
    assert "DRY-RUN" in markdown
    assert "rm -rf" not in markdown and "mkfs" not in markdown and "kill -9" not in markdown
    assert "logs/dmesg_node-01.log:L3" in postmortem
    assert "1/1 verified" in postmortem
    assert (tmp_path / "run" / "postmortem.md").read_text(encoding="utf-8") == written["postmortem"]
    assert (tmp_path / "run" / "runbook.md").read_text(encoding="utf-8") == written["runbook"]
