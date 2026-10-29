from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi.testclient import TestClient

from app import main
from app.limits import RunLimiter


async def _skip_startup():
    return None


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    incidents = tmp_path / "incidents"
    sample = incidents / "xid79_demo"
    logs = sample / "logs"
    logs.mkdir(parents=True)
    (logs / "rank_013.log").write_text("2026-10-01T12:00:00Z INFO step 1\n", encoding="utf-8")
    (logs / "dmesg_node-01.log").write_text(
        "2026-10-01T12:00:00Z kernel: ready\n2026-10-01T12:01:00Z kernel: Xid 79 synthetic\n",
        encoding="utf-8")
    (logs / "scheduler.log").write_text("job_id: synthetic\n", encoding="utf-8")
    (sample / "job_config.yaml").write_text("nodes: 1\n", encoding="utf-8")
    (sample / "dashboard.png").write_bytes(b"synthetic test image")
    (sample / "truth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "_ensure_demo_data", _skip_startup)
    monkeypatch.setattr(main, "settings", replace(
        main.settings, incidents_dir=str(incidents), runs_dir=str(tmp_path / "runs")))
    main.app.state.limiter = RunLimiter()
    main.app.state.uploads = set()
    return TestClient(main.app)


def test_home_health_dashboard_and_source_routes(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert b"CrashCart" in home.content
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/api/incidents").json()[0]["synthetic"] is True
        assert client.get("/api/dashboard/xid79_demo").headers["content-type"] == "image/png"
        source = client.get("/api/source", params={
            "incident": "xid79_demo", "file": "logs/dmesg_node-01.log", "line": 2,
        })
        assert source.status_code == 200
        assert "Xid" in source.json()["lines"][1]["text"]
        assert client.get("/api/source", params={
            "incident": "xid79_demo", "file": "../truth.json",
        }).status_code == 400
        assert client.get("/api/source", params={
            "incident": "xid79_demo", "file": "logs/truth.json",
        }).status_code == 400
        assert client.get("/api/report/bad!id").status_code == 400


def _zip_file(path: str, content: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(path, content)
    return buffer.getvalue()


def test_upload_rejects_traversal_and_truth_then_accepts_text_bundle(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        bad = _zip_file("../escape.log", b"not safe")
        response = client.post("/api/upload", files={"file": ("bad.zip", bad, "application/zip")})
        assert response.status_code == 400

        truth = _zip_file("bundle/truth.json", b"{}")
        response = client.post("/api/upload", files={"file": ("truth.zip", truth, "application/zip")})
        assert response.status_code == 400

        good = _zip_file("bundle/logs/rank_000.log", b"2026-10-01T12:00:00Z INFO healthy\n")
        response = client.post("/api/upload", files={"file": ("logs.zip", good, "application/zip")})
        assert response.status_code == 201
        assert response.json()["file_count"] == 1
        assert response.json()["has_dashboard"] is False
        assert client.get("/api/incidents").json()[-1]["synthetic"] is False


def test_run_sse_report_and_replay_routes(tmp_path, monkeypatch):
    from app import llm

    monkeypatch.setattr(llm, "settings", replace(llm.settings, mode="mock"))
    with _client(tmp_path, monkeypatch) as client:
        started = client.post("/api/run", json={
            "incident_id": "xid79_demo", "swarm_concurrency": 4, "use_eyes": True,
        })
        assert started.status_code == 202
        run_id = started.json()["run_id"]
        stream = client.get(f"/api/stream/{run_id}")
        assert stream.status_code == 200
        event_types = [line[7:] for line in stream.text.splitlines() if line.startswith("event: ")]
        assert event_types[0] == "run_started"
        assert "metrics" in event_types
        assert event_types[-1] == "done"
        report = client.get(f"/api/report/{run_id}")
        assert report.status_code == 200
        assert report.json()["diagnosis"]["root_cause"]["category"] == "gpu_fell_off_bus"
        assert client.get(f"/api/report/{run_id}/postmortem.md").status_code == 200

        sample_events = Path(main.settings.runs_dir) / main.app.state.sample_replay_id / "events.jsonl"
        sample_events.parent.mkdir(parents=True, exist_ok=True)
        sample_events.write_text(
            '{"t_rel":0,"event":{"type":"run_started"}}\n'
            '{"t_rel":0.01,"event":{"type":"done"}}\n', encoding="utf-8")
        replay = client.get(f"/api/replay/{main.app.state.sample_replay_id}?speed=20")
        assert replay.status_code == 200
        assert "event: run_started" in replay.text and "event: done" in replay.text
