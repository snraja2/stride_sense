"""FastAPI application for local and hackathon CrashCart runs."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app import llm
from app.config import settings
from app.events import EventRecorder, read_events, replay_events
from app.ingest import load_bundle
from app.limits import RunLimiter
from app.pipeline import run as run_pipeline
from tools.make_incident import FAULTS, make_incident


ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = ROOT / "app" / "static" / "index.html"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_UNPACKED_BYTES = 200 * 1024 * 1024
MAX_UPLOAD_TOKENS = 1_200_000
TEXT_SUFFIXES = {".log", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini"}
_INCIDENT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}$")


class RunRequest(BaseModel):
    incident_id: str
    swarm_concurrency: int = Field(default=24, ge=4, le=48)
    use_eyes: bool = True
    baseline: bool = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _ensure_demo_data()
    yield


app = FastAPI(title="CrashCart", version="0.1.0", lifespan=lifespan)
app.state.runs = {}
app.state.tasks = {}
app.state.sample_replay_id = "sample_xid79_demo"
app.state.uploads = set()
app.state.limiter = RunLimiter(settings.ip_runs_per_hour, settings.daily_run_cap)


def _incident_root() -> Path:
    return Path(settings.incidents_dir).resolve()


def _runs_root() -> Path:
    return Path(settings.runs_dir).resolve()


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    return run_id


def _incident_path(incident_id: str) -> Path:
    if not _INCIDENT_ID.fullmatch(incident_id):
        raise HTTPException(status_code=400, detail="invalid incident id")
    if incident_id in app.state.uploads:
        path = _incident_root() / "uploads" / incident_id
    else:
        path = _incident_root() / incident_id
    try:
        path.resolve().relative_to(_incident_root())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid incident path") from exc
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="incident not found")
    return path


async def _ensure_demo_data() -> None:
    root = _incident_root()
    root.mkdir(parents=True, exist_ok=True)
    missing = [fault_id for fault_id in FAULTS if not (root / fault_id).is_dir()]
    for fault_id in missing:
        make_incident(fault_id, root, 7)

    sample_id = app.state.sample_replay_id
    sample_dir = _runs_root() / sample_id
    sample_events = sample_dir / "events.jsonl"
    if sample_events.exists() and sample_events.stat().st_size:
        return
    sample_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(sample_events)
    bundle = load_bundle(root / "xid79_demo")
    original_settings = llm.settings
    llm.settings = replace(original_settings, mode="mock")
    try:
        report = await run_pipeline(bundle, {
            "run_id": sample_id,
            "incident_id": "xid79_demo",
            "swarm_concurrency": 24,
            "use_eyes": True,
            "baseline": True,
        }, recorder.emit, sample_dir)
        (sample_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        await recorder.finish()
        llm.settings = original_settings


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_INDEX)


@app.get("/healthz")
@app.get("/api/healthz")
async def healthz() -> dict:
    return {"status": "ok", "mode": llm.settings.mode}


@app.get("/api/incidents")
async def incidents() -> list[dict]:
    items = []
    for incident_id in FAULTS:
        path = _incident_root() / incident_id
        if not path.is_dir():
            continue
        bundle = load_bundle(path)
        items.append({
            "id": incident_id,
            "title": incident_id.replace("_", " ").title(),
            "file_count": len(bundle.files),
            "est_tokens": bundle.est_tokens,
            "has_dashboard": any(record.kind == "dashboard" for record in bundle.files.values()),
            "has_truth": (path / "truth.json").is_file(),
            "synthetic": True,
        })
    for incident_id in sorted(app.state.uploads):
        path = _incident_root() / "uploads" / incident_id
        if not path.is_dir():
            continue
        bundle = load_bundle(path)
        items.append({
            "id": incident_id,
            "title": "Uploaded incident",
            "file_count": len(bundle.files),
            "est_tokens": bundle.est_tokens,
            "has_dashboard": any(record.kind == "dashboard" for record in bundle.files.values()),
            "has_truth": False,
            "synthetic": False,
        })
    return items


async def _execute_run(run_id: str, incident_id: str, options: dict, recorder: EventRecorder, run_dir: Path) -> None:
    try:
        bundle = load_bundle(_incident_path(incident_id))
        report = await run_pipeline(bundle, options, recorder.emit, run_dir)
        report["run_id"] = run_id
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        app.state.runs[run_id]["report"] = report
        app.state.runs[run_id]["status"] = "complete"
    except Exception as exc:
        await recorder.emit({"type": "error", "message": f"Run failed ({type(exc).__name__})."})
        app.state.runs[run_id]["status"] = "failed"
    finally:
        await recorder.finish()


@app.post("/api/run", status_code=202)
async def start_run(body: RunRequest, request: Request) -> dict:
    path = _incident_path(body.incident_id)
    ip = request.client.host if request.client else "unknown"
    if not app.state.limiter.allow(ip, live=llm.settings.mode == "live"):
        return JSONResponse(status_code=429, content={"error": "limit", "replay_run_id": app.state.sample_replay_id})
    run_id = uuid.uuid4().hex
    run_dir = _runs_root() / run_id
    recorder = EventRecorder(run_dir / "events.jsonl")
    options = {
        "run_id": run_id,
        "incident_id": body.incident_id,
        "swarm_concurrency": body.swarm_concurrency,
        "use_eyes": body.use_eyes,
        "baseline": body.baseline,
    }
    app.state.runs[run_id] = {"recorder": recorder, "status": "running", "report": None}
    task = asyncio.create_task(_execute_run(run_id, body.incident_id, options, recorder, run_dir))
    app.state.tasks[run_id] = task
    task.add_done_callback(lambda _task: app.state.tasks.pop(run_id, None))
    return {"run_id": run_id}


def _sse(event: dict) -> str:
    name = str(event.get("type") or "message")
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


def _read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, remaining: int) -> bytes:
    """Read a zip member in bounded chunks, even if its declared size is dishonest."""
    if remaining < 0:
        raise HTTPException(status_code=413, detail="zip uncompressed size exceeds the safety limit")
    chunks = bytearray()
    try:
        with archive.open(info, "r") as member:
            while True:
                block = member.read(min(1024 * 1024, remaining + 1 - len(chunks)))
                if not block:
                    break
                chunks.extend(block)
                if len(chunks) > remaining:
                    raise HTTPException(status_code=413, detail="zip uncompressed size exceeds the safety limit")
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail="zip contains an unreadable entry") from exc
    return bytes(chunks)


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str) -> StreamingResponse:
    run_id = _safe_run_id(run_id)
    state = app.state.runs.get(run_id)
    if not state:
        path = _runs_root() / run_id / "events.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="run not found")

        async def completed_stream():
            for record in read_events(path):
                yield _sse(record["event"])

        return StreamingResponse(completed_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    recorder: EventRecorder = state["recorder"]

    async def live_stream():
        index = 0
        while True:
            while index < len(recorder.events):
                record = recorder.events[index]
                index += 1
                yield _sse(record["event"])
            if recorder.done:
                break
            try:
                record = await asyncio.wait_for(recorder.queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if record is None or record["seq"] < index:
                continue
            while index < record["seq"]:
                previous = recorder.events[index]
                index += 1
                yield _sse(previous["event"])
            index = record["seq"] + 1
            yield _sse(record["event"])

    return StreamingResponse(live_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _replay_path(run_id: str) -> Path | None:
    if run_id == app.state.sample_replay_id:
        return _runs_root() / run_id / "events.jsonl"
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
        return None
    live_path = _runs_root() / run_id / "events.jsonl"
    if live_path.is_file():
        return live_path
    sample_path = ROOT / "samples" / "runs" / f"{run_id}.jsonl"
    return sample_path if sample_path.is_file() else None


@app.get("/api/replay/{run_id}")
async def replay_run(run_id: str, speed: float = Query(default=1.0, gt=0, le=20)) -> StreamingResponse:
    path = _replay_path(run_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="replay not found")

    async def replay_stream():
        async for event in replay_events(path, speed):
            yield _sse(event)

    return StreamingResponse(replay_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/report/{run_id}")
async def get_report(run_id: str) -> dict:
    run_id = _safe_run_id(run_id)
    state = app.state.runs.get(run_id)
    if state and state.get("report") is not None:
        return state["report"]
    path = _runs_root() / run_id / "report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/report/{run_id}/postmortem.md")
async def get_postmortem(run_id: str) -> Response:
    run_id = _safe_run_id(run_id)
    path = _runs_root() / run_id / "postmortem.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="postmortem not found")
    return Response(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@app.get("/api/source")
async def source(incident: str, file: str, line: int = 1, ctx: int = 15) -> dict:
    root = _incident_path(incident).resolve()
    normalized = file.replace("\\", "/")
    parts = PurePosixPath(normalized)
    if parts.is_absolute() or ".." in parts.parts or parts.name.lower() == "truth.json":
        raise HTTPException(status_code=400, detail="invalid source path")
    target = (root / Path(*parts.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid source path") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="source file not found")
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="source file could not be read") from exc
    context = max(0, min(100, int(ctx)))
    center = max(1, min(max(1, len(lines)), int(line)))
    start = max(1, center - context)
    end = min(len(lines), center + context)
    return {"file": normalized, "start_line": start, "end_line": end,
            "selected_line": center, "lines": [{"line": number, "text": lines[number - 1]}
                                                 for number in range(start, end + 1)]}


@app.get("/api/dashboard/{incident_id}")
async def dashboard(incident_id: str) -> FileResponse:
    root = _incident_path(incident_id).resolve()
    target = (root / "dashboard.png").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid dashboard path") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="dashboard not available")
    return FileResponse(target, media_type="image/png")


@app.get("/api/truth/{incident_id}")
async def truth_badge(incident_id: str) -> dict:
    if incident_id not in FAULTS:
        raise HTTPException(status_code=404, detail="truth is only available for synthetic samples")
    truth_path = _incident_path(incident_id) / "truth.json"
    if not truth_path.is_file():
        raise HTTPException(status_code=404, detail="ground truth unavailable")
    return json.loads(truth_path.read_text(encoding="utf-8"))


@app.post("/api/upload", status_code=201)
async def upload_bundle(file: UploadFile = File(...)) -> dict:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload exceeds 30 MB")
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="upload must be a zip file") from exc
    entries = [info for info in archive.infolist() if not info.is_dir()]
    if len(entries) > 10_000:
        raise HTTPException(status_code=400, detail="zip contains too many files")
    unpacked_size = sum(info.file_size for info in entries)
    if unpacked_size > MAX_UNPACKED_BYTES:
        raise HTTPException(status_code=413, detail="zip uncompressed size exceeds the safety limit")
    if not entries:
        raise HTTPException(status_code=400, detail="zip contains no files")

    validated: list[tuple[str, bytes, bool]] = []
    names = set()
    dashboard_count = 0
    estimated_chars = 0
    actual_unpacked_size = 0
    for info in entries:
        name = info.filename.replace("\\", "/")
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or (parts.parts and ":" in parts.parts[0]):
            raise HTTPException(status_code=400, detail="zip contains an unsafe path")
        clean_parts = [part for part in parts.parts if part not in {"", "."}]
        if not clean_parts or clean_parts[-1].lower() == "truth.json":
            raise HTTPException(status_code=400, detail="zip may not include truth.json")
        clean_name = PurePosixPath(*clean_parts).as_posix()
        name_key = clean_name.casefold()
        if name_key in names:
            raise HTTPException(status_code=400, detail="zip contains duplicate paths")
        names.add(name_key)
        suffix = Path(clean_name).suffix.lower()
        if suffix == ".png":
            dashboard_count += 1
            if dashboard_count > 1:
                raise HTTPException(status_code=400, detail="zip may contain only one PNG dashboard")
            data = _read_zip_member(archive, info, MAX_UNPACKED_BYTES - actual_unpacked_size)
            actual_unpacked_size += len(data)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise HTTPException(status_code=400, detail=f"{clean_name} is not a PNG image")
            validated.append(("dashboard.png", data, True))
        elif suffix in TEXT_SUFFIXES:
            data = _read_zip_member(archive, info, MAX_UNPACKED_BYTES - actual_unpacked_size)
            actual_unpacked_size += len(data)
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"{clean_name} is not UTF-8 text") from exc
            estimated_chars += len(data)
            validated.append((clean_name, data, False))
        else:
            raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix or '(none)'}")
    if estimated_chars // 4 > MAX_UPLOAD_TOKENS:
        raise HTTPException(status_code=413, detail="upload exceeds 1.2M estimated text tokens")
    if not any(not is_image for _, _, is_image in validated):
        raise HTTPException(status_code=400, detail="zip must contain at least one text file")
    archive.close()

    incident_id = f"upload-{uuid.uuid4().hex[:12]}"
    target_root = _incident_root() / "uploads" / incident_id
    target_root.mkdir(parents=True, exist_ok=False)
    root_resolved = target_root.resolve()
    for relpath, data, _ in validated:
        target = (target_root / Path(*PurePosixPath(relpath).parts)).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="zip contains an unsafe path") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    app.state.uploads.add(incident_id)
    bundle = load_bundle(target_root)
    return {"incident_id": incident_id, "file_count": len(bundle.files),
            "est_tokens": bundle.est_tokens,
            "has_dashboard": any(record.kind == "dashboard" for record in bundle.files.values())}
