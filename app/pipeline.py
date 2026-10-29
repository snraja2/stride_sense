"""Coordinate one run across the baseline, dashboard reader, swarm, and reducer."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Callable

from app import llm
from app.cluster import cluster_findings
from app.config import settings
from app.eyes import analyze_dashboard
from app.export import write_exports
from app.ingest import Bundle, file_stats, silence_findings
from app.reduce import diagnose
from app.swarm import run_swarm


def _metric_snapshot() -> dict:
    current = llm.current_metrics()
    return {
        "calls": dict(current.calls),
        "prompt_tokens": dict(current.prompt_tokens),
        "completion_tokens": dict(current.completion_tokens),
        "latency_s": dict(current.latency_s),
    }


def _delta(current: dict, start: dict, role: str) -> float:
    return current.get(role, 0) - start.get(role, 0)


def _cost(snapshot: dict, start: dict) -> float:
    role_prices = {
        "reduce": (settings.price_reduce_input, settings.price_reduce_output),
        "baseline": (settings.price_reduce_input, settings.price_reduce_output),
        "swarm": (settings.price_swarm_input, settings.price_swarm_output),
        "eyes": (settings.price_eyes_input, settings.price_eyes_output),
        "skeptic": (settings.price_skeptic_input, settings.price_skeptic_output),
    }
    total = 0.0
    for role, (input_price, output_price) in role_prices.items():
        prompt = _delta(snapshot["prompt_tokens"], start["prompt_tokens"], role)
        completion = _delta(snapshot["completion_tokens"], start["completion_tokens"], role)
        total += (prompt * input_price + completion * output_price) / 1_000_000
    return total


def _metrics_payload(started: float, snapshot: dict, baseline: dict) -> dict:
    elapsed = max(0.001, time.monotonic() - started)
    calls = {}
    latencies = {}
    for role, count in snapshot["calls"].items():
        delta_calls = int(_delta(snapshot["calls"], baseline["calls"], role))
        if delta_calls <= 0:
            continue
        latency = _delta(snapshot["latency_s"], baseline["latency_s"], role)
        calls[role] = delta_calls
        latencies[role] = latency / delta_calls
    tokens = sum(_delta(snapshot["prompt_tokens"], baseline["prompt_tokens"], role) +
                 _delta(snapshot["completion_tokens"], baseline["completion_tokens"], role)
                 for role in snapshot["calls"])
    return {
        "type": "metrics",
        "elapsed_s": elapsed,
        "tokens_processed": int(tokens),
        "tokens_per_second": tokens / elapsed,
        "cost_usd": _cost(snapshot, baseline),
        "calls": calls,
        "average_latency_s": latencies,
    }


async def _emit(emit: Callable, event: dict) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


async def _metrics_loop(emit: Callable, started: float, baseline: dict, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            await _emit(emit, _metrics_payload(started, _metric_snapshot(), baseline))


def _file_index(bundle: Bundle) -> list[dict]:
    chunks_by_file: dict[str, list[str]] = {}
    for chunk in bundle.chunks:
        chunks_by_file.setdefault(chunk.file, []).append(chunk.id)
    rows = []
    for path, record in bundle.files.items():
        rows.append({
            "id": (chunks_by_file.get(path) or [f"file-{hashlib.sha256(path.encode()).hexdigest()[:16]}"])[0],
            "chunk_ids": chunks_by_file.get(path, []),
            "file": path,
            "kind": record.kind,
            "rank": record.rank,
            "node": record.node,
            "est_tokens": record.est_tokens,
        })
    return rows


async def _run_baseline(bundle: Bundle, emit: Callable) -> dict | None:
    record = bundle.files.get("logs/rank_000.log")
    if record is None or not record.lines:
        return None
    prompt_path = Path(__file__).resolve().parent / "prompts" / "tail_baseline.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else (
        "Diagnose only from the final log lines. Return JSON with category, node, gpu, explanation and next_steps."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(record.lines[-200:])},
    ]
    try:
        response, latency = await llm.call("baseline", messages, temperature=0, max_tokens=500)
        if isinstance(response, dict):
            parsed = response
            text = json.dumps(response, ensure_ascii=False)
        else:
            choices = getattr(response, "choices", []) or []
            text = getattr(getattr(choices[0], "message", None), "content", "") if choices else str(response)
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed = {"category": "other", "explanation": str(text), "next_steps": []}
        result = {"text": text, "category": parsed.get("category", "other"), "latency_s": latency}
        await _emit(emit, {"type": "baseline", **result})
        return result
    except Exception as exc:
        await _emit(emit, {"type": "warning", "message": f"Tail baseline unavailable ({type(exc).__name__})."})
        return None


async def _run_scoped(bundle: Bundle, options: dict | None = None, emit: Callable | None = None,
                      run_dir: str | Path | None = None) -> dict:
    """Run the mock/live-compatible diagnosis pipeline for one loaded bundle."""
    if emit is None:
        async def emit(_event: dict) -> None:
            return None
    options = options or {}
    started = time.monotonic()
    metric_start = _metric_snapshot()
    stop_metrics = asyncio.Event()
    metrics_task = asyncio.create_task(_metrics_loop(emit, started, metric_start, stop_metrics))
    files = _file_index(bundle)
    await _emit(emit, {
        "type": "run_started",
        "run_id": options.get("run_id"),
        "incident_id": options.get("incident_id", bundle.root.name),
        "files": files,
        "total_tokens": bundle.est_tokens,
        "has_dashboard": any(record.kind == "dashboard" for record in bundle.files.values()),
    })

    use_eyes = bool(options.get("use_eyes", True))
    baseline_enabled = bool(options.get("baseline", True))
    concurrency = max(1, int(options.get("swarm_concurrency", settings.concurrency)))
    baseline_task = asyncio.create_task(_run_baseline(bundle, emit)) if baseline_enabled else None
    eyes_task = asyncio.create_task(analyze_dashboard(bundle, emit, enabled=use_eyes))
    swarm_task = asyncio.create_task(run_swarm(bundle.chunks, bundle, emit, concurrency=concurrency))
    deterministic_findings = silence_findings(bundle)

    tasks = [eyes_task, swarm_task] + ([baseline_task] if baseline_task else [])
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    eyes_result = completed[0] if isinstance(completed[0], dict) else {"observations": [], "summary": "Dashboard analysis unavailable."}
    swarm_findings = completed[1] if isinstance(completed[1], list) else []
    baseline_result = completed[2] if baseline_task and len(completed) > 2 and isinstance(completed[2], dict) else None
    if isinstance(completed[0], BaseException):
        await _emit(emit, {"type": "warning", "message": "Dashboard analysis failed; continuing without visual evidence."})
    if isinstance(completed[1], BaseException):
        await _emit(emit, {"type": "warning", "message": "Log scan failed; continuing with deterministic silence findings."})

    await _emit(emit, {"type": "omni", **eyes_result})
    findings = [*swarm_findings, *deterministic_findings]
    clusters = cluster_findings(findings)
    await _emit(emit, {"type": "clusters", "n": len(clusters), "top": clusters[:20]})
    diagnosis = await diagnose(bundle, clusters, eyes_result.get("observations", []), emit)
    result = diagnosis["result"]
    result["verification"] = diagnosis["verification"]
    exports = write_exports(run_dir, result, diagnosis.get("narrative", "")) if run_dir else {
        "postmortem": "", "runbook": ""
    }
    highlight_files = {item.get("file") for item in result.get("evidence", [])}
    highlight_tiles = [chunk.id for chunk in bundle.chunks if chunk.file in highlight_files]
    await _emit(emit, {
        "type": "verdict",
        "diagnosis": result,
        "verification": diagnosis["verification"],
        "highlight_tiles": highlight_tiles,
    })

    elapsed = time.monotonic() - started
    metrics_snapshot = _metric_snapshot()
    metric_event = _metrics_payload(started, metrics_snapshot, metric_start)
    await _emit(emit, metric_event)
    await _emit(emit, {"type": "done", "elapsed_s": elapsed,
                       "cost_usd": metric_event["cost_usd"], "truth_match": None})
    stop_metrics.set()
    await metrics_task

    report = {
        "narrative": diagnosis.get("narrative", ""),
        "diagnosis": result,
        "verification": diagnosis["verification"],
        "clusters": clusters,
        "findings": findings,
        "observations": eyes_result.get("observations", []),
        "baseline": baseline_result,
        "files": files,
        "elapsed_s": elapsed,
        "cost_usd": metric_event["cost_usd"],
        "postmortem_md": exports["postmortem"],
        "runbook_md": exports["runbook"],
        "file_stats": file_stats(bundle),
    }
    return report


async def run(bundle: Bundle, options: dict | None = None, emit: Callable | None = None,
              run_dir: str | Path | None = None) -> dict:
    """Run one isolated metrics scope so concurrent users don't mix costs."""
    with llm.metrics_scope():
        return await _run_scoped(bundle, options, emit, run_dir)
