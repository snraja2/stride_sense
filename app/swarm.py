"""Parallel log-scan workers with strict source-quote grounding."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from app import llm
from app.ingest import Bundle, Chunk
from app.verify import verify_quote


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "scan.txt"


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return json.dumps(response)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


def _token_usage(response: Any, chunk: Chunk, content: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is not None and completion is not None:
            return int(prompt), int(completion)
    return chunk.est_tokens, len(content) // 4


async def _emit(emit: Callable | None, event: dict) -> None:
    if emit is None:
        return
    result = emit(event)
    if inspect.isawaitable(result):
        await result


def _ground_findings(parsed: Any, bundle: Bundle, chunk: Chunk) -> list[dict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        return []
    record = bundle.files[chunk.file]
    grounded = []
    for raw in parsed["findings"][:12]:
        if not isinstance(raw, dict):
            continue
        try:
            line_start = int(raw.get("line_start"))
            line_end = int(raw.get("line_end"))
            quote = str(raw.get("quote") or "")
        except (TypeError, ValueError):
            continue
        if line_start < chunk.line_start or line_end > chunk.line_end:
            continue
        if not verify_quote(bundle, chunk.file, line_start, line_end, quote):
            continue
        finding = dict(raw)
        finding.update({
            "file": chunk.file,
            "line_start": line_start,
            "line_end": line_end,
            "quote": quote,
            "rank": record.rank,
            "node": f"node-{record.node:02d}" if record.node is not None else None,
        })
        timestamp = record.timestamps[line_start - 1] if line_start <= len(record.timestamps) else None
        if timestamp is not None:
            finding["timestamp"] = timestamp.isoformat()
        grounded.append(finding)
    return grounded


async def scan(chunk: Chunk, bundle: Bundle, emit: Callable | None = None) -> list[dict]:
    """Scan one chunk, retry a malformed JSON response once, and ground hits."""
    await _emit(emit, {"type": "tile", "id": chunk.id, "state": "running"})
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": chunk.text},
    ]
    request_options: dict[str, Any] = {"temperature": 0, "max_tokens": 700}
    if os.getenv("LLM_JSON_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        request_options["response_format"] = {"type": "json_object"}

    total_latency = 0.0
    response = None
    content = ""
    parsed = None
    for attempt in range(2):
        if attempt:
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Return valid JSON only, matching the required findings schema. Do not add prose."},
            ])
        try:
            response, latency = await llm.call("swarm", messages, **request_options)
        except Exception as exc:
            await _emit(emit, {"type": "tile", "id": chunk.id, "state": "error", "hits": 0,
                               "tokens_in": chunk.est_tokens, "tokens_out": 0, "latency_s": total_latency,
                               "error": type(exc).__name__})
            return []
        total_latency += latency
        content = _content(response)
        try:
            parsed = json.loads(content)
            break
        except (TypeError, json.JSONDecodeError):
            parsed = None

    if parsed is None:
        await _emit(emit, {"type": "tile", "id": chunk.id, "state": "error", "hits": 0,
                           "tokens_in": chunk.est_tokens, "tokens_out": len(content) // 4,
                           "latency_s": total_latency})
        return []

    findings = _ground_findings(parsed, bundle, chunk)
    _, tokens_out = _token_usage(response, chunk, content)
    await _emit(emit, {"type": "tile", "id": chunk.id,
                       "state": "hit" if findings else "clean", "hits": len(findings),
                       "tokens_in": chunk.est_tokens, "tokens_out": tokens_out,
                       "latency_s": total_latency})
    return findings


async def run_swarm(chunks: Iterable[Chunk], bundle: Bundle, emit: Callable | None = None) -> list[dict]:
    """Scan chunks concurrently; the shared LLM semaphore bounds provider load."""
    results = await asyncio.gather(*(scan(chunk, bundle, emit) for chunk in chunks))
    return [finding for group in results for finding in group]
