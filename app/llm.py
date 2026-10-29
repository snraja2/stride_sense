"""OpenAI-compatible LLM client with bounded concurrency and deterministic mock mode."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.config import settings

try:
    from openai import AsyncOpenAI
    from openai import APIStatusError
except ImportError:  # Keeps mock mode importable in a minimal environment.
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc]


@dataclass
class Metrics:
    calls: dict[str, int] = field(default_factory=dict)
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    completion_tokens: dict[str, int] = field(default_factory=dict)
    latency_s: dict[str, float] = field(default_factory=dict)

    def record(self, role: str, prompt: int, completion: int, latency: float) -> None:
        self.calls[role] = self.calls.get(role, 0) + 1
        self.prompt_tokens[role] = self.prompt_tokens.get(role, 0) + prompt
        self.completion_tokens[role] = self.completion_tokens.get(role, 0) + completion
        self.latency_s[role] = self.latency_s.get(role, 0.0) + latency


@dataclass
class StreamDelta:
    text: str = ""
    usage: Any | None = None


metrics = Metrics()
_semaphore = asyncio.Semaphore(max(1, settings.concurrency))
_pause_lock = asyncio.Lock()
_pause_until = 0.0
_client = None


def _client_for_live():
    global _client
    if _client is None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is required for live mode")
        _client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return _client


def _usage(response: Any, prompt_fallback: int = 0, completion_fallback: int = 0) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return prompt_fallback, completion_fallback
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _estimate_messages(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


async def _respect_pause() -> None:
    while True:
        async with _pause_lock:
            delay = _pause_until - time.monotonic()
        if delay <= 0:
            return
        await asyncio.sleep(delay)


def _retry_delay(exc: Exception, attempt: int) -> float:
    delay = min(20.0, 2**attempt) + random.random()
    if getattr(exc, "status_code", None) == 429:
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        try:
            delay = max(delay, float(retry_after))
        except (TypeError, ValueError):
            pass
    return delay


async def _set_pause(delay: float) -> None:
    global _pause_until
    async with _pause_lock:
        _pause_until = max(_pause_until, time.monotonic() + delay)


_FINDING_RE = re.compile(
    r"\b(?:Xid|NCCL\s+WARN|Watchdog|OutOfMemory|nan|Traceback|ECC|Link down|not responding|"
    r"Killed process|SIGTERM|illegal memory|version)\b", re.IGNORECASE
)


def _mock_response(role: str, messages: list[dict]) -> dict:
    source = "\n".join(str(m.get("content", "")) for m in messages)
    if role == "swarm":
        findings = []
        for line in source.splitlines():
            match = re.match(r"L(\d+)\|(.*)", line)
            if not match or not _FINDING_RE.search(match.group(2)):
                continue
            number, quote = int(match.group(1)), match.group(2)[:160]
            findings.append({"line_start": number, "line_end": number, "quote": quote,
                             "kind": "hardware" if re.search(r"Xid|ECC|Link down", quote, re.I) else "error",
                             "severity": 5 if re.search(r"Xid|OutOfMemory|illegal memory", quote, re.I) else 3,
                             "why": "Potentially relevant anomaly in log evidence."})
            if len(findings) == 12:
                break
        return {"findings": findings}
    if role == "eyes":
        return {
            "observations": [{
                "id": "O1",
                "claim": "Synthetic mock observation: dashboard telemetry varies across the GPU grid.",
                "entity": None,
                "metric": "GPU telemetry",
                "time_ref": None,
                "severity": 1,
            }],
            "summary": "Synthetic mock observation; this is not visual analysis.",
        }
    if role == "reduce":
        return _mock_reduce_response(messages)
    if role == "skeptic":
        return {"disagree": False, "alternative": "No independent alternative in mock mode.", "check": "Review source evidence."}
    return {"text": "mock response"}


def _mock_reduce_response(messages: list[dict]) -> dict:
    payload = {}
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("clusters", candidate.get("CLUSTERS")), list):
            payload = candidate
            break

    clusters = payload.get("clusters", payload.get("CLUSTERS", []))
    visual = payload.get("visual", payload.get("VISUAL", {}))
    observations = visual.get("observations", []) if isinstance(visual, dict) else []
    category_rules = (
        ("gpu_fell_off_bus", lambda text: "xid" in text and "79" in text),
        ("ecc_error", lambda text: "xid 48" in text or "double-bit ecc" in text),
        ("ib_link_failure", lambda text: "link down" in text or "net/ib" in text),
        ("cuda_oom", lambda text: "cuda out of memory" in text or "outofmemoryerror" in text),
        ("nan_divergence", lambda text: "loss nan" in text or "floatingpointerror" in text),
        ("filesystem_stall", lambda text: "nfs: server" in text and "not responding" in text),
        ("nccl_version_mismatch", lambda text: "reimaged with older image" in text or "version 2.18.3" in text),
        ("dataloader_oom_kill", lambda text: "killed process" in text or "dataloader worker" in text),
        ("preemption", lambda text: "preempted by higher priority" in text),
        ("illegal_memory_access", lambda text: "illegal memory access" in text or "xid 31" in text),
    )
    flattened = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        for exemplar in cluster.get("exemplars", []):
            if isinstance(exemplar, dict):
                flattened.append((cluster, exemplar))
    text = "\n".join(str(exemplar.get("quote", "")) for _, exemplar in flattened).lower()
    category = next((name for name, matcher in category_rules if matcher(text)), None)
    selected = []
    if category:
        matcher = next(rule for name, rule in category_rules if name == category)
        selected = [pair for pair in flattened if matcher(str(pair[1].get("quote", "")).lower())]
        selected.sort(key=lambda pair: (-int(pair[0].get("severity", 1) or 1), str(pair[0].get("first_ts") or "")))

    if not category and observations:
        visual_text = "\n".join(str(item.get("claim", "")) for item in observations if isinstance(item, dict)).lower()
        if "temperature" in visual_text and ("90" in visual_text or "94" in visual_text):
            category = "thermal_throttle"
        elif "power" in visual_text and "clock" in visual_text:
            category = "power_cap"

    chosen: list[tuple[dict, dict]] = []
    if selected:
        chosen.append(selected[0])
    silence = [pair for pair in flattened if pair[0].get("kind") == "silence"]
    if category and silence:
        silence.sort(key=lambda pair: str(pair[0].get("first_ts") or ""))
        if silence[0] not in chosen:
            chosen.append(silence[0])

    evidence = []
    timeline = []
    for index, (cluster, exemplar) in enumerate(chosen, 1):
        evidence_id = f"E{index}"
        item = {
            "id": evidence_id,
            "file": exemplar.get("file", ""),
            "line_start": exemplar.get("line_start", 0),
            "line_end": exemplar.get("line_end", 0),
            "quote": exemplar.get("quote", ""),
            "observation_id": None,
            "supports": "Direct root-cause evidence." if index == 1 else "Locates the affected rank and corroborates the failure timeline.",
        }
        evidence.append(item)
        timeline.append({"ts": exemplar.get("timestamp") or cluster.get("first_ts") or "", "event": item["supports"], "evidence_ids": [evidence_id]})

    root_node = None
    root_rank = None
    for _, exemplar in chosen:
        file_name = str(exemplar.get("file", ""))
        node_match = re.search(r"dmesg_node-(\d+)", file_name)
        rank_match = re.search(r"rank_(\d+)", file_name)
        if node_match and root_node is None:
            root_node = f"node-{int(node_match.group(1)):02d}"
        if rank_match and root_rank is None:
            root_rank = int(rank_match.group(1))
            root_node = root_node or f"node-{root_rank // 8:02d}"
    root_gpu = root_rank % 8 if root_rank is not None else None
    first_seen = min((item["ts"] for item in timeline if item["ts"]), default="")
    diagnosis = category is not None and bool(evidence)
    summary = {
        "gpu_fell_off_bus": "GPU 5 on node-01 fell off the bus, interrupting its rank and leaving peers to time out.",
        "ecc_error": "A double-bit GPU memory error is the earliest hardware fault in the evidence.",
        "ib_link_failure": "An InfiniBand link failure is the earliest localized network fault in the evidence.",
        "cuda_oom": "A rank ran out of CUDA memory before its peers timed out.",
        "nan_divergence": "A non-finite gradient and loss caused the training ranks to fail.",
        "filesystem_stall": "The node stopped responding to the shared filesystem during checkpoint activity.",
        "nccl_version_mismatch": "A node was reimaged with a different NCCL version before the collective failure.",
        "dataloader_oom_kill": "The operating system killed a data-loader worker after an out-of-memory event.",
        "preemption": "The scheduler preempted a node, interrupting the distributed job.",
        "illegal_memory_access": "A CUDA illegal-memory-access fault occurred on the affected rank.",
        "thermal_throttle": "A GPU thermal anomaly reduced its clock and slowed the training job.",
        "power_cap": "A GPU power limit reduced its clock and slowed the training job.",
    }
    narrative = (
        f"{summary[category]} [E1]"
        if diagnosis else "The available mock evidence does not support a specific root cause; more source data is needed."
    )
    result = {
        "verdict": "diagnosis" if diagnosis else "abstain",
        "root_cause": {
            "summary": summary.get(category, "Mock mode found no grounded root cause."),
            "category": category or "other",
            "node": root_node,
            "gpu": root_gpu,
            "rank": root_rank,
            "first_seen_ts": first_seen,
        },
        "confidence": 0.9 if diagnosis else 0.0,
        "symptoms": [str(cluster.get("kind")) for cluster in clusters if cluster.get("kind") == "timeout"],
        "timeline": timeline,
        "evidence": evidence,
        "ruled_out": [],
        "runbook": ([
            {"step": 1, "command": "nvidia-smi -q", "purpose": "Inspect GPU health before resuming."},
            {"step": 2, "command": "dcgmi diag -r 1", "purpose": "Run a non-destructive diagnostic after review."},
        ] if diagnosis else []),
        "missing_data": None if diagnosis else "A grounded root-cause citation is not available.",
    }
    return {"narrative": narrative, "result": result}


async def call(role: str, messages: list[dict], **kw: Any) -> tuple[Any, float]:
    """Call a configured role and return (response, measured latency seconds)."""
    if role not in settings.models:
        raise ValueError(f"unknown LLM role: {role}")
    async with _semaphore:
        start = time.perf_counter()
        if settings.mode == "mock":
            await asyncio.sleep(0.2 + random.Random(role + str(len(messages))).random() * 0.6)
            response = _mock_response(role, messages)
            prompt_tokens = _estimate_messages(messages)
            completion_tokens = len(json.dumps(response)) // 4
            elapsed = time.perf_counter() - start
            metrics.record(role, prompt_tokens, completion_tokens, elapsed)
            return response, elapsed
        if not settings.api_key:
            raise RuntimeError("Set LLM_API_KEY or NEBIUS_API_KEY when LLM_MODE=live")
        client = _client_for_live()
        request = {"model": settings.models[role], "messages": messages, **kw}
        extra = settings.extra_body(role)
        if extra:
            request["extra_body"] = extra
        for attempt in range(6):
            await _respect_pause()
            try:
                response = await client.chat.completions.create(**request)
                prompt_tokens, completion_tokens = _usage(response, _estimate_messages(messages))
                elapsed = time.perf_counter() - start
                metrics.record(role, prompt_tokens, completion_tokens, elapsed)
                return response, elapsed
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise
                delay = _retry_delay(exc, attempt)
                if status == 429:
                    await _set_pause(delay)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable retry state")


async def call_stream(role: str, messages: list[dict], **kw: Any) -> AsyncIterator[StreamDelta]:
    """Stream text deltas; the final item carries usage when the API provides it."""
    if settings.mode == "mock":
        response, _ = await call(role, messages, **kw)
        if isinstance(response, dict) and "narrative" in response:
            text = response["narrative"] + "\n---JSON---\n" + json.dumps(response["result"])
        else:
            text = json.dumps(response)
        for i in range(0, len(text), 48):
            yield StreamDelta(text=text[i:i + 48])
        yield StreamDelta(usage={"estimated": True})
        return
    client = _client_for_live()
    request = {"model": settings.models[role], "messages": messages, "stream": True,
               "stream_options": {"include_usage": True}, **kw}
    extra = settings.extra_body(role)
    if extra:
        request["extra_body"] = extra
    async with _semaphore:
        start = time.perf_counter()
        prompt_tokens = _estimate_messages(messages)
        completion_chars = 0
        stream = await client.chat.completions.create(**request)
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            choices = getattr(chunk, "choices", []) or []
            for choice in choices:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    completion_chars += len(content)
                    yield StreamDelta(text=content)
            if usage is not None:
                yield StreamDelta(usage=usage)
        metrics.record(role, prompt_tokens, completion_chars // 4, time.perf_counter() - start)
