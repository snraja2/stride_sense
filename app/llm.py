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
    r"Xid|NCCL WARN|Watchdog|OutOfMemory|nan|Traceback|ECC|Link down|not responding|"
    r"Killed process|SIGTERM|illegal memory|version", re.IGNORECASE
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
        return {"observations": [], "summary": "Synthetic mock observation: dashboard inspected."}
    if role == "reduce":
        result = {
            "verdict": "abstain",
            "root_cause": {"summary": "Mock mode found no grounded root cause.", "category": "other", "node": None, "gpu": None, "rank": None, "first_seen_ts": ""},
            "confidence": 0.0, "symptoms": [], "timeline": [], "evidence": [], "ruled_out": [],
            "runbook": [], "missing_data": "No verified evidence was supplied to the mock responder.",
        }
        return {"narrative": "Mock mode could not identify a grounded root cause.", "result": result}
    if role == "skeptic":
        return {"disagree": False, "alternative": "No independent alternative in mock mode.", "check": "Review source evidence."}
    return {"text": "mock response"}


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
