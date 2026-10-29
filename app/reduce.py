"""Stream the final diagnosis, verify its citations, and retry bad evidence."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable

from app import llm
from app.ingest import Bundle, file_stats
from app.verify import verify_quote


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "reduce.txt"
_DELIMITER = "---JSON---"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CATEGORIES = [
    "gpu_fell_off_bus", "ecc_error", "ib_link_failure", "cuda_oom", "nan_divergence",
    "thermal_throttle", "power_cap", "filesystem_stall", "nccl_version_mismatch",
    "dataloader_oom_kill", "preemption", "illegal_memory_access", "other",
]


class _ThinkStripper:
    """Incrementally remove hidden think tags without leaking partial tags."""

    def __init__(self) -> None:
        self.pending = ""
        self.in_think = False

    def feed(self, value: str, *, final: bool = False) -> str:
        self.pending += value
        output = []
        while self.pending:
            if self.in_think:
                close = self.pending.find(_THINK_CLOSE)
                if close >= 0:
                    self.pending = self.pending[close + len(_THINK_CLOSE):]
                    self.in_think = False
                    continue
                if final:
                    self.pending = ""
                else:
                    self.pending = self.pending[-(len(_THINK_CLOSE) - 1):]
                break

            opening = self.pending.find(_THINK_OPEN)
            if opening >= 0:
                output.append(self.pending[:opening])
                self.pending = self.pending[opening + len(_THINK_OPEN):]
                self.in_think = True
                continue
            if final:
                output.append(self.pending)
                self.pending = ""
            else:
                safe = max(0, len(self.pending) - (len(_THINK_OPEN) - 1))
                if safe:
                    output.append(self.pending[:safe])
                    self.pending = self.pending[safe:]
            break
        return "".join(output)


async def _emit(emit: Callable | None, event: dict) -> None:
    if emit is None:
        return
    result = emit(event)
    if inspect.isawaitable(result):
        await result


def _emit_text(emit: Callable | None, text: str, attempt: int) -> list:
    if not text:
        return []
    return [{"type": "narrative_token", "text": text[index:index + 48], "attempt": attempt}
            for index in range(0, len(text), 48)]


async def _stream_once(messages: list[dict], emit: Callable | None, attempt: int) -> tuple[str, dict, int]:
    pre_delimiter = ""
    json_text = ""
    narrative_parts: list[str] = []
    stripper = _ThinkStripper()
    found_delimiter = False

    async for delta in llm.call_stream("reduce", messages, temperature=0, max_tokens=3000):
        text = getattr(delta, "text", "") or ""
        if not text:
            continue
        if found_delimiter:
            json_text += text
            continue
        pre_delimiter += text
        marker_index = pre_delimiter.find(_DELIMITER)
        if marker_index >= 0:
            raw_narrative = pre_delimiter[:marker_index]
            cleaned = stripper.feed(raw_narrative, final=True)
            narrative_parts.append(cleaned)
            for token_event in _emit_text(emit, cleaned, attempt):
                await _emit(emit, token_event)
            json_text = pre_delimiter[marker_index + len(_DELIMITER):]
            pre_delimiter = ""
            found_delimiter = True
            continue

        safe_length = max(0, len(pre_delimiter) - len(_DELIMITER) + 1)
        if safe_length:
            safe_text = pre_delimiter[:safe_length]
            pre_delimiter = pre_delimiter[safe_length:]
            cleaned = stripper.feed(safe_text)
            if cleaned:
                narrative_parts.append(cleaned)
                for token_event in _emit_text(emit, cleaned, attempt):
                    await _emit(emit, token_event)

    if not found_delimiter:
        cleaned = stripper.feed(pre_delimiter, final=True)
        if cleaned:
            narrative_parts.append(cleaned)
            for token_event in _emit_text(emit, cleaned, attempt):
                await _emit(emit, token_event)
        raise ValueError("reducer response did not include the ---JSON--- delimiter")
    try:
        parsed = json.loads(json_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("reducer returned invalid JSON after the delimiter") from exc
    if not isinstance(parsed, dict):
        raise ValueError("reducer JSON must be an object")
    return "".join(narrative_parts).strip(), parsed, len(json_text) // 4


def _schema_message() -> str:
    schema = {
        "verdict": "diagnosis | abstain",
        "root_cause": {"summary": "str", "category": _CATEGORIES, "node": "node-XX|null", "gpu": "int|null", "rank": "int|null", "first_seen_ts": "str"},
        "confidence": "number from 0 to 1",
        "symptoms": ["str"],
        "timeline": [{"ts": "str", "event": "str", "evidence_ids": ["E1"]}],
        "evidence": [{"id": "E1", "file": "str", "line_start": 0, "line_end": 0, "quote": "str", "observation_id": "O1|null", "supports": "str"}],
        "ruled_out": [{"hypothesis": "str", "why": "str", "evidence_ids": ["E2"]}],
        "runbook": [{"step": 1, "command": "str", "purpose": "str"}],
        "missing_data": "str|null",
    }
    return "Return one JSON object matching this schema exactly:\n" + json.dumps(schema, ensure_ascii=False)


def _text_excerpt(bundle: Bundle, kind: str, limit: int) -> str:
    parts = []
    for record in bundle.files.values():
        if record.kind == kind:
            parts.append(f"[{record.path}]\n" + "\n".join(record.lines))
    return "\n".join(parts)[:limit]


def _messages(bundle: Bundle, clusters: list[dict], observations: list[dict]) -> list[dict]:
    job = {
        "scheduler": _text_excerpt(bundle, "scheduler", 5_000),
        "config": _text_excerpt(bundle, "config", 6_000),
    }
    payload = {
        "JOB": job,
        "FILE_STATS": file_stats(bundle),
        "CLUSTERS": clusters,
        "VISUAL": {"observations": observations},
    }
    return [
        {"role": "system", "content": _PROMPT_PATH.read_text(encoding="utf-8")},
        {"role": "user", "content": _schema_message()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _abstain() -> dict:
    return {
        "verdict": "abstain",
        "root_cause": {"summary": "The available evidence is insufficient to make a grounded diagnosis.", "category": "other", "node": None, "gpu": None, "rank": None, "first_seen_ts": ""},
        "confidence": 0.0,
        "symptoms": [],
        "timeline": [],
        "evidence": [],
        "ruled_out": [],
        "runbook": [],
        "missing_data": "A verified root-cause citation is missing.",
    }


def _valid_evidence(result: dict, bundle: Bundle, observations: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = []
    failed = []
    observation_ids = {str(item.get("id")) for item in observations if isinstance(item, dict) and item.get("id")}
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    for item in evidence:
        if not isinstance(item, dict):
            failed.append({"item": item, "reason": "evidence item is not an object"})
            continue
        file = str(item.get("file") or "")
        if file == "dashboard.png":
            okay = str(item.get("observation_id") or "") in observation_ids
            reason = "unknown dashboard observation id"
        else:
            try:
                start = int(item.get("line_start"))
                end = int(item.get("line_end"))
            except (TypeError, ValueError):
                start, end = 0, 0
            okay = verify_quote(bundle, file, start, end, str(item.get("quote") or ""))
            reason = "quote does not match the cited source lines"
        if okay:
            valid.append(item)
        else:
            failed.append({"item": item, "reason": reason})
    return valid, failed


def _clean_result(parsed: dict) -> dict:
    result = parsed if isinstance(parsed, dict) else _abstain()
    if result.get("verdict") not in {"diagnosis", "abstain"}:
        result["verdict"] = "abstain"
    if not isinstance(result.get("root_cause"), dict):
        result["root_cause"] = _abstain()["root_cause"]
    if result["root_cause"].get("category") not in _CATEGORIES:
        result["root_cause"]["category"] = "other"
    for field in ("symptoms", "timeline", "evidence", "ruled_out", "runbook"):
        if not isinstance(result.get(field), list):
            result[field] = []
    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    return result


async def diagnose(bundle: Bundle, clusters: list[dict], observations: list[dict] | None = None,
                   emit: Callable | None = None) -> dict:
    """Generate a diagnosis, retry once for bad evidence, and remove any that remains invalid."""
    observations = observations or []
    messages = _messages(bundle, clusters, observations)
    narrative = ""
    result = _abstain()
    tokens_out = 0
    try:
        narrative, parsed, tokens_out = await _stream_once(messages, emit, 1)
        result = _clean_result(parsed)
    except Exception as exc:
        await _emit(emit, {"type": "warning", "message": f"Reducer response unavailable ({type(exc).__name__}); abstaining."})
        result["verification"] = {"verified": 0, "total": 0, "failed": 0}
        return {"narrative": "The reducer could not produce a verified diagnosis, so the system is abstaining.",
                "result": result, "verification": {"verified": 0, "total": 0, "failed": 0}, "tokens_out": tokens_out}

    valid, failed = _valid_evidence(result, bundle, observations)
    if failed:
        repair = {
            "failed_evidence": failed,
            "instruction": "Fix or remove these items; use only exemplar text from the supplied clusters. Return the same narrative and JSON delimiter format.",
        }
        retry_messages = messages + [{"role": "user", "content": json.dumps(repair, ensure_ascii=False)}]
        try:
            retry_narrative, retry_parsed, retry_tokens = await _stream_once(retry_messages, emit, 2)
            narrative, result, tokens_out = retry_narrative, _clean_result(retry_parsed), retry_tokens
            valid, failed = _valid_evidence(result, bundle, observations)
        except Exception as exc:
            await _emit(emit, {"type": "warning", "message": f"Evidence repair failed ({type(exc).__name__}); removing invalid citations."})

    total = len(result.get("evidence", []))
    result["evidence"] = valid
    root_support = any(
        re.search(r"root[- ]cause|direct root", str(item.get("supports", "")), re.IGNORECASE)
        for item in valid
    )
    if result.get("verdict") == "diagnosis" and (not valid or not root_support):
        result["verdict"] = "abstain"
        result["confidence"] = 0.0
        result["missing_data"] = "No verified evidence supports the proposed root cause."
        if not valid:
            narrative = "No source citation could be verified, so the system is abstaining from a root-cause diagnosis."
    result["verification"] = {
        "verified": len(valid),
        "total": total,
        "failed": max(0, total - len(valid)),
    }
    return {"narrative": narrative, "result": result, "verification": result["verification"], "tokens_out": tokens_out}
