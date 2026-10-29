"""Optional dashboard vision analysis; failures never stop the incident run."""

from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from app import llm
from app.ingest import Bundle


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dash.txt"


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
    return content if isinstance(content, str) else ""


async def _emit(emit: Callable | None, event: dict) -> None:
    if emit is None:
        return
    result = emit(event)
    if inspect.isawaitable(result):
        await result


def _normalize(parsed: Any) -> dict:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("observations"), list):
        raise ValueError("vision response has no observations list")
    observations = []
    for index, raw in enumerate(parsed["observations"][:20], 1):
        if not isinstance(raw, dict) or not raw.get("claim"):
            continue
        try:
            severity = max(1, min(5, int(raw.get("severity", 1))))
        except (TypeError, ValueError):
            severity = 1
        observations.append({
            "id": str(raw.get("id") or f"O{index}"),
            "claim": str(raw["claim"])[:500],
            "entity": str(raw["entity"])[:100] if raw.get("entity") is not None else None,
            "metric": str(raw.get("metric") or "GPU telemetry")[:100],
            "time_ref": str(raw["time_ref"])[:100] if raw.get("time_ref") is not None else None,
            "severity": severity,
        })
    return {"observations": observations, "summary": str(parsed.get("summary") or "")[:1000]}


async def analyze_dashboard(bundle: Bundle, emit: Callable | None = None, *, enabled: bool = True) -> dict:
    """Read dashboard.png through the eyes role, returning an empty result on failure."""
    if not enabled:
        return {"observations": [], "summary": "Dashboard analysis disabled."}
    dashboard = next((record for record in bundle.files.values() if record.kind == "dashboard"), None)
    if dashboard is None:
        await _emit(emit, {"type": "warning", "message": "No dashboard image was available for visual analysis."})
        return {"observations": [], "summary": "No dashboard image available."}

    try:
        image_bytes = (bundle.root / dashboard.path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": _PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Read the dashboard and report only concrete visual observations. "
                    + (f"Synthetic mock fixture: {bundle.root.name}." if llm.settings.mode == "mock" else "")
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]},
        ]
        response, _ = await llm.call("eyes", messages, temperature=0, max_tokens=600)
        return _normalize(json.loads(_content(response)))
    except Exception as exc:
        await _emit(emit, {"type": "warning", "message": f"Dashboard analysis unavailable ({type(exc).__name__}); continuing without visual evidence."})
        return {"observations": [], "summary": "Dashboard analysis unavailable."}
