"""Render verified incident results and a command-filtered dry-run runbook."""

from __future__ import annotations

import re
from pathlib import Path


_FORBIDDEN = re.compile(r"\brm\s+-rf\b|\bmkfs(?:\.[\w-]+)?\b|\bdd\s+if\s*=|\bshutdown\b|\bkill\s+-9\b", re.IGNORECASE)
_REBOOT = re.compile(r"\breboot\b", re.IGNORECASE)


def safe_runbook_items(result: dict) -> list[dict]:
    """Drop dangerous commands; allow reboot only when visibly labeled a suggestion."""
    items = result.get("runbook", []) if isinstance(result, dict) else []
    safe = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "").strip()
        if not command or _FORBIDDEN.search(command):
            continue
        if _REBOOT.search(command) and not command.upper().startswith("SUGGESTION:"):
            continue
        safe.append({**item, "command": command})
    return safe


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def render_runbook(result: dict) -> str:
    lines = [
        "# Dry-run runbook",
        "",
        "> **DRY-RUN — review before running.** No command was executed.",
        "",
    ]
    items = safe_runbook_items(result)
    if not items:
        lines.append("No safe runbook commands were generated. Review the diagnosis and plan checks manually.")
    for index, item in enumerate(items, 1):
        lines.extend([
            f"{index}. **{_cell(item.get('purpose') or 'Review this diagnostic step.')}**",
            f"   ```text\n   {_cell(item.get('command'))}\n   ```",
        ])
    return "\n".join(lines) + "\n"


def render_postmortem(result: dict, narrative: str = "") -> str:
    root = result.get("root_cause", {}) if isinstance(result, dict) else {}
    lines = [
        "# Incident postmortem",
        "",
        f"**Verdict:** {_cell(result.get('verdict', 'abstain')).upper()}",
        f"**Root cause:** {_cell(root.get('summary') or 'No verified root cause.')}",
        f"**Category:** {_cell(root.get('category') or 'other')}",
        "",
        "## Summary",
        "",
        _cell(narrative or root.get("summary") or "The system abstained because the available evidence was insufficient."),
        "",
        "## Timeline",
        "",
    ]
    timeline = result.get("timeline", []) if isinstance(result.get("timeline"), list) else []
    if timeline:
        lines.extend(["| Time | Event | Evidence |", "|---|---|---|"])
        for item in timeline:
            ids = item.get("evidence_ids", [])
            evidence_ids = ", ".join(str(value) for value in ids) if isinstance(ids, list) else _cell(ids)
            lines.append(f"| {_cell(item.get('ts'))} | {_cell(item.get('event'))} | {_cell(evidence_ids)} |")
    else:
        lines.append("No verified timeline events were available.")
    lines.extend(["", "## Evidence", ""])
    evidence = result.get("evidence", []) if isinstance(result.get("evidence"), list) else []
    if evidence:
        lines.extend(["| ID | Source | Quote | Supports |", "|---|---|---|---|"])
        for item in evidence:
            file = _cell(item.get("file"))
            start = item.get("line_start")
            end = item.get("line_end")
            source = f"{file}:L{start}" if start else file
            if end and end != start:
                source += f"-L{end}"
            if item.get("observation_id"):
                source += f" ({_cell(item['observation_id'])})"
            lines.append(f"| {_cell(item.get('id'))} | {source} | {_cell(item.get('quote'))} | {_cell(item.get('supports'))} |")
    else:
        lines.append("No verified citations were retained.")
    lines.extend(["", "## Ruled out", ""])
    ruled_out = result.get("ruled_out", []) if isinstance(result.get("ruled_out"), list) else []
    lines.extend([f"- **{_cell(item.get('hypothesis'))}:** {_cell(item.get('why'))}" for item in ruled_out] or ["No hypotheses were ruled out."])
    lines.extend(["", "## Runbook", "", "> **DRY-RUN — review before running.** No command was executed.", ""])
    safe_items = safe_runbook_items(result)
    if not safe_items:
        lines.append("No safe runbook commands were generated. Review the diagnosis and plan checks manually.")
    for index, item in enumerate(safe_items, 1):
        lines.extend([
            f"{index}. **{_cell(item.get('purpose') or 'Review this diagnostic step.')}**",
            f"   ```text\n   {_cell(item.get('command'))}\n   ```",
        ])
    verification = result.get("verification")
    if isinstance(verification, dict):
        lines.extend(["", f"**Citation verification:** {verification.get('verified', 0)}/{verification.get('total', 0)} verified."])
    return "\n".join(lines).rstrip() + "\n"


def write_exports(directory: str | Path, result: dict, narrative: str = "") -> dict[str, str]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    postmortem = render_postmortem(result, narrative)
    runbook = render_runbook(result)
    (output / "postmortem.md").write_text(postmortem, encoding="utf-8")
    (output / "runbook.md").write_text(runbook, encoding="utf-8")
    return {"postmortem": postmortem, "runbook": runbook}
