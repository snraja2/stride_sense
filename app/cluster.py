"""Normalize and deduplicate grounded findings from log workers."""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Iterable


_HEX = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
_RANK = re.compile(r"\brank(?:\s*[_=:#-]?\s*)\d+\b", re.IGNORECASE)
_NODE = re.compile(r"\bnode(?:\s*[_=:#-]?\s*)\d+\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def normalize_quote(quote: str) -> str:
    """Normalize volatile identifiers and numbers while preserving wording."""
    value = _HEX.sub("0x#", quote)
    value = _RANK.sub("rank #", value)
    value = _NODE.sub("node #", value)
    value = _NUMBER.sub("#", value)
    return _SPACE.sub(" ", value).strip().lower()


def _timestamp(finding: dict) -> str | None:
    value = finding.get("timestamp") or finding.get("ts") or finding.get("first_ts")
    if not value:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _ts_key(value: str | None) -> tuple[int, str]:
    if value is None:
        return (1, "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, parsed.astimezone(timezone.utc).isoformat())
    except ValueError:
        return (0, value)


def cluster_findings(findings: Iterable[dict], *, limit: int = 80) -> list[dict]:
    """Group duplicate findings by kind and normalized quote."""
    if limit < 1:
        raise ValueError("limit must be positive")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for original in findings:
        finding = dict(original)
        kind = str(finding.get("kind") or "unknown").strip().lower()
        quote = str(finding.get("quote") or "")
        groups[(kind, normalize_quote(quote))].append(finding)

    clusters = []
    for (kind, key), members in groups.items():
        ordered = sorted(
            members,
            key=lambda item: (
                _ts_key(_timestamp(item)),
                str(item.get("file", "")),
                int(item.get("line_start", 0) or 0),
            ),
        )
        timestamps = [_timestamp(item) for item in ordered if _timestamp(item) is not None]
        files = sorted({str(item.get("file", "")) for item in members if item.get("file")})
        severity = max((int(item.get("severity", 1) or 1) for item in members), default=1)
        exemplars = [deepcopy(item) for item in ordered[:3]]
        clusters.append({
            "kind": kind,
            "key": key,
            "count": len(members),
            "files": files[:10],
            "file_count": len(files),
            "first_ts": min(timestamps, key=_ts_key) if timestamps else None,
            "last_ts": max(timestamps, key=_ts_key) if timestamps else None,
            "severity": severity,
            "exemplars": exemplars,
        })

    clusters.sort(key=lambda item: (-item["severity"], _ts_key(item["first_ts"]), item["kind"], item["key"]))
    if len(clusters) <= limit:
        return clusters
    selected = clusters[:limit]
    selected_keys = {(item["kind"], item["key"]) for item in selected}
    protected = [
        item for item in clusters[limit:]
        if item["count"] == 1 and item["severity"] >= 4
        and (item["kind"], item["key"]) not in selected_keys
    ]
    return selected + protected
