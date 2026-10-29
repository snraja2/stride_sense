"""Load incident bundles into numbered, timestamped text and stable chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


_TIMESTAMP = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_RANK_FILE = re.compile(r"(?:^|/)rank_(\d+)\.log$", re.IGNORECASE)
_NODE_FILE = re.compile(r"(?:^|/)dmesg_node-(\d+)\.log$", re.IGNORECASE)
_STEP = re.compile(r"\bstep\s+(\d+)\b", re.IGNORECASE)
DEFAULT_CHUNK_TOKENS = 16_000


def parse_timestamp(line: str) -> datetime | None:
    """Parse a leading ISO-like timestamp, returning None for ordinary lines."""
    match = _TIMESTAMP.match(line)
    if not match:
        return None
    value = match.group(1).replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", value):
        value = value[:-2] + ":" + value[-2:]
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass
class FileRecord:
    path: str
    kind: str
    rank: int | None = None
    node: int | None = None
    lines: list[str] = field(default_factory=list)
    numbered_lines: list[str] = field(default_factory=list)
    timestamps: list[datetime | None] = field(default_factory=list)
    est_tokens: int = 0


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    kind: str
    rank: int | None
    node: int | None
    line_start: int
    line_end: int
    text: str
    est_tokens: int


@dataclass
class Bundle:
    root: Path
    files: dict[str, FileRecord]
    chunks: list[Chunk]
    est_tokens: int


def _classify(relpath: str) -> tuple[str, int | None, int | None]:
    rank_match = _RANK_FILE.search(relpath)
    if rank_match:
        rank = int(rank_match.group(1))
        return "rank_log", rank, rank // 8
    node_match = _NODE_FILE.search(relpath)
    if node_match:
        return "dmesg", None, int(node_match.group(1))
    name = Path(relpath).name.lower()
    if name == "scheduler.log":
        return "scheduler", None, None
    if name == "dashboard.png":
        return "dashboard", None, None
    if Path(relpath).suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".txt"}:
        return "config", None, None
    return "unknown", None, None


def _chunks_for(record: FileRecord, max_tokens: int) -> list[Chunk]:
    if not record.numbered_lines:
        return []
    max_chars = max_tokens * 4
    result: list[Chunk] = []
    current: list[str] = []
    current_chars = 0
    start_line = 1

    def flush() -> None:
        nonlocal current, current_chars, start_line
        if not current:
            return
        end_line = start_line + len(current) - 1
        text = "\n".join(current)
        digest = hashlib.sha256(f"{record.path}:{start_line}:{end_line}:{text}".encode()).hexdigest()[:16]
        result.append(Chunk(
            id=digest,
            file=record.path,
            kind=record.kind,
            rank=record.rank,
            node=record.node,
            line_start=start_line,
            line_end=end_line,
            text=text,
            est_tokens=len(text) // 4,
        ))
        start_line = end_line + 1
        current = []
        current_chars = 0

    for numbered in record.numbered_lines:
        extra = len(numbered) + (1 if current else 0)
        if current and current_chars + extra > max_chars:
            flush()
            extra = len(numbered)
        current.append(numbered)
        current_chars += extra
    flush()
    return result


def load_bundle(path: str | Path, *, max_chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Bundle:
    """Load supported bundle files, explicitly excluding every truth.json."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"bundle directory does not exist: {path}")
    if max_chunk_tokens < 1:
        raise ValueError("max_chunk_tokens must be positive")

    files: dict[str, FileRecord] = {}
    chunks: list[Chunk] = []
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.name.lower() == "truth.json":
            continue
        relpath = source.relative_to(root).as_posix()
        kind, rank, node = _classify(relpath)
        record = FileRecord(path=relpath, kind=kind, rank=rank, node=node)
        if kind != "dashboard":
            try:
                record.lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            record.numbered_lines = [f"L{number:05d}|{line}" for number, line in enumerate(record.lines, 1)]
            record.timestamps = [parse_timestamp(line) for line in record.lines]
            numbered_text = "\n".join(record.numbered_lines)
            record.est_tokens = len(numbered_text) // 4
            chunks.extend(_chunks_for(record, max_chunk_tokens))
        files[relpath] = record
    chunks.sort(key=lambda chunk: (chunk.file, chunk.line_start))
    return Bundle(root=root, files=files, chunks=chunks, est_tokens=sum(chunk.est_tokens for chunk in chunks))


def file_stats(bundle: Bundle) -> dict[str, dict[str, str | int | None]]:
    """Summarize timestamp bounds and the last training step for each file."""
    stats: dict[str, dict[str, str | int | None]] = {}
    for path, record in bundle.files.items():
        found = [value for value in record.timestamps if value is not None]
        last_step = None
        for line in reversed(record.lines):
            match = _STEP.search(line)
            if match:
                last_step = int(match.group(1))
                break
        stats[path] = {
            "first_ts": found[0].isoformat() if found else None,
            "last_ts": found[-1].isoformat() if found else None,
            "last_step": last_step,
        }
    return stats


def silence_findings(bundle: Bundle) -> list[dict]:
    """Find rank logs that stop more than two minutes before peer logs."""
    rank_records = [record for record in bundle.files.values() if record.kind == "rank_log"]
    last_values = []
    for record in rank_records:
        last_ts = next((value for value in reversed(record.timestamps) if value is not None), None)
        if last_ts is not None:
            last_values.append(last_ts)
    if not last_values:
        return []
    last_values.sort()
    middle = len(last_values) // 2
    if len(last_values) % 2:
        median = last_values[middle]
    else:
        median = last_values[middle - 1] + (last_values[middle] - last_values[middle - 1]) / 2

    findings = []
    for record in rank_records:
        last_ts = next((value for value in reversed(record.timestamps) if value is not None), None)
        last_line_index = next((i for i in range(len(record.lines) - 1, -1, -1) if record.lines[i].strip()), None)
        if last_ts is None or last_line_index is None:
            continue
        seconds = (median - last_ts).total_seconds()
        if seconds <= 120:
            continue
        findings.append({
            "file": record.path,
            "kind": "silence",
            "severity": 4,
            "line_start": last_line_index + 1,
            "line_end": last_line_index + 1,
            "quote": record.lines[last_line_index],
            "timestamp": last_ts.isoformat(),
            "why": f"log ends {int(seconds)} s before peers",
        })
    return sorted(findings, key=lambda item: (item["timestamp"], item["file"]))
