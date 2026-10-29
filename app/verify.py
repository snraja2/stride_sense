"""Strict source-quote verification shared by workers and final reports."""

from __future__ import annotations

from app.ingest import Bundle


def _compact(value: str) -> str:
    return " ".join(value.split())


def verify_quote(bundle: Bundle, file: str, line_start: int, line_end: int, quote: str) -> bool:
    """Accept only a short quote that appears verbatim within the cited lines."""
    if not file or not isinstance(quote, str) or not quote.strip() or len(quote) > 200:
        return False
    if line_start < 1 or line_end < line_start:
        return False
    record = bundle.files.get(file.replace("\\", "/"))
    if record is None or record.kind == "dashboard" or line_end > len(record.lines):
        return False
    source = _compact(" ".join(record.lines[line_start - 1:line_end]))
    return _compact(quote) in source
