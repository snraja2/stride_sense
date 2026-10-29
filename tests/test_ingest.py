from datetime import datetime, timezone

from app.ingest import file_stats, load_bundle, parse_timestamp, silence_findings
from tools.make_incident import make_incident


def test_parse_timestamp_handles_iso_z_and_invalid_lines():
    value = parse_timestamp("2026-10-01T12:34:56.123Z INFO ready")
    assert value == datetime(2026, 10, 1, 12, 34, 56, 123000, tzinfo=timezone.utc)
    assert parse_timestamp("2026-10-01 12:34:56") == datetime(2026, 10, 1, 12, 34, 56, tzinfo=timezone.utc)
    assert parse_timestamp("ordinary line") is None
    assert parse_timestamp("2026-99-99T12:34:56Z bad date") is None


def test_bundle_excludes_truth_and_chunks_at_line_boundaries(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "rank_013.log").write_text("\n".join(f"2026-10-01T12:00:{i:02d}Z message-{i:02d}" for i in range(30)))
    (tmp_path / "truth.json").write_text('{"secret":true}')

    bundle = load_bundle(tmp_path, max_chunk_tokens=20)
    repeated = load_bundle(tmp_path, max_chunk_tokens=20)

    assert "truth.json" not in bundle.files
    assert len(bundle.chunks) > 1
    assert [chunk.id for chunk in bundle.chunks] == [chunk.id for chunk in repeated.chunks]
    assert all(chunk.rank == 13 and chunk.kind == "rank_log" for chunk in bundle.chunks)
    assert all(chunk.est_tokens <= 20 for chunk in bundle.chunks)
    assert bundle.chunks[0].line_start == 1
    assert bundle.chunks[-1].line_end == 30
    assert all(chunk.line_end + 1 == next_chunk.line_start for chunk, next_chunk in zip(bundle.chunks, bundle.chunks[1:]))


def test_file_stats_and_silence_detect_only_demo_rank_013(tmp_path):
    root = make_incident("xid79_demo", tmp_path, 7)
    bundle = load_bundle(root)

    stats = file_stats(bundle)
    findings = silence_findings(bundle)

    assert stats["logs/rank_013.log"]["last_ts"] == "2026-10-01T12:55:00+00:00"
    assert [item["file"] for item in findings] == ["logs/rank_013.log"]
    assert findings[0]["severity"] == 4
    assert findings[0]["quote"] == bundle.files["logs/rank_013.log"].lines[findings[0]["line_start"] - 1]
