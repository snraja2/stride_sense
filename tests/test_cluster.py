from app.cluster import cluster_findings, normalize_quote


def test_watchdog_duplicates_cluster_and_keep_earliest_exemplar():
    findings = [
        {
            "file": f"logs/rank_{rank:03d}.log",
            "line_start": 200 + rank,
            "line_end": 200 + rank,
            "timestamp": f"2026-10-01T13:05:{rank:02d}+00:00",
            "kind": "timeout",
            "severity": 3 + (rank == 0),
            "quote": f"rank={rank} node-{rank // 8:02d} Watchdog timeout 600{rank:03d} ms; fault at 0x{rank + 10:x}",
        }
        for rank in range(63)
    ]

    clusters = cluster_findings(findings)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["count"] == 63
    assert cluster["file_count"] == 63
    assert len(cluster["files"]) == 10
    assert len(cluster["exemplars"]) == 3
    assert cluster["exemplars"][0]["file"] == "logs/rank_000.log"
    assert cluster["severity"] == 4


def test_normalization_removes_rank_node_and_number_variation():
    assert normalize_quote("rank=013 node-01 CUDA Xid 79 at 0x8a00") == normalize_quote(
        "rank=014 node-02 CUDA Xid 48 at 0x8a01"
    )


def test_singleton_high_severity_clusters_survive_the_limit():
    findings = [
        {
            "file": f"f{i}.log",
            "kind": "error",
            "severity": 4,
            "quote": f"unique failure {chr(65 + i // 26)}{chr(65 + i % 26)}",
        }
        for i in range(85)
    ]

    clusters = cluster_findings(findings, limit=80)

    assert len(clusters) == 85
    assert {item["count"] for item in clusters} == {1}
