import json
from pathlib import Path

from tools.make_incident import FAULTS, make_incident


def test_generator_is_deterministic_and_truth_line_exists(tmp_path):
    first = make_incident("xid79_demo", tmp_path / "one", 7)
    second = make_incident("xid79_demo", tmp_path / "two", 7)
    truth = json.loads((first / "truth.json").read_text())
    source = first / truth["evidence_file"]
    lines = source.read_text().splitlines()
    assert truth["category"] == "gpu_fell_off_bus"
    assert "Xid" in lines[truth["evidence_line"] - 1]
    assert (first / "logs/rank_013.log").read_bytes() == (second / "logs/rank_013.log").read_bytes()
    assert (first / "dashboard.png").read_bytes() == (second / "dashboard.png").read_bytes()


def test_every_fault_has_real_truth_evidence_and_expected_text_volume(tmp_path):
    expected_evidence = {
        "xid79_demo": "Xid (PCI:0000:8a:00): 79",
        "ecc_dbe": "Xid 48",
        "ib_flap": "Link down",
        "oom_one_rank": "CUDA out of memory",
        "nan_divergence": "loss nan",
        "thermal_throttle": "DUE TO TIME LIMIT",
        "power_cap": "DUE TO TIME LIMIT",
        "fs_stall": "nfs: server fs01 not responding",
        "nccl_version_mismatch": "NCCL 2.18.3",
        "dataloader_oom_kill": "Killed process 9931",
        "preemption": "Job preempted by higher priority job",
        "illegal_access": "Xid 31: MMU fault",
    }
    for fault_id in FAULTS:
        root = make_incident(fault_id, tmp_path / fault_id, 7)
        truth = json.loads((root / "truth.json").read_text())
        evidence_lines = (root / truth["evidence_file"]).read_text().splitlines()

        assert 1 <= truth["evidence_line"] <= len(evidence_lines), fault_id
        assert expected_evidence[fault_id] in evidence_lines[truth["evidence_line"] - 1], fault_id

        text_files = [*root.rglob("*.log"), *root.rglob("*.yaml")]
        estimated_tokens = sum(path.stat().st_size for path in text_files) // 4
        assert 700_000 <= estimated_tokens <= 900_000, (fault_id, estimated_tokens)


def test_demo_rank_stops_at_xid_and_only_peers_timeout(tmp_path):
    root = make_incident("xid79_demo", tmp_path, 7)
    faulty = (root / "logs/rank_013.log").read_text().splitlines()
    peer = (root / "logs/rank_012.log").read_text()

    assert "2026-10-01T12:55:00Z" in faulty[-1]
    assert "Watchdog" not in "\n".join(faulty)
    assert "Watchdog caught collective operation timeout" in peer
