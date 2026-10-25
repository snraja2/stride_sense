import json
from pathlib import Path

from tools.make_incident import make_incident


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
