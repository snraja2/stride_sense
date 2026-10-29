from app.ingest import load_bundle
from app.verify import verify_quote


def test_verifier_accepts_whitespace_normalization_and_rejects_bad_citations(tmp_path):
    (tmp_path / "rank_000.log").write_text("first line\nNVRM:   Xid 79, GPU fell off the bus.\nlast line\n")
    bundle = load_bundle(tmp_path)

    assert verify_quote(bundle, "rank_000.log", 2, 2, "NVRM: Xid 79, GPU fell off the bus.")
    assert verify_quote(bundle, "rank_000.log", 2, 3, "Xid 79, GPU fell off the bus. last line")
    assert not verify_quote(bundle, "rank_000.log", 1, 1, "Xid 79")
    assert not verify_quote(bundle, "rank_000.log", 0, 2, "NVRM: Xid 79")
    assert not verify_quote(bundle, "missing.log", 1, 1, "NVRM: Xid 79")
    assert not verify_quote(bundle, "rank_000.log", 2, 3, "Xid 79" * 40)


def test_dashboard_has_no_verifiable_text_quote(tmp_path):
    (tmp_path / "dashboard.png").write_bytes(b"synthetic-png")
    bundle = load_bundle(tmp_path)
    assert not verify_quote(bundle, "dashboard.png", 1, 1, "synthetic-png")
