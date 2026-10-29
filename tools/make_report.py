"""Summarize recorded evaluation rows into a markdown table and PNG chart."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("records", []) if not row.get("skipped")]
    systems = payload.get("systems", sorted({row.get("system", "unknown") for row in rows}))
    lines = [
        f"# Evaluation results ({payload.get('mode', 'unknown')} mode)",
        "",
        "These values summarize the rows recorded in `results.json`; mock-mode results are workflow checks, not model-quality or provider performance measurements.",
        "",
        "| System | Runs | Category accuracy | Location accuracy | Median time (s) | Mean cost (USD) | Citation validity | Abstain rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    chart = {"labels": [], "category_accuracy": [], "location_accuracy": []}
    for system in systems:
        group = [row for row in rows if row.get("system") == system]
        if not group:
            continue
        count = len(group)
        cat = sum(bool(row.get("category_correct")) for row in group) / count
        loc = sum(bool(row.get("location_correct")) for row in group) / count
        times = [float(row.get("elapsed_s", 0)) for row in group]
        costs = [float(row.get("estimated_cost_usd", 0)) for row in group]
        citations = sum(int(row.get("verified_citations", 0)) for row in group)
        citation_total = sum(int(row.get("total_citations", 0)) for row in group)
        validity = citations / citation_total if citation_total else 0.0
        abstain = sum(row.get("verdict") == "abstain" for row in group) / count
        lines.append(f"| {system} | {count} | {cat:.1%} | {loc:.1%} | {statistics.median(times):.2f} | {statistics.mean(costs):.4f} | {validity:.1%} | {abstain:.1%} |")
        chart["labels"].append(system)
        chart["category_accuracy"].append(cat)
        chart["location_accuracy"].append(loc)
    if payload.get("sequential_vs_parallel_s"):
        timing = payload["sequential_vs_parallel_s"]
        lines.extend(["", f"Recorded full-pipeline elapsed time at swarm concurrency 1: {timing.get('sequential_s', 'not recorded')} s; at concurrency {timing.get('parallel_concurrency', 'not recorded')}: {timing.get('parallel_s', 'not recorded')} s."])
    lines.extend(["", f"Evaluation mode: **{payload.get('mode', 'unknown')}**. Live provider results must be recorded separately before using these values as model benchmarks."])
    return "\n".join(lines) + "\n", chart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--out", type=Path, default=Path("eval"))
    args = parser.parse_args()
    report, chart = summarize(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.md").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = list(range(len(chart["labels"])))
    width = 0.36
    ax.bar([i - width / 2 for i in x], chart["category_accuracy"], width, label="Category")
    ax.bar([i + width / 2 for i in x], chart["location_accuracy"], width, label="Location")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, chart["labels"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(args.out / "chart.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out / 'RESULTS.md'} and {args.out / 'chart.png'}")


if __name__ == "__main__":
    main()
