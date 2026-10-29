"""Run a small, reproducible incident evaluation (truth is read only here)."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm
from app.config import CATEGORIES, settings
from app.ingest import load_bundle
from app.pipeline import run as run_pipeline
from tools.make_incident import FAULTS


def _json_response(response) -> dict:
    if isinstance(response, dict):
        return response
    choices = getattr(response, "choices", []) or []
    content = getattr(getattr(choices[0], "message", None), "content", "") if choices else ""
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {}


def _snapshot() -> dict:
    return {
        "calls": dict(llm.metrics.calls), "prompt_tokens": dict(llm.metrics.prompt_tokens),
        "completion_tokens": dict(llm.metrics.completion_tokens), "latency_s": dict(llm.metrics.latency_s),
    }


def _usage(before: dict, after: dict) -> dict:
    roles = set(before["calls"]) | set(after["calls"])
    return {
        role: {
            "calls": after["calls"].get(role, 0) - before["calls"].get(role, 0),
            "prompt_tokens": after["prompt_tokens"].get(role, 0) - before["prompt_tokens"].get(role, 0),
            "completion_tokens": after["completion_tokens"].get(role, 0) - before["completion_tokens"].get(role, 0),
            "latency_s": round(after["latency_s"].get(role, 0) - before["latency_s"].get(role, 0), 4),
        } for role in sorted(roles)
    }


def _cost(usage: dict) -> float:
    prices = {
        "reduce": (settings.price_reduce_input, settings.price_reduce_output),
        "baseline": (settings.price_reduce_input, settings.price_reduce_output),
        "swarm": (settings.price_swarm_input, settings.price_swarm_output),
        "eyes": (settings.price_eyes_input, settings.price_eyes_output),
        "skeptic": (settings.price_skeptic_input, settings.price_skeptic_output),
    }
    return sum((row["prompt_tokens"] * prices[role][0] + row["completion_tokens"] * prices[role][1]) / 1_000_000
               for role, row in usage.items() if role in prices)


def _correct(result: dict, truth: dict) -> tuple[bool, bool]:
    root = result.get("root_cause", {}) if isinstance(result, dict) else {}
    category_ok = root.get("category") == truth.get("category")
    location_keys = [key for key in ("node", "gpu") if truth.get(key) is not None]
    location_ok = bool(location_keys) and all(root.get(key) == truth.get(key) for key in location_keys)
    return category_ok, location_ok


async def _run_one(incident_id: str, system: str, incident_root: Path) -> dict:
    bundle_root = incident_root / incident_id
    truth_path = bundle_root / "truth.json"  # eval harness is the only non-badge reader
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    bundle = load_bundle(bundle_root)
    before = _snapshot()
    started = time.monotonic()
    report: dict = {}
    if system in {"crashcart", "no_eyes"}:
        report = await run_pipeline(bundle, {"use_eyes": system != "no_eyes", "baseline": True})
        result = report.get("diagnosis", {})
    elif system == "tail":
        record = bundle.files.get("logs/rank_000.log")
        tail = "\n".join(record.lines[-200:]) if record else ""
        response, _ = await llm.call("baseline", [
            {"role": "system", "content": f"Return JSON with a diagnosis category from: {', '.join(CATEGORIES)}. Use only the final log lines."},
            {"role": "user", "content": tail},
        ], temperature=0, max_tokens=500)
        answer = response if isinstance(response, dict) else _json_response(response)
        result = {"verdict": "diagnosis" if answer.get("category") in CATEGORIES else "abstain",
                  "root_cause": {"category": answer.get("category", "other"), "node": answer.get("node"), "gpu": answer.get("gpu")},
                  "verification": {"verified": 0, "total": 0}}
    elif system == "full_ultra":
        if bundle.est_tokens > 1_000_000:
            return {"incident_id": incident_id, "system": system, "skipped": "bundle exceeds 1.0M estimated tokens"}
        source = "\n\n".join(f"### {path}\n" + "\n".join(record.lines) for path, record in bundle.files.items() if record.kind != "dashboard")
        response, _ = await llm.call("reduce", [
            {"role": "system", "content": f"Diagnose this training failure. Choose one category: {', '.join(CATEGORIES)}. Return JSON."},
            {"role": "user", "content": source},
        ], temperature=0, max_tokens=1200)
        answer = response if isinstance(response, dict) else _json_response(response)
        root = answer.get("root_cause", {}) if isinstance(answer.get("root_cause"), dict) else answer
        result = {"verdict": answer.get("verdict", "abstain"), "root_cause": root,
                  "verification": {"verified": 0, "total": 0}}
    else:
        raise ValueError(f"unknown system: {system}")
    after = _snapshot()
    usage = _usage(before, after)
    category_ok, location_ok = _correct(result, truth)
    return {
        "incident_id": incident_id, "system": system, "mode": llm.settings.mode,
        "category_correct": category_ok, "location_correct": location_ok,
        "elapsed_s": round(time.monotonic() - started, 4),
        "tokens_by_role": usage,
        "estimated_cost_usd": round(_cost(usage), 8),
        "verified_citations": result.get("verification", {}).get("verified", 0),
        "total_citations": result.get("verification", {}).get("total", 0),
        "verdict": result.get("verdict", "abstain"), "predicted_category": result.get("root_cause", {}).get("category"),
    }


async def _compare_swarm(bundle_root: Path) -> dict:
    bundle = load_bundle(bundle_root)
    parallel = max(1, min(24, settings.concurrency))
    report = {"parallel_concurrency": parallel}
    for concurrency in dict.fromkeys((1, parallel)):
        started = time.monotonic()
        await run_pipeline(bundle, {"swarm_concurrency": concurrency, "use_eyes": False, "baseline": False})
        report["sequential_s" if concurrency == 1 else "parallel_s"] = round(time.monotonic() - started, 4)
    return report


async def evaluate(args) -> dict:
    incident_root = Path(args.incidents_dir)
    ids = list(FAULTS) if args.incidents == "all" else [item.strip() for item in args.incidents.split(",") if item.strip()]
    unknown = sorted(set(ids) - set(FAULTS))
    if unknown:
        raise ValueError(f"unknown incident ids: {', '.join(unknown)}")
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    allowed = {"crashcart", "tail", "no_eyes", "full_ultra"}
    if set(systems) - allowed:
        raise ValueError(f"unknown systems: {', '.join(sorted(set(systems) - allowed))}")
    if "full_ultra" in systems:
        ids = ids[:max(0, args.full_limit)]
    records = []
    for incident_id in ids:
        for system in systems:
            records.append(await _run_one(incident_id, system, incident_root))
    result = {"mode": llm.settings.mode, "systems": systems, "incidents": ids, "records": records}
    if args.sequential_compare:
        compare_id = ids[0] if args.sequential_compare == 1 else str(args.sequential_compare)
        result["sequential_vs_parallel_s"] = await _compare_swarm(incident_root / compare_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incidents", default="all", help="all or comma-separated sample ids")
    parser.add_argument("--systems", default="crashcart,tail,no_eyes,full_ultra")
    parser.add_argument("--full-limit", type=int, default=4)
    parser.add_argument("--incidents-dir", default=settings.incidents_dir)
    parser.add_argument("--out", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--sequential-compare", type=int, default=0,
                        help="also time one full-pipeline run with sequential and default swarm concurrency")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['records'])} {result['mode']} evaluation rows to {args.out}")
    if "sequential_vs_parallel_s" in result:
        print(f"Swarm sequential/parallel elapsed seconds: {result['sequential_vs_parallel_s']}")


if __name__ == "__main__":
    main()
