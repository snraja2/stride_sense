"""Generate deterministic synthetic multi-node GPU training incidents."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.make_dashboard import render_dashboard


FAULTS = {
    "xid79_demo": ("gpu_fell_off_bus", "NVRM: Xid (PCI:0000:8a:00): 79, pid=4128, name=python, GPU has fallen off the bus.", False),
    "ecc_dbe": ("ecc_error", "NVRM: Xid 48: double-bit ECC error on framebuffer memory", False),
    "ib_flap": ("ib_link_failure", "mlx5_core 0000:af:00.0: Link down", False),
    "oom_one_rank": ("cuda_oom", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", False),
    "nan_divergence": ("nan_divergence", "FloatingPointError: loss is nan after gnorm inf", False),
    "thermal_throttle": ("thermal_throttle", "DUE TO TIME LIMIT after GPU slowdown", True),
    "power_cap": ("power_cap", "DUE TO TIME LIMIT after GPU slowdown", True),
    "fs_stall": ("filesystem_stall", "nfs: server fs01 not responding, still trying", False),
    "nccl_version_mismatch": ("nccl_version_mismatch", "NCCL version 2.18.3", False),
    "dataloader_oom_kill": ("dataloader_oom_kill", "Out of memory: Killed process 9931 (pt_data_worker)", False),
    "preemption": ("preemption", "Job preempted by higher priority job", False),
    "illegal_access": ("illegal_memory_access", "CUDA error: an illegal memory access was encountered", False),
}
BASE_TIME = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def timestamp(seconds: int) -> str:
    return (BASE_TIME + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rank_log(path: Path, rank: int, fault: str, faulty_rank: int, faulty_node: int,
              faulty_gpu: int, fault_second: int, rng: random.Random) -> None:
    lines = []
    node, gpu = divmod(rank, 8)
    for i in range(300):
        seconds = i * 11
        if seconds >= fault_second:
            break
        if i % 11 == 0:
            lines.append(f"{timestamp(seconds)} INFO rank={rank} NCCL communicator initialized on node-{node:02d} gpu={gpu}")
        if fault == "nccl_version_mismatch" and node == faulty_node and i == 2:
            lines.append(f"{timestamp(seconds)} INFO rank={rank} NCCL version 2.18.3 loaded")
        if fault == "ib_flap" and node == faulty_node and i == 25:
            lines.append(f"{timestamp(seconds)} NCCL WARN NET/IB : ibv_poll_cq error 12, vendor err 129")
        if fault == "fs_stall" and node == faulty_node and i == 22:
            lines.append(f"{timestamp(seconds)} INFO saving checkpoint to /mnt/checkpoint/step-0022")
        if fault == "oom_one_rank" and rank == faulty_rank and i == 290:
            lines.append(f"{timestamp(seconds)} ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        if fault == "nan_divergence" and rank == faulty_rank and i == 288:
            lines.append(f"{timestamp(seconds)} ERROR step {i} loss nan lr 0.0002 tok/s 17320 gnorm inf mem 71.2 GB")
        if fault == "illegal_access" and rank == faulty_rank and i == 290:
            lines.append(f"{timestamp(seconds)} ERROR CUDA error: an illegal memory access was encountered")
        if fault == "dataloader_oom_kill" and rank == faulty_rank and i == 290:
            lines.append(f"{timestamp(seconds)} ERROR DataLoader worker (pid 9931) is killed by signal: Killed")
        loss = 2.6 - i * .004 + rng.uniform(-.008, .008)
        gnorm = 1.1 + rng.uniform(-.15, .15)
        elapsed = seconds
        if fault in {"thermal_throttle", "power_cap"} and elapsed > fault_second:
            elapsed = int(elapsed * 27 / 11)
        lines.append(f"{timestamp(seconds)} INFO rank={rank} step {i} loss {loss:.4f} lr 0.0002 tok/s 17320 gnorm {gnorm:.3f} mem {58 + rng.random()*5:.1f} GB")
        if i in (40, 120, 220):
            lines.append(f"{timestamp(seconds)} INFO rank={rank} checkpoint saved path=/mnt/checkpoint/step-{i:04d}")
        if i in (60, 180):
            lines.append(f"{timestamp(seconds)} INFO rank={rank} evaluation completed samples=4096 accuracy=0.78")
        if rank == faulty_rank and i == 290 and fault in {"oom_one_rank", "nan_divergence", "dataloader_oom_kill", "illegal_access", "preemption"}:
            break
    if fault in {"xid79_demo", "ecc_dbe", "ib_flap", "oom_one_rank", "nan_divergence", "thermal_throttle", "power_cap", "fs_stall", "nccl_version_mismatch", "dataloader_oom_kill", "preemption", "illegal_access"}:
        if rank != faulty_rank:
            for j in range(25):
                lines.append(f"{timestamp(fault_second + 600 + j)} ERROR ProcessGroupNCCL.cpp:{800+j} Watchdog caught collective operation timeout: 600{j:03d} milliseconds before timing out; Traceback frame {j}")
            lines.append(f"{timestamp(fault_second + 625)} ERROR ProcessGroupNCCL communicator aborted after timeout")
    path.write_text("\n".join(lines) + "\n")


def make_incident(fault_id: str, out: Path, seed: int) -> Path:
    category, evidence, needs_dashboard = FAULTS[fault_id]
    rng = random.Random(seed + sum(map(ord, fault_id)))
    if fault_id == "xid79_demo":
        node, gpu = 1, 5
    else:
        node, gpu = rng.randrange(8), rng.randrange(8)
    rank = node * 8 + gpu
    fault_second = 3300
    root = out / fault_id
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    evidence_file = "logs/scheduler.log" if needs_dashboard else f"logs/dmesg_node-{node:02d}.log"
    evidence_line = 4 if needs_dashboard else 3
    for r in range(64):
        _rank_log(logs / f"rank_{r:03d}.log", r, fault_id, rank, node, gpu, fault_second, rng)
    for n in range(8):
        entries = [f"{timestamp(0)} kernel: Linux version 6.8.0-synthetic", f"{timestamp(120)} kernel: GPU driver initialized"]
        if n == node and not needs_dashboard:
            if fault_id == "ecc_dbe":
                evidence = "NVRM: Xid 48: double-bit ECC error on framebuffer memory"
            elif fault_id == "ib_flap":
                evidence = "mlx5_core 0000:af:00.0: Link down"
            elif fault_id == "fs_stall":
                evidence = "nfs: server fs01 not responding, still trying"
            elif fault_id == "dataloader_oom_kill":
                evidence = "Out of memory: Killed process 9931 (pt_data_worker)"
            elif fault_id == "illegal_access":
                evidence = "NVRM: Xid 31: MMU fault"
            entries.append(f"{timestamp(fault_second)} kernel: {evidence}")
        else:
            entries.append(f"{timestamp(400)} kernel: node healthy")
        (logs / f"dmesg_node-{n:02d}.log").write_text("\n".join(entries) + "\n")
    scheduler = [f"{timestamp(0)} job_id: synthetic-{fault_id}", f"{timestamp(1)} allocated nodes: node-00..node-07", f"{timestamp(2)} state: RUNNING"]
    if fault_id == "preemption":
        scheduler.append(f"{timestamp(fault_second)} event: Job preempted by higher priority job")
        evidence_file, evidence_line = "logs/scheduler.log", 4
    elif needs_dashboard:
        scheduler.append(f"{timestamp(fault_second + 600)} state: FAILED DUE TO TIME LIMIT")
    else:
        scheduler.append(f"{timestamp(fault_second + 625)} state: FAILED")
    (logs / "scheduler.log").write_text("\n".join(scheduler) + "\n")
    config = "# Synthetic job configuration\njob_name: training-demo\nnodes: 8\ngpus_per_node: 8\nprecision: bf16\ncheckpoint_interval: 40\n"
    (root / "job_config.yaml").write_text(config)
    render_dashboard(root / "dashboard.png", fault_id, node, gpu)
    truth = {"fault_id": fault_id, "category": category, "node": f"node-{node:02d}", "gpu": gpu, "rank": rank,
             "evidence_file": evidence_file, "evidence_line": evidence_line,
             "needs_dashboard": needs_dashboard, "notes": "Synthetic incident; planted ground truth for evaluation only."}
    (root / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", choices=sorted(FAULTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("incidents"))
    args = parser.parse_args()
    if bool(args.id) == args.all:
        parser.error("choose exactly one of --id or --all")
    ids = [args.id] if args.id else list(FAULTS)
    for fault_id in ids:
        path = make_incident(fault_id, args.out, args.seed)
        print(f"Generated synthetic incident {fault_id}: {path}")


if __name__ == "__main__":
    main()
