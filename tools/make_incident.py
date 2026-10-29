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
    slow_fault = fault in {"thermal_throttle", "power_cap"}
    for i in range(323):
        seconds = i * 11 if i <= 300 else fault_second + (i - 300) * 27
        if seconds > fault_second + (600 if slow_fault else 0):
            break
        if seconds > fault_second and not slow_fault and not (fault == "nan_divergence" and i == 301):
            break
        if i % 11 == 0:
            lines.append(f"{timestamp(seconds)} INFO rank={rank} NCCL communicator initialized on node-{node:02d} gpu={gpu}")
        if fault == "nccl_version_mismatch" and i == 2:
            version = "2.18.3" if node == faulty_node else "2.20.5"
            lines.append(f"{timestamp(seconds)} INFO rank={rank} NCCL version {version} loaded")
        if fault == "ib_flap" and node == faulty_node and i == 25:
            lines.append(f"{timestamp(seconds)} NCCL WARN NET/IB : ibv_poll_cq error 12, vendor err 129")
        if fault == "fs_stall" and node == faulty_node and i == 22:
            lines.append(f"{timestamp(seconds)} INFO saving checkpoint to /mnt/checkpoint/step-0022")
        if fault == "oom_one_rank" and rank == faulty_rank and i == 300:
            lines.append(f"{timestamp(seconds)} ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
            break
        if fault == "nan_divergence" and rank == faulty_rank and i == 300:
            lines.append(f"{timestamp(seconds)} ERROR step {i} loss nan lr 0.0002 tok/s 17320 gnorm inf mem 71.2 GB")
            break
        if fault == "nan_divergence" and rank != faulty_rank and i == 301:
            lines.append(f"{timestamp(seconds)} ERROR FloatingPointError: loss is nan after gnorm inf")
            break
        if fault == "illegal_access" and rank == faulty_rank and i == 300:
            lines.append(f"{timestamp(seconds)} ERROR CUDA error: an illegal memory access was encountered")
            break
        if fault == "dataloader_oom_kill" and rank == faulty_rank and i == 300:
            lines.append(f"{timestamp(seconds)} ERROR DataLoader worker (pid 9931) is killed by signal: Killed")
            break
        if fault == "preemption" and node == faulty_node and i == 300:
            lines.append(f"{timestamp(seconds)} WARNING Received signal 15 from scheduler")
            break
        loss = 2.6 - i * .004 + rng.uniform(-.008, .008)
        gnorm = 1.1 + rng.uniform(-.15, .15)
        step_time = 27 if slow_fault and seconds > fault_second else 11
        lines.append(f"{timestamp(seconds)} INFO rank={rank} step {i} loss {loss:.4f} lr 0.0002 tok/s {int(19000 / step_time)} gnorm {gnorm:.3f} mem {58 + rng.random()*5:.1f} GB step_time={step_time}s")
        if i % 3 == 0:
            queued = rng.randrange(3, 9)
            batch_wait = rng.randrange(8, 22)
            io_wait = rng.randrange(2, 14)
            lines.append(f"{timestamp(seconds)} INFO rank={rank} dataloader prefetch queue={queued}/8 batch_wait={batch_wait}ms io_wait={io_wait}ms")
        if i in (40, 120, 220):
            lines.append(f"{timestamp(seconds)} INFO rank={rank} checkpoint saved path=/mnt/checkpoint/step-{i:04d}")
        if i in (60, 180):
            lines.append(f"{timestamp(seconds)} INFO rank={rank} evaluation completed samples=4096 accuracy=0.78")
    has_watchdog = fault not in {"nan_divergence"} and rank != faulty_rank
    if fault == "preemption" and node == faulty_node:
        has_watchdog = False
    if has_watchdog:
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
    evidence_line = 3
    for r in range(64):
        _rank_log(logs / f"rank_{r:03d}.log", r, fault_id, rank, node, gpu, fault_second, rng)
    dmesg_evidence = {
        "xid79_demo": "NVRM: Xid (PCI:0000:8a:00): 79, pid=4128, name=python, GPU has fallen off the bus.",
        "ecc_dbe": "NVRM: Xid 48: double-bit ECC error on framebuffer memory",
        "ib_flap": "mlx5_core 0000:af:00.0: Link down",
        "fs_stall": "nfs: server fs01 not responding, still trying",
        "dataloader_oom_kill": "Out of memory: Killed process 9931 (pt_data_worker)",
        "illegal_access": "NVRM: Xid 31: MMU fault",
    }
    for n in range(8):
        entries = [f"{timestamp(0)} kernel: Linux version 6.8.0-synthetic", f"{timestamp(120)} kernel: GPU driver initialized"]
        if n == node and not needs_dashboard and fault_id in dmesg_evidence:
            entries.append(f"{timestamp(fault_second)} kernel: {dmesg_evidence[fault_id]}")
        else:
            entries.append(f"{timestamp(400)} kernel: node healthy")
        (logs / f"dmesg_node-{n:02d}.log").write_text("\n".join(entries) + "\n")
    scheduler = [f"{timestamp(0)} job_id: synthetic-{fault_id}", f"{timestamp(1)} allocated nodes: node-00..node-07", f"{timestamp(2)} state: RUNNING"]
    if fault_id == "preemption":
        scheduler.append(f"{timestamp(fault_second)} event: Job preempted by higher priority job")
        evidence_file, evidence_line = "logs/scheduler.log", 4
    elif fault_id == "nccl_version_mismatch":
        scheduler.append(f"{timestamp(180)} node-{node:02d} reimaged with older image NCCL 2.18.3")
        evidence_file, evidence_line = "logs/scheduler.log", 4
        scheduler.append(f"{timestamp(fault_second + 625)} state: FAILED")
    elif needs_dashboard:
        scheduler.append(f"{timestamp(fault_second + 600)} state: FAILED DUE TO TIME LIMIT")
        evidence_line = 4
    elif fault_id in {"oom_one_rank", "nan_divergence"}:
        scheduler.append(f"{timestamp(fault_second + 625)} state: FAILED")
        evidence_file = f"logs/rank_{rank:03d}.log"
        marker = "CUDA out of memory" if fault_id == "oom_one_rank" else "loss nan"
        source_lines = (root / evidence_file).read_text().splitlines()
        evidence_line = next(i for i, line in enumerate(source_lines, 1) if marker in line)
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
