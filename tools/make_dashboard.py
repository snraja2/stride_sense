"""Create a synthetic, unbranded GPU telemetry dashboard."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_dashboard(path, fault_id: str, node: int, gpu: int) -> None:
    rng = np.random.default_rng(731 + node * 8 + gpu)
    times = np.arange(0, 601, 30)
    util = np.clip(91 + rng.normal(0, 2, (len(times), 64)), 0, 100)
    clock = np.clip(1500 + rng.normal(0, 25, (len(times), 64)), 800, 1600)
    power = np.clip(285 + rng.normal(0, 8, (len(times), 64)), 100, 350)
    util[:, node * 8 + gpu] = np.clip(88 + rng.normal(0, 2, len(times)), 0, 100)
    clock[:, node * 8 + gpu] = np.clip(1490 + rng.normal(0, 15, len(times)), 800, 1600)
    power[:, node * 8 + gpu] = np.clip(280 + rng.normal(0, 5, len(times)), 100, 350)
    temp = np.clip(68 + rng.normal(0, 2, 64), 40, 99).reshape(8, 8)
    sampled_util = util[-1].reshape(8, 8)
    sampled_clock = clock[-1].reshape(8, 8)
    sampled_power = power[-1].reshape(8, 8)
    if fault_id in {"xid79_demo", "ecc_dbe", "ib_flap", "oom_one_rank", "dataloader_oom_kill", "illegal_access"}:
        sampled_util[node, gpu] = 0
        sampled_clock[node, gpu] = 0
    elif fault_id == "thermal_throttle":
        temp[node, gpu] = 94
        sampled_clock[node, gpu] = 510
        clock[:, node * 8 + gpu] = np.linspace(1490, 510, len(times))
    elif fault_id == "power_cap":
        sampled_clock[node, gpu] = 620
        sampled_power[node, gpu] = 160
        clock[:, node * 8 + gpu] = 620
        power[:, node * 8 + gpu] = 160
    elif fault_id in {"nccl_version_mismatch", "preemption", "fs_stall", "nan_divergence"}:
        sampled_util[node, gpu] = 0
        sampled_clock[node, gpu] = 0
    mean_util, mean_clock, mean_power = util.mean(axis=1), clock.mean(axis=1), power.mean(axis=1)
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.suptitle("GPU telemetry", fontsize=18)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=.38, wspace=.25)
    for idx, (values, title, ylabel, color) in enumerate((
        (util, "GPU utilization", "%", "#4c9f70"),
        (clock, "SM clock", "MHz", "#5287b5"),
        (power, "Power", "W", "#d18c45"),
    )):
        ax = fig.add_subplot(grid[0, idx])
        mean = values.mean(axis=1)
        ax.fill_between(times, values.min(axis=1), values.max(axis=1), color=color, alpha=.18)
        ax.plot(times, mean, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    heatmaps = ((sampled_util, "Utilization (%)", "viridis", 0, 100),
                (temp, "Temperature (°C)", "inferno", 40, 100),
                (sampled_clock, "SM clock (MHz)", "cividis", 0, 1600))
    for idx, (values, title, cmap, lo, hi) in enumerate(heatmaps):
        ax = fig.add_subplot(grid[1, idx])
        image = ax.imshow(values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title + " at T+120 s")
        ax.set_xticks(range(8), [str(x) for x in range(8)])
        ax.set_yticks(range(8), [f"node-{x:02d}" for x in range(8)])
        ax.set_xlabel("GPU")
        for row in range(8):
            for col in range(8):
                ax.text(col, row, f"{values[row, col]:.0f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.savefig(path, dpi=100)
    plt.close(fig)
