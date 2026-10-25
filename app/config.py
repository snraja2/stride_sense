"""Environment-backed CrashCart configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("LLM_API_KEY") or os.getenv("NEBIUS_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    mode: str = os.getenv("LLM_MODE", "live").lower()
    concurrency: int = _int_env("CONCURRENCY", 24)
    daily_run_cap: int = _int_env("DAILY_RUN_CAP", 60)
    ip_runs_per_hour: int = _int_env("IP_RUNS_PER_HOUR", 3)
    runs_dir: str = os.getenv("RUNS_DIR", "./runs")
    incidents_dir: str = os.getenv("INCIDENTS_DIR", "./incidents")
    port: int = _int_env("PORT", 8080)
    model_reduce: str = os.getenv("LLM_MODEL_REDUCE", "nvidia/Nemotron-3-Ultra-550b-a55b")
    model_baseline: str = os.getenv("LLM_MODEL_BASELINE", "nvidia/Nemotron-3-Ultra-550b-a55b")
    model_swarm: str = os.getenv("LLM_MODEL_SWARM", "nvidia/Nemotron-3_5-Lightning")
    model_eyes: str = os.getenv("LLM_MODEL_EYES", "")
    model_skeptic: str = os.getenv("LLM_MODEL_SKEPTIC", "nvidia/nemotron-3-super-120b-a12b")
    price_reduce_input: float = _float_env("PRICE_REDUCE_INPUT", 1.00)
    price_reduce_output: float = _float_env("PRICE_REDUCE_OUTPUT", 3.00)
    price_swarm_input: float = _float_env("PRICE_SWARM_INPUT", 0.06)
    price_swarm_output: float = _float_env("PRICE_SWARM_OUTPUT", 0.24)
    price_eyes_input: float = _float_env("PRICE_EYES_INPUT", 0.06)
    price_eyes_output: float = _float_env("PRICE_EYES_OUTPUT", 0.24)
    price_skeptic_input: float = _float_env("PRICE_SKEPTIC_INPUT", 0.30)
    price_skeptic_output: float = _float_env("PRICE_SKEPTIC_OUTPUT", 0.90)

    @property
    def models(self) -> dict[str, str]:
        return {
            "reduce": self.model_reduce,
            "baseline": self.model_baseline,
            "swarm": self.model_swarm,
            "eyes": self.model_eyes,
            "skeptic": self.model_skeptic,
        }

    def extra_body(self, role: str) -> dict:
        raw = os.getenv(f"LLM_EXTRA_BODY_{role.upper()}", "")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


settings = Settings()

CATEGORIES = (
    "gpu_fell_off_bus", "ecc_error", "ib_link_failure", "cuda_oom",
    "nan_divergence", "thermal_throttle", "power_cap", "filesystem_stall",
    "nccl_version_mismatch", "dataloader_oom_kill", "preemption",
    "illegal_memory_access", "other",
)
