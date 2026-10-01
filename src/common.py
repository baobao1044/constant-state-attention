"""Shared helpers: seeding, environment manifest, versioned run directories.

Every experiment driver in ``experiments/`` writes its config, environment and metrics
into ``results/<name>/<utc-timestamp>/`` so a number in the README can always be traced
back to the run that produced it.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"


def seed_all(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_manifest() -> dict:
    man = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_rev": _git_rev(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        man["cuda"] = {
            "name": p.name,
            "total_memory_gb": round(p.total_memory / 1024 ** 3, 1),
            "sm": f"{p.major}{p.minor}",
            "count": torch.cuda.device_count(),
        }
    return man


def run_dir(name: str) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = RESULTS / name / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run(name: str, config: dict, metrics, env: dict | None = None) -> Path:
    d = run_dir(name)
    (d / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (d / "env.json").write_text(json.dumps(env or env_manifest(), indent=2, sort_keys=True))
    metrics.to_csv(d / "metrics.csv", index=False)
    return d


def results_path(name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / name


def device_or_default(arg: str | None) -> str:
    if arg:
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"
