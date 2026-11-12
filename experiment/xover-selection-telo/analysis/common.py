"""What every analysis shares: the config, the result tables and where the output goes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from xover import design

STUDY = Path(__file__).resolve().parents[1]

MECHANISM_COLUMNS = (
    "offspring_step",
    "parent_separation_final",
    "parent_diameter_final",
    "diversity_retained",
    "diversity_half_life",
)
"""Offspring step and parent separation are normalised by the box diagonal. Diversity retained is the
final over the initial parent diameter. The half-life is the generation at which the diameter first
halves, NaN (dropped from medians) for runs where it never does."""


def load_config() -> dict[str, Any]:
    """Return `config/config.yaml`."""
    return yaml.safe_load((STUDY / "config" / "config.yaml").read_text())


def results_dir(config: dict[str, Any]) -> Path:
    """Return the configured results directory."""
    return STUDY / config["results"]


def _scan(config: dict[str, Any], kind: str) -> pl.DataFrame:
    # Tables of instances the config no longer selects (re37) stay on disk; leave them out.
    instances = list(design.selected_instances(config))
    frame = pl.scan_parquet(results_dir(config) / kind / "*" / "*.parquet").filter(pl.col("instance").is_in(instances))
    return frame.collect()


def load_indicators(config: dict[str, Any]) -> pl.DataFrame:
    """Return the indicator table: one row per run and budget fraction."""
    return _scan(config, "indicators")


def load_mechanism(config: dict[str, Any]) -> pl.DataFrame:
    """Return seed-median crossover diagnostics per (instance, algorithm, crossover)."""
    raw = _scan(config, "mechanism")
    return raw.group_by(["instance", "algorithm", "crossover"]).agg(
        [pl.col(c).fill_nan(None).median() for c in MECHANISM_COLUMNS]
    )


def provenance() -> dict[str, Any]:
    """Return the commit the analysis ran from, and whether tracked files in the study had changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=STUDY, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607

    return {"git_sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "-uno", "--", "."))}


def write(config: dict[str, Any], name: str, payload: dict[str, Any]) -> Path:
    """Write one analysis's JSON under `results/analysis/` and return the path."""
    path = results_dir(config) / "analysis" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
