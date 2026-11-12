"""What every analysis shares: the config, the result tables and where the output goes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import yaml

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


def probe_arms(config: dict[str, Any]) -> list[str]:
    """Return the arms that change the mutation as well as the crossover (`run.mutation_overrides`).

    They answer a separate question, so the operator comparison leaves them out: an arm in the table
    shifts every other arm's min-max regret and every mean rank.
    """
    return list(config["run"].get("mutation_overrides", {}))


def _scan(config: dict[str, Any], kind: str, *, probes: bool) -> pl.DataFrame:
    frame = pl.scan_parquet(results_dir(config) / kind / "*" / "*.parquet")
    if not probes:
        frame = frame.filter(~pl.col("crossover").is_in(probe_arms(config)))
    return frame.collect()


def load_indicators(config: dict[str, Any], *, probes: bool = False) -> pl.DataFrame:
    """Return the indicator table: one row per run and budget fraction, without the probes by default."""
    return _scan(config, "indicators", probes=probes)


def load_mechanism(config: dict[str, Any], *, probes: bool = False) -> pl.DataFrame:
    """Return seed-median crossover diagnostics per (instance, algorithm, crossover), without the probes by default."""
    raw = _scan(config, "mechanism", probes=probes)
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
