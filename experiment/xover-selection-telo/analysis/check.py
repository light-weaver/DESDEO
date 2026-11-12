"""Check that a copied `results/` tree is complete and consistent with the config before analysing it.

Every check compares the tables with what `config/config.yaml` asks for:

- every instance has a reference front built from every configured algorithm, crossover and front seed;
- every (instance, algorithm) has an indicator and a mechanism table, holding every crossover, seed and
  budget fraction exactly once;
- no table is older than its instance's front, which would mean it was scored against an earlier one
  (this relies on the copy keeping modification times, as `rsync -a` does);
- the rankable indicators have no missing or non-finite values, and `hv_relative` is at most 1.

    python -m analysis.check

Exits with status 1 and lists the failures if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from analysis.common import load_config, results_dir
from xover import design
from xover.indicators import RANKABLE

SHOWN = 5
"""Failures listed per check; the rest are counted."""


def check(root: Path, config: dict) -> dict[str, list[str]]:
    """Return every failure, grouped by check."""
    run = config["run"]
    instances = list(design.selected_instances(config))
    crossovers = list(run["crossovers"])
    seeds = list(run["seeds"])
    fractions = list(config["fractions"])
    front_algorithms = design.front_selection(config, "algorithms")
    front_crossovers = design.front_selection(config, "crossovers")
    front_runs = len(front_algorithms) * len(design.front_pairs(config)) * len(config["reference_front"]["seeds"])
    runs_per_table = len(design.arm_mutations(config)) * len(seeds)
    expected_rows = runs_per_table * len(fractions)

    failures: dict[str, list[str]] = {
        "front missing": [],
        "front built from other runs": [],
        "table missing": [],
        "table incomplete": [],
        "table older than its front": [],
        "indicator missing or not finite": [],
        "hv_relative above 1": [],
    }
    for instance in instances:
        front = root / "fronts" / f"{instance}.npz"
        meta_path = root / "fronts" / f"{instance}.json"
        if not front.exists() or not meta_path.exists():
            failures["front missing"].append(instance)
            continue
        meta = json.loads(meta_path.read_text())
        if meta["n_archived_runs"] != front_runs or meta["front_runs"]["crossovers"] != front_crossovers:
            failures["front built from other runs"].append(f"{instance}: {meta['n_archived_runs']} runs")
        for algorithm in run["algorithms"]:
            name = f"{instance}/{algorithm}"
            paths = [root / kind / instance / f"{algorithm}.parquet" for kind in ("indicators", "mechanism")]
            if not all(path.exists() for path in paths):
                failures["table missing"].append(name)
                continue
            if any(path.stat().st_mtime < front.stat().st_mtime for path in paths):
                failures["table older than its front"].append(name)
            table = pl.read_parquet(paths[0])
            cells = table.select("crossover", "mutation", "seed", "fraction").unique()
            if (
                table.height != expected_rows
                or cells.height != expected_rows
                or set(table["crossover"]) != set(crossovers)
                or set(table["seed"]) != set(seeds)
                or not np.allclose(sorted(table["fraction"].unique()), fractions)
                or set(table.select("crossover", "mutation").unique().iter_rows()) != set(design.arm_mutations(config))
                or pl.read_parquet(paths[1]).height != runs_per_table
            ):
                failures["table incomplete"].append(f"{name}: {table.height} of {expected_rows} rows")
            values = table.select(RANKABLE).to_numpy()
            if not np.isfinite(values).all():
                failures["indicator missing or not finite"].append(name)
            if (table["hv_relative"] > 1).any():
                failures["hv_relative above 1"].append(name)
    return failures


def main() -> None:
    """Run the checks and report."""
    config = load_config()
    failures = check(results_dir(config), config)
    for name, items in failures.items():
        status = "ok" if not items else f"FAILED ({len(items)})"
        print(f"{name:34s} {status}")
        for item in items[:SHOWN]:
            print(f"    {item}")
        if len(items) > SHOWN:
            print(f"    ... and {len(items) - SHOWN} more")
    if any(failures.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
