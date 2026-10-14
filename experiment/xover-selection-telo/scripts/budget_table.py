"""Phase 1: collect the per-instance budgets into one table, for reading rather than for the workflow.

Every downstream job reads its own instance's budget file, so a changed budget reruns only that
instance's jobs.
"""

import json
from pathlib import Path

import polars as pl

rows = []
for path in snakemake.input:
    record = json.loads(Path(path).read_text())
    rows.append(
        {
            "instance": record["instance"],
            "suite": record["suite"],
            "m": record["m"],
            "n_var": record["n_var"],
            "budget": record["budget"],
            "final_median_igd_plus": record["final_median_igd_plus"],
            "wall_clock_s": record["wall_clock_s"],
        }
    )
pl.DataFrame(rows).sort("instance").write_csv(snakemake.output[0])
