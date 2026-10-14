"""Phase 4: score every run of one instance x algorithm against the instance's reference front.

Writes two tables. The indicator table has one row per run and budget fraction; the mechanism table
has one row per run, with the mean and final value of each crossover mechanism series.
"""

from pathlib import Path

import polars as pl

from xover import design, fronts
from xover.indicators import score_run
from xover.logger import load_run, mechanism_summary, unpack_snapshots

config = snakemake.config
slug, algorithm = snakemake.wildcards.inst, snakemake.wildcards.alg
instance = design.instance(config, slug)
front = fronts.load(snakemake.input.front)
population = design.population_size(config, instance.m)

keys = {
    "instance": slug,
    "suite": instance.suite,
    "base": instance.base,
    "family": instance.family,
    "m": instance.m,
    "n_var": instance.n_var,
    "algorithm": algorithm,
}
rows, mechanism = [], []
for path in sorted(map(Path, snakemake.input.runs)):
    # .../{inst}/{alg}/{xo}/{mut}/s{seed}.npz
    run = keys | {"crossover": path.parent.parent.name, "mutation": path.parent.name, "seed": int(path.stem[1:])}
    arrays = load_run(path)
    mechanism.append(run | mechanism_summary(arrays))
    for row in score_run(
        unpack_snapshots(arrays),
        arrays["snap_evals"],
        front,
        fractions=config["fractions"],
        m=instance.m,
        population=population,
        exact_hv_max_m=config["indicators"]["exact_hv_max_m"],
        r2_weights=config["indicators"]["r2_weights"],
    ):
        rows.append(run | row)

Path(snakemake.output.indicators).parent.mkdir(parents=True, exist_ok=True)
Path(snakemake.output.mechanism).parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows, infer_schema_length=None).write_parquet(snakemake.output.indicators)
pl.DataFrame(mechanism, infer_schema_length=None).write_parquet(snakemake.output.mechanism)
print(f"{slug} {algorithm}: {len(mechanism)} runs, {len(rows)} indicator rows")
