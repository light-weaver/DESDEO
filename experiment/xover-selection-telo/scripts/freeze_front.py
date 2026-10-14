"""Phase 2: merge one instance's archives into its frozen reference front.

Archives are merged in the order of their run names (`{instance}__{algorithm}__{crossover}__
{mutation}__s{seed}`, sorted as strings), which keeps the front independent of how the files happen
to be listed.
"""

from pathlib import Path

import numpy as np

from xover import design, fronts
from xover.indicators import hv_reference_component

config = snakemake.config
slug = snakemake.wildcards.inst
instance = design.instance(config, slug)

archives = {}
for path in map(Path, snakemake.input.archives):
    mut, xo, alg = path.stem, path.parent.name, path.parent.parent.name
    with np.load(path, allow_pickle=False) as handle:
        for key in handle.files:
            seed = key.removeprefix("archive_F_")
            archives[f"{slug}__{alg}__{xo}__{mut}__{seed}"] = handle[key]

# Snakemake guarantees every block's file exists; this checks each holds every front seed.
expected = len(snakemake.input.archives) * len(config["reference_front"]["seeds"])
if len(archives) != expected:
    msg = f"{slug}: {len(archives)} archives, expected {expected}; refusing to freeze a partial pool"
    raise RuntimeError(msg)

population = design.population_size(config, instance.m)
front = fronts.freeze(
    [archives[key].astype(float) for key in sorted(archives) if len(archives[key])],
    cap=design.subset_cap(config, instance.m),
    hv_reference=hv_reference_component(instance.m, population),
    exact_hv_max_m=config["indicators"]["exact_hv_max_m"],
)
fronts.save(
    front,
    snakemake.output.npz,
    {
        "instance": slug,
        "m": instance.m,
        "n_archived_runs": len(archives),
        "cap": design.subset_cap(config, instance.m),
        "front_runs": {key: design.front_selection(config, key) for key in ("algorithms", "crossovers", "mutations")}
        | {"seeds": config["reference_front"]["seeds"]},
    },
)
print(f"{slug}: {len(front.points):,} points from {front.pool_size:,} non-dominated, {len(archives)} runs")
