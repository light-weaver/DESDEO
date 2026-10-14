"""Phase 5: the multi-objective landscape features of one instance's sample."""

from pathlib import Path

import numpy as np
import polars as pl

from xover import design
from xover.landscape import mo_features

config = snakemake.config
instance = design.instance(config, snakemake.wildcards.inst)

with np.load(snakemake.input[0], allow_pickle=False) as handle:
    landscapes = {str(name): handle["S"][:, i].astype(float) for i, name in enumerate(handle["S_names"])}
    values = mo_features(
        handle["X"].astype(float),
        handle["Y"].astype(float),
        handle["lower"].astype(float),
        handle["upper"].astype(float),
        landscapes,
    )

row = {"instance": instance.slug, "suite": instance.suite, "base": instance.base, "family": instance.family}
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame([row | values]).write_parquet(snakemake.output[0])
