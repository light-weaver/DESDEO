"""Phase 3: run one block of the main grid at every seed, and dump the runs into one tar.

The tar holds `s{seed}.npz`, the populations at the configured budget fractions plus the crossover
mechanism series, and `s{seed}.json`, its timings and provenance, for each seed.
"""

import json
from pathlib import Path

from xover.design import Block
from xover.run import run_cell, write_block

config = snakemake.config
w = snakemake.wildcards
block = Block(w.inst, w.alg, w.xo, w.mut)
budget = json.loads(Path(snakemake.input.budget).read_text())["budget"]

runs = {}
for seed in snakemake.params.seeds:
    arrays, record = run_cell(config, block, seed, budget)
    runs[seed] = (arrays, record)
    print(f"{w.inst} {w.alg} {w.xo} {w.mut} s{seed}: {record['solve_s']:.1f} s, {record['generations']} generations")
write_block(snakemake.output[0], runs)
