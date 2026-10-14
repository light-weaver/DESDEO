"""Phase 3: run one block of the main grid at every seed, and dump each run.

Each run writes `s{seed}.npz`, the populations at the configured budget fractions plus the crossover
mechanism series, and `s{seed}.json`, its timings and provenance.
"""

import json
from pathlib import Path

from xover.design import Block
from xover.run import run_cell, write_json, write_npz

config = snakemake.config
w = snakemake.wildcards
block = Block(w.inst, w.alg, w.xo, w.mut)
budget = json.loads(Path(snakemake.input.budget).read_text())["budget"]

for seed, npz, record_path in zip(config["run"]["seeds"], snakemake.output.npz, snakemake.output.json, strict=True):
    arrays, record = run_cell(config, block, seed, budget)
    write_npz(npz, arrays)
    write_json(record_path, record)
    print(f"{w.inst} {w.alg} {w.xo} {w.mut} s{seed}: {record['solve_s']:.1f} s, {record['generations']} generations")
