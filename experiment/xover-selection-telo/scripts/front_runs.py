"""Phase 2: the archive runs of one block, at every front seed, for the reference fronts.

These repeat some cells of the main grid with a non-dominated archive attached. They are separate
runs so that the archive's CPU cost stays out of the grid's timings; the archive only listens, so the
trajectories are the same as their grid twins'.
"""

import json
from pathlib import Path

from xover.design import Block
from xover.run import run_cell, write_json, write_npz

config = snakemake.config
w = snakemake.wildcards
block = Block(w.inst, w.alg, w.xo, w.mut)
budget = json.loads(Path(snakemake.input.budget).read_text())["budget"]

arrays, records = {}, []
for seed in config["reference_front"]["seeds"]:
    run_arrays, record = run_cell(config, block, seed, budget, archive=True)
    arrays[f"archive_F_s{seed}"] = run_arrays["archive_F"]
    records.append(record)
    print(f"{w.inst} {w.alg} {w.xo} {w.mut} s{seed}: {record['archive_rows']} archived points")

write_npz(snakemake.output.npz, arrays)
write_json(snakemake.output.json, {"runs": records})
