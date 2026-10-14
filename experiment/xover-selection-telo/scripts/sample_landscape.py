"""Phase 5: draw and evaluate one instance's landscape sample, and its derived scalar landscapes.

Plain float32 arrays only, so the ELA environment's older numpy can read the file.
"""

import numpy as np

from xover import design
from xover.landscape import derived_landscapes, sample
from xover.problems import build_problem
from xover.run import write_npz

config = snakemake.config
settings = config["landscape"]
instance = design.instance(config, snakemake.wildcards.inst)
problem = build_problem(instance.spec)

arrays = sample(problem, settings["sample_size_per_var"] * instance.n_var, settings["sample_seed"])
names, landscapes = derived_landscapes(arrays["Y"], config["indicators"]["exact_hv_max_m"])
arrays["S"] = landscapes.astype(np.float32)
# The landscape names travel in the file as fixed-width unicode, which needs no pickling.
arrays["S_names"] = np.asarray(names)
write_npz(snakemake.output[0], arrays)
print(f"{instance.slug}: {len(arrays['X']):,} points")
