"""Phase 1: the evaluation budget of one instance. See `xover.calibration` for the rule."""

from xover.calibration import calibrate
from xover.run import provenance, write_json

record = calibrate(snakemake.config, snakemake.wildcards.inst)
write_json(snakemake.output[0], record | {"provenance": provenance()})
print(f"{record['instance']}: budget {record['budget']:,} ({record['wall_clock_s']:.0f} s)")
