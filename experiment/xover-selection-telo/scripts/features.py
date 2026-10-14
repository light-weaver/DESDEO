"""Phase 5: join the multi-objective and ELA features, one row per instance, and screen them.

The join is a left join on the instance. Nothing is imputed: a feature that could not be computed on
an instance stays missing, and gets a `<feature>__available` flag.

The screen, in order:

1. drop features missing on more than `max_missing` of the instances;
2. drop features whose variance across instances is below `min_variance`;
3. cluster what remains at |Pearson r| > `correlation_cap` and keep one feature per cluster,
   preferring the feature present on more instances, then the earlier column.

With one sample per instance there is no seed-to-seed spread to measure, so the old study's ICC
stability screen has no counterpart here.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ("instance", "suite", "base", "family")
settings = snakemake.config["landscape"]

mo = pl.concat([pl.read_parquet(path) for path in snakemake.input.mo], how="diagonal_relaxed")
ela = pl.concat([pl.read_parquet(path) for path in snakemake.input.ela], how="diagonal_relaxed")
ela_columns = [c for c in ela.columns if c not in KEYS]
table = mo.join(ela.select(["instance", *ela_columns]), on="instance", how="left")

candidates = [c for c in table.columns if c not in KEYS and table[c].dtype.is_numeric()]
# NaN and null both mean "not computed".
table = table.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in candidates])
report: dict = {"dropped_missing": [], "dropped_variance": [], "clustered": []}

kept = []
for name in candidates:
    missing = table[name].null_count() / table.height
    if missing > settings["max_missing"]:
        report["dropped_missing"].append({"feature": name, "missing": round(missing, 4)})
        continue
    values = table[name].drop_nulls().to_numpy()
    if len(values) == 0 or np.var(values) < settings["min_variance"]:
        report["dropped_variance"].append(name)
        continue
    kept.append(name)

matrix = np.ma.masked_invalid(table.select(kept).to_numpy().astype(float))
with np.errstate(invalid="ignore", divide="ignore"):
    correlation = np.ma.corrcoef(matrix, rowvar=False).filled(np.nan) if len(kept) > 1 else np.ones((1, 1))
order = sorted(range(len(kept)), key=lambda i: (table[kept[i]].null_count(), i))
chosen: list[int] = []
for i in order:
    partner = next((j for j in chosen if abs(correlation[i, j]) > settings["correlation_cap"]), None)
    if partner is None:
        chosen.append(i)
    else:
        report["clustered"].append({"feature": kept[i], "represented_by": kept[partner]})
retained = [kept[i] for i in sorted(chosen)]

flags = [pl.col(n).is_not_null().cast(pl.Int8).alias(f"{n}__available") for n in retained if table[n].null_count()]
table = table.with_columns(flags).select([*KEYS, *retained, *(f.meta.output_name() for f in flags)])

report |= {
    "n_instances": table.height,
    "n_candidates": len(candidates),
    "n_ela_features": len(ela_columns),
    "n_retained": len(retained),
    "retained": retained,
}
Path(snakemake.output.features).parent.mkdir(parents=True, exist_ok=True)
table.write_parquet(snakemake.output.features)
Path(snakemake.output.report).write_text(json.dumps(report, indent=2) + "\n")
print(f"{table.height} instances, {len(candidates)} candidate features, {len(retained)} retained")
