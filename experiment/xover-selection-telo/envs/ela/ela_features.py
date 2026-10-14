"""Phase 5: pflacco's ELA features for one instance's landscape sample.

Runs in this directory's own environment (Python 3.11, numpy 1.24, pflacco 1.2.2), never imports
desdeo or moocore, and reads nothing but the `.npz` that `scripts/sample_landscape.py` writes:

    uv run --project envs/ela python envs/ela/ela_features.py SAMPLE.npz OUT.parquet --sets ...

Features are computed per objective and aggregated across objectives by mean, min, max and std, so
relabelling the objectives cannot change the feature vector, and once on each derived scalar
landscape (mean normalised objective, achievement scalarising value, Pareto rank, negated
hypervolume contribution), which is where the multi-objective structure enters.

Only feature sets that need no extra objective evaluations are used. A set that raises is left
missing rather than imputed. `ela_meta`'s models are fitted only when they have fewer parameters than
the sample has points; pflacco otherwise returns a perfect adjusted R-squared, silently.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _parameter_count(kind: str, d: int) -> int:
    """Parameters in each `ela_meta` model at dimension `d`."""
    return {
        "lin_simple": 1 + d,
        "lin_w_interact": 1 + d + d * (d - 1) // 2,
        "quad_simple": 1 + 2 * d,
        "quad_w_interact": 1 + d + d * (d + 1) // 2,
    }.get(kind, 0)


def _is_identifiable(name: str, n: int, d: int) -> bool:
    """Whether an `ela_meta` feature's model is determined by the sample it was fitted on."""
    if not name.startswith("ela_meta."):
        return True
    needed = _parameter_count(name.split(".")[1], d)
    return needed == 0 or needed < n


def compute_for_landscape(x: np.ndarray, values: np.ndarray, lower, upper, sets: list[str]) -> dict[str, float]:
    """Every requested feature set on one scalar landscape."""
    import pflacco.classical_ela_features as classical
    import pflacco.misc_features as misc

    n, d = x.shape
    out: dict[str, float] = {}
    for set_name in sets:
        function = getattr(classical, set_name, None) or getattr(misc, set_name, None)
        if function is None:
            msg = f"pflacco has no feature set {set_name!r}"
            raise ValueError(msg)
        parameters = function.__code__.co_varnames[: function.__code__.co_argcount]
        kwargs = {}
        if "lower_bound" in parameters:
            kwargs["lower_bound"] = lower
        if "upper_bound" in parameters:
            kwargs["upper_bound"] = upper
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = function(x, values, **kwargs)
        except Exception:  # noqa: S112 -- missing, never imputed
            continue
        for key, value in dict(result).items():
            if key.endswith("costs_runtime") or not isinstance(value, (int, float, np.floating)):
                continue
            if _is_identifiable(key, n, d):
                out[key] = float(value)
    return out


def aggregate(per_objective: list[dict[str, float]], prefix: str) -> dict[str, float]:
    """Mean, min, max and std of each feature across objectives."""
    out: dict[str, float] = {}
    for key in sorted(set().union(*(set(entry) for entry in per_objective))):
        values = np.array([entry[key] for entry in per_objective if key in entry and np.isfinite(entry[key])])
        if len(values) == 0:
            continue
        out[f"{prefix}{key}.mean"] = float(values.mean())
        out[f"{prefix}{key}.min"] = float(values.min())
        out[f"{prefix}{key}.max"] = float(values.max())
        out[f"{prefix}{key}.std"] = float(values.std())
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sets", nargs="+", required=True, help="pflacco feature functions to run")
    args = parser.parse_args()

    with np.load(args.sample, allow_pickle=False) as handle:
        x = handle["X"].astype(float)
        y = handle["Y"].astype(float)
        lower = handle["lower"].astype(float).tolist()
        upper = handle["upper"].astype(float).tolist()
        landscapes = handle["S"].astype(float)
        names = [str(name) for name in handle["S_names"]]

    finite = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y, landscapes = x[finite], y[finite], landscapes[finite]

    row: dict[str, object] = {"instance": args.sample.stem}
    per_objective = [compute_for_landscape(x, y[:, j], lower, upper, args.sets) for j in range(y.shape[1])]
    row |= aggregate(per_objective, prefix="obj.")
    for index, name in enumerate(names):
        values = landscapes[:, index]
        if np.ptp(values) == 0:
            continue
        for key, value in compute_for_landscape(x, values, lower, upper, args.sets).items():
            row[f"{name}.{key}"] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(args.out, index=False)
    print(f"{args.sample.stem}: {len(row) - 1} ELA features")


if __name__ == "__main__":
    main()
