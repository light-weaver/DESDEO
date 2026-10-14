"""Phase 1: one evaluation budget per instance.

The budget is the smallest number of evaluations at which the calibration configuration (NSGA-III
with `sbx_default` and polynomial mutation, at a fixed population of 100) is within 2 % of its own
200 000-evaluation quality:

    budget(p) = min { e : median_seeds IGD+(e) <= 1.02 * median_seeds IGD+(200 000) }

rounded up to the next 5 000 and clipped to [10 000, 200 000]. All of these numbers live in
`config["calibration"]`.

Quality is IGD+ against a reference front built from the calibration runs' own merged archives, the
same construction as the study's reference fronts. That is self-referential, which is fine because
the criterion only detects *saturation*, the point past which more evaluations stop buying quality,
and this front is never used for scoring. IGD+ rather than hypervolume because it stays cheap and
finite at every objective count.

**Caveat for the paper:** the budget is a property of NSGA-III with SBX, one of the configurations
being compared, not of the problem. Every run is recorded at 21 budget fractions, so any conclusion
can be re-read at a shorter budget.
"""

from __future__ import annotations

import time
from typing import Any

import moocore
import numpy as np

from xover import design
from xover.fronts import build_reference_front
from xover.logger import unpack_snapshots
from xover.problems import build_problem
from xover.run import run

NO_MECHANISM = 10**9
"""A mechanism stride no run reaches: calibration does not need the crossover diagnostics."""


def calibrate(config: dict[str, Any], slug: str) -> dict[str, Any]:
    """Run the calibration for one instance and return its record, budget included."""
    settings = config["calibration"]
    instance = design.instance(config, slug)
    problem = build_problem(instance.spec)
    max_evaluations = settings["max_evaluations"]

    started = time.perf_counter()
    traces: list[tuple[np.ndarray, list[np.ndarray]]] = []
    archives: list[np.ndarray] = []
    for seed in settings["seeds"]:
        options = design.make_options(
            config,
            settings["algorithm"],
            settings["crossover"],
            settings["mutation"],
            m=instance.m,
            max_evaluations=max_evaluations,
            seed=seed,
            use_archive=True,
            population=settings["population"],
        )
        arrays, _ = run(
            problem,
            options,
            max_evaluations,
            n_checkpoints=settings["n_checkpoints"],
            mechanism_stride=NO_MECHANISM,
            archive_dtype=float,
        )
        traces.append((arrays["snap_evals"].copy(), unpack_snapshots(arrays)))
        archives.append(arrays["archive_F"])
    wall_clock = time.perf_counter() - started

    reference = build_reference_front(archives, cap=design.subset_cap(config, instance.m))
    span = np.where(reference.nadir - reference.ideal > 0, reference.nadir - reference.ideal, 1.0)
    reference_scaled = (reference.points - reference.ideal) / span

    # One quality curve per seed, interpolated onto a shared evaluation grid so the medians line up.
    grid = shared_grid(traces, settings["n_checkpoints"])
    curves = []
    for evaluations, snapshots in traces:
        quality = np.array(
            [moocore.igd_plus((snap - reference.ideal) / span, ref=reference_scaled) for snap in snapshots]
        )
        curves.append(np.interp(grid, evaluations, quality))
    median = np.median(np.vstack(curves), axis=0)

    return {
        "instance": slug,
        "budget": saturation_budget(grid, median, settings),
        "m": instance.m,
        "n_var": instance.n_var,
        "suite": instance.suite,
        "wall_clock_s": round(wall_clock, 2),
        "saturation_curve": {
            "evaluations": [int(e) for e in grid],
            "median_igd_plus": [float(q) for q in median],
        },
        "reference_front": reference.to_dict(),
        "final_median_igd_plus": float(median[-1]),
        "rule": settings,
    }


def shared_grid(traces: list[tuple[np.ndarray, list[np.ndarray]]], n_checkpoints: int) -> np.ndarray:
    """A common evaluation axis for the per-seed curves.

    Runs overshoot their budget by different amounts, so the seeds' checkpoints do not coincide; the
    grid spans only the range every seed covers.
    """
    lowest_start = max(int(evaluations[0]) for evaluations, _ in traces)
    highest_end = min(int(evaluations[-1]) for evaluations, _ in traces)
    return np.unique(np.geomspace(max(1, lowest_start), highest_end, num=n_checkpoints).astype(int))


def saturation_budget(grid: np.ndarray, median: np.ndarray, settings: dict[str, Any]) -> int:
    """The smallest grid point whose median quality is within `tolerance` of the final one."""
    target = median[-1] * settings["tolerance"]
    reached = np.flatnonzero(median <= target)
    raw = int(grid[reached[0]]) if len(reached) else int(grid[-1])
    rounded = int(np.ceil(raw / settings["rounding"]) * settings["rounding"])
    return int(np.clip(rounded, settings["floor"], settings["ceiling"]))
