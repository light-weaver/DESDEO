"""The pipeline reproduces the EMO-paper study's code exactly, given the same inputs.

Runs the old study (`experiment/xover-selection` on the `experiment/xover-selection-2` branch) and this
pipeline on the same cells, budgets and archives, and requires identical output: every array of every
run dump, the calibration curve and budget, the reference front, and every indicator value. Both
sides use the same DESDEO, so any difference is the pipeline's.

Skipped unless `LEGACY_DIR` points at the old study's directory, for example a worktree:

    git worktree add /tmp/legacy experiment/xover-selection-2
    LEGACY_DIR=/tmp/legacy/experiment/xover-selection pytest tests/test_legacy_equivalence.py

The old code is driven in subprocesses, because it picks its problem registry from the `XOVER_RUN`
environment variable at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xover import calibration, design, fronts
from xover.design import Block
from xover.indicators import hv_reference_component, score_run
from xover.logger import load_run, unpack_snapshots
from xover.run import run_cell

LEGACY_DIR = os.environ.get("LEGACY_DIR")
pytestmark = pytest.mark.skipif(not LEGACY_DIR, reason="set LEGACY_DIR to the old study's directory")

BUDGET = 3000

# (XOVER_RUN, instance, algorithm, crossover, seed, budget)
CELLS = [
    *(
        ("", "dtlz2_m3_n12", algorithm, crossover, 0, BUDGET)
        for algorithm in ("nsga3", "nsga2", "ibea")
        for crossover in (
            "sbx_default",
            "sbx_platemo",
            "sbx_jmetal",
            "sbx_no_uniform",
            "blend",
            "bex",
            "bex_uniform",
            "uniform",
            "sax",
            "de",
            "pcx",
            "composite",
            "none",
        )
    ),
    ("", "re33_m3", "nsga2", "blend", 1, BUDGET),
    ("", "metallappl_mop2_m5", "ibea", "pcx", 0, BUDGET),
    ("", "gaa_m3", "nsga3", "sbx_jmetal", 0, 2000),
    ("", "river_pollution_m4_f_1-f_2-f_3-f_4", "nsga3", "uniform", 3, BUDGET),
    ("zcat", "zcat3_l2_ps1_m3_n30", "nsga2", "de", 2, BUDGET),
    ("rotated_dtlz", "r2_dtlz2_m5_n30", "ibea", "sbx_default", 0, BUDGET),
]

FRONT_INSTANCE = "dtlz2_m3_n12"
FRONT_CELLS = [
    (algorithm, crossover, seed)
    for algorithm in ("nsga3", "ibea")
    for crossover in ("sbx_default", "de")
    for seed in (0, 1)
]

CALIBRATION = {"seeds": [0, 1], "max_evaluations": 20_000, "n_checkpoints": 30}

DRIVER = r"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

job = json.loads(sys.argv[1])
out = Path(job.get("out", "."))

if job["mode"] == "runs":
    import runner
    for instance, algorithm, crossover, seed, budget, archive in job["cells"]:
        record = runner.run_one(runner.Cell(instance, algorithm, crossover, "polynomial", seed), budget, out,
                                archive=archive)
        assert record["status"] == "ok", record

elif job["mode"] == "front":
    import config, indicators, refset
    paths = sorted((out / job["instance"]).glob("*.npz"))
    archives = [np.load(p)["archive_F"].astype(float) for p in paths]
    front = refset.build_reference_front([a for a in archives if len(a)])
    m = front.points.shape[1]
    hv = indicators._hypervolume(front.normalise(front.points),
                                 np.full(m, config.hv_reference_component(m, config.POPULATION_SIZE[m])), m)
    np.savez(job["result"], points=front.points, ideal=front.ideal, nadir=front.nadir, hv=hv,
             pool_size=front.pool_size)

elif job["mode"] == "score":
    import config, hooks, indicators, refset
    with np.load(job["front"]) as f:
        front = refset.ReferenceFront(points=f["points"].astype(float), ideal=f["ideal"].astype(float),
                                      nadir=f["nadir"].astype(float), pool_size=int(f["pool_size"]),
                                      hypervolume=float(f["hv"]))
    rows = []
    for path in job["runs"]:
        with np.load(path) as handle:
            arrays = {k: handle[k] for k in handle.files}
        snapshots, evaluations = hooks.unpack_snapshots(arrays), arrays["snap_evals"]
        budget = int(evaluations[-1])
        for fraction in indicators.FRACTIONS:
            index = int(np.argmin(np.abs(evaluations - fraction * budget)))
            rows.append(indicators.score(snapshots[index], front, m=3, population=91,
                                         reference_hv=front.hypervolume))
    Path(job["result"]).write_text(json.dumps(rows))

elif job["mode"] == "calibrate":
    import calibrate
    calibrate.CALIBRATION_SEEDS = tuple(job["seeds"])
    calibrate.MAX_EVALUATIONS = job["max_evaluations"]
    calibrate.N_CHECKPOINTS = job["n_checkpoints"]
    Path(job["result"]).write_text(json.dumps(calibrate.calibrate_instance(job["instance"])))
"""


def _legacy(job: dict, run: str = "") -> None:
    env = os.environ | {"XOVER_RUN": run, "PYTHONPATH": LEGACY_DIR}
    subprocess.run([sys.executable, "-c", DRIVER, json.dumps(job)], cwd=LEGACY_DIR, env=env, check=True)  # noqa: S603


@pytest.fixture(scope="module")
def legacy_runs(tmp_path_factory) -> Path:
    """The old runner's dumps of every cell, plus the archive runs behind one front."""
    out = tmp_path_factory.mktemp("legacy_runs")
    for run in ("", "zcat", "rotated_dtlz"):
        cells = [[i, a, x, s, b, False] for r, i, a, x, s, b in CELLS if r == run]
        _legacy({"mode": "runs", "out": str(out), "cells": cells}, run)
    archive_root = out / "archives"
    cells = [[FRONT_INSTANCE, a, x, s, BUDGET, True] for a, x, s in FRONT_CELLS]
    _legacy({"mode": "runs", "out": str(archive_root), "cells": cells})
    return out


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: f"{c[1]}-{c[2]}-{c[3]}-s{c[4]}")
def test_run_dumps_are_identical(config, legacy_runs, cell):
    _, instance, algorithm, crossover, seed, budget = cell
    new, _ = run_cell(config, Block(instance, algorithm, crossover, "polynomial"), seed, budget)
    old = load_run(legacy_runs / instance / f"{instance}__{algorithm}__{crossover}__polynomial__s{seed}.npz")
    assert set(new) == set(old)
    for key in old:
        assert np.array_equal(new[key], old[key], equal_nan=True), key


def test_reference_front_is_identical(config, legacy_runs, tmp_path):
    archives = {}
    for algorithm, crossover, seed in FRONT_CELLS:
        arrays, _ = run_cell(
            config, Block(FRONT_INSTANCE, algorithm, crossover, "polynomial"), seed, BUDGET, archive=True
        )
        archives[f"{FRONT_INSTANCE}__{algorithm}__{crossover}__polynomial__s{seed}"] = arrays["archive_F"]
    new = fronts.freeze(
        archives,
        cap=design.subset_cap(config, 3),
        hv_reference=hv_reference_component(3, 91),
        exact_hv_max_m=config["indicators"]["exact_hv_max_m"],
    )
    result = tmp_path / "legacy_front.npz"
    _legacy({"mode": "front", "out": str(legacy_runs / "archives"), "instance": FRONT_INSTANCE, "result": str(result)})
    with np.load(result) as old:
        assert np.array_equal(new.points, old["points"])
        assert np.array_equal(new.ideal, old["ideal"])
        assert np.array_equal(new.nadir, old["nadir"])
        assert new.pool_size == int(old["pool_size"])
        assert new.hypervolume == float(old["hv"])


def test_indicators_are_identical(config, legacy_runs, tmp_path):
    # A front built from a few of the legacy runs' own snapshots is enough to compare the scoring.
    paths = sorted((legacy_runs / FRONT_INSTANCE).glob("*.npz"))[:6]
    pool = {path.stem: load_run(path)["snap_F"] for path in paths}
    front = fronts.freeze(pool, cap=300, hv_reference=hv_reference_component(3, 91), exact_hv_max_m=5)
    front_path = tmp_path / "front.npz"
    np.savez(front_path, points=front.points, ideal=front.ideal, nadir=front.nadir, hv=front.hypervolume,
             pool_size=front.pool_size)  # fmt: skip

    new = []
    for path in paths:
        arrays = load_run(path)
        new += score_run(
            unpack_snapshots(arrays), arrays["snap_evals"], front, fractions=config["fractions"], m=3, population=91
        )
    result = tmp_path / "legacy_scores.json"
    _legacy({"mode": "score", "front": str(front_path), "runs": [str(p) for p in paths], "result": str(result)})
    old = json.loads(result.read_text())
    assert len(new) == len(old)
    for new_row, old_row in zip(new, old, strict=True):
        values = {k: v for k, v in new_row.items() if k not in {"fraction", "evaluations"}}
        assert values.keys() == old_row.keys()
        for key, value in values.items():
            assert value == old_row[key] or (np.isnan(value) and np.isnan(old_row[key])), key


def test_calibration_is_identical(config, tmp_path):
    settings = config["calibration"] | CALIBRATION
    new = calibration.calibrate(config | {"calibration": settings}, "dtlz2_m3_n12")
    result = tmp_path / "legacy_calibration.json"
    _legacy({"mode": "calibrate", "instance": "dtlz2_m3_n12", "result": str(result)} | CALIBRATION)
    old = json.loads(result.read_text())
    assert new["budget"] == old["budget"]
    assert new["saturation_curve"] == old["saturation_curve"]
    assert new["reference_front"] == old["reference_front"]
