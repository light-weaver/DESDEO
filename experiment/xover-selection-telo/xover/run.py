"""Run one EA with the snapshot logger attached, and write what it recorded.

Shared by every phase that runs an algorithm: calibration (phase 1), the archive runs behind the
reference fronts (phase 2) and the main grid (phase 3).
"""

from __future__ import annotations

import io
import json
import platform
import socket
import subprocess
import tarfile
import time
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from xover.logger import DEFAULT_MECHANISM_STRIDE, SnapshotLogger


def run(
    problem,
    options,
    budget: int,
    *,
    fractions: list[float] | None = None,
    n_checkpoints: int | None = None,
    mechanism_stride: int = DEFAULT_MECHANISM_STRIDE,
    archive_dtype=np.float32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Solve `problem` under `options` and return the logger's arrays plus a record of the run.

    Args:
        problem: the DESDEO problem.
        options: its `EMOOptions`; `options.template.use_archive` decides whether `archive_F`, the
            final non-dominated archive, is added to the arrays.
        budget: the evaluation budget the options were built with, for placing checkpoints.
        fractions: budget fractions to snapshot at (the grid's schedule).
        n_checkpoints: log-spaced checkpoints instead of `fractions` (calibration).
        mechanism_stride: record the crossover diagnostics every Nth generation.
        archive_dtype: the archive's dtype. The grid stores float32; calibration keeps float64.
    """
    from desdeo.emo.options.algorithms import emo_constructor

    setup_start = time.perf_counter()
    solve, extras = emo_constructor(options, problem)
    logger_options: dict[str, Any] = {"n_checkpoints": n_checkpoints, "mechanism_stride": mechanism_stride}
    if fractions is not None:
        logger_options["fractions"] = fractions
    logger = SnapshotLogger(problem=problem, publisher=extras.publisher, max_evaluations=budget, **logger_options)
    setup_s = time.perf_counter() - setup_start

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    result = solve()
    # Wall-clock on a loaded machine measures the machine, so cost claims use the CPU time.
    solve_s, solve_cpu_s = time.perf_counter() - wall_start, time.process_time() - cpu_start

    logger.finalise(result.optimal_outputs)
    # Behavioural rather than `publisher.check_consistency()`, which reports False for the composite
    # crossover even though its sub-operators' messages flow.
    logger.check_messages_arrived()

    arrays = logger.arrays()
    if options.template.use_archive:
        arrays["archive_F"] = archive_targets(extras, problem, archive_dtype)

    record = {
        "budget": budget,
        "setup_s": round(setup_s, 4),
        "solve_s": round(solve_s, 4),
        "solve_cpu_s": round(solve_cpu_s, 4),
        "final_evaluations": logger.evaluations,
        "generations": logger.generation,
        "n_snapshots": len(arrays["snap_n"]),
        "archive_rows": int(arrays["archive_F"].shape[0]) if "archive_F" in arrays else 0,
        "metadata": logger.metadata(),
    }
    return arrays, record


def archive_targets(extras, problem, dtype=np.float32) -> np.ndarray:
    """The run's final non-dominated archive, in minimisation sense."""
    archive = getattr(extras, "archive", None)
    solutions = getattr(archive, "solutions", None) if archive is not None else None
    n_obj = len(problem.objectives)
    if solutions is None or solutions.height == 0:
        return np.zeros((0, n_obj), dtype=dtype)
    targets = [f"{obj.symbol}_min" for obj in problem.objectives]
    if all(t in solutions.columns for t in targets):
        return solutions.select(targets).to_numpy().astype(dtype)
    signs = np.array([-1.0 if obj.maximize else 1.0 for obj in problem.objectives], dtype=dtype)
    raw = solutions.select([obj.symbol for obj in problem.objectives]).to_numpy().astype(dtype)
    return raw * signs


def write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write `arrays` compressed, atomically: a killed job never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    """Write `record` as indented JSON, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=2, default=str) + "\n")
    partial.replace(path)


def write_block(path: str | Path, runs: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]]) -> None:
    """Write a block's runs, `{seed: (arrays, record)}`, into one tar, atomically.

    Each seed becomes `s{seed}.npz` and `s{seed}.json`. The tar itself is uncompressed, since the
    `.npz` members already are. One file per block instead of two per run keeps the study inside
    Roihu's scratch file quota.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")

    def add(tar: tarfile.TarFile, name: str, content: bytes) -> None:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))

    with tarfile.open(partial, "w") as tar:
        for seed, (arrays, record) in runs.items():
            npz = io.BytesIO()
            np.savez_compressed(npz, **arrays)
            add(tar, f"s{seed}.npz", npz.getvalue())
            add(tar, f"s{seed}.json", (json.dumps(record, indent=2, default=str) + "\n").encode())
    partial.replace(path)


@cache
def provenance() -> dict[str, Any]:
    """Which code and environment produced a result."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "unknown"

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--", ".", "../../desdeo") not in {"", "unknown"},
        "versions": {p: version(p) for p in ("desdeo", "numpy", "scipy", "polars", "moocore", "pymoo", "scikit-learn")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def run_cell(config: dict[str, Any], block, seed: int, budget: int, *, archive: bool = False):
    """Run one cell of the grid: `block` (a `xover.design.Block`) at `seed`.

    The problem is rebuilt for every cell, so no state can carry over from one run to the next.
    Returns the arrays for the run's `.npz` and the record for its `.json`.
    """
    from xover import design
    from xover.problems import build_problem

    instance = design.instance(config, block.instance)
    problem = build_problem(instance.spec)
    options = design.make_options(
        config,
        block.algorithm,
        block.crossover,
        block.mutation,
        m=instance.m,
        max_evaluations=budget,
        seed=seed,
        use_archive=archive,
    )
    arrays, record = run(
        problem,
        options,
        budget,
        fractions=config["fractions"],
        mechanism_stride=config["mechanism_stride"],
    )
    identity = {
        "instance": block.instance,
        "algorithm": block.algorithm,
        "crossover": block.crossover,
        "mutation": block.mutation,
        "seed": seed,
    }
    return arrays, identity | record | {"provenance": provenance()}
