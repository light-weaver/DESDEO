"""The problem registry: expand `config["suites"]` into instances, and build any instance's problem.

An instance is a slug plus a serialisable recipe (`spec`), so a job can rebuild its problem from the
slug alone. Building is lazy: listing the instances builds only the cheap analytic RE problems, to
read their objective symbols, and never MetallAppl or GAA, whose builders read datasets and fit
surrogate models.

Each instance has a suite, the set of problems it was generated from, and a family. The DTLZ/WFG,
ZCAT and rotated-DTLZ suites are all benchmark problems; the real-world suite is the only real-world
family.

Slugs:

- DTLZ and WFG: `{problem}_m{m}_n{n_var}`, for example `dtlz2_m3_n12`;
- real-world problems: `{base}_m{m}` for the full objective set and `{base}_m{m}_{f_1-f_3-...}` for a
  sampled subset, `gaa_m{m}` and `metallappl_{variant}_m{m}`;
- ZCAT: `{problem}_l{level}_ps{0|1}_m3_n30`;
- rotated DTLZ: `r2_{dtlz2|dtlz3}[_unrotated]_m5_n30`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from functools import cache
from typing import Any


@dataclass(frozen=True)
class Instance:
    """One problem instance: a slug, how to rebuild it, and what it is grouped with."""

    slug: str
    suite: str
    """`dtlz_wfg`, `zcat`, `rotated_dtlz` or `real_world`: the key under `config["suites"]`."""
    base: str
    """The base problem. Subsets of one base problem share a decision space (or, for MetallAppl,
    datasets and fitted models), so analyses that need independent instances group by this."""
    family: str
    """`benchmark` (the DTLZ/WFG, ZCAT and rotated-DTLZ suites) or `real_world`."""
    m: int
    n_var: int
    spec: dict[str, Any] = field(compare=False)


def instances(suites: dict[str, Any]) -> dict[str, Instance]:
    """Every instance the suites generate, keyed by slug, in a fixed order."""
    out: dict[str, Instance] = {}
    generated = [
        *_dtlz_wfg(suites.get("dtlz_wfg")),
        *_real_world(suites.get("real_world")),
        *_zcat(suites.get("zcat")),
        *_rotated_dtlz(suites.get("rotated_dtlz")),
    ]
    for instance in generated:
        if instance.slug in out:
            msg = f"duplicate instance slug {instance.slug!r}"
            raise ValueError(msg)
        out[instance.slug] = instance
    return out


# --------------------------------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------------------------------


def _dtlz_wfg(suite: dict[str, Any] | None) -> list[Instance]:
    if not suite:
        return []
    return [
        Instance(
            slug=f"{name}_m{m}_n{n_var}",
            suite="dtlz_wfg",
            base=name,
            family="benchmark",
            m=m,
            n_var=n_var,
            spec={"kind": "pymoo", "name": name, "n_var": n_var, "n_obj": m},
        )
        for m, n_var in suite["blocks"]
        for name in suite["problems"]
    ]


def _real_world(suite: dict[str, Any] | None) -> list[Instance]:
    if not suite:
        return []
    out: list[Instance] = []
    for base, recipe in suite.get("bases", {}).items():
        spec = {"kind": "testproblem", "func": recipe["func"]}
        if recipe.get("kwargs"):
            spec["kwargs"] = dict(recipe["kwargs"])
        symbols, n_var = _shape(spec)
        required = tuple(tuple(subset) for subset in recipe.get("required", ()))
        subsets = objective_subsets(
            symbols,
            cap=suite["subset_cap"],
            seed=suite["subset_seed"] + hash_str(base),
            min_objectives=suite["min_objectives"],
            required=required,
        )
        for subset in subsets:
            m = len(subset)
            full = m == len(symbols)
            out.append(
                Instance(
                    slug=f"{base}_m{m}" if full else f"{base}_m{m}_{'-'.join(subset)}",
                    suite="real_world",
                    base=base,
                    family="real_world",
                    m=m,
                    n_var=n_var,
                    spec=spec | {"objectives": None if full else list(subset)},
                )
            )

    for m, subset in suite.get("gaa_ladder", {}).items():
        out.append(
            Instance(
                slug=f"gaa_m{m}",
                suite="real_world",
                base="gaa",
                family="real_world",
                m=int(m),
                n_var=27,
                spec={"kind": "gaa", "objectives": None if subset == "all" else list(subset)},
            )
        )

    for name, objectives in suite.get("metallappl_variants", {}).items():
        out.append(
            Instance(
                slug=f"metallappl_{name}_m{len(objectives)}",
                suite="real_world",
                base="metallappl",
                family="real_world",
                m=len(objectives),
                n_var=17,
                spec={"kind": "metallappl", "objectives": list(objectives)},
            )
        )

    excluded = set(suite.get("exclude") or ())
    return [instance for instance in out if instance.slug not in excluded]


def _zcat(suite: dict[str, Any] | None) -> list[Instance]:
    if not suite:
        return []
    m, n_var = suite["m"], suite["n_var"]
    return [
        Instance(
            slug=f"{problem}_l{level}_ps{int(linked)}_m{m}_n{n_var}",
            suite="zcat",
            base=problem,
            family="benchmark",
            m=m,
            n_var=n_var,
            spec={
                "kind": "pymoo",
                "name": problem,
                "n_var": n_var,
                "n_obj": m,
                "extra": {"level": level, "complicated_pareto_set": linked},
            },
        )
        for problem in suite["problems"]
        for level in suite["levels"]
        for linked in suite["linked"]
    ]


def _rotated_dtlz(suite: dict[str, Any] | None) -> list[Instance]:
    if not suite:
        return []
    from xover.rotated_dtlz import N_OBJ, N_VAR, slug

    return [
        Instance(
            slug=slug(base, rotated=rotated),
            suite="rotated_dtlz",
            base=f"r2_{base}",
            family="benchmark",
            m=N_OBJ,
            n_var=N_VAR,
            spec={"kind": "cec2009_rotated", "base": base, "rotated": rotated},
        )
        for base in suite["bases"]
        for rotated in suite["rotated"]
    ]


# --------------------------------------------------------------------------------------------------
# Objective subsets of the real-world problems
# --------------------------------------------------------------------------------------------------


def objective_subsets(
    symbols: tuple[str, ...],
    *,
    cap: int,
    seed: int,
    min_objectives: int,
    required: tuple[tuple[str, ...], ...] = (),
) -> list[tuple[str, ...]]:
    """Choose up to `cap` objective subsets of `symbols`, always including the full set.

    `required` subsets are always included. The remaining slots are filled round-robin over
    decreasing sizes, m-1, m-2, ..., `min_objectives`, and round again, each size drawing from its own
    shuffled pool, so the instances spread across objective counts instead of piling up on drop-one
    subsets.
    """
    m = len(symbols)
    chosen: list[tuple[str, ...]] = [symbols]
    if m <= min_objectives:
        return chosen
    order = {s: i for i, s in enumerate(symbols)}
    for subset in required:
        ordered = tuple(sorted(subset, key=order.__getitem__))
        if ordered not in chosen:
            chosen.append(ordered)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    pools = {size: sorted(tuple(c) for c in itertools.combinations(symbols, size)) for size in range(min_objectives, m)}
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = dict.fromkeys(pools, 0)
    sizes = list(range(m - 1, min_objectives - 1, -1))
    seen = set(chosen)
    while len(chosen) < cap and any(cursors[s] < len(pools[s]) for s in sizes):
        for size in sizes:
            if len(chosen) >= cap:
                break
            while cursors[size] < len(pools[size]) and pools[size][cursors[size]] in seen:
                cursors[size] += 1
            if cursors[size] < len(pools[size]):
                candidate = pools[size][cursors[size]]
                chosen.append(candidate)
                seen.add(candidate)
                cursors[size] += 1
    return chosen


def hash_str(text: str) -> int:
    """A stable small integer from a string, so subset sampling does not depend on `PYTHONHASHSEED`."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


@cache
def _shape_cached(spec_key: str) -> tuple[tuple[str, ...], int]:
    problem = build_problem(json.loads(spec_key))
    return tuple(obj.symbol for obj in problem.objectives), len(problem.get_flattened_variables())


def _shape(spec: dict[str, Any]) -> tuple[tuple[str, ...], int]:
    """The objective symbols and variable count of a problem, read from the problem itself."""
    return _shape_cached(json.dumps(spec, sort_keys=True))


# --------------------------------------------------------------------------------------------------
# Building problems
# --------------------------------------------------------------------------------------------------


def build_problem(spec: dict[str, Any]):
    """Rebuild a DESDEO problem from its specification."""
    kind = spec["kind"]
    objectives = spec.get("objectives")

    if kind == "pymoo":
        from desdeo.problem.external.pymoo_provider import PymooProblemParams, create_pymoo_problem

        return create_pymoo_problem(
            PymooProblemParams(name=spec["name"], n_var=spec["n_var"], n_obj=spec["n_obj"], extra=spec.get("extra"))
        )

    if kind == "cec2009_rotated":
        from xover.rotated_dtlz import build_problem as build_rotated_dtlz

        return build_rotated_dtlz(spec)

    if kind == "metallappl":
        from desdeo.problem.testproblems import metallurgical_application

        # MetallAppl takes the subset itself, because its decision box is derived from the datasets
        # behind the chosen surrogates. Subsetting with model_copy would keep the wrong box.
        return metallurgical_application(objectives=tuple(objectives))

    if kind == "gaa":
        from desdeo.problem.testproblems import gaa

        problem = gaa(fold_constraint=True)
    elif kind == "testproblem":
        from desdeo.problem import testproblems

        problem = getattr(testproblems, spec["func"])(**spec.get("kwargs", {}))
    else:
        msg = f"unknown problem kind {kind!r}"
        raise ValueError(msg)

    if objectives is None:
        return problem
    keep = {o.symbol: o for o in problem.objectives}
    return problem.model_copy(update={"objectives": [keep[s] for s in objectives]})
