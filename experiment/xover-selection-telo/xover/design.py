"""The design: which runs exist, and the DESDEO options of each.

Every function takes the workflow config, the plain dict Snakemake loads from `config/config.yaml`,
so the same code serves the Snakefile, the scripts it runs and the tests.

A *cell* is one run: (instance, algorithm, crossover, mutation, seed). A *block* is a cell without
its seed; the workflow runs one job per block, covering every seed, so that each job pays for its
Python start-up once rather than once per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from xover.problems import Instance, instances

ALGORITHMS = ("nsga3", "nsga2", "ibea", "spea2", "agemoea2")
"""The algorithms `make_options` can build."""


class ConfigError(ValueError):
    """The workflow config is inconsistent."""


@dataclass(frozen=True)
class Block:
    """One instance x algorithm x crossover x mutation, run at every seed by one job."""

    instance: str
    algorithm: str
    crossover: str
    mutation: str


# --------------------------------------------------------------------------------------------------
# Selections
# --------------------------------------------------------------------------------------------------


@cache
def _all_instances(suites_key: str) -> dict[str, Instance]:
    import json

    return instances(json.loads(suites_key))


def all_instances(config: dict[str, Any]) -> dict[str, Instance]:
    """Every instance the suites generate."""
    import json

    return _all_instances(json.dumps(config["suites"], sort_keys=True))


def selected_instances(config: dict[str, Any]) -> dict[str, Instance]:
    """The instances `run.instances` names, in registry order."""
    registry = all_instances(config)
    wanted = config["run"]["instances"]
    if wanted == "all":
        return registry
    unknown = [slug for slug in wanted if slug not in registry]
    if unknown:
        msg = f"run.instances names unknown instances: {unknown}"
        raise ConfigError(msg)
    return {slug: registry[slug] for slug in registry if slug in set(wanted)}


def instance(config: dict[str, Any], slug: str) -> Instance:
    """One instance by slug."""
    return all_instances(config)[slug]


def run_blocks(config: dict[str, Any]) -> list[Block]:
    """Every block of the main grid (phase 3)."""
    run = config["run"]
    return [
        Block(slug, algorithm, crossover, mutation)
        for slug in selected_instances(config)
        for algorithm in run["algorithms"]
        for crossover in run["crossovers"]
        for mutation in run["mutations"]
    ]


def front_selection(config: dict[str, Any], key: str) -> list[str]:
    """`reference_front[key]`, with `all` meaning the list under `run`."""
    value = config["reference_front"][key]
    return list(config["run"][key]) if value == "all" else list(value)


def front_blocks(config: dict[str, Any]) -> list[Block]:
    """Every block whose archives feed the reference fronts (phase 2)."""
    return [
        Block(slug, algorithm, crossover, mutation)
        for slug in selected_instances(config)
        for algorithm in front_selection(config, "algorithms")
        for crossover in front_selection(config, "crossovers")
        for mutation in front_selection(config, "mutations")
    ]


def population_size(config: dict[str, Any], m: int) -> int:
    """The population size, and NSGA-III's reference-vector count, at `m` objectives."""
    table = {int(k): int(v) for k, v in config["population_size"].items()}
    if m not in table:
        msg = f"no population size for m={m}; add it under population_size"
        raise ConfigError(msg)
    return table[m]


def subset_cap(config: dict[str, Any], m: int) -> int:
    """How many points a reference front keeps at `m` objectives.

    An exact entry wins; otherwise the largest entry below `m`, or the smallest entry when `m` is
    below them all, or `default` above them all.
    """
    raw = config["reference_front"]["subset_cap"]
    caps = {int(k): int(v) for k, v in raw.items() if k != "default"}
    if m in caps:
        return caps[m]
    if m < min(caps):
        return caps[min(caps)]
    if m <= max(caps):
        return caps[max(k for k in caps if k <= m)]
    return int(raw["default"])


# --------------------------------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------------------------------


def crossover_options(config: dict[str, Any], name: str):
    """The DESDEO `CrossoverOptions` for one crossover of the config."""
    from desdeo.emo.options.crossover import CrossoverOptions
    from pydantic import TypeAdapter

    return TypeAdapter(CrossoverOptions).validate_python(config["crossovers"][name])


def mutation_options(config: dict[str, Any], name: str):
    """The DESDEO `MutationOptions` for one mutation of the config."""
    from desdeo.emo.options.mutation import MutationOptions
    from pydantic import TypeAdapter

    return TypeAdapter(MutationOptions).validate_python(config["mutations"][name])


def make_options(
    config: dict[str, Any],
    algorithm: str,
    crossover: str,
    mutation: str,
    *,
    m: int,
    max_evaluations: int,
    seed: int,
    use_archive: bool,
    population: int | None = None,
):
    """The `EMOOptions` of one run.

    Every operator parameter is set here rather than inherited from DESDEO's per-algorithm
    factories, so the algorithm factor differs in the selection machinery and nothing else. All
    algorithms run at verbosity 2, which the mating tournament of every algorithm except NSGA-III
    needs, and which the snapshot logger needs for the population frames. One seed drives every
    random component.

    Args:
        config: the workflow config.
        algorithm: a name in `ALGORITHMS`.
        crossover: a key of `config["crossovers"]`.
        mutation: a key of `config["mutations"]`.
        m: the number of objectives, which sets the population size.
        max_evaluations: the evaluation budget.
        seed: the run's seed.
        use_archive: keep a non-dominated archive of every evaluated solution.
        population: override the population size (calibration uses a fixed 100).
    """
    from desdeo.emo.options.algorithms import EMOOptions, Template1Options, Template2Options
    from desdeo.emo.options.generator import LHSGeneratorOptions
    from desdeo.emo.options.repair import ClipRepairOptions
    from desdeo.emo.options.scalar_selection import TournamentSelectionOptions
    from desdeo.emo.options.selection import (
        AGEMOEA2SelectorOptions,
        IBEASelectorOptions,
        NSGA2SelectorOptions,
        NSGA3SelectorOptions,
        ReferenceVectorOptions,
        SPEA2SelectorOptions,
    )
    from desdeo.emo.options.termination import MaxEvaluationsTerminatorOptions

    n = population or population_size(config, m)
    settings = config["algorithms"][algorithm]
    common: dict[str, Any] = {
        "crossover": crossover_options(config, crossover),
        "mutation": mutation_options(config, mutation),
        "generator": LHSGeneratorOptions(n_points=n),
        "repair": ClipRepairOptions(),
        "termination": MaxEvaluationsTerminatorOptions(max_evaluations=max_evaluations),
        "use_archive": use_archive,
        "verbosity": 2,
        "seed": seed,
    }

    if algorithm == "nsga3":
        template = Template1Options(
            algorithm_name="NSGA3",
            selection=NSGA3SelectorOptions(
                reference_vector_options=ReferenceVectorOptions(
                    adaptation_frequency=0,
                    creation_type=settings["reference_vectors"],
                    vector_type=settings["vector_type"],
                    lattice_resolution=None,
                    number_of_vectors=n,
                ),
            ),
            **common,
        )
    elif algorithm == "nsga2":
        # Population size, tournament winners and generator size are three separate knobs in DESDEO;
        # all three come from `n`.
        template = Template2Options(
            algorithm_name="NSGA-II",
            selection=NSGA2SelectorOptions(population_size=n),
            mate_selection=TournamentSelectionOptions(tournament_size=settings["tournament_size"], winner_size=n),
            **common,
        )
    elif algorithm == "ibea":
        template = Template2Options(
            algorithm_name="IBEA",
            selection=IBEASelectorOptions(
                population_size=n, kappa=settings["kappa"], binary_indicator=settings["binary_indicator"]
            ),
            mate_selection=TournamentSelectionOptions(tournament_size=settings["tournament_size"], winner_size=n),
            **common,
        )
    elif algorithm == "spea2":
        # `population_size` is the archive, N-bar in the paper, and the tournament winners are the
        # population N. The paper's continuous experiments set the two equal.
        template = Template2Options(
            algorithm_name="SPEA2",
            selection=SPEA2SelectorOptions(population_size=n),
            mate_selection=TournamentSelectionOptions(tournament_size=settings["tournament_size"], winner_size=n),
            **common,
        )
    elif algorithm == "agemoea2":
        template = Template2Options(
            algorithm_name="AGE-MOEA-II",
            selection=AGEMOEA2SelectorOptions(population_size=n),
            mate_selection=TournamentSelectionOptions(tournament_size=settings["tournament_size"], winner_size=n),
            **common,
        )
    else:
        msg = f"unknown algorithm {algorithm!r}; supported: {ALGORITHMS}"
        raise ConfigError(msg)

    return EMOOptions(preference=None, template=template)


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------


def validate(config: dict[str, Any]) -> None:
    """Check the config before any job is planned. Raises `ConfigError` naming every problem."""
    from pydantic import ValidationError

    problems: list[str] = []
    run = config["run"]

    for key in ("algorithms", "crossovers", "mutations"):
        problems += [f"run.{key} names {name!r}, which is not defined" for name in run[key] if name not in config[key]]
        problems += [
            f"reference_front.{key} names {name!r}, which run.{key} does not"
            for name in front_selection(config, key)
            if name not in run[key]
        ]
        chosen = config["calibration"][key[:-1]]
        if chosen not in config[key]:
            problems.append(f"calibration.{key[:-1]} names {chosen!r}, which is not defined")
    problems += [f"unknown algorithm {name!r}" for name in config["algorithms"] if name not in ALGORITHMS]

    for kind, build in (("crossovers", crossover_options), ("mutations", mutation_options)):
        for name in config[kind]:
            try:
                build(config, name)
            except ValidationError as error:
                problems += [f"{kind}.{name}.{entry['loc']}: {entry['msg']}" for entry in readable_errors(error)]

    try:
        chosen = selected_instances(config)
    except ConfigError as error:
        problems.append(str(error))
        chosen = {}
    for slug, inst in chosen.items():
        if str(inst.m) not in {str(k) for k in config["population_size"]}:
            problems.append(f"{slug}: no population size for m={inst.m}")
        # WFG needs k = 2(m-1) position parameters plus at least one distance parameter; with fewer
        # variables pymoo evaluates it to all-NaN, silently.
        if inst.spec.get("name", "").startswith("wfg") and inst.n_var < 2 * (inst.m - 1) + 1:
            problems.append(f"{slug}: WFG needs n_var >= 2(m-1)+1")
        if inst.m < config["suites"].get("real_world", {}).get("min_objectives", 3):
            problems.append(f"{slug}: fewer objectives than min_objectives")

    if problems:
        raise ConfigError("invalid workflow config:\n  " + "\n  ".join(problems))


def readable_errors(error) -> list[dict[str, str]]:
    """Reduce a Pydantic error on an untagged union to the branch the config actually chose.

    DESDEO's option unions are untagged, so one wrong field makes Pydantic try every member and report
    a failure for each; a single out-of-range probability yields dozens of errors. The rejections of
    the other members each carry a `literal_error` on their `name` field, so every error nested under
    such a branch is dropped, and the `*Options` branch tags are stripped from what remains. Ported
    from `experiment/ea_builder/runner.py`.
    """
    entries = error.errors()
    rejected = {
        entry["loc"][:-1]
        for entry in entries
        if entry["loc"] and entry["loc"][-1] == "name" and entry["type"] == "literal_error"
    }
    readable = []
    for entry in entries:
        loc = entry["loc"]
        if any(loc[: len(prefix)] == prefix for prefix in rejected):
            continue
        path = [
            str(part) for part in loc if not (isinstance(part, str) and part.endswith("Options") and part[:1].isupper())
        ]
        readable.append({"loc": ".".join(path), "msg": entry["msg"]})
    return readable
