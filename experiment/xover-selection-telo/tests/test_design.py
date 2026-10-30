"""The design as the config defines it: instances, grid size, options and validation."""

import copy

import pytest

from xover import design


def test_config_is_valid(config):
    design.validate(config)


def test_grid_size(config):
    """150 instances x 5 algorithms x 14 crossovers x 1 mutation, at 11 seeds each."""
    instances = design.selected_instances(config)
    assert len(instances) == 150
    suites = [i.suite for i in instances.values()]
    assert {s: suites.count(s) for s in set(suites)} == {
        "dtlz_wfg": 64,
        "zcat": 40,
        "rotated_dtlz": 4,
        "real_world": 42,
    }
    families = [i.family for i in instances.values()]
    assert {f: families.count(f) for f in set(families)} == {"benchmark": 108, "real_world": 42}
    assert len(design.run_blocks(config)) * len(config["run"]["seeds"]) == 115_500
    assert len(design.front_blocks(config)) * len(config["reference_front"]["seeds"]) == 150 * 140


def test_readmitted_instance_is_present(config):
    assert "re41_m3_f_2-f_3-f_4" in design.selected_instances(config)


def test_every_option_combination_builds(config):
    for algorithm in config["run"]["algorithms"]:
        for crossover in config["run"]["crossovers"]:
            options = design.make_options(
                config, algorithm, crossover, "polynomial", m=5, max_evaluations=1000, seed=0, use_archive=False
            )
            template = options.template
            assert template.verbosity == 2
            assert template.generator.n_points == 210
            assert template.seed == 0


@pytest.mark.parametrize(
    ("algorithm", "name", "selector"),
    [("spea2", "SPEA2", "SPEA2Selector"), ("agemoea2", "AGE-MOEA-II", "AGEMOEA2Selector")],
)
def test_added_algorithms_differ_from_the_others_in_selection_only(config, algorithm, name, selector):
    def build(chosen):
        return design.make_options(
            config, chosen, "de_current", "polynomial", m=5, max_evaluations=1000, seed=0, use_archive=False
        ).template

    added, nsga2 = build(algorithm), build("nsga2")
    assert (added.algorithm_name, added.selection.name) == (name, selector)
    assert added.selection.population_size == added.mate_selection.winner_size == 210
    differing = {key for key in type(added).model_fields if getattr(added, key) != getattr(nsga2, key)}
    assert differing == {"algorithm_name", "selection"}


def test_the_two_de_arms_are_different_operators(config):
    rand = design.crossover_options(config, "de")
    current = design.crossover_options(config, "de_current")
    assert (rand.name, rand.scaling_factor, rand.xover_probability) == ("DifferentialEvolutionCrossover", 0.5, 0.9)
    assert (current.name, current.scaling_factor, current.xover_probability) == (
        "DifferentialEvolutionCurrentCrossover",
        0.5,
        1.0,
    )


def test_a_bad_operator_value_gives_one_readable_error(config):
    broken = copy.deepcopy(config)
    broken["crossovers"]["sbx_default"]["xover_probability"] = 1.5
    with pytest.raises(design.ConfigError) as error:
        design.validate(broken)
    lines = str(error.value).splitlines()[1:]
    assert lines == ["  crossovers.sbx_default.xover_probability: Input should be less than or equal to 1"]


def test_an_undefined_name_is_refused(config):
    broken = copy.deepcopy(config)
    broken["run"]["crossovers"] = [*broken["run"]["crossovers"], "missing"]
    with pytest.raises(design.ConfigError, match="'missing'"):
        design.validate(broken)


def test_front_selection_all_follows_run(config):
    assert design.front_selection(config, "crossovers") == config["run"]["crossovers"]


@pytest.mark.parametrize(("m", "cap"), [(3, 1500), (4, 2000), (5, 2000), (6, 3000), (8, 4000), (11, 4000)])
def test_subset_cap(config, m, cap):
    assert design.subset_cap(config, m) == cap
