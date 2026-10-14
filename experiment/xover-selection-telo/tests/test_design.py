"""The design as the config defines it: instances, grid size, options and validation."""

import copy

import pytest

from xover import design


def test_config_is_valid(config):
    design.validate(config)


def test_grid_size(config):
    """150 instances x 3 algorithms x 13 crossovers x 1 mutation, at 11 seeds each."""
    instances = design.selected_instances(config)
    assert len(instances) == 150
    assert {suite: sum(i.suite == suite for i in instances.values()) for suite in ("zcat", "rotated_dtlz")} == {
        "zcat": 40,
        "rotated_dtlz": 4,
    }
    assert len(design.run_blocks(config)) * len(config["run"]["seeds"]) == 64_350
    assert len(design.front_blocks(config)) * len(config["reference_front"]["seeds"]) == 150 * 78


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
