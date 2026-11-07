"""Regret, ratios, anytime summaries, problem labels and arm attributes."""

import numpy as np
import polars as pl
import pytest

from xover import anytime, design, plotstyle, problem_properties
from xover.indicators import LOWER_IS_BETTER, add_regret, log_ratios


def _one_instance(indicator: str, values: list[float]) -> pl.DataFrame:
    """Return one instance at one fraction with one row per arm."""
    return pl.DataFrame(
        {
            "instance": ["i"] * len(values),
            "fraction": [1.0] * len(values),
            "algorithm": ["p"] * len(values),
            "crossover": [f"x{i}" for i in range(len(values))],
            "mutation": ["polynomial"] * len(values),
            indicator: values,
        }
    )


@pytest.mark.parametrize("indicator", ["igd_plus", "r2"])
def test_regret_is_zero_for_the_best_configuration_whichever_way_the_indicator_points(indicator):
    values = [1.0, 2.0, 3.0] if LOWER_IS_BETTER[indicator] else [-1.0, -2.0, -3.0]
    regret = dict(add_regret(_one_instance(indicator, values), indicator).select(["crossover", "regret"]).iter_rows())
    assert regret == pytest.approx({"x0": 0.0, "x1": 0.5, "x2": 1.0})


def test_regret_keeps_the_raw_value():
    out = add_regret(_one_instance("r2", [-3.0, -1.0]), "r2")
    assert sorted(out["value"].to_list()) == [-3.0, -1.0]


def test_regret_refuses_an_indicator_without_an_orientation():
    with pytest.raises(ValueError, match="no orientation"):
        add_regret(_one_instance("spacing", [0.1, 0.2]), "spacing")


def test_regret_depends_on_the_rows_it_is_given():
    # Pitfall 1 of docs/phase6.md: the bounds come from the rows passed in, so filter first.
    table = _one_instance("igd_plus", [1.0, 2.0, 9.0])
    kept = table.filter(pl.col("crossover") != "x2")
    assert add_regret(kept, "igd_plus").filter(pl.col("crossover") == "x1")["regret"].item() == 1.0
    filtered_after = add_regret(table, "igd_plus").filter(pl.col("crossover") == "x1")
    assert filtered_after["regret"].item() == pytest.approx(0.125)


def test_log_ratios_are_zero_for_each_blocks_best_arm_and_floor_the_hypervolume_gap():
    table = pl.DataFrame(
        {
            "instance": ["i"] * 4,
            "algorithm": ["p", "p", "q", "q"],
            "crossover": ["a", "b", "a", "b"],
            "igd_plus": [0.1, 1.0, 0.4, 0.2],
            "hv_relative": [-0.01, 0.01, 0.1, 0.1],
            "r2": [-1.0, -2.0, -1.0, -2.0],
        }
    )
    ratios = {(r["algorithm"], r["crossover"]): r["log_ratio"] for r in log_ratios(table, "igd_plus").to_dicts()}
    assert ratios == pytest.approx({("p", "a"): 0.0, ("p", "b"): 1.0, ("q", "a"): np.log10(2.0), ("q", "b"): 0.0})
    floored = {(r["algorithm"], r["crossover"]): r["log_ratio"] for r in log_ratios(table, "hv_relative").to_dicts()}
    assert floored[("p", "b")] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no meaningful ratio"):
        log_ratios(table, "r2")


def _curves(values: dict[str, list[float]], indicator: str = "hv_relative") -> pl.DataFrame:
    """Return one run per arm on one instance, at fractions 0.0, 0.5 and 1.0."""
    return pl.DataFrame(
        [
            {
                "instance": "i",
                "algorithm": "p",
                "crossover": arm,
                "mutation": "polynomial",
                "seed": 0,
                "fraction": fraction,
                indicator: value,
            }
            for arm, series in values.items()
            for fraction, value in zip((0.0, 0.5, 1.0), series, strict=True)
        ]
    )


def test_aocc_separates_two_arms_that_end_equal():
    table = _curves({"early": [0.5, 0.1, 0.1], "late": [0.5, 0.5, 0.1]})
    areas = dict(anytime.aocc(table).select(["crossover", "aocc"]).iter_rows())
    assert areas == pytest.approx({"early": 0.2, "late": 0.4})
    finals = anytime.aocc(table)["final"].to_list()
    assert finals == pytest.approx([0.1, 0.1])


def test_aocc_orients_a_higher_is_better_indicator():
    table = _curves({"a": [-3.0, -2.0, -1.0]}, "r2")
    assert anytime.aocc(table, "r2")["aocc"].item() == pytest.approx(2.0)


def test_aocc_refuses_an_indicator_with_no_direction():
    with pytest.raises(ValueError, match="no orientation"):
        anytime.aocc(_curves({"a": [1.0, 1.0, 1.0]}, "spacing"), "spacing")


def test_time_to_target_is_censored_at_the_end_when_the_target_is_never_reached():
    table = _curves({"fast": [1.0, 0.0, 0.0], "slow": [1.0, 1.0, 0.0], "never": [1.0, 1.0, 1.0]})
    out = {row["crossover"]: row for row in anytime.time_to_target(table).to_dicts()}
    assert (out["fast"]["reached_at"], out["fast"]["censored"]) == (0.5, False)
    assert (out["slow"]["reached_at"], out["slow"]["censored"]) == (1.0, False)
    assert (out["never"]["reached_at"], out["never"]["censored"]) == (1.0, True)


@pytest.fixture(scope="module")
def labelled(config) -> pl.DataFrame:
    """Return all 150 instances with their labels."""
    frame = pl.DataFrame(
        [
            {"instance": i.slug, "suite": i.suite, "base": i.base, "family": i.family, "m": i.m}
            for i in design.selected_instances(config).values()
        ]
    )
    return problem_properties.annotate(frame)


def test_every_one_of_the_150_instances_is_labelled(labelled):
    assert labelled.height == 150
    assert labelled["instance"].n_unique() == 150
    for factor, levels in problem_properties.FACTORS.items():
        assert set(labelled[factor].unique().to_list()) <= set(levels), factor


def test_the_level_counts(labelled):
    counts = {
        factor: {level: labelled.filter(pl.col(factor) == level).height for level in levels}
        for factor, levels in problem_properties.FACTORS.items()
    }
    assert counts["separable"] == {"yes": 54, "no": 52, "unknown": 44}
    assert counts["modality"] == {"unimodal": 58, "multimodal": 42, "deceptive": 8, "unknown": 42}
    assert counts["pareto_set"] == {"product": 78, "linked": 30, "unknown": 42}
    assert counts["family"] == {"benchmark": 108, "real_world": 42}
    assert counts["objectives"] == {"3": 96, "4-5": 33, "6-11": 21}


def test_the_readmitted_re41_subset_takes_the_real_world_labels(labelled):
    row = labelled.filter(pl.col("instance") == "re41_m3_f_2-f_3-f_4").to_dicts()[0]
    assert (row["base"], row["pareto_set"], row["objectives"]) == ("re41", "unknown", "3")


def test_only_real_world_problems_are_unknown(labelled):
    for factor in ("front", "modality", "pareto_set"):
        unknown = labelled.filter(pl.col(factor) == "unknown")
        assert set(unknown["family"].to_list()) == {"real_world"}, factor


def test_every_linked_instance_is_non_separable(labelled):
    linked = labelled.filter(pl.col("pareto_set") == "linked")
    assert set(linked["separable"].to_list()) == {"no"}


def test_zcat_labels_follow_the_level_and_the_pareto_set_switch(config):
    instances = [i for i in design.selected_instances(config).values() if i.suite == "zcat"]
    assert len(instances) == 40
    for instance in instances:
        extra = instance.spec["extra"]
        level, linked = extra["level"], extra["complicated_pareto_set"]
        assert problem_properties.parse_zcat_slug(instance.slug) == (instance.base, level, linked)
        labels = problem_properties.describe(instance.slug, instance.base, instance.family, instance.m)
        assert labels["pareto_set"] == ("linked" if linked else "product")
        assert labels["modality"] == ("multimodal" if problem_properties.ZCAT_MULTIMODAL[level] else "unimodal")
        separable = problem_properties.ZCAT_SEPARABLE[level] and not linked
        assert labels["separable"] == ("yes" if separable else "no")


def test_a_rotation_makes_the_problem_non_separable_and_its_pareto_set_linked(config):
    instances = [i for i in design.selected_instances(config).values() if i.suite == "rotated_dtlz"]
    assert len(instances) == 4
    for instance in instances:
        rotated = instance.spec["rotated"]
        assert problem_properties.is_rotated(instance.slug) is rotated
        labels = problem_properties.describe(instance.slug, instance.base, instance.family, instance.m)
        assert (labels["separable"], labels["pareto_set"]) == (("no", "linked") if rotated else ("yes", "product"))


def test_an_unknown_benchmark_base_or_slug_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no labels"):
        problem_properties.describe("zdt1_m3_n30", "zdt1", "benchmark", 3)
    with pytest.raises(ValueError, match="not a ZCAT"):
        problem_properties.parse_zcat_slug("zcat1_l9_ps0_m3_n30")
    with pytest.raises(ValueError, match="not a rotated"):
        problem_properties.is_rotated("r2_dtlz4_m5_n30")


def test_every_split_names_levels_of_its_factor():
    for split in problem_properties.SPLITS:
        levels = problem_properties.FACTORS[split.factor]
        assert set(split.sides) <= set(levels), split.name
        for factor, level in split.within:
            assert level in problem_properties.FACTORS[factor], split.name


def test_arm_attributes(config):
    crossovers = config["crossovers"]
    assert plotstyle.has_uniform(crossovers["sbx_default"]) is True
    assert plotstyle.has_uniform(crossovers["sbx_no_uniform"]) is False
    assert plotstyle.has_uniform(crossovers["none"]) is False
    assert plotstyle.has_uniform(crossovers["composite"]) is plotstyle.MIXED
    assert plotstyle.is_sbx(crossovers["composite"]) is plotstyle.MIXED
    assert plotstyle.is_sbx(crossovers["none"]) is False
    assert not plotstyle.crosses_anything(crossovers["none"])


def test_every_arm_has_a_distinct_colour_and_a_line_style(config):
    crossovers = config["crossovers"]
    colours = [plotstyle.colour(arm, crossovers) for arm in crossovers]
    assert len(colours) == len(set(colours)) == len(crossovers)
    assert all(plotstyle.arm_style(arm, crossovers)["linestyle"] is not None for arm in crossovers)
