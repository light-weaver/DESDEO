"""The tier classification and the falsification tests in `analysis/tier.py`."""

import polars as pl
import pytest

from analysis.tier import (
    MIXING_SIDE,
    STEP_CONTROL_EXCLUDE,
    budget_consistency,
    cuts,
    falsification,
    step_control,
    summarise,
    tier,
    variance_share,
)
from xover import problem_properties

EXPECTED = {
    "sbx_default": "mixing",
    "sbx_default_eta20": "mixing",
    "sbx_platemo": "mixing",
    "sbx_platemo_eta20": "mixing",
    "sbx_jmetal": "mixing",
    "sbx_pisa": "mixing",
    "bex_uniform": "mixing",
    "uniform": "mixing",
    "uniform_pm50": "mixing",
    "uniform_pm100": "mixing",
    "composite": "portfolio",
    "sbx_no_uniform": "non_mixing",
    "bex": "non_mixing",
    "blend": "non_mixing",
    "sax": "non_mixing",
    "pcx": "non_mixing",
    "de_current": "non_mixing",
    "de": "de",
    "none": "off",
}


def test_every_arm_has_the_expected_tier(config):
    assert {arm: tier(config["crossovers"][arm]) for arm in config["run"]["crossovers"]} == EXPECTED


def test_the_mixing_side_has_eleven_arms():
    assert sum(t in MIXING_SIDE for t in EXPECTED.values()) == 11


def test_the_mutation_probes_are_outside_the_step_control(config):
    # Their offspring step is measured before their mutation, so it would understate how far they move.
    assert set(config["run"]["mutation_overrides"]) <= set(STEP_CONTROL_EXCLUDE)


def test_the_off_arm_is_not_counted_as_mixing_despite_its_uniform_rate(config):
    assert config["crossovers"]["none"]["uniform_xover_probability"] > 0
    assert tier(config["crossovers"]["none"]) == "off"


def test_summarise_names_an_arm_that_breaks_into_the_top_places():
    tiers = {"a": "mixing", "b": "mixing", "c": "non_mixing", "d": "non_mixing"}
    clean = summarise({"a": 1.2, "b": 1.8, "c": 3.1, "d": 3.9}, tiers, 10)
    assert clean["top_k_is_mixing_side"]
    assert clean["best_other"] == {"arm": "c", "position": 3}
    broken = summarise({"a": 1.2, "c": 1.8, "b": 3.1, "d": 3.9}, tiers, 10)
    assert not broken["top_k_is_mixing_side"]
    assert broken["intruders"] == ["c"]
    assert broken["displaced"] == ["b"]


def _rows(scores: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Return indicator rows at the final fraction: {algorithm: {arm: igd_plus}} on four instances."""
    return pl.DataFrame(
        [
            {
                "instance": f"i{instance}",
                "fraction": 1.0,
                "algorithm": algorithm,
                "crossover": arm,
                "mutation": "polynomial",
                "seed": 0,
                "igd_plus": value + 0.01 * instance,
            }
            for instance in range(4)
            for algorithm, arms in scores.items()
            for arm, value in arms.items()
        ]
    )


TIERS = {"m1": "mixing", "m2": "mixing", "d": "de", "n1": "non_mixing"}


def test_falsification_passes_when_the_mixing_side_leads_everywhere():
    out = falsification(_rows({"p": {"m1": 0.1, "m2": 0.2, "d": 0.8, "n1": 0.9}}), "igd_plus", TIERS)
    assert not out["all"]["falsified"]
    assert not out["p"]["falsified"]
    assert out["p"]["mixing_split_share"] > 0.5


def test_falsification_catches_a_failure_inside_one_algorithm_that_the_pool_hides():
    rows = _rows(
        {
            "p": {"m1": 0.1, "m2": 0.2, "d": 0.8, "n1": 0.9},
            "q": {"m1": 0.1, "m2": 0.2, "d": 0.8, "n1": 0.9},
            "r": {"m1": 0.1, "m2": 0.9, "d": 0.8, "n1": 0.2},
        }
    )
    out = falsification(rows, "igd_plus", TIERS)
    assert not out["all"]["falsified"]
    assert not out["p"]["falsified"]
    assert out["r"]["falsified"]
    assert out["r"]["non_mixing_in_top_k"] == ["n1"]


def test_a_de_arm_in_the_top_places_is_reported_and_does_not_falsify():
    tiers = {**TIERS, "n2": "non_mixing", "n3": "non_mixing"}
    rows = _rows({"p": {"m1": 0.1, "d": 0.2, "m2": 0.21, "n1": 0.9, "n2": 0.9, "n3": 0.9}})
    out = falsification(rows, "igd_plus", tiers)["p"]
    assert out["de_in_top_k"] == ["d"]
    assert out["non_mixing_in_top_k"] == []
    assert not out["falsified"]


def test_a_minority_share_falsifies_even_with_the_right_top_places():
    tiers = {"m": "mixing", "n1": "non_mixing", "n2": "non_mixing", "n3": "non_mixing"}
    # The top place is right, but most of the gap between arms lies inside the non-mixing group.
    rows = _rows({"p": {"m": 0.30, "n1": 0.31, "n2": 0.32, "n3": 1.0}})
    out = falsification(rows, "igd_plus", tiers)["p"]
    assert out["non_mixing_in_top_k"] == []
    assert out["mixing_split_share"] < 0.5
    assert out["falsified"]


def test_the_variance_share_is_one_when_only_the_split_separates_the_arms():
    frame = pl.DataFrame(
        {
            "instance": ["i"] * 4,
            "algorithm": ["p"] * 4,
            "crossover": ["a", "b", "c", "d"],
            "regret": [0.0, 0.0, 1.0, 1.0],
        }
    )
    tiers = {"a": "mixing", "b": "portfolio", "c": "non_mixing", "d": "off"}
    assert variance_share(frame, tiers)["share_of_between_crossover_ss"] == pytest.approx(1.0)


def test_cuts_follow_the_problem_properties():
    frame = problem_properties.annotate(
        pl.DataFrame(
            {
                "instance": [
                    "wfg2_m3_n12",
                    "wfg8_m3_n12",
                    "zcat1_l2_ps0_m3_n30",
                    "zcat1_l1_ps1_m3_n30",
                    "r2_dtlz2_m5_n30",
                    "river_pollution_m5",
                    "dtlz1_m8_n24",
                ],
                "base": ["wfg2", "wfg8", "zcat1", "zcat1", "r2_dtlz2", "river_pollution", "dtlz1"],
                "family": ["benchmark"] * 5 + ["real_world", "benchmark"],
                "m": [3, 3, 3, 3, 5, 5, 8],
                "algorithm": ["a"] * 7,
            }
        )
    )
    out = cuts(frame)

    def members(cut: str) -> list[str]:
        return sorted(out[cut]["instance"].to_list())

    assert members("pareto_set=linked") == ["r2_dtlz2_m5_n30", "wfg8_m3_n12", "zcat1_l1_ps1_m3_n30"]
    assert members("separable=no&pareto_set=product") == ["wfg2_m3_n12", "zcat1_l2_ps0_m3_n30"]
    assert members("separable=yes") == ["dtlz1_m8_n24", "river_pollution_m5"]
    assert members("objectives=6-11") == ["dtlz1_m8_n24"]
    assert members("family=real_world") == ["river_pollution_m5"]
    assert "pareto_set=unknown" in out
    assert "modality=deceptive" not in out


def test_step_control_credits_mixing_when_the_step_does_not_follow_the_ranks():
    tiers = {"m1": "mixing", "m2": "mixing", "n1": "non_mixing", "n2": "non_mixing"}
    arms = [("m1", 0.1, 0.02), ("m2", 0.2, 0.05), ("n1", 0.6, 0.06), ("n2", 0.7, 0.01)]
    rows = [(f"i{b}", "a", arm, regret, step) for b in range(6) for arm, regret, step in arms]
    frame = pl.DataFrame(rows, schema=["instance", "algorithm", "crossover", "regret", "offspring_step"], orient="row")
    out = step_control(frame, frame, tiers, pairs=(("m1", "n1"),), exclude=())
    # Centred ranks (-1.5, -0.5, 0.5, 1.5): the mixing dummy leaves residual SS 1 of 5, the step rank 4.8 of 5.
    assert out["r2_within_block"]["mixing"] == pytest.approx(0.8)
    assert out["r2_within_block"]["step_rank"] == pytest.approx(0.04)
    assert out["matched"]["m1_vs_n1"] == {"blocks_where_b_steps_further": 6, "a_wins_there": 1.0, "a_wins_overall": 1.0}


def test_budget_consistency_compares_the_ranking_at_every_pair_of_fractions():
    order = {0.1: ["a", "b", "c"], 0.5: ["a", "b", "c"], 1.0: ["c", "b", "a"]}
    rows = [
        {"instance": f"i{i}", "algorithm": "p", "crossover": arm, "fraction": fraction, "regret": float(rank)}
        for i in range(4)
        for fraction, arms in order.items()
        for rank, arm in enumerate(arms)
    ]
    out = budget_consistency(pl.DataFrame(rows), [0.1, 0.5, 1.0])
    assert out["tau"][0][1] == pytest.approx(1.0)
    assert out["tau_to_final"] == pytest.approx([-1.0, -1.0, 1.0])
    assert out["mean_pairwise_tau"] == pytest.approx(-1 / 3, abs=1e-4)
    assert out["min_pairwise_tau"] == {"tau": -1.0, "fractions": [0.1, 1.0]}
