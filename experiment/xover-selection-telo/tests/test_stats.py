"""The statistics the analyses rest on, on small grids with known answers."""

import numpy as np
import polars as pl
import pytest

from xover import stats


@pytest.fixture
def grid() -> pl.DataFrame:
    """Twelve instances x four arms x two algorithms, with arm `a` best and a little noise."""
    rng = np.random.default_rng(3)
    rows = []
    for instance in range(12):
        for algorithm in ("p", "q"):
            for arm, offset in (("a", 0.0), ("b", 0.2), ("c", 0.4), ("d", 0.6)):
                rows.append(
                    {
                        "instance": f"i{instance}",
                        "algorithm": algorithm,
                        "crossover": arm,
                        "regret": offset + rng.normal(0, 0.15),
                    }
                )
    return pl.DataFrame(rows)


def test_within_block_ranks_reproduce_the_friedman_mean_ranks(grid):
    ranks = stats.within_block_ranks(grid, "regret", "crossover")
    friedman = stats.friedman(grid, "regret", "crossover")["mean_ranks"]
    for arm, rank in ranks.items():
        assert rank == pytest.approx(friedman[arm], abs=1e-4)


def test_crossover_blocks_rank_each_algorithm_separately_and_agree_with_friedman(grid):
    blocks = stats.CROSSOVER_BLOCKS
    ranks = stats.within_block_ranks(grid, "regret", "crossover", blocks=blocks)
    result = stats.friedman(grid, "regret", "crossover", block=blocks)
    assert result["n_blocks"] == grid["instance"].n_unique() * grid["algorithm"].n_unique()
    assert result["treatments"] == sorted(result["treatments"])
    for arm, rank in ranks.items():
        assert rank == pytest.approx(result["mean_ranks"][arm], abs=1e-4)
    pairs = stats.pairwise_wilcoxon(grid, "regret", "crossover", block=blocks)
    assert {row["n"] for row in pairs} == {result["n_blocks"]}


def test_within_block_ranks_are_ordered_best_first(grid):
    ranks = stats.within_block_ranks(grid, "regret", "crossover")
    values = list(ranks.values())
    assert values == sorted(values)
    assert next(iter(ranks)) == "a"


def test_holm_correction_steps_down_and_keeps_the_input_order():
    assert stats.holm_correct([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_critical_difference_matches_demsar_at_four_treatments():
    # Demsar 2006, section 3.2.2: k = 4, N = 14 gives CD = 2.569 sqrt(20 / 84) = 1.25.
    assert stats.critical_difference(4, 14) == pytest.approx(1.2536, abs=1e-3)
    assert np.isnan(stats.critical_difference(1, 14))


@pytest.mark.parametrize(("k", "q"), [(2, 1.960), (5, 2.728), (10, 3.164), (14, 3.354), (15, 3.391)])
def test_critical_difference_reproduces_the_tabulated_q_values(k, q):
    # q_0.05 from Demsar 2006 (k <= 10) and the tables that extend it, recovered from CD with N = 1.
    assert stats.critical_difference(k, 1) / np.sqrt(k * (k + 1) / 6) == pytest.approx(q, abs=5e-4)


def test_the_single_best_level_sits_between_no_knowledge_and_per_instance_knowledge(grid):
    gain = stats.oracle_gain(grid, "regret", ["crossover", "algorithm"])
    for factor in ("crossover", "algorithm"):
        assert gain[f"know_{factor}"] <= gain[f"best_fixed_{factor}"] <= gain["no_knowledge"]
    assert stats.best_fixed_levels(grid, "regret", ["crossover"]) == {"crossover": "a"}


def _factorial_frame(effect_sizes: tuple[float, float, float] = (1.0, 3.0, 0.5)) -> pl.DataFrame:
    """Return a balanced instance x crossover x algorithm design with known, unequal effects."""
    rng = np.random.default_rng(0)
    crossover_effect, algorithm_effect, instance_effect = effect_sizes
    return pl.DataFrame(
        [
            {
                "instance": f"i{instance}",
                "crossover": f"x{crossover}",
                "algorithm": f"a{algorithm}",
                "regret": crossover_effect * crossover
                + algorithm_effect * algorithm
                + instance_effect * instance
                + rng.normal(0, 0.3),
            }
            for instance in range(8)
            for crossover in range(5)
            for algorithm in range(3)
        ]
    )


def test_omega_squared_orders_the_factors_by_their_true_effects():
    effects = {e.term: e for e in stats.omega_squared(_factorial_frame(), "regret", ["crossover", "algorithm"])}
    assert effects["algorithm"].omega_squared > effects["crossover"].omega_squared > 0
    assert effects["crossover:algorithm"].omega_squared < 0.01
    assert effects["crossover"].df == 4


def test_generalized_omega_squared_reduces_to_the_classical_share_with_no_manipulated_factor():
    frame = _factorial_frame()
    factors = ["crossover", "algorithm", "instance"]
    classical = {e.term: e.omega_squared for e in stats.omega_squared(frame, "regret", factors)}
    generalized = {
        e.term: e.omega_squared
        for e in stats.generalized_omega_squared(frame, "regret", manipulated=[], measured=factors)
    }
    assert generalized.keys() == classical.keys()
    for term, value in classical.items():
        assert generalized[term] == pytest.approx(value, rel=1e-9)


def test_generalized_omega_squared_drops_the_other_manipulated_factor_from_the_denominator():
    frame = _factorial_frame(effect_sizes=(1.0, 6.0, 0.5))
    classical = {e.term: e.omega_squared for e in stats.omega_squared(frame, "regret", ["crossover", "algorithm"])}
    generalized = {
        e.term: e.omega_squared
        for e in stats.generalized_omega_squared(
            frame, "regret", manipulated=["crossover", "algorithm"], measured=["instance"]
        )
    }
    assert generalized["crossover"] > classical["crossover"]
    assert generalized["algorithm"] > classical["algorithm"]
    assert 0.0 < generalized["crossover"] < 1.0


def test_the_block_rank_matrix_averages_to_the_within_block_ranks(grid):
    keys, arms, ranks = stats.block_rank_matrix(grid, "regret")
    expected = stats.within_block_ranks(grid, "regret", blocks=stats.CROSSOVER_BLOCKS)
    assert keys.columns == list(stats.CROSSOVER_BLOCKS)
    assert ranks.shape == (grid.select(stats.CROSSOVER_BLOCKS).unique().height, len(arms))
    for arm, mean in zip(arms, ranks.mean(axis=0), strict=True):
        assert mean == pytest.approx(expected[arm])


def test_ordering_tau_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    ranks = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    same = stats.ordering_tau(ranks, {k: v * 2 for k, v in ranks.items()})
    reverse = stats.ordering_tau(ranks, {k: -v for k, v in ranks.items()})
    assert (same["kendall_tau"], same["same_best"]) == (pytest.approx(1.0), True)
    assert (reverse["kendall_tau"], reverse["same_best"]) == (pytest.approx(-1.0), False)


def test_the_split_null_moves_whole_groups_and_reproduces_from_its_seed(grid):
    keys, _, ranks = stats.block_rank_matrix(grid, "regret")
    groups = keys["instance"].to_numpy()
    first = stats.split_tau_null(ranks, groups, (6, 6), n_permutations=50, seed=[1, 2])
    again = stats.split_tau_null(ranks, groups, (6, 6), n_permutations=50, seed=[1, 2])
    np.testing.assert_array_equal(first, again)
    assert np.median(first) > 0.5
    with pytest.raises(ValueError, match="cannot be split"):
        stats.split_tau_null(ranks, groups, (6, 5))


def test_the_performance_profile_counts_blocks_within_each_factor():
    ratios = np.log10([1.0, 1.05, 1.5, 10.0])
    np.testing.assert_allclose(
        stats.performance_profile(ratios, np.array([1.0, 1.1, 2.0, 10.0])), [0.25, 0.5, 0.75, 1.0]
    )
