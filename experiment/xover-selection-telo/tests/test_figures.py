"""The logic behind what the figures claim: which arms a CD bar joins, and when a ranking settles."""

from analysis.figures import clique_rows, cliques, settled_at


def test_cliques_join_arms_closer_than_the_critical_difference():
    assert cliques([1.0, 1.5, 2.2, 4.0], 1.0) == [(0, 1), (1, 2)]


def test_a_clique_inside_a_longer_one_is_dropped():
    assert cliques([1.0, 1.1, 1.2], 1.0) == [(0, 2)]


def test_an_arm_with_no_neighbour_within_the_critical_difference_has_no_bar():
    assert cliques([1.0, 3.0, 5.0], 1.0) == []


def test_overlapping_cliques_get_separate_rows():
    assert clique_rows([1.0, 1.5, 2.2, 4.0], 1.0) == [0, 1]


def test_settled_is_the_first_fraction_from_which_tau_stays_above_the_threshold():
    fractions = [0.1, 0.5, 0.9, 1.0]
    assert settled_at(fractions, [0.96, 0.9, 0.96, 1.0]) == 0.9
    assert settled_at(fractions, [0.96, 0.96, 0.96, 1.0]) == 0.1
    assert settled_at(fractions, [0.5, 0.6, 0.7, 0.8]) is None
