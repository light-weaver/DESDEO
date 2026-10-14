"""The problem registry and the problems it builds."""

import numpy as np
import pytest

from xover import design, problems, rotated_dtlz


def test_slugs_are_stable(config):
    """A handful of slugs from each suite, as the EMO-paper study named them."""
    slugs = set(design.all_instances(config))
    assert {
        "dtlz2_m3_n12",
        "wfg9_m8_n24",
        "re61_m6",
        "river_pollution_m4_f_1-f_2-f_3-f_4",
        "gaa_m11",
        "metallappl_mop2_m5",
        "zcat14_l3_ps1_m3_n30",
        "r2_dtlz3_unrotated_m5_n30",
    } <= slugs


def test_subset_sampling_is_deterministic():
    symbols = tuple(f"f_{i}" for i in range(1, 7))
    first = problems.objective_subsets(symbols, cap=6, seed=7, min_objectives=3)
    again = problems.objective_subsets(symbols, cap=6, seed=7, min_objectives=3)
    assert first == again
    assert first[0] == symbols
    assert len(first) == 6


def test_required_subsets_are_kept_in_declared_order():
    symbols = ("f_1", "f_2", "f_3", "f_4", "f_5")
    chosen = problems.objective_subsets(symbols, cap=2, seed=0, min_objectives=3, required=(("f_4", "f_1", "f_3"),))
    assert chosen == [symbols, ("f_1", "f_3", "f_4")]


@pytest.mark.parametrize("slug", ["dtlz2_m3_n12", "re33_m3", "zcat1_l2_ps1_m3_n30", "r2_dtlz2_m5_n30"])
def test_problems_build_and_evaluate(config, slug):
    from xover.landscape import sample

    instance = design.instance(config, slug)
    problem = problems.build_problem(instance.spec)
    assert len(problem.objectives) == instance.m
    assert len(problem.get_flattened_variables()) == instance.n_var
    evaluated = sample(problem, 20, seed=0)
    assert evaluated["Y"].shape == (20, instance.m)
    assert np.isfinite(evaluated["Y"]).all()


def test_uf11_keeps_the_published_matrix_with_its_defect():
    """The report calls M orthogonal. UF12's is; UF11's published one is not, and it stays as published."""
    uf11 = np.asarray(rotated_dtlz.PUBLISHED["dtlz2"]["matrix"])
    uf12 = np.asarray(rotated_dtlz.PUBLISHED["dtlz3"]["matrix"])
    assert np.allclose(uf12 @ uf12.T, np.eye(rotated_dtlz.N_VAR), atol=1e-3)
    assert np.linalg.norm(uf11[3]) == pytest.approx(0.791, abs=1e-3)
    assert np.array_equal(uf11[3, 9:15], uf11[4, 9:15])
    assert np.linalg.cond(uf11) == pytest.approx(5.19, abs=0.01)
