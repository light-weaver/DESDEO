"""Indicators and reference fronts on small, hand-checkable sets."""

import numpy as np
import pytest

from xover import fronts
from xover.indicators import LOWER_IS_BETTER, PARETO_COMPLIANT, RANKABLE, hv_reference_component, score


@pytest.mark.parametrize(("m", "n", "r"), [(3, 91, 1 + 1 / 12), (5, 210, 1 + 1 / 6), (8, 120, 1 + 1 / 3)])
def test_hv_reference_component(m, n, r):
    assert hv_reference_component(m, n) == pytest.approx(r)


def _front() -> fronts.ReferenceFront:
    t = np.linspace(0, 1, 50)
    points = np.column_stack([t, 1 - t, np.zeros_like(t)])
    return fronts.freeze({"line": points}, cap=20, hv_reference=hv_reference_component(3, 91), exact_hv_max_m=5)


def test_front_keeps_extremes_and_cap():
    front = _front()
    assert len(front.points) == 20
    assert front.pool_size == 50
    assert {tuple(p) for p in front.points} >= {(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)}
    assert front.hypervolume > 0


def test_a_set_on_the_front_scores_perfectly_and_a_worse_set_does_not():
    front = _front()
    on = score(front.points, front, m=3, population=91, reference_hv=front.hypervolume)
    off = score(front.points + 0.2, front, m=3, population=91, reference_hv=front.hypervolume)
    assert on["igd_plus"] == pytest.approx(0.0)
    assert on["hv_relative"] == pytest.approx(0.0)
    assert off["igd_plus"] > on["igd_plus"]
    assert off["hv_relative"] > on["hv_relative"]
    assert off["r2"] < on["r2"]  # R2 is the one higher-is-better indicator


def test_orientation_covers_every_rankable_indicator():
    assert set(RANKABLE) <= set(LOWER_IS_BETTER)
    assert set(PARETO_COMPLIANT) <= set(RANKABLE)
