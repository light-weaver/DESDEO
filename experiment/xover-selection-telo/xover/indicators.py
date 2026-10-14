"""Phase 4: indicators, every one measured against an instance's frozen reference front.

The indicator set is chosen so the indicators have *different* biases, which is what makes agreement
between them informative: convergence (IGD+, IGD, additive epsilon), distance (GD, GD_p, GD+,
averaged Hausdorff, Sayin's coverage error), hypervolume, decomposition (R2) and distribution
(spacing, spread, uniformity).

Both a run's points and the front are normalised into the front's unit box, by the ideal and nadir
of the merged archive pool the front was built from. **The hypervolume reference is derived, not
1.1:** `r = 1 + 1/H`, with `H` recovered from the population size (Ishibuchi, Imada, Setoguchi and
Nojima, DOI 10.1162/evco_a_00226).
"""

from __future__ import annotations

import math
from functools import lru_cache

import moocore
import numpy as np

from xover.fronts import ReferenceFront

EXACT_HV_MAX_M = 5
"""Above five objectives exact hypervolume costs 141 ms against IGD+'s 0.09 ms, so `hv_approx` takes
over. The workflow passes `config["indicators"]["exact_hv_max_m"]`."""

R2_WEIGHTS = 100
"""s-energy weight vectors for R2. The workflow passes `config["indicators"]["r2_weights"]`."""


def hv_reference_component(m: int, n: int) -> float:
    """Return the hypervolume reference component `r` for `m` objectives at population size `n`.

    Ishibuchi, Imada, Setoguchi and Nojima give `r = 1 + 1/H`, with `H` the largest simplex divisor
    whose Das-Dennis point count does not exceed the population size. GECCO 2017 precursor DOI
    10.1145/3071178.3071264; journal version Evolutionary Computation 26(3):411-440, 2018, DOI
    10.1162/evco_a_00226.

    A flat 1.1 corresponds to H = 10, which is right for no `m` in this study.
    """
    h = 1
    while math.comb(h + 1 + m - 1, m - 1) <= n:
        h += 1
    return 1.0 + 1.0 / h


def score(
    points: np.ndarray,
    front: ReferenceFront,
    *,
    m: int,
    population: int,
    reference_hv: float | None = None,
    exact_hv_max_m: int = EXACT_HV_MAX_M,
    r2_weights: int = R2_WEIGHTS,
) -> dict[str, float]:
    """Every indicator for one snapshot, on values normalised to the unit box by `R_p`'s box.

    `reference_hv` is `HV(R)` under the same fixed reference point, which depends only on the front,
    `m` and `population` and is therefore constant for a whole instance. The frozen front carries it;
    left at `None` it is computed here.
    """
    if len(points) == 0:
        return {}

    approx = front.normalise(points)
    reference = front.normalise(front.points)
    finite = np.isfinite(approx).all(axis=1)
    approx = approx[finite]
    if len(approx) == 0:
        return {"n_nondom": 0}

    out: dict[str, float] = {}

    out["igd_plus"] = float(moocore.igd_plus(approx, ref=reference))
    out["igd"] = float(moocore.igd(approx, ref=reference))
    out["epsilon_additive"] = float(moocore.epsilon_additive(approx, ref=reference))

    distances = _distance_indicators(approx, reference)
    out |= distances

    # One hypervolume evaluation per row, against a reference point no approximation set can move.
    # `r` comes from the objective count and the *population size*, not from any front's values
    # (Ishibuchi et al.'s r = 1 + 1/H), and both `approx` and `reference` were normalised into the
    # frozen front's box above. So HV(A) and HV(R) share the fixed reference `full(m, r)`.
    #
    # An earlier `hv_attained` used `r * approx.max(axis=0)` instead -- a reference point derived from
    # the set being measured, which is neither Pareto-compliant nor comparable between sets. A worse,
    # more diffuse front pushed its own reference outward and collected a larger volume, and since
    # larger read as better it ranked the arms roughly in reverse: Spearman +0.50 against igd_plus,
    # +0.51 against frac_outside_front_ref. `hv_relative` saturates at 1.0 on the hard instances
    # instead, which is well defined.
    r = hv_reference_component(m, population)
    out["hv_front"] = hypervolume(approx, np.full(m, r), exact_hv_max_m)
    out["hv_reference_r"] = float(r)

    # Relative hypervolume, 1 - HV(X)/HV(R): the hypervolume relative deviation of Bezerra, Lopez-Ibanez
    # and Stuetzle, Eq. 2 of *Evolutionary Computation* 26(4):621-656 (2018). Their 2017 EMO paper,
    # which this comment used to cite, is about inverted generational distance instead.
    #
    # This is the study's one indicator whose magnitude is comparable *across*
    # instances without a per-arm normalisation, because it is a ratio against the instance's own
    # reference front rather than against the spread of whichever arms happen to be in the grid.
    #
    # Written out rather than called as `moocore.RelativeHypervolume`, which computes exactly this and
    # agrees with it to the last bit -- checked at m=3 and m=5. The class always calls the exact
    # `moocore.hypervolume`, which is intractable above m=5 on a four-thousand-point front;
    # `hypervolume` switches to `hv_approx` there.
    if reference_hv is None:
        reference_hv = hypervolume(reference, np.full(m, r), exact_hv_max_m)
    out["hv_relative"] = (reference_hv - out["hv_front"]) / reference_hv if reference_hv > 0 else float("nan")

    out["r2"] = _r2(approx, m, r2_weights)
    out |= _distribution(approx)

    out["frac_outside_front_ref"] = float(np.mean((approx > r).any(axis=1)))
    out["n_nondom"] = len(moocore.filter_dominated(approx))
    out["n_points"] = len(approx)
    return out


def _distance_indicators(approx: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """GD, the Hausdorff view, GD+, and Sayin's coverage error.

    `distance_indicators` supplies `igd`, `igd_p`, `gd`, `gd_p` and `ahd`; the AHD it returns is the
    symmetric Hausdorff view, `max(IGD_p, GD_p)`, which penalises both missing regions and stray
    points. It has no GD+, so that one is computed here from its definition.

    `distance_indicators(F, R, p=inf).igd_p` reproduces Sayin's coverage error exactly -- the
    *maximum* distance from any reference point to its nearest solution, so a front with one large
    hole is penalised even when its mean is good. Only the right `p` is needed, not new code.
    """
    from desdeo.tools.indicators_unary import distance_indicators

    out: dict[str, float] = {}
    try:
        base = distance_indicators(approx, reference)
        out["gd"] = float(base.gd)
        out["gd_p"] = float(base.gd_p)
        out["ahd"] = float(base.ahd)
    except Exception:
        out |= dict.fromkeys(("gd", "gd_p", "ahd"), float("nan"))
    out["gd_plus"] = _gd_plus(approx, reference)
    try:
        out["coverage_error"] = float(distance_indicators(approx, reference, p=np.inf).igd_p)
    except Exception:
        out["coverage_error"] = float("nan")
    return out


def _gd_plus(approx: np.ndarray, reference: np.ndarray, chunk: int = 512) -> float:
    """GD+, the dominance-aware convergence measure.

    Ishibuchi, Masuda, Tanigaki & Nojima (2015): the distance from a solution to a reference point
    counts only the components where the solution is *worse*, so a solution that dominates its
    nearest reference point is not penalised for it. Paired with IGD+, the difference between the
    plus and non-plus forms is what shows where dominance-awareness matters.
    """
    if len(approx) == 0 or len(reference) == 0:
        return float("nan")
    total = 0.0
    for start in range(0, len(approx), chunk):
        block = approx[start : start + chunk]
        excess = np.maximum(block[:, None, :] - reference[None, :, :], 0.0)
        total += np.sqrt(np.einsum("ijk,ijk->ij", excess, excess)).min(axis=1).sum()
    return float(total / len(approx))


def hypervolume(points: np.ndarray, reference: np.ndarray, exact_hv_max_m: int = EXACT_HV_MAX_M) -> float:
    """Exact hypervolume up to `exact_hv_max_m` objectives, `moocore.hv_approx` above."""
    inside = points[(points < reference).all(axis=1)]
    if len(inside) == 0:
        return 0.0
    try:
        if len(reference) <= exact_hv_max_m:
            return float(moocore.hypervolume(inside, ref=reference))
        return float(moocore.hv_approx(inside, ref=reference))
    except Exception:
        return float("nan")


def _r2(approx: np.ndarray, m: int, n_weights: int) -> float:
    """The decomposition view: the only indicator aligned with how RVEA and NSGA-III select.

    The weight set is the same s-energy construction the reference-vector algorithms use, which is
    what makes this the decomposition view rather than a second distance measure. `moocore.r2_exact`
    is not usable here -- it supports two objectives only, and every instance in this study has at
    least three.

    The ideal is the origin, because `approx` is already normalised into `R_p`'s unit box.
    """
    try:
        from desdeo.tools.indicators_unary import r2_indicator

        weights = _weight_vectors(m, n_weights)
        result = r2_indicator(approx, weights, np.zeros(m))
        return float(result.r2_value)
    except Exception:
        return float("nan")


@lru_cache(maxsize=16)
def _weight_vectors(m: int, n: int) -> np.ndarray:
    """Cached s-energy weight vectors for R2.

    About 1 s per construction, against thousands of indicator evaluations per job.
    """
    from desdeo.tools.reference_vectors import create_s_energy

    return create_s_energy(number_of_objectives=m, number_of_vectors=n, seed=0)


def _distribution(approx: np.ndarray, cap: int = 2000) -> dict[str, float]:
    """Spacing, spread and Sayin's uniformity level: distribution only, indifferent to convergence."""
    if len(approx) < 2:  # noqa: PLR2004
        return {"spacing": float("nan"), "spread": float("nan"), "uniformity_level": float("nan")}
    sample = approx[:cap]
    diff = sample[:, None, :] - sample[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return {
        "spacing": float(nearest.std()),
        "spread": float(nearest.mean()),
        "uniformity_level": float(nearest.min()),
    }


def score_run(
    snapshots: list[np.ndarray],
    evaluations: np.ndarray,
    front: ReferenceFront,
    *,
    fractions: list[float],
    m: int,
    population: int,
    exact_hv_max_m: int = EXACT_HV_MAX_M,
    r2_weights: int = R2_WEIGHTS,
) -> list[dict[str, float]]:
    """One row of indicators per budget fraction for one run.

    Comparisons are at budget *fractions* rather than evaluation counts, because the terminator
    overshoots by a different amount per algorithm and no two arms stop at the same count. The
    budget is the run's own last snapshot, so 1.0 is each run's endpoint, and each fraction takes the
    snapshot nearest to it (the first, on a tie).
    """
    budget = int(evaluations[-1])
    rows = []
    for fraction in fractions:
        index = int(np.argmin(np.abs(evaluations - fraction * budget)))
        values = score(
            snapshots[index],
            front,
            m=m,
            population=population,
            reference_hv=front.hypervolume,
            exact_hv_max_m=exact_hv_max_m,
            r2_weights=r2_weights,
        )
        rows.append({"fraction": fraction, "evaluations": int(evaluations[index]), **values})
    return rows


LOWER_IS_BETTER: dict[str, bool] = {
    "igd_plus": True,
    "igd": True,
    "epsilon_additive": True,
    "gd": True,
    "gd_p": True,
    "ahd": True,
    "gd_plus": True,
    "coverage_error": True,
    "hv_relative": True,
    "hv_front": False,
    # The one indicator here where higher is better, and it is always negative. DESDEO reports the
    # *utility* form of unary R2 (Brockhoff, Wagner & Trautmann 2012): the mean over weight vectors
    # of the best utility any solution achieves, the utility being a negated Tchebycheff distance.
    # PlatEMO and jMetal report the distance form, which is non-negative and minimised. Checked
    # rather than assumed -- DESDEO's value is the exact negation of a hand-written distance-form R2,
    # to the last bit -- so this is a convention, not a defect.
    #
    # It was written as True here first, and RQ10 caught it within one run: r2 ranked the arms almost
    # exactly opposite to the other twelve indicators (Kendall tau -0.855 against igd_plus), putting
    # `local` first and `sbx` last. That is what a wrong orientation looks like, and it is why
    # anything that ranks or integrates refuses an indicator missing from this map.
    "r2": False,
}
"""Orientation, for the indicators that have an unambiguous one.

`spacing`, `spread` and `uniformity_level` are deliberately absent. They measure how evenly a set is
distributed and say nothing about whether it is in the right place, so "better" depends on what is
being asked; a degenerate front can score perfectly on all three. They stay descriptive, and anything
that needs a direction -- ranking, an integral -- refuses to take one silently."""

COMPARABLE_ACROSS_INSTANCES: tuple[str, ...] = ("hv_relative",)
"""Indicators whose magnitude means the same thing on every instance.

`hv_relative` is `1 - HV(A)/HV(R)`: a ratio against the instance's own reference front, so 0.1 is the
same shortfall on DTLZ2 at m=3 as on GAA at m=11. Everything else is computed in the reference
front's unit box, which removes the objectives' scales but not the front's geometry -- an IGD+ of
0.05 is not the same achievement on a concave front as on a disconnected one.

This matters for anything that aggregates *across* instances without first normalising within them.
The per-instance min-max regret in `add_regret` handles that by normalising, but its scale then
depends on which arms are in the grid, since the maximum is the worst arm's value. That is fine for
rank-based tests and wrong for an integral, which is why `anytime.aocc` defaults here instead."""

RANKABLE: tuple[str, ...] = (
    "igd_plus",
    "igd",
    "epsilon_additive",
    "gd",
    "gd_p",
    "ahd",
    "gd_plus",
    "coverage_error",
    "hv_relative",
    "r2",
)
"""The columns an analysis may rank, aggregate or integrate. Everything else in the table is a
diagnostic, and the analyses read this rather than each maintaining an exclusion list.

Having an orientation in `LOWER_IS_BETTER` is necessary and not sufficient, which is the lesson of
bug-045. `hv_attained` had one, so every "does it have a direction?" gate passed it through, and
RQ10 then ranked thirteen indicators of which one measured a set against a reference point derived
from that same set. It alone produced a minimum pairwise Kendall tau of -0.513; without it the
remaining 66 pairs agree unanimously on the winner at mean tau 0.80.

`hv_front` is deliberately absent despite having an orientation: it is `hv_relative`'s numerator, so
including both would weight hypervolume twice in any multi-indicator aggregate. `spacing`, `spread`
and `uniformity_level` are absent because they describe how evenly a set is distributed and say
nothing about where it is. `hv_reference_r`, `frac_outside_front_ref`, `n_points` and `n_nondom` are
run metadata."""

PARETO_COMPLIANT: tuple[str, ...] = ("igd_plus", "epsilon_additive", "hv_relative", "r2")
"""The rankable indicators that are at least weakly Pareto compliant, which the paper leads with.

A set that dominates another never scores worse on these. `hv_relative` inherits it from hypervolume,
because HV(R) and the reference point are fixed per instance. IGD+ and the additive epsilon are
weakly compliant (Ishibuchi, Masuda, Tanigaki & Nojima, EMO 2015, doi:10.1007/978-3-319-15892-1_8).
R2 is weakly compliant when its reference point is utopian (Brockhoff, Wagner & Trautmann, GECCO
2012); here that point is the reference pool's ideal, which holds unless a run beats the pool on an
objective.

IGD and GD+ are not in the set. The same EMO 2015 paper gives a counterexample for each -- "the GD+
indicator is not Pareto compliant" -- and an earlier plan counted both among "the compliant six". GD,
GD_p, AHD and coverage error are not compliant either."""
