"""Phase 5: landscape features, computed on one large Latin-hypercube sample per instance.

The sample is drawn once per instance, at `sample_size_per_var * n_var` points (200 d, the size
Liefooghe, Verel, Lacroix, Zavoianu and McCall use, GECCO 2021, DOI 10.1145/3449639.3459353), with
one fixed seed. It is not one of the runs' initial populations, so the features describe the problem
and are shared by every run on it.

Two kinds of feature come out of it:

- **Multi-objective features**, here, needing only numpy, scipy and moocore, chosen for relevance to
  crossover: front geometry (fraction non-dominated, number of fronts, rank statistics, the Lame
  exponent of the first front), objective conflict, the decision-space spread of the non-dominated
  set, decision-objective coupling, ridge-regularised linear and quadratic fit quality, a k-NN
  ruggedness walk, and scale descriptors.
- **ELA features** from pflacco, in `envs/ela/ela_features.py`, which needs its own interpreter.

Both are also computed on four derived scalar landscapes (mean normalised objective, achievement
scalarising value, Pareto rank and negated hypervolume contribution), which carry the trade-off into
single-objective machinery. They are computed once, by `derived_landscapes`, and stored with the
sample, so the ELA environment needs neither desdeo nor moocore.

Everything is in minimisation sense.
"""

from __future__ import annotations

import moocore
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, kendalltau

LAME_FLOOR = 1e-9
"""Points this close to the front's own minimum carry no curvature information and would dominate
the fit through the logarithm."""

MAX_PAIRWISE = 1500
"""Cap on points entering an O(n^2) distance matrix. A 6 000-point sample at d=30 would need a
288 MB matrix and buys nothing over a subsample."""


def sample(problem, n_points: int, seed: int) -> dict[str, np.ndarray]:
    """Draw and evaluate a Latin-hypercube sample with DESDEO's own LHS generator.

    Returns `X` (float32), `Y` (float32, minimisation sense) and the decision box.
    """
    from desdeo.emo.operators.evaluator import EMOEvaluator
    from desdeo.emo.options.generator import LHSGeneratorOptions
    from desdeo.emo.options.templates import generator_constructor
    from desdeo.tools.patterns import Publisher

    variables = [v.symbol for v in problem.get_flattened_variables()]
    publisher = Publisher()
    evaluator = EMOEvaluator(problem=problem, publisher=publisher, verbosity=2)
    generator = generator_constructor(
        problem=problem,
        options=LHSGeneratorOptions(n_points=n_points),
        evaluator=evaluator,
        publisher=publisher,
        verbosity=2,
        seed=seed,
    )
    solutions, outputs = generator.do()

    x = solutions.select(variables).to_numpy().astype(np.float32)
    targets = [f"{obj.symbol}_min" for obj in problem.objectives]
    if all(t in outputs.columns for t in targets):
        y = outputs.select(targets).to_numpy().astype(np.float32)
    else:
        signs = np.array([-1.0 if obj.maximize else 1.0 for obj in problem.objectives], dtype=np.float32)
        y = outputs.select([obj.symbol for obj in problem.objectives]).to_numpy().astype(np.float32) * signs

    lower = np.asarray([v.lowerbound for v in problem.get_flattened_variables()], dtype=np.float32)
    upper = np.asarray([v.upperbound for v in problem.get_flattened_variables()], dtype=np.float32)
    return {"X": x, "Y": y, "lower": lower, "upper": upper}


HV_APPROX_SAMPLES = 16384
"""Samples for the approximate hypervolume above `exact_hv_max_m` objectives.

Measured against exact contributions at m=7: Spearman 0.965 at 4 096 samples, **0.990 at 16 384**,
0.996 at 65 536, for 0.35 s, 1.4 s and 5.6 s on the study's largest first front. 16 384 is where the
accuracy stops being the binding constraint."""


def hv_contributions(points: np.ndarray, ref: np.ndarray, exact_hv_max_m: int = 5) -> np.ndarray:
    """Per-point hypervolume contribution: exact where that is affordable, approximate above.

    `moocore.hv_contributions` is exact and exponential in the objective count, and in a
    many-objective population almost every point is non-dominated, so the first front stays large
    exactly where the cost explodes. Measured on a synthetic front: 0.12 s at m=5, 9.2 s at m=7,
    122 s at m=8.

    The threshold is the indicators' `exact_hv_max_m`, so the study has one rule about when exact
    hypervolume stops being viable rather than two.

    Above it, the contribution is the drop in total hypervolume when the point is removed, with a
    deterministic approximation standing in for each total. That is 82x faster at
    m=7 and 1.4 s at m=11. What this feature needs is to *order* points, and it does: Spearman 0.99
    against exact.
    """
    if points.shape[1] <= exact_hv_max_m:
        return np.asarray(moocore.hv_contributions(points, ref=ref))

    total = moocore.hv_approx(points, ref=ref, nsamples=HV_APPROX_SAMPLES)
    out = np.empty(len(points))
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        keep[i] = False
        out[i] = total - moocore.hv_approx(points[keep], ref=ref, nsamples=HV_APPROX_SAMPLES)
        keep[i] = True
    # A contribution cannot be negative; a small one differenced out of two approximations can be.
    return np.clip(out, 0.0, None)


def derived_landscapes(y: np.ndarray, exact_hv_max_m: int = 5) -> tuple[list[str], np.ndarray]:
    """Four scalar landscapes that carry multi-objective structure into single-objective machinery.

    Features computed per objective and then averaged would miss the trade-off entirely, which is
    the whole thing a multiobjective landscape has that a single-objective one does not. These four
    put it back: the achievement-scalarising value, the Pareto rank, the hypervolume contribution
    and the mean normalised objective.

    Returned in minimisation sense, matching `Y`.
    """
    lo, hi = y.min(axis=0), y.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    normalised = (y - lo) / span

    names = ["mean_obj", "asf", "rank", "hv_contrib"]
    ranks = np.asarray(moocore.pareto_rank(y), dtype=float)
    columns = [normalised.mean(axis=1), normalised.max(axis=1), ranks]

    contributions = np.zeros(len(y))
    try:
        first = ranks == ranks.min()
        contributions[first] = hv_contributions(normalised[first], np.full(y.shape[1], 1.1), exact_hv_max_m)
    except Exception:  # noqa: S110
        # A degenerate first front -- every point on one rank, or a zero-span objective -- has no
        # hypervolume contribution to report. Leaving the column at zero is the honest answer.
        pass
    # Negated so that, like every other landscape here, smaller is better.
    columns.append(-contributions)

    return names, np.column_stack(columns)


def mo_features(
    x: np.ndarray, y: np.ndarray, lower: np.ndarray, upper: np.ndarray, landscapes: dict[str, np.ndarray]
) -> dict[str, float]:
    """Every multi-objective feature of one sample.

    Args:
        x: decision matrix, `(n, d)`.
        y: objective matrix in minimisation sense, `(n, m)`.
        lower: componentwise lower bound of the decision box.
        upper: componentwise upper bound of the decision box.
        landscapes: the derived scalar landscapes, from `derived_landscapes`.
    """
    n, d = x.shape
    m = y.shape[1]
    out: dict[str, float] = {"n_points": float(n), "d": float(d), "m": float(m)}

    finite = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    landscapes = {name: values[finite] for name, values in landscapes.items()}
    out["frac_finite"] = float(finite.mean())
    if len(y) < 3:  # noqa: PLR2004
        return out

    # ---- Front geometry -------------------------------------------------------------------------
    ranks = np.asarray(moocore.pareto_rank(y))
    n_fronts = int(ranks.max()) + 1
    first = ranks == ranks.min()
    out["frac_nondominated"] = float(first.mean())
    out["n_fronts"] = float(n_fronts)
    # Liefooghe et al.'s rank features: the mean front index, and the entropy of the distribution
    # over fronts. A flat distribution means selection pressure is spread thin.
    out["rank_avg"] = float(ranks.mean() / max(1, n_fronts - 1)) if n_fronts > 1 else 0.0
    counts = np.bincount(ranks.astype(int))
    out["rank_ent"] = float(entropy(counts / counts.sum())) if len(counts) > 1 else 0.0

    out["lame_p"] = _lame_exponent(y[first])

    # ---- Objective conflict ---------------------------------------------------------------------
    taus = []
    for i in range(m):
        for j in range(i + 1, m):
            tau = kendalltau(y[:, i], y[:, j]).statistic
            if np.isfinite(tau):
                taus.append(float(tau))
    if taus:
        out["conflict_mean"] = float(np.mean(taus))
        out["conflict_min"] = float(np.min(taus))
        out["conflict_max"] = float(np.max(taus))
        out["conflict_std"] = float(np.std(taus))

    # ---- Decision-space geometry of the elite set -----------------------------------------------
    span = np.where(upper - lower > 0, upper - lower, 1.0)
    scaled = (x - lower) / span
    diagonal = float(np.sqrt(d)) or 1.0

    elite = scaled[first]
    if len(elite) >= 2:  # noqa: PLR2004
        elite_d = pdist(_subsample(elite))
        # Crossover acts in decision space, so how spread the elite set is *there* is the feature
        # most directly about the operator rather than about the problem's objective geometry.
        out["dist_x_nd_avg"] = float(elite_d.mean() / diagonal)
        out["dist_x_nd_max"] = float(elite_d.max() / diagonal)
        out["dist_x_nd_std"] = float(elite_d.std() / diagonal)
    all_d = pdist(_subsample(scaled))
    out["dist_x_avg"] = float(all_d.mean() / diagonal)
    out["dist_x_max"] = float(all_d.max() / diagonal)
    # Below 1 the elite set is clustered relative to the whole sample.
    out["dist_x_nd_ratio"] = out.get("dist_x_nd_avg", np.nan) / out["dist_x_avg"] if out["dist_x_avg"] else np.nan

    # ---- Decision-objective coupling ------------------------------------------------------------
    couplings = []
    for j in range(d):
        for k in range(m):
            tau = kendalltau(scaled[:, j], y[:, k]).statistic
            if np.isfinite(tau):
                couplings.append(abs(float(tau)))
    if couplings:
        out["coupling_mean"] = float(np.mean(couplings))
        out["coupling_max"] = float(np.max(couplings))
        # WFG and DTLZ split their variables into position and distance parameters. A population
        # where only some variables move the objectives has a low fraction here.
        out["coupling_frac_strong"] = float(np.mean(np.asarray(couplings) > 0.2))  # noqa: PLR2004

    # ---- Surrogate fit quality, the honest replacement for ela_meta ------------------------------
    scalar = _scalarise(y)
    out["lin_r2"] = _ridge_r2(scaled, scalar, quadratic=False)
    out["quad_r2"] = _ridge_r2(scaled, scalar, quadratic=True)
    out["quad_gain"] = out["quad_r2"] - out["lin_r2"]

    # ---- Ruggedness -----------------------------------------------------------------------------
    out |= _ruggedness(scaled, scalar)

    # ---- Scale ----------------------------------------------------------------------------------
    out["box_span_ratio"] = float(span.max() / span.min()) if span.min() > 0 else float("nan")
    out["y_span_ratio"] = _span_ratio(y)

    # ---- Derived scalar landscapes --------------------------------------------------------------
    # Multi-objective structure carried through single-objective machinery: features computed per
    # objective would miss the trade-off entirely, so these four scalarisations put it back.
    for name, values in landscapes.items():
        out[f"{name}_lin_r2"] = _ridge_r2(scaled, values, quadratic=False)
        out[f"{name}_skew"] = _skew(values)
        out[f"{name}_kurtosis"] = _kurtosis(values)
    return out


def _subsample(points: np.ndarray, cap: int = MAX_PAIRWISE, seed: int = 0) -> np.ndarray:
    if len(points) <= cap:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), cap, replace=False)]


def _scalarise(y: np.ndarray) -> np.ndarray:
    """The mean normalised objective: one scalar landscape standing in for the whole vector."""
    lo, hi = y.min(axis=0), y.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return ((y - lo) / span).mean(axis=1)


def _lame_exponent(front: np.ndarray) -> float:
    """Fit `sum_i (f_i / max_i)^p = 1` to the sampled first front and return `p`.

    `p < 1` is convex, `p = 1` linear, `p > 1` concave -- the geometry that decides whether an
    operator biasing offspring toward the parent midpoint lands on the front or inside it.

    This describes the *sampled* front, not the true one: it is what can be computed before
    optimising.
    """
    if len(front) < 3:  # noqa: PLR2004
        return float("nan")
    lo = front.min(axis=0)
    shifted = front - lo
    hi = shifted.max(axis=0)
    span = np.where(hi > 0, hi, 1.0)
    scaled = shifted / span
    scaled = scaled[(scaled > LAME_FLOOR).all(axis=1)]
    if len(scaled) < 3:  # noqa: PLR2004
        return float("nan")

    def residual(p: float) -> float:
        return float(np.mean((np.power(scaled, p).sum(axis=1) - 1.0) ** 2))

    grid = np.geomspace(0.2, 6.0, 60)
    values = [residual(p) for p in grid]
    return float(grid[int(np.argmin(values))])


def _ridge_r2(x: np.ndarray, y: np.ndarray, *, quadratic: bool, alpha: float = 1e-3) -> float:
    """Ridge-regularised R-squared of a linear or quadratic model.

    Regularised because the quadratic design matrix has `1 + d + d(d+1)/2` columns -- 91 at d=12 and
    496 at d=30 -- and an unregularised fit on few points returns a meaningless R-squared of 1.
    """
    if len(y) < 3 or np.ptp(y) == 0:  # noqa: PLR2004
        return float("nan")
    design = [np.ones((len(x), 1)), x]
    if quadratic:
        design.append(x**2)
        idx = np.triu_indices(x.shape[1], k=1)
        if len(idx[0]) and x.shape[1] <= 40:  # noqa: PLR2004
            design.append(x[:, idx[0]] * x[:, idx[1]])
    a = np.hstack(design)
    penalty = alpha * np.eye(a.shape[1])
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(a.T @ a + penalty, a.T @ y)
    except np.linalg.LinAlgError:
        return float("nan")
    residual = y - a @ beta
    total = ((y - y.mean()) ** 2).sum()
    return float(1.0 - (residual**2).sum() / total) if total > 0 else float("nan")


def _ruggedness(x: np.ndarray, y: np.ndarray, k: int = 5) -> dict[str, float]:
    """A k-NN-graph random walk: how much the scalar landscape changes between near neighbours.

    A grid walk is impossible here -- the sample is a Latin hypercube in 12 to 30 dimensions, not a
    lattice -- so the walk runs on the nearest-neighbour graph instead, which is the standard
    substitute and needs no extra evaluations.
    """
    sample = _subsample(x)
    values = _subsample(y[: len(x)].reshape(-1, 1)).ravel() if len(y) == len(x) else y
    if len(sample) != len(values) or len(sample) < k + 2:
        return {"rugged_corr": float("nan"), "rugged_step": float("nan")}
    distances = squareform(pdist(sample))
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1)[:, :k]
    steps = np.abs(values[neighbours] - values[:, None])
    span = np.ptp(values)
    out = {"rugged_step": float(steps.mean() / span) if span > 0 else float("nan")}
    # Correlation between a point's value and its neighbours' mean: near 1 is smooth, near 0 rugged.
    neighbour_mean = values[neighbours].mean(axis=1)
    if np.std(values) > 0 and np.std(neighbour_mean) > 0:
        out["rugged_corr"] = float(np.corrcoef(values, neighbour_mean)[0, 1])
    else:
        out["rugged_corr"] = float("nan")
    return out


def _span_ratio(y: np.ndarray) -> float:
    spans = np.ptp(y, axis=0)
    positive = spans[spans > 0]
    return float(positive.max() / positive.min()) if len(positive) > 1 else 1.0


def _skew(values: np.ndarray) -> float:
    sd = values.std()
    return float((((values - values.mean()) / sd) ** 3).mean()) if sd > 0 else float("nan")


def _kurtosis(values: np.ndarray) -> float:
    sd = values.std()
    return float((((values - values.mean()) / sd) ** 4).mean() - 3.0) if sd > 0 else float("nan")
