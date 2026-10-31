"""Phase 6: convergence curves and the two summaries of them.

Every run is sampled at 21 budget fractions, so each configuration has a curve rather than a point.
That shows whether an arm that ends well got there quickly, which a final-population number cannot.

**Everything here integrates a comparable indicator, never per-instance regret.** `add_regret`
normalises within an instance by the best and worst arm in the grid, so its scale moves when an arm
is added or dropped. Ranks survive that, and an area does not.

**The curve is the median over seeds, and the integral is of the median curve** rather than the
median of per-seed integrals. The first is what the plots show.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from xover.indicators import COMPARABLE_ACROSS_INSTANCES, LOWER_IS_BETTER

CONFIG_KEYS = ["instance", "algorithm", "crossover", "mutation"]
"""One convergence curve per configuration, pooling the seeds."""

DEFAULT_INDICATOR = COMPARABLE_ACROSS_INSTANCES[0]

TARGET_TOLERANCE = 0.10
"""Within 10 % of the range any arm reaches on that instance. A relative band, because the attainable
range differs by orders of magnitude across instances."""


def _direction(indicator: str) -> bool:
    """Return whether lower is better, refusing indicators that have no unambiguous orientation."""
    if indicator not in LOWER_IS_BETTER:
        msg = (
            f"{indicator!r} has no orientation in LOWER_IS_BETTER, so it cannot be ranked or integrated. "
            f"The distribution indicators are deliberately excluded."
        )
        raise ValueError(msg)
    return LOWER_IS_BETTER[indicator]


def curves(table: pl.DataFrame, indicator: str = DEFAULT_INDICATOR) -> pl.DataFrame:
    """Return the median-over-seeds indicator per configuration and fraction, oriented so lower is better."""
    lower_better = _direction(indicator)
    median = table.group_by([*CONFIG_KEYS, "fraction"]).agg(pl.col(indicator).median().alias("value"))
    if not lower_better:
        median = median.with_columns((-pl.col("value")).alias("value"))
    return median.sort([*CONFIG_KEYS, "fraction"])


def aocc(table: pl.DataFrame, indicator: str = DEFAULT_INDICATOR) -> pl.DataFrame:
    """Area over the convergence curve per configuration, lower being better.

    The trapezoid of the oriented indicator against budget fraction, divided by the width of the sampled
    interval so the result is on the indicator's own scale. An arm that reaches a given quality early
    scores below one that reaches it late, which is what separates this from the endpoint value.

    The integral starts at the first sampled fraction, 0.02. Extrapolating back to an unmeasured origin
    would put the largest trapezoid over the region where nothing was observed.
    """
    traced = curves(table, indicator)
    wide = traced.pivot(values="value", index=CONFIG_KEYS, on="fraction")
    columns = sorted((c for c in wide.columns if c not in CONFIG_KEYS), key=float)
    if len(columns) < 2:  # noqa: PLR2004
        msg = f"AOCC needs at least two sampled fractions, found {len(columns)}"
        raise ValueError(msg)

    grid = np.array([float(c) for c in columns])
    values = wide.select(columns).to_numpy().astype(float)
    span = float(grid[-1] - grid[0])
    return wide.select(CONFIG_KEYS).with_columns(
        pl.Series("aocc", np.trapezoid(values, grid, axis=1) / span),
        pl.Series("final", values[:, -1]),
    )


def time_to_target(
    table: pl.DataFrame,
    indicator: str = DEFAULT_INDICATOR,
    *,
    tolerance: float = TARGET_TOLERANCE,
) -> pl.DataFrame:
    """Return the smallest fraction at which a configuration comes within `tolerance` of the instance's best.

    The target is set by the best value any arm reaches on that instance at any sampled fraction, so it
    is a property of the instance rather than of the arm being scored. Censored at 1.0 when the target is
    never reached, and flagged.
    """
    traced = curves(table, indicator)
    # Oriented so lower is better, so the instance's best is the minimum over everything sampled.
    best = traced.group_by("instance").agg(pl.col("value").min().alias("best"), pl.col("value").max().alias("worst"))
    joined = traced.join(best, on="instance").with_columns(
        (pl.col("best") + tolerance * (pl.col("worst") - pl.col("best"))).alias("target")
    )
    reached = (
        joined.filter(pl.col("value") <= pl.col("target"))
        .group_by(CONFIG_KEYS)
        .agg(pl.col("fraction").min().alias("reached_at"))
    )
    return (
        joined.select(CONFIG_KEYS)
        .unique()
        .join(reached, on=CONFIG_KEYS, how="left")
        .with_columns(
            pl.col("reached_at").is_null().alias("censored"),
            pl.col("reached_at").fill_null(1.0).alias("reached_at"),
        )
    )
