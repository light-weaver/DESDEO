"""Phase 6: the statistics shared by the analyses.

Omega-squared says how much of the variance a choice accounts for. Friedman, Holm-corrected Wilcoxon
and the Nemenyi critical difference compare the arms. Kendall tau compares two orderings of the arms,
and the performance profile shows how far each arm falls behind the best one.

With 750 blocks almost anything reaches significance, so the effect size is reported next to every
p-value. The p-value only says the ordering is not noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

# --------------------------------------------------------------------------------------------------
# Variance decomposition
# --------------------------------------------------------------------------------------------------


@dataclass
class Effect:
    """One term's share of the variance."""

    term: str
    omega_squared: float
    eta_squared: float
    df: int

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {
            "term": self.term,
            "omega_squared": round(self.omega_squared, 6),
            "eta_squared": round(self.eta_squared, 6),
            "df": self.df,
        }


def omega_squared(
    table: pl.DataFrame,
    response: str,
    factors: list[str],
    *,
    interactions: list[tuple[str, ...]] | None = None,
) -> list[Effect]:
    """Decompose the variance of `response` over `factors` and their interactions.

    A type-I sequential decomposition, exact here because every instance sees every combination.

    Omega-squared rather than eta-squared as the headline: eta-squared is biased upward by the number
    of levels, and the factors have different level counts (14 crossovers, 5 algorithms).
    """
    terms: list[tuple[str, ...]] = [(f,) for f in factors]
    terms += interactions if interactions is not None else _all_pairs(factors)

    fit = _sequential_fit(table, response, terms, factors)
    if fit is None:
        return []

    return [
        Effect(
            term=":".join(term),
            # Olejnik & Algina's design "abc" (2003, Table 2): every source other than the term itself
            # sits in the denominator. With a measured factor in the design, such as the instance,
            # report `generalized_omega_squared` instead.
            omega_squared=float(max(0.0, (fit.ss[term] - fit.df[term] * fit.mse) / (fit.total_ss + fit.mse))),
            eta_squared=float(fit.ss[term] / fit.total_ss),
            df=fit.df[term],
        )
        for term in terms
    ]


@dataclass
class _Fit:
    """What a sequential decomposition leaves behind, for the omega-squared estimators to divide up."""

    ss: dict[tuple[str, ...], float]
    df: dict[tuple[str, ...], int]
    mse: float
    total_ss: float
    n: int


def _sequential_fit(
    table: pl.DataFrame, response: str, terms: list[tuple[str, ...]], factors: list[str]
) -> _Fit | None:
    """Type-I sequential decomposition of `response` over `terms`, exact on a balanced design."""
    frame = table.drop_nulls([response, *factors])
    y = frame[response].to_numpy().astype(float)
    n = len(y)
    if n < len(terms) + 2:
        return None

    grand = y.mean()
    total_ss = float(((y - grand) ** 2).sum())
    if total_ss == 0:
        return None

    ss: dict[tuple[str, ...], float] = {}
    df: dict[tuple[str, ...], int] = {}
    residual = y - grand
    used_df = 0

    for term in terms:
        labels = np.array(["\x00".join(map(str, values)) for values in frame.select(term).iter_rows()])
        unique, inverse = np.unique(labels, return_inverse=True)
        # Fit each term on what the previous terms left behind: sequential, so the shares sum to at
        # most the total rather than double-counting shared variance.
        counts = np.bincount(inverse, minlength=len(unique))
        means = np.bincount(inverse, weights=residual, minlength=len(unique)) / counts
        contribution = means[inverse]
        ss[term] = float((contribution**2).sum())
        df[term] = len(unique) - 1
        residual = residual - contribution
        used_df += df[term]

    error_df = max(1, n - 1 - used_df)
    return _Fit(ss=ss, df=df, mse=float((residual**2).sum()) / error_df, total_ss=total_ss, n=n)


def generalized_omega_squared(
    table: pl.DataFrame,
    response: str,
    manipulated: list[str],
    measured: list[str],
    *,
    interactions: list[tuple[str, ...]] | None = None,
) -> list[Effect]:
    """Olejnik & Algina's generalized omega squared (2003, Eq. 7), for designs with a measured factor.

    The study manipulates the crossover and the algorithm and samples the instance from a population of
    problems. Olejnik & Algina show that the classical share of total variance is then the wrong
    comparison, because an instance set with more between-problem variance shrinks every effect
    (pp. 441-445). Generalized omega squared keeps the measured factor's variance in the denominator
    and removes the other manipulated factors'.

        omega^2_G = (SS_e - df_e MSE) / [delta (SS_e - df_e MSE) + sum_meas (SS_m - df_m MSE) + N MSE]

    with delta = 1 when the effect is manipulated and 0 when it is measured, and the sum running over
    every source that involves a measured factor. With no manipulated factor at all this reduces to
    the classical formula.
    """
    factors = [*manipulated, *measured]
    terms: list[tuple[str, ...]] = [(f,) for f in factors]
    terms += interactions if interactions is not None else _all_pairs(factors)

    fit = _sequential_fit(table, response, terms, factors)
    if fit is None:
        return []

    measured_set = set(measured)
    adjusted = {term: fit.ss[term] - fit.df[term] * fit.mse for term in terms}
    from_measured = sum(value for term, value in adjusted.items() if measured_set & set(term))

    effects: list[Effect] = []
    for term in terms:
        delta = 0.0 if measured_set & set(term) else 1.0
        denominator = delta * adjusted[term] + from_measured + fit.n * fit.mse
        omega = adjusted[term] / denominator if denominator > 0 else 0.0
        effects.append(
            Effect(
                term=":".join(term),
                omega_squared=float(max(0.0, omega)),
                eta_squared=float(fit.ss[term] / fit.total_ss),
                df=fit.df[term],
            )
        )
    return effects


def _all_pairs(factors: list[str]) -> list[tuple[str, ...]]:
    return [(a, b) for i, a in enumerate(factors) for b in factors[i + 1 :]]


# --------------------------------------------------------------------------------------------------
# Rank-based comparisons
# --------------------------------------------------------------------------------------------------


CROSSOVER_BLOCKS: tuple[str, ...] = ("instance", "algorithm")
"""The blocks crossovers are ranked in: one instance under one algorithm.

The crossover is compared with the algorithm held fixed, so each (instance, algorithm) pair is a block
and the median over its seeds is the cell. The five blocks of one instance share its problem, and many
instances share a base problem, so tests over them are somewhat optimistic."""


def _block_columns(block: str | tuple[str, ...]) -> list[str]:
    return [block] if isinstance(block, str) else list(block)


def friedman(table: pl.DataFrame, response: str, treatment: str, block: str | tuple[str, ...] = "instance") -> dict:
    """Run a Friedman test over treatments, blocked by `block` (one column or several)."""
    blocks = _block_columns(block)
    wide = table.pivot(values=response, index=blocks, on=treatment, aggregate_function="median").drop_nulls()
    # Sorted, because the pivot's column order follows the data and would reorder the output between runs.
    treatments = sorted(c for c in wide.columns if c not in blocks)
    if len(treatments) < 3 or wide.height < 3:  # noqa: PLR2004
        return {"statistic": None, "p_value": None, "n_blocks": wide.height, "treatments": treatments}
    columns = [wide[t].to_numpy().astype(float) for t in treatments]
    statistic, p_value = stats.friedmanchisquare(*columns)
    ranks = np.array([stats.rankdata(row) for row in np.column_stack(columns)])
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_blocks": wide.height,
        "treatments": treatments,
        "mean_ranks": dict(zip(treatments, ranks.mean(axis=0).round(4).tolist(), strict=True)),
    }


def pairwise_wilcoxon(
    table: pl.DataFrame, response: str, treatment: str, block: str | tuple[str, ...] = "instance"
) -> list[dict]:
    """Run a paired Wilcoxon signed-rank test over every treatment pair, Holm-corrected."""
    blocks = _block_columns(block)
    wide = table.pivot(values=response, index=blocks, on=treatment, aggregate_function="median").drop_nulls()
    treatments = sorted(c for c in wide.columns if c not in blocks)
    rows = []
    for i, a in enumerate(treatments):
        for b in treatments[i + 1 :]:
            x, y = wide[a].to_numpy().astype(float), wide[b].to_numpy().astype(float)
            if np.allclose(x, y):
                rows.append({"a": a, "b": b, "p_value": 1.0, "median_difference": 0.0, "n": len(x)})
                continue
            try:
                _, p = stats.wilcoxon(x, y)
            except ValueError:
                p = 1.0
            rows.append(
                {"a": a, "b": b, "p_value": float(p), "median_difference": float(np.median(x - y)), "n": len(x)}
            )
    for row, corrected in zip(rows, holm_correct([r["p_value"] for r in rows]), strict=True):
        row["p_holm"] = corrected
    return rows


def holm_correct(p_values: list[float]) -> list[float]:
    """Apply the Holm-Bonferroni step-down correction, preserving input order."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * p_values[index]
        running = max(running, value)
        corrected[index] = min(1.0, running)
    return corrected.tolist()


def critical_difference(n_treatments: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical difference for a Friedman test at `alpha`."""
    q_alpha = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
    }.get(n_treatments)
    if q_alpha is None or n_blocks == 0:
        return float("nan")
    return float(q_alpha * np.sqrt(n_treatments * (n_treatments + 1) / (6 * n_blocks)))


def within_block_ranks(
    table: pl.DataFrame, response: str, treatment: str = "crossover", blocks: tuple[str, ...] = ("instance",)
) -> dict[str, float]:
    """Mean rank of each treatment level within blocks, best first (rank 1 = lowest response).

    The aggregation the study leads with. Each block's cell is the median of `response` over everything
    else inside it, the levels are ranked inside the block with ties averaged, and the ranks are averaged
    over blocks. With `blocks=("instance",)` and complete blocks this is exactly `friedman`'s `mean_ranks`.

    A median pooled over instances is the wrong ordering: it is dominated by the many instances where
    the top arms sit near zero regret, so near-tied arms swap places on noise.
    """
    keys = list(blocks)
    per_block = (
        table.group_by([*keys, treatment])
        .agg(pl.col(response).median().alias("_value"))
        .filter(pl.col("_value").is_not_null() & pl.col("_value").is_not_nan())
    )
    ranked = per_block.with_columns(pl.col("_value").rank("average").over(keys).alias("_rank"))
    means = ranked.group_by(treatment).agg(pl.col("_rank").mean().alias("_mean")).sort(["_mean", treatment])
    return dict(zip([str(t) for t in means[treatment].to_list()], means["_mean"].to_list(), strict=True))


def block_rank_matrix(
    table: pl.DataFrame, response: str, treatment: str = "crossover", blocks: tuple[str, ...] = CROSSOVER_BLOCKS
) -> tuple[pl.DataFrame, list[str], np.ndarray]:
    """Each complete block's ranks of every level, as a matrix: (block keys, levels, ranks).

    The rows are `within_block_ranks`'s per-block ranks, so the mean over any set of rows is that set's
    mean ranks. Built once, it lets a permutation test re-split the blocks thousands of times without
    regrouping the table. Rows are sorted by the block keys and columns by level name.
    """
    keys = list(blocks)
    per_block = table.group_by([*keys, treatment]).agg(pl.col(response).median().alias("_value"))
    wide = per_block.pivot(values="_value", index=keys, on=treatment).sort(keys).drop_nulls()
    levels = sorted(c for c in wide.columns if c not in keys)
    ranks = stats.rankdata(wide.select(levels).to_numpy().astype(float), axis=1)
    return wide.select(keys), levels, ranks


def ordering_tau(left: dict[str, float], right: dict[str, float]) -> dict:
    """Kendall tau-b between two mean-rank vectors over the same levels, and whether the best agrees."""
    levels = sorted(set(left) & set(right))
    result = stats.kendalltau([left[x] for x in levels], [right[x] for x in levels])
    return {
        "kendall_tau": float(result.statistic),
        "p_value": float(result.pvalue),
        "same_best": min(levels, key=left.get) == min(levels, key=right.get),
        "n_levels": len(levels),
    }


def split_tau_null(
    ranks: np.ndarray,
    groups: np.ndarray,
    sizes: tuple[int, int],
    *,
    n_permutations: int = 999,
    seed: int | list[int] = 0,
) -> np.ndarray:
    """Kendall tau between the two halves of random splits of `groups`, at the observed sizes.

    `ranks` holds one row per block and `groups` each row's group (an instance), so a split moves whole
    instances, as the property splits do. Two random halves of the same instances also disagree
    somewhat, so an observed tau means something only against this distribution.
    """
    rng = np.random.default_rng(seed)
    labels = np.unique(groups)
    if len(labels) != sum(sizes):
        msg = f"{len(labels)} groups cannot be split into {sizes}"
        raise ValueError(msg)
    index = np.searchsorted(labels, groups)
    out = np.empty(n_permutations)
    for i in range(n_permutations):
        first = np.zeros(len(labels), dtype=bool)
        first[rng.choice(len(labels), size=sizes[0], replace=False)] = True
        mask = first[index]
        out[i] = stats.kendalltau(ranks[mask].mean(axis=0), ranks[~mask].mean(axis=0)).statistic
    return out


def performance_profile(log_ratios: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Dolan and More's profile: the share of blocks within a factor tau of the block's best.

    `log_ratios` are log10(value / best value in the block), so 0 is the best and 1 is ten times worse.
    """
    log_ratios = np.asarray(log_ratios, dtype=float)
    return np.array([np.mean(log_ratios <= np.log10(tau)) for tau in np.asarray(taus, dtype=float)])


# --------------------------------------------------------------------------------------------------
# What selection per problem could gain
# --------------------------------------------------------------------------------------------------


def oracle_gain(table: pl.DataFrame, response: str, factors: list[str]) -> dict[str, float]:
    """Expected response if you knew the best level of each factor per instance, and of all of them."""
    out: dict[str, float] = {}
    baseline = table.group_by("instance").agg(pl.col(response).mean().alias("value"))
    out["no_knowledge"] = float(baseline["value"].mean())

    for factor in factors:
        # Best level of this factor per instance, averaging over everything else.
        per_level = table.group_by(["instance", factor]).agg(pl.col(response).mean().alias("value"))
        best = per_level.group_by("instance").agg(pl.col("value").min().alias("value"))
        out[f"know_{factor}"] = float(best["value"].mean())
        # One level for every instance: the single best solver of the algorithm-selection literature.
        # `know_` minus this is what per-instance selection could at most add over always using one arm.
        fixed = per_level.group_by(factor).agg(pl.col("value").mean().alias("value"))
        out[f"best_fixed_{factor}"] = float(fixed["value"].min())

    per_cell = table.group_by(["instance", *factors]).agg(pl.col(response).mean().alias("value"))
    best_all = per_cell.group_by("instance").agg(pl.col("value").min().alias("value"))
    out["know_everything"] = float(best_all["value"].mean())
    return out


def best_fixed_levels(table: pl.DataFrame, response: str, factors: list[str]) -> dict[str, str]:
    """Return the level of each factor that is best on average over instances: the single best solver.

    The companion to the `best_fixed_` entries of `oracle_gain`, which carry the value. This carries the
    name, kept apart so `oracle_gain` stays a map of numbers.
    """
    out: dict[str, str] = {}
    for factor in factors:
        per_level = table.group_by(["instance", factor]).agg(pl.col(response).mean().alias("value"))
        fixed = per_level.group_by(factor).agg(pl.col("value").mean().alias("value")).sort(["value", factor])
        out[factor] = str(fixed[factor][0])
    return out
