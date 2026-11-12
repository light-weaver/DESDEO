"""The mixing tier: does a per-variable mixing step decide which crossovers work?

PLAN.md's claim is that crossovers whose offspring take whole variable values from either parent sit
above those that do not. This script answers it and produces every number the claim rests on:

1. **Each arm's tier**, derived from the config, never listed by hand.
2. **The falsification tests**, pooled and on each of the five algorithms.
3. **Mean ranks for every cut**: indicator, algorithm, and every level of every problem property.
4. **The mixing split's share of the between-crossover variance**, per indicator.
5. **Paired ablations**: how often the version of an operator with the mixing step beats the version
   without it, per indicator, algorithm and family.
6. **The budget**: whether the top places are the mixing side at every sampled fraction, and how
   consistent the ranking is between fractions.
7. **The step-size control**: whether the arms' offspring step explains their ranks as well as the
   mixing split does, and how often a mixing arm beats a non-mixing one in the blocks where the
   non-mixing one steps further.
8. **The mechanism by tier**: median crossover diagnostics per tier and arm.

    python -m analysis.tier
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import polars as pl

from analysis.common import (
    MECHANISM_COLUMNS,
    load_config,
    load_indicators,
    load_mechanism,
    provenance,
    write,
)
from xover import plotstyle, problem_properties, stats
from xover.indicators import PARETO_COMPLIANT, add_regret

MIXING_SIDE = ("mixing", "portfolio")
"""The tiers counted on the mixing side. The variance share is also reported without the portfolio
(`composite`), so that it cannot carry the result."""

NON_MIXING = ("non_mixing", "off")
"""The tiers that count against the claim when they reach the top places. `de` is in neither group:
its binomial crossover mixes the mutant and the target per variable, so a top place for it does not
falsify the claim. It is still reported among the intruders."""

MINORITY_SHARE = 0.5
"""The mixing split carrying less than this of the between-crossover sum of squares falsifies the claim."""

TIER_RULE = (
    "mixing: every crossover operation can give an offspring a whole variable value from the other "
    "parent; portfolio: some members do and some do not; non_mixing: none do, including de_current, whose "
    "offspring is its target plus a difference vector at CR = 1; de: DE/rand/1/bin's binomial crossover "
    "exchanges variables with a mutant, not between two parents, so it is kept apart; off: the arm "
    "recombines nothing."
)

COMPOUND_CUTS: tuple[tuple[tuple[str, str], ...], ...] = ((("separable", "no"), ("pareto_set", "product")),)
"""Cuts on two properties at once. Non-separable problems whose Pareto set is still a product of
per-variable sets (WFG2, 3 and 6, ZCAT at levels 2 and 4 unlinked): an offspring assembled from two
optimal parents' values is optimal there, whatever the objectives' separability. Every linked instance
is non-separable, so `separable=no` alone mixes the two cases."""

STEP_CONTROL_EXCLUDE = ("none", "de", "de_current", "composite", "uniform_pm50", "uniform_pm100")
"""Arms outside the step-size control: no crossover, both DE arms, whose steps come from a difference
vector, the portfolio, and the two mutation probes. `offspring_step` is measured on the crossover's
output before mutation, so for the probes it is uniform crossover's step and leaves out their heavy
mutation."""

STEP_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("sbx_default", "pcx"),
    ("sbx_default", "bex"),
    ("sbx_default", "blend"),
    ("uniform", "pcx"),
    ("uniform", "blend"),
)
"""(mixing arm, non-mixing arm) pairs. A win for the mixing arm in the blocks where the non-mixing arm
steps further cannot come from a larger step."""

ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("sbx_default", "sbx_no_uniform", "the step within SBX"),
    ("bex_uniform", "bex", "the step moved to a second base operator"),
    ("uniform", "sbx_no_uniform", "the step alone against SBX without it"),
    ("uniform", "none", "the step alone against no crossover"),
    ("sbx_no_uniform", "none", "SBX without the step against no crossover"),
    ("uniform", "sbx_default", "the step alone against the step plus SBX's spread"),
    ("uniform_pm50", "sbx_default_eta20", "the swap plus mutation on half the variables against SBX on half"),
    ("uniform_pm100", "sbx_pisa", "the swap plus mutation on every variable against SBX on every one"),
    ("uniform_pm50", "uniform", "mutation on half the variables against 1/n, both after the swap"),
    ("uniform_pm100", "uniform", "mutation on every variable against 1/n, both after the swap"),
    ("sbx_default_eta20", "uniform", "SBX's spread on half the variables against no spread, both with the swap"),
)
"""(A, B, what the contrast isolates): share of blocks where A beats B.

The last five test the mutation probes. Each probe is matched with the SBX arm that perturbs the same
share of variables at the same distribution index, 20. If SBX's spread acts like a mutation, each probe
ties its SBX partner. The SBX arms also mutate at 1/n, which the probes replace."""


def tier(options: dict[str, Any]) -> str:
    """Return the arm's mixing tier. See `TIER_RULE`."""
    match options["name"]:
        case "DifferentialEvolutionCrossover":
            return "de"
        case "DifferentialEvolutionCurrentCrossover":
            return "non_mixing"
    if not plotstyle.crosses_anything(options):
        return "off"
    mixes = plotstyle.has_uniform(options)
    if mixes is plotstyle.MIXED:
        return "portfolio"
    return "mixing" if mixes else "non_mixing"


def summarise(mean_ranks: dict[str, float], tiers: dict[str, str], n_instances: int) -> dict:
    """Return where the mixing side lands in one ordering."""
    order = list(mean_ranks)
    side = [arm for arm in order if tiers[arm] in MIXING_SIDE]
    others = [arm for arm in order if tiers[arm] not in MIXING_SIDE]
    top = order[: len(side)]
    return {
        "n_instances": n_instances,
        "mean_ranks": {arm: round(rank, 4) for arm, rank in mean_ranks.items()},
        "order": order,
        "k_mixing_side": len(side),
        "top_k_is_mixing_side": all(tiers[arm] in MIXING_SIDE for arm in top),
        "intruders": [arm for arm in top if tiers[arm] not in MIXING_SIDE],
        "displaced": [arm for arm in side if arm not in top],
        "best_other": {"arm": others[0], "position": order.index(others[0]) + 1} if others else None,
        "worst_mixing_side_position": max(order.index(arm) for arm in side) + 1 if side else None,
        "position": {arm: order.index(arm) + 1 for arm in ("uniform", "sbx_no_uniform") if arm in order},
    }


def cuts(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the subsets the tier is checked on, each ranked on its own instances.

    Every algorithm, every level of every factor in `problem_properties.FACTORS`, and the
    `COMPOUND_CUTS`. `frame` must carry the property columns. Empty cuts are dropped.
    """
    out = {"all": frame}
    for algorithm in sorted(frame["algorithm"].unique().to_list()):
        out[f"algorithm={algorithm}"] = frame.filter(pl.col("algorithm") == algorithm)
    for factor, levels in problem_properties.FACTORS.items():
        for level in levels:
            out[f"{factor}={level}"] = frame.filter(pl.col(factor) == level)
    for conditions in COMPOUND_CUTS:
        sub = frame
        for factor, level in conditions:
            sub = sub.filter(pl.col(factor) == level)
        out["&".join(f"{factor}={level}" for factor, level in conditions)] = sub
    return {name: sub for name, sub in out.items() if sub.height}


def variance_share(frame: pl.DataFrame, tiers: dict[str, str], exclude: tuple[str, ...] = ()) -> dict:
    """Return the mixing split's share of the between-crossover sum of squares.

    Regret is centred within (instance, algorithm) blocks, so what is left is the crossover's effect
    and its interactions. The between-crossover sum of squares is then split into the part the
    two-group contrast (mixing side against the rest) carries and the part between arms inside a
    group. Every block holds every arm, so the design is balanced and the split is exact.
    """
    kept = frame.filter(~pl.col("crossover").is_in(list(exclude)))
    side = [arm for arm, t in tiers.items() if t in MIXING_SIDE]
    centred = kept.with_columns(
        (pl.col("regret") - pl.col("regret").mean().over(["instance", "algorithm"])).alias("_c"),
        pl.col("crossover").is_in(side).alias("_side"),
    )
    per_arm = centred.group_by("crossover").agg(pl.col("_c").mean().alias("mu"), pl.len().alias("n"))
    per_side = centred.group_by("_side").agg(pl.col("_c").mean().alias("mu"), pl.len().alias("n"))
    ss_arm = float((per_arm["n"] * per_arm["mu"] ** 2).sum())
    ss_side = float((per_side["n"] * per_side["mu"] ** 2).sum())
    ss_total = float((centred["_c"] ** 2).sum())
    return {
        "excluded": list(exclude),
        "n_arms": per_arm.height,
        "n_blocks": centred.select(["instance", "algorithm"]).unique().height,
        "share_of_between_crossover_ss": ss_side / ss_arm if ss_arm > 0 else None,
        "between_crossover_share_of_within_block_ss": ss_arm / ss_total if ss_total > 0 else None,
    }


def falsification(endpoint: pl.DataFrame, indicator: str, tiers: dict[str, str]) -> dict:
    """Run PLAN.md's two tests on one indicator, pooled and on each algorithm.

    `endpoint` is raw indicator rows at the final fraction. Regret is computed inside each scope, so
    that the sum of squares of one algorithm is not scaled by the others (pitfall 1 of docs/phase6.md).
    The claim fails in a scope if a non-mixing arm reaches the top places, as many as the mixing side
    has arms, or if the mixing split carries a minority of the between-crossover sum of squares over
    all the arms.
    """
    scopes = {"all": endpoint}
    for algorithm in sorted(endpoint["algorithm"].unique().to_list()):
        scopes[algorithm] = endpoint.filter(pl.col("algorithm") == algorithm)

    out = {}
    for scope, rows in scopes.items():
        regret = add_regret(rows, indicator)
        summary = summarise(
            stats.within_block_ranks(regret, "regret", blocks=stats.CROSSOVER_BLOCKS),
            tiers,
            regret["instance"].n_unique(),
        )
        non_mixing = [arm for arm in summary["intruders"] if tiers[arm] in NON_MIXING]
        share = variance_share(regret, tiers)["share_of_between_crossover_ss"]
        out[scope] = {
            "top_k": summary["order"][: summary["k_mixing_side"]],
            "non_mixing_in_top_k": non_mixing,
            "de_in_top_k": [arm for arm in summary["intruders"] if tiers[arm] == "de"],
            "mixing_split_share": share,
            "falsified": bool(non_mixing) or share < MINORITY_SHARE,
        }
    return out


def median_regret_by_tier(frame: pl.DataFrame, tiers: dict[str, str]) -> dict[str, float]:
    """Return median endpoint regret per tier, over instance x algorithm cells."""
    labels = pl.DataFrame({"crossover": list(tiers), "tier": list(tiers.values())})
    joined = frame.join(labels, on="crossover")
    rows = joined.group_by("tier").agg(pl.col("regret").median().alias("m")).sort("m")
    return {row["tier"]: round(row["m"], 4) for row in rows.to_dicts()}


def ablation(frame: pl.DataFrame, better: str, worse: str) -> dict:
    """Return the share of (instance, algorithm) blocks where `better` has lower regret than `worse`, ties half."""
    cells = (
        frame.filter(pl.col("crossover").is_in([better, worse]))
        .pivot(values="regret", index=["instance", "family", "algorithm"], on="crossover")
        .drop_nulls([better, worse])
    )

    def rate(block: pl.DataFrame) -> float | None:
        if block.height == 0:
            return None
        wins = (block[better] < block[worse]).sum() + 0.5 * (block[better] == block[worse]).sum()
        return round(float(wins) / block.height, 4)

    return {
        "overall": rate(cells),
        "n_blocks": cells.height,
        "by_algorithm": {a: rate(cells.filter(pl.col("algorithm") == a)) for a in sorted(cells["algorithm"].unique())},
        "by_family": {f: rate(cells.filter(pl.col("family") == f)) for f in sorted(cells["family"].unique())},
    }


def budget(frame: pl.DataFrame, tiers: dict[str, str], fractions: list[float]) -> list[dict]:
    """Return the mixing side's place at every sampled fraction."""
    rows = []
    for fraction in fractions:
        at = frame.filter(pl.col("fraction") == fraction)
        summary = summarise(
            stats.within_block_ranks(at, "regret", blocks=stats.CROSSOVER_BLOCKS), tiers, at["instance"].n_unique()
        )
        rows.append(
            {
                "fraction": fraction,
                "best": summary["order"][0],
                "top_k_is_mixing_side": summary["top_k_is_mixing_side"],
                "intruders": summary["intruders"],
                "mean_ranks": summary["mean_ranks"],
            }
        )
    return rows


def budget_consistency(frame: pl.DataFrame, fractions: list[float]) -> dict:
    """Return Kendall tau between the rankings at every pair of fractions."""
    ranks = [
        stats.within_block_ranks(frame.filter(pl.col("fraction") == f), "regret", blocks=stats.CROSSOVER_BLOCKS)
        for f in fractions
    ]
    n = len(fractions)
    tau = [[1.0] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        tau[i][j] = tau[j][i] = round(stats.ordering_tau(ranks[i], ranks[j])["kendall_tau"], 4)
    pairs = [(tau[i][j], i, j) for i, j in itertools.combinations(range(n), 2)]
    worst, i, j = min(pairs)
    return {
        "fractions": fractions,
        "tau": tau,
        "mean_pairwise_tau": round(float(np.mean([value for value, _, _ in pairs])), 4),
        "min_pairwise_tau": {"tau": worst, "fractions": [fractions[i], fractions[j]]},
        "tau_to_final": [tau[i][n - 1] for i in range(n)],
    }


def step_control(
    endpoint: pl.DataFrame,
    mechanism: pl.DataFrame,
    tiers: dict[str, str],
    pairs: tuple[tuple[str, str], ...] = STEP_CONTROL_PAIRS,
    exclude: tuple[str, ...] = STEP_CONTROL_EXCLUDE,
) -> dict:
    """Return whether the mixing split explains the ranks, or the offspring step does.

    Removing the offspring swap from SBX also changes its offspring step, so "mixing matters" could be
    read as "step size matters". Within each (instance, algorithm) block the arms' regret ranks, centred
    on the block mean, are regressed on a mixing indicator and on the rank of the arm's offspring step.
    The R^2 of each alone and of both says which one carries the ordering.
    """
    block = ["instance", "algorithm"]
    joined = (
        endpoint.select([*block, "crossover", "regret"])
        .join(mechanism.select([*block, "crossover", "offspring_step"]), on=[*block, "crossover"])
        .filter(~pl.col("crossover").is_in(list(exclude)))
        .with_columns(
            pl.col("regret").rank().over(block).alias("regret_rank"),
            pl.col("offspring_step").rank().over(block).alias("step_rank"),
            (pl.col("crossover").replace_strict(tiers, return_dtype=pl.String) == "mixing")
            .cast(pl.Float64)
            .alias("mixes"),
        )
    )
    centred = joined.with_columns(
        [(pl.col(c) - pl.col(c).mean().over(block)).alias(c) for c in ("regret_rank", "step_rank", "mixes")]
    )
    y = centred["regret_rank"].to_numpy()

    def fit(columns: list[str]) -> tuple[float, np.ndarray]:
        x = centred.select(columns).to_numpy()
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return float(1 - np.sum((y - x @ beta) ** 2) / np.sum(y**2)), beta

    r2_mixing, _ = fit(["mixes"])
    r2_step, _ = fit(["step_rank"])
    r2_both, beta = fit(["mixes", "step_rank"])

    wide = joined.pivot(on="crossover", index=block, values=["regret", "offspring_step"])
    matched = {}
    for a, b in pairs:
        further = wide.filter(pl.col(f"offspring_step_{b}") >= pl.col(f"offspring_step_{a}"))
        matched[f"{a}_vs_{b}"] = {
            "blocks_where_b_steps_further": further.height,
            "a_wins_there": (
                round(float((further[f"regret_{a}"] < further[f"regret_{b}"]).mean()), 4) if further.height else None
            ),
            "a_wins_overall": round(float((wide[f"regret_{a}"] < wide[f"regret_{b}"]).mean()), 4),
        }
    steps = joined.group_by("crossover").agg(pl.col("offspring_step").median()).sort("crossover")
    return {
        "arms": sorted(joined["crossover"].unique().to_list()),
        "n_blocks": joined.select(block).unique().height,
        "r2_within_block": {"mixing": round(r2_mixing, 4), "step_rank": round(r2_step, 4), "both": round(r2_both, 4)},
        "coefficients": {"mixing": round(float(beta[0]), 4), "step_rank": round(float(beta[1]), 4)},
        "median_offspring_step": {arm: round(step, 5) for arm, step in steps.iter_rows()},
        "matched": matched,
    }


def mechanism_by_tier(mechanism: pl.DataFrame, tiers: dict[str, str]) -> dict:
    """Return median crossover diagnostics per tier and per arm, over (instance, algorithm) blocks."""
    with_tier = mechanism.with_columns(pl.col("crossover").replace_strict(tiers, return_dtype=pl.String).alias("tier"))

    def medians(key: str) -> dict:
        table = with_tier.group_by(key).agg([pl.col(c).median() for c in MECHANISM_COLUMNS]).sort(key)
        return {
            row[key]: {c: None if row[c] is None else round(row[c], 4) for c in MECHANISM_COLUMNS}
            for row in table.iter_rows(named=True)
        }

    retains_less = {}
    for a, b, _ in ABLATIONS:
        pair = with_tier.filter(pl.col("crossover").is_in([a, b])).pivot(
            on="crossover", index=["instance", "algorithm"], values="diversity_retained"
        )
        retains_less[f"{a}_vs_{b}"] = round(float((pair[a] < pair[b]).mean()), 4)
    return {
        "by_tier": medians("tier"),
        "by_arm": medians("crossover"),
        "a_retains_less_diversity_than_b": retains_less,
    }


def main() -> None:
    """Run every section and write `results/analysis/tier.json`."""
    config = load_config()
    table = load_indicators(config)
    meta = problem_properties.annotate(table.select(["instance", "base", "family", "m"]).unique())
    fractions = sorted(table["fraction"].unique().to_list())
    final = fractions[-1]
    tiers = {arm: tier(config["crossovers"][arm]) for arm in config["run"]["crossovers"]}
    mechanism = load_mechanism(config)

    payload: dict = {
        **provenance(),
        "aggregation": "mean rank within (instance, algorithm) blocks of seed-median regret",
        "tier_rule": TIER_RULE,
        "tiers": tiers,
        "mixing_side": sorted(arm for arm, t in tiers.items() if t in MIXING_SIDE),
        "indicators": list(PARETO_COMPLIANT),
        "fractions": fractions,
        "falsification": {},
        "cuts": {},
        "cut_ablations": {},
        "variance_share": {},
        "median_regret_by_tier": {},
        "ablations": {},
        "budget": {},
        "budget_consistency": {},
        "step_control": {},
        "mechanism_by_tier": mechanism_by_tier(mechanism, tiers),
    }

    for indicator in PARETO_COMPLIANT:
        payload["falsification"][indicator] = falsification(table.filter(pl.col("fraction") == final), indicator, tiers)
        regret = add_regret(table, indicator).join(meta, on="instance")
        endpoint = regret.filter(pl.col("fraction") == final)
        by_cut = cuts(endpoint)
        payload["cuts"][indicator] = {
            name: summarise(
                stats.within_block_ranks(sub, "regret", blocks=stats.CROSSOVER_BLOCKS),
                tiers,
                sub["instance"].n_unique(),
            )
            for name, sub in by_cut.items()
        }
        # Where an arm off the mixing side breaks into the top places, the question is whether the swap
        # still pays inside that cut.
        payload["cut_ablations"][indicator] = {
            name: {f"{a}_vs_{b}": ablation(sub, a, b)["overall"] for a, b, _ in ABLATIONS}
            for name, sub in by_cut.items()
        }
        payload["variance_share"][indicator] = [
            variance_share(endpoint, tiers),
            variance_share(endpoint, tiers, ("none", "de", "de_current")),
            variance_share(endpoint, tiers, ("none", "de", "de_current", "composite")),
        ]
        payload["median_regret_by_tier"][indicator] = median_regret_by_tier(endpoint, tiers)
        payload["ablations"][indicator] = {
            f"{a}_vs_{b}": {"isolates": what, **ablation(endpoint, a, b)} for a, b, what in ABLATIONS
        }
        payload["budget"][indicator] = budget(regret, tiers, fractions)
        payload["budget_consistency"][indicator] = budget_consistency(regret, fractions)
        payload["step_control"][indicator] = step_control(endpoint, mechanism, tiers)

    path = write(config, "tier", payload)
    _report(payload)
    print(f"\nwrote {path}")


def _report(payload: dict) -> None:
    print(f"tiers: {payload['tiers']}")
    print(f"mixing side ({len(payload['mixing_side'])}): {payload['mixing_side']}")
    for indicator in payload["indicators"]:
        print(f"\n=== {indicator} ===")
        for scope, entry in payload["falsification"][indicator].items():
            print(
                f"  {scope:10s} falsified: {entry['falsified']!s:5s}  share {entry['mixing_split_share']:.3f}  "
                f"non-mixing in top: {entry['non_mixing_in_top_k']}  de: {entry['de_in_top_k']}"
            )
        for name, entry in payload["cuts"][indicator].items():
            other = entry["best_other"]
            print(
                f"  {name:36s} n={entry['n_instances']:3d}  top {entry['k_mixing_side']} = mixing side: "
                f"{entry['top_k_is_mixing_side']!s:5s}  intruders {entry['intruders']}  best other "
                f"{other['arm']} (#{other['position']})"
            )
        shares = ", ".join(
            f"{v['share_of_between_crossover_ss']:.3f} ({v['n_arms']} arms)"
            for v in payload["variance_share"][indicator]
        )
        print(f"  mixing split's share of the between-crossover SS: {shares}")
        print(f"  median regret by tier: {payload['median_regret_by_tier'][indicator]}")
        for pair, entry in payload["ablations"][indicator].items():
            print(f"  {pair:34s} {entry['overall']:.3f}  by algorithm {entry['by_algorithm']}")
        rows = payload["budget"][indicator]
        held = [r["fraction"] for r in rows if r["top_k_is_mixing_side"]]
        settled = next(
            (r["fraction"] for i, r in enumerate(rows) if all(x["top_k_is_mixing_side"] for x in rows[i:])), None
        )
        print(
            f"  top places are the mixing side at {len(held)} of {len(rows)} fractions, "
            f"and at every fraction from {settled}"
        )
        control = payload["step_control"][indicator]
        print(f"  step control R^2: {control['r2_within_block']}")
        ranks = ", ".join(f"{arm} {rank:.2f}" for arm, rank in payload["cuts"][indicator]["all"]["mean_ranks"].items())
        print(f"  pooled mean ranks: {ranks}")


if __name__ == "__main__":
    main()
