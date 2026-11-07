"""Phase 6: the figures that `tier.json` and the indicator tables already support.

- `cd_indicators`: critical-difference diagrams of the arms under each compliant indicator.
- `cd_family`: the same on benchmark and on real-world problems, under IGD+.
- `rank_grid`: every arm's mean rank pooled and under each algorithm, with the top places outlined.
- `profiles`: performance profiles under each algorithm, IGD+.
- `anytime`: Kendall tau between each fraction's ranking and the final one.
- `algorithm_indicator`: each algorithm's mean rank on each indicator.
- `regret_grid`: mean regret of every algorithm and crossover combination, one panel per indicator.

Run `python -m analysis.tier` first. Figures go to `results/plots/`, as PDF and PNG.

    python -m analysis.figures
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from analysis.common import load_config, load_indicators, results_dir
from xover import plotstyle, stats
from xover.indicators import add_regret, log_ratios

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TIER_ORDER = ("mixing", "portfolio", "non_mixing", "de", "off")
TIER_MARKER = {"mixing": "o", "portfolio": "D", "non_mixing": "s", "de": "^", "off": "X"}
TIER_LABEL = {
    "mixing": "mixing",
    "portfolio": "portfolio",
    "non_mixing": "non-mixing",
    "de": "DE",
    "off": "no crossover",
}
INDICATOR_LABEL = {"igd_plus": "IGD+", "epsilon_additive": "Additive epsilon", "hv_relative": "HV", "r2": "R2"}
ALGORITHM_LABEL = {
    "nsga3": "NSGA-III",
    "nsga2": "NSGA-II",
    "ibea": "IBEA",
    "spea2": "SPEA2",
    "agemoea2": "AGE-MOEA-II",
}

TAU_THRESHOLD = 0.95
"""The agreement with the final ranking that counts as settled."""

WHITE_TEXT_SHARE = 0.38
"""Cells in the darkest part of a heatmap's colour range, as a share of it, get white text."""

CD_ABOVE = 0.40
"""Inches above a CD diagram's axis: tick labels, the panel title and the CD bracket."""

CD_CLIQUE_GAP = 0.08
"""Inches between rows of clique bars, and between the axis and the first row."""

CD_CLIQUE_SEPARATION = 0.25
"""Rank units two bars in one row must keep between them."""

CD_ROW = 0.15
"""Inches between label rows."""

CD_LABEL_OFFSET = 0.45
"""Rank units between the ends of the axis and the label markers."""

CD_AXIS_SHARE = 0.46
"""Share of the panel width the rank axis takes. The rest holds the labels."""


@dataclass
class Panel:
    """One critical-difference diagram: a title, every arm's mean rank, and the critical difference."""

    title: str
    mean_ranks: dict[str, float]
    critical_difference: float


def display_tiers(tiers: dict[str, str]) -> dict[str, str]:
    """Return the tiers as the figures mark them, with both DE arms under `de`.

    `analysis.tier` keeps `de_current` in `non_mixing`, because PLAN.md's falsification test counts it
    there. A reader of a figure needs to see the two DE arms as one operator family.
    """
    return {**tiers, "de_current": "de"}


def arm_order(tier: dict) -> list[str]:
    """Return the arms ordered by their pooled mean rank, averaged over the indicators."""
    indicators = tier["indicators"]
    return sorted(tier["tiers"], key=lambda a: np.mean([tier["cuts"][i]["all"]["mean_ranks"][a] for i in indicators]))


def cliques(ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Return maximal runs of consecutive arms (sorted by rank) whose ranks span less than `cd`."""
    out: list[tuple[int, int]] = []
    for i in range(len(ranks)):
        j = max(j for j in range(i, len(ranks)) if ranks[j] - ranks[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


def clique_rows(ranks: list[float], cd: float) -> list[int]:
    """Return the row for each clique: the first row where it does not touch a bar already placed."""
    ends: list[float] = []
    rows = []
    for i, j in cliques(ranks, cd):
        row = next((r for r, end in enumerate(ends) if end + CD_CLIQUE_SEPARATION < ranks[i]), len(ends))
        if row == len(ends):
            ends.append(ranks[j])
        else:
            ends[row] = ranks[j]
        rows.append(row)
    return rows


def panel_height(panel: Panel) -> float:
    """Return the panel's height in inches."""
    ranks = sorted(panel.mean_ranks.values())
    n_rows = max(clique_rows(ranks, panel.critical_difference), default=-1) + 1
    return CD_ABOVE + CD_CLIQUE_GAP * (n_rows + 1) + CD_ROW * math.ceil(len(ranks) / 2) + 0.08


def draw_cd(ax, panel: Panel, tiers: dict[str, str], crossovers: dict[str, Any]) -> None:
    """Draw one diagram on `ax`, whose height in inches must be `panel_height(panel)`."""
    names = sorted(panel.mean_ranks, key=panel.mean_ranks.get)
    ranks = [panel.mean_ranks[name] for name in names]
    k = len(names)
    margin = (k - 1) * (1 - CD_AXIS_SHARE) / (2 * CD_AXIS_SHARE)
    ax.set_xlim(1 - margin, k + margin)
    ax.axis("off")

    ax.plot([1, k], [0, 0], color=INK_SECONDARY, linewidth=0.8, solid_capstyle="butt")
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.04], color=INK_SECONDARY, linewidth=0.8)
        ax.text(r, 0.07, str(r), ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)

    # The title sits over the left labels and the bracket over the right end of the axis.
    top = 0.29
    cd = panel.critical_difference
    ax.plot([k - cd, k], [top, top], color=INK, linewidth=1.0, solid_capstyle="butt")
    for x in (k - cd, k):
        ax.plot([x, x], [top - 0.025, top + 0.025], color=INK, linewidth=1.0)
    ax.text(k - cd - 0.15, top, f"CD = {cd:.2f}", ha="right", va="center", fontsize=7)
    ax.text(1 - margin, top, panel.title, va="center", fontweight="bold")

    # Bars joining the arms the Nemenyi test cannot separate.
    rows = clique_rows(ranks, cd)
    for (i, j), row in zip(cliques(ranks, cd), rows, strict=True):
        y = -CD_CLIQUE_GAP * (row + 1)
        ax.plot([ranks[i] - 0.06, ranks[j] + 0.06], [y, y], color=INK, linewidth=1.8, solid_capstyle="butt")

    # Elbow connectors to the labels: the better half left, the worse half right, never crossing.
    first_row = -CD_CLIQUE_GAP * (max(rows, default=-1) + 2) - 0.06
    half = math.ceil(k / 2)
    sides = [(i, i, 1 - CD_LABEL_OFFSET, "right") for i in range(half)]
    sides += [(i, k - 1 - i, k + CD_LABEL_OFFSET, "left") for i in range(half, k)]
    for index, row, x_end, align in sides:
        name, rank = names[index], ranks[index]
        y = first_row - CD_ROW * row
        ax.plot([rank, rank, x_end], [0, y, y], color=AXIS, linewidth=0.5, zorder=1)
        ax.plot(
            x_end,
            y,
            marker=TIER_MARKER[tiers[name]],
            color=plotstyle.colour(name, crossovers),
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        pad = 0.28 if align == "left" else -0.28
        ax.text(x_end + pad, y, f"{name.replace('_', '-')}  {rank:.2f}", ha=align, va="center", fontsize=8)
    ax.set_ylim(first_row - CD_ROW * (half - 1) - 0.08, CD_ABOVE - 0.02)


def tier_handles() -> list[Line2D]:
    """Return legend entries for the tiers' marker shapes, in ink because colour belongs to the arms."""
    return [
        Line2D([], [], color=INK_SECONDARY, marker=TIER_MARKER[t], markersize=5, linewidth=0, label=TIER_LABEL[t])
        for t in TIER_ORDER
    ]


def cd_figure(panels: list[Panel], tiers: dict[str, str], crossovers: dict[str, Any]) -> plt.Figure:
    """Stack one diagram per panel at full width, with the tier legend underneath."""
    heights = [panel_height(panel) for panel in panels]
    legend = 0.3
    total = sum(heights) + legend
    fig = plt.figure(figsize=(plotstyle.FIGURE_WIDTH, total))
    bottom = 1.0
    for panel, height in zip(panels, heights, strict=True):
        bottom -= height / total
        draw_cd(fig.add_axes((0, bottom, 1, height / total)), panel, tiers, crossovers)
    fig.legend(handles=tier_handles(), loc="lower center", ncol=len(TIER_ORDER), bbox_to_anchor=(0.5, 0.0))
    return fig


def rank_grid(tier: dict, crossovers: dict[str, Any]) -> plt.Figure:
    """Mean ranks, one panel per indicator, with each column's top places outlined, one per mixing-side arm.

    An outlined cell on a non-mixing arm is a failure of PLAN.md's first falsification test. Ranks are
    read from the cuts, so the pooled column is ranked over all 750 blocks and the others over 150.
    """
    columns = ["all", *(f"algorithm={a}" for a in ALGORITHM_LABEL)]
    labels = ["Pooled", *ALGORITHM_LABEL.values()]
    tiers = display_tiers(tier["tiers"])
    indicators = tier["indicators"]
    arms = arm_order(tier)

    fig, axes = plt.subplots(2, 2, figsize=(plotstyle.FIGURE_WIDTH, 6.6), sharey=True)
    fig.subplots_adjust(wspace=0.06, hspace=0.26, left=0.2, right=0.995, top=0.97, bottom=0.1)
    for ax, indicator in zip(axes.flat, indicators, strict=True):
        cuts = [tier["cuts"][indicator][column] for column in columns]
        values = np.array([[cut["mean_ranks"][arm] for cut in cuts] for arm in arms])
        ax.imshow(values, cmap="Greens_r", vmin=1, vmax=len(arms), aspect="auto")
        ax.grid(visible=False)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0, labelsize=6.5)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="left")
        ax.set_yticks(range(len(arms)), [arm.replace("_", "-") for arm in arms], fontsize=7.5)
        ax.tick_params(axis="y", pad=11)
        ax.set_title(INDICATOR_LABEL[indicator], loc="left", fontweight="bold", fontsize=9, y=-0.085)
        for row, arm in enumerate(arms):
            if ax in axes[:, 0]:
                ax.plot(
                    -0.72,
                    row,
                    marker=TIER_MARKER[tiers[arm]],
                    color=plotstyle.colour(arm, crossovers),
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    clip_on=False,
                )
            for col, cut in enumerate(cuts):
                rank = values[row, col]
                colour = "white" if (rank - 1) / (len(arms) - 1) < WHITE_TEXT_SHARE else INK
                ax.text(col, row, f"{rank:.1f}", ha="center", va="center", fontsize=6.5, color=colour)
                if arm in cut["order"][: cut["k_mixing_side"]]:
                    outline = tiers[arm] in ("non_mixing", "off", "de")
                    ax.add_patch(
                        Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#D55E00" if outline else "white",
                            linewidth=1.8 if outline else 0.8,
                            zorder=3,
                        )
                    )
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(len(arms) - 0.5, -0.5)

    handles = [
        *tier_handles(),
        Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="#D55E00", linewidth=1.8, label="top places, not on the mixing side"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0), fontsize=7.5)
    return fig


def profiles(endpoint: pl.DataFrame, tier: dict, crossovers: dict[str, Any], indicator: str = "igd_plus") -> plt.Figure:
    """Performance profiles of the arms under each algorithm."""
    factors = np.logspace(0, 1, 240)
    ratios = log_ratios(endpoint, indicator)
    arms = tier["cuts"][indicator]["all"]["order"]

    fig, axes = plt.subplots(2, 3, figsize=(plotstyle.FIGURE_WIDTH, 4.3), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.2, left=0.08, right=0.99, top=0.95, bottom=0.11)
    for ax, (algorithm, label) in zip(axes.flat, ALGORITHM_LABEL.items(), strict=False):
        part = ratios.filter(pl.col("algorithm") == algorithm)
        # Non-mixing arms first, so the mixing arms' solid lines are drawn on top.
        for arm in sorted(arms, key=lambda a: plotstyle.has_uniform(crossovers[a]) is True):
            profile = stats.performance_profile(
                part.filter(pl.col("crossover") == arm)["log_ratio"].to_numpy(), factors
            )
            ax.plot(factors, profile, linewidth=1.1, **plotstyle.arm_style(arm, crossovers))
        ax.axvline(2, color=GRID, linewidth=0.5, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        ax.minorticks_off()
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1.01)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, labelsize=7)
        ax.set_title(label, loc="left", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of instances", fontsize=8)
    fig.supxlabel(f"within this factor of the best arm's {INDICATOR_LABEL[indicator]}", fontsize=8, x=0.37, y=0.0)

    axes[0, 2].tick_params(labelbottom=True)
    key = axes[1, 2]
    key.axis("off")
    handles = [Line2D([], [], label=a.replace("_", "-"), **plotstyle.arm_style(a, crossovers)) for a in arms]
    key.legend(handles=handles, loc="center", ncol=2, fontsize=6.5, handlelength=2.4, columnspacing=0.9)
    return fig


def settled_at(fractions: list[float], tau: list[float], threshold: float = TAU_THRESHOLD) -> float | None:
    """Return the earliest fraction from which tau stays at or above `threshold` to the end."""
    for index, fraction in enumerate(fractions):
        if all(value >= threshold for value in tau[index:]):
            return fraction
    return None


def anytime(tier: dict) -> plt.Figure:
    """Show the agreement of each fraction's ranking with the final one, one panel per indicator."""
    indicators = tier["indicators"]
    fig, axes = plt.subplots(1, len(indicators), figsize=(plotstyle.FIGURE_WIDTH, 2.0), sharey=True)
    fig.subplots_adjust(wspace=0.2, left=0.08, right=0.98, top=0.88, bottom=0.2)
    for ax, indicator in zip(axes, indicators, strict=True):
        entry = tier["budget_consistency"][indicator]
        fractions, tau = entry["fractions"], entry["tau_to_final"]
        ax.axhline(TAU_THRESHOLD, color=AXIS, linewidth=0.5, zorder=0)
        ax.plot(fractions, tau, color="#08519C", linewidth=1.6)
        settled = settled_at(fractions, tau)
        if settled is not None:
            ax.axvline(settled, color=AXIS, linewidth=0.5, zorder=0)
            ax.text(
                0.97, 0.06, f"τ ≥ {TAU_THRESHOLD} from {settled:.0%}", transform=ax.transAxes, ha="right", fontsize=7
            )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
        ax.grid(visible=False)
        ax.tick_params(length=0, labelsize=7)
        ax.set_title(INDICATOR_LABEL[indicator], loc="left", fontsize=8.5)
    axes[0].set_ylabel("Kendall τ with final", fontsize=8)
    fig.supxlabel("share of the budget (%)", fontsize=8, y=0.0)
    return fig


def algorithm_ranks(endpoint: pl.DataFrame, indicators: list[str]) -> np.ndarray:
    """Return mean ranks of the algorithms, one row per indicator and one column per algorithm.

    Each block is one instance under one crossover, so the crossover is held fixed. Regret carries the
    orientation, so rank 1 is the best algorithm on every indicator.
    """
    rows = []
    for indicator in indicators:
        regret = add_regret(endpoint, indicator)
        ranks = stats.within_block_ranks(regret, "regret", "algorithm", blocks=("instance", "crossover"))
        rows.append([ranks[algorithm] for algorithm in ALGORITHM_LABEL])
    return np.array(rows)


def mean_regret(endpoint: pl.DataFrame, indicator: str, arms: list[str]) -> np.ndarray:
    """Return mean regret over instances, one row per arm and one column per algorithm.

    `add_regret` is given all the combinations, so on each instance 0 is the best algorithm and
    crossover combination and 1 the worst. The scale therefore depends on the worst combination.
    """
    means = add_regret(endpoint, indicator).group_by(["algorithm", "crossover"]).agg(pl.col("regret").mean())
    lookup = {(row["algorithm"], row["crossover"]): row["regret"] for row in means.to_dicts()}
    return np.array([[lookup[algorithm, arm] for algorithm in ALGORITHM_LABEL] for arm in arms])


def annotate_cells(ax, values: np.ndarray, *, vmin: float, vmax: float, fontsize: float) -> None:
    """Draw `values` as a dark-is-good heatmap with each cell's value written in it."""
    ax.imshow(values, cmap="Greens_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for (row, col), value in np.ndenumerate(values):
        colour = "white" if (value - vmin) / (vmax - vmin) < WHITE_TEXT_SHARE else INK
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=fontsize, color=colour)


def algorithm_indicator(endpoint: pl.DataFrame, indicators: list[str]) -> plt.Figure:
    """Show each algorithm's mean rank on each indicator, ranked within instance and crossover."""
    fig, ax = plt.subplots(figsize=(plotstyle.FIGURE_WIDTH * 0.72, 2.0))
    annotate_cells(ax, algorithm_ranks(endpoint, indicators), vmin=1, vmax=len(ALGORITHM_LABEL), fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(ALGORITHM_LABEL)), ALGORITHM_LABEL.values(), fontsize=7.5)
    ax.set_yticks(range(len(indicators)), [INDICATOR_LABEL[i] for i in indicators], fontsize=8)
    return fig


def regret_grid(endpoint: pl.DataFrame, tier: dict, crossovers: dict[str, Any]) -> plt.Figure:
    """Show the mean regret of each algorithm and crossover combination, one panel per indicator."""
    tiers = display_tiers(tier["tiers"])
    indicators = tier["indicators"]
    arms = arm_order(tier)
    tables = {indicator: mean_regret(endpoint, indicator, arms) for indicator in indicators}
    vmax = max(float(table.max()) for table in tables.values())

    fig, axes = plt.subplots(2, 2, figsize=(plotstyle.FIGURE_WIDTH, 6.6), sharey=True)
    fig.subplots_adjust(wspace=0.06, hspace=0.26, left=0.2, right=0.995, top=0.97, bottom=0.11)
    for ax, indicator in zip(axes.flat, indicators, strict=True):
        table = tables[indicator]
        annotate_cells(ax, table, vmin=0, vmax=vmax, fontsize=6.5)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(ALGORITHM_LABEL)), ALGORITHM_LABEL.values(), rotation=30, ha="left", fontsize=6.5)
        ax.set_yticks(range(len(arms)), [arm.replace("_", "-") for arm in arms], fontsize=7.5)
        ax.tick_params(axis="y", pad=11)
        ax.set_title(INDICATOR_LABEL[indicator], loc="left", fontweight="bold", fontsize=9, y=-0.085)
        row, col = np.unravel_index(int(table.argmin()), table.shape)
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#D55E00", linewidth=1.8, zorder=3))
        if ax in axes[:, 0]:
            for position, arm in enumerate(arms):
                ax.plot(
                    -0.72,
                    position,
                    marker=TIER_MARKER[tiers[arm]],
                    color=plotstyle.colour(arm, crossovers),
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    clip_on=False,
                )
        ax.set_xlim(-0.5, len(ALGORITHM_LABEL) - 0.5)
        ax.set_ylim(len(arms) - 0.5, -0.5)

    handles = [
        *tier_handles(),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#D55E00", linewidth=1.8, label="lowest mean regret"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0), fontsize=7.5)
    return fig


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    """Write one figure as PDF and PNG."""
    plotstyle.save(fig, directory / f"{name}.pdf", close=False)
    plotstyle.save(fig, directory / f"{name}.png")
    print(f"wrote {directory / name}.png")


def main() -> None:
    """Draw every figure."""
    plotstyle.use()
    config = load_config()
    crossovers = config["crossovers"]
    root = results_dir(config)
    plots = root / "plots"
    tier = json.loads((root / "analysis" / "tier.json").read_text())
    tiers = display_tiers(tier["tiers"])
    arms = len(tiers)
    n_algorithms = len(ALGORITHM_LABEL)

    def panel(title: str, indicator: str, cut: str) -> Panel:
        entry = tier["cuts"][indicator][cut]
        scale = 1 if cut.startswith("algorithm=") else n_algorithms
        return Panel(title, entry["mean_ranks"], stats.critical_difference(arms, entry["n_instances"] * scale))

    pooled = [panel(INDICATOR_LABEL[i], i, "all") for i in tier["indicators"]]
    save(cd_figure(pooled, tiers, crossovers), plots, "cd_indicators")
    family = [
        panel("Benchmark problems, IGD+", "igd_plus", "family=benchmark"),
        panel("Real-world problems, IGD+", "igd_plus", "family=real_world"),
    ]
    save(cd_figure(family, tiers, crossovers), plots, "cd_family")
    save(rank_grid(tier, crossovers), plots, "rank_grid")
    endpoint = load_indicators(config).filter(pl.col("fraction") == 1.0)
    save(profiles(endpoint, tier, crossovers), plots, "profiles")
    save(anytime(tier), plots, "anytime")
    save(algorithm_indicator(endpoint, tier["indicators"]), plots, "algorithm_indicator")
    save(regret_grid(endpoint, tier, crossovers), plots, "regret_grid")


if __name__ == "__main__":
    main()
