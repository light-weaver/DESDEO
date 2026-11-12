"""Phase 6: shared matplotlib styling, and the two attributes each arm is judged on.

Every function that needs to know about an arm takes its option dictionary, or the `crossovers`
section of the config, so nothing here lists arms by attribute.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGURE_WIDTH = 6.5
"""Inches, a single column of a two-column paper at full bleed."""

# Two arms need care, and a naive rule gets both wrong:
#
#   `none` declares `uniform_xover_probability = 0.5`, but its `xover_probability` and
#   `pair_xover_probability` are both 0.0, so no variable is ever crossed and the component never
#   fires. Testing `uniform_xover_probability > 0` alone would call the off arm a mixing arm.
#
#   `composite` is a portfolio: one SBX member and two others, two of the three carrying the
#   component. It is neither yes nor no on either axis, so both predicates return `MIXED` for it.

MIXED = None
"""What `is_sbx` and `has_uniform` return for an arm that is partly both, currently `composite`."""


def is_sbx(options: dict[str, Any]) -> bool | None:
    """Return whether the arm is one of the operators shipped under the name SBX, or `MIXED`."""
    if options["name"] == "CompositeCrossover":
        names = {member["name"] for member in options["crossovers"]}
        if names == {"SimulatedBinaryCrossover"}:
            return True
        return MIXED if "SimulatedBinaryCrossover" in names else False
    if not crosses_anything(options):
        return False  # the off arm runs SBX code but performs no crossover
    return options["name"] == "SimulatedBinaryCrossover"


def has_uniform(options: dict[str, Any]) -> bool | None:
    """Return whether the arm performs per-variable uniform mixing, or `MIXED` for a split portfolio."""
    if options["name"] == "CompositeCrossover":
        flags = {_member_mixes(member) for member in options["crossovers"]}
        return flags.pop() if len(flags) == 1 else MIXED
    return _member_mixes(options)


def crosses_anything(options: dict[str, Any]) -> bool:
    """Return False when the arm is configured so that no decision variable is ever recombined."""
    return all(options.get(gate) != 0.0 for gate in ("xover_probability", "pair_xover_probability"))


def _member_mixes(options: dict[str, Any]) -> bool:
    """Return whether one operator's settings actually exercise the uniform-mixing component."""
    if options["name"] == "UniformCrossover":
        return bool(options.get("uniform_xover_probability", 0) > 0)
    rate = options.get("uniform_xover_probability")
    return bool(rate and rate > 0 and crosses_anything(options))


# Colour carries the operator family and line style the mixing component, so `sbx_no_uniform` is the
# one dashed line among the solid blues, and `bex` and `bex_uniform` are one purple, dashed and solid.

FAMILY_COLOURS = {
    "sbx": ["#08306B", "#08519C", "#2171B5", "#4292C6", "#0E7C9B", "#2AA5C4", "#6BAED6"],
    "bex": ["#54278F", "#807DBA"],
    "uniform": ["#009E73", "#00593F", "#7FCDBB"],
}
"""Ramps for the families, assigned in config order, so `uniform` keeps its green and the two mutation
probes after it take darker and lighter ones. Seven SBX arms are more than blues can keep apart, so the
ramp turns towards cyan; figures that must tell SBX arms apart need another encoding."""

SOLO_COLOURS = {
    "blend": "#D55E00",
    "sax": "#8C564B",
    "de": "#E69F00",
    "de_current": "#A66F00",
    "pcx": "#CC79A7",
    "composite": "#000000",
    "none": "#999999",
}
"""Every other arm gets a hue of its own, colour-blind safe (Okabe & Ito where possible)."""

MIX_LINESTYLE = {True: "-", False: (0, (5, 2)), MIXED: (0, (3, 1, 1, 1))}
"""Solid when the arm mixes per variable, dashed when it does not, dash-dot for the portfolio."""


def family(arm: str, options: dict[str, Any]) -> str:
    """Return which operator family an arm belongs to, for colouring."""
    if arm in SOLO_COLOURS:
        return arm
    names = {"SimulatedBinaryCrossover": "sbx", "BoundedExponentialCrossover": "bex", "UniformCrossover": "uniform"}
    return names.get(options["name"], arm)


def colour(arm: str, crossovers: dict[str, dict[str, Any]]) -> str:
    """Return the arm's colour: a family ramp for SBX and BEX, a hue of its own for everything else."""
    group = family(arm, crossovers[arm])
    if group in FAMILY_COLOURS:
        members = [a for a, options in crossovers.items() if family(a, options) == group]
        return FAMILY_COLOURS[group][members.index(arm) % len(FAMILY_COLOURS[group])]
    return SOLO_COLOURS[arm]


def arm_style(arm: str, crossovers: dict[str, dict[str, Any]]) -> dict:
    """Return line kwargs: colour for the operator family, dash for whether it mixes per variable."""
    return {"color": colour(arm, crossovers), "linestyle": MIX_LINESTYLE[has_uniform(crossovers[arm])]}


def use() -> None:
    """Apply the study's style to the current matplotlib session."""
    mpl.rcParams.update(
        {
            "figure.figsize": (FIGURE_WIDTH, FIGURE_WIDTH * 0.62),
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
        }
    )


def save(fig, path, *, close: bool = True) -> None:
    """Write a figure, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
