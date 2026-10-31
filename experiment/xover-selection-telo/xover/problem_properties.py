"""Phase 6: the problem properties the analyses split the instances by.

Every instance gets five labels besides its `family`, with levels in `FACTORS`:

- `front`: the Pareto front's shape. Huband, Hingston, Barone & While 2006 for DTLZ and WFG;
  Zapotecas-Martinez, Coello Coello, Aguirre & Tanaka 2023, section 4.1, for ZCAT.
- `separable`: Huband et al. 2006; ZCAT's Table 4, the level's Z function.
- `modality`: Huband et al. 2006; ZCAT's Table 4.
- `pareto_set`: a product when each variable's optimal values do not depend on the others.
- `objectives`: the band of m.

Notes on the choices:
- No problem here has a purely convex front. WFG1 (convex and mixed) is `mixed`, WFG2 (convex and
  disconnected) is `disconnected`. ZCAT1 is `mixed`: at three objectives its front is a cone, with
  circular (concave) cross-sections and straight sides. The rotated DTLZ problems keep DTLZ2's and
  DTLZ3's sphere.
- A ZCAT instance with its Pareto set linked is non-separable in x whatever its level, because the
  optimal distance values depend on the position variables.
- A rotated DTLZ problem is non-separable and its Pareto set is linked, since the rotation mixes the
  variables the optimal set is a product in. Its M = I control keeps DTLZ's labels.
- `unknown` means nobody has established it: most real-world problems, and the separability of
  DTLZ5/6, whose angle mapping ties the position variables to g.

The labels are not independent of each other or of the problem families: every unknown front,
modality and Pareto set belongs to a real-world problem, every linked Pareto set is also
non-separable, and every ZCAT instance has m = 3.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import polars as pl

DTLZ: dict[str, tuple[str, str]] = {
    "dtlz1": ("linear", "multimodal"),
    "dtlz2": ("concave", "unimodal"),
    "dtlz3": ("concave", "multimodal"),
    "dtlz4": ("concave", "unimodal"),
    "dtlz5": ("degenerate", "unimodal"),
    "dtlz6": ("degenerate", "unimodal"),
    "dtlz7": ("disconnected", "multimodal"),
}
"""Front and modality (Huband et al. 2006). Every DTLZ Pareto set is a product."""

DTLZ_SEPARABILITY_UNKNOWN: tuple[str, ...] = ("dtlz5", "dtlz6")

WFG: dict[str, tuple[str, str, str, str]] = {
    "wfg1": ("mixed", "unimodal", "yes", "product"),
    "wfg2": ("disconnected", "multimodal", "no", "product"),
    "wfg3": ("degenerate", "unimodal", "no", "product"),
    "wfg4": ("concave", "multimodal", "yes", "product"),
    "wfg5": ("concave", "deceptive", "yes", "product"),
    "wfg6": ("concave", "unimodal", "no", "product"),
    "wfg7": ("concave", "unimodal", "yes", "product"),
    "wfg8": ("concave", "unimodal", "no", "linked"),
    "wfg9": ("concave", "deceptive", "no", "linked"),
}
"""Front, modality, separability and Pareto set (Huband et al. 2006).

Separability is Huband's, which is about the objectives. The Pareto set splits the non-separable ones
further: in WFG2, 3 and 6 the optimal distance variables stay at 0.35 whatever the position variables
are, so the Pareto set is still a product of per-variable sets. In WFG8 and 9 the optimal distance
values depend on the position variables (pymoo `wfg.py`, `_positional_to_optimal`)."""

ZCAT_FRONT: dict[str, str] = {
    "zcat1": "mixed",
    "zcat3": "linear",
    "zcat11": "disconnected",
    "zcat14": "degenerate",
    "zcat18": "mixed",
}
"""Front classes, section 4.1 of the ZCAT paper, for the problems in the study."""

ZCAT_SEPARABLE: dict[int, bool] = {1: True, 2: False, 3: True, 4: False}
ZCAT_MULTIMODAL: dict[int, bool] = {1: False, 2: False, 3: True, 4: True}
"""ZCAT's levels cross separability with modality (its Table 4)."""

ROTATED_DTLZ_MODALITY: dict[str, str] = {"r2_dtlz2": "unimodal", "r2_dtlz3": "multimodal"}
"""CEC 2009's UF11 (rotated DTLZ2) and UF12 (rotated DTLZ3), by their `base`."""

SEPARABLE_REAL_WORLD: tuple[str, ...] = ("river_pollution",)
"""River pollution's f1-f4 are additively separable in (x1, x2) and f5 is a max of single-variable
terms. No other real-world problem here has a published separability class."""

FACTORS: dict[str, tuple[str, ...]] = {
    "family": ("benchmark", "real_world"),
    "front": ("linear", "concave", "mixed", "disconnected", "degenerate", "unknown"),
    "separable": ("yes", "no", "unknown"),
    "modality": ("unimodal", "multimodal", "deceptive", "unknown"),
    "pareto_set": ("product", "linked", "unknown"),
    "objectives": ("3", "4-5", "6-11"),
}
"""Every factor and its levels, in display order. `family` comes with the data; `annotate` adds the rest."""

LABELS: tuple[str, ...] = ("front", "separable", "modality", "pareto_set", "objectives")
"""The columns `annotate` adds."""

MIN_M, LOW_BAND_MAX_M, MAX_M = 3, 5, 11
"""The objective counts the grid spans, and the top of the 4-5 band: exact hypervolume runs to m = 5
and is approximated above it, so the bands split where the measurement itself changes."""

_ZCAT_SLUG = re.compile(r"^(zcat\d+)_l([1-4])_ps([01])_m\d+_n\d+$")
_ROTATED_SLUG = re.compile(r"^r2_dtlz[23](_unrotated)?_m\d+_n\d+$")


class Split(NamedTuple):
    """Two levels of one factor whose crossover rankings are compared, optionally inside a subset."""

    name: str
    factor: str
    sides: tuple[str, str]
    within: tuple[tuple[str, str], ...] = ()
    """(factor, level) conditions both sides must also meet."""


SPLITS: tuple[Split, ...] = (
    Split("separable", "separable", ("yes", "no")),
    Split("modality", "modality", ("unimodal", "multimodal")),
    Split("pareto_set", "pareto_set", ("product", "linked")),
    # Every linked instance is non-separable, so the plain Pareto-set split also moves separability.
    # Inside the non-separable instances only the Pareto set changes.
    Split("pareto_set_within_non_separable", "pareto_set", ("product", "linked"), (("separable", "no"),)),
    Split("family", "family", ("benchmark", "real_world")),
    Split("objectives_3_vs_4-5", "objectives", ("3", "4-5")),
    Split("objectives_3_vs_6-11", "objectives", ("3", "6-11")),
    Split("objectives_4-5_vs_6-11", "objectives", ("4-5", "6-11")),
)
"""The contrasts the paper's problem-property table reports. Deceptive and unknown levels are left out:
the deceptive level has two base problems and unknown is a label for missing knowledge, not a property."""


def objectives_band(m: int) -> str:
    """Return the objective-count band: "3", "4-5" or "6-11"."""
    if not MIN_M <= m <= MAX_M:
        msg = f"m = {m} is outside the grid's {MIN_M}-{MAX_M} objectives"
        raise ValueError(msg)
    if m == MIN_M:
        return "3"
    return "4-5" if m <= LOW_BAND_MAX_M else "6-11"


def parse_zcat_slug(instance: str) -> tuple[str, int, bool]:
    """Return (problem, level, linked) from a ZCAT slug. Raises on anything else."""
    match = _ZCAT_SLUG.match(instance)
    if match is None:
        msg = f"{instance!r} is not a ZCAT slug"
        raise ValueError(msg)
    return match[1], int(match[2]), match[3] == "1"


def is_rotated(instance: str) -> bool:
    """Return whether a rotated-DTLZ slug is the rotated problem rather than its M = I control."""
    match = _ROTATED_SLUG.match(instance)
    if match is None:
        msg = f"{instance!r} is not a rotated DTLZ slug"
        raise ValueError(msg)
    return match[1] is None


def describe(instance: str, base: str, family: str, m: int) -> dict[str, str]:
    """Return the five labels of one instance. A benchmark base without a label raises rather than guessing."""
    band = {"objectives": objectives_band(m)}
    if base in DTLZ:
        front, modality = DTLZ[base]
        separable = "unknown" if base in DTLZ_SEPARABILITY_UNKNOWN else "yes"
        return {**band, "front": front, "modality": modality, "separable": separable, "pareto_set": "product"}
    if base in WFG:
        front, modality, separable, pareto_set = WFG[base]
        return {**band, "front": front, "modality": modality, "separable": separable, "pareto_set": pareto_set}
    if base in ZCAT_FRONT:
        _, level, linked = parse_zcat_slug(instance)
        return {
            **band,
            "front": ZCAT_FRONT[base],
            "modality": "multimodal" if ZCAT_MULTIMODAL[level] else "unimodal",
            "separable": "yes" if ZCAT_SEPARABLE[level] and not linked else "no",
            "pareto_set": "linked" if linked else "product",
        }
    if base in ROTATED_DTLZ_MODALITY:
        rotated = is_rotated(instance)
        return {
            **band,
            "front": "concave",
            "modality": ROTATED_DTLZ_MODALITY[base],
            "separable": "no" if rotated else "yes",
            "pareto_set": "linked" if rotated else "product",
        }
    if family != "real_world":
        msg = f"{instance}: benchmark base {base!r} has no labels in problem_properties"
        raise ValueError(msg)
    separable = "yes" if base in SEPARABLE_REAL_WORLD else "unknown"
    return {**band, "front": "unknown", "modality": "unknown", "separable": separable, "pareto_set": "unknown"}


def annotate(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the five label columns to a frame with `instance`, `base`, `family` and `m`."""
    meta = frame.select(["instance", "base", "family", "m"]).unique()
    labels = pl.DataFrame(
        [{"instance": row[0], **describe(*row)} for row in meta.iter_rows()],
        schema={"instance": pl.String, **dict.fromkeys(LABELS, pl.String)},
    )
    return frame.drop([c for c in LABELS if c in frame.columns]).join(labels, on="instance")


def split_sides(frame: pl.DataFrame, split: Split) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the rows on each side of a split, after its `within` conditions."""
    inside = frame
    for factor, level in split.within:
        inside = inside.filter(pl.col(factor) == level)
    a, b = split.sides
    return inside.filter(pl.col(split.factor) == a), inside.filter(pl.col(split.factor) == b)
