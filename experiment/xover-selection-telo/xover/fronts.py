"""Reference fronts: a uniform subset of merged archives.

Shared by calibration (phase 1) and the reference fronts (phase 2), so the two are built the same way. The construction
is the one the RE suite uses for its own published fronts, which makes this study's fronts and the
suite's comparable in construction as well as in content:

    Tanabe, R., Ishibuchi, H., & Oyama, A. (2017), *Benchmarking multi- and many-objective
    evolutionary algorithms under two optimization scenarios*, IEEE Access 5:19597-19619,
    DOI 10.1109/ACCESS.2017.2751071.

Three steps: merge the archives, filter to the non-dominated set, then reduce to a uniform subset by
greedy max-min distance-based selection.

**The ideal and nadir come from the merged pool, before subsetting, and are frozen there.** Seeding
the selection with the argmins alone would preserve the ideal but not the nadir: the point maximising
objective *j* over a non-dominated set need not be the argmin of any objective. Taking the box from
the pool makes it independent of the cap, so changing the cap changes the resolution of `R_p` and
nothing else. Seeding with all 2m extremes is still worth doing, because a subset that has dropped
the corners gives a misleadingly small IGD+ at the edges of the front.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import moocore
import numpy as np


@dataclass(frozen=True)
class ReferenceFront:
    """A frozen reference front and the box it was measured in."""

    points: np.ndarray
    """The uniform subset, in minimisation sense."""
    ideal: np.ndarray
    """Componentwise minimum of the merged pool, before subsetting."""
    nadir: np.ndarray
    """Componentwise maximum of the merged pool, before subsetting."""
    pool_size: int
    """How many non-dominated points the pool held before the subset was taken."""
    hypervolume: float | None = None
    """`HV(R)` in this front's own unit box under the study's fixed reference point, the denominator
    of `hv_relative`. Computed once when the front is frozen and carried in its sidecar."""

    @property
    def m(self) -> int:
        """The objective count."""
        return int(self.points.shape[1])

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """Map objective vectors into the unit box defined by this front's ideal and nadir."""
        span = self.nadir - self.ideal
        span = np.where(span > 0, span, 1.0)
        return (np.asarray(values, dtype=float) - self.ideal) / span

    def to_dict(self) -> dict:
        """A JSON-serialisable summary, without the point cloud."""
        return {
            "n_points": int(self.points.shape[0]),
            "pool_size": self.pool_size,
            "hypervolume": self.hypervolume,
            "ideal": self.ideal.tolist(),
            "nadir": self.nadir.tolist(),
        }


def build_reference_front(pools: list[np.ndarray], *, cap: int) -> ReferenceFront:
    """Build `R_p` from a list of archives, each in minimisation sense.

    Args:
        pools: archives to merge. Empty entries are ignored.
        cap: how many points to keep (`xover.design.subset_cap`).

    Returns:
        The frozen reference front, its ideal and its nadir.
    """
    stacked = [np.asarray(p, dtype=float) for p in pools if p is not None and len(p) > 0]
    if not stacked:
        msg = "no archives to build a reference front from"
        raise ValueError(msg)
    merged = np.concatenate(stacked, axis=0)
    merged = merged[np.isfinite(merged).all(axis=1)]
    if len(merged) == 0:
        msg = "every archived point was non-finite"
        raise ValueError(msg)

    front = np.asarray(moocore.filter_dominated(merged))
    ideal = front.min(axis=0)
    nadir = front.max(axis=0)

    subset = uniform_subset(front, cap, ideal=ideal, nadir=nadir)
    return ReferenceFront(points=subset, ideal=ideal, nadir=nadir, pool_size=int(front.shape[0]))


def uniform_subset(
    points: np.ndarray,
    cap: int,
    *,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce `points` to at most `cap` of them by greedy max-min distance selection.

    Selection runs on objectives normalised to the pool's own box, so no objective dominates the
    distance through its units alone, and is seeded with the 2m extreme points -- the argmin and
    argmax of every objective -- so the subset keeps the corners of the front rather than trimming
    them off.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= cap:
        return points

    if ideal is None:
        ideal = points.min(axis=0)
    if nadir is None:
        nadir = points.max(axis=0)
    span = nadir - ideal
    span = np.where(span > 0, span, 1.0)
    scaled = (points - ideal) / span

    seeds: list[int] = []
    for j in range(points.shape[1]):
        seeds.append(int(np.argmin(scaled[:, j])))
        seeds.append(int(np.argmax(scaled[:, j])))
    # dict.fromkeys rather than set(), so the seed order is deterministic.
    seeds = list(dict.fromkeys(seeds))[:cap]

    chosen = np.zeros(n, dtype=bool)
    # Running nearest-distance-to-the-chosen-set, updated one point at a time. That keeps the whole
    # selection O(n * cap * m) rather than recomputing a full pairwise matrix per step, which is what
    # makes a 400 000-point pool tractable at all.
    nearest = np.full(n, np.inf)
    for index in seeds:
        chosen[index] = True
        nearest = np.minimum(nearest, _distance_to(scaled, scaled[index]))

    while chosen.sum() < cap:
        masked = np.where(chosen, -np.inf, nearest)
        index = int(np.argmax(masked))
        if not np.isfinite(masked[index]):
            break  # every remaining point is a duplicate of one already chosen
        chosen[index] = True
        nearest = np.minimum(nearest, _distance_to(scaled, scaled[index]))

    return points[chosen]


def _distance_to(scaled: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Euclidean distance from every row to one point."""
    diff = scaled - point
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def freeze(
    archives: list[np.ndarray],
    *,
    cap: int,
    hv_reference: float,
    exact_hv_max_m: int,
) -> ReferenceFront:
    """Build a reference front from archives and attach `HV(R)`, its hypervolume in its own unit box.

    The archives are merged in the order given; the workflow sorts them by run, so the front does not
    depend on the order files happen to be listed in.
    """
    from xover.indicators import hypervolume

    front = build_reference_front(archives, cap=cap)
    m = front.points.shape[1]
    value = hypervolume(front.normalise(front.points), np.full(m, hv_reference), exact_hv_max_m)
    return replace(front, hypervolume=float(value))


def save(front: ReferenceFront, npz_path, record: dict) -> None:
    """Write a frozen front (`points`, `ideal`, `nadir` as float32) and its JSON sidecar."""
    from pathlib import Path

    from xover.run import write_json, write_npz

    npz_path = Path(npz_path)
    write_npz(
        npz_path,
        {
            "points": front.points.astype(np.float32),
            "ideal": front.ideal.astype(np.float32),
            "nadir": front.nadir.astype(np.float32),
        },
    )
    write_json(npz_path.with_suffix(".json"), record | front.to_dict())


def load(npz_path) -> ReferenceFront:
    """Read a frozen front and its sidecar back."""
    import json
    from pathlib import Path

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as handle:
        points, ideal, nadir = handle["points"], handle["ideal"], handle["nadir"]
    sidecar = json.loads(npz_path.with_suffix(".json").read_text())
    return ReferenceFront(
        points=points.astype(float),
        ideal=ideal.astype(float),
        nadir=nadir.astype(float),
        pool_size=int(sidecar["pool_size"]),
        hypervolume=sidecar["hypervolume"],
    )
