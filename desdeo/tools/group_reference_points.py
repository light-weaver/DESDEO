"""Generate IPR reference points inside the convex hull of a group's aspiration points.

In "favourite" method (name is WIP) each decision maker (DM) has a corresponding aspiration point, and the convex hull
of these points is the region of compromise between them. The functions here sample reference points uniformly inside
that hull, so that the iterative Pareto representer (`desdeo.tools.iterative_pareto_representer`) only searches for
solutions between what the DMs asked for.

The aspiration points may contain duplicates, some may lie inside the hull of the others, and together they may
span an affine subspace of any dimension up to the number of objectives (a line, a plane, a hyperplane). The hull
is built in the lowest dimensional space that holds the points. A singular value decomposition finds the affine
span, the points are written in an orthonormal basis of that span, the hull is triangulated and sampled there, and
the samples are mapped back to the objective space.

IPR reference points live in the normalized objective space (ideal at 0, nadir at 1, every objective minimized),
on the plane where the components sum to the number of objectives. Moving a reference point along the ideal-nadir
diagonal does not change the solution of the achievement scalarizing function IPR uses
(`desdeo.tools.scalarization.add_asf_diff`), so the aspirations are projected onto that plane before the hull is
built.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from desdeo.problem import Problem
from desdeo.problem.utils import objective_dict_to_numpy_array
from desdeo.tools.scalarization import objective_dict_has_all_symbols
from desdeo.tools.utils import flip_maximized_objective_values, get_corrected_ideal, get_corrected_nadir


@dataclass(frozen=True, eq=False)
class AffineSubspace:
    """An affine subspace given by a point in it and an orthonormal basis.

    The subspace holds the points `origin + y @ basis.T` for every `y` in R^d, where d is the number of columns of
    `basis`. With d = 0 the subspace is the single point `origin`.

    Attributes:
        origin (np.ndarray): a point in the subspace, shape (k,).
        basis (np.ndarray): orthonormal basis vectors of the subspace as columns, shape (k, d).
    """

    origin: np.ndarray
    basis: np.ndarray

    @property
    def dim(self) -> int:
        """The dimension of the subspace."""
        return self.basis.shape[1]

    @classmethod
    def from_points(cls, points: np.ndarray, *, rtol: float = 1e-8, atol: float | None = None) -> "AffineSubspace":
        """Find the smallest affine subspace that holds the given points.

        The origin is the mean of the points. The basis comes from a singular value decomposition of the centered
        points: a direction counts as part of the subspace when its singular value exceeds
        `max(rtol * largest_singular_value, atol)`. Duplicates and points inside the hull of the others do not
        change the result.

        Args:
            points (np.ndarray): the points, shape (m, k).
            rtol (float, optional): tolerance relative to the largest singular value. Defaults to 1e-8.
            atol (float | None, optional): absolute tolerance on the singular values. If None, it is
                `1e-12 * max(1, max(abs(points)))`. The absolute floor keeps rounding noise, such as the residue
                left when identical points are averaged, from counting as a dimension. Defaults to None.

        Raises:
            ValueError: `points` is not a non-empty 2-D array of finite values.

        Returns:
            AffineSubspace: the subspace, with dimension at most min(m - 1, k).
        """
        points = _as_point_array(points)
        origin = points.mean(axis=0)
        _, singular_values, vt = np.linalg.svd(points - origin, full_matrices=False)

        if atol is None:
            atol = 1e-12 * max(1.0, float(np.abs(points).max()))
        threshold = max(rtol * singular_values[0], atol)
        dim = int(np.sum(singular_values > threshold))

        return cls(origin=origin, basis=vt[:dim].T)

    def flatten(self, x: np.ndarray) -> np.ndarray:
        """Write points in the coordinates of the subspace.

        Points outside the subspace are projected orthogonally onto it.

        Args:
            x (np.ndarray): points in the full space, shape (n, k) or (k,).

        Returns:
            np.ndarray: the coordinates of the points in the subspace, shape (n, d).
        """
        return (np.atleast_2d(np.asarray(x, dtype=float)) - self.origin) @ self.basis

    def unflatten(self, y: np.ndarray) -> np.ndarray:
        """Map coordinates in the subspace back to the full space.

        Args:
            y (np.ndarray): coordinates in the subspace, shape (n, d).

        Returns:
            np.ndarray: the points in the full space, shape (n, k).
        """
        y = np.asarray(y, dtype=float).reshape(-1, self.dim)
        return self.origin + y @ self.basis.T

    def truncated(self, dim: int) -> "AffineSubspace":
        """Return the subspace spanned by the first `dim` basis vectors.

        The basis vectors from `from_points` are ordered by decreasing singular value, so the truncated subspace
        drops the directions in which the points spread the least.

        Args:
            dim (int): the dimension of the truncated subspace, between 0 and `self.dim`.

        Raises:
            ValueError: `dim` is out of range.

        Returns:
            AffineSubspace: the truncated subspace.
        """
        if not 0 <= dim <= self.dim:
            msg = f"Cannot truncate a {self.dim}-dimensional subspace to {dim} dimensions."
            raise ValueError(msg)
        return AffineSubspace(origin=self.origin, basis=self.basis[:, :dim])


def sample_convex_hull(flat_points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points uniformly inside the convex hull of full-dimensional points.

    In one dimension the hull is an interval and the samples are uniform on it. In more dimensions the vertices of
    the hull are triangulated (Delaunay), a simplex is picked with probability proportional to its volume, and a
    point is drawn uniformly inside it with Dirichlet(1, ..., 1) barycentric weights. The result is exactly uniform
    over the hull, with no rejection step.

    Args:
        flat_points (np.ndarray): the points, shape (m, d) with d >= 1. They must span all d dimensions, for
            example the output of `AffineSubspace.flatten`.
        num_points (int): the number of samples.
        rng (np.random.Generator): the random number generator.

    Raises:
        ValueError: the input is malformed or the hull has zero volume.
        scipy.spatial.QhullError: qhull cannot build the hull, which happens when the points are (numerically)
            confined to a lower dimensional subspace.

    Returns:
        np.ndarray: the samples, shape (num_points, d).
    """
    flat_points = _as_point_array(flat_points)
    _check_num_points(num_points)
    dim = flat_points.shape[1]

    if dim == 1:
        return rng.uniform(flat_points.min(), flat_points.max(), size=(num_points, 1))

    vertices = flat_points[ConvexHull(flat_points).vertices]
    simplices = Delaunay(vertices).simplices
    corners = vertices[simplices]
    # The volume of a simplex is |det| / d!, and the d! cancels when the volumes are normalized.
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
    total_volume = volumes.sum()
    if not total_volume > 0:
        msg = "The convex hull of the points has zero volume."
        raise ValueError(msg)

    chosen = rng.choice(len(simplices), size=num_points, p=volumes / total_volume)
    weights = rng.dirichlet(np.ones(dim + 1), size=num_points)

    # One vertex slot at a time, so memory stays O(num_points * d).
    samples = np.zeros((num_points, dim))
    for j in range(dim + 1):
        samples += weights[:, j, np.newaxis] * vertices[simplices[chosen, j]]
    return samples


def generate_points_in_hull(
    points: np.ndarray,
    num_points: int,
    *,
    rtol: float = 1e-8,
    atol: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points uniformly inside the convex hull of a set of points.

    The points may be degenerate: duplicated, collinear, coplanar, or confined to any lower dimensional affine
    subspace. The affine span of the points is found first (see `AffineSubspace.from_points`), the points are
    flattened into it, the hull is sampled there, and the samples are mapped back.

    Args:
        points (np.ndarray): the points, shape (m, k).
        num_points (int): the number of samples.
        rtol (float, optional): relative tolerance for the dimension of the span. Defaults to 1e-8.
        atol (float | None, optional): absolute tolerance for the dimension of the span. See
            `AffineSubspace.from_points`. Defaults to None.
        seed (int | np.random.Generator | None, optional): seed or generator for the random numbers.
            Defaults to None.

    Raises:
        ValueError: `points` is malformed or `num_points` is not a positive integer.

    Returns:
        np.ndarray: the samples, shape (num_points, k). If all points coincide (the span has dimension 0), the
            hull is a single point and it is returned alone, shape (1, k).
    """
    _check_num_points(num_points)
    points = _as_point_array(points)
    subspace = AffineSubspace.from_points(points, rtol=rtol, atol=atol)

    if subspace.dim == 0:
        return subspace.origin[np.newaxis, :].copy()

    rng = np.random.default_rng(seed)
    while True:
        try:
            return subspace.unflatten(sample_convex_hull(subspace.flatten(points), num_points, rng))
        except QhullError:
            # Never raised in one dimension, so the loop ends by the time dim reaches 1.
            warnings.warn(
                f"Qhull could not build the {subspace.dim}-dimensional hull of the points, which are nearly confined "
                f"to fewer dimensions. Retrying in {subspace.dim - 1} dimensions. A larger rtol or atol avoids this.",
                stacklevel=2,
            )
            subspace = subspace.truncated(subspace.dim - 1)


def normalize_objective_vectors(
    problem: Problem,
    vectors: list[dict[str, float]],
    *,
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
) -> np.ndarray:
    """Normalize objective vectors so that the ideal is at 0, the nadir at 1, and every objective is minimized.

    Args:
        problem (Problem): the problem the vectors belong to.
        vectors (list[dict[str, float]]): objective vectors in original units, keyed by objective symbol.
        ideal (dict[str, float] | None, optional): the ideal point with maximized objectives multiplied by -1, as
            in `add_asf_diff`. If None, it is read from the problem. Defaults to None.
        nadir (dict[str, float] | None, optional): the nadir point with maximized objectives multiplied by -1. If
            None, it is read from the problem. Defaults to None.

    Raises:
        ValueError: a vector, the ideal or the nadir misses an objective, the problem has no ideal or nadir, or
            the nadir is not worse than the ideal in every (minimized) objective.

    Returns:
        np.ndarray: the normalized vectors, shape (len(vectors), k), with columns in the order of
            `problem.objectives`.
    """
    ideal_arr, nadir_arr = _resolve_ideal_nadir(problem, ideal, nadir)
    if len(vectors) == 0:
        msg = "At least one objective vector is needed."
        raise ValueError(msg)

    rows = []
    for vector in vectors:
        if not objective_dict_has_all_symbols(problem, vector):
            msg = f"The objective vector {vector} is missing a value for one or more objectives."
            raise ValueError(msg)
        rows.append(objective_dict_to_numpy_array(problem, flip_maximized_objective_values(problem, vector)))

    return (np.array(rows, dtype=float) - ideal_arr) / (nadir_arr - ideal_arr)


def project_to_reference_plane(points: np.ndarray) -> np.ndarray:
    """Project normalized points onto the IPR reference plane along the ideal-nadir diagonal.

    The plane holds the points whose components sum to the number of objectives k. It is perpendicular to the
    diagonal and passes through the nadir (1, ..., 1).

    Args:
        points (np.ndarray): normalized points, shape (n, k) or (k,).

    Returns:
        np.ndarray: the projected points, shape (n, k).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    num_objectives = points.shape[1]
    return points + ((num_objectives - points.sum(axis=1)) / num_objectives)[:, np.newaxis]


def generate_group_reference_points(
    problem: Problem,
    aspirations: list[dict[str, float]],
    num_points: int,
    *,
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
    rtol: float = 1e-8,
    atol: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate IPR reference points inside the convex hull of the DMs' aspiration points.

    The aspirations are normalized (see `normalize_objective_vectors`) and projected onto the IPR reference plane
    (see `project_to_reference_plane`). The dimension of their affine span on that plane is found, and points are
    sampled uniformly inside their convex hull (see `generate_points_in_hull`). The result can be passed to
    `choose_reference_point` as the array of candidate reference points, and a chosen point converted back to
    original units with `denormalize_reference_point`.

    Two aspirations that differ only by a step along the ideal-nadir diagonal lead to the same achievement
    scalarizing problem, so they are projected to the same point. If all aspirations are projected to one point,
    that point is the only reference point returned and IPR has nothing more to search after evaluating it.
    Aspirations outside the box between the ideal and the nadir are not clipped.

    IPR discards candidates near evaluated solutions (`_find_bad_RPs` uses a thickness of 0.02), so a hull
    narrower than that runs out of candidates after few evaluations.

    Args:
        problem (Problem): the problem.
        aspirations (list[dict[str, float]]): one aspiration point per DM in original units, keyed by objective
            symbol. Duplicates are allowed.
        num_points (int): the number of reference points to generate.
        ideal (dict[str, float] | None, optional): the ideal point with maximized objectives multiplied by -1. If
            None, it is read from the problem. Defaults to None.
        nadir (dict[str, float] | None, optional): the nadir point with maximized objectives multiplied by -1. If
            None, it is read from the problem. Defaults to None.
        rtol (float, optional): relative tolerance for the dimension of the aspirations' span. Defaults to 1e-8.
        atol (float | None, optional): absolute tolerance for the dimension of the aspirations' span. See
            `AffineSubspace.from_points`. Defaults to None.
        seed (int | np.random.Generator | None, optional): seed or generator for the random numbers.
            Defaults to None.

    Raises:
        ValueError: see `normalize_objective_vectors`, or `num_points` is not a positive integer.

    Returns:
        np.ndarray: the reference points in the normalized space, shape (num_points, k), or (1, k) when all
            aspirations are projected to one point. Columns follow the order of `problem.objectives`, and every
            row sums to k.
    """
    normalized = normalize_objective_vectors(problem, aspirations, ideal=ideal, nadir=nadir)
    return generate_points_in_hull(project_to_reference_plane(normalized), num_points, rtol=rtol, atol=atol, seed=seed)


def denormalize_reference_point(
    problem: Problem,
    reference_point: np.ndarray,
    *,
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert a normalized reference point back to original units.

    This undoes `normalize_objective_vectors`, including the sign flip of maximized objectives. The output can be
    passed to `add_asf_diff` as the reference point.

    Args:
        problem (Problem): the problem.
        reference_point (np.ndarray): a normalized reference point, shape (k,), in the order of
            `problem.objectives`.
        ideal (dict[str, float] | None, optional): the ideal point with maximized objectives multiplied by -1. If
            None, it is read from the problem. Defaults to None.
        nadir (dict[str, float] | None, optional): the nadir point with maximized objectives multiplied by -1. If
            None, it is read from the problem. Defaults to None.

    Raises:
        ValueError: the reference point has the wrong shape, or see `normalize_objective_vectors`.

    Returns:
        dict[str, float]: the reference point in original units, keyed by objective symbol.
    """
    ideal_arr, nadir_arr = _resolve_ideal_nadir(problem, ideal, nadir)
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != ideal_arr.shape:
        msg = f"Expected a reference point of shape {ideal_arr.shape}, got {reference_point.shape}."
        raise ValueError(msg)

    values = ideal_arr + reference_point * (nadir_arr - ideal_arr)
    minimized = {obj.symbol: float(value) for obj, value in zip(problem.objectives, values, strict=True)}
    return flip_maximized_objective_values(problem, minimized)


def _resolve_ideal_nadir(
    problem: Problem, ideal: dict[str, float] | None, nadir: dict[str, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ideal and nadir as arrays with maximized objectives negated, and check them."""
    ideal = get_corrected_ideal(problem) if ideal is None else ideal
    nadir = get_corrected_nadir(problem) if nadir is None else nadir

    for name, point in (("ideal", ideal), ("nadir", nadir)):
        if not objective_dict_has_all_symbols(problem, point):
            msg = f"The {name} point is missing a value for one or more objectives."
            raise ValueError(msg)

    ideal_arr = np.asarray(objective_dict_to_numpy_array(problem, ideal), dtype=float)
    nadir_arr = np.asarray(objective_dict_to_numpy_array(problem, nadir), dtype=float)

    not_worse = [obj.symbol for obj, gap in zip(problem.objectives, nadir_arr - ideal_arr, strict=True) if not gap > 0]
    if not_worse:
        msg = (
            f"The nadir must be worse than the ideal in every objective, but it is not for {not_worse}. The ideal and "
            "nadir are expected with maximized objectives multiplied by -1."
        )
        raise ValueError(msg)

    return ideal_arr, nadir_arr


def _as_point_array(points: np.ndarray) -> np.ndarray:
    """Convert to a float array of shape (m, k) with m, k >= 1 and finite values, or raise ValueError."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:  # noqa: PLR2004
        msg = f"Expected a non-empty 2-D array of points, got shape {arr.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "The points must be finite."
        raise ValueError(msg)
    return arr


def _check_num_points(num_points: int) -> None:
    """Raise ValueError unless num_points is a positive integer."""
    if not isinstance(num_points, int | np.integer) or num_points < 1:
        msg = f"num_points must be a positive integer, got {num_points!r}."
        raise ValueError(msg)
