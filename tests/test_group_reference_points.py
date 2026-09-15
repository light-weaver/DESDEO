"""Tests for generating IPR reference points inside the convex hull of a group's aspirations."""

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.spatial import ConvexHull, QhullError

from desdeo.problem.testproblems import dtlz2, river_pollution_problem, simple_test_problem
from desdeo.tools import group_reference_points
from desdeo.tools.group_reference_points import (
    AffineSubspace,
    denormalize_reference_point,
    generate_group_reference_points,
    generate_points_in_hull,
    normalize_objective_vectors,
    project_to_reference_plane,
)
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.utils import get_corrected_ideal, get_corrected_nadir


def _random_affine_points(rng: np.random.Generator, num_dims: int, dim: int, num_points: int) -> np.ndarray:
    """Random points spanning a random dim-dimensional affine subspace of R^num_dims."""
    basis, _ = np.linalg.qr(rng.normal(size=(num_dims, num_dims)))
    return rng.normal(size=num_dims) + rng.normal(size=(num_points, dim)) @ basis[:, :dim].T


def _hull_residuals(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Residual of writing each sample as a convex combination of the points (0 when inside the hull).

    Solved with non-negative least squares; the sum-to-one row is weighted so that it acts as a hard constraint.
    """
    weight = 1e3
    a = np.vstack((points.T, weight * np.ones(len(points))))
    return np.array([nnls(a, np.append(x, weight))[1] for x in np.atleast_2d(samples)])


def _original_ideal_nadir(problem) -> tuple[dict[str, float], dict[str, float]]:
    """The ideal and nadir in original units, without flipping maximized objectives."""
    return {o.symbol: o.ideal for o in problem.objectives}, {o.symbol: o.nadir for o in problem.objectives}


DTLZ2_ASPIRATIONS = [
    {"f_1": 0.2, "f_2": 0.6, "f_3": 0.7},
    {"f_1": 0.7, "f_2": 0.2, "f_3": 0.5},
    {"f_1": 0.5, "f_2": 0.5, "f_3": 0.1},
    {"f_1": 0.5, "f_2": 0.5, "f_3": 0.1},  # duplicate
    {"f_1": 0.45, "f_2": 0.45, "f_3": 0.45},  # inside the hull of the others after projection
]

RIVER_ASPIRATIONS = [
    {"f_1": 5.5, "f_2": 3.0, "f_3": 3.0, "f_4": -4.0, "f_5": 0.2},
    {"f_1": 6.0, "f_2": 3.2, "f_3": 1.0, "f_4": -6.0, "f_5": 0.1},
    {"f_1": 5.0, "f_2": 3.3, "f_3": 5.0, "f_4": -2.0, "f_5": 0.3},
]


@pytest.mark.gdmtools
@pytest.mark.parametrize("num_dims", [3, 4, 5])
def test_affine_dimension_detection(num_dims):
    """The dimension of the affine span is found for every dimension from 0 to the full space."""
    rng = np.random.default_rng(num_dims)
    for dim in range(num_dims + 1):
        points = _random_affine_points(rng, num_dims, dim, num_points=dim + 3)
        assert AffineSubspace.from_points(points).dim == dim


@pytest.mark.gdmtools
def test_dimension_special_cases():
    """Identical, duplicated, collinear, coplanar, few and noisy points get the expected dimension."""
    rng = np.random.default_rng(0)

    assert AffineSubspace.from_points(np.tile(rng.random(4), (5, 1))).dim == 0
    # The mean of these is not exactly 0.1, so a purely relative tolerance would see a dimension.
    assert AffineSubspace.from_points(np.full((7, 3), 0.1)).dim == 0

    a, b = rng.random(4), rng.random(4)
    assert AffineSubspace.from_points(np.array([a, b, a, b, b])).dim == 1
    assert AffineSubspace.from_points(_random_affine_points(rng, 5, 1, 6)).dim == 1
    assert AffineSubspace.from_points(_random_affine_points(rng, 4, 2, 4)).dim == 2
    assert AffineSubspace.from_points(_random_affine_points(rng, 5, 3, 8)).dim == 3
    # m points span at most m - 1 dimensions.
    assert AffineSubspace.from_points(rng.random((3, 6))).dim == 2

    plane = _random_affine_points(rng, 5, 2, 10)
    assert AffineSubspace.from_points(plane + 1e-13 * rng.normal(size=plane.shape)).dim == 2
    assert AffineSubspace.from_points(plane + 1e-4 * rng.normal(size=plane.shape)).dim == 5


@pytest.mark.gdmtools
def test_flatten_unflatten_roundtrip():
    """The basis is orthonormal, points in the subspace survive a round trip, and truncation keeps leading columns."""
    points = _random_affine_points(np.random.default_rng(1), 5, 3, 7)
    subspace = AffineSubspace.from_points(points)

    np.testing.assert_allclose(subspace.basis.T @ subspace.basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(subspace.unflatten(subspace.flatten(points)), points, atol=1e-12)

    truncated = subspace.truncated(1)
    assert truncated.dim == 1
    np.testing.assert_array_equal(truncated.basis, subspace.basis[:, :1])
    with pytest.raises(ValueError):
        subspace.truncated(4)


@pytest.mark.gdmtools
@pytest.mark.parametrize("case", ["full", "coplanar", "collinear", "interior", "duplicates"])
def test_samples_inside_hull(case):
    """Every sample is a convex combination of the input points, whatever their degeneracy."""
    rng = np.random.default_rng(2)
    match case:
        case "full":
            points = rng.random((6, 3))
        case "coplanar":
            points = _random_affine_points(rng, 4, 2, 5)
        case "collinear":
            points = _random_affine_points(rng, 5, 1, 4)
        case "interior":
            tetrahedron = rng.random((4, 3))
            centre = tetrahedron.mean(axis=0)
            points = np.vstack((tetrahedron, centre, 0.5 * (tetrahedron[0] + centre)))
        case "duplicates":
            base = _random_affine_points(rng, 4, 3, 5)
            points = np.vstack((base, base[:2]))

    samples = generate_points_in_hull(points, 2000, seed=3)

    assert samples.shape == (2000, points.shape[1])
    assert _hull_residuals(points, samples[:300]).max() < 1e-9
    # The oracle itself must reject a point outside the hull.
    assert _hull_residuals(points, points.max(axis=0) + 1.0).min() > 1e-3


@pytest.mark.gdmtools
def test_uniform_on_simplex():
    """Samples in a simplex have the simplex's centroid as their mean."""
    vertices = np.random.default_rng(4).random((4, 4))  # a 3-simplex in R^4
    samples = generate_points_in_hull(vertices, 100_000, seed=5)
    np.testing.assert_allclose(samples.mean(axis=0), vertices.mean(axis=0), atol=5e-3)


@pytest.mark.gdmtools
def test_uniform_ignores_interior_and_duplicate_points():
    """A tilted square with an off-centre interior point and a duplicated corner is sampled uniformly."""
    rng = np.random.default_rng(6)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    flat = np.vstack((square, [[0.8, 0.7]], square[:1]))
    basis, _ = np.linalg.qr(rng.normal(size=(3, 2)))
    offset = rng.normal(size=3)

    samples = generate_points_in_hull(offset + flat @ basis.T, 100_000, seed=7)

    np.testing.assert_allclose(samples.mean(axis=0), offset + np.array([0.5, 0.5]) @ basis.T, atol=5e-3)


@pytest.mark.gdmtools
def test_identical_points_return_one_row():
    """When all points coincide, the hull is that point and it is returned alone."""
    point = np.array([0.2, 0.5, 0.3])
    out = generate_points_in_hull(np.tile(point, (4, 1)), 100, seed=0)
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], point, atol=1e-15)


@pytest.mark.gdmtools
def test_collinear_points_fill_the_segment():
    """Collinear points give samples on the segment between the two extreme points, covering all of it."""
    start, end = np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 0.0, 5.0])
    points = start + np.outer([0.0, 0.3, 1.0, 0.6, 1.0], end - start)

    samples = generate_points_in_hull(points, 10_000, seed=0)
    t = (samples - start) @ (end - start) / np.dot(end - start, end - start)

    np.testing.assert_allclose(start + np.outer(t, end - start), samples, atol=1e-12)
    assert t.min() >= -1e-12
    assert t.max() <= 1 + 1e-12
    assert t.min() < 0.01
    assert t.max() > 0.99


@pytest.mark.gdmtools
def test_seed_reproducibility():
    """The same seed or an equally seeded generator reproduces the samples; another seed does not."""
    points = np.random.default_rng(8).random((5, 3))
    samples = generate_points_in_hull(points, 50, seed=9)

    np.testing.assert_array_equal(samples, generate_points_in_hull(points, 50, seed=9))
    np.testing.assert_array_equal(samples, generate_points_in_hull(points, 50, seed=np.random.default_rng(9)))
    assert not np.array_equal(samples, generate_points_in_hull(points, 50, seed=10))


@pytest.mark.gdmtools
@pytest.mark.parametrize(
    ("points", "num_points"),
    [
        (np.random.default_rng(0).random((3, 2)), 0),
        (np.random.default_rng(0).random((3, 2)), 2.5),
        (np.zeros((0, 3)), 10),
        (np.array([[0.0, np.nan], [1.0, 1.0]]), 10),
        (np.array([1.0, 2.0, 3.0]), 10),
    ],
)
def test_invalid_inputs_raise(points, num_points):
    """Malformed points or a non-positive number of samples raise ValueError."""
    with pytest.raises(ValueError):
        generate_points_in_hull(points, num_points)


@pytest.mark.gdmtools
def test_qhull_failure_falls_back_to_fewer_dimensions(monkeypatch):
    """If qhull fails, a warning is issued and the hull is sampled with one dimension fewer."""

    def failing_in_3d(points):
        if points.shape[1] == 3:
            raise QhullError("simulated failure")
        return ConvexHull(points)

    monkeypatch.setattr(group_reference_points, "ConvexHull", failing_in_3d)
    points = np.random.default_rng(11).random((6, 3))

    with pytest.warns(UserWarning, match="Retrying in 2 dimensions"):
        samples = generate_points_in_hull(points, 500, seed=0)

    plane = AffineSubspace.from_points(points).truncated(2)
    assert samples.shape == (500, 3)
    np.testing.assert_allclose(plane.unflatten(plane.flatten(samples)), samples, atol=1e-12)


@pytest.mark.gdmtools
def test_output_on_ipr_plane():
    """The reference points lie on the IPR plane, inside the hull of the projected aspirations."""
    problem = dtlz2(5, 3)
    out = generate_group_reference_points(problem, DTLZ2_ASPIRATIONS, 5000, seed=0)

    assert out.shape == (5000, 3)
    assert np.abs(out.sum(axis=1) - 3).max() < 1e-12
    projected = project_to_reference_plane(normalize_objective_vectors(problem, DTLZ2_ASPIRATIONS))
    assert _hull_residuals(projected, out[:300]).max() < 1e-9


@pytest.mark.gdmtools
def test_projection_removes_the_diagonal():
    """The origin and the k unit vectors span k dimensions, and k - 1 once projected onto the IPR plane."""
    k = 4
    points = np.vstack((np.zeros(k), np.eye(k)))
    assert AffineSubspace.from_points(points).dim == k
    assert AffineSubspace.from_points(project_to_reference_plane(points)).dim == k - 1


@pytest.mark.gdmtools
def test_maximized_objectives_normalization():
    """With maximized objectives, the ideal maps to 0, the nadir to 1, and both project to (1, ..., 1)."""
    problem = river_pollution_problem()
    ideal, nadir = _original_ideal_nadir(problem)
    k = len(problem.objectives)

    normalized = normalize_objective_vectors(problem, [ideal, nadir])
    np.testing.assert_allclose(normalized, [np.zeros(k), np.ones(k)], atol=1e-12)
    np.testing.assert_allclose(project_to_reference_plane(normalized), np.ones((2, k)), atol=1e-12)

    for i, obj in enumerate(problem.objectives):
        at_nadir_in_one = normalize_objective_vectors(problem, [{**ideal, obj.symbol: obj.nadir}])
        np.testing.assert_allclose(project_to_reference_plane(at_nadir_in_one)[0], np.eye(k)[i] + (k - 1) / k)


@pytest.mark.gdmtools
def test_denormalize_roundtrip():
    """Denormalizing a projected aspiration moves it only along the ideal-nadir direction in original units."""
    problem = river_pollution_problem()
    ideal, nadir = _original_ideal_nadir(problem)
    symbols = [o.symbol for o in problem.objectives]
    direction = np.array([nadir[s] - ideal[s] for s in symbols])

    for aspiration in RIVER_ASPIRATIONS:
        projected = project_to_reference_plane(normalize_objective_vectors(problem, [aspiration]))[0]
        back = denormalize_reference_point(problem, projected)
        shift = np.array([back[s] - aspiration[s] for s in symbols])
        np.testing.assert_allclose(shift, (shift @ direction) / (direction @ direction) * direction, atol=1e-12)

    k = len(symbols)
    assert denormalize_reference_point(problem, np.zeros(k)) == pytest.approx(ideal)
    assert denormalize_reference_point(problem, np.ones(k)) == pytest.approx(nadir)
    with pytest.raises(ValueError):
        denormalize_reference_point(problem, np.zeros(k + 1))


@pytest.mark.gdmtools
def test_explicit_ideal_nadir_matches_problem():
    """Passing the corrected ideal and nadir gives the same points as reading them from the problem."""
    problem = river_pollution_problem()
    from_problem = generate_group_reference_points(problem, RIVER_ASPIRATIONS, 200, seed=1)
    explicit = generate_group_reference_points(
        problem,
        RIVER_ASPIRATIONS,
        200,
        ideal=get_corrected_ideal(problem),
        nadir=get_corrected_nadir(problem),
        seed=1,
    )
    np.testing.assert_array_equal(from_problem, explicit)


@pytest.mark.gdmtools
def test_errors():
    """Missing ideal or nadir, missing symbols, a degenerate or wrongly signed ideal-nadir box raise ValueError."""
    dtlz = dtlz2(5, 3)
    river = river_pollution_problem()
    simple = simple_test_problem()

    with pytest.raises(ValueError):
        generate_group_reference_points(simple, [{o.symbol: 0.0 for o in simple.objectives}], 10)
    with pytest.raises(ValueError, match="missing"):
        generate_group_reference_points(dtlz, [{"f_1": 0.1, "f_2": 0.2}], 10)
    with pytest.raises(ValueError):
        generate_group_reference_points(dtlz, [], 10)
    with pytest.raises(ValueError, match="worse than the ideal"):
        generate_group_reference_points(
            dtlz,
            DTLZ2_ASPIRATIONS,
            10,
            ideal={"f_1": 0.0, "f_2": 0.0, "f_3": 0.0},
            nadir={"f_1": 1.0, "f_2": 0.0, "f_3": 1.0},
        )
    ideal, nadir = _original_ideal_nadir(river)
    with pytest.raises(ValueError, match="maximized"):
        generate_group_reference_points(river, RIVER_ASPIRATIONS, 10, ideal=ideal, nadir=nadir)


@pytest.mark.gdmtools
def test_diagonal_aspirations_collapse():
    """Aspirations differing only along the ideal-nadir direction give a single reference point."""
    problem = river_pollution_problem()
    ideal, nadir = _original_ideal_nadir(problem)
    a = RIVER_ASPIRATIONS[0]
    b = {s: a[s] + 0.3 * (nadir[s] - ideal[s]) for s in a}

    out = generate_group_reference_points(problem, [a, b, a], 100, seed=0)

    assert out.shape == (1, 5)
    assert np.abs(out.sum(axis=1) - 5).max() < 1e-12


@pytest.mark.gdmtools
def test_choose_reference_point_compatibility():
    """The output passes IPR's plane checks and choose_reference_point picks its points from it."""
    problem = dtlz2(5, 3)
    symbols = [o.symbol for o in problem.objectives]
    refp = generate_group_reference_points(problem, DTLZ2_ASPIRATIONS, 2000, seed=0)

    first, _ = choose_reference_point(refp, None)
    assert (refp == first).all(axis=1).any()

    solution = first / np.linalg.norm(first)  # a point on DTLZ2's front, as if found from the first reference point
    evaluated = [
        _EvaluatedPoint(
            reference_point=dict(zip(symbols, first.tolist(), strict=True)),
            targets=dict(zip(symbols, solution.tolist(), strict=True)),
            objectives=dict(zip(symbols, solution.tolist(), strict=True)),
        )
    ]
    second, bad_mask = choose_reference_point(refp, evaluated)

    assert (refp == second).all(axis=1).any()
    assert bad_mask.shape == (len(refp),)
