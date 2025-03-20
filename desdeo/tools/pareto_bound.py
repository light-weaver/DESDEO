"""Code for the Bounded representation of the Pareto front."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull

from .GenerateReferencePoints import get_rotation_matrix


def project(points: np.ndarray) -> np.ndarray:
    """Project a point or set of points to the reference plane.

    The points are assumed to be normalized to an approximate (and unchanging) ideal (0, 0, ..., 0) and
    nadir (1, 1, ..., 1) points. The direction of projection is in the nadir-ideal direction.
    The plane is assumed to pass through the nadir, thus having the equation
    sum_{i=1}^{k} f_i = k, where k is the number of objectives.

    Args:
        points (np.ndarray): The point or points to be projected.

    Returns:
        np.ndarray: The projected point/points.
    """
    points = np.atleast_2d(points)
    current_norm = points.sum(axis=1)
    final_norm = points.shape[1]
    return points + ((final_norm - current_norm) / final_norm)[:, None]


@dataclass
class _Transformations:
    unit_simplex: np.ndarray
    """Equations of the unit simplex, projected down to (k-1) dimensions."""
    inverted_unit_simplex: np.ndarray
    """Equations of the inverted unit simplex, projected down to (k-1) dimensions."""
    transformation_matrix: Callable[[np.ndarray], np.ndarray]
    """A function that transforms points anywhere in the k-dimensional objective space to the (k-1)-dimensional
    compressed space."""
    inverse_transformation_matrix: Callable[[np.ndarray], np.ndarray]
    """A function that transforms points in the (k-1)-dimensional compressed space to the k-dimensional objective space
    (on the reference plane)."""


def get_transformations(num_dims: int) -> _Transformations:
    """Get unit simplex and inverted unit simplex in the (k-1) dimensional compressed space.

    Args:
        num_dims (int): The number of objectives.

    Returns:
        Transformations: A dataclass containing the unit simplex, inverted unit simplex, transformation matrix and
        inverse transformation matrix.
    """
    unit_simplex = np.eye(num_dims)
    inverted_unit_simplex = np.ones((num_dims, num_dims)) - unit_simplex
    # project simplexes to reference plane
    unit_simplex = project(unit_simplex)
    inverted_unit_simplex = project(inverted_unit_simplex)
    # Rotate the diagonal vector to the last axis
    first_rotation_matrix = get_rotation_matrix((np.ones(num_dims)), np.array([0] * (num_dims - 1) + [1]))
    # Apply the rotation matrix to the unit simplexes
    unit_simplex = unit_simplex @ first_rotation_matrix.T
    inverted_unit_simplex = inverted_unit_simplex @ first_rotation_matrix.T
    # Remove the last axis
    unit_simplex = unit_simplex[:, :-1]
    inverted_unit_simplex = inverted_unit_simplex[:, :-1]
    intercept = unit_simplex[0, -1]
    # Rotate and scale the  unit simplex such that the first vertex is at (0, 0, .., 1)
    second_rotation_matrix = get_rotation_matrix(unit_simplex[0], np.array([0] * (num_dims - 2) + [1]))
    unit_simplex = unit_simplex @ second_rotation_matrix.T
    scale = np.abs(unit_simplex[0, -1])
    unit_simplex = unit_simplex / scale

    inverted_unit_simplex = inverted_unit_simplex @ second_rotation_matrix.T
    inverted_unit_simplex = inverted_unit_simplex / scale

    inv_first = np.linalg.inv(first_rotation_matrix.T)
    inv_second = np.linalg.inv(second_rotation_matrix.T)

    def transformation_matrix(x: np.ndarray) -> np.ndarray:
        """Transform points anywhere in the k-dimensional objective space to the compressed space.

        Args:
            x (np.ndarray): The points to be transformed.

        Returns:
            np.ndarray: The transformed points. Has one less dimension than the input points.
        """
        return (x @ first_rotation_matrix.T)[:, :-1] @ second_rotation_matrix.T / scale

    def inverse_transformation_matrix(x: np.ndarray) -> np.ndarray:
        """Transform points in the compressed space to the reference plane in the k-dimensional objective space.

        Args:
            x (np.ndarray): The points to be transformed. The 1-norm must be equal to the number of objectives.

        Returns:
            np.ndarray: The transformed points. Has one more dimension than the input
        """
        temp = x @ inv_second * scale
        temp = np.concatenate((temp, np.ones((temp.shape[0], 1)) * intercept), axis=1)
        temp = temp @ inv_first
        return project(temp)

    return _Transformations(
        unit_simplex=unit_simplex,
        inverted_unit_simplex=inverted_unit_simplex,
        transformation_matrix=transformation_matrix,
        inverse_transformation_matrix=inverse_transformation_matrix,
    )


@njit
def hull_distance(hull_eqn: np.ndarray, point: np.ndarray) -> float:
    """Get the distance of a point from the convex hull.

    Assumes that the hull is unit/inverted unit simplex. Assumes that the distance at the centre of the hull is zero.

    Related stackoverflow question:
    https://stackoverflow.com/questions/41000123/computing-the-distance-to-a-convex-hull.

    Args:
        hull_eqn (np.ndarray): The equations of the convex hull generated by scipy.spatial.ConvexHull.
        point (np.ndarray): The point to get the distance from, assumes that the center of the hull is at the origin.

    Returns:
        float: The distance of the point from the convex hull.
    """
    max_ = -np.inf
    num_eqns = hull_eqn.shape[0]
    max_centre_ = -np.inf
    for i in range(num_eqns):
        current_slopes = hull_eqn[i, :-1]
        current_intercept = hull_eqn[i, -1]
        temp = current_slopes.dot(point) + current_intercept
        temp_centre = current_slopes.dot(np.zeros(point.shape)) + current_intercept
        if temp_centre > max_centre_:
            max_centre_ = temp_centre
        if temp > max_:
            max_ = temp
    return max_ - max_centre_


@njit
def hull_distance_batch(hull_eqn, points):
    """Get the distance of a set of points from the convex hull. See `hull_distance` method for more details."""
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = hull_distance(hull_eqn, points[i])
    return distances


def get_scale_factor(hull_eqns: np.ndarray, transform_matrix: Callable) -> float:
    """Get the scale factor that defines the slope of the dominance cone."""
    n_dims = hull_eqns.shape[1]
    # some random solution
    soln = np.random.random((1, n_dims))
    # central reference point
    refp = np.ones((1, n_dims))
    soln_projected = project(soln)
    asf_val1 = np.max(soln - refp)
    asf_val2 = np.max(soln - soln_projected)
    # asf_val1 = ReferencePointASF(np.ones((1, n_dims)), np.ones((1, n_dims)), np.zeros((1, n_dims)))(soln, refp)
    # asf_val2 = ReferencePointASF(np.ones((1, n_dims)), np.ones((1, n_dims)), np.zeros((1, n_dims)))(
    #    soln, soln_projected
    # )
    y_delta = np.abs(asf_val2 - asf_val1)
    distance = hull_distance(hull_eqns, transform_matrix(soln_projected)[0] - transform_matrix(refp)[0])
    return y_delta / distance


def optimistic_bound(
    known_solutions: np.ndarray,
    ideal: np.ndarray,
    scale_factor: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the optimistic bound of the Pareto front.

    Args:
        known_solutions (np.ndarray): The currently known solutions. This includes fake solutions generated for pruning.
        ideal (np.ndarray): The ideal point.
        scale_factor (float, optional): The scale factor that defines the slope of the dominance cone. Defaults to 1.0.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The distance function. Maps reference points (on the reference plane) to
            their distance from the optimistic bound.
    """
    transformations = get_transformations(known_solutions.shape[1])
    unit_sim, inv_sim, transformation_matrix = (
        transformations.unit_simplex,
        transformations.inverted_unit_simplex,
        transformations.transformation_matrix,
    )
    optimistic_hull_eqns = ConvexHull(inv_sim).equations
    pessimistic_hull_eqns = ConvexHull(unit_sim).equations

    points_proj = transformation_matrix(known_solutions)
    ideal = np.atleast_2d(ideal + 1e-10)
    ideal_proj = transformation_matrix(ideal)
    asf_vals = np.max(known_solutions - project(known_solutions), axis=1)
    ideal_asf = np.max(ideal - project(ideal), axis=1)
    scale = get_scale_factor(optimistic_hull_eqns, transformation_matrix)
    scale = scale_factor * scale

    @njit
    def distance(projected_reference_points: np.ndarray) -> np.ndarray:
        distances = np.zeros((projected_reference_points.shape[0], points_proj.shape[0] + 1))
        for i in range(points_proj.shape[0]):
            distances[:, i] = asf_vals[i] - scale * hull_distance_batch(
                optimistic_hull_eqns, np.atleast_2d(projected_reference_points - points_proj[i])
            )
        distances[:, -1] = ideal_asf + scale * hull_distance_batch(
            pessimistic_hull_eqns, np.atleast_2d(projected_reference_points - ideal_proj)
        )
        actual_distances = np.zeros(projected_reference_points.shape[0])
        for i in range(projected_reference_points.shape[0]):
            actual_distances[i] = distances[i].max()
        return actual_distances

    def optimistic_projection(reference_points: np.ndarray) -> np.ndarray:
        """Get the distance of the reference points from the optimistic bound."""
        return distance(transformation_matrix(reference_points))[:, None]

    return optimistic_projection


def pessimistic_bound(
    known_solutions: np.ndarray, nadir: np.ndarray, scale_factor: float = 1.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the pessimistic bound of the Pareto front.

    Args:
        known_solutions (np.ndarray): The currently known solutions. This includes fake solutions generated for pruning.
        nadir (np.ndarray): The nadir point.
        scale_factor (float, optional): The scale factor defining the slope of the dominance cone. Defaults to 1.0.

    Returns:
        Callable[[np.ndarray], np.ndarray]: The distance function. Maps reference points (on the reference plane) to
            their distance from the pessimistic bound.
    """
    transformations = get_transformations(known_solutions.shape[1])
    unit_sim, inv_sim, transformation_matrix = (
        transformations.unit_simplex,
        transformations.inverted_unit_simplex,
        transformations.transformation_matrix,
    )
    optimistic_hull_eqns = ConvexHull(inv_sim).equations
    pessimistic_hull_eqns = ConvexHull(unit_sim).equations

    points_proj = transformation_matrix(known_solutions)
    nadir = np.atleast_2d(nadir)
    nadir_proj = transformation_matrix(nadir)
    asf_vals = np.max(known_solutions - project(known_solutions), axis=1)
    nadir_asf = np.max(nadir - project(nadir), axis=1)  # Should technically be zero
    scale = get_scale_factor(optimistic_hull_eqns, transformation_matrix)
    scale = scale_factor * scale

    @njit
    def distance(projected_reference_points):
        distances = np.zeros((projected_reference_points.shape[0], points_proj.shape[0] + 1))
        for i in range(points_proj.shape[0]):
            distances[:, i] = asf_vals[i] + scale * hull_distance_batch(
                pessimistic_hull_eqns, np.atleast_2d(projected_reference_points - points_proj[i])
            )
        distances[:, -1] = nadir_asf - scale * hull_distance_batch(
            optimistic_hull_eqns, np.atleast_2d(projected_reference_points - nadir_proj)
        )
        actual_distances = np.zeros(projected_reference_points.shape[0])
        for i in range(projected_reference_points.shape[0]):
            actual_distances[i] = distances[i].min()
        return actual_distances

    def pessimistic_projection(reference_points: np.ndarray) -> np.ndarray:
        """Get the distance of the reference points from the pessimistic bound."""
        return distance(transformation_matrix(reference_points))[:, None]

    return pessimistic_projection
