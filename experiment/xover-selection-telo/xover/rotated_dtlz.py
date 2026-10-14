"""The rotated DTLZ run: does the mixing tier survive the published rotated DTLZ problems?

This replaces run A's first design, which rotated DTLZ3's distance variables by a 45-degree Givens
rotation of the study's own making. The problems are now the published rotated DTLZ problems, followed
exactly:

    Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu and S. Tiwari. Multiobjective optimization test
    instances for the CEC 2009 special session and competition. Technical Report CES-487, School of
    Computer Science and Electronic Engineering, University of Essex, 2009 (draft of 20 April 2009).
    Unconstrained problems 11 and 12, R2_DTLZ2_M5 and R2_DTLZ3_M5, known as UF11 and UF12.

The construction, extending the box and rotating the search space, is the CEC 2007 suite's, whose
R_DTLZ2 the 2009 report builds on and cites as its reference [4]:

    V. L. Huang, A. K. Qin, K. Deb, E. Zitzler, P. N. Suganthan, J. J. Liang, M. Preuss and S. Huband.
    Problem definitions for performance assessment of multi-objective optimization algorithms.
    Technical Report, Nanyang Technological University, Singapore, 2007.

What the citation fixes, and this module keeps (report pp. 12-15):

- Five objectives and 30 variables, the size the report's data files cover.
- z = M x with the published M, search box and scale factors lambda, in `rotated_dtlz_data.py`.
- A coordinate outside [0, 1] is folded back, z' = -lambda z below 0 and 1 - lambda (z - 1) above 1,
  and penalised by its distance from the interval.
- Each objective is DTLZ2's or DTLZ3's own, computed on z', shifted by one and stretched by
  S = 2 / (1 + exp(-p)), where p is the norm of the penalties on the variables that objective reads.
  Where those all lie in [0, 1], S is 1, so on the Pareto set the objectives lie on DTLZ's sphere,
  moved to (1, ..., 1).

Two places where the report's pages and its reference code differ, settled in favour of the code,
because the code is what the competition evaluated and what published UF11 and UF12 results were
computed with:

- The problem pages print z' = lambda z above 1. The general definition (p. 13) and the code both use
  1 - lambda (z - 1).
- UF11's M is not orthogonal. The report calls M an "orthogonal matrix, with condition number=1", and
  UF12's is (condition number 1.0003). UF11's fourth row has norm 0.79 and ends in the same six entries
  as its fifth, which reads as a slip in the competition's data file; the condition number is 5.19. The
  published matrix is kept, since it is UF11 as everyone has run it, and `tests/test_problems.py` pins
  the defect so that nobody corrects it into a different problem. UF11 is a rotation with some shear,
  UF12 a rotation.

M acts on the first 15 variables and is the identity on the other 15. Its block mixes the four position
variables with eleven distance variables, so the Pareto set, z_5 = ... = z_30 = 0.5, is an affine set
in x rather than a product of per-variable sets.

The published box does not hold all of that set, which the report does not say. Of Pareto-optimal
points sampled uniformly over z_1..z_4, 71% lie inside it for UF11 and 28% for UF12. The controls reach
97.6% of it, because the box stops x_4 at 0.976. Reference fronts are built from the runs, as for every
instance in the study, so the scoring needs nothing different; but a pair differs in which part of the
sphere a solver can reach as well as in the rotation.

The control is the study's, not the citation's: the same construction with M = I, keeping the box,
lambda, the penalty and the shift, so that each pair differs in the rotation alone. Plain DTLZ would not
do, because its box and its objective offset differ as well.

The instances are generated from `config["suites"]["rotated_dtlz"]` by `xover.problems`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from desdeo.problem import Objective, Problem, Simulator, Url, Variable, VariableTypeEnum
from desdeo.problem.external import register_provider
from desdeo.problem.external.core import ExternalProblemInfo, ExternalProblemParams, Provider
from pydantic import Field

from xover import rotated_dtlz_data as data

PROVIDER_NAME = "xovercec09"
LOCATOR = f"desdeo://external/{PROVIDER_NAME}/evaluate"

N_VAR = 30
N_OBJ = 5
"""The size the report defines R2_DTLZ2_M5 and R2_DTLZ3_M5 at, and the only one its data files cover."""

VARIABLES = [f"x_{i}" for i in range(1, N_VAR + 1)]
OBJECTIVES = [f"f_{i}" for i in range(1, N_OBJ + 1)]

PUBLISHED: dict[str, dict[str, Any]] = {
    "dtlz2": {
        "label": "UF11 (R2_DTLZ2_M5)",
        "matrix": data.DTLZ2_MATRIX,
        "lower": data.DTLZ2_LOWER,
        "upper": data.DTLZ2_UPPER,
        "scale": data.DTLZ2_LAMBDA,
    },
    "dtlz3": {
        "label": "UF12 (R2_DTLZ3_M5)",
        "matrix": data.DTLZ3_MATRIX,
        "lower": data.DTLZ3_LOWER,
        "upper": data.DTLZ3_UPPER,
        "scale": data.DTLZ3_LAMBDA,
    },
}
"""Each problem's published data, keyed by the DTLZ problem it rotates."""


class RotatedDTLZParams(ExternalProblemParams):
    """Which published problem, and whether its rotation is applied."""

    base: str = Field(description="'dtlz2' for UF11, 'dtlz3' for UF12.")
    rotated: bool = Field(description="True applies the published M; False is the control, M = I.")


def arrays(base: str, *, rotated: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """M, the box's lower and upper bounds, and lambda, for one instance."""
    entry = PUBLISHED[base]
    matrix = np.asarray(entry["matrix"], dtype=float) if rotated else np.eye(N_VAR)
    return (
        matrix,
        np.asarray(entry["lower"], dtype=float),
        np.asarray(entry["upper"], dtype=float),
        np.asarray(entry["scale"], dtype=float),
    )


def transform(x: np.ndarray, matrix: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute z = M x per row, fold it back into [0, 1], and penalise each coordinate that left."""
    z = x @ matrix.T
    below, above = z < 0.0, z > 1.0
    folded = np.where(below, -scale * z, np.where(above, 1.0 - scale * (z - 1.0), z))
    penalty = np.where(below, -z, np.where(above, z - 1.0, 0.0))
    return folded, penalty


def penalty_norms(penalty: np.ndarray, n_obj: int) -> np.ndarray:
    """Each objective's p, the norm of the penalties on the variables it reads.

    That is every distance variable, and the position variables the objective reads: the first m - i,
    plus, from the second objective on, the one its sine takes.
    """
    distance = np.sum(penalty[:, n_obj - 1 :] ** 2, axis=1)
    out = np.empty((penalty.shape[0], n_obj))
    for m in range(1, n_obj + 1):
        read = list(range(n_obj - m)) + ([n_obj - m] if m > 1 else [])
        out[:, m - 1] = np.sqrt(distance + np.sum(penalty[:, read] ** 2, axis=1))
    return out


def dtlz(base: str, folded: np.ndarray, n_obj: int) -> np.ndarray:
    """DTLZ2's or DTLZ3's objectives at the folded coordinates."""
    distance = folded[:, n_obj - 1 :] - 0.5
    if base == "dtlz2":
        g = np.sum(distance**2, axis=1)
    else:
        g = 100.0 * (distance.shape[1] + np.sum(distance**2 - np.cos(20.0 * math.pi * distance), axis=1))
    angles = folded[:, : n_obj - 1] * math.pi / 2.0
    out = np.empty((folded.shape[0], n_obj))
    for m in range(1, n_obj + 1):
        value = (1.0 + g) * np.prod(np.cos(angles[:, : n_obj - m]), axis=1)
        if m > 1:
            value = value * np.sin(angles[:, n_obj - m])
        out[:, m - 1] = value
    return out


def evaluate_rows(x: np.ndarray, base: str, *, rotated: bool) -> np.ndarray:
    """The objectives of each row of `x`, as the reference code computes them."""
    matrix, _, _, scale = arrays(base, rotated=rotated)
    folded, penalty = transform(x, matrix, scale)
    stretch = 2.0 / (1.0 + np.exp(-penalty_norms(penalty, N_OBJ)))
    return stretch * (dtlz(base, folded, N_OBJ) + 1.0)


class RotatedDTLZProvider(Provider):
    """Evaluates UF11, UF12 and their unrotated controls."""

    def info(self, params: RotatedDTLZParams | dict[str, Any]) -> ExternalProblemInfo:
        """The published box, and the front's ideal and nadir, (1, ..., 1) and (2, ..., 2)."""
        raw = params if isinstance(params, dict) else params.model_dump()
        _, lower, upper, _ = arrays(raw["base"], rotated=raw["rotated"])
        label = PUBLISHED[raw["base"]]["label"]
        return ExternalProblemInfo(
            name=label if raw["rotated"] else f"{label}, unrotated control",
            description=(
                f"{label} from the CEC 2009 competition (Zhang et al., CES-487)"
                + ("." if raw["rotated"] else ", with M = I in place of the published rotation.")
            ),
            variable_symbols=VARIABLES,
            variable_names=None,
            variable_type=None,
            variable_lower_bounds=dict(zip(VARIABLES, lower.tolist(), strict=True)),
            variable_upper_bounds=dict(zip(VARIABLES, upper.tolist(), strict=True)),
            objective_symbols=OBJECTIVES,
            objective_names=None,
            objective_maximize=None,
            ideal_point=dict.fromkeys(OBJECTIVES, 1.0),
            nadir_point=dict.fromkeys(OBJECTIVES, 2.0),
            # Declared without defaults in DESDEO's model, so they are passed even though UF11 and UF12
            # are unconstrained.
            constraint_symbols=None,
            constraint_names=None,
            constraint_types=None,
        )

    def evaluate(self, xs: dict[str, Any], params: RotatedDTLZParams | dict[str, Any]) -> dict[str, list[float]]:
        """Evaluate every point in `xs`, given as {variable symbol: [values]}."""
        raw = params if isinstance(params, dict) else params.model_dump()
        x = np.atleast_2d(np.column_stack([np.asarray(xs[s], dtype=float) for s in VARIABLES]))
        values = evaluate_rows(x, raw["base"], rotated=raw["rotated"])
        return dict(zip(OBJECTIVES, values.T.tolist(), strict=True))


register_provider(PROVIDER_NAME, RotatedDTLZProvider(), overwrite=True)


def build_problem(spec: dict[str, Any]) -> Problem:
    """The DESDEO problem for one instance, from its spec alone, as a spawned worker has to build it."""
    params = RotatedDTLZParams(base=spec["base"], rotated=spec["rotated"])
    info = RotatedDTLZProvider().info(params)
    url = Url(url=LOCATOR)
    variables = [
        Variable(
            name=symbol,
            symbol=symbol,
            lowerbound=info.variable_lower_bounds[symbol],
            upperbound=info.variable_upper_bounds[symbol],
            variable_type=VariableTypeEnum.real,
        )
        for symbol in info.variable_symbols
    ]
    objectives = [
        Objective(name=symbol, symbol=symbol, simulator_path=url, objective_type="simulator", maximize=False)
        for symbol in info.objective_symbols
    ]
    problem = Problem(
        name=info.name,
        description=info.description,
        variables=variables,
        objectives=objectives,
        simulators=[
            Simulator(
                name="xovercec09_sim",
                symbol="xovercec09_sim",
                url=url,
                parameter_options=params.model_dump(exclude_none=True),
            )
        ],
    )
    return problem.update_ideal_and_nadir(info.ideal_point, info.nadir_point)


BASES: tuple[str, ...] = ("dtlz2", "dtlz3")
"""UF11 and UF12: a spherical g and a Rastrigin one, which is what Salomon's objection is about."""


def slug(base: str, *, rotated: bool) -> str:
    """`r2_dtlz2_m5_n30` is UF11 itself; `r2_dtlz2_unrotated_m5_n30` is its control."""
    return f"r2_{base}{'' if rotated else '_unrotated'}_m{N_OBJ}_n{N_VAR}"
