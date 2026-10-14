"""`SnapshotLogger`, the study's instrumentation.

A `Subscriber` attached to the publisher after `emo_constructor` builds the algorithm. It records
anytime traces (the population at fixed fractions of the evaluation budget) and per-generation
crossover mechanism statistics, which `xover.run` writes as one `.npz` per run.

Design constraints:

- **Cheap on the hot path.** Population messages store only a *reference*; the conversion to
  `float32` happens only when the evaluation counter crosses a checkpoint. Measured: NSGA-III on
  DTLZ2 at 30 000 evaluations runs in 1.79 s with the hook against 1.83 s without.
- **Plain arrays only** in the `.npz` -- `float32`/`int32`, no strings, no pickling -- so any numpy
  can read them. Everything that is not a number goes in the run's `.json`.
- **Ragged snapshots.** The final snapshot comes from the result and need not have the population's
  size, so `snap_F` is stored flat and sliced by the `cumsum` of `snap_n`.
- **Minimisation sense throughout.** Snapshots hold the `{symbol}_min` target columns, which the
  evaluator already sign-corrects. The objective senses are recorded in the JSON so original units
  can be recovered, but nothing downstream has to remember to flip a sign.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from desdeo.tools.message import (
    CrossoverMessageTopics,
    EvaluatorMessageTopics,
    GeneratorMessageTopics,
    Message,
    PolarsDataFrameMessage,
    SelectorMessageTopics,
)
from desdeo.tools.patterns import Subscriber

if TYPE_CHECKING:
    from collections.abc import Sequence

    from desdeo.problem import Problem
    from desdeo.tools.patterns import Publisher

SAMPLE_FRACTIONS: tuple[float, ...] = (0.02, *(k / 20 for k in range(1, 21)))
"""The default snapshot schedule, as fractions of the evaluation budget; `config["fractions"]` is
what the workflow passes. 0.02 is about as fine as a study can go: a snapshot can only be taken
between generations, and at a population of 91 and a budget of 10 000 one generation is 0.9 % of the
budget."""


DEFAULT_MECHANISM_STRIDE = 5
"""Record the crossover diagnostics every Nth generation. They are the only per-generation cost that
touches the decision matrix, and a trend is all that is needed."""


class SnapshotLogger(Subscriber):
    """Records anytime traces and crossover mechanism statistics from a running EA.

    Attach after `emo_constructor` and before calling the solver:

        solve, extras = emo_constructor(options, problem)
        logger = SnapshotLogger(problem=problem, publisher=extras.publisher, max_evaluations=budget)
        result = solve()
        logger.finalise(result.optimal_outputs)
        arrays = logger.arrays()
    """

    @property
    def interested_topics(self) -> Sequence[Any]:
        """The topics the logger listens to."""
        return [
            # Generation 0, which the generator publishes rather than the evaluator.
            GeneratorMessageTopics.VERBOSE_OUTPUTS,
            GeneratorMessageTopics.NEW_EVALUATIONS,
            # Every later generation.
            EvaluatorMessageTopics.NEW_EVALUATIONS,
            SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS,
            # The mechanism columns.
            CrossoverMessageTopics.PARENTS,
            CrossoverMessageTopics.OFFSPRINGS,
        ]

    @property
    def provided_topics(self) -> dict[int, Sequence[Any]]:
        """The logger publishes nothing; it only listens."""
        return {0: [], 1: [], 2: []}

    def __init__(
        self,
        *,
        problem: Problem,
        publisher: Publisher,
        max_evaluations: int,
        n_checkpoints: int | None = None,
        fractions: Sequence[float] = SAMPLE_FRACTIONS,
        mechanism_stride: int = DEFAULT_MECHANISM_STRIDE,
    ) -> None:
        """Attach a logger to a publisher.

        Args:
            problem: the problem being solved, for the objective symbols and the decision box.
            publisher: the publisher from `emo_constructor`'s extras.
            max_evaluations: the run's evaluation budget, used to place the checkpoints.
            n_checkpoints: place this many *log-spaced* checkpoints instead of using `fractions`.
                Only calibration wants this: it reads a budget off a quality curve and needs the curve
                denser than the grid does, particularly early.
            fractions: budget fractions at which to snapshot, the same list the indicators are
                scored at.
            mechanism_stride: record the crossover diagnostics every Nth generation.
        """
        # Verbosity 1 rather than 0: `Subscriber.notify` refuses verbosity 0, and although this
        # logger never calls notify, a zero here would be a trap for anyone who later did.
        super().__init__(publisher=publisher, verbosity=1)
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.mechanism_stride = mechanism_stride

        self.objective_symbols = [obj.symbol for obj in problem.objectives]
        self.target_symbols = [f"{symbol}_min" for symbol in self.objective_symbols]
        self.variable_symbols = [var.symbol for var in problem.get_flattened_variables()]
        self.maximise = [bool(obj.maximize) for obj in problem.objectives]

        lower = [var.lowerbound for var in problem.get_flattened_variables()]
        upper = [var.upperbound for var in problem.get_flattened_variables()]
        self._has_box = all(b is not None for b in (*lower, *upper))
        if self._has_box:
            self._lower = np.asarray(lower, dtype=float)
            self._upper = np.asarray(upper, dtype=float)
            span = self._upper - self._lower
            # The box diagonal, which normalises the distance statistics so they are comparable
            # across instances of different dimension and scale.
            self._diagonal = float(np.linalg.norm(span)) or 1.0
        else:
            self._lower = self._upper = None
            self._diagonal = 1.0

        self.checkpoints = (
            _log_checkpoints(max_evaluations, n_checkpoints)
            if n_checkpoints is not None
            else checkpoints(max_evaluations, fractions)
        )
        self._next_checkpoint = 0

        self.evaluations = 0
        self.generation = 0
        self._snapshot_due = False
        self._initial_pending = False
        self._pending_population: pl.DataFrame | None = None

        # Snapshots.
        self._snap_targets: list[np.ndarray] = []
        self._snap_evals: list[int] = []
        self._snap_gen: list[int] = []
        self._snap_n: list[int] = []
        self._initial_variables: np.ndarray | None = None

        # Per-generation trace, one row per generation, cheap enough to keep in full.
        self._gen_evals: list[int] = []
        self._gen_n: list[int] = []

        # Crossover mechanism statistics, subsampled by `mechanism_stride`.
        self._mech: dict[str, list[float]] = {
            "gen": [],
            "evals": [],
            "parent_separation": [],
            "offspring_step": [],
            "frac_on_bound": [],
            "parent_diameter": [],
        }
        self._pending_parents: np.ndarray | None = None

        # Behavioural message counts, which is how the runner asserts the instrumentation worked.
        self.counts: dict[str, int] = {
            "parents": 0,
            "offsprings": 0,
            "population": 0,
            "new_evaluations": 0,
        }

        publisher.auto_subscribe(self)

    # ----------------------------------------------------------------------------------------------
    # Subscriber protocol
    # ----------------------------------------------------------------------------------------------

    def state(self) -> Sequence[Message]:
        """The logger publishes nothing."""
        return []

    def update(self, message: Message) -> None:
        """Route one message. Kept branch-light: this runs on every message of every generation."""
        topic = message.topic

        if topic is EvaluatorMessageTopics.NEW_EVALUATIONS or topic is GeneratorMessageTopics.NEW_EVALUATIONS:
            self._on_new_evaluations(int(message.value))
            return

        if topic is SelectorMessageTopics.SELECTED_VERBOSE_OUTPUTS:
            self.counts["population"] += 1
            self.generation += 1
            self._on_population(message)
            return

        if topic is GeneratorMessageTopics.VERBOSE_OUTPUTS:
            self.counts["population"] += 1
            self._on_initial_population(message)
            return

        if topic is CrossoverMessageTopics.PARENTS:
            self.counts["parents"] += 1
            if self._recording_mechanism():
                self._pending_parents = _to_matrix(message, self.variable_symbols)
            return

        if topic is CrossoverMessageTopics.OFFSPRINGS:
            self.counts["offsprings"] += 1
            if self._recording_mechanism() and self._pending_parents is not None:
                self._record_mechanism(_to_matrix(message, self.variable_symbols))
            self._pending_parents = None

    # ----------------------------------------------------------------------------------------------
    # Handlers
    # ----------------------------------------------------------------------------------------------

    def _on_new_evaluations(self, new: int) -> None:
        """Accumulate exactly as `BaseTerminator.update` does, so checkpoints mean what it means."""
        self.counts["new_evaluations"] += 1
        self.evaluations += new
        while (
            self._next_checkpoint < len(self.checkpoints)
            and self.evaluations >= self.checkpoints[self._next_checkpoint]
        ):
            self._next_checkpoint += 1
            self._snapshot_due = True

        # The generator publishes its frame *before* its NEW_EVALUATIONS; the evaluator publishes
        # them the other way round. So the initial population is already in hand here, and waiting
        # for a later population message would misattribute it to generation 1.
        if self._initial_pending:
            self._initial_pending = False
            self._snapshot_due = False
            self._take_snapshot()

    def _on_initial_population(self, message: Message) -> None:
        """Hold generation 0 until its evaluation count arrives."""
        frame = _frame(message)
        self._pending_population = frame
        self._initial_pending = True
        if self._initial_variables is None:
            present = [s for s in self.variable_symbols if s in frame.columns]
            if present:
                self._initial_variables = frame.select(present).to_numpy().astype(np.float32)

    def _on_population(self, message: Message) -> None:
        """Store a reference, and convert only if a checkpoint is owed."""
        self._pending_population = _frame(message)
        self._gen_evals.append(self.evaluations)
        self._gen_n.append(self._pending_population.height)
        if self._snapshot_due:
            self._snapshot_due = False
            self._take_snapshot()

    def _take_snapshot(self) -> None:
        """Convert the held population to `float32` and record it."""
        frame = self._pending_population
        if frame is None:
            return
        present = [s for s in self.target_symbols if s in frame.columns]
        if len(present) != len(self.target_symbols):
            # Fall back to the raw objective columns, sign-corrected here, so a frame that carries
            # objectives but not targets still yields a minimisation-sense snapshot.
            raw = frame.select(self.objective_symbols).to_numpy().astype(np.float32)
            signs = np.array([-1.0 if m else 1.0 for m in self.maximise], dtype=np.float32)
            targets = raw * signs
        else:
            targets = frame.select(present).to_numpy().astype(np.float32)
        self._snap_targets.append(targets)
        self._snap_n.append(int(targets.shape[0]))
        self._snap_evals.append(self.evaluations)
        self._snap_gen.append(self.generation)

    def _recording_mechanism(self) -> bool:
        return self.generation % self.mechanism_stride == 0

    def _record_mechanism(self, offspring: np.ndarray) -> None:
        """Aggregate crossover diagnostics that do not depend on parent-offspring row pairing.

        `BaseCrossover.get_parents` shuffles the mating pool and the module's own TODO records that
        offspring row order is not consistent across operators -- some vstack their children, the
        untruncated SBX and `LocalCrossover` interleave them. A naive `offspring[0::2]` against
        `population[0::2]` measures nothing, and returns an identical value for every operator, which
        is the signature of a destroyed pairing rather than of no effect.

        So the statistics here are pairing-free by construction. Mated *pairs* are still well defined
        -- consecutive rows of the PARENTS message are the mating order -- so parent separation is
        exact; offspring displacement is measured to the nearest parent rather than to an assumed
        partner.
        """
        parents = self._pending_parents
        if parents is None or parents.size == 0 or offspring.size == 0:
            return

        pairs = parents[: len(parents) - len(parents) % 2]
        if len(pairs) >= 2:  # noqa: PLR2004
            separation = np.linalg.norm(pairs[0::2] - pairs[1::2], axis=1).mean() / self._diagonal
        else:
            separation = float("nan")

        # Distance from every offspring to its nearest parent, which needs no pairing assumption.
        step = _nearest_distance(offspring, parents).mean() / self._diagonal

        # DESDEO clips every crossover's output onto the box, so an out-of-bounds *rate* is
        # identically zero and cannot separate the truncated from the untruncated arm. What survives
        # clipping is the footprint it leaves: gene values sitting exactly on a bound. That is the
        # quantity that distinguishes `sbx` from `sbxt`.
        if self._has_box:
            on_bound = np.isclose(offspring, self._lower) | np.isclose(offspring, self._upper)
            frac_on_bound = float(on_bound.mean())
        else:
            frac_on_bound = float("nan")

        self._mech["gen"].append(self.generation)
        self._mech["evals"].append(self.evaluations)
        self._mech["parent_separation"].append(float(separation))
        self._mech["offspring_step"].append(float(step))
        self._mech["frac_on_bound"].append(frac_on_bound)
        self._mech["parent_diameter"].append(float(_diameter(parents) / self._diagonal))

    # ----------------------------------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------------------------------

    def finalise(self, final_outputs: pl.DataFrame | None = None) -> None:
        """Record the end of the run.

        The terminator overshoots by up to one iteration and by a different amount per algorithm, so
        the last checkpoint rarely lands exactly on the budget. Taking a final snapshot from
        `EMOResult` makes every run comparable at its own endpoint as well as at budget fractions.
        """
        if final_outputs is not None:
            self._pending_population = final_outputs
            self._take_snapshot()

    def arrays(self) -> dict[str, np.ndarray]:
        """The `.npz` payload: plain `float32`/`int32` arrays, nothing that needs pickling."""
        n_obj = len(self.objective_symbols)
        targets = (
            np.concatenate(self._snap_targets, axis=0) if self._snap_targets else np.zeros((0, n_obj), dtype=np.float32)
        )
        out = {
            "snap_F": targets.astype(np.float32),
            "snap_n": np.asarray(self._snap_n, dtype=np.int32),
            "snap_evals": np.asarray(self._snap_evals, dtype=np.int32),
            "snap_gen": np.asarray(self._snap_gen, dtype=np.int32),
            "gen_evals": np.asarray(self._gen_evals, dtype=np.int32),
            "gen_n": np.asarray(self._gen_n, dtype=np.int32),
            "checkpoints": np.asarray(self.checkpoints, dtype=np.int32),
        }
        if self._initial_variables is not None:
            out["snap_X0"] = self._initial_variables.astype(np.float32)
        for key, values in self._mech.items():
            dtype = np.int32 if key in {"gen", "evals"} else np.float32
            out[f"mech_{key}"] = np.asarray(values, dtype=dtype)
        return out

    def metadata(self) -> dict[str, Any]:
        """Everything that is not a number, for the sibling `.json`."""
        return {
            "objective_symbols": self.objective_symbols,
            "target_symbols": self.target_symbols,
            "variable_symbols": self.variable_symbols,
            "maximise": self.maximise,
            "snapshot_sense": "minimisation",
            "max_evaluations": self.max_evaluations,
            "final_evaluations": self.evaluations,
            "generations": self.generation,
            "n_snapshots": len(self._snap_n),
            "mechanism_stride": self.mechanism_stride,
            "message_counts": dict(self.counts),
        }

    def check_messages_arrived(self) -> None:
        """Assert the instrumentation actually received data. Raises `InstrumentationError`.

        Behavioural rather than registry-based on purpose. `Publisher.check_consistency()` reports
        `False` for the composite crossover, because `emo_constructor` registers only the composite
        -- whose `provided_topics` is empty -- while its sub-operators notify through the same
        publisher. The messages do flow. Checking that they arrived tests the thing that matters and
        covers every arm uniformly, so no framework change is needed and `check_consistency()` stays
        a recorded provenance column rather than an abort condition.
        """
        if self.counts["population"] == 0:
            msg = "no population messages arrived; is the logger attached and verbosity 2?"
            raise InstrumentationError(msg)
        if self.counts["new_evaluations"] == 0:
            msg = "no evaluation-count messages arrived; checkpoints cannot be placed"
            raise InstrumentationError(msg)
        if not self._snap_n:
            msg = "no snapshots were taken"
            raise InstrumentationError(msg)
        if self.counts["parents"] == 0 or self.counts["offsprings"] == 0:
            msg = (
                f"crossover messages missing: {self.counts['parents']} PARENTS and "
                f"{self.counts['offsprings']} OFFSPRINGS; the mechanism columns would be empty"
            )
            raise InstrumentationError(msg)


class InstrumentationError(RuntimeError):
    """The logger did not receive the messages a run should have produced."""


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def checkpoints(max_evaluations: int, fractions: Sequence[float] = SAMPLE_FRACTIONS) -> list[int]:
    """Evaluation counts at which to snapshot, one per requested fraction.

    Deduplicated and sorted, so a budget small enough that two fractions round together yields one
    checkpoint rather than two identical ones. `xover.indicators.score_run` still picks the snapshot
    nearest each fraction, because a checkpoint is crossed part way through a generation and the
    realised evaluation count overshoots it by up to a population; what this schedule guarantees is
    that no two fractions can be served by the same snapshot at a realistic budget.
    """
    return sorted({max(1, round(fraction * max_evaluations)) for fraction in fractions})


def _log_checkpoints(max_evaluations: int, n: int) -> list[int]:
    """Log-spaced evaluation checkpoints, ending at the budget.

    Log spacing rather than linear because the interesting part of an anytime trace is the early
    convergence, and a linear grid spends most of its points on a flat tail.
    """
    start = max(1, min(100, max_evaluations))
    raw = np.geomspace(start, max_evaluations, num=max(2, n))
    return sorted({round(float(value)) for value in raw} | {max_evaluations})


def _frame(message: Message) -> pl.DataFrame:
    """The DataFrame carried by a population message."""
    if isinstance(message, PolarsDataFrameMessage):
        return message.value
    value = message.value
    if isinstance(value, pl.DataFrame):
        return value
    msg = f"expected a population DataFrame, got {type(value).__name__}"
    raise TypeError(msg)


def _to_matrix(message: Message, symbols: list[str]) -> np.ndarray:
    """The decision matrix carried by a crossover message."""
    value = message.value
    if isinstance(value, pl.DataFrame):
        present = [s for s in symbols if s in value.columns]
        return value.select(present or value.columns).to_numpy().astype(float)
    return np.atleast_2d(np.asarray(value, dtype=float))


def _nearest_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from every row of `query` to its nearest row of `reference`."""
    # Chunked so a large mating pool does not allocate an n-by-m matrix all at once.
    out = np.empty(len(query), dtype=float)
    chunk = max(1, 4096 // max(1, len(reference)))
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        out[start : start + chunk] = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    return out


def _diameter(points: np.ndarray, cap: int = 256) -> float:
    """The spread of a point set, as the diagonal of its bounding box.

    The bounding box rather than the maximum pairwise distance: it is linear in the number of points
    where the exact diameter is quadratic, and this is on the per-generation path.
    """
    if len(points) == 0:
        return float("nan")
    sample = points[:cap]
    return float(np.linalg.norm(sample.max(axis=0) - sample.min(axis=0)))


def unpack_snapshots(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Slice the flat `snap_F` back into one matrix per checkpoint, using the `cumsum` of `snap_n`."""
    counts = np.asarray(arrays["snap_n"], dtype=int)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    flat = arrays["snap_F"]
    return [flat[bounds[i] : bounds[i + 1]] for i in range(len(counts))]


MECHANISM_COLUMNS = ("parent_separation", "offspring_step", "frac_on_bound", "parent_diameter")
"""The per-generation mechanism series `_record_mechanism` writes."""


def mechanism_summary(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce one run's per-generation mechanism series to scalars: mean and final value of each."""
    if "mech_gen" not in arrays or len(arrays["mech_gen"]) == 0:
        return {}

    out: dict[str, float] = {}
    for column in MECHANISM_COLUMNS:
        values = arrays.get(f"mech_{column}")
        if values is None or len(values) == 0:
            continue
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        out[column] = float(finite.mean())
        out[f"{column}_final"] = float(finite[-1])
    return out | _diversity_decay(arrays)


def _diversity_decay(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Half-life of the parent-set diameter, in generations.

    How fast an operator burns through the population's spread in decision space. A midpoint-biased
    operator should show a shorter half-life than a parent-centric one.
    """
    diameter = arrays.get("mech_parent_diameter")
    generations = arrays.get("mech_gen")
    if diameter is None or generations is None or len(diameter) < 3:  # noqa: PLR2004
        return {}
    finite = np.isfinite(diameter)
    diameter, generations = diameter[finite], generations[finite]
    if len(diameter) < 3 or diameter[0] <= 0:  # noqa: PLR2004
        return {}
    half = diameter[0] / 2.0
    below = np.flatnonzero(diameter <= half)
    return {
        "diversity_half_life": float(generations[below[0]]) if len(below) else float("nan"),
        "diversity_retained": float(diameter[-1] / diameter[0]),
    }


def load_run(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Read one run's `.npz`."""
    with np.load(npz_path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
