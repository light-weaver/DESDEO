"""The snapshot schedule and the dump of one short run."""

import numpy as np

from xover.design import Block
from xover.logger import checkpoints, mechanism_summary, unpack_snapshots
from xover.run import run_cell


def test_checkpoints_follow_the_fractions():
    assert checkpoints(10_000, [0.02, 0.5, 1.0]) == [200, 5000, 10_000]
    # Two fractions rounding to one evaluation count give one checkpoint, not two.
    assert checkpoints(10, [0.01, 0.02]) == [1]


def test_one_run_dumps_every_fraction(config):
    # At 10 000 evaluations and 91 points per generation no two fractions fall in one generation.
    arrays, record = run_cell(config, Block("dtlz2_m3_n12", "nsga3", "sbx_default", "polynomial"), 0, 10_000)
    snapshots = unpack_snapshots(arrays)
    # The initial population, one per checkpoint, and the final population.
    assert len(snapshots) == len(arrays["snap_evals"]) == 1 + len(config["fractions"]) + 1
    assert all(s.shape == (91, 3) for s in snapshots)
    assert arrays["snap_F"].dtype == np.float32
    assert arrays["snap_X0"].shape == (91, 12)
    assert record["provenance"]["versions"]["desdeo"] != "unknown"
    assert set(mechanism_summary(arrays)) >= {"parent_separation", "offspring_step", "diversity_retained"}


def test_same_seed_same_run_different_seed_different_run(config):
    block = Block("re33_m3", "ibea", "de", "polynomial")
    first, _ = run_cell(config, block, 1, 2000)
    again, _ = run_cell(config, block, 1, 2000)
    other, _ = run_cell(config, block, 2, 2000)
    assert all(np.array_equal(first[k], again[k], equal_nan=True) for k in first)
    assert not np.array_equal(first["snap_F"], other["snap_F"])
