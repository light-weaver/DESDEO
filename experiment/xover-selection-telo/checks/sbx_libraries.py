"""Compare the study's SBX arms with the libraries they are named after.

Each library's own code and DESDEO's operator recombine the same parent pair 200 000 times, and the
offspring are compared per variable: Kolmogorov-Smirnov on each offspring's value, on the gap between
the two offspring and on the offspring's mean, plus the share of offspring equal to either parent.
Variables 0 and 1 are a wide interior gap in both orders, 2 and 3 sit near a bound, 4 and 6 on one, 5
has equal values and 7 is a narrow gap. Rows marked `differ` are controls that must fail, which shows the test can see a
difference of that kind.

    python checks/sbx_libraries.py --platemo <PlatEMO>/PlatEMO --jars <jars> --out <dir>

`<jars>` holds jmetal-core and commons-lang3, commons-math3, commons-collections4 and commons-io.
Needs Octave, a JVM and uv. PLAN.md records the versions compared.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from desdeo.emo.operators.crossover import SimulatedBinaryCrossover
from desdeo.problem.testproblems import dtlz2
from desdeo.tools.patterns import Publisher
from pymoo.core.problem import Problem as PymooProblem
from pymoo.operators.crossover.sbx import SBX
from scipy.stats import chi2_contingency, ks_2samp

HERE = Path(__file__).parent
P1 = np.array([0.2, 0.8, 0.01, 0.97, 0.0, 0.4, 0.5, 0.3])
P2 = np.array([0.8, 0.2, 0.03, 0.995, 0.5, 0.4, 1.0, 0.31])
N = 200_000
D = P1.size
ALPHA = 0.01


def desdeo_sbx(*, eta: float, truncated: bool, swap: bool, uniform: float, pair: float = 1.0, seed: int = 2):
    """DESDEO's offspring as an (N, 2, D) array."""
    problem = dtlz2(n_objectives=3, n_variables=D)
    symbols = [v.symbol for v in problem.get_flattened_variables()]
    operator = SimulatedBinaryCrossover(
        problem=problem,
        seed=seed,
        verbosity=0,
        publisher=Publisher(),
        pair_xover_probability=pair,
        xover_probability=0.5,
        uniform_xover_probability=uniform,
        xover_distribution=eta,
        truncated=truncated,
        swap_uncrossed_variables=swap,
    )
    population = pl.DataFrame(np.tile(np.vstack([P1, P2]), (N, 1)), schema=symbols, orient="row")
    offspring = operator.do(population=population, to_mate=list(range(2 * N)))
    return offspring.to_numpy().reshape(N, 2, D)


def pymoo_sbx(*, eta: float, prob_bin: float, seed: int = 3):
    """Pymoo's offspring as an (N, 2, D) array, through `SBX._do`, which is what pymoo calls per mating."""
    problem = PymooProblem(n_var=D, n_obj=2, xl=np.zeros(D), xu=np.ones(D))
    parents = np.stack([np.tile(P1, (N, 1)), np.tile(P2, (N, 1))])
    offspring = SBX(prob_var=0.5, eta=eta, prob_exch=1.0, prob_bin=prob_bin)._do(
        problem, parents, random_state=np.random.default_rng(seed)
    )
    return offspring.transpose(1, 0, 2)


def platemo_sbx(path: Path, out: Path, *, proc: float, disc: float):
    """PlatEMO's `OperatorGA` run in Octave, as an (N, 2, D) array."""
    target = out / f"platemo_{proc}_{disc}.bin"
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "octave", "--no-gui", "--quiet", "--path", str(path / "Algorithms" / "Utility functions"),
            str(HERE / "platemo_sbx.m"), str(target), str(proc), str(disc),
        ],
        check=True,
    )  # fmt: skip
    stacked = np.fromfile(target).reshape(2 * N, D)
    return np.stack([stacked[:N], stacked[N:]], axis=1)


def jmetal_sbx(jars: Path, out: Path, *, probability: float, index: float):
    """JMetal's `SBXCrossover` run on the JVM, as an (N, 2, D) array."""
    target = out / f"jmetal_{probability}_{index}.bin"
    command = ["uv", "run", "--no-project", "--with", "jpype1", "python", str(HERE / "jmetal_sbx.py")]
    subprocess.run([*command, str(jars), str(target), str(probability), str(index)], check=True)  # noqa: S603
    return np.fromfile(target).reshape(N, 2, D)


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, list[int]]:
    """The smallest Bonferroni-corrected p-value over every test, and the variables that fail.

    Offspring are rounded first. The untruncated form computes an uncrossed gene as `avg + diff`, one
    ulp away from the parent, and Kolmogorov-Smirnov would read that atom as a shift in location.
    """
    a, b = np.round(a, 12), np.round(b, 12)
    tests: dict[tuple[str, int], float] = {}
    for var in range(D):
        for slot in (0, 1):
            x, y = a[:, slot, var], b[:, slot, var]
            if x.std() or y.std():
                tests["value", var, slot] = ks_2samp(x, y, method="asymp").pvalue
            for label, parent in (("p1", P1[var]), ("p2", P2[var])):
                table = [[(x == parent).sum(), N - (x == parent).sum()], [(y == parent).sum(), N - (y == parent).sum()]]
                if min(table[0][0] + table[1][0], table[0][1] + table[1][1]) > 0:
                    tests[label, var, slot] = chi2_contingency(table).pvalue
        pairs = {
            "gap": (a[:, 0, var] - a[:, 1, var], b[:, 0, var] - b[:, 1, var]),
            "mean": (a[:, :, var].mean(1), b[:, :, var].mean(1)),
        }
        for name, (x, y) in pairs.items():
            x, y = np.round(x, 12), np.round(y, 12)  # noqa: PLW2901
            if x.std() or y.std():
                tests[name, var, 0] = ks_2samp(x, y, method="asymp").pvalue
    corrected = {key: min(p * len(tests), 1.0) for key, p in tests.items()}
    return min(corrected.values()), sorted({key[1] for key, p in corrected.items() if p <= ALPHA})


def main() -> None:
    """Run every comparison and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platemo", type=Path, required=True)
    parser.add_argument("--jars", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    default = desdeo_sbx(eta=30, truncated=True, swap=False, uniform=0.5)
    platemo_arm = desdeo_sbx(eta=30, truncated=False, swap=False, uniform=0.5)
    jmetal_arm = desdeo_sbx(eta=20, truncated=True, swap=True, uniform=0.5)
    no_uniform = desdeo_sbx(eta=30, truncated=True, swap=False, uniform=0.0)

    platemo_30 = platemo_sbx(args.platemo, args.out, proc=1, disc=30)
    platemo_20 = platemo_sbx(args.platemo, args.out, proc=1, disc=20)
    jmetal_10 = jmetal_sbx(args.jars, args.out, probability=1.0, index=20)
    jmetal_09 = jmetal_sbx(args.jars, args.out, probability=0.9, index=20)
    pymoo_30 = pymoo_sbx(eta=30, prob_bin=0.5)

    rows = [
        ("match", "sbx_default", "pymoo SBX eta=30 prob=1 (pymoo's NSGA-III)", default, pymoo_30),
        ("match", "sbx_platemo", "PlatEMO OperatorGA {1,30}", platemo_arm, platemo_30),
        ("match", "sbx_platemo at eta=20", "PlatEMO OperatorGA defaults {1,20}",
         desdeo_sbx(eta=20, truncated=False, swap=False, uniform=0.5), platemo_20),
        ("match", "sbx_jmetal", "jMetal SBXCrossover(1.0, 20)", jmetal_arm, jmetal_10),
        ("match", "sbx_jmetal at pair 0.9", "jMetal SBXCrossover(0.9, 20) (its NSGA-II example)",
         desdeo_sbx(eta=20, truncated=True, swap=True, uniform=0.5, pair=0.9), jmetal_09),
        ("differ", "sbx_platemo", "PlatEMO OperatorGA defaults {1,20}", platemo_arm, platemo_20),
        ("differ", "sbx_jmetal", "jMetal SBXCrossover(0.9, 20) (its NSGA-II example)", jmetal_arm, jmetal_09),
        ("differ", "sbx_default", "pymoo SBX eta=15 (pymoo's NSGA-II, pair 1)", default,
         pymoo_sbx(eta=15, prob_bin=0.5)),
        ("differ", "sbx_default", "PlatEMO OperatorGA {1,30}", default, platemo_30),
        ("differ", "sbx_platemo", "pymoo SBX eta=30", platemo_arm, pymoo_30),
        ("differ", "sbx_default at eta=20", "jMetal SBXCrossover(1.0, 20)",
         desdeo_sbx(eta=20, truncated=True, swap=False, uniform=0.5), jmetal_10),
        ("differ", "sbx_no_uniform", "pymoo SBX eta=30 prob_bin=0", no_uniform, pymoo_sbx(eta=30, prob_bin=0.0)),
    ]  # fmt: skip

    print(f"{'want':7}{'verdict':9}{'p (Bonferroni)':16}{'DESDEO':24}library")
    failed = False
    for want, arm, library, ours, theirs in rows:
        p, variables = compare(ours, theirs)
        verdict = "match" if p > ALPHA else "differ"
        failed |= verdict != want
        note = f"  variables {variables}" if variables else ""
        print(f"{want:7}{verdict:9}{p:<16.3g}{arm:24}{library}{note}")
    print("\nunexpected verdicts above" if failed else "\nevery row as expected")


if __name__ == "__main__":
    main()
