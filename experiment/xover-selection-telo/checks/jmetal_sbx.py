"""Write jMetal's SBXCrossover offspring for the fixed parent pairs of `sbx_libraries.py`.

Runs in its own environment, because it needs JPype to call the jar:

    uv run --no-project --with jpype1 python jmetal_sbx.py <jars> <out> <probability> <distribution index>
"""

import sys
from pathlib import Path

import jpype
import jpype.imports
import numpy as np

P1 = [0.2, 0.8, 0.01, 0.97, 0.0, 0.4, 0.5, 0.3]
P2 = [0.8, 0.2, 0.03, 0.995, 0.5, 0.4, 1.0, 0.31]
N = 200_000


def main() -> None:
    """Recombine the pair N times and write the offspring as little-endian doubles."""
    jars, out, probability, index = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    jpype.startJVM(classpath=[str(jar) for jar in jars.glob("*.jar") if not jar.name.startswith("ecj")])

    from java.util import ArrayList
    from org.uma.jmetal.operator.crossover.impl import SBXCrossover
    from org.uma.jmetal.solution.doublesolution.impl import DefaultDoubleSolution
    from org.uma.jmetal.util.bounds import Bounds
    from org.uma.jmetal.util.pseudorandom import JMetalRandom

    JMetalRandom.getInstance().setSeed(1)
    bounds = ArrayList()
    for _ in P1:
        bounds.add(Bounds.create(jpype.java.lang.Double(0.0), jpype.java.lang.Double(1.0)))

    def solution(values: list[float]):
        made = DefaultDoubleSolution(bounds, 2, 0)
        for i, value in enumerate(values):
            made.variables().set(i, jpype.java.lang.Double(value))
        return made

    crossover = SBXCrossover(probability, index)
    parents = ArrayList()
    parents.add(solution(P1))
    parents.add(solution(P2))
    offspring = np.empty((N, 2, len(P1)))
    for pair in range(N):
        first, second = crossover.execute(parents)
        offspring[pair, 0] = [float(value) for value in first.variables()]
        offspring[pair, 1] = [float(value) for value in second.variables()]
    offspring.astype("<f8").tofile(out)


if __name__ == "__main__":
    main()
