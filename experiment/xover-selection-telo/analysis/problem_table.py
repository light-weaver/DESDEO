"""The paper's table of test problems, built from the instance registry and the calibrated budgets.

One row per base problem: its decision and objective counts, how many instances it contributes, the
range of its evaluation budgets, and, for the benchmark problems, the property labels the analysis
splits on (`xover/problem_properties.py`). A property that differs between a row's instances is shown
as both levels. Citation keys are those of `experiment/xover-selection/paper/references.bib` on the
experiment/xover-selection-2 branch.

The table is a `sidewaystable*` (package `rotating`, plus `booktabs`) with two panels side by side,
benchmark problems on the left and real-world problems on the right, so that it fits one landscape
page.

    python -m analysis.problem_table
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from analysis.common import load_config, results_dir
from xover import design, problem_properties

BENCHMARK_SUITES = (
    ("dtlz_wfg", r"DTLZ~\cite{deb2005scalable} and WFG~\cite{huband2006review}"),
    ("zcat", r"ZCAT~\cite{zapotecas2023challenging}"),
    ("rotated_dtlz", r"Rotated DTLZ, CEC 2009~\cite{zhang2009multiobjective}"),
)
"""Benchmark suites, in table order, with their citations in the section header."""

LABELS = {
    **{f"dtlz{i}": f"DTLZ{i}" for i in range(1, 8)},
    **{f"wfg{i}": f"WFG{i}" for i in range(1, 10)},
    **{f"zcat{i}": f"ZCAT{i}" for i in (1, 3, 11, 14, 18)},
    "r2_dtlz2": "UF11 (R2-DTLZ2-M5)",
    "r2_dtlz3": "UF12 (R2-DTLZ3-M5)",
}
"""Benchmark base problem -> row label."""

REAL_WORLD: dict[str, tuple[str, str, str]] = {
    "re31": ("RE3-1", "tanabe2020easy", "Two-bar truss design"),
    "re32": ("RE3-2", "tanabe2020easy", "Welded beam design"),
    "re33": ("RE3-3", "tanabe2020easy", "Disc brake design"),
    "re34": ("RE3-4", "tanabe2020easy", "Vehicle crashworthiness design"),
    "re41": ("RE4-1", "tanabe2020easy", "Car side-impact design"),
    "re42": ("RE4-2", "tanabe2020easy", "Conceptual marine design"),
    "re61": ("RE6-1", "tanabe2020easy", "Water resource planning"),
    "rocket_injector_design": ("Rocket injector", "vaidyanathan2004computational", "Four-objective original"),
    "river_pollution": ("River pollution", "narula1989flexible", "Five-objective variant"),
    "gaa": ("GAA", "woodruff2013many", "Aircraft family design"),
    "metallappl": ("MetallAppl", "saini2023interactive", "Microalloyed steel design, surrogates"),
}
"""Real-world base problem -> (label, citation key, application). Their properties are unknown, so the
application stands in for the property columns."""

ABBREVIATIONS = {
    "front": {
        "linear": "Lin.",
        "concave": "Conc.",
        "mixed": "Mixed",
        "disconnected": "Disc.",
        "degenerate": "Degen.",
    },
    "separable": {"yes": "S", "no": "NS", "unknown": "--"},
    "modality": {"unimodal": "U", "multimodal": "M", "deceptive": "D"},
    "pareto_set": {"product": "P", "linked": "L"},
}
PROPERTIES = tuple(ABBREVIATIONS)

CAPTION = r"Test problems included in the study"


def joined(values) -> str:
    """Return the distinct values: one or two listed, three or more as a range to keep the columns narrow."""
    distinct = sorted(set(values))
    if len(distinct) > 2:  # noqa: PLR2004
        return f"{distinct[0]}--{distinct[-1]}"
    return ", ".join(str(v) for v in distinct)


def budget_range(budgets: list[int]) -> str:
    """Return the budget range in thousands of evaluations."""
    low, high = min(budgets) // 1000, max(budgets) // 1000
    return f"{low}" if low == high else f"{low}--{high}"


def counts(instances: list, budgets: dict[str, int]) -> list[str]:
    """Return the n, m, instance-count and budget cells of one row."""
    return [
        joined(i.n_var for i in instances),
        joined(i.m for i in instances),
        str(len(instances)),
        budget_range([budgets[i.slug] for i in instances]),
    ]


def properties(instances: list) -> list[str]:
    """Return the four property cells of one benchmark row, both levels where its instances differ."""
    labels = [problem_properties.describe(i.slug, i.base, i.family, i.m) for i in instances]
    cells = []
    for prop in PROPERTIES:
        levels = ABBREVIATIONS[prop]
        seen = {levels[d[prop]] for d in labels}
        cells.append("/".join(v for v in dict.fromkeys(levels.values()) if v in seen))
    return cells


def grouped(config: dict) -> dict[tuple[str, str], list]:
    """Return the selected instances grouped by (suite, base problem), in registry order."""
    groups = defaultdict(list)
    for instance in design.selected_instances(config).values():
        groups[instance.suite, instance.base].append(instance)
    return groups


def row(cells: list[str]) -> str:
    """Return one table row."""
    return " & ".join(cells) + r" \\"


def benchmark_panel(groups: dict, budgets: dict[str, int]) -> list[str]:
    """Return the left panel: one section per benchmark suite, with property columns."""
    lines = [
        r"\begin{tabular}[t]{lrrrrllll}",
        r"\toprule",
        row(["Problem", "$n$", "$m$", "Inst.", "Budget", "Front", "Sep.", "Mod.", "PS"]),
    ]
    for suite, header in BENCHMARK_SUITES:
        members = {base: instances for (s, base), instances in groups.items() if s == suite}
        total = sum(len(instances) for instances in members.values())
        lines += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{\textit{{{header}}}, {total} instances}} \\"]
        lines += [
            row([LABELS[base], *counts(instances, budgets), *properties(instances)])
            for base, instances in members.items()
        ]
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def real_world_panel(groups: dict, budgets: dict[str, int], n_instances: int) -> list[str]:
    """Return the right panel: the real-world problems, each with its application, and the total."""
    members = {base: instances for (s, base), instances in groups.items() if s == "real_world"}
    total = sum(len(instances) for instances in members.values())
    lines = [
        r"\begin{tabular}[t]{lrrrrl}",
        r"\toprule",
        row(["Problem", "$n$", "$m$", "Inst.", "Budget", "Application"]),
        r"\midrule",
        rf"\multicolumn{{6}}{{l}}{{\textit{{Real-world problems}}, {total} instances}} \\",
    ]
    for base, instances in members.items():
        label, key, application = REAL_WORLD[base]
        lines.append(row([rf"{label}~\cite{{{key}}}", *counts(instances, budgets), application]))
    return [
        *lines,
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{All problems}} & {n_instances} & & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]


def latex(config: dict, budgets: dict[str, int]) -> str:
    """Return the table as LaTeX: a rotated float holding the two panels side by side."""
    groups = grouped(config)
    n_instances = len(design.selected_instances(config))
    return "\n".join(
        [
            r"\begin{sidewaystable*}",
            r"\centering",
            rf"\caption{{{CAPTION}}}",
            r"\label{tab:problems}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            *benchmark_panel(groups, budgets),
            r"\hspace{1em}",
            *real_world_panel(groups, budgets, n_instances),
            r"\end{sidewaystable*}",
            "",
        ]
    )


def main() -> None:
    """Write `results/analysis/problems.tex` and print it."""
    config = load_config()
    table = pl.read_csv(results_dir(config) / "budgets.csv")
    budgets = dict(zip(table["instance"], table["budget"], strict=True))
    text = latex(config, budgets)
    path = results_dir(config) / "analysis" / "problems.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
