"""The paper's two tables of test problems, built from the instance registry and the calibrated budgets.

The benchmark table has one row per base problem: its (variables, objectives) pairs, how many
instances it contributes, the range of its evaluation budgets, and the property labels the analysis
splits on (`xover/problem_properties.py`). A property that differs between a row's instances lists
both levels. It is a `sidewaystable*` (package `rotating`).

The real-world table has one row per instance: the problem, its application, its variables, the exact
objectives the instance optimises and its budget.

Both use `booktabs`, and the real-world table `array`. Citation keys are those of
`experiment/xover-selection/paper/references.bib` on the experiment/xover-selection-2 branch.

    python -m analysis.problem_table
"""

from __future__ import annotations

import re
from collections import defaultdict

import polars as pl

from analysis.common import load_config, results_dir
from xover import design, problem_properties, problems

BENCHMARK_CAPTION = "Benchmark problems included in the study"
REAL_WORLD_CAPTION = "Real-world problems included in the study, with the objectives of each instance"

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
    "re34": ("RE3-4", "tanabe2020easy", "Vehicle crashworthiness"),
    "re41": ("RE4-1", "tanabe2020easy", "Car side-impact design"),
    "re42": ("RE4-2", "tanabe2020easy", "Conceptual marine design"),
    "re61": ("RE6-1", "tanabe2020easy", "Water resource planning"),
    "rocket_injector_design": ("Rocket injector", "vaidyanathan2004computational", "Rocket injector design"),
    "river_pollution": ("River pollution", "narula1989flexible", "Water quality management"),
    "gaa": ("GAA", "woodruff2013many", "Aircraft family design"),
    "metallappl": ("MetallAppl", "saini2023interactive", "Microalloyed steel design"),
}
"""Real-world base problem -> (label, citation key, application), in table order."""

LEVELS = {
    "front": {
        "linear": "Linear",
        "concave": "Concave",
        "mixed": "Mixed",
        "disconnected": "Disconnected",
        "degenerate": "Degenerate",
    },
    "separable": {"yes": "Yes", "no": "No", "unknown": "Unknown"},
    "modality": {"unimodal": "Unimodal", "multimodal": "Multimodal", "deceptive": "Deceptive"},
    "pareto_set": {"product": "Product", "linked": "Linked"},
}
"""Property label -> the word printed for it."""

SYMBOLS = {
    "TF_max": r"$\mathit{TF}_{\max}$",
    "TW4": r"$\mathit{TW}_{4}$",
    "TT_max": r"$\mathit{TT}_{\max}$",
    "Xcc_max": r"$X_{\mathit{cc},\max}$",
}
"""Objective symbols that need more than `f_k` -> $f_{k}$. MetallAppl's (YS, UTS, ...) print as text."""


def budget_range(budgets: list[int]) -> str:
    """Return the budget range in thousands of evaluations."""
    low, high = min(budgets) // 1000, max(budgets) // 1000
    return f"{low}" if low == high else f"{low}--{high}"


def pairs(instances: list) -> str:
    """Return the distinct (variables, objectives) pairs, ordered by objectives and then variables."""
    distinct = sorted({(i.m, i.n_var) for i in instances})
    return ", ".join(f"({n}, {m})" for m, n in distinct)


def properties(instances: list) -> list[str]:
    """Return the four property cells of one benchmark row, both levels where its instances differ."""
    labels = [problem_properties.describe(i.slug, i.base, i.family, i.m) for i in instances]
    cells = []
    for prop, words in LEVELS.items():
        seen = [words[level] for level in words if any(d[prop] == level for d in labels)]
        cells.append(", ".join([seen[0], *(word.lower() for word in seen[1:])]))
    return cells


def objectives(instance) -> list[str]:
    """Return the objective symbols the instance optimises, read from the problem when it uses them all."""
    chosen = instance.spec.get("objectives")
    return list(chosen) if chosen else list(problems._shape(instance.spec)[0])


def symbol(name: str) -> str:
    """Return one objective symbol as LaTeX."""
    if name in SYMBOLS:
        return SYMBOLS[name]
    match = re.fullmatch(r"f_(\d+)", name)
    return rf"$f_{{{match.group(1)}}}$" if match else name


def grouped(config: dict) -> dict[tuple[str, str], list]:
    """Return the selected instances grouped by (suite, base problem), in registry order."""
    groups = defaultdict(list)
    for instance in design.selected_instances(config).values():
        groups[instance.suite, instance.base].append(instance)
    return groups


def row(cells: list[str]) -> str:
    """Return one table row."""
    return " & ".join(cells) + r" \\"


def benchmark_table(groups: dict, budgets: dict[str, int]) -> str:
    """Return the benchmark table: one section per suite, one row per base problem."""
    header = [
        "Problem",
        "(Variables, objectives)",
        "Instances",
        "Evaluations (thousands)",
        "Pareto front",
        "Separable",
        "Modality",
        "Pareto set",
    ]
    lines = [r"\begin{tabular}{llrrllll}", r"\toprule", row(header)]
    for suite, title in BENCHMARK_SUITES:
        members = {base: instances for (s, base), instances in groups.items() if s == suite}
        total = sum(len(instances) for instances in members.values())
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}, {total} instances}} \\"]
        lines += [
            row(
                [
                    LABELS[base],
                    pairs(instances),
                    str(len(instances)),
                    budget_range([budgets[i.slug] for i in instances]),
                    *properties(instances),
                ]
            )
            for base, instances in members.items()
        ]
    return float_(
        "sidewaystable*",
        BENCHMARK_CAPTION,
        "tab:benchmark-problems",
        [*lines, r"\bottomrule", r"\end{tabular}"],
    )


def real_world_table(groups: dict, budgets: dict[str, int]) -> str:
    """Return the real-world table: one row per instance, the problem named on its first row."""
    header = [
        "Problem",
        "Application",
        "Variables",
        "Objectives",
        "Objectives used",
        "Evaluations (thousands)",
    ]
    # The application column wraps, so that the table fits the width of a two-column page.
    lines = [r"\begin{tabular}{l>{\raggedright\arraybackslash}p{3.4cm}rrlr}", r"\toprule", row(header), r"\midrule"]
    for number, (base, (label, key, application)) in enumerate(REAL_WORLD.items()):
        instances = groups["real_world", base]
        order = objectives(max(instances, key=lambda i: i.m))
        instances = sorted(instances, key=lambda i: (-i.m, [order.index(s) for s in objectives(i)]))
        if number:
            lines.append(r"\addlinespace")
        for index, instance in enumerate(instances):
            first = index == 0
            lines.append(
                row(
                    [
                        rf"{label}~\cite{{{key}}}" if first else "",
                        application if first else "",
                        str(instance.n_var) if first else "",
                        str(instance.m),
                        ", ".join(symbol(s) for s in objectives(instance)),
                        str(budgets[instance.slug] // 1000),
                    ]
                )
            )
    total = sum(len(instances) for (suite, _), instances in groups.items() if suite == "real_world")
    lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{{total} instances}} \\", r"\bottomrule", r"\end{tabular}"]
    return float_("table*", REAL_WORLD_CAPTION, "tab:real-world-problems", lines)


def float_(environment: str, caption: str, label: str, body: list[str]) -> str:
    """Return a float holding one tabular."""
    return "\n".join(
        [
            rf"\begin{{{environment}}}",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            *body,
            rf"\end{{{environment}}}",
            "",
        ]
    )


def main() -> None:
    """Write `results/analysis/problems_benchmark.tex` and `problems_real_world.tex`."""
    config = load_config()
    table = pl.read_csv(results_dir(config) / "budgets.csv")
    budgets = dict(zip(table["instance"], table["budget"], strict=True))
    groups = grouped(config)
    directory = results_dir(config) / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("problems_benchmark.tex", benchmark_table(groups, budgets)),
        ("problems_real_world.tex", real_world_table(groups, budgets)),
    ):
        (directory / name).write_text(text)
        print(f"wrote {directory / name}")


if __name__ == "__main__":
    main()
