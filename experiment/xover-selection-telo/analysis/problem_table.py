"""The paper's two tables of test problems, built from the instance registry and the calibrated budgets.

The benchmark table stacks one block per suite and gives every instance's budget, with the property
labels the analysis splits on (`xover/problem_properties.py`). DTLZ and WFG list a (variables,
objectives, budget) triple per instance. ZCAT and rotated DTLZ, whose instances differ in their
properties rather than their size, have one budget column per property combination.

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

BENCHMARK_CAPTION = "Benchmark problems included in the study. Budgets are in thousands of function evaluations"
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


def thousands(evaluations: int) -> str:
    """Return a budget in thousands of evaluations. Every budget is a multiple of 5 000, so this is exact."""
    return str(evaluations // 1000)


def label_words(instance) -> dict[str, str]:
    """Return the instance's property labels as the words the table prints."""
    labels = problem_properties.describe(instance.slug, instance.base, instance.family, instance.m)
    return {prop: words[labels[prop]] for prop, words in LEVELS.items()}


def suite_shape(instances: list) -> str:
    """Return the variables and objectives shared by every instance of a suite, as a title fragment."""
    (n_var,), (m,) = {i.n_var for i in instances}, {i.m for i in instances}
    return f"{n_var} variables, {m} objectives"


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


def zcat_level(level: int) -> str:
    """Return the separability and modality of a ZCAT level's distance function (Table 4 of the ZCAT paper)."""
    separable = "Separable" if problem_properties.ZCAT_SEPARABLE[level] else "Non-separable"
    return f"{separable}, {'multimodal' if problem_properties.ZCAT_MULTIMODAL[level] else 'unimodal'}"


def dtlz_wfg_block(members: dict, budgets: dict[str, int]) -> list[str]:
    """Return the DTLZ and WFG block: fixed properties per problem, a (variables, objectives, budget) per instance."""
    total = sum(len(instances) for instances in members.values())
    header = ["Problem", "Pareto front", "Separable", "Modality", "Pareto set", "(Variables, objectives, budget)"]
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        rf"\multicolumn{{6}}{{l}}{{\textit{{{dict(BENCHMARK_SUITES)['dtlz_wfg']}}}, {total} instances}} \\",
        r"\midrule",
        row(header),
        r"\midrule",
    ]
    for base, instances in members.items():
        words = label_words(instances[0])
        triples = ", ".join(
            f"({i.n_var}, {i.m}, {thousands(budgets[i.slug])})" for i in sorted(instances, key=lambda i: (i.m, i.n_var))
        )
        lines.append(row([LABELS[base], *words.values(), triples]))
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def zcat_block(members: dict, budgets: dict[str, int]) -> list[str]:
    """Return the ZCAT block: one budget column per (level, Pareto set), the level's properties in the header.

    The level fixes the separability and modality of the distance function (Table 4 of the ZCAT paper); a
    linked Pareto set makes the instance non-separable whatever the level, so the header gives the level's
    own properties and the Pareto set separately.
    """
    instances = [i for group in members.values() for i in group]
    levels = sorted({problem_properties.parse_zcat_slug(i.slug)[1] for i in instances})
    columns = [(level, linked) for level in levels for linked in (False, True)]
    title = f"{dict(BENCHMARK_SUITES)['zcat']}, {suite_shape(instances)}, {len(instances)} instances"
    lines = [
        rf"\begin{{tabular}}{{ll*{{{len(columns)}}}{{c}}}}",
        r"\toprule",
        rf"\multicolumn{{{2 + len(columns)}}}{{l}}{{\textit{{{title}}}}} \\",
        r"\midrule",
        row(["", "", *(rf"\multicolumn{{2}}{{c}}{{Level {level}}}" for level in levels)]),
        row(
            [
                "",
                "",
                *(rf"\multicolumn{{2}}{{c}}{{{zcat_level(level)}}}" for level in levels),
            ]
        ),
        "".join(rf"\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(levels))),
        row(["Problem", "Pareto front", *("Linked" if linked else "Product" for _, linked in columns)]),
        r"\midrule",
    ]
    for base, group in members.items():
        by_column = {problem_properties.parse_zcat_slug(i.slug)[1:]: i for i in group}
        cells = [thousands(budgets[by_column[column].slug]) for column in columns]
        lines.append(row([LABELS[base], label_words(group[0])["front"], *cells]))
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def rotated_block(members: dict, budgets: dict[str, int]) -> list[str]:
    """Return the rotated DTLZ block: the budget of the unrotated control and of the rotated problem."""
    instances = [i for group in members.values() for i in group]
    title = f"{dict(BENCHMARK_SUITES)['rotated_dtlz']}, {suite_shape(instances)}, {len(instances)} instances"
    lines = [
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\",
        r"\midrule",
        row(["", "", "", "Unrotated control", "Rotated"]),
        row(["Problem", "Pareto front", "Modality", "Separable, product", "Non-separable, linked"]),
        r"\midrule",
    ]
    for base, group in members.items():
        by_rotation = {problem_properties.is_rotated(i.slug): i for i in group}
        words = label_words(group[0])
        cells = [thousands(budgets[by_rotation[rotated].slug]) for rotated in (False, True)]
        lines.append(row([LABELS[base], words["front"], words["modality"], *cells]))
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def benchmark_table(groups: dict, budgets: dict[str, int]) -> str:
    """Return the benchmark table: one block per suite, stacked, each listing every instance's budget."""
    blocks = {"dtlz_wfg": dtlz_wfg_block, "zcat": zcat_block, "rotated_dtlz": rotated_block}
    body = []
    for suite, _ in BENCHMARK_SUITES:
        members = {base: instances for (s, base), instances in groups.items() if s == suite}
        if body:
            body.append(r"\par\bigskip")
        body += blocks[suite](members, budgets)
    return float_("table*", BENCHMARK_CAPTION, "tab:benchmark-problems", body)


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
