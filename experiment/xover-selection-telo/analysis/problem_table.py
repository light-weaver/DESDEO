"""The paper's table of test problems, built from the instance registry and the calibrated budgets.

One row per base problem: its decision and objective counts, how many instances it contributes, the
range of its evaluation budgets, and, for the benchmark problems, the property labels the analysis
splits on (`xover/problem_properties.py`). A property that differs between a row's instances is shown
as both levels. Citation keys are placeholders for the paper's bibliography.

    python -m analysis.problem_table
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from analysis.common import load_config, results_dir
from xover import design, problem_properties

SUITES = (
    ("dtlz_wfg", "DTLZ and WFG"),
    ("zcat", "ZCAT"),
    ("rotated_dtlz", "Rotated DTLZ (CEC 2009)"),
    ("real_world", "Real-world problems"),
)

ROWS: dict[str, tuple[str, str, str]] = {
    **{f"dtlz{i}": (f"DTLZ{i}", "deb2005dtlz", "") for i in range(1, 8)},
    **{f"wfg{i}": (f"WFG{i}", "huband2006wfg", "") for i in range(1, 10)},
    **{f"zcat{i}": (f"ZCAT{i}", "zapotecas2023zcat", "") for i in (1, 3, 11, 14, 18)},
    "r2_dtlz2": ("UF11 (R2-DTLZ2-M5)", "zhang2008cec09", ""),
    "r2_dtlz3": ("UF12 (R2-DTLZ3-M5)", "zhang2008cec09", ""),
    "re31": ("RE3-1", "tanabe2020re", "Two-bar truss design"),
    "re32": ("RE3-2", "tanabe2020re", "Welded beam design"),
    "re33": ("RE3-3", "tanabe2020re", "Disc brake design"),
    "re34": ("RE3-4", "tanabe2020re", "Vehicle crashworthiness design"),
    "re37": ("RE3-7", "tanabe2020re", "Rocket injector design, three objectives"),
    "re41": ("RE4-1", "tanabe2020re", "Car side-impact design"),
    "re42": ("RE4-2", "tanabe2020re", "Conceptual marine design"),
    "re61": ("RE6-1", "tanabe2020re", "Water resource planning"),
    "rocket_injector_design": ("Rocket injector", "vaidyanathan2003rocket", "Original four-objective version"),
    "river_pollution": ("River pollution", "narula1989river", "Five-objective variant"),
    "gaa": ("GAA", "simpson1996gaa", "Aircraft family design, constraint as objective"),
    "metallappl": ("MetallAppl", "saini2023steel", "Microalloyed steel design, surrogates"),
}
"""Base problem -> (label, citation key, application). The application replaces the property columns
on real-world rows, whose properties are unknown."""

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


def joined(values) -> str:
    """Return the distinct values, sorted, separated by commas."""
    return ", ".join(str(v) for v in sorted(set(values)))


def budget_range(budgets: list[int]) -> str:
    """Return the budget range in thousands of evaluations."""
    low, high = min(budgets) // 1000, max(budgets) // 1000
    return f"{low}" if low == high else f"{low}--{high}"


def rows(config: dict, budgets: dict[str, int]) -> list[tuple[str, list[str]]]:
    """Return (suite header, table lines) per suite."""
    groups = defaultdict(list)
    for instance in design.selected_instances(config).values():
        groups[instance.suite, instance.base].append(instance)

    out = []
    for suite, header in SUITES:
        lines = []
        for (group_suite, base), instances in groups.items():
            if group_suite != suite:
                continue
            label, key, application = ROWS[base]
            cells = [
                label,
                rf"\cite{{{key}}}",
                joined(i.n_var for i in instances),
                joined(i.m for i in instances),
                str(len(instances)),
                budget_range([budgets[i.slug] for i in instances]),
            ]
            if suite == "real_world":
                cells.append(rf"\multicolumn{{4}}{{l}}{{{application}}}")
            else:
                labels = [problem_properties.describe(i.slug, i.base, i.family, i.m) for i in instances]
                for prop in PROPERTIES:
                    levels = ABBREVIATIONS[prop]
                    seen = [levels[d[prop]] for d in labels]
                    cells.append("/".join(dict.fromkeys(v for v in levels.values() if v in seen)))
            lines.append(" & ".join(cells) + r" \\")
        out.append((header, lines))
    return out


def latex(config: dict, budgets: dict[str, int]) -> str:
    """Return the table as LaTeX (booktabs)."""
    sections = rows(config, budgets)
    total = len(design.selected_instances(config))
    body = []
    for index, (header, lines) in enumerate(sections):
        if index:
            body.append(r"\midrule")
        count = sum(int(line.split(" & ")[4]) for line in lines)
        body.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{header}}} ({count} instances)}} \\")
        body += lines
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Test problems. $n$: decision variables; $m$: objectives; Inst.: instances in the"
            r" study; Budget: function evaluations per run, in thousands, calibrated per instance."
            r" Each DTLZ and WFG problem is run at $(m, n) = (3, 12), (3, 30), (5, 12)$ and $(8, 24)$."
            r" Each ZCAT problem is run at levels 1--4 (separable unimodal, non-separable unimodal,"
            r" separable multimodal, non-separable multimodal), each with and without a linked Pareto set."
            r" Each rotated problem is paired with its unrotated control. The RE, river pollution and rocket"
            r" injector problems with more than three objectives are run on the full objective set and on up to"
            r" five sampled subsets of at least three objectives; GAA on a nested ladder of 3 to 11 objectives;"
            r" MetallAppl on its two published formulations and three objective triples."
            r" Front: Pareto front shape (Lin.\ linear, Conc.\ concave, Disc.\ disconnected, Degen.\ degenerate);"
            r" Sep.: separable (S) or not (NS); Mod.: unimodal (U), multimodal (M) or deceptive (D);"
            r" PS: Pareto set a product of per-variable sets (P) or linked (L).}",
            r"\label{tab:problems}",
            r"\small",
            r"\begin{tabular}{llrrrrllll}",
            r"\toprule",
            r"Problem & Ref. & $n$ & $m$ & Inst. & Budget & Front & Sep. & Mod. & PS \\",
            r"\midrule",
            *body,
            r"\midrule",
            rf"\multicolumn{{4}}{{l}}{{Total}} & {total} & \multicolumn{{5}}{{l}}{{}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
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
