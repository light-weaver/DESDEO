"""Phase 0: build every selected instance once, and record what each one is.

Building MetallAppl fits its surrogate models and caches them, and DESDEO writes that cache through
one fixed temporary path, so jobs that fitted the same model at once could corrupt it. Every job that
builds a problem therefore waits for this one.

The record lists each instance's objectives, senses and decision box, as built.
"""

from xover import design
from xover.problems import build_problem
from xover.run import provenance, write_json

config = snakemake.config
records = []
for slug, instance in design.selected_instances(config).items():
    problem = build_problem(instance.spec)
    variables = problem.get_flattened_variables()
    records.append(
        {
            "instance": slug,
            "suite": instance.suite,
            "base": instance.base,
            "family": instance.family,
            "m": instance.m,
            "n_var": instance.n_var,
            "objectives": [objective.symbol for objective in problem.objectives],
            "maximise": [bool(objective.maximize) for objective in problem.objectives],
            "lower": [variable.lowerbound for variable in variables],
            "upper": [variable.upperbound for variable in variables],
            "spec": instance.spec,
        }
    )
    print(f"built {slug}")

write_json(snakemake.output[0], {"instances": records, "provenance": provenance()})
