# Crossover selection in EMO: experiment pipeline

This directory regenerates every raw result of the crossover-selection study: per-instance
evaluation budgets, reference fronts, population dumps of every run, quality indicators, and
landscape features. It is a [Snakemake](https://snakemake.readthedocs.io) workflow over a small
Python package (`xover/`) built on [DESDEO](https://github.com/industrial-optimization-group/DESDEO).

The study compares 14 real-coded crossover operators inside 5 algorithms (NSGA-III, NSGA-II, IBEA,
SPEA2, AGE-MOEA-II) on 150 problem instances with 3 to 11 objectives (108 benchmark instances from
DTLZ, WFG, ZCAT and rotated DTLZ, and 42 real-world ones), at 11 seeds per configuration: 115 500
runs.

## Running it

From this directory, inside a DESDEO checkout:

```bash
uv sync --inexact --group experiment                                                # DESDEO + Snakemake
uv run --group experiment snakemake --cores 8 --configfile config/smoke.yaml        # every phase, in about a minute
uv run --group experiment snakemake --cores 128                                     # the full study
uv run --group experiment snakemake --cores 128 budgets                             # stop after one phase
```

`--inexact` keeps any other dependency groups you already have installed. The first job that needs
landscape features creates a second, small environment for pflacco (`envs/ela/`, Python 3.11); `uv`
must be on the `PATH` for that, and it needs network access once to download the packages.

The workflow is resumable: interrupt it and run the same command again. A second run with nothing
changed reports "Nothing to be done".

On a Slurm cluster, `slurm/run.sh` runs the whole workflow inside one allocation (see its header).

## The phases

| Phase | Target | What it does | Unit of work |
|---|---|---|---|
| 0 | `problem_data` | Builds every instance once, recording its objectives and box, and fits the MetallAppl surrogate models into the cache before any parallel job needs them | once |
| 1 | `budgets` | Calibrates each instance's evaluation budget: the smallest budget at which NSGA-III with `sbx_default` reaches within 2 % of its own 200 000-evaluation median IGD+, rounded up to 5 000 and clipped to [10 000, 200 000] | instance |
| 2 | `fronts` | Runs every algorithm × crossover at seeds 0 and 1 with a non-dominated archive, merges the archives, and freezes one reference front per instance (greedy max-min subset of 1 500 to 4 000 points) | block, then instance |
| 3 | `runs` | Runs the main grid. Each run dumps its population (objective vectors) at 21 budget fractions, 0.02 and then every 5 %, plus per-generation crossover mechanism statistics | block (all seeds) |
| 4 | `indicators` | Scores every dump against its instance's front: IGD+, IGD, additive epsilon, GD, GD_p, GD+, averaged Hausdorff, coverage error, hypervolume (relative), R2, spacing, spread, uniformity | instance × algorithm |
| 5 | `features` | Draws one Latin-hypercube sample of 200 × n_var points per instance and computes multi-objective landscape features and pflacco ELA features on it, then screens them | instance |

A *block* is one instance × algorithm × crossover × mutation; its job runs every seed, so each job
starts Python once rather than once per run. Phases 2 and 3 both need only the budgets and run side
by side. Phase 5 needs only the problems. Phase 6, the statistical analysis, is described in
[docs/phase6.md](docs/phase6.md) and not yet part of the workflow.

## Changing the setup

Everything is in [config/config.yaml](config/config.yaml): the problem suites, which instances,
algorithms, crossovers, mutations and seeds are run, the operator parameters (as DESDEO option
dictionaries), the population sizes, and the settings of every phase. Two ways to change it:

```bash
uv run --group experiment snakemake --cores 8 --config mechanism_stride=10          # one value
uv run --group experiment snakemake --cores 8 --configfile my_changes.yaml          # a file layered on top
```

Dictionaries in a layered file merge into the defaults and lists replace them, as in
[config/smoke.yaml](config/smoke.yaml). The config is validated before anything runs, and an invalid
operator parameter is reported as one line, such as
`crossovers.sbx_default.xover_probability: Input should be less than or equal to 1`.

Each job receives its own slice of the config, so Snakemake reruns exactly the jobs a change affects.
Editing one crossover's parameters reruns that crossover's runs, the fronts they feed, and the
scoring of their instances, and nothing else.

To choose which runs build the reference fronts, edit `reference_front` (`all` means the list under
`run`). A front built from some of the crossovers favours those crossovers in every indicator.

## Outputs

Under `results/` (`results_smoke/` for the smoke configuration):

| Path | Contents |
|---|---|
| `problem_data.json` | every instance: suite, base problem, objectives and senses, decision box |
| `budgets/{instance}.json`, `budgets.csv` | the budget, the median IGD+ saturation curve, timings |
| `front_runs/{instance}/{algorithm}/{crossover}/{mutation}.npz` | the archives of the front runs (`archive_F_s{seed}`) |
| `fronts/{instance}.npz` + `.json` | the frozen front (`points`, `ideal`, `nadir`, float32) and its pool size and HV(R) |
| `runs/{instance}/{algorithm}/{crossover}/{mutation}/s{seed}.npz` | one run's dump, described below |
| `runs/.../s{seed}.json` | its timings, generations, message counts and provenance (git SHA, package versions) |
| `indicators/{instance}/{algorithm}.parquet` | one row per run and budget fraction |
| `mechanism/{instance}/{algorithm}.parquet` | one row per run: mean and final value of each mechanism statistic |
| `landscape/sample/{instance}.npz` | the landscape sample and its derived scalar landscapes |
| `landscape/{mo,ela}/{instance}.parquet` | features per instance |
| `features.parquet`, `features_report.json` | the screened feature table, and why each feature was dropped |

A run dump (`.npz`, plain float32/int32 arrays) holds:

- `snap_F`: the population's objective vectors at every snapshot, stacked; `snap_n` gives the rows
  per snapshot, `snap_evals` and `snap_gen` when it was taken. Objectives are in minimisation sense.
- `snap_X0`: the initial population's decision vectors (identical across configurations at one seed).
- `gen_evals`, `gen_n`: evaluations and population size after every generation.
- `mech_*`: crossover diagnostics every 5th generation (parent separation, offspring step, fraction
  of genes on a bound, parent-set diameter), normalised by the decision box diagonal.

`xover.logger.load_run` and `xover.logger.unpack_snapshots` read them back.

## Reproducibility

- One integer seed per run drives every random component. The initial population depends only on
  the seed and the population size, so all configurations at one seed start from the same points.
- The objective subsets of the real-world problems are drawn from fixed seeds and do not depend on
  `PYTHONHASHSEED`.
- Every result file records the git commit, whether the tree was dirty, and the versions of DESDEO,
  numpy, scipy, polars, moocore, pymoo and scikit-learn.
- `tests/test_legacy_equivalence.py` shows that this pipeline produces bit-identical run dumps,
  fronts, indicators and calibration results to the code the EMO-paper version of the study ran
  (branch `experiment/xover-selection-2`), given the same inputs. This covers the EMO paper's 3
  algorithms and 13 crossovers; SPEA2, AGE-MOEA-II and `de_current` have no old code to match.
- `checks/sbx_libraries.py` compares the SBX arms with pymoo, PlatEMO and jMetal, running each
  library's own code. [PLAN.md](PLAN.md) records the result, including the parameters that differ.

```bash
uv run pytest tests                                                                 # about 10 s
git worktree add /tmp/legacy experiment/xover-selection-2
LEGACY_DIR=/tmp/legacy/experiment/xover-selection uv run pytest tests               # about 90 s more
```

## Cost

The EMO-paper version of the study ran a 105-instance grid on one 128-core node of CSC's Roihu
cluster. Its 8 034 front runs took 1.21 h and its 61 776 grid runs took 140 min. This version runs
150 instances (21 000 front runs, 115 500 grid runs) and adds calibration and landscape features.
GAA is the expensive problem, at about 27 times the mean cost per run. [PLAN.md](PLAN.md) has the
estimate and the measured cost of each algorithm.
