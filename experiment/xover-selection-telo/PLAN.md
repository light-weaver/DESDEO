# Design record: crossover selection in EMO (TELO version)

Branch `experiment/xover-selection-telo`, cut from `master` at `8c7e9d6a` (DESDEO 2.8.0 plus two
documentation commits), with `master` merged in at `0553d414` on 2026-10-30 (`2126de30`). Never
merged back. The EMO-paper version of the study stays on `experiment/xover-selection-2`, in
`experiment/xover-selection/`.

## Claim

In NSGA-III, NSGA-II, IBEA, SPEA2 and AGE-MOEA-II, on problems whose Pareto set is close to a product
of per-variable sets, what makes a real-coded crossover work is per-variable mixing: offspring take
whole variable values from either parent. In SBX that is the offspring swap,
`uniform_xover_probability = 0.5`.

**What would falsify it:** a non-mixing arm in the top places, one per mixing-side arm (six of 14 arms,
nine of 17, eleven of 19), under any Pareto-compliant indicator, or
the mixing split carrying a minority of the between-crossover sum of squares. Both tests run pooled
and on each of the five algorithms, so that no algorithm can hide a failure in another. `de_current`
counts as non-mixing (see *The two DE arms*).

**Result (2026-10-31, `analysis/tier.py`):** the claim as written is falsified. It passes pooled on all
four indicators, and fails in 9 of the 20 algorithm × indicator tests: `sax` reaches the top six under
NSGA-II and SPEA2 on all four indicators, and on IBEA the mixing split carries 0.403 of the sum of
squares under the additive epsilon. NSGA-III and AGE-MOEA-II pass all four. `sax` averages one variable
and swaps none, so it is non-mixing. This was on the 14-arm grid.

**Result on 17 arms (2026-11-09):** still falsified. The algorithm × indicator tests fail in the same
9 of 20 cells as on 14 arms, with the IBEA share under the additive epsilon now 0.411. The pooled test
now fails under IGD+, HV and R2, because `sbx_pisa` falls to 10th or 11th place and `sax` takes the
ninth. The mixing split's share of the sum of squares stays between 0.68 and 0.76 pooled.
`sbx_pisa` differs from `sbx_default_eta20` only in crossing every variable, and its mean rank is 1.6
to 2.8 places worse, more than the critical difference of 0.90 under every indicator. The swap alone
does not make SBX work: crossing about half the variables matters too. η = 20 against 30 moves the
mean rank by at most 0.3, for both `sbx_default` and `sbx_platemo`. Rescored on 149 instances against
fronts that include the two mutation probes (2026-11-12), the same 9 cells and the same three pooled
tests fail, the IBEA share is 0.415 and the pooled shares are 0.68 to 0.76.

**Result on 19 arms (2026-11-12):** falsified in every scope. The two mutation probes now count as
arms of their own, and both are on the mixing side because they swap whole variables. All 24 tests
fail (pooled and the five algorithms, under four indicators). The probes rank 12th or 13th
(`uniform_pm50`) and 17th to 19th (`uniform_pm100`), so two of the eleven top places go to arms off
the mixing side: `sax` in every scope except IBEA under the additive epsilon and HV, where
`de_current` takes it, and in most scopes `de` as well. The mixing split's share of the sum of squares
drops below one half everywhere: 0.33 to 0.42 pooled, at most 0.49 under any one algorithm and 0.09 to
0.16 under IBEA. Inside each block the other 17 arms keep their order, because regret is a monotone
rescaling within an instance; their mean ranks still shift by where the probes land. Per-variable swapping is therefore not enough on its own: uniform
crossover followed by heavy mutation swaps as much as `uniform` and ranks among the non-mixing arms.
The 17-arm tests above are as of `bf89d803`; `analysis/tier.py` no longer computes them.

## Factors

| Factor | Levels |
|---|---|
| Crossover | 19: `sbx_default`, `sbx_default_eta20`, `sbx_platemo`, `sbx_platemo_eta20`, `sbx_jmetal`, `sbx_pisa`, `sbx_no_uniform`, `blend`, `bex`, `bex_uniform`, `uniform`, `uniform_pm50`, `uniform_pm100`, `sax`, `de`, `de_current`, `pcx`, `composite`, `none` |
| Algorithm | 5: NSGA-III, NSGA-II, IBEA, SPEA2, AGE-MOEA-II |
| Instance | 149: 108 benchmark (64 DTLZ/WFG in four (m, n) blocks, 40 ZCAT, 4 rotated DTLZ) and 41 real-world (RE, rocket injector, river pollution, GAA ladder, MetallAppl) |
| Seed | 11 (0 to 10) |
| Mutation probe | 2 of the 19 arms, `uniform_pm50` and `uniform_pm100`: uniform crossover with polynomial mutation at rate 0.5 and 1.0 |

Held fixed:
- polynomial mutation at rate 1/n and index 20, except in the two mutation probes;
- Latin-hypercube initialisation and clip repair;
- population size by m (Das-Dennis counts, `population_size` in the config);
- NSGA-III with s-energy planar reference vectors and no adaptation;
- IBEA with kappa 0.05 and the additive epsilon indicator;
- binary tournaments on the selector's published fitness for every algorithm except NSGA-III;
- SPEA2's archive size equal to its population size;
- verbosity 2;
- no archive in the grid runs.

Every value is in `config/config.yaml`. Only the selection differs between algorithms:
`make_options` builds SPEA2 and AGE-MOEA-II from the same operators as the others rather than from
DESDEO's `spea2_options` and `agemoea2_options`.

**Seeds.** One integer per run seeds every random component, including NSGA-III's reference
vectors. The initial population depends only on the seed and the population size, so at one seed
every configuration starts from the same points, and comparisons across crossovers are paired by
seed.

**Budget.** One per instance, from phase 1 (see `xover/calibration.py`). This is a property of
NSGA-III with SBX, one of the compared configurations, and the paper says so. Every run is dumped at
21 budget fractions, so the analysis can be repeated at any shorter budget.

**Mutation probes.** Uniform crossover swaps whole variables and changes no value. SBX also moves the
values of the variables it crosses, half of them in `sbx_default_eta20` and all of them in `sbx_pisa`.
The probes ask whether that movement is doing a mutation's job, so that the swap is all SBX adds. Each
runs `uniform` with polynomial mutation (index 20, as SBX's η here) on half or on all of the
variables, in place of the 1/n mutation, and is compared with the SBX arm that moves the same share:
`uniform_pm50` with `sbx_default_eta20`, `uniform_pm100` with `sbx_pisa`. A tie in both pairs supports
the reading. The match is loose in one respect: polynomial mutation scales its step by the variable's
bounds, while SBX scales it by the distance between the parents, so SBX's step shrinks as the
population converges and the mutation's does not. A probe that loses to its SBX partner may be losing
on that alone. The probes are arms of the comparison like any other, on the mixing side, and the
matched contrasts are among `analysis/tier.py`'s ablations. They stay out of the step-size control
only: `offspring_step` is measured on the crossover's output before mutation, so for them it is
uniform crossover's step without their mutation.

**Probe result (2026-11-12):** the reading fails. Each figure below is the share of the 745
(instance, algorithm) blocks where the first arm has the lower regret at the final budget, for IGD+,
additive epsilon, HV and R2 in that order.

| Contrast | IGD+ | ε+ | HV | R2 |
|---|---|---|---|---|
| `uniform_pm50` beats `sbx_default_eta20` | 0.18 | 0.27 | 0.20 | 0.20 |
| `uniform_pm100` beats `sbx_pisa` | 0.14 | 0.23 | 0.17 | 0.17 |
| `uniform_pm50` beats `uniform` | 0.14 | 0.26 | 0.17 | 0.16 |
| `uniform_pm100` beats `uniform` | 0.10 | 0.19 | 0.11 | 0.09 |
| `sbx_default_eta20` beats `uniform` | 0.34 | 0.47 | 0.37 | 0.37 |

Both probes lose to their SBX partner in 73 to 86% of blocks, and in at least 69% under each algorithm. More
mutation makes uniform crossover worse at each step: among all 19 arms `uniform` has a mean rank of
5.4 to 7.2, `uniform_pm50` 11.2 to 12.2 and `uniform_pm100` 14.0 to 15.6, last or next to last with
`sbx_no_uniform`. So polynomial mutation on half or all of the variables does not do what SBX's
spread does. The last row shows that the spread is not what makes SBX good either. Plain `uniform`
beats `sbx_default_eta20` in 53 to 66% of blocks. That margin depends on the algorithm: under IGD+
`uniform` wins about 4 blocks in 5 under NSGA-II and SPEA2 and about half under AGE-MOEA-II and IBEA.
The bounds caveat above is still untested. `offspring_step` is measured on the crossover's output
before mutation, so the mechanism tables cannot compare the two step sizes.

**Reference fronts.** Merged archives of all 5 algorithms × 19 arms (17 crossovers and the 2 probes) at seeds 0 and 1, reduced
to a uniform subset of 1 500 to 4 000 points by greedy max-min selection. The ideal and nadir come
from the whole merged pool. Indicators normalise into that box. Hypervolume uses
`r = 1 + 1/H` (Ishibuchi et al. 2018).

## The two DE arms

| | `de` | `de_current` |
|---|---|---|
| Operator | DE/rand/1/bin (`DifferentialEvolutionCrossover`) | DE/current/1 (`DifferentialEvolutionCurrentCrossover`) |
| Mutant | `x_r1 + F (x_r2 - x_r3)` | `x_i + F (x_r1 - x_r2)` |
| Donors | Three distinct, none the target | Two distinct, either may be the target |
| F, CR | 0.5, 0.9 | 0.5, 1.0 (Li and Zhang 2009) |
| Forced component | One, from the mutant | None |

`de` keeps its name so that its results line up with the EMO paper's arm. `composite` keeps
DE/rand/1/bin as its DE member.

- **Tier.** `de_current` is non-mixing by mechanism: at CR = 1 every component comes from the mutant,
  so an offspring is its target plus a difference vector and takes no whole value from a second
  solution. It therefore enters the falsification test, and a top place for it counts against
  the claim. `de` stays in its own tier, because binomial crossover at CR = 0.9 does mix the mutant
  and the target per variable.
- **Not an ablation.** The arms differ in base vector, donor rule, CR and the forced component, so
  a gap between them cannot be pinned on any one of these.
- **Context.** DESDEO's MOEA/D-DE draws donors from a neighbourhood and replaces per subproblem. Here
  the operator runs inside generational algorithms with no neighbourhood, so `de_current` measures
  DE/current/1 and says nothing about MOEA/D-DE.
- **Mating pool.** NSGA-III passes the population, and every member is a target once. The other four
  algorithms pass only the tournament winners, so targets and donors all come from the mating pool.
  A pool can hold copies of one solution, and two such donors give a zero difference vector.
- **Mechanism columns.** The operators publish their targets as `PARENTS`, in no pairing, so
  `parent_separation` means nothing for either arm. `offspring_step` still reads as distance to the
  nearest target.

## SBX arms against their libraries (2026-10-30)

The operators match; two arms' parameters do not match what their library ships.

| Arm | Library | Operator | Parameters |
|---|---|---|---|
| `sbx_default` | pymoo 0.6.2 | Matches | Matches pymoo's NSGA-III (η = 30, pair rate 1). pymoo's NSGA-II, SPEA2 and AGE-MOEA-II use η = 15 and pair rate 0.9 |
| `sbx_platemo` | PlatEMO `948b670e`, `OperatorGA` | Matches | **Differs:** η = 30 here. PlatEMO's default is 20, and NSGA-II, NSGA-III, IBEA, SPEA2 and AGE-MOEA-II all call it with the default |
| `sbx_jmetal` | jMetal 7.9, `SBXCrossover` | Matches, including the swap of uncrossed variables | η = 20 matches. **Pair rate differs:** 1.0 here, 0.9 in jMetal's NSGA-II example |
| `sbx_no_uniform` | None, it is a control | Not pymoo at `prob_bin = 0` | |

Method: `checks/sbx_libraries.py` runs each library's own code (pymoo in-process, `OperatorGA.m` in
Octave, the jMetal jar through JPype) and DESDEO's operator on one parent pair, 200 000 times. The
pair covers wide interior gaps in both orders, values near and on both bounds, equal values and a
narrow gap. Per variable it compares each offspring's distribution, the gap between the offspring,
their mean and the share copied from each parent, Bonferroni-corrected within a row at 0.01.

- Five comparisons of matching settings show no difference, including `sbx_platemo` at η = 20 against
  PlatEMO's defaults and `sbx_jmetal` at pair rate 0.9 against jMetal at 0.9.
- Seven controls all differ. Truncated against untruncated SBX differs only on the variables near or
  on a bound, so `sbx_default` and `sbx_platemo` are the same operator in the interior.
- `sbx_no_uniform` differs from pymoo at `prob_bin = 0` only where parent 1 holds the larger value:
  pymoo's first offspring always takes the lower value there, and DESDEO's keeps to its own parent.
- jMetalPy copies uncrossed variables instead of swapping them, so the swap belongs to Java jMetal
  alone. This comes from reading its source; it was not run.

This checks the crossover operator only. It does not cover tournament selection or how a framework
handles an odd mating pool.

**Decided: `sbx_platemo` stays at η = 30.** The arm reproduces PlatEMO's operator but not what PlatEMO
ships, whose default is 20. At 30 it differs from `sbx_default` in `truncated` alone, so the pair is a
clean bound-handling contrast, and it stays the EMO paper's arm. Any statement about PlatEMO in the
paper has to name this difference.

## Changes from the EMO-paper version

| Change | Why |
|---|---|
| Budgets recalibrated | The committed `budgets.json` came from an earlier framework and arm naming, and cannot be regenerated by the old code |
| `re41_m3_f_2-f_3-f_4` re-admitted (149 → 150 instances) | It was excluded because IBEA divided by zero there; DESDEO's kappa fallback (`455fdf2d`) fixed that. Checked: 6 IBEA runs at 60 000 evaluations finish |
| Mutation factorial dropped (Study A: 3 extra mutations on 13 core instances) | Polynomial mutation only. **This removes the evidence for the draft's "mutation matters little" result** (ω² ≈ 0.0004 for mutation) |
| ZCAT and rotated-DTLZ side runs folded into the main design | One config, one result tree, no `XOVER_RUN` switch or per-run config hashes |
| Landscape features on one 200 d sample per instance | Replaces 11 seeded initial populations. The ICC (seed-stability) screen has nothing to measure and is gone |
| Raw run dumps kept, one tar per block | The cluster pipeline deleted each run after scoring it. Two files per run would take the study past Roihu's default 500 000 scratch files |
| SPEA2 and AGE-MOEA-II added (3 → 5 algorithms) | Two selection schemes the first three do not cover: strength fitness with nearest-neighbour density, and a survival score from the estimated front geometry |
| `de_current` added (13 → 14 crossovers) | DESDEO split DE/current/1 out of `DifferentialEvolutionCrossover` (`6b023c55`), which gives a DE variant with no per-variable mixing |
| `sbx_default_eta20`, `sbx_platemo_eta20` and `sbx_pisa` added (14 → 17 crossovers, 2026-11-07) | One-parameter contrasts for the SBX question. `sbx_platemo_eta20` is PlatEMO's operator at its shipped η = 20. `sbx_default_eta20` against `sbx_default` isolates η, and against `sbx_jmetal` isolates the swap of uncrossed variables. `sbx_pisa` is `sbx_default_eta20` with every variable crossed (`xover_probability` 1.0), so the two isolate the per-variable rate. Its name follows the request; its parameters have not been checked against PISA's code |
| `re37` removed (150 → 149 instances, 2026-11-12) | RE3-7 is the rocket injector problem without TW4. The subset sampler drew the same objective triple from the four-objective rocket injector (`rocket_injector_design_m3_TF_max-TT_max-Xcc_max`): same functions, same bounds. Counting it twice weighted one problem double in every rank. Its runs and tables stay on disk; the config no longer selects it and the analysis reads only selected instances. The 17-arm result is on 149; the 14-arm one was on 150 |
| Mutation probes `uniform_pm50` and `uniform_pm100` added (2026-11-12) | Tests whether SBX's spread does a mutation's job. `run.mutation_overrides` gives an arm its own mutation instead of the full crossover × mutation product. See *Mutation probes* |
| Mutation probes counted as arms (17 → 19 arms in the analysis, 2026-11-12) | First kept out of the operator comparison, then counted as full arms on request, so that the figures and the tests show them as operators of their own. The falsification tests now run on 19 arms with eleven on the mixing side |

## Verification

### 2026-10-30, after the merge and the additions

- `tests/test_legacy_equivalence.py` passes on the merged DESDEO (48 of 48). Both sides of that test
  use the same DESDEO, so it shows the pipeline still equals the old code. It does not show that the
  merge left the framework's own results unchanged, so numbers must not be carried over from runs on
  the older framework.
- It covers the 3 old algorithms and 13 old arms only. SPEA2, AGE-MOEA-II and `de_current` have no
  old code to match; `test_design.py` checks that the added algorithms differ from NSGA-II in
  `algorithm_name` and `selection` alone.
- Pilot: SPEA2 and AGE-MOEA-II × `sbx_default` and `de_current` on one instance of each of the 34
  base problems, for 6 generations each. All 136 runs finished with finite objectives.
- SPEA2 and AGE-MOEA-II also ran 60 generations on `wfg4_m8_n24` and 12 on `gaa_m11`. See *Cost*.
- The smoke configuration, now with four algorithms and both DE arms, runs phases 0 to 4 (82 jobs, no
  NaN in the four Pareto-compliant indicators). A second invocation does nothing. Phase 5 was not
  rerun: it depends on the problems alone.
- This branch had lost the `experiment` dependency group, so `slurm/run.sh` stopped before its first
  job. The group is back in `pyproject.toml`. Snakemake 9.27 caps its dependencies, which moves the
  branch's lock from `sqlmodel` 0.0.42 to 0.0.37 and `packaging` 26.3 to 25.0.
- Runs moved from two files per seed into one tar per block. On the smoke configuration all 16
  indicator and mechanism tables are identical to those from the separate files, exactly and row for
  row. That configuration has seeds 0 and 1 only. The full grid now scores seeds in numeric order
  instead of `s0, s1, s10, s2`, so its rows sort differently while their values do not change. A
  changed seed list still reruns the blocks, because the list is a param of `run_block`.
- The full dry run through `slurm/run.sh -n` plans 22 504 jobs:

  | Rule | Jobs |
  |---|---|
  | `run_block` | 10 500 |
  | `front_runs` | 10 500 |
  | `score` | 750 |
  | `calibrate`, `reference_front`, `sample_landscape`, `mo_features`, `ela_features` | 149 each |

### 2026-10-14, before them

- `tests/test_legacy_equivalence.py` passes against a worktree of `experiment/xover-selection-2` on
  the same DESDEO. The following are bit-identical to the old code's:
  - run dumps for 45 cells: all 13 crossovers × 3 algorithms on `dtlz2_m3_n12`, plus RE33, MetallAppl,
    GAA, river pollution, ZCAT and rotated DTLZ;
  - a reference front built from 8 archive runs;
  - every indicator on 6 runs × 21 fractions;
  - the calibration curve and budget at a reduced setting.
- The instance registry reproduces the old one's 150 slugs and specs, and `make_options` reproduces
  the old `EMOOptions` for all 624 algorithm × crossover × m × archive combinations.
- The smoke configuration runs every phase. A second invocation does nothing. Changing one
  crossover's parameter reruns only its 4 blocks and their 2 fronts and 4 scoring jobs.

## Cost

The grid has 115 500 runs and the fronts 21 000, against 64 350 and 11 700 before. At the EMO paper's
rates on 128 cores (61 776 grid runs in 140 min, 8 034 front runs in 1.21 h) that is about 7.5 hours,
before calibration and landscape features. Treat that as a floor, since those rates are from a
smaller instance set and AGE-MOEA-II costs more than the old algorithms.

The study writes about 34 000 result files, and Snakemake keeps one metadata file for each. With the
uv environments that comes to about 133 000 files on scratch, against a default quota of 500 000.
Two files per run would have been about 570 000.

Time per evaluation with `sbx_default` relative to NSGA-III on the same instance, on one core, from
single runs while other jobs shared the machine:

| Instance (population) | NSGA-II | IBEA | SPEA2 | AGE-MOEA-II |
|---|---|---|---|---|
| `dtlz2_m3_n12` (91) | 0.29 | 0.33 | 1.44 | 0.57 |
| `zcat3_l2_ps1_m3_n30` (91) | 0.97 | 0.99 | 0.90 | 1.10 |
| `dtlz2_m5_n12` (210) | 0.33 | 0.47 | 0.62 | 2.29 |
| `wfg4_m8_n24` (120) | 0.43 | 0.47 | 0.66 | 1.61 |
| `gaa_m11` (220) | 0.61 | 0.58 | 0.60 | 0.68 |

GAA's evaluator dominates it at about 1.7 s per 1 000 evaluations, so the algorithm hardly matters
there.

## Framework changes

None. Everything the study needs is on `master`, including `DifferentialEvolutionCurrentCrossover`,
SPEA2 and AGE-MOEA-II.
