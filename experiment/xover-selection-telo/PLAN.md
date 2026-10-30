# Design record: crossover selection in EMO (TELO version)

Branch `experiment/xover-selection-telo`, cut from `master` at `8c7e9d6a` (DESDEO 2.8.0 plus two
documentation commits), with `master` merged in at `0553d414` on 2026-10-30 (`2126de30`). Never
merged back. The EMO-paper version of the study stays on `experiment/xover-selection-2`, in
`experiment/xover-selection/`.

## Claim

In NSGA-III, NSGA-II, IBEA, SPEA2 and AGE-MOEA-II, on problems whose Pareto set is close to a product
of per-variable sets, what makes a real-coded crossover work is per-variable mixing: offspring take
whole variable values from either parent. In SBX that is the offspring swap,
`uniform_xover_probability = 0.5`. The EMO-paper evidence: the mixing-side arms take the top six
places under all four Pareto-compliant indicators, and SBX without the swap (`sbx_no_uniform`) comes
13th of 13, below no crossover at all. That evidence covers NSGA-III, NSGA-II and IBEA only, so
SPEA2 and AGE-MOEA-II are out-of-sample tests of the claim.

**What would falsify it:** on the regenerated data, a non-mixing arm in the top six under any
Pareto-compliant indicator, or the mixing split carrying a minority of the between-crossover sum of
squares. Both tests also run on SPEA2 alone and on AGE-MOEA-II alone, so that the three older
algorithms cannot average away a failure in the new two. `de_current` counts as non-mixing (see
*The two DE arms*).

## Factors

| Factor | Levels |
|---|---|
| Crossover | 14: `sbx_default`, `sbx_platemo`, `sbx_jmetal`, `sbx_no_uniform`, `blend`, `bex`, `bex_uniform`, `uniform`, `sax`, `de`, `de_current`, `pcx`, `composite`, `none` |
| Algorithm | 5: NSGA-III, NSGA-II, IBEA, SPEA2, AGE-MOEA-II |
| Instance | 150: 108 benchmark (64 DTLZ/WFG in four (m, n) blocks, 40 ZCAT, 4 rotated DTLZ) and 42 real-world (RE, rocket injector, river pollution, GAA ladder, MetallAppl) |
| Seed | 11 (0 to 10) |

Held fixed:
- polynomial mutation at rate 1/n and index 20;
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

**Reference fronts.** Merged archives of all 5 algorithms × 14 crossovers at seeds 0 and 1, reduced
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
  solution. It therefore enters the falsification test, and a top-six place for it counts against
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

**Open: `sbx_platemo` at η = 30.** The arm reproduces PlatEMO's operator but not what PlatEMO ships.
At 30 it differs from `sbx_default` in `truncated` alone, which makes the pair a clean bound-handling
contrast. At 20 it would be PlatEMO's real default, and that is what the framework question asks
about. The config stays at 30 until this is decided. The value has to be settled before the grid
runs, because `test_legacy_equivalence.py` pins the arm to the EMO paper's.

## Changes from the EMO-paper version

| Change | Why |
|---|---|
| Budgets recalibrated | The committed `budgets.json` came from an earlier framework and arm naming, and cannot be regenerated by the old code |
| `re41_m3_f_2-f_3-f_4` re-admitted (149 → 150 instances) | It was excluded because IBEA divided by zero there; DESDEO's kappa fallback (`455fdf2d`) fixed that. Checked: 6 IBEA runs at 60 000 evaluations finish |
| Mutation factorial dropped (Study A: 3 extra mutations on 13 core instances) | Polynomial mutation only. **This removes the evidence for the draft's "mutation matters little" result** (ω² ≈ 0.0004 for mutation) |
| ZCAT and rotated-DTLZ side runs folded into the main design | One config, one result tree, no `XOVER_RUN` switch or per-run config hashes |
| Landscape features on one 200 d sample per instance | Replaces 11 seeded initial populations. The ICC (seed-stability) screen has nothing to measure and is gone |
| Raw run dumps kept | The cluster pipeline deleted each run after scoring it |
| SPEA2 and AGE-MOEA-II added (3 → 5 algorithms) | Two selection schemes the first three do not cover: strength fitness with nearest-neighbour density, and a survival score from the estimated front geometry |
| `de_current` added (13 → 14 crossovers) | DESDEO split DE/current/1 out of `DifferentialEvolutionCrossover` (`6b023c55`), which gives a DE variant with no per-variable mixing |

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
- The full dry run plans 22 504 jobs:

  | Rule | Jobs |
  |---|---|
  | `run_block` | 10 500 |
  | `front_runs` | 10 500 |
  | `score` | 750 |
  | `calibrate`, `reference_front`, `sample_landscape`, `mo_features`, `ela_features` | 150 each |

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
