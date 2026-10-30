# Phase 6: the analyses (to be ported)

Phases 1-5 produce every input the analyses need. This note is the brief for porting the EMO-paper
analyses onto those outputs. The old scripts are on branch `experiment/xover-selection-2`, in
`experiment/xover-selection/`; paths below are relative to that directory.

## What changes for the analyses

| EMO-paper version | This pipeline |
|---|---|
| Three result trees (`results/`, `results_zcat/`, `results_rotated_dtlz/`) pooled by `XOVER_RUN=pooled` | One tree. Every table has a `suite` column (`dtlz_wfg`, `zcat`, `rotated_dtlz`, `real_world`) and a `family` column (`benchmark` for the first three, `real_world`) |
| `indicators/{instance}.parquet` | `indicators/{instance}/{algorithm}.parquet`. Same 30 columns, minus `config_hash`, plus `suite` |
| `mechanism/{instance}.parquet` | `mechanism/{instance}/{algorithm}.parquet`, same treatment |
| A `config_hash` check on every load | Nothing to check: Snakemake keeps the tables consistent with the config |
| Mutation factorial on 13 core instances (Study A) | **Gone.** Every run uses polynomial mutation |
| `features.parquet`: one row per instance × seed (11 initial populations) | One row per instance (one 200 d sample). No `seed` or `arm` column |

Load everything with one scan:

```python
import polars as pl
indicators = pl.scan_parquet("results/indicators/*/*.parquet").collect()
mechanism = pl.scan_parquet("results/mechanism/*/*.parquet").collect()
```

The pooled view in the old code always filtered to the polynomial plane first. Here every row is
already on it.

## Shared machinery to port first

- **`stats.py`**:
  - `omega_squared`, `generalized_omega_squared` (Olejnik and Algina, with the instance as a measured
    factor);
  - `friedman` and `pairwise_wilcoxon` (Holm), blocked by `CROSSOVER_BLOCKS = ("instance", "algorithm")`;
  - `critical_difference` (Nemenyi);
  - `within_block_ranks`, the primary ordering everywhere;
  - `block_rank_matrix`, `ordering_tau`, `split_tau_null`;
  - `performance_profile` (Dolan and Moré);
  - `oracle_gain`, `best_fixed_levels`;
  - the bootstrap power functions.

  Tests: `test_stats.py`.
- **`indicators.py`**:
  - `add_regret`: median over seeds, oriented by `LOWER_IS_BETTER`, min-max normalised within
    (instance, fraction);
  - `log_ratios`;
  - `RATIO_FLOOR`.

  The orientation constants (`LOWER_IS_BETTER`, `RANKABLE`, `PARETO_COMPLIANT`,
  `COMPARABLE_ACROSS_INSTANCES`) are already in `xover/indicators.py`.
- **`anytime.py`**:
  - `curves`;
  - `aocc`: trapezoid area over the 0.02-1.0 grid of `hv_relative`, divided by 0.98;
  - `time_to_target`: target = best + 10 % of the range, censored at 1.0.
- **`problem_properties.py`**: hand labels (front shape, separability, modality, Pareto-set linkage,
  objective band) assigned from each instance's base problem and m, and the `SPLITS` used by tier and
  splits. Its tests pin the old 149-instance set; the re-admitted `re41_m3_f_2-f_3-f_4` gets re41's
  labels.
- **`plotstyle.py`**: the palette and the `is_sbx` / `has_uniform` predicates that decide each arm's tier.
  `de_current` needs a colour of its own, and `analysis_pairs.TRACKED` should list it.

## The analyses, in the order to port them

Statistical defaults for every script: endpoint at `fraction == 1.0`, blocks of (instance,
algorithm), arms ordered by within-block ranks, the four Pareto-compliant indicators
(`igd_plus`, `epsilon_additive`, `hv_relative`, `r2`).

| Script (old) | Question | Method and output | In the EMO draft |
|---|---|---|---|
| `analysis_tier.py` | Does the mixing side outrank the rest? Each arm's tier is read from its options: `mixing` (`sbx_default`, `sbx_platemo`, `sbx_jmetal`, `bex_uniform`, `uniform`), `portfolio` (`composite`), `de`, `off` (`none`) or `non_mixing` (which includes `de_current`); the mixing side is the first two | Per indicator: cuts by algorithm and by every problem property; whether the top k places are all mixing-side; between-crossover sum-of-squares share of the split, with and without `none`/`de`/`de_current`/`composite`; the top-k and share tests repeated on SPEA2 alone and AGE-MOEA-II alone; paired ablations (e.g. `sbx_default` vs `sbx_no_uniform`); budget consistency (τ between fractions); step-size control (does offspring step explain the split?); mechanism by tier. → `tier.json` | §4.1 Fig. 1, Fig. 2; §4.4 ablations, step control, mechanism. **The claim.** |
| `analysis_splits.py` | Does the ordering survive problem properties? | For each split in `SPLITS`: block ranks on each side, τ between sides, permutation p (999 whole-instance shuffles); per algorithm and within m = 3; per-side Friedman/CD; performance profiles from `log_ratios`. → `splits.json` | §4.3 Table 2, Fig. 3, Fig. 4 |
| `analysis_rq7_algorithms.py` | Do algorithms differ, and does the best crossover depend on the algorithm? | Classical and generalised ω² of algorithm, crossover and their interaction, over 5 algorithms; Friedman over algorithms; per-algorithm orderings and τ between them; whether the pooled winner wins inside each algorithm. → `rq7_algorithms.json` | §4.1 ω² values, §4.2 |
| `analysis_rq10_indicators.py` | Does the answer depend on the indicator? | Per-indicator orderings, pairwise τ, agreement within the compliant four (PCA/PERMANOVA are companion-paper material). → `rq10_indicators.json` | §4.1 Fig. 2a |
| `analysis_rq8_anytime.py` | Does the ranking depend on when it is measured? | τ of each fraction's ranking against the final one; earliest fraction from which τ ≥ 0.95 holds to the end. → `rq8_anytime.json` | §4.1 "settles at 55 %" |
| `analysis_rq9_aocc.py` | Convergence speed vs final quality | AOCC on `hv_relative` against endpoint regret on `igd_plus`; framework spread (range of SBX-arm medians ÷ range of others). → `rq9_aocc.json` | §4.4 spread ratios |
| `analysis_rq2_factors.py` | How much does the crossover matter, and is the best one predictable? | On the plane: Friedman, Holm-Wilcoxon, CD, oracle gain against the single best arm, bootstrap power by number of instances and by base problem. The factorial-core block (ω² of crossover vs mutation, ART, `rq2_gate`) **has no data any more**. → `rq2_factors.json` | §4.1 oracle and SBS/VBS, §4.4 power; §3's mutation ω² |
| `analysis_rq1_frameworks.py` | Do the SBX variants frameworks ship differ? | Hand-curated variant table (V1-V8 across ten frameworks), to be read with PLAN.md's library check: `sbx_platemo` is PlatEMO's operator at η = 30, not its default 20, and both `sbx_default` and `sbx_jmetal` use a pair rate of 1; inheritance diagnostics driving `SimulatedBinaryCrossover.do` directly on DTLZ2 (identity retention, linkage, step size); performance of the SBX arms. → `rq1_frameworks.json` | §4.4 Table 3 |
| `analysis_pairs.py` | Do the switches in the ZCAT and rotated-DTLZ suites (separability, Pareto-set linkage, rotation) weaken the mixing tier? | Matched pairs (`zcat_arm.PAIRS`, `rotated_dtlz.PAIRS`): tier summary per instance, counts of weakened / strengthened / reversed. → `pairs.json` | scope and threats sections |
| `analysis_rq3_rule.py`, `analysis_rq4_transfer.py` | Can landscape features predict the best arm? Do benchmark and real-world problems differ? | Multi-output decision tree (depth 3) on the instance × crossover regret matrix, leave-one-base-out CV; benchmark ↔ real-world transfer; energy distance between feature distributions | Cut from the EMO paper (companion material). Inputs now come from phase 5 |

`analysis_rq5_mechanism.py` is superseded by tier's step-control and mechanism-by-tier sections.
`make_plots.py` and `render_plan_evidence.py` produced the old main-grid figures and are not used by
the paper; the paper's figures came from `paper/scripts/make_figures.py`.

### RQ numbering

The numbers disagree between file names, script docstrings and the paper. File names follow
`PLAN2.md`. Docstrings follow an older plan: `rq1` says RQ6, `rq2` says RQ1, `rq3` says RQ2/RQ5, `rq4`
says RQ3, `rq5` says RQ4. The paper uses neither and is organised as §4.1-4.4. Pick one scheme for
the journal version and rename the scripts to match.

## Pitfalls already paid for

Each of these produced wrong numbers at least once (bug numbers refer to `.wolf/buglog.json`).

1. **Filter before `add_regret`, never after** (bug-050, 052, 053). It normalises within (instance,
   fraction) over whatever rows it is given. With one mutation level this is automatic, but the same
   applies to any other filter, such as one algorithm or one suite. Still wrong in the old `rq1`,
   `rq3`, `rq4`, `rq5` and `make_plots`.
2. **Order arms by `within_block_ranks`, not by a median pooled over instances** (bug-052). Still
   wrong in the old `rq1` (`performance`) and `rq4` (`rank_agreement`). The old `rq3`/`rq4` also
   block on instance alone, from before the switch to (instance, algorithm) blocks on 2026-09-19.
3. **R2 is higher-is-better** (DESDEO reports the utility form). Never negate it before
   `add_regret` (bug-051).
4. **Rank only `RANKABLE`** (bug-045). `hv_front` is `hv_relative`'s numerator, and spacing, spread
   and uniformity have no direction.
5. **Never integrate regret.** Its scale depends on which arms are in the grid. AOCC uses
   `hv_relative`, the one indicator comparable across instances.
6. **`hv_relative` can be negative** when a run beats the reference front (bug-049). Floor it
   (`RATIO_FLOOR = 1e-4`) before taking logs.
7. **Report variance decompositions with and without the off levels** (`none`). The `none` arm
   inflates interaction terms.
8. ω² results serialised through `e.__dict__` (rq7, rq8, rq9) are unrounded, while the others use
   `.to_dict()`. Use one.
9. **`tier` matches `DifferentialEvolutionCrossover` by its exact name.** `de_current` is a different
   class, so it falls through to `non_mixing`. PLAN.md wants that outcome on purpose; give it an
   explicit branch, so the tier of the arm that can falsify the claim does not rest on a fall-through.

## Inputs the paper text relied on that this pipeline no longer produces

- **The mutation result.** The draft's setup section justifies fixing polynomial mutation by the
  Study A result, mutation ω² = 0.0004 among real operators (`paper/setup.md:42`,
  `PAPER-PLAN.md:122`). Options:
  - cite the EMO paper for it;
  - or add mutation levels back to `config.yaml` and run them on a core subset. Every block runs every
    listed mutation, so a core subset needs a small extension to `xover.design.run_blocks`.
- **Feature stability (ICC).** RQ3's gate used split-half reproducibility and ICC over 11 seeded
  samples. With one sample per instance, stability can only be checked by drawing a second sample
  (change `landscape.sample_seed` into a list).
