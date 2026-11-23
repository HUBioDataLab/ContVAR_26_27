# Recorded experiments and integration validation

These records describe the **original standalone two-scale, seed-42 experiment
completed on 9 November 2026**. They are included for traceability; integrating
the implementation into this repository is not a new full training experiment.
Large source datasets, embedding files, predictions and checkpoints are not
committed to Git.

## Records

| File | Meaning |
|---|---|
| `seed42_summary.json` | Unmodified 12-fit historical summary, including original source hashes |
| `selection_before_test.json` | Historical validation-based lambda choices |
| `all_runs.csv` | All 12 historical conditions |
| `selected_models.csv` | Eight candidates after validation-based lambda selection |
| `train_only_baselines.json` | Train-majority, GO-only and protein-GO reference rules |
| `seed42_environment.json` | Recorded Python/platform/package environment |
| `seed42_requirements-lock.txt` | Original training dependency snapshot, not the shared repository lock |
| `parallel_coactivation_audit.json` | 13 November audit of existing parallel test predictions |
| `migration_validation.json` | Current model/checkpoint and preparation compatibility checks |

Historical `results/.../best.pt` paths identify the standalone delivery's
checkpoint locations. Those binary files are shared separately, not present in
this directory. Source hashes in the historical summary refer to the original
implementation; this formatted, integrated version has different source hashes.

## Main findings

Test Macro-F1 for the validation-selected λ=0 candidates:

| Model | Easy | Hard |
|---|---:|---:|
| Parallel score difference | 0.5699 | 0.5041 |
| Parallel joint | 0.6755 | 0.5661 |
| Multiclass score difference | 0.6017 | 0.4947 |
| Multiclass joint | 0.6871 | 0.5665 |

The overall validation winner was **parallel joint λ=0** in each family.
Observed test ranking does not retroactively change that choice. The hard
multiclass/parallel joint difference is approximately 0.0004, with only one seed.
Joint λ=0.1 reduced reverse MSE while lowering test Macro-F1 in all four tested
joint comparisons. Separate-score reversal error is structurally zero and is
not evidence of better classification.

The easy protein-GO train-majority reference achieved **0.7031**, above all new
models. Hard GO-only achieved **0.5536**, versus 0.5661 for the validation winner.
Mutation-sensitivity ablations, repeated seeds and confidence intervals remain
future work.

## Parallel coactivation

At the unchanged 0.5 cutoffs, simultaneous positive heads occurred in 2 easy
and 4 hard joint λ=0 test examples, versus 64 in each score-difference model.
Of those score-difference overlaps, 48 easy and 46 hard predictions were wrong;
33 easy and 15 hard cases had both scores at least 0.8. The audit JSON also
records joint λ=0.1 counts. These are model-example events, not necessarily
distinct biological variants across models.

Two independent sigmoids need not be exclusive. The larger-score decision
remains in the code. Abstention, calibration and tuned thresholds were discussed
but not implemented. Multiclass is a natural main candidate for a single-label
task, while retaining parallel as an experimental comparison; this design
preference does not rewrite the historical selection.

## Refactor verification

The integration preserves model state keys, parameter counts and numerical
equations. `migration_validation.json` records:

- Identical seeded initialization and outputs for all four combinations.
- All **12 original checkpoints** read back on their **complete validation
  partitions**, with every reported metric exactly equal to the stored values.
- All **12 native prepared tables** rebuilt from real original inputs and found
  byte-identical to the frozen baseline tables.

The model/data/CLI tests additionally cover native directories versus ZIPs,
single-mutation scope, retained multiple-label groups, augmentation treatment,
the actual train→predict commands, the 12-fit smoke suite and historical
parallel tie/overlap decisions. Smoke fits are software checks and are not
included in the historical performance table.

To regenerate a separate report from the recorded summary:

```bash
python -m decoder.new_decoder.report \
  --suite decoder/new_decoder/docs/experiments/seed42_summary.json \
  --out decoder/new_decoder/runs/historical_report
```
