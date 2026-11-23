# MF / window-7 experiment protocol

- This is the user-approved replacement experiment suite; historical sources/results remain in `../contvar_decoder_v1` unchanged.
- Source cohorts are exactly `prepared_old_subset_without_1499_proteins` and `prepared_full_all_proteins` in `../New_Decoder_Data_with_local_Context`.
- Preserve each source's easy/hard train/validation/test assignments, labels, conflicting labels and duplicate multiplicities.
- Select only is_combinatorial=False and GO namespace molecular_function. Exclude augmented rows only from validation/test. Do not generate augmentation.
- A few genuinely unusable input records may be excluded with explicit reasons, identically for all model conditions in that cohort/split.
- All embeddings are frozen. Use raw window positions 13:20, total length 7, including masks. Never reuse historical mean-33 features.
- Conditions: none and mean with parallel/multiclass x joint/score_difference; pair_mean and esim with joint only. Seed 42. Exactly 48 full training runs. No reverse penalty or additional seeds.
- Pair branches replace separate window means: compare mean vectors (one token) or aligned raw positions (seven tokens); GO reads the comparison memory, with no mean after the ESIM comparison.
- Select checkpoints only with validation macro-F1. Finish fitting/selection before final test evaluation.
- Evaluate each model on its own test and on the intersection of both cohorts' tests, preserving matching label/duplicate multiplicities and verifying train/validation exclusion.
- New reports/plots/tables must use only new runs and newly computed baselines. Existing results are historical archives, not report inputs.
