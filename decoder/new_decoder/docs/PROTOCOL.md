# Data and experiment protocol

## Sources and filtering

Use the supplied native `data/splits/multiclass`, `parallel` and `cascade`
directories or the matching original ZIPs. `prepare_all` verifies identical
source rows, native multiclass and binary targets, cascade stage-1 equivalence,
and the exact stage-2 LOF/GOF subset. Cascade is a reference, not extra training
data. Multiple matching ZIPs are rejected rather than silently selecting a
version. Extracted directory fingerprints cover all expected source tables.

Preserve source train/val/test assignments and labels. Easy and hard are
separate protocols. In order:

1. Remove augmented validation/test rows; keep train augmentation.
2. Reject combinatorial evidence, even when the SNP text looks like one change.
   An explicit `is_combinatorial=False` is required.
3. Validate one X123Y substitution, an existing sequence, position within the
   sequence and cache prefix (≤1022), matching WT residue, and changed amino acid.
4. Require every WT/VAR global and mutation-site feature: floating, finite, 1280-D.
5. Require the GO ID in the selected artifact. No random/zero-vector fallback.

Every exclusion is recorded. GO lookup uses `matched_go_id`, not `go_idx`.
The existing audited GO reader is reused without weakening legacy object-array
checks. Descriptions and flags are not input features.

Neutral augmentation was supplied in the source datasets. This implementation
does not crawl UniProt or regenerate augmentation. Its treatment is controlled
by the source `augmented` flag.

## Multiple source labels

The explicit project decision is to **retain source labels and report groups**.
Single-mutation `(protein_id,snp,GO)` groups with different labels are preserved,
flagged by `multiple_labels_for_input`, and evaluated separately. No majority
vote, relabeling or silent deletion is applied.

These groups are not automatically annotation errors: a broad GO term can
collapse different experimental subactivities. They differ from combinatorial
evidence (excluded) and simultaneous positive model heads (a prediction issue).

## Original available-cache cohorts

| Family | Train | Validation | Test | Train augmented |
|---|---:|---:|---:|---:|
| Easy | 41,720 | 4,677 | 4,515 | 1,731 |
| Hard | 41,872 | 4,973 | 4,915 | 1,408 |

These are historical counts for the original cache and source version, not a
promise that a newer sequence dictionary or cache gives the same exclusions.
Current preparation always records actual counts and fingerprints. New artifact
versions require distinct experiments.

Original multi-label groups: easy 86 keys / 180 rows; hard 94 keys / 196 rows.
Easy shares proteins across partitions but no protein+mutation pair. Hard is
protein-ID disjoint. Homology-level disjointness has not been established.

## Training and selection

Original settings: AdamW, lr 3e-4, weight decay 1e-4; batch 128; width 128;
four attention heads; joint fusion dropout 0.1; gradient norm clip 5; max 20
epochs; patience 5; seed 42; CPU, four threads; unweighted classes.

Per family, fit four baseline combinations and λ=0.1 controls for the two joint
models. The score-difference λ control is structurally redundant. Total: 12
fits. All fitting finishes before test evaluation. Best checkpoints and joint
lambda are selected by validation Macro-F1, preferring λ=0 on a tie. Selection
is persisted in `selection_before_test.json`, then test examples are evaluated.
The single-model CLI defaults to train/validation only; `--evaluate-test` is an
explicit final evaluation option.

The suite verifies prepared-table hashes before training. New run plans hash
the integrated module sources and its shared GO reader. Old run records retain
their historical source hashes; formatting/integration does not retroactively
change those training records.

Report class precision/recall/F1/AP/AUROC, Macro-F1, accuracy, MCC, confusion
matrix, signed reversal, score RMS, class swap, head overlap, and seen/unseen-GO
and multi-label subgroups. Unsupported metrics may be null. Parallel neutral
scores are auxiliary; CE and binary BCE loss magnitudes are not directly comparable.

## Interpretation boundaries

Historical results use one seed without confidence intervals. Easy protein-GO
train-majority rules beat the neural models; high F1 alone does not establish
mutation sensitivity. λ=0.1 reduced reversal error and lowered test F1 in all
four joint comparisons, which is not a statement about every possible lambda.

The previous test results have been inspected. New architecture, threshold or
calibration decisions belong on validation; future evaluation on that same test
must disclose its prior use. A proposed parallel overlap/abstention policy is
not implemented; abstention would not be the biological NEUTRAL class.
