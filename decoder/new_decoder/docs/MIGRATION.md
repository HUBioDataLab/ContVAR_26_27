# Standalone delivery to repository integration

Development now uses `decoder.new_decoder` from the repository root. The old
`contvar_decoder` namespace belongs to the separately archived standalone
delivery. No existing repository directory is moved, renamed or replaced.
All committed integration changes are confined to `decoder/new_decoder`.

## Preserved behavior

The four combinations, layer ordering, dimensions, parameter names, raw-score
subtraction, losses, label order and parallel decision rule are unchanged.
The historical checkpoint tag `contvar_go_decoder_starter_v1` is intentionally
retained to load existing checkpoints with `weights_only=True`. The tag does
not imply that a different architecture is silently loaded.

Single-mutation evidence is required in preparation and loading. Train
augmentation remains, evaluation augmentation is excluded, original partition
assignments are preserved, and multiple-source-label groups remain present
with separate reporting. No biological labels or reverse class labels are
invented. Future comparison readers, window/pocket inputs and abstention are
not enabled by this integration.

## Integration changes

- Centralized repository paths, feature constants and common CLI defaults.
- Added input/output type annotations, tensor contracts and module/function
  documentation; applied consistent Black formatting.
- Reused the existing identical audited GO reader instead of duplicating it.
- Separated deterministic evaluation/metrics from optimizer code.
- Accepted native root `data/splits/{layout}` directories in addition to the
  original ZIP artifacts, with explicit ambiguity/target checks.
- Kept new prepared outputs under `data/prepared/new_decoder`; older consumers
  of `data/prepared/*.tsv` are unaffected.
- Kept run outputs under this module's ignored `runs/` directory.
- Corrected the standalone train entry-point ordering: dataset loading is now
  defined before the single main invocation. Public train/predict commands are
  covered by an end-to-end test.
- Validated invalid hyperparameters before creating outputs, added explicit
  feature-store cleanup, and retained fingerprinted plans and checkpoints.

The root requirements and ignore rules are unchanged. Module-specific runtime
minimums and development tools are under this directory. Runtime package
minimums are separate from the preserved historical training environment.

## CLI path changes

| Standalone | Repository integration |
|---|---|
| `python -m contvar_decoder.train` | `python -m decoder.new_decoder.train` |
| `--package /path/to/encoder_package` | `--data-root /path/to/encoder_package/data` |
| `prepared_splits/{head}` | `data/prepared/new_decoder/{head}` by default |
| `source_splits/*-*.zip` | `data/splits/{layout}/*.tsv` or supplied ZIP source |
| `results/<run>` | `decoder/new_decoder/runs/<run>` by default |

`--data-dir` overrides the prepared root, independently of `--data-root` for
frozen input files. A supplied data root also accepts the legacy `go/` NPZ
location. Prediction works with the cached cohort, not arbitrary raw sequences.

The standalone archives and the original `results/full_seed42` directory were
left unchanged. Historical summaries remain historical; current integration
validation is recorded separately in `experiments/migration_validation.json`.
