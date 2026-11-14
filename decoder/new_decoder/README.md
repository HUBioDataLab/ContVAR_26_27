# GO-Conditioned Paired Decoder

Predict **LOF, GOF or NEUTRAL for a queried GO term** from frozen wild-type and
single-residue variant representations. This implementation integrates the
previous standalone two-scale decoder into the existing repository layout.
Existing encoders, baseline decoders, root configuration and shared data paths
are unchanged.

## Implemented scope

Each WT/VAR side supplies a mutation-site vector and a global vector, both
1280-D. The GO query is a frozen 200-D Anc2vec vector. Scale projections,
type embeddings and one shared attention reader produce GO-conditioned WT/VAR
summaries. The reader is initialized from scratch; no STAR-GO weights are loaded.

| Output | Pair processing | CLI options |
|---|---|---|
| Two binary heads | Separate shared scores, then VAR minus WT | `--head parallel --pair-mode score_difference` |
| Two binary heads | Joint WT / VAR / GO fusion | `--head parallel --pair-mode joint` |
| Three-class head | Separate shared scores, then VAR minus WT | `--head multiclass --pair-mode score_difference` |
| Three-class head | Joint WT / VAR / GO fusion | `--head multiclass --pair-mode joint` |

**Score difference means subtracting raw scorer outputs, not protein embeddings.**
The current `local.h5` input is a single mutation-site residue, not a sequence
window. The repository's window artifact is not consumed by this baseline yet.
Local-window/pocket integration, a separate comparison reader and aligned
`LOCAL_PAIR` are future extensions; see [architecture](docs/ARCHITECTURE.md).

## Directory and data contract

```text
decoder/new_decoder/
  config.py              repository paths, input constants, shared CLI defaults
  model.py               shared reader, four model combinations and losses
  data.py                read-only feature store and validated paired dataset
  go_embeddings.py       existing audited Anc2vec reader integration
  sources.py             native TSV directories or original split ZIPs
  prepare.py             filtering, exclusions and input-label review
  prepare_all.py         native target/cohort checks and manifest
  evaluation.py          deterministic prediction and metrics
  train.py               single-model/four-model validation-based training
  suite.py               12 distinct easy/hard fits, then final tests
  predict.py             saved-checkpoint readback on cached-feature splits
  report.py              completed-suite CSV/Markdown export
  docs/                  architecture, protocol and historical experiment records
  tests/                 model, data, policy and CLI integration tests
  runs/                  local outputs, ignored by Git
```

Large inputs stay in the existing root `data/` directory:

```text
data/embeddings/phase3_seq_phase1_global.h5
data/embeddings/phase3_seq_phase1_local.h5
data/embeddings/go_terms_embeddings_anc2vec.npz
data/sequences/uniprot_dict.json
data/splits/{multiclass,parallel,cascade}/*.tsv
```

The original `multiclass-*.zip`, `parallel-*.zip` and `cascade-*.zip` layouts
are also supported under a supplied `--source`. An external encoder data root
can be supplied with `--data-root`; its legacy `go/` location is supported.

Prepared tables default to **`data/prepared/new_decoder/{multiclass,parallel}`**.
This keeps the root `data/prepared/*.tsv` tables used by other tools untouched.
GO lookup uses `matched_go_id`, never the dataset's numeric `go_idx`.
Descriptions, source labels and evaluation flags are not model features.

## Setup and tests

Run commands from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r decoder/new_decoder/requirements-dev.txt
python -m pytest -c decoder/new_decoder/pyproject.toml decoder/new_decoder/tests -q
python -m black --check --config decoder/new_decoder/pyproject.toml decoder/new_decoder
```

Tests use small synthetic files; no encoder downloads or large shared data are
needed. Training uses the supplied frozen artifacts and does not update ESM-2.

## Prepare the correct native targets

```bash
python -m decoder.new_decoder.prepare_all
```

Use `--source /path/to/splits --data-root /path/to/data --out /new/prepared/path`
to keep shared inputs elsewhere. Preparation keeps source partition assignments
and labels, requires explicit single-mutation evidence, excludes augmented
validation/test rows, retains train augmentation and reports exclusions.
Single-mutation groups with multiple source labels are **retained and reported**.
Details are in the [data and experiment protocol](docs/PROTOCOL.md).

## Train and evaluate

One model, with checkpoint selection on validation:

```bash
python -m decoder.new_decoder.train \
  --family hard --head multiclass --pair-mode joint \
  --lambda-reverse 0 --seed 42 \
  --out decoder/new_decoder/runs/hard_multiclass_joint_seed42
```

The single-model command accesses test only with explicit `--evaluate-test`.
For the complete protocol:

```bash
python -m decoder.new_decoder.suite \
  --epochs 20 --patience 5 --seed 42 \
  --out decoder/new_decoder/runs/full_seed42
```

The suite fits four baseline combinations plus two joint reverse-loss controls
per family: **12 distinct fits**. Separate-score reversal is structural, so its
lambda control is not a duplicate training run. Checkpoints and joint lambda
are selected on validation Macro-F1; the selection is written before any test
predictions are evaluated. `--smoke` runs one epoch on limited subsets and is
only a software check. Use new output directories to preserve completed runs.

Read an existing compatible checkpoint without training:

```bash
python -m decoder.new_decoder.predict \
  decoder/new_decoder/runs/hard_multiclass_joint_seed42/best.pt \
  --family hard --split val \
  --out decoder/new_decoder/runs/validation_readback

python -m decoder.new_decoder.report \
  --suite decoder/new_decoder/runs/full_seed42 \
  --out decoder/new_decoder/runs/full_seed42_report
```

These commands accept `--data-root` and `--data-dir` where applicable. Inference
requires cached WT/VAR features; raw-sequence embedding extraction belongs to
the encoders and is not performed here.

## Decision rule and existing evidence

Parallel heads use independent sigmoid scores and the historical fixed **0.5**
cutoff: neither positive means NEUTRAL; otherwise the larger score wins,
including when both are positive. Their auxiliary neutral score is not a learned
third probability. Abstention, calibration and optimized thresholds are not
implemented. Multiclass uses one softmax with label order LOF / GOF / NEUTRAL.

The [experiment records](docs/experiments/README.md) contain the original
single-seed results and limitations, not a retraining claim for this refactor.
The joint parallel model won validation in both families. Multiclass joint had
slightly higher observed test Macro-F1, but the hard difference was approximately
0.0004 and does not establish superiority. An easy protein-GO reference rule
outperformed every new model; mutation-sensitivity controls remain necessary.

The original checkpoint format and state-dict keys are preserved. Refactor
validation is recorded separately from historical performance results.
