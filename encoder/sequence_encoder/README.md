# ContVAR — Sequence Model

Predicts the functional impact of protein variants using a multi-stage training pipeline built on top of ESM-2.

```
Raw protein sequences
        │
        ▼
 Phase 0 — GO Pre-training
   ESM-2 backbone fine-tuned with GO semantic triplet loss
        │  backbone_best.pt
        ▼
 Phase 1 — DMS Fine-tuning
   Backbone further fine-tuned on experimental fitness data
        │  phase1_backbone_best.pt
        ▼
 Phase 2 — GO Function Decoder
   FFN decoder maps protein embeddings → GO term probabilities
        │  decoder_best_{F|P|C}.pt
        ▼
 ContVAR_fitness score + LOF/GOF benchmark evaluation
```

The decoder lives in [phase_2_decoder/](phase_2_decoder/). Phases 0 and 1 are
optional: the pipeline ships three orchestrator scripts that build the decoder
on top of three different backbones, so the contribution of each fine-tuning
stage can be measured directly (see [Running the pipeline](#running-the-pipeline)).

---

## Stages

### Phase 0 — GO Pre-training

**Goal:** Teach the ESM-2 backbone to embed proteins with similar Gene Ontology annotations close together in embedding space.

**Entry point:** `phase_0_go_pretrained/train_go.py`

**How it works:**

1. A pre-built dataset (`go_pretrain_dataset.pkl`) contains protein sequences and their pairwise GO semantic similarity scores across three ontologies: Molecular Function (MF), Biological Process (BP), and Cellular Component (CC).
2. At each training step, a triplet is sampled: an **anchor** protein, a **positive** (semantically similar GO profile), and a **negative** (dissimilar GO profile). Negatives are bucketed by similarity score (0.1–0.2 = hard, 0.0–0.1 = easy) and sampled with balanced probability.
3. The model is `GOPretrainModel`: ESM-2 backbone + three independent projection heads (one per ontology). Each head is `Linear(D→256) → ReLU → Linear(256→256)` followed by L2 normalisation, projecting to a 256-dim unit sphere.
4. **Loss:** `TripletMarginLoss` (margin=0.3) applied independently per ontology on the corresponding head's outputs.
5. **Optimizer:** AdamW with separate learning rates — backbone: `5e-6`, heads: `1e-4`. Gradient accumulation (default 8 steps) gives an effective batch of 8 with `batch_size=1`.
6. Early layers of ESM-2 are frozen (default: first 25 of 33) to preserve general representations and reduce memory.
7. Validation runs mid-epoch every `val_every_samples` training samples and reports loss, triplet accuracy, MRR, alignment, and uniformity per ontology.
8. Best checkpoint is selected by validation triplet loss.

**Outputs** (written to `artifacts/pipeline_run/phase0/`):
- `backbone_best.pt` — ESM-2 weights only, used by Phase 1 and the Decoder.
- `full_model_best.pt` — backbone + all three projection heads.
- `backbone_last.pt` / `full_model_last.pt` — overwritten every epoch.

**Key config fields** (`pipeline_config.json → phase0`):

| Field | Default | Meaning |
|---|---|---|
| `epochs` | 30 | Training epochs |
| `batch_size` | 1 | Per-step batch (memory-limited) |
| `accum_steps` | 8 | Gradient accumulation → effective batch 8 |
| `freeze_layers` | 25 | Frozen ESM-2 transformer layers |
| `samples_per_epoch` | 10000 | Triplets drawn per epoch |
| `val_every_samples` | 2000 | Mid-epoch validation cadence |

---

### Phase 1 — DMS Fine-tuning

**Goal:** Further refine the backbone on experimental Deep Mutational Scanning (DMS) data so that the embedding space reflects actual fitness effects of mutations.

**Entry point:** `phase_1_dms_fine_tuning/phase1_dms_backbone_finetune.py`

**How it works:**

1. The Phase 0 backbone (`backbone_best.pt`) is loaded into ESM-2. Only the last N transformer layers are kept trainable (default: 8); all earlier layers are frozen.
2. Two new lightweight heads are added on top of the backbone:
   - **Global head** — `Linear(D→256) → ReLU → Linear(256→256)`, L2-normalised. Operates on mean-pooled whole-sequence embeddings.
   - **Local head** — same architecture. Operates on the token embedding at the mutated residue position only.
3. Training data comes from ProteinGym DMS substitution assays. Each sample is a triplet: wild-type (WT) sequence, a high-fitness variant (positive), and a low-fitness variant (negative).
4. **Loss:** two `TripletMarginLoss` terms summed — one from the global head (whole-protein shape) and one from the local head (mutation-site residue).
5. **Optimizer:** AdamW; backbone layers are trained at a lower rate than the new heads.
6. Proteins are split to prevent data leakage between train and validation. `--split_json` is **required**. [dms_data.py](phase_1_dms_fine_tuning/dms_data.py) auto-detects three split formats:
   - **family-level** — `{"family_to_split": {"<family>_wt_model": "train|val|test"}}`
   - **protein-level** — `{"protein_to_split": {"P12345": "train|val|test"}}`; the UniProt ID is recovered from the tokens of each family key, so the same `phase0_protein_split_removed_graphless.json` used by Phase 0 can drive Phase 1 directly.
   - **legacy mutation-level** — `{"families": {"<family>_wt_model": {"val_positives": [...], "val_negatives": [...]}}}`

**Outputs** (written to `artifacts/pipeline_run/phase1/`):
- `phase1_backbone_best.pt` — the further fine-tuned ESM-2 weights.

**Key config fields** (`pipeline_config.json → phase1`):

| Field | Default | Meaning |
|---|---|---|
| `epochs` | 20 | Training epochs |
| `batch_size` | 4 | Per-step batch |
| `last_trainable_layers` | 8 | Unfrozen ESM-2 layers |

---

### Phase 2 — GO Function Decoder

**Goal:** Train a lightweight classifier that predicts GO term annotations from a single protein embedding, and use it at inference time to score how much a variant preserves or changes the wild-type protein's function.

**Entry point:** [phase_2_decoder/train.py](phase_2_decoder/train.py)

#### Step 1 — Build embeddings

[phase_2_decoder/build_embeddings.py](phase_2_decoder/build_embeddings.py) runs the (optionally fine-tuned) backbone over every GO-annotated protein in the dataset and saves the results to an H5 file.

- Input: raw amino-acid sequences (up to 1022 residues, truncated if longer).
- ESM-2 produces per-token embeddings `[L, D]`; BOS and EOS tokens are excluded and the remainder is mean-pooled to a single vector `[D]` (D = 1280 for ESM-2 650M).
- Output: `{protein_id → [1280]}` stored in an HDF5 file.

#### Step 2 — Train the decoder

[phase_2_decoder/train.py](phase_2_decoder/train.py) trains `FFNDecoder` on the pre-computed embeddings.

**Architecture** ([phase_2_decoder/models/ffn_decoder.py](phase_2_decoder/models/ffn_decoder.py)):

```
[B, 1280]
  Linear(1280 → 512) → BatchNorm → ReLU → Dropout(0.30)
  Linear(512 → 1024) → BatchNorm → ReLU → Dropout(0.30)
  Linear(1024 → 512) → BatchNorm → ReLU → Dropout(0.20)
  Linear(512 → N_GO)
[B, N_GO]  ← raw logits
```

`N_GO` = number of GO terms appearing ≥10 times in the training data, plus one `NULL_FUNCTION` sentinel. The aspect (MF / BP / CC) is selected independently; a separate decoder is trained per aspect.

**GO annotation preprocessing ([phase_2_decoder/data/parse_goa.py](phase_2_decoder/data/parse_goa.py)):**

The raw GOA file (GAF 2.2) is processed in two steps before training:

1. **Qualifier filtering** — GAF 2.2 records the biological relationship between a protein and a GO term (e.g. `enables`, `involved_in`, `acts_upstream_of_or_within`, `part_of`). Only semantically valid qualifiers per aspect are retained; records with no direct functional relationship are dropped.
2. **Ancestor propagation** — GO's "true path rule": if a protein has term T, it implicitly has all ancestor terms of T in the hierarchy. `go-basic.obo` is loaded via `obonet` and for each annotated term, all ancestor terms (via `is_a` / `part_of` edges) are added to the protein's label set. On average, this expands the annotation set ~2.5× per protein (e.g. TP53: 34 → 82 MF terms).

**Training details:**

- **Loss:** `BCEWithLogitsLoss` — multi-label binary classification (a protein can have many GO terms).
- **Class weights:** `pos_weight[i] = n_negatives[i] / n_positives[i]`, clamped at 20. The `NULL_FUNCTION` class gets an extra 3× multiplier so the model learns to predict it for unannotated proteins.
- **Optimizer:** AdamW + cosine annealing LR.
- **Early stopping:** monitors validation mAP; stops after 15 epochs without improvement.
- No sigmoid is applied inside `forward()` — `BCEWithLogitsLoss` fuses it internally for numerical stability.
- **Test monitoring:** test split is evaluated every epoch alongside validation for observability; model selection (checkpoint saving, early stopping) uses only validation mAP.

#### Step 3 — Variant scoring (inference)

[phase_2_decoder/inference.py](phase_2_decoder/inference.py) / [phase_2_decoder/predict_variants.py](phase_2_decoder/predict_variants.py) produce the final variant fitness score.

```
1. Load decoder and go_vocab_{aspect}.json
2. For each variant in embeddings_variable.h5:
     a. Mean-pool the variant's token embeddings  →  [1280]
     b. var_probs = sigmoid(decoder(var_emb))      →  [N_GO]  (one probability per GO term)
     c. For each GO term:
          wt_label = 1 if term in WT's GOA annotations else 0
          if wt_label == 1 and var_prob < (1 - threshold):  status = LOSS
          if wt_label == 0 and var_prob > threshold:        status = GAIN
          else:                                              status = NEUTRAL
     d. ContVAR_fitness = mean(var_prob over WT-annotated terms)
                          1.0 = all WT functions retained
                          0.0 = complete functional loss
3. Write per-variant CSV with fitness, n_loss, n_gain, n_neutral
```

The WT's GO annotations come from the GOA database (`goa_2025-12-04_swissprot_noiea.tsv`), not from the decoder — the decoder only evaluates the **variant** sequence.

**Outputs** (written to `artifacts/pipeline_run/decoder/`):
- `decoder_best_{f|p|c}.pt` — trained decoder weights.
- `go_vocab_{f|p|c}.json` — GO term → index mapping.
- `predictions_{aspect}.csv` — per-variant fitness scores.

**Key config fields** (`pipeline_config.json → decoder`):

| Field | Default | Meaning |
|---|---|---|
| `aspect` | `"F"` | GO ontology: F=MF, P=BP, C=CC |
| `embedding_batch_size` | 4 | Proteins per batch during embedding |
| `max_seq_len` | 1022 | Sequence truncation length |

#### Step 4 — LOF/GOF benchmark evaluation

The decoder is validated against a curated **variant-specific GO benchmark**
(`data/raw/variant_specific_go_benchmark.tsv`), where each row is a known
loss-of-function (LOF) or gain-of-function (GOF) variant paired with a GO term.

1. [phase_2_decoder/build_variant_embeddings.py](phase_2_decoder/build_variant_embeddings.py) — embeds one FASTA per variant. The file stem `{protein_id}_{snp}` (e.g. `A0A061I403_E231G`) becomes the H5 key, matching what the evaluator expects.
2. [phase_2_decoder/benchmark_eval.py](phase_2_decoder/benchmark_eval.py) — scores each benchmark entry. With `--variant_h5`, it computes a **delta** between the variant and WT decoder scores for the matched GO term (LOF → delta < 0, GOF → delta > 0); without it, it falls back to WT-only scoring. Reports ROC-AUC, average precision, precision/recall/F1.
3. [phase_2_decoder/delta_analysis.py](phase_2_decoder/delta_analysis.py) — diagnostic that compares the LOF vs GOF delta distributions (overlap, AUROC, "dead signal" check) across the three backbones (`raw` / `phase_0` / `phase_1`), to see which fine-tuning stage actually sharpens the variant-effect signal.

---

## Running the pipeline

Three orchestrator scripts build the decoder on top of three different
backbones. They share `pipeline_config.json` and only differ in which
fine-tuning stages run first — this is the ablation that isolates the effect of
each backbone:

```bash
# Raw ESM-2 (no fine-tuning)  →  Decoder
python run_raw_esm_decoder.py --config pipeline_config.json

# Phase 0 (GO pre-train)      →  Decoder
python run_phase0_decoder.py --config pipeline_config.json

# Phase 0 + Phase 1 (full)    →  Decoder
python run_phase0_phase1_decoder.py --config pipeline_config.json

# Quick smoke test (tiny ESM-2, 1 epoch, 64 samples)
python run_phase0_phase1_decoder.py --config pipeline_config_smoke.json
```

Each script chains `build_embeddings.py → train.py → test_eval.py` in
`phase_2_decoder/`. Individual stages can also be run directly:

```bash
# Phase 0 only
python phase_0_go_pretrained/train_go.py \
    --dataset data/processed/go_pretrain_dataset.pkl \
    --split_file phase0_protein_split_removed_graphless.json \
    --esm_model esm2_t33_650M_UR50D \
    --out artifacts/phase0

# Phase 1 only (--split_json is required)
python phase_1_dms_fine_tuning/phase1_dms_backbone_finetune.py \
    --split_json phase0_protein_split_removed_graphless.json \
    --out_dir artifacts/phase1

# Decoder only (after embeddings exist)
python -m phase_2_decoder.train --aspect F
python -m phase_2_decoder.predict_variants --var_h5 embeddings_variable.h5 --aspect F

# Benchmark evaluation
python -m phase_2_decoder.benchmark_eval --aspect F \
    --embeddings_h5 artifacts/pipeline_run/decoder/raw_embeddings.h5 \
    --checkpoint artifacts/pipeline_run/decoder/decoder_best_f_raw.pt \
    --go_vocab_json artifacts/pipeline_run/decoder/go_vocab_f_raw.json
```

---

## Required data files

| File | Used by | Description |
|---|---|---|
| `data/processed/go_pretrain_dataset.pkl` | Phase 0 | Protein sequences + GO semantic similarity pairs |
| `data/raw/uniprot_dict.json` | Decoder embedding builder | UniProt accession → amino-acid sequence |
| `data/raw/goa_2025-12-04_swissprot_noiea.tsv` | Decoder (train + inference) | SwissProt GO annotations, GAF 2.2, IEA/ND excluded |
| `data/raw/go-basic.obo` | Decoder (train + inference) | GO ontology hierarchy for ancestor propagation |
| `data/raw/DMS_ProteinGym_substitutions/` | Phase 1 | ProteinGym DMS assay CSVs |
| `data/raw/variant_specific_go_benchmark.tsv` | Benchmark eval | LOF/GOF variants paired with matched GO terms |
| `data/raw/variant_specific_go_benchmark_filtered_fastas/` | Benchmark eval | One FASTA per benchmark variant (`{protein_id}_{snp}.fasta`) |
| `phase0_protein_split_removed_graphless.json` | Phase 0 + Phase 1 + Decoder | Train/val/test protein split |
| `dms_protein_split.json` | Phase 1 | DMS family-level split |
| `triplet_data.json` | Phase 1 | Pre-built DMS triplet index |

---

## ESM-2 model options

| Model | Parameters | Notes |
|---|---|---|
| `esm2_t6_8M_UR50D` | 8M | Fast, low memory — use for smoke tests |
| `esm2_t12_35M_UR50D` | 35M | Medium |
| `esm2_t30_150M_UR50D` | 150M | Large |
| `esm2_t33_650M_UR50D` | 650M | Best results, embed dim = 1280 |

Set `"esm_model"` in `pipeline_config.json` to switch. The decoder input dim is inferred automatically from the H5 file.

---

## Outputs summary

```
artifacts/pipeline_run/
├── phase0/
│   ├── backbone_best.pt        ← ESM-2 weights after GO pre-training
│   └── full_model_best.pt      ← backbone + 3 GO projection heads
├── phase1/
│   └── phase1_backbone_best.pt ← ESM-2 weights after DMS fine-tuning
└── decoder/
    ├── go_vocab_f.json          ← GO term → index (MF)
    ├── decoder_best_f.pt        ← trained FFN decoder (MF)
    └── predictions_f.csv        ← per-variant fitness scores
```
