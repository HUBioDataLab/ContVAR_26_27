# ContVAR (2026–2027): Predicting Function-Specific Effects of Single Amino Acid Substitutions

**Hacettepe University — Department of Computer Engineering (BBM479)**  
**Laboratory:** [HUBioDataLab](https://github.com/HUBioDataLab)  
**Supervisor:** Prof. Dr. Tunca Doğan  
**Team Members:** Ömer Taha Gögen, İsmet Yiğit Arı, Gayem Oran, Nihat Guliyev

---

## 1. Project Overview

**ContVAR** predicts how a single amino acid substitution affects a specific protein function. Given a **wild-type (WT) protein**, a **single-residue variant**, and a target **Gene Ontology (GO) term**, the model classifies the functional consequence into one of three classes:
- **LOF (Loss of Function)** (`label_idx = 0`)
- **GOF (Gain of Function)** (`label_idx = 1`)
- **NEUTRAL (No detectable effect)** (`label_idx = 2`)

```text
(WT Protein, Variant, Target GO Term)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Multi-Scale Protein Encoders (encoder/)                           │
│    Extracts WT, Variant, and Delta representations across scales:    │
│    • Mutation Site [1280]          • Local Sequence Window [33, 1280]│
│    • 3D Spatial Pocket (Graph)     • Global Context [1280]           │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. GO-Conditioned Paired Decoder (decoder/)                          │
│    Combines WT, Variant, and comparison features with Anc2vec GO     │
│    embeddings [200] via GO-queried attention and local comparison    │
│    ──► Predicts: LOF | GOF | NEUTRAL                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```text
ContVAR_26_27/
├── encoder/                       # Protein representation learning & feature extraction
│   ├── sequence_encoder/          # Sequence-based encoder (ESM-2 650M: Phase 0 GO & Phase 1 DMS)
│   │   ├── phase_0_go_pretrained/            # Phase 0 GO semantic triplet pretraining
│   │   ├── phase_1_dms_fine_tuning/          # Phase 1 ProteinGym DMS fine-tuning
│   │   ├── phase_2_decoder/                  # WT GO term prediction & ESM utilities
│   │   ├── extract_multiscale_embeddings.py  # Extracts Global + Mut-Site + Local Window H5s
│   │   └── run_extract_server.sh             # Server launcher for multi-scale extraction
│   │
│   └── graph_encoder/             # Structure-based graph encoder (GATv2 + ESM-2 node features)
│       ├── contvar/                          # Residue graph builder, Phase 0/1 training, inference
│       └── phase2_decoder/                   # Graph-based Phase 2 GO decoder
│
├── decoder/                       # Variant–GO effect classification models
│   ├── phase_3_variant_decoder/   # Baseline decoders (MLP, Cascade, Parallel), data loaders & metrics
│   ├── new_decoder/               # New GO-conditioned attention & local-comparison (`LOCAL_PAIR`) decoder
│   └── example_load.py            # Quick verification script to load WT/VAR/Window/GO batches
│
├── data/                          # Local datasets & embeddings (git-ignored; see Section 4)
│   ├── embeddings/                # Protein H5 files (global, local, window) & GO Anc2vec NPZ
│   ├── prepared/                  # Ready-to-use easy/hard train, val, test TSVs & GO vocabs
│   ├── sequences/                 # UniProt sequence dictionaries (168,112 proteins)
│   └── splits/                    # V4 variant–GO dataset splits (multiclass, parallel, cascade)
│
└── docs/
    └── proposal.pdf               # Full BBM479 Project Proposal Report
```

---

## 3. Encoder → Decoder Data Contract

All extracted embeddings live in `data/embeddings/` and follow a unified key and shape convention so the decoder team can load any scale directly:

| Feature / File | H5 / NPZ File | Key Format (WT / Variant) | Shape & Dtype | Description |
|---|---|---|---|---|
| **Global (`WT_global`, `VAR_global`)** | `phase3_seq_phase1_global.h5` | WT: `{protein_id}`<br>VAR: `{protein_id}_{snp}` | `[1280]` (`float32`) | Mean-pooled whole-sequence representation from Phase 1 ESM-2 backbone. |
| **Mutation Site (`WT_mut`, `VAR_mut`)** | `phase3_seq_phase1_local.h5` | WT: `{protein_id}_{pos}`<br>VAR: `{protein_id}_{snp}` | `[1280]` (`float32`) | Contextual residue embedding at the exact 1-based mutated position `pos`. |
| **Local Window (`WT_window`, `VAR_window`)** | `phase3_seq_phase1_window.h5` | WT: `{protein_id}_{pos}`<br>VAR: `{protein_id}_{snp}` | `[33, 1280]` (`float16`) | **Unpooled** local sequence window $[p-16 \dots p+16]$ centered at index `16`. Any smaller window $k' \le 16$ can be sliced via `window[16-k' : 16+k'+1]`. |
| **Window Padding Mask (`WINDOW_mask`)** | `phase3_seq_phase1_window.h5` | `mask/{key}` | `[33]` (`bool`) | `True` for valid amino acids, `False` for N/C-terminal boundary padding. |
| **GO Query (`GO`)** | `go_terms_embeddings_anc2vec.npz` | Looked up by `matched_go_id` | `[200]` (`float32`) | Pretrained Anc2vec embedding of the queried GO term. |

---

## 4. Setup & How to Run

### Step 1: Clone the Repository & Install Dependencies
```bash
git clone https://github.com/HUBioDataLab/ContVAR_26_27.git
cd ContVAR_26_27

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Place Local Data Files in `data/`
Because large `.h5`, `.npz`, and `.json` files are excluded from Git via `.gitignore`, place the shared `data/` subdirectories (`embeddings/`, `prepared/`, `sequences/`, `splits/`) into the `ContVAR_26_27/data/` folder. See [data/README.md](data/README.md) for the exact file list.

### Step 3: Verify Data Loading (For Decoder Development)
Run `decoder/example_load.py` to verify that all embeddings (`WT_global`, `VAR_global`, `WT_mut`, `VAR_mut`, `WT_window`, `VAR_window`, `GO`, and deltas) load properly without loading the full H5 files into RAM:

```bash
# Load a sample batch of 4 rows from the hard train split:
python decoder/example_load.py --family hard --split train --n 4

# Optional: slice a smaller local window radius (e.g., k=10 -> 21 residues) on the fly:
python decoder/example_load.py --family hard --split train --n 4 --window_k 10
```

### Step 4: Running Baseline Decoder Experiments
```bash
python -m decoder.phase_3_variant_decoder.compare_four_decoders \
    --dataset_dir data/splits \
    --sequences_json data/sequences/uniprot_dict.json \
    --go_embeddings data/embeddings/go_terms_embeddings_anc2vec.npz \
    --global_h5 data/embeddings/phase3_seq_phase1_global.h5 \
    --local_h5 data/embeddings/phase3_seq_phase1_local.h5 \
    --family both --device auto
```

### Step 5: Extracting Multi-Scale Embeddings on the GPU Server (For Encoder Development)
```bash
bash encoder/sequence_encoder/run_extract_server.sh --cuda 6 --batch_size 8
```
