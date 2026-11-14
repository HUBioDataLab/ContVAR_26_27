# Data Directory

All large datasets, embeddings, and sequence dictionaries are consolidated here and excluded from Git tracking.

```text
data/
├── embeddings/
│   ├── phase3_seq_phase1_global.h5        # [1280] global mean-pooled sequence embeddings
│   ├── phase3_seq_phase1_local.h5         # [1280] mutation-site residue embeddings
│   ├── phase3_seq_phase1_window.h5        # [33, 1280] local context window (p-16..p+16) + mask
│   └── go_terms_embeddings_anc2vec.npz    # [200] Anc2vec GO term embeddings
├── prepared/
│   ├── easy_{train,val,test}.tsv          # Prepared easy split tables
│   ├── hard_{train,val,test}.tsv          # Prepared hard split tables
│   └── {easy,hard}_go_vocab.json          # GO term vocabularies
├── sequences/
│   ├── uniprot_dict.json                  # Active full sequence dictionary (168,112 proteins: 166,613 + 1,499 V4)
│   ├── uniprot_dict_v1_original_166613.json # Backup of original GOA-scoped dictionary (166,613 proteins)
│   ├── uniprot_dict_v4_full_168112.json   # Explicit copy of full merged dictionary (168,112 proteins)
│   └── uniprot_dict_v4_missing_1499.json  # Only the 1,499 V4 proteins recovered from Swiss-Prot
└── splits/
    ├── multiclass/                        # V4 3-class (LOF / GOF / NEUTRAL) splits
    ├── parallel/                          # V4 parallel binary splits
    └── cascade/                           # V4 2-stage cascade splits
```
