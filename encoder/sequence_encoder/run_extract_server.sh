#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# 1) Backbone checkpoint on the server
BACKBONE="${BACKBONE:-/media/ubuntu/8TB/mfurkan/contvar/checkpoints/phase1_backbone_best.pt}"

# 2) Sequence dictionary: prefers our new full 168,112-protein JSON (with the 1,499 missing V4 proteins added)
if [[ -f "$REPO_ROOT/data/sequences/uniprot_dict.json" ]]; then
  SEQUENCES_JSON="${SEQUENCES_JSON:-$REPO_ROOT/data/sequences/uniprot_dict.json}"
elif [[ -f "/media/ubuntu/8TB/omertaha/contvar_phase3/sequences/uniprot_dict_v4_full_168112.json" ]]; then
  SEQUENCES_JSON="${SEQUENCES_JSON:-/media/ubuntu/8TB/omertaha/contvar_phase3/sequences/uniprot_dict_v4_full_168112.json}"
else
  SEQUENCES_JSON="${SEQUENCES_JSON:-/media/ubuntu/8TB/mfurkan/contvar_data/raw/uniprot_dict.json}"
fi

# 3) V4 Dataset Splits directory
if [[ -d "$REPO_ROOT/data/splits/multiclass" ]]; then
  DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/data/splits/multiclass}"
elif [[ -d "/media/ubuntu/8TB/omertaha/contvar_phase3/datasets/splits_v4_neutral_augmentation/multiclass" ]]; then
  DATASET_DIR="${DATASET_DIR:-/media/ubuntu/8TB/omertaha/contvar_phase3/datasets/splits_v4_neutral_augmentation/multiclass}"
else
  DATASET_DIR="${DATASET_DIR:-/home/omertaha/contvar_v4_sequence_audit/splits_v4_neutral_augmentation/multiclass}"
fi

# 4) Output prefix for _global.h5, _local.h5, and _window.h5
OUT_PREFIX="${OUT_PREFIX:-/media/ubuntu/8TB/omertaha/contvar_phase3/embeddings/phase3_seq_phase1}"

exec python3 -u "$SCRIPT_DIR/extract_multiscale_embeddings.py" \
  --backbone "$BACKBONE" \
  --sequences_json "$SEQUENCES_JSON" \
  --dataset_dir "$DATASET_DIR" \
  --out_prefix "$OUT_PREFIX" \
  --window_k 16 \
  --window_dtype float16 \
  --center_long_sequences \
  --batch_size 4 \
  --cuda 0 \
  --resume \
  "$@"
