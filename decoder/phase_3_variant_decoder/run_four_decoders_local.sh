#!/usr/bin/env bash
set -euo pipefail

decoder_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
decoder_workspace_dir="$(cd -- "$decoder_repo_dir/.." && pwd)"
decoder_kit_dir="$decoder_workspace_dir/gonderilecek_decoder_baslangic_v1"
decoder_python="${CONTVAR_DECODER_PYTHON:-$decoder_kit_dir/.venv/bin/python}"

cd "$decoder_repo_dir"
exec "$decoder_python" -u -m phase_3_variant_decoder.compare_four_decoders \
  --dataset_dir "$decoder_workspace_dir/splits_v4_neutral_augmentation" \
  --sequences_json "$decoder_kit_dir/data/sequences/uniprot_dict.json" \
  --go_embeddings "$decoder_kit_dir/data/go/go_terms_embeddings_anc2vec.npz" \
  --global_h5 "$decoder_kit_dir/data/embeddings/phase3_seq_phase1_global.h5" \
  --local_h5 "$decoder_kit_dir/data/embeddings/phase3_seq_phase1_local.h5" \
  --go_obo "$decoder_repo_dir/data/ontology/go-basic.obo" \
  --device auto --cpu_threads 4 \
  --out_dir "$decoder_workspace_dir/output/four_decoder_experiments" \
  "$@"
