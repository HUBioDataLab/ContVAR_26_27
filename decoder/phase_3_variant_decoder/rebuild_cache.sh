#!/bin/bash
set -e
cd /home/mfurkan/contvar/CONTVAR-Sekans-model-
rm -rf data/processed/phase3_cache
CD=data/processed/phase3_cache
for FAM in hard easy; do
  for MODE in both global; do
    for E in raw_esm seq_phase0 seq_phase1; do
      python -m phase_3_variant_decoder.data.pair_dataset --build_cache --family $FAM --embedding $E --input_mode $MODE --global_h5 data/processed/phase3_${E}_global.h5 --local_h5 data/processed/phase3_${E}_local.h5 --cache_dir $CD
    done
    for P in phase0 phase1; do
      python -m phase_3_variant_decoder.data.pair_dataset --build_cache --family $FAM --embedding concat_$P --input_mode $MODE --global_h5 data/processed/phase3_seq_${P}_global.h5 --local_h5 data/processed/phase3_seq_${P}_local.h5 --global_h5_secondary data/processed/phase3_raw_esm_global.h5 --local_h5_secondary data/processed/phase3_raw_esm_local.h5 --cache_dir $CD
    done
  done
done
echo "20 CACHE HAZIR: $(ls $CD/*_wt.npy | wc -l) dosya, $(du -sh $CD | cut -f1)"
