# GO-Conditioned Paired Decoder (`new_decoder`)

This directory contains the new decoder architectures developed in this project:
- **Scale Projections & Type Embeddings**: Mapping mutation-site, local sequence window (`[2k+1, 1280]`), pocket, and global representations into a shared hidden space.
- **GO-Conditioned Attention Readers**: Shared wild-type/variant cross-attention readers queried by the target GO embedding, and a separate comparison reader.
- **Aligned Local Residue Comparison (`LOCAL_PAIR`)**: Residue-by-residue comparison across the $[p-k, p+k]$ window prior to pooling:
  $$m_i = \text{MLP}\big([h_i^{\text{WT}};\, h_i^{\text{VAR}};\, h_i^{\text{VAR}} - h_i^{\text{WT}};\, h_i^{\text{WT}} \odot h_i^{\text{VAR}}]\big)$$
- **Multi-Scale Fusion & Classification Heads**: Predicting **LOF**, **GOF**, or **NEUTRAL** for the queried GO term.
