"""Reuse the repository's audited Anc2vec reader without changing baselines.

This reader is identical to the reader used by the original standalone decoder.
It uses GO IDs rather than dataset row indices and restricts legacy pickle input
to the audited artifact. No separate GO loading implementation is introduced.
"""
from decoder.phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable

__all__ = ["GOEmbeddingTable"]
