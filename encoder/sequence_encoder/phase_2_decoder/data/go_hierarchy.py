"""
GO ontoloji hiyerarşisi — tahmin propagation matrisi için.

Sadece is_a + part_of ilişkileri kullanılır (CAFA standardı).
"""

import numpy as np


class GOHierarchy:

    def __init__(self, obo_path: str):
        import obonet
        import networkx as nx

        print("[INFO] go-basic.obo yükleniyor...", flush=True)
        g_full = obonet.read_obo(obo_path)

        # Sadece is_a + part_of kenarlarını al (obonet MultiDiGraph'ta key = relation type)
        self._g = nx.DiGraph()
        self._g.add_nodes_from(g_full.nodes())
        for u, v, key in g_full.edges(keys=True):
            if key in ("is_a", "part_of"):
                self._g.add_edge(u, v)

        print(
            f"[INFO] OBO yüklendi: {self._g.number_of_nodes()} term, "
            f"{self._g.number_of_edges()} kenar (is_a+part_of)",
            flush=True,
        )

    def build_propagation_matrix(self, go_vocab: dict[str, int]) -> np.ndarray:
        """
        [N_go, N_go] float32 propagation matrisi döndürür.

        prop_mat[i, j] = 1  →  GO terimi i'nin j'yi ancestor olarak içerdiği
                               (ya da i == j: diagonal 1).

        Kullanım (evaluation post-processing):
            preds_propagated = (preds @ prop_mat) > 0
            # preds: [N_samples, N_go] binary

        NULL_FUNCTION satır/sütunu sıfır bırakılır — propagate edilmez.
        """
        import networkx as nx

        n = len(go_vocab)
        mat = np.zeros((n, n), dtype=np.float32)
        null_idx = go_vocab.get("NULL_FUNCTION")

        for term, idx in go_vocab.items():
            if term == "NULL_FUNCTION":
                continue
            mat[idx, idx] = 1.0
            if term in self._g:
                for ancestor in nx.descendants(self._g, term):
                    anc_idx = go_vocab.get(ancestor)
                    if anc_idx is not None and anc_idx != null_idx:
                        mat[idx, anc_idx] = 1.0

        print(
            f"[INFO] Propagation matrisi: {n}x{n}, "
            f"non-zero: {int(mat.sum())}",
            flush=True,
        )
        return mat
