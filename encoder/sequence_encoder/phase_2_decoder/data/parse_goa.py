"""
GO annotation parsing ve vocabulary oluşturma.

Kaynak: goa_2025-12-04_swissprot_noiea.tsv
GAF 2.2 formatı (başlıksız, tab-separated):
    col 2  → UniProt accession   (P04637)
    col 4  → Qualifier           (enables, involved_in, ...)
    col 5  → GO term ID          (GO:0003700)
    col 7  → Evidence code       (IDA, IBA, ND, ...)
    col 9  → Aspect              (F=MF, P=BP, C=CC)
    col 13 → Taxon               (taxon:9606)
"""

import json
from collections import defaultdict, Counter


# ND ve IEA evidence kodları düşük güvenilirlikli → hariç tutulur
EXCLUDE_EVIDENCE = {"ND", "IEA"}

# GAF 2.2: aspect başına geçerli qualifier'lar.
# Negatif etki qualifier'ları pozitif label olamaz.
# GAF 2.2 qualifier'larından hangilerinin pozitif label sayılacağı.
# NOT qualifier'ı (eski GAF 2.1) zaten EXCLUDE_EVIDENCE ile eleniyordu;
# burada sadece geçersiz/belirsiz ilişki türleri dışlanır.
# CAFA standardına göre negative_effect qualifier'ları da dahildir —
# "protein X süreci inhibe eder" yine de bir ilişkidir.
VALID_QUALIFIERS: dict[str, set] = {
    "F": {"enables", "contributes_to"},
    "P": {
        "involved_in",
        "acts_upstream_of",
        "acts_upstream_of_or_within",
        "acts_upstream_of_positive_effect",
        "acts_upstream_of_or_within_positive_effect",
        "acts_upstream_of_negative_effect",
        "acts_upstream_of_or_within_negative_effect",
    },
    "C": {"is_active_in", "part_of", "located_in", "colocalizes_with"},
}


def build_ancestor_map(obo_path: str, go_terms: set) -> dict[str, frozenset]:
    """
    Verilen GO term seti için her terimin GO hiyerarşisindeki
    tüm ancestor'larını döndürür.

    Sadece is_a + part_of kenarları kullanılır (CAFA standardı).
    obonet'te kenarlar child→parent yönünde kurulur, bu nedenle
    nx.descendants(g, term) gerçek GO ancestor'larını verir.
    """
    import obonet
    import networkx as nx

    print("[INFO] go-basic.obo yükleniyor...", flush=True)
    g_full = obonet.read_obo(obo_path)

    # Sadece is_a + part_of kenarlarını al (CAFA standardı)
    g = nx.DiGraph()
    g.add_nodes_from(g_full.nodes())
    for u, v, key in g_full.edges(keys=True):
        if key in ("is_a", "part_of"):
            g.add_edge(u, v)

    print(f"[INFO] OBO yüklendi: {g.number_of_nodes()} term, {g.number_of_edges()} kenar (is_a+part_of)", flush=True)

    ancestor_map: dict[str, frozenset] = {}
    missing = 0
    for term in go_terms:
        if term in g:
            ancestor_map[term] = frozenset(nx.descendants(g, term))
        else:
            ancestor_map[term] = frozenset()
            missing += 1

    if missing:
        print(f"[WARN] OBO'da bulunamayan GO term sayısı: {missing}", flush=True)
    return ancestor_map


def parse_goa_tsv(tsv_path: str, aspect: str = "F",
                  taxon_filter: str = None,
                  obo_path: str = None,
                  use_qualifier_filter: bool = True) -> dict[str, set]:
    """
    GO anotasyon TSV dosyasını parse eder.

    Args:
        tsv_path:     GOA TSV dosya yolu (GAF 2.2)
        aspect:       "F" (MF) | "P" (BP) | "C" (CC)
        taxon_filter: None → tüm organizmalar | "taxon:9606" → sadece insan
        obo_path:     go-basic.obo yolu. Verilirse her annotation'ın
                      GO hiyerarşisindeki tüm ancestor'ları da eklenir.
        use_qualifier_filter: False → VALID_QUALIFIERS kontrolü atlanır,
                      aspect'e uyan tüm qualifier'lar kabul edilir.

    Returns:
        {uniprot_id: {GO:XXXXXXX, ...}}
    """
    valid_quals = VALID_QUALIFIERS[aspect]
    annotations: dict[str, set] = defaultdict(set)

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 13:
                continue

            uniprot_id = cols[1]
            qualifier  = cols[3]
            go_id      = cols[4]
            evidence   = cols[6]
            asp        = cols[8]
            taxon      = cols[12].split("|")[0]

            if asp != aspect:
                continue
            if use_qualifier_filter and qualifier not in valid_quals:
                continue
            if evidence in EXCLUDE_EVIDENCE:
                continue
            if taxon_filter and taxon != taxon_filter:
                continue

            annotations[uniprot_id].add(go_id)

    if obo_path is not None:
        all_terms: set = set()
        for terms in annotations.values():
            all_terms.update(terms)

        print(f"[INFO] Propagation: {len(all_terms)} unique GO term için ancestor hesaplanıyor...", flush=True)
        ancestor_map = build_ancestor_map(obo_path, all_terms)

        for pid in annotations:
            expanded: set = set()
            for term in annotations[pid]:
                expanded.add(term)
                expanded.update(ancestor_map.get(term, frozenset()))
            annotations[pid] = expanded

        total_before = sum(len(v) for v in annotations.values())
        print(f"[INFO] Propagation tamamlandı. Ortalama GO term/protein: "
              f"{total_before / max(len(annotations), 1):.1f}", flush=True)

    return dict(annotations)


def build_go_vocab(annotations: dict[str, set], min_freq: int = 10,
                   save_path: str = None) -> dict[str, int]:
    """
    GO term → integer index eşlemesi oluşturur.

    min_freq altındaki GO terimleri atılır.
    NULL_FUNCTION sınıfı sona eklenir (hiçbir GO terimi atanmamış proteinler için).

    Args:
        annotations: parse_goa_tsv() çıktısı
        min_freq:    Minimum görülme sıklığı
        save_path:   JSON olarak kaydetmek için yol (None → kaydetme)

    Returns:
        {"GO:0003700": 0, ..., "NULL_FUNCTION": N}
    """
    counts: Counter = Counter()
    for go_terms in annotations.values():
        counts.update(go_terms)

    filtered = [(go, cnt) for go, cnt in sorted(counts.items()) if cnt >= min_freq]
    vocab: dict[str, int] = {go: idx for idx, (go, cnt) in enumerate(filtered)}
    vocab["NULL_FUNCTION"] = len(vocab)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab
