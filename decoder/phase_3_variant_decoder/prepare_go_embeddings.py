"""Download, audit and export Anc2vec GO inputs (NumPy + Python stdlib only).

Run from the repository root; see docs/ANC2VEC.md for missing-term policies.
No labels are used to construct embeddings. No source TSV is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANC_COMMIT = "d828ad7ce9606e98cd349a5c48d6fc825edcf4c1"
GO_COMMIT = "214d7d0359359db8c47ec6f7a2fdce84900a5636"
ANC_BASE = f"https://raw.githubusercontent.com/aedera/anc2vec/{ANC_COMMIT}/"
SOURCES = {
    "embeddings.npz": (ANC_BASE + "anc2vec/data/embeddings.npz",
                       "9d0768115e077d54679f614b6e1617630a699c5b20bf332cd532129bda838517"),
    "anc2vec_go.obo": (ANC_BASE + "anc2vec/data/go.obo",
                       "e9205ff2483cf7e36e08ed1406f38ba60dc16fcc9dfc2f2adeb66a519d48ff09"),
    "anc2vec_LICENSE": (ANC_BASE + "LICENSE",
                        "5a797ab1159240423c90455a96ecee2d88a5fadb2399d909a61df72bcaa32fc5"),
    "go-current.obo": (
        f"https://raw.githubusercontent.com/geneontology/go-ontology/{GO_COMMIT}/src/ontology/go-edit.obo",
        "157cca77e9e9e9f4ce7ec3c2ef08381d2d408a5a2022e18aba168b69278769b7"),
    # The published graph contains inferred ancestry absent from go-edit.obo.
    # Cache once; record its data-version and actual hash in every audit.
    "go-basic.obo": ("https://purl.obolibrary.org/obo/go/go-basic.obo", None),
}
ROOT_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}
GO_RE = re.compile(r"GO:\d{7}\Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire_sources(directory: Path, offline: bool = False) -> dict:
    """Verified cache; downloads are atomic. Never unpickle an unverified file."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, expected) in SOURCES.items():
        path = directory / name
        if path.exists():
            if expected is not None and sha256(path) != expected:
                raise ValueError(f"Checksum mismatch: {path}. Move the corrupt file and retry.")
        else:
            if offline:
                raise FileNotFoundError(f"Offline cache missing: {path}")
            print(f"[DOWNLOAD] {name}", flush=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
                    temporary = Path(f.name)
                    request = urllib.request.Request(url, headers={"User-Agent": "ContVAR-GO-audit/1.0"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        while block := response.read(1024 * 1024):
                            f.write(block)
                if expected is not None and sha256(temporary) != expected:
                    raise ValueError(f"Downloaded checksum mismatch: {url}")
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        manifest[name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
    return manifest


@dataclass
class Term:
    id: str = ""
    name: str = ""
    namespace: str = ""
    obsolete: bool = False
    alt_ids: list[str] = field(default_factory=list)
    replaced_by: list[str] = field(default_factory=list)
    consider: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)


def read_obo(path: Path) -> tuple[dict[str, Term], dict[str, str], str]:
    """Only is_a/part_of for optional ancestor proxies; retain obsolete stanzas."""
    terms, version, term = {}, "", None
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("data-version: "):
                version = line.split(": ", 1)[1]
            if line.startswith("["):
                if term is not None and term.id:
                    terms[term.id] = term
                term = Term() if line == "[Term]" else None
                continue
            if term is None or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            value = value.split(" ! ", 1)[0]
            if key in ("id", "name", "namespace"):
                setattr(term, key, value)
            elif key == "is_obsolete":
                term.obsolete = value == "true"
            elif key in ("alt_id", "replaced_by", "consider", "is_a"):
                attr = {"alt_id": "alt_ids", "is_a": "parents"}.get(key, key)
                getattr(term, attr).append(value.split()[0])
            elif key == "relationship" and value.startswith("part_of "):
                term.parents.append(value.split()[1])
    if term is not None and term.id:
        terms[term.id] = term
    aliases = {a: t.id for t in terms.values() for a in t.alt_ids}
    return terms, aliases, version


def load_pretrained(path: Path) -> dict[str, np.ndarray]:
    # Upstream is an object dictionary, which requires pickle. Restrict this
    # exception to the exact, pinned upstream bytes, then export pickle-free NPZ.
    if sha256(path) != SOURCES["embeddings.npz"][1]:
        raise ValueError("Refusing to unpickle an unverified Anc2vec archive")
    with np.load(path, allow_pickle=True) as archive:
        vectors = archive["embds"].item()
    if not isinstance(vectors, dict) or len(vectors) != 44261:
        raise ValueError("Unexpected upstream Anc2vec dictionary")
    for go, vector in vectors.items():
        a = np.asarray(vector)
        if not GO_RE.fullmatch(go) or a.shape != (200,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid source vector: {go}")
    return {go: np.asarray(v, dtype=np.float32) for go, v in vectors.items()}


def read_rows(directory: Path) -> tuple[list[dict], dict]:
    rows, manifest = [], {}
    required = {"protein_id", "snp", "matched_go_id", "label"}
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            stage1 = directory / f"stage1_{family}_{split}.tsv"
            stage2 = directory / f"stage2_{family}_{split}.tsv"
            if stage1.exists() or stage2.exists():
                from phase_3_variant_decoder.data.variant_dataset import (
                    _read_and_validate_cascade_split,
                )
                frame = _read_and_validate_cascade_split(directory, family, split)
                source_paths = (stage1, stage2)
                source_rows = frame.drop(columns="split").to_dict("records")
            else:
                path = directory / f"{family}_{split}.tsv"
                with path.open(encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    if not required.issubset(reader.fieldnames or []):
                        raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
                    source_rows = list(reader)
                source_paths = (path,)
            for index, row in enumerate(source_rows):
                if not GO_RE.fullmatch(row["matched_go_id"]):
                    raise ValueError(f"{source_paths[0]}:{index + 2}: invalid GO ID")
                if row["label"] not in ("LOF", "GOF", "NEUTRAL"):
                    raise ValueError(f"{source_paths[0]}:{index + 2}: invalid label")
                rows.append({**row, "family": family, "split": split, "row_index": index})
            for path in source_paths:
                manifest[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return rows, manifest


def sequence_eligible(rows: list[dict], path: Path, max_seq_len: int):
    """Audit the phase-3 sequence filter chain without modifying input rows."""
    raw = json.loads(path.read_text())
    sequences = ({k: str(v) for k, v in raw.items()} if isinstance(raw, dict)
                 else {r["Entry"]: r["Sequence"] for r in raw if r.get("Sequence")})
    eligible, excluded, counts = [], [], defaultdict(Counter)
    for row in rows:
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", row["snp"])
        seq = sequences.get(row["protein_id"])
        if match is None:
            reason = "malformed_snp"
        elif seq is None:
            reason = "missing_sequence"
        elif int(match[2]) > max_seq_len:
            reason = "beyond_truncation"
        elif not 1 <= int(match[2]) <= len(seq):
            reason = "position_out_of_bounds"
        elif seq[int(match[2]) - 1] != match[1]:
            reason = "wt_residue_mismatch"
        else:
            reason = "kept"
        counts[row["family"]][reason] += 1
        if reason == "kept":
            eligible.append(row)
        else:
            excluded.append({**row, "reason": reason})
    return eligible, excluded, {
        "path": str(path.resolve()), "sha256": sha256(path), "sequence_count": len(sequences),
        "max_seq_len": max_seq_len, "counts": {k: dict(v) for k, v in counts.items()},
        "drop_contradictory": False,
    }


def resolve_identity(go: str, vectors: dict, terms: dict, aliases: dict) -> tuple[str, str]:
    """Prefer the original historical vector; never interpret 'consider' as identity."""
    if go in vectors:
        return go, "direct"
    current, seen, changes = go, set(), []
    while current not in seen:
        seen.add(current)
        if current in aliases:
            current = aliases[current]
            changes.append("alt_id")
        else:
            term = terms.get(current)
            if term is None or not term.obsolete or len(term.replaced_by) != 1:
                break
            current = term.replaced_by[0]
            changes.append("replaced_by")
        if current in vectors:
            return current, "+".join(changes)
    return current, "missing"


def nearest_ancestors(go: str, vectors: dict, terms: dict, aliases: dict) -> tuple[list[str], int]:
    """First BFS level with available, non-root, active ancestors of the same aspect.

    This is an approximation, NOT an Anc2vec vector learned for the query term.
    Do not cross obsolete terms or regulators; do not use generic GO root vectors.
    """
    go = aliases.get(go, go)
    start = terms.get(go)
    if start is None or start.obsolete:
        return [], 0
    frontier, seen, depth = set(start.parents), {go}, 1
    while frontier:
        level = {aliases.get(g, g) for g in frontier} - seen
        seen.update(level)
        level = {g for g in level if g in terms and not terms[g].obsolete
                 and terms[g].namespace == start.namespace and g not in ROOT_TERMS}
        candidates = sorted(g for g in level if g in vectors)
        if candidates:
            return candidates, depth
        frontier = {p for g in level for p in terms[g].parents}
        depth += 1
    return [], 0


def build_records(go_ids, vectors, terms, aliases, graph_terms=None) -> dict:
    records = {}
    for go in sorted(go_ids):
        canonical, status = resolve_identity(go, vectors, terms, aliases)
        term = terms.get(aliases.get(go, go))
        graph = terms if graph_terms is None else graph_terms
        ancestors, depth = nearest_ancestors(canonical, vectors, graph, aliases) if status == "missing" and not (term and term.obsolete) else ([], 0)
        records[go] = {
            "go_id": go, "name": term.name if term else "",
            "namespace": term.namespace if term else "unknown",
            "ontology_status": "unknown" if term is None else "obsolete" if term.obsolete else "active",
            "identity_status": status, "canonical_id": canonical,
            "source_ids": [canonical] if status != "missing" else [],
            "proxy_candidates": ancestors, "proxy_distance": depth,
            "replaced_by": term.replaced_by if term else [],
            "consider": term.consider if term else [],
        }
    return records


def group_stats(rows, records, covered) -> dict:
    ids = {r["matched_go_id"] for r in rows}
    exact = {g for g in ids if records[g]["identity_status"] != "missing"}
    proxy = {g for g in ids if records[g]["proxy_candidates"]}
    n, covered_n = len(rows), sum(r["matched_go_id"] in covered for r in rows)
    return {
        "rows": n, "terms": len(ids),
        "direct_terms": sum(records[g]["identity_status"] == "direct" for g in ids),
        "identity_resolved_terms": len(exact),
        "identity_resolved_rows": sum(r["matched_go_id"] in exact for r in rows),
        "proxy_possible_terms": len(proxy),
        "proxy_possible_rows": sum(r["matched_go_id"] in proxy for r in rows),
        "obsolete_terms": sum(records[g]["ontology_status"] == "obsolete" for g in ids),
        "obsolete_rows": sum(records[r["matched_go_id"]]["ontology_status"] == "obsolete" for r in rows),
        "covered_terms": len(ids & covered), "covered_rows": covered_n,
        "unresolved_terms": len(ids - covered), "unresolved_rows": n - covered_n,
        "covered_row_percent": round(100 * covered_n / n, 4) if n else 0,
    }


def write_tsv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items() if k in fields})


def write_markdown(path: Path, report: dict, records: dict):
    g = report["groups"]
    total = g["all_split_rows"]
    lines = ["# Anc2vec kapsam analizi", "",
             f"Üretim: {report['created_utc']} · Anc2vec boyutu: {report['dimension']}", "",
             "Bu ölçümler ham TSV satırları içindir; sekans ve pozisyon filtreleri henüz uygulanmadı.",
             "Easy ve hard alternatif split aileleridir; toplam satır sayısı bağımsız örnek sayısı değildir.", "",
             f"- Benzersiz GO: **{total['terms']}**",
             f"- Doğrudan hazır vektörü olan: **{total['direct_terms']}**",
             f"- Kimlik eşlemeleri dahil gerçek vektörü olan: **{total['identity_resolved_terms']}**",
             f"- Ata yaklaşımı mümkün ek terim: **{total['proxy_possible_terms']}**",
             f"- Bu çıktıda kapsanan: **{total['covered_terms']}**; çözülemeyen: **{total['unresolved_terms']}**",
             f"- Güncel editör ontolojisinde obsolete: **{total['obsolete_terms']}**",
             f"- Ata yaklaşımı etkin: **{report['ancestor_proxy_enabled']}**", "",
             "## Split bazında kapsam", "",
             "| Split | Satır | Gerçek vektörlü satır | Export kapsamı | Kapsam % | Eksik satır |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            key = f"{family}/{split}"
            s = g[key]
            lines.append(f"| {key} | {s['rows']} | {s['identity_resolved_rows']} | {s['covered_rows']} | {s['covered_row_percent']:.2f} | {s['unresolved_rows']} |")
    lines += ["", "## Etiket bazında eksiklik", "",
              "| Split / sınıf | Satır | Export dışında |",
              "|---|---:|---:|"]
    for key, s in g.items():
        if key.count("/") == 2 and key.rsplit("/", 1)[-1] in ("LOF", "GOF", "NEUTRAL"):
            lines.append(f"| {key} | {s['rows']} | {s['unresolved_rows']} |")
    if report.get("sequence_audit"):
        lines += ["", "## Sekans filtrelerinden sonraki kapsam", "",
                  "Sekans bulunması, 1022 (veya --max-seq-len) sınırı, pozisyon ve WT rezidü eşleşmesi uygulanır.",
                  "Çelişkili etiket filtresi uygulanmaz. Embedding exportu ham terimlerin kapsamını korur.", "",
                  "| Split | Kalan satır | GO sayısı | Gerçek vektörlü satır | Export kapsamı % | Eksik satır |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key, s in report["eligible_groups"].items():
            if key.count("/") == 1:
                lines.append(f"| {key} | {s['rows']} | {s['terms']} | {s['identity_resolved_rows']} | {s['covered_row_percent']:.2f} | {s['unresolved_rows']} |")
    lines += ["", "## Hazır dosyada bulunmayan en sık 20 terim", "",
              "Satır sayıları iki split ailesinin toplamıdır; birbirinden bağımsız gözlemler değildir.", "",
              "| GO | Ad | Satır | Ontoloji durumu | Export durumu |",
              "|---|---|---:|---|---|"]
    missing = sorted((r for r in records.values() if r["identity_status"] == "missing"),
                     key=lambda r: (-r["rows_across_families"], r["go_id"]))
    for r in missing[:20]:
        name = r["name"].replace("|", "/")
        lines.append(f"| {r['go_id']} | {name} | {r['rows_across_families']} | {r['ontology_status']} | {r['export_status']} |")
    lines += ["", "## Yorum ve kullanım", "",
              "Hazır dosyada bulunmayan bir terimin ata vektörünü kullanmak o terimin gerçek Anc2vec embedding'ini üretmez.",
              "Aynı ataya eşlenen farklı GO terimleri aynı vektöre dönüşebilir; fonksiyon ayrımı kaybolabilir.",
              f"Bu exportta aynı kaynak vektör(ler)ini paylaşan {len(report['shared_source_groups'])} grup var (coverage.json).",
              "Obsolete terimlerin mevcut tarihsel vektörleri korunur ve işaretlenir; bu onların güncel geçerliliğini kanıtlamaz.",
              "`consider` önerileri otomatik eşdeğerlik olarak kullanılmaz. Eksik terimler için sıfır/rastgele vektör üretilmez.",
              "Tam ve öğrenilmiş kapsam için sürümü seçilmiş ontolojide tüm Anc2vec uzayını yeniden eğitmek gerekir.",
              "Sadece eksikleri ayrı eğitip eski vektörlerle birleştirmek aynı koordinat uzayını garanti etmez.",
              "Yeni uzayda da obsolete terimler için biyolojik kimlik/etiket incelemesi gerekir.", "",
              "## Kaynaklar ve tekrar üretilebilirlik", "",
              f"- Anc2vec commit: `{report['anc2vec_commit']}`",
              f"- Hazır vektörlerin ontolojisi: `{report['pretrained_go_version']}`",
              f"- Kimlik denetimi GO edit commit: `{report['go_commit']}`",
              f"- Ata grafı: `{report['ancestor_graph_version']}`",
              "- Tüm kaynak ve veri SHA-256 değerleri: `coverage.json`.",
              "- Terim düzeyindeki sonuçlar: `term_audit.tsv`; eksikler: `unresolved_terms.tsv` / `unresolved_rows.tsv`.",
              "- Tüm etiket, kaynak ve aspect kırılımları: `coverage_by_group.tsv`.",
              "- Model girdisi: `go_embeddings.npz` (pickle gerektirmez).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare(dataset_dir: Path, source_dir: Path, out_dir: Path, *, offline=False,
            ancestor_proxy=False, missing="error", sequences_json=None, max_seq_len=1022) -> dict:
    rows, datasets = read_rows(dataset_dir)
    sources = acquire_sources(source_dir, offline)
    vectors = load_pretrained(source_dir / "embeddings.npz")
    historical, historical_aliases, historical_version = read_obo(source_dir / "anc2vec_go.obo")
    terms, current_aliases, current_version = read_obo(source_dir / "go-current.obo")
    graph_terms, _, graph_version = read_obo(source_dir / "go-basic.obo")
    if len(graph_terms) < 10000 or not graph_version:
        raise ValueError("Invalid or incomplete published go-basic ontology")
    # Current official IDs take precedence over historical alternative mappings.
    aliases = {k: v for k, v in historical_aliases.items() if k not in terms}
    aliases.update(current_aliases)
    records = build_records({r["matched_go_id"] for r in rows}, vectors, terms, aliases, graph_terms)
    selected, selected_status = {}, {}
    for go, record in records.items():
        if record["identity_status"] != "missing":
            selected[go] = vectors[record["canonical_id"]]
            selected_status[go] = record["identity_status"]
        elif ancestor_proxy and record["proxy_candidates"]:
            selected[go] = np.stack([vectors[p] for p in record["proxy_candidates"]]).mean(axis=0)
            selected_status[go] = "ancestor_mean_proxy"
            record["source_ids"] = record["proxy_candidates"]
        record["export_status"] = selected_status.get(go, "unresolved")
        record["present_in_2020_ontology"] = go in historical or go in historical_aliases
    covered = set(selected)
    eligible_groups, sequence_audit, excluded = {}, None, []
    if sequences_json is not None:
        eligible, excluded, sequence_audit = sequence_eligible(rows, sequences_json, max_seq_len)
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                part = [r for r in eligible if r["family"] == family and r["split"] == split]
                eligible_groups[f"{family}/{split}"] = group_stats(part, records, covered)
                for label in ("LOF", "GOF", "NEUTRAL"):
                    eligible_groups[f"{family}/{split}/{label}"] = group_stats([r for r in part if r["label"] == label], records, covered)
    groups = {"all_split_rows": group_stats(rows, records, covered)}
    for family in ("easy", "hard"):
        family_rows = [r for r in rows if r["family"] == family]
        groups[family] = group_stats(family_rows, records, covered)
        train_ids = {r["matched_go_id"] for r in family_rows if r["split"] == "train"}
        for split in ("train", "val", "test"):
            part = [r for r in family_rows if r["split"] == split]
            key = f"{family}/{split}"
            groups[key] = group_stats(part, records, covered)
            groups[key]["unseen_in_train_rows"] = sum(r["matched_go_id"] not in train_ids for r in part)
            groups[key]["unseen_in_train_covered_rows"] = sum(r["matched_go_id"] not in train_ids and r["matched_go_id"] in covered for r in part)
            for label in ("LOF", "GOF", "NEUTRAL"):
                groups[f"{key}/{label}"] = group_stats([r for r in part if r["label"] == label], records, covered)
        for source in sorted({r.get("match_source", "") for r in family_rows}):
            groups[f"{family}/source/{source}"] = group_stats([r for r in family_rows if r.get("match_source", "") == source], records, covered)
        for aspect in sorted({r["namespace"] for r in records.values()}):
            groups[f"{family}/aspect/{aspect}"] = group_stats([r for r in family_rows if records[r["matched_go_id"]]["namespace"] == aspect], records, covered)
    # Counts below combine the two alternative split families; they are not
    # independent observations. Report family-specific counts above as primary.
    counts = Counter(r["matched_go_id"] for r in rows)
    for go, record in records.items():
        record["rows_across_families"] = counts[go]
    identity_collisions = defaultdict(list)
    for go, record in records.items():
        if record["source_ids"]:
            identity_collisions[tuple(record["source_ids"])].append(go)
    collisions = [v for v in identity_collisions.values() if len(v) > 1]
    collision_conflicts = {}
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            inputs = defaultdict(list)
            for row in rows:
                if row["family"] != family or row["split"] != split:
                    continue
                record = records[row["matched_go_id"]]
                if record["source_ids"]:
                    inputs[(row["protein_id"], row["snp"], tuple(record["source_ids"]))].append(row)
            new_conflicts = []
            for group in inputs.values():
                per_term = defaultdict(set)
                for row in group:
                    per_term[row["matched_go_id"]].add(row["label"])
                if (len(per_term) > 1 and all(len(labels) == 1 for labels in per_term.values())
                        and len({r["label"] for r in group}) > 1):
                    new_conflicts.append(group)
            collision_conflicts[f"{family}/{split}"] = {
                "new_conflicting_input_groups": len(new_conflicts),
                "rows": sum(len(group) for group in new_conflicts),
            }
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()), "dataset_files": datasets,
        "source_files": sources, "anc2vec_commit": ANC_COMMIT, "go_commit": GO_COMMIT,
        "pretrained_go_version": historical_version, "audit_go_version": current_version,
        "ancestor_graph_version": graph_version,
        "pretrained_terms": len(vectors), "dimension": 200,
        "ancestor_proxy_enabled": ancestor_proxy, "missing_policy": missing,
        "sequence_filters_applied": False, "groups": groups,
        "sequence_audit": sequence_audit, "eligible_groups": eligible_groups,
        "shared_source_groups": collisions,
        "new_input_conflicts_from_shared_vectors": collision_conflicts,
        "notes": ["Raw TSV coverage, before sequence/position filters.",
                  "Easy and hard are alternative splits; combined row counts double-count shared examples.",
                  "Direct historical vectors of now-obsolete terms are retained and flagged.",
                  "Ancestor means are proxies, not learned Anc2vec vectors for those terms.",
                  "Consider links are not equivalence mappings and are never auto-resolved.",
                  "No variant labels, protein annotations or split outcomes construct GO vectors."],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(selected)
    # No zero/random replacement rows: unsupported terms are absent and the
    # runtime lookup raises unless the caller explicitly filters those rows.
    np.savez_compressed(out_dir / "go_embeddings.npz",
                        go_ids=np.asarray(ids, dtype="U10"),
                        embeddings=np.stack([selected[g] for g in ids]).astype(np.float32) if ids else np.empty((0, 200), np.float32),
                        status=np.asarray([selected_status[g] for g in ids], dtype="U64"),
                        metadata_json=np.asarray(json.dumps({k: v for k, v in report.items() if k not in ("groups", "eligible_groups", "shared_source_groups", "new_input_conflicts_from_shared_vectors")})))
    report["artifact_sha256"] = sha256(out_dir / "go_embeddings.npz")
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    fields = list(next(iter(records.values())).keys()) if records else ["go_id"]
    write_tsv(out_dir / "term_audit.tsv", list(records.values()), fields)
    write_tsv(out_dir / "unresolved_terms.tsv", [r for r in records.values() if r["export_status"] == "unresolved"], fields)
    write_tsv(out_dir / "coverage_by_group.tsv", [{"group": g, **v} for g, v in groups.items()], ["group", *groups["all_split_rows"].keys(), "unseen_in_train_rows", "unseen_in_train_covered_rows"])
    row_fields = ["family", "split", "row_index", "protein_id", "snp", "label", "matched_go_id", "matched_go_name", "match_source"]
    write_tsv(out_dir / "unresolved_rows.tsv", [r for r in rows if r["matched_go_id"] not in covered], row_fields)
    if sequence_audit:
        write_tsv(out_dir / "sequence_excluded_rows.tsv", excluded, [*row_fields, "reason"])
        write_tsv(out_dir / "coverage_after_sequence_filters.tsv", [{"group": g, **v} for g, v in eligible_groups.items()], ["group", *groups["all_split_rows"].keys()])
    write_markdown(out_dir / "coverage.md", report, records)
    summary = groups["all_split_rows"]
    print(f"[COVERAGE] {summary['covered_terms']}/{summary['terms']} terms; "
          f"{summary['covered_rows']}/{summary['rows']} split rows; "
          f"unresolved={summary['unresolved_terms']} terms", flush=True)
    print(f"[OUTPUT] {out_dir.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/processed/anc2vec/source")
    parser.add_argument("--out-dir", type=Path, help="Default: data/processed/anc2vec/exact (or ancestor_proxy)")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sequences-json", type=Path, help="Optional: also audit coverage after the phase-3 sequence filters")
    parser.add_argument("--max-seq-len", type=int, default=1022)
    parser.add_argument("--ancestor-proxy", action="store_true", help="Explicitly allow nearest-ancestor mean approximations")
    parser.add_argument("--missing", choices=("error", "report"), default="error",
                        help="error: exit 2 after audit if any GO remains unresolved; report: allow partial export")
    args = parser.parse_args()
    if args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    if args.out_dir is None:
        args.out_dir = ROOT / "data/processed/anc2vec" / ("ancestor_proxy" if args.ancestor_proxy else "exact")
    report = prepare(args.dataset_dir, args.source_dir, args.out_dir,
                     offline=args.offline, ancestor_proxy=args.ancestor_proxy, missing=args.missing,
                     sequences_json=args.sequences_json, max_seq_len=args.max_seq_len)
    if args.missing == "error" and report["groups"]["all_split_rows"]["unresolved_terms"]:
        parser.exit(2, "Unresolved GO terms remain; see unresolved_terms.tsv. Partial vectors were exported, but runtime lookup refuses missing IDs.\n")


if __name__ == "__main__":
    main()
