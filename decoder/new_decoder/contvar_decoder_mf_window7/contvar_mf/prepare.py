"""Prepare the two supplied cohorts, MF-only, with one shared window-7 cache."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pandas as pd

from .go_embeddings import GOEmbeddingTable
from .util import DATA_ROOT, LABELS, WORKSPACE, sha256_file, utc_now, write_json

COHORTS = {"old": "prepared_old_subset_without_1499_proteins", "full": "prepared_full_all_proteins"}
KEY = ["protein_id", "snp", "matched_go_id"]
SNP = re.compile(r"([A-Z])([1-9]\d*)([A-Z])")


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def boolean_column(values, name):
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError(f"{name} must explicitly contain True or False")
    return normalized.eq("true")


def ontology_namespaces(path):
    graph = json.loads(Path(path).read_text(encoding="utf-8"))["graphs"][0]
    table = {}
    for node in graph["nodes"]:
        if not re.search(r"/GO_\d{7}$", node["id"]):
            continue
        go = node["id"].rsplit("/", 1)[1].replace("_", ":")
        meta = node.get("meta", {})
        properties = meta.get("basicPropertyValues", [])
        namespace = next((p["val"] for p in properties if p["pred"].endswith("#hasOBONamespace")), None)
        record = dict(go_id=go, canonical_go_id=go, namespace=namespace,
                      ontology_name=node.get("lbl", ""), obsolete=bool(meta.get("deprecated", False)))
        table[go] = record
        for p in properties:
            if p["pred"].endswith("#hasAlternativeId") and re.fullmatch(r"GO:\d{7}", p["val"]):
                table[p["val"]] = {**record, "go_id": p["val"]}
    return table, dict(graph_id=graph.get("id"), version=graph.get("meta", {}).get("version"),
                       sha256=sha256_file(path))


def annotate_augmentation(frame):
    inferred = frame.match_source.eq("broad_neutral_broadcast")
    if "augmented" in frame and not boolean_column(frame.augmented, "augmented").equals(inferred):
        raise ValueError("Source augmented flags disagree with broad_neutral_broadcast rule")
    return inferred


def source_input_reason(row, sequences):
    match = SNP.fullmatch(row.snp)
    if not match:
        return "invalid_single_substitution_format"
    wt, pos, _ = match.groups()
    seq = sequences.get(row.protein_id)
    if seq is None:
        return "missing_sequence"
    pos = int(pos)
    if pos > len(seq):
        return "position_outside_sequence"
    if seq[pos - 1] != wt:
        return "wt_residue_mismatch"
    return ""


def _cache_keys(source, output, keys, scale, invalid):
    group = output.create_group(scale)
    group.create_dataset("keys", data=np.asarray(keys, dtype=object), dtype=h5py.string_dtype("utf-8"))
    vectors = group.create_dataset("vectors", shape=(len(keys), 1280), dtype="f4")
    for i, key in enumerate(keys):
        try:
            value = np.asarray(source[key][:], dtype=np.float32)
            if value.shape != (1280,) or not np.isfinite(value).all():
                raise ValueError("invalid shape or nonfinite value")
            vectors[i] = value
        except (KeyError, ValueError, OSError) as exc:
            invalid[f"{scale}:{key}"] = str(exc)
            vectors[i] = 0
        if (i + 1) % 5000 == 0 or i + 1 == len(keys):
            print(f"CACHE {scale} {i+1}/{len(keys)}", flush=True)


def build_cache(data, frames, destination, hashes):
    requested = {"global": set(), "mut": set()}
    for frame in frames.values():
        for scale in requested:
            for side in ("wt", "var"):
                requested[scale].update(frame[f"{side}_{scale}_key"])
    keys = {k: sorted(v) for k, v in requested.items()}
    fingerprint = {**hashes, "crop": [13, 20], "keys_sha256": __import__("hashlib").sha256(
        json.dumps(keys, sort_keys=True).encode()).hexdigest()}
    audit_path = destination.with_suffix(".audit.json")
    if destination.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["fingerprint"] != fingerprint:
            raise ValueError("Existing feature cache does not match this preparation")
        return audit
    partial = destination.with_name(destination.name + ".partial")
    invalid = {}
    window = next((data / "embeddings").glob("phase3_seq_phase1_window*.h5"))
    padded = 0
    with h5py.File(data / "embeddings/phase3_seq_phase1_global.h5", "r") as global_file, \
            h5py.File(data / "embeddings/phase3_seq_phase1_local.h5", "r") as mut_file, \
            h5py.File(window, "r") as raw, h5py.File(partial, "w") as out:
        if int(raw.attrs["center_index"]) != 16 or int(raw.attrs["window_len"]) != 33:
            raise ValueError("Unexpected raw window layout")
        _cache_keys(global_file, out, keys["global"], "global", invalid)
        _cache_keys(mut_file, out, keys["mut"], "mut", invalid)
        group = out.create_group("window")
        group.create_dataset("keys", data=np.asarray(keys["mut"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        tokens = group.create_dataset("tokens", shape=(len(keys["mut"]), 7, 1280), dtype="f2")
        masks = group.create_dataset("mask", shape=(len(keys["mut"]), 7), dtype="bool")
        means = group.create_dataset("mean", shape=(len(keys["mut"]), 1280), dtype="f4")
        for i, key in enumerate(keys["mut"]):
            try:
                ds, ms = raw[key], raw["mask"][key]
                if ds.shape != (33, 1280) or ms.shape != (33,):
                    raise ValueError("invalid raw window/mask shape")
                w = np.asarray(ds[13:20], dtype=np.float16)
                m = np.asarray(ms[13:20])
                if not np.isin(m, [0, 1]).all() or not bool(m[3]):
                    raise ValueError("invalid mask or absent center")
                m = m.astype(bool)
                if not np.isfinite(w[m]).all():
                    raise ValueError("nonfinite valid window token")
                if not np.array_equal(w[3], np.asarray(mut_file[key][:], dtype=np.float16)):
                    raise ValueError("window center does not match mutation-site embedding")
                w[~m] = 0
                tokens[i], masks[i], means[i] = w, m, w[m].astype(np.float32).mean(0, dtype=np.float32)
                padded += int(not m.all())
            except (KeyError, ValueError, OSError) as exc:
                invalid[f"window:{key}"] = str(exc)
                tokens[i], masks[i], means[i] = 0, False, 0
            if (i + 1) % 3000 == 0 or i + 1 == len(keys["mut"]):
                print(f"CACHE window7 {i+1}/{len(keys['mut'])}", flush=True)
        out.attrs["complete"] = True
        out.attrs["window_size"] = 7
        out.attrs["crop_start"], out.attrs["crop_stop"] = 13, 20
        out.attrs["fingerprint"] = json.dumps(fingerprint, sort_keys=True)
    partial.replace(destination)
    audit = dict(fingerprint=fingerprint, invalid_inputs=invalid,
                 global_keys=len(keys["global"]), mutation_window_keys=len(keys["mut"]), padded_windows=padded,
                 window_source=str(window.resolve()), window_dtype="source float16 retained; mean float32")
    write_json(audit_path, audit)
    return audit


def row_statistics(frame):
    counts = frame.groupby(KEY, sort=False).label.nunique()
    conflicts = set(counts[counts > 1].index)
    conflict_mask = pd.Series([tuple(r) in conflicts for r in frame[KEY].itertuples(index=False, name=None)])
    return dict(rows=len(frame), proteins=int(frame.protein_id.nunique()),
                variants=int(frame[KEY[:2]].drop_duplicates().shape[0]), go_terms=int(frame.matched_go_id.nunique()),
                labels={label: int(frame.label.eq(label).sum()) for label in LABELS},
                augmented=int(frame.augmented.astype(str).str.lower().eq("true").sum()),
                conflicting_keys=len(conflicts), conflicting_rows=int(conflict_mask.sum()),
                same_label_duplicate_extra=int(frame.duplicated(KEY + ["label"]).sum()))


def split_overlaps(frames, family):
    result = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        proteins = len(set(frames[a].protein_id) & set(frames[b].protein_id))
        variants = len(set(frames[a][KEY[:2]].itertuples(index=False, name=None)) &
                       set(frames[b][KEY[:2]].itertuples(index=False, name=None)))
        if variants or (family == "hard" and proteins):
            raise ValueError(f"Unexpected within-cohort split overlap: {family}/{a}/{b}")
        result[f"{a}/{b}"] = dict(protein_overlap=proteins, variant_overlap=variants)
    return result


def prepare(data=DATA_ROOT, output=WORKSPACE / "prepared", ontology=WORKSPACE / "ontology/go-basic.json"):
    data, output, ontology = Path(data), Path(output), Path(ontology)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared manifest already exists; use the completed preparation")
    output.mkdir(exist_ok=True, parents=True)
    ns, ontology_meta = ontology_namespaces(ontology)
    write_json(output / "ontology_provenance.json", {**ontology_meta, **json.loads(ontology.with_suffix(".source.json").read_text())})
    go = GOEmbeddingTable(data / "embeddings/go_terms_embeddings_anc2vec.npz")
    sequences_path = data / "sequences/uniprot_dict.json"
    raw_sequences = json.loads(sequences_path.read_text(encoding="utf-8"))
    sequences = raw_sequences if isinstance(raw_sequences, dict) else {r["Entry"]: r["Sequence"] for r in raw_sequences}
    # Validate the augmentation interpretation against the native source flags.
    for path in sorted((data / "splits/multiclass").glob("*.tsv")):
        annotate_augmentation(read_table(path))
    hashes = {"global": sha256_file(data / "embeddings/phase3_seq_phase1_global.h5"),
              "mut": sha256_file(data / "embeddings/phase3_seq_phase1_local.h5"),
              "window": sha256_file(next((data / "embeddings").glob("phase3_seq_phase1_window*.h5"))),
              "go": go.artifact_sha256}
    frames, reports, excluded, source_hashes, terms_used = {}, {}, {}, {}, set()
    for cohort, folder in COHORTS.items():
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                name = f"{cohort}/{family}/{split}"
                path = data / folder / f"{family}_{split}.tsv"
                frame = read_table(path)
                frame["source_row"] = np.arange(len(frame))
                frame["source_sample_id"] = frame.get("sample_id", frame.source_row.astype(str))
                frame["sample_id"] = [f"{cohort}:{family}:{split}:{i}" for i in frame.source_row]
                frame["cohort"], frame["family"], frame["split"] = cohort, family, split
                terms_used.update(frame.matched_go_id)
                unknown = set(frame.matched_go_id) - ns.keys()
                if unknown:
                    raise ValueError(f"Unresolved ontology namespaces: {sorted(unknown)}")
                frame["namespace"] = frame.matched_go_id.map(lambda x: ns[x]["namespace"])
                frame["is_combinatorial"] = boolean_column(frame.is_combinatorial, "is_combinatorial")
                frame["augmented"] = annotate_augmentation(frame)
                reasons = pd.Series("", index=frame.index)
                reasons.loc[frame.is_combinatorial] = "is_combinatorial_true"
                reasons.loc[reasons.eq("") & frame.namespace.ne("molecular_function")] = "not_molecular_function"
                if split != "train":
                    reasons.loc[reasons.eq("") & frame.augmented] = "evaluation_augmentation"
                kept = frame.loc[reasons.eq("")].copy()
                kept["input_issue"] = [source_input_reason(r, sequences) for r in kept.itertuples(index=False)]
                for index, reason in kept.loc[kept.input_issue.ne(""), "input_issue"].items():
                    reasons.loc[index] = reason
                selected = kept.loc[kept.input_issue.eq("")].drop(columns="input_issue").copy()
                if not selected.label.isin(LABELS).all():
                    raise ValueError("Unexpected label")
                selected["label_idx"] = selected.label.map({label: i for i, label in enumerate(LABELS)})
                selected["target"] = selected.label
                selected["target_lof"], selected["target_gof"] = selected.label.eq("LOF").astype(int), selected.label.eq("GOF").astype(int)
                selected["pos"] = selected.snp.str.extract(r"[A-Z](\d+)[A-Z]", expand=False).astype(int)
                for side in ("wt", "var"):
                    selected[f"{side}_window_key"] = selected[f"{side}_mut_key"]
                selected["common_test"] = False
                selected["common_key"] = ""
                frame["exclusion_reason"] = reasons
                excluded[name] = frame.loc[reasons.ne(""), ["sample_id", "source_sample_id"] + KEY + ["label", "exclusion_reason"]]
                frames[name] = selected.reset_index(drop=True)
                reports[name] = dict(source_file=str(path.resolve()), source_rows=len(frame),
                    selected_before_embedding_check=len(selected), exclusions=dict(Counter(reasons[reasons.ne("")])) )
                source_hashes[str(path.relative_to(data))] = sha256_file(path)
                print(f"SELECT {name}: source={len(frame)} selected_MF={len(selected)}", flush=True)
    pd.DataFrame([ns[x] for x in sorted(terms_used)]).to_csv(output / "go_namespace_lookup.tsv", sep="\t", index=False)
    cache = build_cache(data, frames, output / "features_window7.h5", hashes)
    invalid = cache["invalid_inputs"]
    for name, frame in frames.items():
        reasons = []
        for row in frame.itertuples(index=False):
            bad = [f"{scale}:{getattr(row, f'{side}_{scale}_key')}" for scale in ("global", "mut", "window")
                   for side in ("wt", "var") if f"{scale}:{getattr(row, f'{side}_{scale}_key')}" in invalid]
            if row.matched_go_id not in go.go_to_row:
                bad.append("missing_go_embedding")
            reasons.append(";".join(bad))
        mask = np.array([bool(r) for r in reasons])
        if mask.any():
            bad = frame.loc[mask, ["sample_id", "source_sample_id"] + KEY + ["label"]].copy()
            bad["exclusion_reason"] = np.asarray(reasons)[mask]
            excluded[name] = pd.concat([excluded[name], bad], ignore_index=True)
            frame = frame.loc[~mask].reset_index(drop=True)
        technical = excluded[name].loc[~excluded[name].exclusion_reason.isin(
            ["is_combinatorial_true", "not_molecular_function", "evaluation_augmentation"])]
        if len(technical) > 20:
            technical.to_csv(output / (name.replace("/", "_") + "_technical_issues.tsv"), sep="\t", index=False)
            raise ValueError(f"Systematic input issue in {name}: {len(technical)} rows; inspect before training")
        frames[name] = frame
        reports[name]["technical_excluded_rows"] = len(technical)
        reports[name]["technical_exclusions"] = dict(Counter(technical.exclusion_reason))
        excluded[name].to_csv(output / (name.replace("/", "_") + "_excluded.tsv"), sep="\t", index=False)
    for cohort in COHORTS:
        for family in ("easy", "hard"):
            groups = {s: frames[f"{cohort}/{family}/{s}"] for s in ("train", "val", "test")}
            combined = pd.concat(groups.values(), ignore_index=True)
            counts = combined.groupby(KEY, sort=False).label.nunique()
            conflicts = set(counts[counts > 1].index)
            train_go = set(groups["train"].matched_go_id)
            for frame in groups.values():
                frame["seen_in_train"] = frame.matched_go_id.isin(train_go)
                frame["multiple_labels_for_input"] = [tuple(k) in conflicts for k in frame[KEY].itertuples(index=False, name=None)]
            reports[f"{cohort}/{family}/overlap"] = split_overlaps(groups, family)
    common_report = {}
    for family in ("easy", "hard"):
        tests = {c: frames[f"{c}/{family}/test"] for c in COHORTS}
        for frame in tests.values():
            occurrence = frame.groupby(KEY + ["label"], sort=False).cumcount()
            frame["common_key"] = ["|".join(map(str, (*values, n))) for values, n in zip(
                frame[KEY + ["label"]].itertuples(index=False, name=None), occurrence)]
        intersection = set(tests["old"].common_key) & set(tests["full"].common_key)
        if not intersection:
            raise ValueError(f"No common MF test records for {family}")
        common = tests["old"].loc[tests["old"].common_key.isin(intersection)].copy()
        common.to_csv(output / f"common_{family}_test.tsv", sep="\t", index=False)
        for cohort in COHORTS:
            tests[cohort]["common_test"] = tests[cohort].common_key.isin(intersection)
            trainval = pd.concat([frames[f"{cohort}/{family}/train"], frames[f"{cohort}/{family}/val"]])
            columns = KEY[:2] if family == "easy" else KEY[:1]
            if set(common[columns].itertuples(index=False, name=None)) & set(trainval[columns].itertuples(index=False, name=None)):
                raise ValueError("Common test overlaps training/validation")
        common_report[family] = row_statistics(common)
    for name, frame in frames.items():
        cohort, family, split = name.split("/")
        folder = output / cohort
        folder.mkdir(exist_ok=True)
        target = folder / f"{family}_{split}.tsv"
        frame.to_csv(target, sep="\t", index=False)
        reports[name]["final"] = row_statistics(frame)
        reports[name]["sha256"] = sha256_file(target)
    manifest = dict(format="contvar_mf_window7_prepared_v1", created=utc_now(), source_data=str(data.resolve()),
        source_cohorts=COHORTS, source_table_sha256=source_hashes, source_feature_sha256=hashes,
        sequences_sha256=sha256_file(sequences_path), ontology=ontology_meta,
        feature_cache="features_window7.h5", feature_cache_sha256=sha256_file(output / "features_window7.h5"),
        policy=dict(namespace="molecular_function", combinatorial=False, train_augmentation="retain existing only",
                    evaluation_augmentation=False, augmentation_detection="source-validated broad_neutral_broadcast",
                    conflicts="retain", duplicates="retain multiplicity", source_partitions="preserve per cohort",
                    window_size=7, crop=[13, 20], additional_biological_filters=False,
                    technical_exclusions="only genuinely unusable inputs, logged and shared by every model condition"),
        tables=reports, common_test=common_report, cache_audit=cache,
        preparation_code_sha256=sha256_file(__file__))
    write_json(output / "manifest.json", manifest)
    print("PREPARATION_COMPLETE " + json.dumps({"common_test": common_report,
          "tables": {k: v["final"]["rows"] for k, v in reports.items() if "final" in v}}), flush=True)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_ROOT)
    p.add_argument("--out", type=Path, default=WORKSPACE / "prepared")
    p.add_argument("--ontology", type=Path, default=WORKSPACE / "ontology/go-basic.json")
    a = p.parse_args()
    prepare(a.data, a.out, a.ontology)
