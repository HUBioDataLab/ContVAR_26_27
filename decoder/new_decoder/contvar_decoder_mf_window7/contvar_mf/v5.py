"""Read the corrected V5 bundle and prepare one shared MF cohort."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
from pathlib import Path
import re
from shutil import copyfileobj
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
import torch

from .go_embeddings import GOEmbeddingTable
from .prepare import annotate_augmentation, boolean_column, ontology_namespaces, row_statistics, split_overlaps
from .util import LABELS, sha256_file, utc_now, write_json

ENCODERS = ("mean", "distance")
SCALES = ("global", "mut", "window")
SPLITS = ("train", "val", "test")
FAMILIES = ("easy", "hard")
V5_FORMAT = "contvar_mf_window7_v5"
VARIANT = ["protein_id", "mutations"]
INPUT_KEY = VARIANT + ["matched_go_id"]
FEATURE_KEYS = [f"{side}_{scale}_key" for scale in SCALES for side in ("wt", "var")]
SOURCE_COLUMNS = [
    "sample_id", "family", "split", "protein_id", "snp", "mutations", "n_mutations", "variant_key",
    "matched_go_id", "label", "label_idx", "is_combinatorial", "augmented", "wt_status",
] + FEATURE_KEYS
VECTOR_FILES = {"global": "phase3_seq_phase1_global.h5", "mut": "phase3_seq_phase1_local.h5"}


class V5Bundle:
    def __init__(self, root):
        self.root = Path(root)
        self.members = {}
        for path in sorted(self.root.glob("Drive_V5_Local_Window_ve_Combinatorial-*.zip")):
            with ZipFile(path) as archive:
                for member in archive.namelist():
                    folder, _, relative = member.partition("/")
                    if folder != "Drive_V5_Local_Window_ve_Combinatorial" or member.endswith("/"):
                        continue
                    if relative in self.members:
                        raise ValueError(f"Duplicate V5 bundle member: {relative}")
                    self.members[relative] = (path, member)
        self.windows = {}
        for path in sorted(self.root.glob("phase3_seq_phase1_window*.h5")):
            with h5py.File(path, "r") as source:
                encoder = Path(source.attrs["backbone_path"]).parent.name.removeprefix("phase1_win7_")
                if encoder not in ENCODERS or encoder in self.windows:
                    raise ValueError(f"Unexpected or duplicate V5 window encoder: {path} ({encoder})")
                self.windows[encoder] = path
        if set(self.windows) != set(ENCODERS):
            raise ValueError("Both V5 encoder windows must be provided as extracted H5 files")

    @contextmanager
    def open(self, name):
        path, member = self.members[name]
        with ZipFile(path) as archive, archive.open(member) as stream:
            yield stream

    @contextmanager
    def feature(self, encoder, scale):
        if scale == "window":
            with h5py.File(self.windows[encoder], "r") as source:
                yield source
            return
        # The archive has deflated HDF5 members. Stage each in RAM once for random reads.
        with io.BytesIO() as buffer:
            with self.open(f"embeddings_v5_win7_{encoder}/{VECTOR_FILES[scale]}") as stream:
                copyfileobj(stream, buffer, 8 * 1024 * 1024)
            with h5py.File(buffer, "r") as source:
                yield source

    def hashes(self):
        paths = set(self.windows.values()) | {path for path, _ in self.members.values()}
        return {p.name: sha256_file(p) for p in sorted(paths)}


def load_namespaces(path):
    if path.suffix != ".obo":
        return ontology_namespaces(path)
    # GO Basic also ships as OBO. Use canonical and alternative IDs as in JSON.
    header, *terms = path.read_text(encoding="utf-8").split("\n[Term]\n")
    version = re.search(r"^data-version: (.+)$", header, flags=re.M)
    if version is None:
        raise ValueError(f"Missing GO Basic ontology version: {path}")
    namespaces = {}
    for term in terms:
        properties = {}
        for line in term.splitlines():
            if not line or line.startswith("["):
                break
            key, _, value = line.partition(": ")
            properties.setdefault(key, []).append(value)
        if "namespace" not in properties:
            continue
        for go_id in properties["id"] + properties.get("alt_id", []):
            namespaces[go_id] = {"namespace": properties["namespace"][0]}
    return namespaces, {"version": version[1], "sha256": sha256_file(path)}


def select_rows(frame, family, split, namespaces):
    if not frame.family.eq(family).all() or not frame.split.eq(split).all() or frame.sample_id.duplicated().any():
        raise ValueError(f"Unexpected source split or sample IDs: {family}/{split}")
    frame["augmented"] = annotate_augmentation(frame)
    frame["is_combinatorial"] = boolean_column(frame.is_combinatorial, "is_combinatorial")
    frame["namespace"] = frame.matched_go_id.map({term: info["namespace"] for term, info in namespaces.items()})
    if frame.namespace.isna().any():
        raise ValueError(f"Unresolved GO namespaces: {sorted(set(frame.loc[frame.namespace.isna(), 'matched_go_id']))}")
    mf = frame.namespace.eq("molecular_function")
    eligible = mf if split == "train" else mf & ~frame.augmented
    selected = frame.loc[eligible, SOURCE_COLUMNS + ["namespace"]].copy().reset_index(drop=True)
    genotype = selected.mutations.str.split(";")
    count = pd.to_numeric(selected.n_mutations, errors="raise")
    if (not genotype.str.len().eq(count).all() or not selected.is_combinatorial.eq(count.gt(1)).all()
            or not selected.mutations.str.fullmatch(r"[A-Z][1-9]\d*[A-Z](;[A-Z][1-9]\d*[A-Z])*").all()):
        raise ValueError(f"Invalid V5 genotype metadata: {family}/{split}")
    label_idx = pd.to_numeric(selected.label_idx, errors="raise")
    if not selected.label.map({label: i for i, label in enumerate(LABELS)}).eq(label_idx).all():
        raise ValueError(f"Unexpected source label indices: {family}/{split}")
    if selected[FEATURE_KEYS].eq("").any().any():
        raise ValueError(f"Missing corrected embedding keys: {family}/{split}")
    selected["source_row"] = np.flatnonzero(eligible)
    selected["label_idx"] = label_idx
    selected["common_test"] = split == "test"
    selected["common_key"] = selected.sample_id if split == "test" else ""
    return selected, dict(source_rows=len(frame), source_combinatorial=int(frame.is_combinatorial.sum()),
                          not_molecular_function=int((~mf).sum()), evaluation_augmentation=int((mf & ~eligible).sum()))


def prepare_bundle(bundle, output, ontology):
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared manifest already exists: {output}")
    namespaces, ontology_meta = load_namespaces(ontology)
    output.mkdir(parents=True, exist_ok=True)
    go_path = output / "go_terms_embeddings_anc2vec.npz"
    with bundle.open(go_path.name) as source, go_path.open("wb") as destination:
        copyfileobj(source, destination)
    go = GOEmbeddingTable(go_path)
    tables, common, overlaps = {}, {}, {}
    table_root = output / "v5"
    table_root.mkdir(exist_ok=True)
    for family in FAMILIES:
        frames, reports = {}, {}
        for split in SPLITS:
            name = f"{family}_{split}.tsv"
            member = f"prepared_v5_combinatorials/{name}"
            with bundle.open(member) as source:
                raw = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
            frame, report = select_rows(raw, family, split, namespaces)
            missing = set(frame.matched_go_id) - go.go_to_row.keys()
            if missing:
                raise ValueError(f"Missing Anc2vec GO IDs: {sorted(missing)}")
            frames[split], reports[split] = frame, {"source_file": member, **report}
        overlaps[family] = split_overlaps(frames, family, VARIANT)
        train_go = set(frames["train"].matched_go_id)
        labels_per_input = pd.concat(frames.values()).groupby(INPUT_KEY, sort=False).label.nunique()
        conflicts = set(labels_per_input[labels_per_input > 1].index)
        for split, frame in frames.items():
            frame["seen_in_train"] = frame.matched_go_id.isin(train_go)
            frame["multiple_labels_for_input"] = [key in conflicts for key in frame[INPUT_KEY].itertuples(index=False, name=None)]
            path = table_root / f"{family}_{split}.tsv"
            frame.to_csv(path, sep="\t", index=False)
            stats = row_statistics(frame, INPUT_KEY)
            stats["combinatorial"] = int(frame.is_combinatorial.sum())
            tables[f"v5/{family}/{split}"] = {**reports[split], "final": stats, "sha256": sha256_file(path)}
            print(f"V5 {family}/{split}: {len(frame)} MF rows, {stats['combinatorial']} combinatorial", flush=True)
        common[family] = tables[f"v5/{family}/test"]["final"]
    manifest = dict(format=V5_FORMAT, created=utc_now(), source_data=str(bundle.root.resolve()),
                    source_cohorts={encoder: "prepared_v5_combinatorials" for encoder in ENCODERS},
                    source_feature_sha256=bundle.hashes(), go_sha256=go.artifact_sha256,
                    ontology=ontology_meta, window_files={k: v.name for k, v in bundle.windows.items()},
                    policy=dict(namespace="molecular_function", combinatorial=True, window_size=7,
                                train_augmentation="retain existing only", evaluation_augmentation=False,
                                conflicts="retain", duplicates="retain multiplicity",
                                feature_lookup="explicit V5 keys; encoder-provided genotype summaries"),
                    tables=tables, split_overlaps=overlaps, common_test=common,
                    preparation_code_sha256=sha256_file(__file__))
    write_json(output / "manifest.json", manifest)
    return manifest


class V5FeatureStore:
    """Index each corrected H5 feature once per encoder for the existing PairDataset."""

    allow_combinatorial = True

    def __init__(self, prepared, bundle, encoder):
        self.go = GOEmbeddingTable(prepared / "go_terms_embeddings_anc2vec.npz")
        frames = [pd.read_csv(prepared / "v5" / f"{family}_{split}.tsv", sep="\t", usecols=FEATURE_KEYS)
                  for family in FAMILIES for split in SPLITS]
        self.maps = {}
        expected_identity = None
        for scale in SCALES:
            keys = sorted({key for frame in frames for side in ("wt", "var") for key in frame[f"{side}_{scale}_key"]})
            self.maps[scale] = {key: i for i, key in enumerate(keys)}
            with bundle.feature(encoder, scale) as source:
                identity = tuple(source.attrs[name] for name in
                                 ("backbone_path", "patch_checkpoint_sha256", "repair_plan_sha256"))
                if (Path(identity[0]).parent.name != f"phase1_win7_{encoder}"
                        or (expected_identity is not None and expected_identity != identity)):
                    raise ValueError(f"Mismatched V5 checkpoint or crop plan: {encoder}/{scale}")
                expected_identity = identity
                if scale == "window":
                    self._load_windows(source, keys)
                else:
                    vectors = np.empty((len(keys), 1280), dtype=np.float32)
                    for i, key in enumerate(keys):
                        value = source[key][:]
                        if value.shape != (1280,) or not np.isfinite(value).all():
                            raise ValueError(f"Invalid {encoder}/{scale} vector: {key}")
                        vectors[i] = value
                    setattr(self, f"{scale}_vectors", torch.from_numpy(vectors))
            print(f"V5 FEATURES {encoder}/{scale}: {len(keys)} keys", flush=True)

    def _load_windows(self, source, keys):
        if source.attrs["window_len"] != 7 or source.attrs["center_index"] != 3:
            raise ValueError("Expected the corrected seven-token windows with center index 3")
        tokens = np.empty((len(keys), 7, 1280), dtype=np.float16)
        masks = np.empty((len(keys), 7), dtype=bool)
        means = np.empty((len(keys), 1280), dtype=np.float32)
        mask_group = source["mask"]
        for i, key in enumerate(keys):
            window, mask = source[key][:], mask_group[key][:]
            if (window.shape != (7, 1280) or mask.shape != (7,) or not np.isin(mask, [0, 1]).all()
                    or not mask[3]):
                raise ValueError(f"Invalid V5 window or mask: {key}")
            mask = mask.astype(bool)
            valid = window[mask]
            if not np.isfinite(valid).all():
                raise ValueError(f"Nonfinite V5 window: {key}")
            window[~mask] = 0
            tokens[i], masks[i], means[i] = window, mask, valid.mean(0, dtype=np.float32)
        self.window_tokens = torch.from_numpy(tokens)
        self.window_masks = torch.from_numpy(masks)
        self.window_means = torch.from_numpy(means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_bundle(V5Bundle(args.bundle), args.out, args.ontology)


if __name__ == "__main__":
    main()
