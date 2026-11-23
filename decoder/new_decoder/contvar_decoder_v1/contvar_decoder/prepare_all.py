"""Validate original head-specific splits and prepare both input layouts."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

from .data import PACKAGE, WORKSPACE
from .prepare import DEFAULT_OUT, KEY, prepare, sha256_file


def read(archive, member):
    return pd.read_csv(io.BytesIO(archive.read(member)), sep="\t", keep_default_na=False, dtype=str)


def prepare_all(source=None, package=PACKAGE, out=DEFAULT_OUT):
    source = Path(source or WORKSPACE / "source_splits")
    paths = {kind: sorted(source.glob(f"{kind}-*.zip"))[-1]
             for kind in ("parallel", "multiclass", "cascade")}
    checks = {}
    with zipfile.ZipFile(paths["multiclass"]) as mc, zipfile.ZipFile(paths["parallel"]) as pa, zipfile.ZipFile(paths["cascade"]) as ca:
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                m = read(mc, f"multiclass/{family}_{split}.tsv")
                p = read(pa, f"parallel/{family}_{split}.tsv")
                s1 = read(ca, f"cascade/stage1_{family}_{split}.tsv")
                s2 = read(ca, f"cascade/stage2_{family}_{split}.tsv")
                common = [c for c in m.columns if c != "target"]
                row = dict(parallel_same_source_rows=m[common].equals(p[common]),
                           multiclass_targets=m.target.equals(m.label),
                           parallel_lof_targets=p.target_lof.equals(m.label.eq("LOF").astype(int).astype(str)),
                           parallel_gof_targets=p.target_gof.equals(m.label.eq("GOF").astype(int).astype(str)),
                           cascade_stage1_same_rows=m[common].equals(s1[common]),
                           cascade_stage1_targets=s1.target.equals(m.label.map({"LOF":"FUNCTIONAL", "GOF":"FUNCTIONAL", "NEUTRAL":"NEUTRAL"})),
                           cascade_stage2_exact_effect_subset=m.loc[m.label.ne("NEUTRAL"),common].reset_index(drop=True).equals(s2[common]))
                if not all(row.values()):
                    raise ValueError(f"Source layout inconsistency: {family}/{split}: {row}")
                checks[f"{family}/{split}"] = row
    reports = {}
    for layout in ("parallel", "multiclass"):
        reports[layout] = prepare(paths[layout], package, Path(out) / layout, layout=layout)
    tables = {}
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            paths_by_head = {h: Path(out) / h / f"{family}_{split}.tsv" for h in reports}
            frames = {h: pd.read_csv(p, sep="\t") for h,p in paths_by_head.items()}
            m, p = frames["multiclass"], frames["parallel"]
            columns = ["sample_id", *KEY, "label", "wt_global_key", "var_global_key", "wt_mut_key", "var_mut_key"]
            if not m[columns].equals(p[columns]):
                raise ValueError(f"Prepared models have different cohorts: {family}/{split}")
            if m.is_combinatorial.any() or (split != "train" and m.augmented.any()):
                raise ValueError(f"Forbidden prepared rows: {family}/{split}")
            tables[f"{family}/{split}"] = dict(rows=len(m), labels=m.label.value_counts().to_dict(),
                                                augmented=int(m.augmented.sum()),
                                                input_label_review_rows=int(m.multiple_labels_for_input.sum()),
                                                paths={h:str(p.relative_to(WORKSPACE)) if p.is_relative_to(WORKSPACE) else str(p)
                                                       for h,p in paths_by_head.items()},
                                                sha256={h:sha256_file(p) for h,p in paths_by_head.items()})
        overlap = reports["multiclass"]["families"][family]["ready_overlap"]
        for pair, counts in overlap.items():
            if counts["protein_id+snp"] or (family == "hard" and counts["protein_id"]):
                raise ValueError(f"Unexpected split overlap: {family}/{pair}")
    manifest = dict(source_sha256={k:sha256_file(v) for k,v in paths.items()},
                    source_layout_checks=checks, prepared=tables,
                    policy=dict(single_mutation_only=True, augmented_eval=False,
                                multiple_labels="retain source labels and report groups",
                                source_partition_assignments="preserved", missing_inputs="explicitly excluded with reasons",
                                parallel_source="parallel ZIP", multiclass_source="multiclass ZIP",
                                cascade="validated reference layout; not used for the four requested models"))
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("Verified both prepared layouts, native targets, no variant leakage, and no augmented evaluation rows.", flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=WORKSPACE/"source_splits")
    p.add_argument("--package",type=Path,default=PACKAGE)
    p.add_argument("--out",type=Path,default=DEFAULT_OUT)
    a=p.parse_args()
    prepare_all(a.source,a.package,a.out)


if __name__=="__main__":
    main()
