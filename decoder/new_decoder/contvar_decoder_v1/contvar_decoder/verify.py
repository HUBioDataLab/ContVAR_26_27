"""Verify the delivery manifest and the native prepared split target mappings."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import WORKSPACE
from .prepare import sha256_file


def verify(root=WORKSPACE, *, full=False):
    root=Path(root)
    path=root/"metadata/file_manifest.json"
    report=json.loads(path.read_text())
    checked=0
    for entry in report["files"]:
        p=root/entry["path"]
        if not p.is_file() or p.stat().st_size!=entry["bytes"]:
            raise ValueError(f"Missing/changed delivery file: {entry['path']}")
        if full or entry["bytes"]<100*1024*1024:
            if sha256_file(p)!=entry["sha256"]:
                raise ValueError(f"Hash mismatch: {entry['path']}")
        checked+=1
    for family in ("easy","hard"):
        for split in ("train","val","test"):
            frames={h:pd.read_csv(root/"prepared_splits"/h/f"{family}_{split}.tsv",sep="\t")
                    for h in ("parallel","multiclass")}
            p,m=frames["parallel"],frames["multiclass"]
            if not p.sample_id.equals(m.sample_id) or not p.label.equals(m.label):
                raise ValueError(f"Head cohorts differ: {family}/{split}")
            if p.is_combinatorial.any() or (split!="train" and p.augmented.any()):
                raise ValueError(f"Forbidden evidence: {family}/{split}")
            if not p.snp.str.fullmatch(r"[A-Z][1-9]\d*[A-Z]").all():
                raise ValueError(f"Non-single SNP: {family}/{split}")
            if not p.target_lof.eq(p.label.eq("LOF").astype(int)).all() or not p.target_gof.eq(p.label.eq("GOF").astype(int)).all():
                raise ValueError(f"Invalid parallel targets: {family}/{split}")
            if not m.target.equals(m.label):
                raise ValueError(f"Invalid multiclass targets: {family}/{split}")
    print(f"Verified {checked} delivery files and all 12 native prepared tables; full_hashes={full}")
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=WORKSPACE)
    p.add_argument("--full",action="store_true",help="Also rehash the large H5 files")
    a=p.parse_args()
    verify(a.root,full=a.full)


if __name__=="__main__":
    main()
