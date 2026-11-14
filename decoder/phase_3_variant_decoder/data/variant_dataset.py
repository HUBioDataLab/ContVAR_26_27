"""
Phase 3 variant dataset: TSV parsing, filters, GO vocabulary.

Design: phase_3_variant_decoder/DESIGN.md §2

Primary input: splits_cascade/stage{1,2}_{easy,hard}_{train,val,test}.tsv
Legacy input : {easy,hard}_{train,val,test}.tsv
Output: filtered DataFrame + go_vocab + a FilterReport listing how many rows each
        step dropped.

CLI:
    python -m phase_3_variant_decoder.data.variant_dataset --family hard
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from phase_3_variant_decoder.config import FAMILIES, LABELS, MAX_SEQ_LEN, VariantDecoderConfig
except ModuleNotFoundError:  # when run directly as a file
    from config import FAMILIES, LABELS, MAX_SEQ_LEN, VariantDecoderConfig


SPLITS = ("train", "val", "test")

# SNP notation in the dataset is 100% clean X123Y (measured, DESIGN §2.2) — the
# regex is still enforced so a malformed row cannot slip through silently.
SNP_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
CASCADE_BASE_COLUMNS = (
    "protein_id", "feature_type", "ftId", "snp", "atomic_description", "label",
    "matched_go_id", "matched_go_name", "cosine_similarity", "match_source",
    "notes", "is_combinatorial",
)


def parse_snp(snp: str) -> tuple[str, int, str] | None:
    """'Y305F' -> ('Y', 305, 'F'). None if the notation does not match."""
    m = SNP_RE.match(str(snp))
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def load_sequences(path: str) -> dict[str, str]:
    """uniprot_dict.json -> {protein_id: sequence}.

    The file is NOT a dict; it is a list of [{"Entry": ..., "Sequence": ...}, ...].
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):          # tolerate a future format change
        return {k: str(v) for k, v in raw.items()}
    return {e["Entry"]: e["Sequence"] for e in raw if e.get("Sequence")}


@dataclass
class FilterReport:
    """Rows dropped by each filter. DESIGN §8.1 verifies this table."""

    family: str
    raw_rows: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)
    kept_rows: int = 0
    n_proteins: int = 0
    n_go_terms: int = 0
    per_split: dict = field(default_factory=dict)
    per_label: dict = field(default_factory=dict)
    source_layout: str = "legacy"

    def add(self, name: str, dropped: int):
        self.steps.append((name, dropped))

    def render(self) -> str:
        lines = [
            f"[FILTER] family={self.family}",
            f"  source layout                              {self.source_layout:>7s}",
            f"  raw rows                                   {self.raw_rows:>7,}",
        ]
        for name, dropped in self.steps:
            lines.append(f"  - {name:<40s} {-dropped:>7,}")
        pct = 100.0 * self.kept_rows / self.raw_rows if self.raw_rows else 0.0
        lines += [
            f"  {'=' * 50}",
            f"  KEPT                                       {self.kept_rows:>7,}  ({pct:.1f}%)",
            f"  proteins={self.n_proteins:,}  GO terms={self.n_go_terms:,}",
            f"  split: {self.per_split}",
            f"  label: {self.per_label}",
        ]
        return "\n".join(lines)


def _read_and_validate_cascade_split(dataset_dir: Path, family: str, split: str) -> pd.DataFrame:
    """Read explicit stage files and prove Stage 2 is Stage 1's effect subset."""
    stage1_path = dataset_dir / f"stage1_{family}_{split}.tsv"
    stage2_path = dataset_dir / f"stage2_{family}_{split}.tsv"
    stage1 = pd.read_csv(stage1_path, sep="\t")
    stage2 = pd.read_csv(stage2_path, sep="\t")

    required = set(CASCADE_BASE_COLUMNS) | {"target"}
    for stage, path, frame in ((1, stage1_path, stage1), (2, stage2_path, stage2)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}")
        if frame["target"].isna().any():
            raise ValueError(f"{path} contains empty Stage {stage} targets")

    expected_stage1 = stage1["label"].map(
        {"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"}
    )
    if expected_stage1.isna().any() or not stage1["target"].equals(expected_stage1):
        raise ValueError(
            f"{stage1_path} has invalid targets; expected LOF/GOF -> FUNCTIONAL "
            "and NEUTRAL -> NEUTRAL"
        )
    if not stage2["label"].isin(("LOF", "GOF")).all():
        raise ValueError(f"{stage2_path} must contain only LOF/GOF rows")
    if not stage2["target"].equals(stage2["label"]):
        raise ValueError(f"{stage2_path} target must equal its LOF/GOF label")

    expected_stage2 = stage1.loc[
        stage1["label"].isin(("LOF", "GOF")), list(CASCADE_BASE_COLUMNS)
    ].reset_index(drop=True)
    actual_stage2 = stage2.loc[:, list(CASCADE_BASE_COLUMNS)].reset_index(drop=True)
    if not actual_stage2.equals(expected_stage2):
        raise ValueError(
            f"{stage2_path} is not the exact ordered LOF/GOF subset of {stage1_path}"
        )
    return stage1.drop(columns="target").assign(split=split)


def _load_source_frames(family: str, dataset_dir: str) -> tuple[list[pd.DataFrame], str]:
    root = Path(dataset_dir)
    cascade_paths = [root / f"stage{stage}_{family}_{split}.tsv"
                     for stage in (1, 2) for split in SPLITS]
    if any(path.exists() for path in cascade_paths):
        missing = [str(path) for path in cascade_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Incomplete cascade dataset; missing stage file(s): " + ", ".join(missing)
            )
        return [
            _read_and_validate_cascade_split(root, family, split) for split in SPLITS
        ], "cascade"

    frames = []
    for split in SPLITS:
        path = root / f"{family}_{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        frames.append(pd.read_csv(path, sep="\t").assign(split=split))
    return frames, "legacy"


def load_family(
    family: str,
    dataset_dir: str,
    sequences: dict[str, str],
    max_seq_len: int = MAX_SEQ_LEN,
    drop_contradictory: bool = False,
) -> tuple[pd.DataFrame, FilterReport]:
    """Load a family's three splits and apply the DESIGN §2.3 filter chain.

    Filter order matters: rows with pos > max_seq_len must be dropped BEFORE
    embedding generation, otherwise the seq[:max_len] truncation silently yields
    the wrong residue.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family!r}")

    frames, source_layout = _load_source_frames(family, dataset_dir)
    df = pd.concat(frames, ignore_index=True)

    rep = FilterReport(family=family, raw_rows=len(df), source_layout=source_layout)

    # 0) SNP notation
    parsed = df["snp"].map(parse_snp)
    ok = parsed.notna()
    rep.add("malformed SNP notation", int((~ok).sum()))
    df, parsed = df[ok].copy(), parsed[ok]
    df["wt_aa"] = [p[0] for p in parsed]
    df["pos"] = [p[1] for p in parsed]
    df["mut_aa"] = [p[2] for p in parsed]

    # 1) proteins with no sequence
    ok = df["protein_id"].isin(sequences)
    rep.add("protein has no sequence", int((~ok).sum()))
    df = df[ok].copy()

    # 2) positions beyond the ESM truncation limit (local embedding undefined)
    ok = df["pos"] <= max_seq_len
    rep.add(f"pos > {max_seq_len}", int((~ok).sum()))
    df = df[ok].copy()

    # 3) position beyond the sequence length
    seq_len = df["protein_id"].map(lambda p: len(sequences[p]))
    ok = df["pos"] <= seq_len
    rep.add("position beyond sequence", int((~ok).sum()))
    df = df[ok].copy()

    # 4) WT residue disagrees with the sequence — this also checks isoform/numbering
    ok = pd.Series(
        [sequences[p][i - 1] == w for p, i, w in zip(df["protein_id"], df["pos"], df["wt_aa"])],
        index=df.index,
    )
    rep.add("WT residue mismatch", int((~ok).sum()))
    df = df[ok].copy()

    # 5) optional: rows with contradictory labels for the same (protein, snp, GO)
    if drop_contradictory:
        key = ["protein_id", "snp", "matched_go_id"]
        nlab = df.groupby(key)["label"].transform("nunique")
        ok = nlab == 1
        rep.add("contradictory label (optional)", int((~ok).sum()))
        df = df[ok].copy()

    # Derived columns — used by the DESIGN §5.2 breakdowns
    df["variant_key"] = df["protein_id"] + "_" + df["snp"]
    df["wt_local_key"] = df["protein_id"] + "_" + df["pos"].astype(str)
    df["is_alanine"] = df["mut_aa"] == "A"

    rep.kept_rows = len(df)
    rep.n_proteins = df["protein_id"].nunique()
    rep.n_go_terms = df["matched_go_id"].nunique()
    rep.per_split = {s: int((df["split"] == s).sum()) for s in SPLITS}
    rep.per_label = {l: int((df["label"] == l).sum()) for l in LABELS}
    return df.reset_index(drop=True), rep


def build_go_vocab(df: pd.DataFrame) -> dict[str, int]:
    """DESIGN §3.3 — vocabulary built from the family's train u val u test terms.

    No row is dropped; the price is that ~16% of hard's val/test is scored on a
    neuron never seen in training (still at random init). This is a deliberate
    decision, made visible through the seen_in_train flag (DESIGN §5.2).
    """
    terms = sorted(df["matched_go_id"].unique())
    return {go: i for i, go in enumerate(terms)}


def attach_indices(df: pd.DataFrame, go_vocab: dict[str, int]) -> pd.DataFrame:
    """Add the go_idx, label_idx and seen_in_train columns."""
    from phase_3_variant_decoder.config import LABEL_TO_IDX

    out = df.copy()
    out["go_idx"] = out["matched_go_id"].map(go_vocab).astype(int)
    out["label_idx"] = out["label"].map(LABEL_TO_IDX).astype(int)
    train_terms = set(out.loc[out["split"] == "train", "matched_go_id"])
    out["seen_in_train"] = out["matched_go_id"].isin(train_terms)
    return out


def prepare(cfg: VariantDecoderConfig) -> tuple[pd.DataFrame, dict[str, int], FilterReport]:
    """End to end: load -> filter -> build vocab -> attach indices."""
    sequences = load_sequences(cfg.sequences_json)
    df, rep = load_family(
        cfg.family, cfg.dataset_dir, sequences,
        max_seq_len=cfg.max_seq_len, drop_contradictory=cfg.drop_contradictory,
    )
    go_vocab = build_go_vocab(df)
    return attach_indices(df, go_vocab), go_vocab, rep


def main():
    parser = argparse.ArgumentParser(description="Phase 3 data preparation report")
    parser.add_argument("--family", choices=FAMILIES, default=None,
                        help="both families are reported if omitted")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--sequences_json", default=None)
    parser.add_argument("--drop_contradictory", action="store_true")
    args = parser.parse_args()

    families = [args.family] if args.family else list(FAMILIES)
    for fam in families:
        cfg = VariantDecoderConfig(family=fam)
        if args.dataset_dir:
            cfg.dataset_dir = args.dataset_dir
        if args.sequences_json:
            cfg.sequences_json = args.sequences_json
        cfg.drop_contradictory = args.drop_contradictory

        df, go_vocab, rep = prepare(cfg)
        print(rep.render())
        seen = df["seen_in_train"]
        for split in ("val", "test"):
            m = df["split"] == split
            if m.sum():
                unseen_pct = 100.0 * (~seen[m]).mean()
                print(f"  {split}: rows with a GO term unseen in train  {unseen_pct:.1f}%")
        print(f"  vocabulary size: {len(go_vocab):,}")
        print(f"  alanine fraction: {100.0 * df['is_alanine'].mean():.1f}%")
        print()


if __name__ == "__main__":
    main()
