"""
Backfill aspect / embedding_type / stage / grid_tag metadata onto existing
WandB runs produced by grid_search.py.

grid_search.py encodes this metadata only inside the run *name*
(e.g. "decoder-grid-sequence_decoder-p-seq_phase1-stage2_lr5e-4_do0p3_..."),
never in wandb.init(config=...). WandB has no substring/"contains" filter in
the run table, so those fields aren't filterable or groupable as-is.

This script parses the run name and, for every matching run:
  - adds structured tags:      aspect_p, emb_seq_phase1, stage_stage2, tag_sequence_decoder
  - backfills run.config with: aspect, embedding_type, stage, grid_tag, hidden_dims_str

Config fields give you exact-match dropdown filtering + "Group by" in the
WandB UI; tags give you quick multi-select filtering. Both are written.

Usage:
    python -m phase_2_decoder.wandb_tag_runs --project contvar_gridsearch
    python -m phase_2_decoder.wandb_tag_runs --project contvar_gridsearch --apply

Runs in dry-run mode (prints what would change) unless --apply is passed.
"""

from __future__ import annotations

import argparse
import re

import wandb
from tqdm import tqdm

try:
    from .config import DecoderConfig
except ImportError:
    from config import DecoderConfig


RUN_NAME_RE = re.compile(
    r"^decoder-grid-(?P<tag>.+)-(?P<aspect>[FPCfpc])-"
    r"(?P<embedding>raw_esm|seq_phase0|seq_phase1|concat_phase0|concat_phase1)-"
    r"(?P<stage>stage1|stage2|stage3|final)"
    r"_lr(?P<lr>[^_]+)_do(?P<dropout>[^_]+)_hd(?P<hidden>[\d\-]+)"
    r"_wd(?P<wd>[^_]+)_pwc(?P<pwc>[^_]+)$"
)


def parse_run_name(name: str) -> dict | None:
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "grid_tag": g["tag"],
        "aspect": g["aspect"].upper(),
        "embedding_type": g["embedding"],
        "stage": g["stage"],
        "hidden_dims_str": g["hidden"],
    }


def desired_tags(meta: dict) -> set[str]:
    return {
        f"aspect_{meta['aspect'].lower()}",
        f"emb_{meta['embedding_type']}",
        f"stage_{meta['stage']}",
        f"tag_{meta['grid_tag']}",
        f"hd_{meta['hidden_dims_str']}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = DecoderConfig().wandb_project
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--entity", default=None, help="WandB entity/workspace (default: your default entity)")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"[INFO] Scanning runs in {path} ({'APPLY' if args.apply else 'DRY-RUN'})")

    runs = api.runs(path)
    n_matched = 0
    n_updated = 0
    n_skipped = 0
    unmatched_names = []

    for run in tqdm(runs, total=len(runs), desc="Tagging runs", unit="run"):
        meta = parse_run_name(run.name)
        if meta is None:
            n_skipped += 1
            unmatched_names.append(run.name)
            continue
        n_matched += 1

        new_tags = desired_tags(meta)
        current_tags = set(run.tags or [])
        tags_changed = not new_tags.issubset(current_tags)

        config_updates = {
            "aspect": meta["aspect"],
            "embedding_type": meta["embedding_type"],
            "stage": meta["stage"],
            "grid_tag": meta["grid_tag"],
            "hidden_dims_str": meta["hidden_dims_str"],
        }
        config_changed = any(run.config.get(k) != v for k, v in config_updates.items())

        if not tags_changed and not config_changed:
            continue

        tqdm.write(f"  {run.id}  {run.name}")
        if tags_changed:
            tqdm.write(f"    tags   + {sorted(new_tags - current_tags)}")
        if config_changed:
            changed = {k: v for k, v in config_updates.items() if run.config.get(k) != v}
            tqdm.write(f"    config + {changed}")

        if args.apply:
            run.tags = sorted(current_tags | new_tags)
            run.config.update(config_updates)
            run.update()
        n_updated += 1

    print(
        f"\n[DONE] matched={n_matched} updated={n_updated} "
        f"unmatched={n_skipped}"
    )
    if unmatched_names:
        print("[WARN] Runs whose name didn't match the expected grid_search.py pattern "
              "(left untouched):")
        for name in unmatched_names[:20]:
            print(f"    - {name}")
        if len(unmatched_names) > 20:
            print(f"    ... and {len(unmatched_names) - 20} more")
    if not args.apply and n_updated:
        print("\n[INFO] Dry-run only. Re-run with --apply to write these changes to WandB.")


if __name__ == "__main__":
    main()
