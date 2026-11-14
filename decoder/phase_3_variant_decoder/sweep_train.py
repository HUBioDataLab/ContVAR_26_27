"""
WandB Sweep agent entry point.

Design: phase_3_variant_decoder/DESIGN.md §5.3 + sweeps/

Uses `train.py`'s `fit()` unchanged — the training logic lives in one place and the
sweep only supplies hyperparameters.

Fixed settings (family, embedding, H5 paths, cache) sit as literal arguments in the
sweep YAML's `command` block; the searched hyperparameters arrive via `${args}`.
That way no environment variables need setting and the YAML is self-contained.

NOTE: single-margin loss — `margin` is shared by LOF/GOF/NEUTRAL, there is no
`margin_neutral`/`margin_ratio` any more (removed at the user's request).

Can also be run directly (outside a sweep, as a single test run).
"""

from __future__ import annotations

import argparse

from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, INPUT_MODES, SECONDARY_REQUIRED, VariantDecoderConfig,
)
from phase_3_variant_decoder import results
from phase_3_variant_decoder.train import fit, prepare_run


def parse_hidden_dims(text: str) -> list[int]:
    """'512,1024,512' -> [512, 1024, 512]"""
    return [int(x) for x in str(text).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Phase 3 sweep trial")
    # ── fixed settings (from the YAML's command block) ──────────────────
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--embedding", choices=EMBEDDING_TYPES, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    p.add_argument("--global_h5", required=True)
    p.add_argument("--local_h5", required=True)
    p.add_argument("--global_h5_secondary", default=None)
    p.add_argument("--local_h5_secondary", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--results_csv", default=str(results.DEFAULT_PATH))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--preset", default="",
                   help="make_sweeps.py preset name (lite/full) — recorded in results.csv "
                        "so batches from different searches can be told apart later")
    p.add_argument("--out_dir", default=str(results.DEFAULT_PATH.parent),
                   help="checkpoint dir — every trial compares against the best saved so far "
                        "for this architecture and only overwrites if it improves (see "
                        "train._save); pass empty string to disable checkpoint writing")
    # ── searched hyperparameters (${args}) ──────────────────────────────
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--w_gof", type=float, default=1.0)
    p.add_argument("--w_neutral", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dims", default="512,1024,512")
    p.add_argument("--cuda", default=None)
    args = p.parse_args()

    if args.embedding in SECONDARY_REQUIRED and not (
        args.global_h5_secondary and args.local_h5_secondary
    ):
        raise SystemExit(f"embedding={args.embedding} requires the *_secondary H5 files")

    cfg = VariantDecoderConfig(family=args.family, embedding_type=args.embedding,
                               input_mode=args.input_mode)
    cfg.source = "sweep"
    cfg.preset = args.preset
    cfg.global_h5 = args.global_h5
    cfg.local_h5 = args.local_h5
    cfg.global_h5_secondary = args.global_h5_secondary
    cfg.local_h5_secondary = args.local_h5_secondary
    cfg.cuda = args.cuda
    cfg.epochs = args.epochs
    cfg.early_stop_patience = args.patience

    cfg.margin = args.margin
    cfg.w_lof = 1.0                                        # fixed: only the ratios matter
    cfg.w_gof = args.w_gof
    cfg.w_neutral = args.w_neutral
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.dropout = args.dropout
    cfg.batch_size = args.batch_size
    cfg.hidden_dims = parse_hidden_dims(args.hidden_dims)

    import wandb
    tags, meta = cfg.wandb_metadata()
    run = wandb.init(
        project=cfg.wandb_project, tags=tags,
        config={
            **meta, "preset": cfg.preset,
            "margin": cfg.margin,
            "w_lof": cfg.w_lof, "w_gof": cfg.w_gof, "w_neutral": cfg.w_neutral,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "dropout": cfg.dropout,
            "batch_size": cfg.batch_size, "hidden_dims": args.hidden_dims,
            "epochs": cfg.epochs,
        },
    )
    cfg.sweep_id = getattr(getattr(run, "sweep", None), "id", "") or ""

    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=args.cache_dir, quiet=True)
    best = fit(cfg, tensors, views, go_vocab, device,
               out_dir=(args.out_dir or None), run=run, quiet=True,
               results_csv=args.results_csv)

    run.summary.update({f"best_{k}": v for k, v in best.items()
                        if isinstance(v, (int, float, str))})
    run.finish()


if __name__ == "__main__":
    main()
