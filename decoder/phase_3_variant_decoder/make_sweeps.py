"""
Generate sweep YAMLs: family x embedding type x input_mode.

Design: phase_3_variant_decoder/DESIGN.md §5.3

Every YAML is self-contained: fixed settings sit as literal arguments in the
`command` block and the searched hyperparameters arrive via ${args}. No environment
variables need setting, which removes the risk of a path vanishing because a shell
variable was undefined.

TWO PRESETS:

  full   80 trials / 8 parameters   — final search on the full dataset
         2 modes x 2 families x 5 types = 20 sweeps x 80 = 1,600 trials

  lite   20 trials / 4 parameters   — quick scan on partial data
         Only the highest-yield axes are searched; the rest are fixed.
         20 sweeps x 20 = 400 trials

What `lite` fixes and why: `weight_decay`, `batch_size` and `hidden_dims` are
architecture/optimisation settings whose optima do not move much when the data
changes. What stays searched — `margin`, `w_gof`, `w_neutral`, `lr` — are exactly
the axes of the measured problems: class imbalance (gof_recall 0.23, neutral_recall
0.08) and the band landing in the wrong place (w/2 = 0.09).

Single-margin loss: `margin` is shared by LOF/GOF/NEUTRAL (no separate
`margin_neutral`/`margin_ratio` any more — removed at the user's request).

Usage:
    python -m phase_3_variant_decoder.make_sweeps --preset lite
    python -m phase_3_variant_decoder.make_sweeps --preset full
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, SECONDARY_REQUIRED, SWEEP_INPUT_MODES, WANDB_PROJECT,
)

# concat_phase0 → primary seq_phase0, secondary raw_esm
PRIMARY_OF = {"concat_phase0": "seq_phase0", "concat_phase1": "seq_phase1"}

_P = {
    "margin": "    distribution: uniform\n    min: 0.05\n    max: 0.6",
    "w_gof": "    distribution: log_uniform_values\n    min: 1.0\n    max: 20.0",
    "w_neutral": "    distribution: log_uniform_values\n    min: 1.0\n    max: 20.0",
    "lr": "    distribution: log_uniform_values\n    min: 0.0001\n    max: 0.005",
    "weight_decay": "    distribution: log_uniform_values\n    min: 0.000001\n    max: 0.01",
    "dropout": "    distribution: uniform\n    min: 0.1\n    max: 0.5",
    "batch_size": "    values: [128, 256, 512, 1024]",
    "hidden_dims": '    values: ["512,1024,512", "1024,512", "512,512", "2048,1024,512"]',
}

PRESETS = {
    "full": {
        "trials": 80,
        "epochs": 60,
        "patience": 12,
        "search": ["margin", "w_gof", "w_neutral", "lr",
                   "weight_decay", "dropout", "batch_size", "hidden_dims"],
        "fixed": {},
    },
    "lite": {
        "trials": 20,
        "epochs": 40,
        "patience": 8,
        "search": ["margin", "w_gof", "w_neutral", "lr"],
        "fixed": {"weight_decay": 1e-4, "dropout": 0.3,
                  "batch_size": 256, "hidden_dims": "512,1024,512"},
    },
}


def _parameters_block(preset: dict) -> str:
    lines = []
    for name in preset["search"]:
        lines.append(f"  {name}:")
        lines.append(_P[name])
    for name, value in preset["fixed"].items():
        # Fixed values are also declared under `parameters` as single values, so
        # they show up in the wandb config and are written in full to results.csv.
        lines.append(f"  {name}:")
        lines.append(f"    value: {value!r}" if isinstance(value, str)
                     else f"    value: {value}")
    return "\n".join(lines)


def make_yaml(family: str, embedding: str, input_mode: str, preset_name: str,
              h5_prefix: str, cache_dir: str, project: str, tag: str = "") -> str:
    preset = PRESETS[preset_name]
    primary = PRIMARY_OF.get(embedding, embedding)

    cmd = [
        "  - ${env}", "  - python", "  - -m",
        "  - phase_3_variant_decoder.sweep_train",
        f"  - --family={family}",
        f"  - --embedding={embedding}",
        f"  - --input_mode={input_mode}",
        f"  - --global_h5={h5_prefix}_{primary}_global.h5",
        f"  - --local_h5={h5_prefix}_{primary}_local.h5",
    ]
    if embedding in SECONDARY_REQUIRED:
        cmd += [
            f"  - --global_h5_secondary={h5_prefix}_raw_esm_global.h5",
            f"  - --local_h5_secondary={h5_prefix}_raw_esm_local.h5",
        ]
    cmd += [
        f"  - --cache_dir={cache_dir}",
        f"  - --epochs={preset['epochs']}",
        f"  - --patience={preset['patience']}",
        f"  - --preset={preset_name}",
        "  - ${args}",
    ]

    stem = f"{family}_{embedding}_{input_mode}"
    n_search = len(preset["search"])
    # `tag` distinguishes runs of the same architecture across design iterations
    # within one wandb project (e.g. "-singlemargin") — separate from `project`,
    # which is the other/coarser way to keep iterations apart (a whole new
    # project, as we did here: contvar_phase3_full -> contvar_phase3_single_margin).
    suffix = f"-{tag}" if tag else ""
    sweep_name = f"phase3-{stem}-{preset_name}{suffix}"
    return f"""# Phase 3 hyperparameter search — {family} / {embedding} / input_mode={input_mode}
# preset={preset_name}  ·  {preset['trials']} trials  ·  {n_search} parameters searched
# Generated by: python -m phase_3_variant_decoder.make_sweeps --preset {preset_name}
#
#   wandb sweep phase_3_variant_decoder/sweeps/{preset_name}/{stem}.yaml
#   CUDA_VISIBLE_DEVICES=<gpu> wandb agent <entity>/{project}/<sweep_id>

program: phase_3_variant_decoder/sweep_train.py
project: {project}
name: {sweep_name}
method: bayes
run_cap: {preset['trials']}

metric:
  # BUG FOUND & FIXED HERE: this used to say "val_macro_f1" (underscore) — but
  # build_wandb_log() (train.py) namespaces every wandb key with "/" for
  # dashboard grouping, so the actually-logged key is "val/macro_f1" (slash).
  # "val_macro_f1" was never once written to any run's history/summary in
  # either sweep (v1 or v2) — the Bayesian search had no way to read its own
  # declared objective, and wandb's sweep.best_run() (used by apply_best.py)
  # silently fell back to some undefined tie-break instead of the true best
  # trial. Confirmed empirically: a "collect" retrain of what wandb called
  # the "best" trial used hyperparameters unrelated to the trial that
  # actually scored highest in results.csv.
  name: val/macro_f1
  goal: maximize

# Trials that collapse into the degenerate solution show it within the first few
# epochs; cutting them early saves most of the budget.
early_terminate:
  type: hyperband
  min_iter: 6
  eta: 3

command:
{chr(10).join(cmd)}

parameters:
{_parameters_block(preset)}
"""


def main():
    p = argparse.ArgumentParser(description="Generate sweep YAMLs")
    p.add_argument("--preset", choices=sorted(PRESETS), default="lite")
    p.add_argument("--input_modes", nargs="+", default=list(SWEEP_INPUT_MODES),
                   help=f"defaults to {list(SWEEP_INPUT_MODES)}; 'local' is deliberately excluded")
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--embeddings", nargs="+", default=list(EMBEDDING_TYPES))
    p.add_argument("--h5_prefix", default="data/processed/phase3")
    p.add_argument("--cache_dir", default="data/processed/phase3_cache")
    p.add_argument("--out_dir", default="phase_3_variant_decoder/sweeps")
    p.add_argument("--project", default=WANDB_PROJECT)
    p.add_argument("--tag", default="",
                   help="suffix added to each sweep's name (e.g. 'singlemargin') to mark "
                        "a design iteration when reusing the same --project")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    out = Path(args.out_dir) / args.preset
    out.mkdir(parents=True, exist_ok=True)

    written = []
    for family in args.families:
        for embedding in args.embeddings:
            for mode in args.input_modes:
                path = out / f"{family}_{embedding}_{mode}.yaml"
                path.write_text(
                    make_yaml(family, embedding, mode, args.preset,
                              args.h5_prefix, args.cache_dir, args.project, args.tag),
                    encoding="utf-8",
                )
                written.append(path)

    total = len(written) * preset["trials"]
    print(f"[DONE] preset={args.preset}  ·  {len(written)} sweep  ·  "
          f"{preset['trials']} trials/sweep  ·  {total:,} trials total")
    print(f"       searched: {', '.join(preset['search'])}")
    if preset["fixed"]:
        print(f"       fixed   : {', '.join(f'{k}={v}' for k, v in preset['fixed'].items())}")
    print(f"       → {out}/")


if __name__ == "__main__":
    main()
