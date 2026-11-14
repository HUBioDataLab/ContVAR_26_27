"""Upload an already completed weighting comparison to Weights & Biases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_3_variant_decoder.compare_weighting import (
    _start_wandb, log_comparison_to_wandb, generate_comparison_report,
)
from phase_3_variant_decoder.config import VariantDecoderConfig


ARTIFACT_FILENAMES = {
    "training_curves": "training_curves.png",
    "test_metrics": "test_metrics.png",
    "confusion_matrices": "test_confusion_matrices.png",
    "summary_csv": "comparison_summary.csv",
    "summary_markdown": "comparison_summary.md",
}


def upload(directory: str | Path, *, project: str | None = None,
           entity: str | None = None, name: str | None = None) -> str:
    root = Path(directory).resolve()
    comparison_path = root / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Comparison JSON not found: {comparison_path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("format") != "contvar_cascade_weighting_comparison_v1":
        raise ValueError(f"Unsupported comparison format: {comparison.get('format')}")

    cfg = VariantDecoderConfig()
    for key, value in comparison["config"].items():
        setattr(cfg, key, value)
    cfg.wandb_enabled = True
    if project:
        cfg.wandb_project = project
    if entity:
        cfg.wandb_entity = entity
    cfg.wandb_run_name = name or (
        f"phase3-weighting-{cfg.family}-{cfg.embedding_type}-{cfg.input_mode}-uploaded"
    )

    histories = {}
    for weighting, result in comparison["runs"].items():
        checkpoint = Path(result["checkpoint"])
        history_path = checkpoint.parent / "history.json"
        if not history_path.is_file():
            raise FileNotFoundError(f"History not found: {history_path}")
        histories[weighting] = json.loads(history_path.read_text(encoding="utf-8"))

    # Re-render from saved metrics so uploads do not reuse obsolete labels.
    # Keep the original comparison files intact.
    artifacts = generate_comparison_report(
        comparison["runs"], histories, root / "presentation_v2", config=vars(cfg))

    run, wandb = _start_wandb(cfg)
    if run is None:
        raise RuntimeError("WandB could not be initialized; inspect the warning above")
    try:
        log_comparison_to_wandb(run, wandb, comparison["runs"], histories, artifacts)
        run.summary["comparison/output_dir"] = str(root)
        run.summary["comparison/posthoc_upload"] = True
        url = run.url
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise
    comparison["wandb_url"] = url
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    print(f"[DONE] WandB comparison -> {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a completed cascade weighting comparison to WandB"
    )
    parser.add_argument("comparison_dir")
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    upload(args.comparison_dir, project=args.project, entity=args.entity, name=args.name)


if __name__ == "__main__":
    main()
