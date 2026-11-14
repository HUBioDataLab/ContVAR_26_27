"""Rebuild presentation artifacts from saved results without training/evaluation.

Original JSON, plots, checkpoints and W&B runs remain intact. Optional upload
creates a new presentation-only run. Diagnostic suites have the same workflow
via present_diagnostics; this entry point also handles all comparison formats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_3_variant_decoder.config import WANDB_ENTITY, WANDB_PROJECT
from phase_3_variant_decoder.presentation_terms import (
    REPORTING_VERSION, configuration_metadata, log_reporting_guide,
)


def _histories(results: dict) -> dict:
    histories = {}
    for condition, result in results.items():
        path = Path(result["checkpoint"]).parent / "history.json"
        if not path.is_file():
            raise FileNotFoundError(f"Saved training history required for {condition}: {path}")
        histories[condition] = json.loads(path.read_text(encoding="utf-8"))
    return histories


def refresh(source: str | Path, *, output_dir: str | Path | None = None,
            upload: bool = False, project: str = WANDB_PROJECT,
            entity: str | None = WANDB_ENTITY, name: str | None = None,
            mode: str = "online", config_json: str | Path | None = None) -> dict:
    path = Path(source).resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    kind = saved.get("format")
    destination = Path(output_dir).resolve() if output_dir else path.parent / "presentation_v3"
    if destination == path.parent:
        raise ValueError("Use a separate output directory to preserve the original artifacts")
    if kind == "contvar_cascade_diagnostic_suite_v1":
        from phase_3_variant_decoder.present_diagnostics import present
        return present(str(path), str(destination), upload=upload, project=project,
                       entity=entity, name=name or "phase3-diagnostic-presentation-v3", mode=mode)
    formats = {
        "contvar_cascade_weighting_comparison_v1": "weighting",
        "contvar_decoder_architecture_comparison_v1": "architectures",
        "contvar_learned_effect_threshold_comparison_v1": "learned_threshold",
        "contvar_cascade_threshold_selection_v1": "threshold_selection",
    }
    if kind not in formats:
        raise ValueError(f"Unsupported saved report format: {kind!r}")
    config = saved.get("config")
    if config_json:
        supplied = json.loads(Path(config_json).read_text(encoding="utf-8"))
        config = supplied.get("config", supplied)
    if not config or not {"family", "embedding_type", "input_mode", "seed"}.issubset(config):
        raise ValueError("Saved configuration is incomplete. Supply --config_json with the original run configuration; encoder history will not be guessed from a filename.")

    condition = formats[kind]
    if condition == "weighting":
        from phase_3_variant_decoder.compare_weighting import generate_comparison_report, log_comparison_to_wandb
        results = saved["runs"]
        histories = _histories(results)
        artifacts = generate_comparison_report(results, histories, destination, config=config)
        logger = lambda run, wandb: log_comparison_to_wandb(run, wandb, results, histories, artifacts)
    elif condition == "architectures":
        from phase_3_variant_decoder.compare_architectures import generate_report, _log_final_wandb
        results = saved["runs"]
        artifacts = generate_report(results, destination, config=config)
        logger = lambda run, wandb: _log_final_wandb(run, wandb, results, artifacts)
    elif condition == "learned_threshold":
        from phase_3_variant_decoder.compare_learned_threshold import generate_report, _log_wandb
        results = saved["runs"]
        histories = _histories(results)
        artifacts = generate_report(results, histories, destination, config=config)
        logger = lambda run, wandb: _log_wandb(run, wandb, results, histories, artifacts)
    else:
        from phase_3_variant_decoder.compare_threshold_selection import generate_report, _log_wandb
        results, curve = saved["results"], saved["curve"]
        artifacts = generate_report(curve, results, destination, config=config)
        logger = lambda run, wandb: _log_wandb(run, wandb, curve, results, artifacts)

    url = None
    if upload:
        import wandb
        run = wandb.init(project=project, entity=entity, mode=mode,
                         name=name or f"phase3-{condition}-presentation-v2",
                         tags=["phase3", "presentation", "reporting-v2", condition],
                         config={**configuration_metadata(config), "presentation_only": True,
                                 "source_json": str(path), "source_wandb_url": saved.get("wandb_url"),
                                 "source_training_config": config, "experiment": condition})
        try:
            log_reporting_guide(run, wandb, [config])
            logger(run, wandb)
            artifact = wandb.Artifact(f"phase3-{condition}-presentation", type="evaluation")
            for filename in artifacts.values():
                artifact.add_file(filename)
            run.log_artifact(artifact)
            url = run.url
            run.finish()
        except Exception:
            run.finish(exit_code=1)
            raise
    result = {"reporting_version": REPORTING_VERSION, "source_json": str(path),
              "presentation_only": True, "artifacts": artifacts, "wandb_url": url}
    (destination / "refreshed_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_json")
    parser.add_argument("--output_dir")
    parser.add_argument("--config_json", help="Original configuration for older result files that did not save it")
    parser.add_argument("--upload", action="store_true", help="Create a separate W&B presentation run")
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--name")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    result = refresh(args.source_json, output_dir=args.output_dir, upload=args.upload,
                     project=args.project, entity=args.entity, name=args.name,
                     mode=args.wandb_mode, config_json=args.config_json)
    print("[DONE] presentation refreshed; no training or evaluation performed")
    for key, value in result["artifacts"].items():
        print(f"  {key}: {value}")
    if result["wandb_url"]:
        print(f"  wandb: {result['wandb_url']}")


if __name__ == "__main__":
    main()
