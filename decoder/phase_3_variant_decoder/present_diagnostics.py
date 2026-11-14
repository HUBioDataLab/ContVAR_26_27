"""Rebuild and upload presentation panels from a completed diagnostic suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_3_variant_decoder.diagnostic_suite import (
    DiagnosticSpec, generate_report, _log_final_wandb,
)
from phase_3_variant_decoder.config import WANDB_PROJECT, WANDB_ENTITY
from phase_3_variant_decoder.presentation_terms import REPORTING_VERSION


def load_suite(path: str | Path) -> tuple[dict, list[DiagnosticSpec]]:
    suite_path = Path(path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("format") != "contvar_cascade_diagnostic_suite_v1":
        raise ValueError(f"Unsupported diagnostic suite format: {suite.get('format')!r}")
    specs = [DiagnosticSpec(
        family=item["family"], embedding=item["embedding"],
        input_mode=item["input_mode"], seed=int(item["seed"]), axis=item["axis"],
    ) for item in suite["specs"]]
    missing = [spec.key for spec in specs if spec.key not in suite["results"]]
    if missing:
        raise ValueError(f"Suite JSON is missing results: {', '.join(missing)}")
    return suite, specs


def present(suite_json: str, output_dir: str | None, *, upload: bool,
            project: str, entity: str | None, name: str, mode: str = "online") -> dict:
    suite, specs = load_suite(suite_json)
    destination = Path(output_dir) if output_dir else Path(suite_json).parent / "presentation"
    reused_keys = set(suite.get("reused_keys", []))
    artifacts = generate_report(specs, suite["results"], reused_keys, destination,
                                config=suite.get("config"))
    wandb_url = None
    if upload:
        run = None
        try:
            import wandb
            run = wandb.init(
                project=project, entity=entity, name=name, mode=mode,
                tags=["phase3", "cascade", "presentation", "diagnostic-suite"],
                config={
                    "source_suite_json": str(Path(suite_json).resolve()),
                    "source_wandb_url": suite.get("wandb_url"),
                    "matrix": suite["specs"],
                    "presentation_only": True,
                    "reporting_version": REPORTING_VERSION,
                    "source_training_config": suite.get("config", {}),
                    "configuration_scope": "Matrix of configurations; see 00_Overview tables.",
                },
            )
            _log_final_wandb(
                run, wandb, specs, suite["results"], reused_keys, artifacts
            )
            artifact = wandb.Artifact("phase3-diagnostic-presentation", type="evaluation")
            for path in artifacts.values():
                artifact.add_file(path)
            run.log_artifact(artifact)
            run.summary["00_Overview/local_report"] = artifacts["presentation_markdown"]
            wandb_url = run.url
            run.finish()
        except Exception as exc:
            if run is not None:
                run.finish(exit_code=1)
            raise RuntimeError(
                "Presentation files were created locally, but W&B upload failed. "
                "Run `wandb login` or use --no_wandb."
            ) from exc
    return {"artifacts": artifacts, "wandb_url": wandb_url,
            "presentation_only": True, "reporting_version": REPORTING_VERSION}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean presentation from an existing diagnostic_suite.json"
    )
    parser.add_argument("suite_json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--name", default="phase3-decoder-presentation")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    result = present(
        args.suite_json, args.output_dir, upload=not args.no_wandb,
        project=args.project, entity=args.entity, name=args.name,
        mode=args.wandb_mode,
    )
    print("[DONE] presentation")
    for key, value in result["artifacts"].items():
        print(f"  {key}: {value}")
    if result["wandb_url"]:
        print(f"  wandb: {result['wandb_url']}")


if __name__ == "__main__":
    main()
