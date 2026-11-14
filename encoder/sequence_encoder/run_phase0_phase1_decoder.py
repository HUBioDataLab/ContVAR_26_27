from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path):
    print("\n[RUN]")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_path(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def choose_embedding_path(root: Path, configured_value: str | None, fallback: Path) -> tuple[Path, bool]:
    if configured_value:
        configured_path = Path(resolve_path(root, configured_value))
        if configured_path.exists():
            print(f"[INFO] Reusing existing embedding: {configured_path}")
            return configured_path, False
        print(f"[INFO] Configured embedding not found, will build locally: {configured_path}")
    if fallback.exists():
        print(f"[INFO] Reusing existing local embedding: {fallback}")
        return fallback, False
    return fallback, True


def main():
    parser = argparse.ArgumentParser(description="Run updated_version Phase0 + Phase1 + Decoder pipeline")
    parser.add_argument("--config", default="pipeline_config.json")
    # Phase0 overrides
    parser.add_argument("--phase0_cuda", default=None,
                        help="Phase0 training device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--phase0_epochs", type=int, default=None, help="Phase0 epoch sayısı override")
    parser.add_argument("--phase0_run_name", default=None, help="Phase0 WandB run name override")
    parser.add_argument("--phase0_no_wandb", action="store_true", help="Phase0 WandB'yi devre dışı bırak")
    # Phase1 overrides
    parser.add_argument("--phase1_cuda", default=None,
                        help="Phase1 training device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--phase1_epochs", type=int, default=None, help="Phase1 epoch sayısı override")
    parser.add_argument("--phase1_run_name", default=None, help="Phase1 WandB run name override")
    parser.add_argument("--phase1_no_wandb", action="store_true", help="Phase1 WandB'yi devre dışı bırak")
    # Decoder overrides
    parser.add_argument("--decoder_cuda", default=None,
                        help="Decoder/build_embeddings device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--aspect", default=None, choices=["F", "P", "C"],
                        help="GO aspect override: F=MF, P=BP, C=CC")
    parser.add_argument("--wandb_run_name", default=None, help="Decoder WandB run name override")
    parser.add_argument("--wandb_project", default=None, help="Decoder WandB project override")
    parser.add_argument("--no_wandb", action="store_true", help="Decoder WandB'yi devre dışı bırak")
    parser.add_argument("--epochs", type=int, default=None, help="Decoder epoch sayısı override")
    parser.add_argument("--lr", type=float, default=None, help="Decoder learning rate override")
    parser.add_argument("--batch_size", type=int, default=None, help="Decoder batch size override")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Decoder early stop patience override")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            config_path = (root / config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    python = cfg.get("python", sys.executable)
    esm_model = cfg.get("esm_model", "esm2_t33_650M_UR50D")
    work_dir = Path(resolve_path(root, cfg.get("work_dir", "artifacts/pipeline_run")))
    work_dir.mkdir(parents=True, exist_ok=True)

    paths = cfg["paths"]
    go_dataset = resolve_path(root, paths["go_dataset"])
    go_split = resolve_path(root, paths["go_split"])
    triplet_json = resolve_path(root, paths["triplet_json"])
    phase1_split_json = resolve_path(root, paths.get("phase1_split_json"))
    phase1_reference_csv = resolve_path(root, paths["phase1_reference_csv"])
    phase1_proteins_csv = resolve_path(root, paths.get("phase1_proteins_csv"))
    goa_tsv = resolve_path(root, paths["goa_tsv"])
    sequences_json = resolve_path(root, paths["sequences_json"])

    phase0_dir = work_dir / "phase0"
    phase1_dir = work_dir / "phase1"
    decoder_dir = work_dir / "decoder"
    phase0_dir.mkdir(exist_ok=True)
    phase1_dir.mkdir(exist_ok=True)
    decoder_dir.mkdir(exist_ok=True)

    trained_phase0_backbone = phase0_dir / "full_model_best.pt"
    trained_phase1_backbone = phase1_dir / "phase1_backbone_best.pt"

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    phase0_cfg = cfg.get("phase0", {})
    configured_phase0_backbone = phase0_cfg.get("checkpoint")
    if configured_phase0_backbone:
        phase0_backbone = Path(resolve_path(root, configured_phase0_backbone))
    else:
        phase0_backbone = trained_phase0_backbone

    if phase0_cfg.get("enabled", True):
        phase0_cuda = args.phase0_cuda if args.phase0_cuda is not None else phase0_cfg.get("cuda")
        phase0_epochs = args.phase0_epochs if args.phase0_epochs is not None else phase0_cfg.get("epochs", 30)
        phase0_run_name = (args.phase0_run_name if args.phase0_run_name is not None
                           else phase0_cfg.get("run_name", "phase0_go_pretrain"))
        phase0_no_wandb = args.phase0_no_wandb or phase0_cfg.get("no_wandb", False)
        cmd = [
            python,
            str(root / "phase_0_go_pretrained" / "train_go.py"),
            "--dataset", go_dataset,
            "--split_file", go_split,
            "--esm_model", esm_model,
            "--epochs", str(phase0_epochs),
            "--batch_size", str(phase0_cfg.get("batch_size", 1)),
            "--accum_steps", str(phase0_cfg.get("accum_steps", 8)),
            "--freeze_layers", str(phase0_cfg.get("freeze_layers", 25)),
            "--samples_per_epoch", str(phase0_cfg.get("samples_per_epoch", 10000)),
            "--max_seq_len", str(phase0_cfg.get("max_seq_len", 1022)),
            "--val_every_samples", str(phase0_cfg.get("val_every_samples", 2000)),
            "--checkpoint_every", str(phase0_cfg.get("checkpoint_every", 5)),
            "--keep_checkpoints", str(phase0_cfg.get("keep_checkpoints", 3)),
            "--out", str(phase0_dir),
            "--run_name", phase0_run_name,
        ]
        if phase0_cuda is not None:
            cmd += ["--cuda", str(phase0_cuda)]
        if phase0_no_wandb:
            cmd.append("--no_wandb")
        resume_path = phase0_cfg.get("resume")
        if resume_path:
            cmd += ["--resume", resolve_path(root, resume_path)]
        run(cmd, cwd=root)
        phase0_backbone = trained_phase0_backbone

    if not phase0_backbone.exists():
        raise FileNotFoundError(f"Phase0 model not found: {phase0_backbone}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    phase1_cfg = cfg.get("phase1", {})
    configured_phase1_backbone = phase1_cfg.get("checkpoint")
    if configured_phase1_backbone:
        phase1_backbone = Path(resolve_path(root, configured_phase1_backbone))
    else:
        phase1_backbone = trained_phase1_backbone

    if phase1_cfg.get("enabled", True):
        phase1_cuda = args.phase1_cuda if args.phase1_cuda is not None else phase1_cfg.get("cuda")
        phase1_epochs = args.phase1_epochs if args.phase1_epochs is not None else phase1_cfg.get("epochs", 20)
        phase1_run_name = (args.phase1_run_name if args.phase1_run_name is not None
                           else phase1_cfg.get("run_name", "phase1_dms_backbone_finetune"))
        phase1_no_wandb = args.phase1_no_wandb or phase1_cfg.get("no_wandb", False)
        cmd = [
            python,
            str(root / "phase_1_dms_fine_tuning" / "phase1_dms_backbone_finetune.py"),
            "--backbone", str(phase0_backbone),
            "--esm_model", esm_model,
            "--triplet_json", triplet_json,
            "--reference_csv", phase1_reference_csv,
            "--proteins_csv", phase1_proteins_csv,
            "--out_dir", str(phase1_dir),
            "--epochs", str(phase1_epochs),
            "--batch_size", str(phase1_cfg.get("batch_size", 4)),
            "--last_trainable_layers", str(phase1_cfg.get("last_trainable_layers", 8)),
            "--max_seq_len", str(phase1_cfg.get("max_seq_len", 1022)),
            "--project_name", str(phase1_cfg.get("project_name", "contvar-phase1-dms-backbone")),
            "--run_name", phase1_run_name,
        ]
        if phase1_cfg.get("limit_families") is not None:
            cmd += ["--limit_families", str(phase1_cfg["limit_families"])]
        if phase1_split_json is not None:
            cmd += ["--split_json", phase1_split_json]
        if phase1_cuda is not None:
            cmd += ["--cuda", str(phase1_cuda)]
        if phase1_no_wandb:
            cmd.append("--no_wandb")
        run(cmd, cwd=root)
        phase1_backbone = trained_phase1_backbone

    if not phase1_backbone.exists():
        raise FileNotFoundError(f"Phase1 backbone not found: {phase1_backbone}")

    # ── Decoder ──────────────────────────────────────────────────────────────
    # CLI args override JSON config; JSON config overrides DecoderConfig defaults
    decoder_cfg = cfg.get("decoder", {})
    configured_decoder_h5 = decoder_cfg.get("seq_phase1_h5")
    decoder_h5, should_build_embeddings = choose_embedding_path(
        root=root,
        configured_value=configured_decoder_h5,
        fallback=decoder_dir / f"{esm_model}_phase1_embeddings.h5",
    )
    decoder_cuda = args.decoder_cuda if args.decoder_cuda is not None else decoder_cfg.get("cuda")
    variant_h5 = decoder_cfg.get("variant_h5")
    aspect = args.aspect if args.aspect is not None else str(decoder_cfg.get("aspect", "F"))
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else decoder_cfg.get("wandb_run_name")
    wandb_project = args.wandb_project if args.wandb_project is not None else decoder_cfg.get("wandb_project")
    no_wandb = args.no_wandb or decoder_cfg.get("no_wandb", False)
    epochs = args.epochs if args.epochs is not None else decoder_cfg.get("epochs")
    lr = args.lr if args.lr is not None else decoder_cfg.get("lr")
    batch_size = args.batch_size if args.batch_size is not None else decoder_cfg.get("batch_size")
    early_stop_patience = (args.early_stop_patience if args.early_stop_patience is not None
                           else decoder_cfg.get("early_stop_patience"))

    if decoder_cfg.get("enabled", True):
        if should_build_embeddings:
            cmd = [
                python,
                str(root / "phase_2_decoder" / "build_embeddings.py"),
                "--goa_tsv", goa_tsv,
                "--sequences_json", sequences_json,
                "--split_json", go_split,
                "--out", str(decoder_h5),
                "--esm_model", esm_model,
                "--backbone", str(phase1_backbone),
                "--batch_size", str(decoder_cfg.get("embedding_batch_size", 4)),
                "--max_seq_len", str(decoder_cfg.get("max_seq_len", 1022)),
            ]
            if decoder_cfg.get("limit") is not None:
                cmd += ["--limit", str(decoder_cfg["limit"])]
            if decoder_cuda is not None:
                cmd += ["--cuda", str(decoder_cuda)]
            run(cmd, cwd=root)

        decoder_ckpt = decoder_dir / f"decoder_best_{aspect.lower()}.pt"
        go_vocab_json = decoder_dir / f"go_vocab_{aspect.lower()}.json"
        metrics_out = decoder_dir / f"test_metrics_{aspect.lower()}.json"

        cmd = [
            python,
            str(root / "phase_2_decoder" / "train.py"),
            "--aspect", aspect,
            "--embedding", "seq_phase1",
            "--seq_phase1_h5", str(decoder_h5),
            "--goa_tsv", goa_tsv,
            "--split_json", go_split,
            "--decoder_checkpoint", str(decoder_ckpt),
            "--go_vocab_json", str(go_vocab_json),
        ]
        if decoder_cuda is not None:
            cmd += ["--cuda", str(decoder_cuda)]
        if wandb_run_name is not None:
            cmd += ["--wandb_run_name", wandb_run_name]
        if wandb_project is not None:
            cmd += ["--wandb_project", wandb_project]
        if no_wandb:
            cmd += ["--no_wandb"]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        if lr is not None:
            cmd += ["--lr", str(lr)]
        if batch_size is not None:
            cmd += ["--batch_size", str(batch_size)]
        if early_stop_patience is not None:
            cmd += ["--early_stop_patience", str(early_stop_patience)]
        if variant_h5 is not None:
            cmd += ["--variant_h5", str(variant_h5)]
        run(cmd, cwd=root)

        cmd = [
            python,
            str(root / "phase_2_decoder" / "test_eval.py"),
            "--aspect", aspect,
            "--embedding", "seq_phase1",
            "--seq_phase1_h5", str(decoder_h5),
            "--goa_tsv", goa_tsv,
            "--split_json", go_split,
            "--decoder_checkpoint", str(decoder_ckpt),
            "--go_vocab_json", str(go_vocab_json),
            "--metrics_out", str(metrics_out),
        ]
        if decoder_cuda is not None:
            cmd += ["--cuda", str(decoder_cuda)]
        run(cmd, cwd=root)

    print("\n[DONE] Phase0 + Phase1 + Decoder pipeline finished.")
    print(f"  Phase0 model    : {phase0_backbone}")
    print(f"  Phase1 backbone : {phase1_backbone}")
    print(f"  Decoder H5      : {decoder_h5}")
    print(f"  Decoder dir     : {decoder_dir}")


if __name__ == "__main__":
    main()
