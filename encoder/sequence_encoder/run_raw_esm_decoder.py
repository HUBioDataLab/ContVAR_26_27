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
    parser = argparse.ArgumentParser(description="Run updated_version raw ESM + Decoder pipeline")
    parser.add_argument("--config", default="pipeline_config.json")
    parser.add_argument("--decoder_cuda", default=None,
                        help="Decoder/build_embeddings device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--wandb_run_name", default=None, help="WandB run name override")
    parser.add_argument("--wandb_project", default=None, help="WandB project name override")
    parser.add_argument("--no_wandb", action="store_true", help="WandB'yi tamamen devre dışı bırak")
    parser.add_argument("--epochs", type=int, default=None, help="Eğitim epoch sayısı override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Early stopping patience override")
    parser.add_argument("--aspect", default=None, choices=["F", "P", "C"],
                        help="GO aspect override: F=MF, P=BP, C=CC")
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
    work_dir = Path(resolve_path(root, cfg.get("raw_work_dir", "artifacts/pipeline_run_raw_esm_decoder")))
    work_dir.mkdir(parents=True, exist_ok=True)

    paths = cfg["paths"]
    go_split = resolve_path(root, paths["go_split"])
    goa_tsv = resolve_path(root, paths["goa_tsv"])
    sequences_json = resolve_path(root, paths["sequences_json"])

    decoder_dir = work_dir / "decoder"
    decoder_dir.mkdir(exist_ok=True)

    decoder_cfg = cfg.get("decoder", {})
    configured_decoder_h5 = decoder_cfg.get("raw_esm_h5")
    decoder_h5, should_build_embeddings = choose_embedding_path(
        root=root,
        configured_value=configured_decoder_h5,
        fallback=decoder_dir / f"{esm_model}_raw_esm_embeddings.h5",
    )

    # CLI args override JSON config; JSON config overrides DecoderConfig defaults
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
            "--embedding", "raw_esm",
            "--raw_esm_h5", str(decoder_h5),
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
            "--embedding", "raw_esm",
            "--raw_esm_h5", str(decoder_h5),
            "--goa_tsv", goa_tsv,
            "--split_json", go_split,
            "--decoder_checkpoint", str(decoder_ckpt),
            "--go_vocab_json", str(go_vocab_json),
            "--metrics_out", str(metrics_out),
        ]
        if decoder_cuda is not None:
            cmd += ["--cuda", str(decoder_cuda)]
        run(cmd, cwd=root)

    print("\n[DONE] Raw ESM + Decoder pipeline finished.")
    print(f"  Decoder H5  : {decoder_h5}")
    print(f"  Decoder dir : {decoder_dir}")


if __name__ == "__main__":
    main()
