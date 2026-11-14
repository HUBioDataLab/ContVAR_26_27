"""
Variant-specific ESM embedding builder.

Her FASTA dosyası bir varyant proteini temsil eder.
Dosya adı formatı: {protein_id}_{snp}.fasta  (ör. A0A061I403_E231G.fasta)
H5 çıktısında key bu stem'dir: "A0A061I403_E231G"

Bu format benchmark_eval.py'nin beklediğiyle birebir eşleşir.

Kullanım:
    python -m phase_2_decoder.build_variant_embeddings
    python -m phase_2_decoder.build_variant_embeddings --cuda 0
    python -m phase_2_decoder.build_variant_embeddings \\
        --fasta_dir data/raw/variant_specific_go_benchmark_filtered_fastas \\
        --out data/processed/variant_embeddings.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

try:
    from .config import pick_device
    from .build_embeddings import load_esm2_model, load_backbone_if_given, mean_pool_last_layer
except ImportError:
    from config import pick_device
    from build_embeddings import load_esm2_model, load_backbone_if_given, mean_pool_last_layer


def parse_fasta(path: Path) -> str:
    """Tek-sequence FASTA'dan sekansı döndürür."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def build_variant_embeddings(args):
    device = torch.device(pick_device(args.cuda))
    print(f"[INFO] Device: {device}")

    fasta_dir = Path(args.fasta_dir)
    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"FASTA dosyası bulunamadı: {fasta_dir}")
    print(f"[INFO] {len(fasta_files)} varyant FASTA bulundu")

    # key → sequence
    variants: list[tuple[str, str]] = []
    for fp in fasta_files:
        key = fp.stem  # "A0A061I403_E231G"
        seq = parse_fasta(fp)
        if seq:
            variants.append((key, seq))
        else:
            print(f"[WARN] Boş sekans, atlanıyor: {fp.name}")

    # Kısa sekansları önce işle (bellek açısından verimli)
    variants.sort(key=lambda x: len(x[1]))
    if args.limit is not None:
        variants = variants[: args.limit]
        print(f"[INFO] --limit ile {args.limit} varyanta kısıtlandı")

    print(f"[INFO] Embed edilecek varyant sayısı: {len(variants)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Mevcut H5'e ekleme ya da sıfırdan oluşturma
    already_done: set[str] = set()
    if out_path.exists() and args.resume:
        with h5py.File(out_path, "r") as h5:
            already_done = set(h5.keys())
        print(f"[INFO] Resume: {len(already_done)} varyant zaten mevcut, atlanacak")

    todo = [(k, s) for k, s in variants if k not in already_done]
    print(f"[INFO] İşlenecek: {len(todo)} varyant")

    if not todo:
        print("[DONE] Yeni varyant yok, H5 güncel.")
        return

    model, alphabet = load_esm2_model(args.esm_model)
    model = load_backbone_if_given(model, args.backbone, device)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    mode = "a" if (out_path.exists() and args.resume) else "w"
    with h5py.File(out_path, mode) as h5:
        for start in tqdm(range(0, len(todo), args.batch_size), desc="Embedding variants"):
            batch = todo[start : start + args.batch_size]
            pooled = mean_pool_last_layer(
                model, batch_converter, alphabet, batch,
                device=device, max_len=args.max_seq_len,
            )
            for (key, _), emb in zip(batch, pooled):
                if key in h5:
                    del h5[key]
                h5.create_dataset(key, data=emb.numpy())

    print(f"[DONE] Varyant embedding'leri kaydedildi → {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build variant-level ESM embeddings for benchmark evaluation"
    )
    parser.add_argument(
        "--fasta_dir",
        default=str(repo_root / "data" / "raw" / "variant_specific_go_benchmark_filtered_fastas"),
        help="Varyant FASTA dosyalarının bulunduğu klasör",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "data" / "processed" / "variant_embeddings.h5"),
        help="Çıktı H5 dosyası",
    )
    parser.add_argument(
        "--esm_model",
        default="esm2_t33_650M_UR50D",
        choices=["esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"],
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Fine-tuned backbone checkpoint (esm_state_dict içeren .pt dosyası)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument(
        "--cuda", default=None,
        help="Device override: örn 0, 1, cuda:0, mps, cpu",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Smoke test için ilk N varyantı işle",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Mevcut H5'e devam et — zaten olan key'leri atla",
    )
    args = parser.parse_args()

    build_variant_embeddings(args)


if __name__ == "__main__":
    main()
