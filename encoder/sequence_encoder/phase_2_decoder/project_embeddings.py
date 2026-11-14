"""
Project protein-level embeddings with the Phase1 DMS global projection head.

Bu script, GO-pretrained backbone'dan üretilmiş protein embedding H5 dosyasını
alır ve Phase1 DMS triplet modelinin global projection head'inden geçirerek
yeni bir H5 üretir.

Amaç:
  pretrained_model + dms fine-tuning + decoder ablation'ında
  DMS-adapted temsil kullanmak.

Not:
  Mevcut Phase1 kodu backbone checkpoint üretmediği için burada kullandığımız
  temsil, "GO-pretrained backbone embedding -> DMS global projection head"
  akışıdır.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .config import pick_device
except ImportError:
    from config import pick_device


class GlobalProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def load_projection_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_dim = int(ckpt["in_dim"])
    out_dim = int(ckpt["out_dim"])
    model = GlobalProjectionHead(in_dim=in_dim, out_dim=out_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, in_dim, out_dim


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def project_h5(in_h5: str, projection_ckpt: str, out_h5: str, batch_size: int = 1024):
    device = torch.device(pick_device())
    print(f"[INFO] Device: {device}")

    model, in_dim, out_dim = load_projection_model(projection_ckpt, device)
    print(f"[INFO] Loaded projection head: in_dim={in_dim}, out_dim={out_dim}")

    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        keys = list(src.keys())
        for batch_keys in tqdm(list(iter_batches(keys, batch_size)), desc="Project embeddings"):
            arr = []
            for key in batch_keys:
                emb = src[key][:]
                if emb.ndim != 1:
                    raise ValueError(f"Expected 1D protein embedding for key={key}, got shape={emb.shape}")
                arr.append(emb)

            batch = torch.tensor(np.stack(arr), dtype=torch.float32, device=device)
            if batch.shape[1] != in_dim:
                raise ValueError(
                    f"Projection input dim mismatch: H5 has {batch.shape[1]}, model expects {in_dim}"
                )

            with torch.no_grad():
                projected = model(batch).cpu().numpy()

            for key, emb in zip(batch_keys, projected):
                dst.create_dataset(key, data=emb, compression="gzip")

    print(f"[DONE] Saved projected embeddings to {out_h5}")


def main():
    parser = argparse.ArgumentParser(description="Project protein-level embeddings with Phase1 DMS global head")
    parser.add_argument("--in_h5", required=True, help="Input protein-level embedding H5")
    parser.add_argument("--projection_ckpt", required=True, help="Phase1 triplet checkpoint (.pt)")
    parser.add_argument("--out_h5", required=True, help="Output projected embedding H5")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    project_h5(args.in_h5, args.projection_ckpt, args.out_h5, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
