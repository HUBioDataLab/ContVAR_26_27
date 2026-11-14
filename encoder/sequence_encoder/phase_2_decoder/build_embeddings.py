"""
Protein-level embedding H5 builder for Sekans_decoder.

Amaç:
  - GO anotasyonlu proteinler için raw ESM (veya opsiyonel fine-tuned backbone)
    kullanarak protein-level embedding üretmek
  - çıktı olarak {protein_id: [D]} formatında bir H5 yazmak

Kullanım:
  python -m Sekans_decoder.build_embeddings
  python -m Sekans_decoder.build_embeddings --aspect F --limit 1000
  python -m Sekans_decoder.build_embeddings --backbone models/go_pretrained/backbone_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

try:
    from .config import DecoderConfig, pick_device
    from .data.parse_goa import EXCLUDE_EVIDENCE
except ImportError:
    from config import DecoderConfig, pick_device
    from data.parse_goa import EXCLUDE_EVIDENCE


def load_esm2_model(model_name: str):
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "ESM paketi bulunamadi. Kurulum icin `pip install fair-esm` calistirin."
        ) from exc

    pretrained_mod = getattr(esm, "pretrained", None)
    errors = []

    if pretrained_mod is not None:
        loader = getattr(pretrained_mod, model_name, None)
        if loader is not None:
            try:
                model, alphabet = loader()
                print(f"[INFO] Loaded {model_name} via esm.pretrained.{model_name}()")
                return model, alphabet
            except Exception as exc:
                errors.append(f"esm.pretrained.{model_name}(): {exc}")

    if pretrained_mod is not None:
        hub_loader = getattr(pretrained_mod, "load_model_and_alphabet_hub", None)
        if hub_loader is not None:
            try:
                model, alphabet = hub_loader(model_name)
                print(f"[INFO] Loaded {model_name} via load_model_and_alphabet_hub()")
                return model, alphabet
            except Exception as exc:
                errors.append(f"load_model_and_alphabet_hub({model_name}): {exc}")

    try:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        print(f"[INFO] Loaded {model_name} via torch.hub.load()")
        return model, alphabet
    except Exception as exc:
        errors.append(f"torch.hub.load(..., {model_name}): {exc}")

    raise RuntimeError(
        f"ESM2 modeli yuklenemedi: {model_name}. Denenen yollar: {' | '.join(errors)}"
    )


def load_backbone_if_given(model, backbone_path: str | None, device: torch.device):
    if not backbone_path:
        return model
    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    if "esm_state_dict" not in ckpt:
        raise KeyError(f"{backbone_path} icinde 'esm_state_dict' yok.")
    model.load_state_dict(ckpt["esm_state_dict"])
    print(f"[INFO] Loaded fine-tuned backbone from {backbone_path}")
    return model


def parse_goa_proteins(tsv_path: str, aspect: str | None = None) -> set[str]:
    proteins = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 13:
                continue
            evidence = cols[6]
            asp = cols[8]
            if evidence in EXCLUDE_EVIDENCE:
                continue
            if aspect is not None and asp != aspect:
                continue
            proteins.add(cols[1])
    return proteins


def load_sequences(json_path: str) -> dict[str, str]:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    return {
        item["Entry"]: item["Sequence"]
        for item in data
        if item.get("Entry") and item.get("Sequence")
    }


def load_split_ids(split_json: str | None) -> set[str] | None:
    if split_json is None:
        return None
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "protein_to_split" in data:
        return set(data["protein_to_split"].keys())
    raise ValueError(
        "Bu builder su an protein-level split bekliyor: split dosyasi 'protein_to_split' icermeli."
    )


def mean_pool_last_layer(model, batch_converter, alphabet, batch_items, device: torch.device, max_len: int):
    labeled = [(pid, seq[:max_len]) for pid, seq in batch_items]
    _, _, tokens = batch_converter(labeled)
    tokens = tokens.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)

    token_emb = results["representations"][model.num_layers].float()  # [B, L, D]
    padding_mask = (tokens != alphabet.padding_idx).float()
    padding_mask[:, 0] = 0  # BOS
    # EOS: son geçerli pozisyonu vektörize bul
    lengths = (tokens != alphabet.padding_idx).sum(dim=1)  # [B]
    eos_idx = lengths - 1                                   # EOS her sekans için son token
    padding_mask[torch.arange(tokens.shape[0]), eos_idx] = 0

    mask = padding_mask.unsqueeze(-1)
    pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.detach().cpu()


def build_embeddings(args):
    device = torch.device(pick_device(args.cuda))
    print(f"[INFO] Device: {device}")

    goa_proteins = parse_goa_proteins(args.goa_tsv, aspect=args.aspect)
    seqs = load_sequences(args.sequences_json)
    split_ids = load_split_ids(args.split_json)

    candidates = set(goa_proteins) & set(seqs.keys())
    if split_ids is not None:
        candidates &= split_ids

    protein_ids = sorted(candidates, key=lambda pid: len(seqs[pid]))
    if args.limit is not None:
        protein_ids = protein_ids[:args.limit]

    print(f"[INFO] GO-annotated proteins: {len(goa_proteins)}")
    print(f"[INFO] Sequence proteins: {len(seqs)}")
    if split_ids is not None:
        print(f"[INFO] Split proteins: {len(split_ids)}")
    print(f"[INFO] Final proteins to embed: {len(protein_ids)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, alphabet = load_esm2_model(args.esm_model)
    model = load_backbone_if_given(model, args.backbone, device)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    with h5py.File(out_path, "w") as h5:
        for start in tqdm(range(0, len(protein_ids), args.batch_size), desc="Embedding proteins"):
            batch_ids = protein_ids[start:start + args.batch_size]
            batch_items = [(pid, seqs[pid]) for pid in batch_ids]
            pooled = mean_pool_last_layer(
                model, batch_converter, alphabet, batch_items, device=device, max_len=args.max_seq_len
            )
            for pid, emb in zip(batch_ids, pooled):
                h5.create_dataset(pid, data=emb.numpy())

    print(f"[DONE] Saved protein-level embeddings to {out_path}")


def main():
    cfg = DecoderConfig()
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Build protein-level ESM embeddings for Sekans_decoder")
    parser.add_argument("--goa_tsv", default=cfg.goa_tsv)
    parser.add_argument("--sequences_json", default=str(repo_root / "data" / "raw" / "uniprot_dict.json"))
    parser.add_argument("--split_json", default=cfg.split_json)
    parser.add_argument("--out", default=cfg.embeddings_h5)
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D",
                        choices=["esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"])
    parser.add_argument("--backbone", default=None,
                        help="Opsiyonel fine-tuned backbone checkpoint (esm_state_dict iceren dosya)")
    parser.add_argument("--aspect", default=None, choices=["F", "P", "C"],
                        help="Sadece belirli ontology icin embedding uretilsin istiyorsan ver")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--cuda", default=None,
                        help="Device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test icin ilk N proteini embed et")
    args = parser.parse_args()

    build_embeddings(args)


if __name__ == "__main__":
    main()
