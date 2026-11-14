"""
Test seti üzerinde final değerlendirme.

Kullanım:
    python -m Sekans_decoder.test_eval
    python -m Sekans_decoder.test_eval --aspect P
"""

import json
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from .config import DecoderConfig, EMBEDDING_TYPES, pick_device
    from .data.parse_goa import parse_goa_tsv, build_go_vocab
    from .data.dataset import GOAnnotationDataset
    from .models.ffn_decoder import FFNDecoder
    from .evaluate import evaluate
except ImportError:
    from config import DecoderConfig, EMBEDDING_TYPES, pick_device
    from data.parse_goa import parse_goa_tsv, build_go_vocab
    from data.dataset import GOAnnotationDataset
    from models.ffn_decoder import FFNDecoder
    from evaluate import evaluate

def test_eval(cfg: DecoderConfig = None):
    if cfg is None:
        cfg = DecoderConfig()
    device = torch.device(pick_device(cfg.cuda))
    print(f"[INFO] Device: {device}")

    print(f"Parsing GO annotations (aspect={cfg.aspect})...")
    _obo = cfg.go_obo_path if cfg.use_go_propagation else None
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect, obo_path=_obo,
                               use_qualifier_filter=cfg.use_qualifier_filter)
    go_vocab    = build_go_vocab(annotations, min_freq=cfg.min_go_freq)
    n_classes   = len(go_vocab)
    print(f"GO vocab size: {n_classes}")

    test_ds = GOAnnotationDataset(
        cfg.embeddings_h5, annotations, go_vocab,
        split="test",
        uniref_tsv=cfg.uniref_tsv,
        split_json=cfg.split_json,
        secondary_h5_path=getattr(cfg, "embeddings_h5_secondary", None),
    )
    print(f"Test: {len(test_ds)} proteins")

    if len(test_ds) == 0:
        raise RuntimeError("Test dataset bos. Embedding H5 ile GO annotation/split kesismiyor.")

    cfg.encoder_output_dim = int(test_ds.embs.shape[1])
    print(f"[INFO] Inferred decoder input dim: {cfg.encoder_output_dim}")

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    print(f"Loading checkpoint: {cfg.decoder_checkpoint}")
    model = FFNDecoder(cfg.encoder_output_dim, n_classes,
                       cfg.hidden_dims, cfg.dropout).to(device)
    checkpoint = torch.load(cfg.decoder_checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    use_wandb = HAS_WANDB
    if use_wandb:
        try:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name="test-eval",
                config={
                    "checkpoint": cfg.decoder_checkpoint,
                    "threshold":  cfg.eval_threshold,
                    "test_size":  len(test_ds),
                },
            )
        except Exception:
            use_wandb = False

    m = evaluate(model, test_loader, device, threshold=cfg.eval_threshold)

    print(
        f"\nTest | threshold={cfg.eval_threshold} | "
        f"mAP={m['mAP']:.4f} | F1={m['F1_macro']:.4f} | "
        f"Precision={m['Precision']:.4f} | Recall={m['Recall']:.4f} | "
        f"Accuracy={m['Accuracy']:.4f} | MCC={m['MCC']:.4f}"
    )

    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in m.items()})
        for k, v in m.items():
            wandb.run.summary[f"test/{k}"] = v
        wandb.finish()

    return m


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"])
    parser.add_argument("--embedding", default=None, choices=list(EMBEDDING_TYPES),
                        help="Embedding kaynağı: raw_esm | seq_phase0 | seq_phase1 | concat_phase0 | concat_phase1")
    parser.add_argument("--raw_esm_h5", default=None, help="Override raw ESM embedding H5 path")
    parser.add_argument("--seq_phase0_h5", default=None, help="Override Phase0 fine-tuned embedding H5 path")
    parser.add_argument("--seq_phase1_h5", default=None, help="Override Phase1 fine-tuned embedding H5 path")
    parser.add_argument("--embeddings_h5", default=None, help="Override protein-level embedding H5 path directly (embedding_type resolution'ı ezer)")
    parser.add_argument("--goa_tsv", default=None, help="Override GOA TSV path")
    parser.add_argument("--split_json", default=None, help="Override split JSON path")
    parser.add_argument("--decoder_checkpoint", default=None, help="Override decoder checkpoint path")
    parser.add_argument("--go_vocab_json", default=None, help="Override GO vocab json path")
    parser.add_argument("--metrics_out", default=None, help="Optional path to dump metrics json")
    parser.add_argument("--cuda", default=None, help="Device override: örn 0, 1, cuda:0, mps, cpu")
    args = parser.parse_args()
    cfg = DecoderConfig(aspect=args.aspect, embedding_type=args.embedding or "raw_esm")
    if args.raw_esm_h5 is not None:
        cfg.raw_esm_h5 = args.raw_esm_h5
    if args.seq_phase0_h5 is not None:
        cfg.seq_phase0_h5 = args.seq_phase0_h5
    if args.seq_phase1_h5 is not None:
        cfg.seq_phase1_h5 = args.seq_phase1_h5
    cfg.resolve_embeddings_h5()
    if args.embeddings_h5 is not None:
        cfg.embeddings_h5 = args.embeddings_h5
    if args.goa_tsv is not None:
        cfg.goa_tsv = args.goa_tsv
    if args.split_json is not None:
        cfg.split_json = args.split_json
    if args.decoder_checkpoint is not None:
        cfg.decoder_checkpoint = args.decoder_checkpoint
    if args.go_vocab_json is not None:
        cfg.go_vocab_json = args.go_vocab_json
    if args.cuda is not None:
        cfg.cuda = args.cuda
    metrics = test_eval(cfg)
    if args.metrics_out is not None:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
