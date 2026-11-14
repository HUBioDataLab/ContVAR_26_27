"""
Decoder eğitimi.

Kullanım:
    python -m Sekans_decoder.train
    python -m Sekans_decoder.train --aspect P
"""

import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

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
    from .evaluate import (
        evaluate, get_probs_labels, compute_metrics_at_threshold,
        find_best_f1_threshold, FINAL_METRIC_KEYS,
    )
    from .benchmark_eval import evaluate_benchmark_in_memory, BM_METRIC_KEYS
except ImportError:
    from config import DecoderConfig, EMBEDDING_TYPES, pick_device
    from data.parse_goa import parse_goa_tsv, build_go_vocab
    from data.dataset import GOAnnotationDataset
    from models.ffn_decoder import FFNDecoder
    from evaluate import (
        evaluate, get_probs_labels, compute_metrics_at_threshold,
        find_best_f1_threshold, FINAL_METRIC_KEYS,
    )
    from benchmark_eval import evaluate_benchmark_in_memory, BM_METRIC_KEYS


# ── Class weights ────────────────────────────────────────────────────────────

def compute_pos_weights(dataset: GOAnnotationDataset,
                        cfg: DecoderConfig) -> torch.Tensor:
    """
    Her GO terimi için pos_weight hesapla.
        pos_weight[i] = n_negative[i] / n_positive[i]

    Nadir GO terimleri için loss'u yüksek tutar → model onları da öğrenir.
    NULL_FUNCTION'a ekstra çarpan uygulanır.
    """
    labels = dataset.labels                  # [N, N_go]
    pos = labels.sum(dim=0)                  # [N_go]
    neg = len(labels) - pos
    pos_weight = neg / (pos + 1e-8)
    pos_weight[-1] *= cfg.null_function_weight
    pos_weight = pos_weight.clamp(max=cfg.pos_weight_clamp)
    return pos_weight


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -float("inf")
        self.counter    = 0

    def step(self, score: float) -> bool:
        """Returns True → eğitimi durdur."""
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Ana eğitim fonksiyonu ────────────────────────────────────────────────────

def train(cfg: DecoderConfig = None):
    if cfg is None:
        cfg = DecoderConfig()
    cfg.validate_paths()
    device = torch.device(pick_device(cfg.cuda))
    print(f"[INFO] Device: {device}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # WandB
    use_wandb = HAS_WANDB and getattr(cfg, "wandb_enabled", True)
    wandb_tags = getattr(cfg, "wandb_tags", None) or None
    wandb_extra_config = getattr(cfg, "wandb_extra_config", None) or {}
    from phase_3_variant_decoder.presentation_terms import ENCODER_LABELS, ENCODER_DEFINITIONS
    encoder_reporting = {
        "task": "Protein GO annotation (multi-label); distinct from variant LOF/GOF/NEUTRAL classification",
        "encoder_label": ENCODER_LABELS[cfg.embedding_type],
        "encoder_training_history": ENCODER_DEFINITIONS[cfg.embedding_type],
        "encoder_history_basis": "Repository pipeline definition; verify external artifact provenance.",
    }
    if use_wandb and cfg.wandb_api_key is not None:
        wandb.login(key=cfg.wandb_api_key)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=wandb_tags,
            config={
                "aspect":               cfg.aspect,
                "embedding_type":       cfg.embedding_type,
                "encoder_output_dim":   cfg.encoder_output_dim,
                "hidden_dims":          cfg.hidden_dims,
                "dropout":              cfg.dropout,
                "min_go_freq":          cfg.min_go_freq,
                "null_function_weight": cfg.null_function_weight,
                "pos_weight_clamp":     cfg.pos_weight_clamp,
                "lr":                   cfg.lr,
                "weight_decay":         cfg.weight_decay,
                "batch_size":           cfg.batch_size,
                "epochs":               cfg.epochs,
                "early_stop_patience":  cfg.early_stop_patience,
                **wandb_extra_config,
                **encoder_reporting,
            },
        )
    elif use_wandb:
        # API key yoksa env'den dene
        try:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                tags=wandb_tags,
                config={
                    "aspect": cfg.aspect,
                    "embedding_type": cfg.embedding_type,
                    "encoder_output_dim": cfg.encoder_output_dim,
                    **wandb_extra_config,
                    **encoder_reporting,
                },
            )
        except Exception:
            print("[WARN] WandB init failed, continuing without logging.")
            use_wandb = False

    # 1. GO annotation parse — HER ZAMAN ham (propagate edilmemiş) anotasyonlarla.
    # Ancestor propagation burada değil, sadece val/test split etiketleri için
    # aşağıdaki adım 2.5'te (GOAnnotationDataset'e ancestor_map olarak) uygulanır.
    print(f"Parsing GO annotations (aspect={cfg.aspect})...")
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect, obo_path=None,
                               use_qualifier_filter=cfg.use_qualifier_filter)

    # 2. Vocabulary
    print("Building vocabulary...")
    go_vocab = build_go_vocab(annotations, min_freq=cfg.min_go_freq,
                              save_path=cfg.go_vocab_json)
    n_classes = len(go_vocab)
    print(f"GO vocab size: {n_classes}")

    # 2.5. Val/test ground-truth propagation eşleme tablosu. SADECE val ve test
    # split etiketlerine uygulanır (GOAnnotationDataset içinde split'e göre
    # gate'lenir) — train split HİÇBİR ZAMAN propagate edilmiş etiketle eğitilmez.
    ancestor_map = None
    if cfg.use_go_propagation and os.path.exists(cfg.go_obo_path):
        try:
            from phase_2_decoder.data.parse_goa import build_ancestor_map
        except ImportError:
            from data.parse_goa import build_ancestor_map
        vocab_terms = set(go_vocab.keys()) - {"NULL_FUNCTION"}
        ancestor_map = build_ancestor_map(cfg.go_obo_path, vocab_terms)
    else:
        reason = "devre dışı (use_go_propagation=False)" if not cfg.use_go_propagation else f"bulunamadı ({cfg.go_obo_path})"
        print(f"[WARN] go.obo {reason}, val/test ancestor propagation devre dışı.")

    # 3. Datasets
    secondary_h5 = getattr(cfg, "embeddings_h5_secondary", None)
    train_ds = GOAnnotationDataset(cfg.embeddings_h5, annotations, go_vocab,
                                   split="train",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   secondary_h5_path=secondary_h5)
    val_ds   = GOAnnotationDataset(cfg.embeddings_h5, annotations, go_vocab,
                                   split="val",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   secondary_h5_path=secondary_h5,
                                   ancestor_map=ancestor_map)
    test_ds  = GOAnnotationDataset(cfg.embeddings_h5, annotations, go_vocab,
                                   split="test",
                                   uniref_tsv=cfg.uniref_tsv,
                                   split_json=cfg.split_json,
                                   secondary_h5_path=secondary_h5,
                                   ancestor_map=ancestor_map)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Train dataset bos. Embedding H5 ile GO annotation/split kesismiyor."
        )

    inferred_input_dim = int(train_ds.embs.shape[1])
    cfg.encoder_output_dim = inferred_input_dim
    print(f"[INFO] Inferred decoder input dim: {cfg.encoder_output_dim}")

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin_memory) if len(test_ds) > 0 else None

    # 4. GO hierarchy propagation matrisi (bir kez oluşturulur, tüm eval'lerde kullanılır)
    prop_mat = None
    if cfg.use_go_propagation and os.path.exists(cfg.go_obo_path):
        try:
            from phase_2_decoder.data.go_hierarchy import GOHierarchy
        except ImportError:
            from data.go_hierarchy import GOHierarchy
        print(f"Building GO hierarchy propagation matrix...")
        prop_mat = GOHierarchy(cfg.go_obo_path).build_propagation_matrix(go_vocab)
    else:
        reason = "devre dışı (use_go_propagation=False)" if not cfg.use_go_propagation else f"bulunamadı ({cfg.go_obo_path})"
        print(f"[WARN] go.obo {reason}, propagation devre dışı.")

    # 5. Class weights
    print("Computing pos_weight...")
    pos_weight = compute_pos_weights(train_ds, cfg)

    # 5. Model, loss, optimizer, scheduler
    decoder   = FFNDecoder(cfg.encoder_output_dim, n_classes,
                           cfg.hidden_dims, cfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(decoder.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=cfg.epochs)
    stopper   = EarlyStopper(patience=cfg.early_stop_patience)

    best_map = 0.0
    best_epoch = 0
    best_val_metrics = None
    best_test_metrics = None
    best_bm_metrics = {}
    best_final_metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        decoder.train()
        train_loss = 0.0

        for emb, label in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            emb   = emb.to(device)
            label = label.to(device)

            logits = decoder(emb)
            loss   = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ──
        metrics = evaluate(decoder, val_loader, device, criterion=criterion,
                           threshold=cfg.eval_threshold, propagation_mat=prop_mat)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Test (gözlem amaçlı, model seçiminde kullanılmaz) ──
        test_metrics = evaluate(decoder, test_loader, device,
                                threshold=cfg.eval_threshold,
                                propagation_mat=prop_mat) if test_loader is not None else None

        # ── Benchmark ──
        bm_metrics = {}
        if (
            os.path.exists(cfg.benchmark_tsv)
            and cfg.variant_h5 is not None
            and os.path.exists(cfg.variant_h5)
        ):
            bm_metrics = evaluate_benchmark_in_memory(
                decoder=decoder,
                go_vocab=go_vocab,
                benchmark_tsv=cfg.benchmark_tsv,
                embeddings_h5=cfg.embeddings_h5,
                split_json=cfg.split_json,
                uniref_tsv=cfg.uniref_tsv,
                aspect=cfg.aspect,
                device=device,
                variant_h5=cfg.variant_h5,
                variant_h5_secondary=getattr(cfg, "variant_h5_secondary", None),
            )

        # ── Final (F1-eşiği optimize edilmiş) WT/GO-terim metrikleri ──
        # val üzerinde F1_macro'yu maksimize eden eşik aranır; test AYNI
        # eşikle değerlendirilir (test kendi eşiğini kendi seçmez). propagation
        # açıkken (prop_mat != None) tahminler de val_*/test_* ile aynı şekilde
        # propagate edilir — böylece final_* skorları, ham tahmin vs propagate
        # edilmiş ground truth gibi asimetrik bir karşılaştırma olmaz.
        final_metrics = {}
        if test_loader is not None:
            val_probs, val_labels = get_probs_labels(decoder, val_loader, device)
            best_final_thr = find_best_f1_threshold(val_probs, val_labels, propagation_mat=prop_mat)
            final_metrics["val"] = compute_metrics_at_threshold(val_probs, val_labels, best_final_thr, propagation_mat=prop_mat)
            test_probs, test_labels = get_probs_labels(decoder, test_loader, device)
            final_metrics["test"] = compute_metrics_at_threshold(test_probs, test_labels, best_final_thr, propagation_mat=prop_mat)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_mAP={metrics['mAP']:.4f} | "
            f"val_F1={metrics['F1_macro']:.4f} | "
            f"val_MCC={metrics['MCC']:.4f} | "
            f"LR={current_lr:.2e}"
        )
        if test_metrics is not None:
            print(
                f"           "
                f"test_mAP={test_metrics['mAP']:.4f} | "
                f"test_F1={test_metrics['F1_macro']:.4f} | "
                f"test_MCC={test_metrics['MCC']:.4f}"
            )
        if bm_metrics:
            for split_name, bm in bm_metrics.items():
                print(
                    f"           bm_{split_name}: "
                    f"AUROC={bm['auroc']:.4f} | "
                    f"AUPRC={bm['auprc']:.4f} | "
                    f"F1={bm['f1']:.4f} | "
                    f"Acc={bm['overall_acc']:.4f} | "
                    f"LOF_Rec={bm['lof_recall']:.4f} | "
                    f"GOF_Rec={bm['gof_recall']:.4f} "
                    f"(n={bm['n_total']}, LOF={bm['n_lof']}, GOF={bm['n_gof']})"
                )
        if final_metrics:
            for split_name, fm in final_metrics.items():
                print(
                    f"           final_{split_name}: "
                    f"F1={fm['F1_macro']:.4f} | Precision={fm['Precision']:.4f} | "
                    f"Recall={fm['Recall']:.4f} | MCC={fm['MCC']:.4f} "
                    f"(threshold={fm['threshold']:.2f})"
                )

        if use_wandb:
            log_dict = {
                "epoch":          epoch,
                "train/loss":     train_loss,
                "val/loss":       metrics["val_loss"],
                "val/mAP":        metrics["mAP"],
                "val/F1_macro":   metrics["F1_macro"],
                "val/Accuracy":   metrics["Accuracy"],
                "val/Precision":  metrics["Precision"],
                "val/Recall":     metrics["Recall"],
                "val/MCC":        metrics["MCC"],
                "val/TP":         metrics["TP"],
                "val/TN":         metrics["TN"],
                "val/FP":         metrics["FP"],
                "val/FN":         metrics["FN"],
                "train/lr":       current_lr,
            }
            if test_metrics is not None:
                log_dict.update({
                    "test/mAP":        test_metrics["mAP"],
                    "test/F1_macro":   test_metrics["F1_macro"],
                    "test/Accuracy":   test_metrics["Accuracy"],
                    "test/Precision":  test_metrics["Precision"],
                    "test/Recall":     test_metrics["Recall"],
                    "test/MCC":        test_metrics["MCC"],
                })
            for split_name, bm in bm_metrics.items():
                log_dict.update({
                    f"benchmark_{split_name}/{key}": bm[key] for key in BM_METRIC_KEYS
                })
                log_dict.update({
                    f"benchmark_{split_name}/tp": bm["tp"],
                    f"benchmark_{split_name}/tn": bm["tn"],
                    f"benchmark_{split_name}/fp": bm["fp"],
                    f"benchmark_{split_name}/fn": bm["fn"],
                })
            for split_name, fm in final_metrics.items():
                log_dict.update({
                    f"final_{split_name}/{key}": fm[key] for key in FINAL_METRIC_KEYS
                })
            wandb.log(log_dict)

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            best_epoch = epoch
            best_val_metrics = metrics
            best_test_metrics = test_metrics
            best_bm_metrics = bm_metrics
            best_final_metrics = final_metrics
            torch.save(
                {
                    "model_state_dict": decoder.state_dict(),
                    "input_dim": cfg.encoder_output_dim,
                    "n_classes": n_classes,
                    "hidden_dims": cfg.hidden_dims,
                    "dropout": cfg.dropout,
                },
                cfg.decoder_checkpoint,
            )
            print(f"  → saved (best mAP={best_map:.4f})")
            if use_wandb:
                wandb.run.summary["best_mAP"]       = best_map
                wandb.run.summary["best_epoch"]      = epoch
                wandb.run.summary["best_F1_macro"]   = metrics["F1_macro"]
                wandb.run.summary["best_MCC"]        = metrics["MCC"]

        if stopper.step(metrics["mAP"]):
            print(f"Early stopping triggered (epoch={epoch})")
            break

    if use_wandb:
        wandb.finish()

    if getattr(cfg, "_return_metrics", False):
        best_val_metrics = best_val_metrics or {}
        best_test_metrics = best_test_metrics or {}
        return {
            "aspect": cfg.aspect,
            "embedding_type": getattr(cfg, "embedding_type", "raw_esm"),
            "hyperparams": {
                "lr": cfg.lr,
                "dropout": cfg.dropout,
                "hidden_dims": cfg.hidden_dims,
                "weight_decay": cfg.weight_decay,
                "pos_weight_clamp": cfg.pos_weight_clamp,
                "epochs": cfg.epochs,
                "early_stop_patience": cfg.early_stop_patience,
            },
            "best_epoch": best_epoch,
            "best_val_map": best_map,
            "val": {
                "mAP": best_val_metrics.get("mAP", 0.0),
                "F1_macro": best_val_metrics.get("F1_macro", 0.0),
                "MCC": best_val_metrics.get("MCC", 0.0),
                "threshold": cfg.eval_threshold,
            },
            "test": {
                "mAP": best_test_metrics.get("mAP", 0.0),
                "F1_macro": best_test_metrics.get("F1_macro", 0.0),
                "MCC": best_test_metrics.get("MCC", 0.0),
                "threshold": cfg.eval_threshold,
            },
            "checkpoint": cfg.decoder_checkpoint,
            "benchmark": best_bm_metrics,
            "final_val": best_final_metrics.get("val", {}),
            "final_test": best_final_metrics.get("test", {}),
        }
    return best_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"],
                        help="GO aspect: F=MF, P=BP, C=CC")
    parser.add_argument("--embedding", default=None, choices=list(EMBEDDING_TYPES),
                        help="Embedding kaynağı: raw_esm | seq_phase0 | seq_phase1 | concat_phase0 | concat_phase1")
    parser.add_argument("--raw_esm_h5", default=None, help="Override raw ESM embedding H5 path")
    parser.add_argument("--seq_phase0_h5", default=None, help="Override Phase0 fine-tuned embedding H5 path")
    parser.add_argument("--seq_phase1_h5", default=None, help="Override Phase1 fine-tuned embedding H5 path")
    parser.add_argument("--embeddings_h5", default=None, help="Override protein-level embedding H5 path directly (embedding_type resolution'ı ezer)")
    parser.add_argument("--goa_tsv", default=None, help="Override GOA TSV path")
    parser.add_argument("--split_json", default=None, help="Override split JSON path")
    parser.add_argument("--decoder_checkpoint", default=None, help="Override decoder checkpoint output path")
    parser.add_argument("--go_vocab_json", default=None, help="Override GO vocab json path")
    parser.add_argument("--cuda", default=None, help="Device override: örn 0, 1, cuda:0, mps, cpu")
    parser.add_argument("--wandb_run_name", default=None, help="WandB run name override")
    parser.add_argument("--wandb_project", default=None, help="WandB project name override")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging entirely")
    parser.add_argument("--epochs", type=int, default=None, help="Training epoch sayısı override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout override")
    parser.add_argument("--hidden_dims", default=None,
                         help="Hidden layer sizes override, dash or comma separated, örn 1024-512 veya 1024,512")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay override")
    parser.add_argument("--pos_weight_clamp", type=float, default=None, help="Pos-weight clamp override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--no_propagation", action="store_true",
                         help="GO ancestor propagation'ı kapat (DecoderConfig varsayılanı True'dur; "
                              "grid_search.py --propagation geçilmediği sürece False kullanır)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Early stopping patience override")
    parser.add_argument("--variant_h5", default=None, help="Variant embeddings H5, keyed by {protein_id}_{snp}; enables benchmark eval during training")
    parser.add_argument("--variant_h5_secondary", default=None, help="concat_phase0/concat_phase1 icin ikinci varyant embedding H5'i")
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
    if args.wandb_run_name is not None:
        cfg.wandb_run_name = args.wandb_run_name
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.hidden_dims is not None:
        cfg.hidden_dims = [int(x) for x in args.hidden_dims.replace(",", "-").split("-") if x]
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.pos_weight_clamp is not None:
        cfg.pos_weight_clamp = args.pos_weight_clamp
    if args.no_propagation:
        cfg.use_go_propagation = False
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.variant_h5 is not None:
        cfg.variant_h5 = args.variant_h5
    if args.variant_h5_secondary is not None:
        cfg.variant_h5_secondary = args.variant_h5_secondary
    train(cfg)
