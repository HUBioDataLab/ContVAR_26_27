"""Training, checkpointing and evaluation for the two-stage scalar decoder.

Stage selection uses validation only. Stage 2 trains/validates on true effect
rows, not stage-1 predictions. Test is evaluated once, after both selections.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)

from phase_3_variant_decoder.cascade_model import CascadeDecoder, stage_targets
from phase_3_variant_decoder.cascade_objectives import cascade_loss_per_row
from phase_3_variant_decoder.config import LABELS, LABEL_TO_IDX, VariantDecoderConfig
from phase_3_variant_decoder.data.pair_dataset import iterate_batches

CHECKPOINT_FORMAT = "contvar_go_scalar_cascade_v1"


def binary_metrics(target, probability, threshold=0.5):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability)
    if not len(target):
        return {"n": 0, "macro_f1": None, "accuracy": None, "auroc": None, "auprc": None}
    pred = probability >= threshold
    both = len(np.unique(target)) == 2
    return {"n": len(target),
            "macro_f1": float(f1_score(target, pred, labels=[0, 1], average="macro", zero_division=0)),
            "accuracy": float(np.mean(target == pred)),
            "auroc": float(roc_auc_score(target, probability)) if both else None,
            "auprc": float(average_precision_score(target, probability)) if both else None,
            "confusion": confusion_matrix(target, pred, labels=[0, 1]).tolist()}


def select_effect_threshold(target, probability):
    """Validation-only macro-F1 grid; deterministic ties prefer 0.5."""
    grid = sorted(np.linspace(0.01, 0.99, 99).tolist() + [0.0, 1.0],
                  key=lambda t: (abs(t - 0.5), t))
    scores = [f1_score(target, probability >= t, labels=[0, 1],
                       average="macro", zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def batch_inputs(tensors, view, idx):
    idx = idx.to(tensors.wt_feat.device)
    return tensors.gather_with_go(view["wt_row"][idx], view["var_row"][idx], view["go_idx"][idx])


@torch.no_grad()
def stage_outputs(stage_model, tensors, view, indices=None, batch_size=4096, include_logits=False):
    stage_model.eval()
    if indices is None:
        indices = torch.arange(view["n"], device=tensors.wt_feat.device)
    collected = {k: [] for k in ("s_wt", "s_var", "delta", "probability")}
    if include_logits:
        collected["logit"] = []
    for pos in iterate_batches(len(indices), batch_size, shuffle=False):
        idx = indices[pos.to(indices.device)]
        out = stage_model(*batch_inputs(tensors, view, idx))
        for key in collected:
            collected[key].append(out[key].cpu().numpy())
    return {k: np.concatenate(v) if v else np.empty(0, dtype=np.float32)
            for k, v in collected.items()}


def _training_rows(view, stage):
    mask, target = stage_targets(view["label_idx"], stage)
    indices = torch.nonzero(mask, as_tuple=True)[0]
    if len(torch.unique(target)) != 2:
        names = "NEUTRAL and effect" if stage == "effect" else "LOF and GOF"
        raise ValueError(f"{stage} requires both {names} in train and validation splits")
    return indices, target


def fit_stage(model, stage, tensors, views, cfg, run=None, quiet=False,
              log_prefix="cascade", diagnostics=False):
    tr_idx, targets = _training_rows(views["train"], stage)
    va_idx, val_targets = _training_rows(views["val"], stage)
    y_val = val_targets.cpu().numpy().astype(np.int64)
    # Both class counts come exclusively from this stage's training subset.
    counts = torch.bincount(targets.long(), minlength=2).float()
    weights = len(targets) / (2 * counts) if cfg.cascade_balance_classes else torch.ones_like(counts)
    loss_kind = getattr(cfg, "cascade_loss", "bce")
    margin = getattr(cfg, "cascade_margin", 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed + (stage == "direction"))
    best_score, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum = 0.0
        for pos in iterate_batches(len(tr_idx), cfg.batch_size, shuffle=True, generator=generator):
            pos = pos.to(tr_idx.device)
            out = model(*batch_inputs(tensors, views["train"], tr_idx[pos]))
            target = targets[pos]
            per_row = cascade_loss_per_row(out["logit"], target, kind=loss_kind, margin=margin)
            loss = (per_row * weights[target.long()]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(pos)
        val_out = stage_outputs(model, tensors, views["val"], va_idx, cfg.batch_size, include_logits=diagnostics)
        p = val_out["probability"]
        if stage == "effect" and cfg.effect_threshold_strategy == "validation":
            threshold, score = select_effect_threshold(y_val, p)
        else:
            # Direction always uses 0.5 to preserve the meaning of WT-minus-
            # variant.  In learned_radius mode Effect also uses 0.5: its
            # trainable radius is then the actual |delta| decision boundary.
            threshold = 0.5
            score = binary_metrics(y_val, p)["macro_f1"]
        record = {"stage": stage, "epoch": epoch, "train_loss": loss_sum / len(tr_idx),
                  "val_macro_f1": score, "threshold": threshold}
        if diagnostics:
            train_out = stage_outputs(model, tensors, views["train"], tr_idx, cfg.batch_size, include_logits=True)
            def clean_loss(output, target):
                target = target.detach().cpu()
                loss = cascade_loss_per_row(torch.from_numpy(output["logit"]), target, kind=loss_kind, margin=margin)
                return float((loss * weights.detach().cpu()[target.long()]).mean())
            train_f1 = binary_metrics(targets.cpu().numpy(), train_out["probability"], threshold)["macro_f1"]
            record.update(train_eval_loss=clean_loss(train_out, targets), val_eval_loss=clean_loss(val_out, val_targets),
                          train_macro_f1=train_f1, f1_gap=train_f1 - score, lr=epoch_lr)
        history.append(record)
        if run is not None:
            run.log({f"{log_prefix}/{stage}/{k}": v
                     for k, v in record.items() if k != "stage"})
        if not quiet:
            print(f"[{stage} {epoch:03d}] {loss_kind} loss={record['train_loss']:.4f} val_macroF1={score:.4f} threshold={threshold:.2f}")
        if score > best_score:
            best_score, stale = score, 0
            best = {**record, "train_n": len(tr_idx), "val_n": len(va_idx),
                    "loss": loss_kind, "margin": margin,
                    "margin_space": "scaled_signed_logit",
                    "threshold_strategy": (
                        cfg.effect_threshold_strategy if stage == "effect" else "fixed_0.5"
                    ),
                    "train_class_counts_0_1": counts.cpu().int().tolist(),
                    "class_weights_0_1": weights.cpu().tolist()}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best, history


def evaluate_cascade(model, tensors, view, effect_threshold, batch_size=4096):
    """Hard cascade metrics, plus direction quality on true effect rows.

    The latter is an oracle-gated diagnostic, never reported as cascade accuracy.
    Probabilities from both models are also combined for ranking metrics; their
    argmax is intentionally NOT used as the hard cascade decision rule.
    """
    model.eval()
    effect = stage_outputs(model.effect_stage, tensors, view, batch_size=batch_size)
    direction = stage_outputs(model.direction_stage, tensors, view, batch_size=batch_size)
    p_effect, p_lof = effect["probability"], direction["probability"]
    routed = p_effect >= effect_threshold
    pred = np.full(view["n"], LABEL_TO_IDX["NEUTRAL"], dtype=np.int64)
    pred[routed] = np.where(p_lof[routed] >= 0.5, LABEL_TO_IDX["LOF"], LABEL_TO_IDX["GOF"])
    y = view["label_idx"].cpu().numpy()
    actual_effect = y != LABEL_TO_IDX["NEUTRAL"]
    probabilities = np.column_stack((p_effect * p_lof, p_effect * (1 - p_lof), 1 - p_effect))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    m = {"n": len(y), "macro_f1": float(f1.mean()), "accuracy": float(np.mean(pred == y)),
         "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
         "routed_n": int(routed.sum()),
         "effect": binary_metrics(actual_effect, p_effect, effect_threshold),
         "direction_true_effect": binary_metrics((y[actual_effect] == 0), p_lof[actual_effect]),
         "effect_missed_n": int((actual_effect & ~routed).sum()),
         "neutral_routed_n": int((~actual_effect & routed).sum())}
    for i, label in enumerate(LABELS):
        target = y == i
        m[f"{label.lower()}_precision"] = float(precision[i])
        m[f"{label.lower()}_recall"] = float(recall[i])
        m[f"{label.lower()}_f1"] = float(f1[i])
        m[f"{label.lower()}_auprc"] = float(average_precision_score(target, probabilities[:, i])) if target.any() else None
    for name in ("seen_in_train", "is_alanine"):
        if name not in view:
            continue
        mask = np.asarray(view[name], dtype=bool)
        for key, selected in ((name, mask), (f"not_{name}", ~mask)):
            m[key] = {"n": int(selected.sum()), "macro_f1":
                      float(f1_score(y[selected], pred[selected], labels=[0, 1, 2], average="macro", zero_division=0))
                      if selected.any() else None}
    scores = {"prediction": pred, "routed": routed, "probabilities": probabilities,
              "effect": effect, "direction": direction}
    return m, scores


def fit_cascade(cfg, tensors, views, go_vocab, device, out_dir=None, run=None,
                quiet=False, log_prefix="cascade", evaluate_test=True, diagnostics=False):
    if tensors.go_features is None:
        raise ValueError("Cascade requires attached GO embeddings")
    for split in (("train", "val", "test") if evaluate_test else ("train", "val")):
        if views[split]["n"] == 0:
            raise ValueError(f"Empty {split} split")
    # Validate both stages before spending time training either one.
    for stage in ("effect", "direction"):
        for split in ("train", "val"):
            _training_rows(views[split], stage)
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.early_stop_patience < 1:
        raise ValueError("epochs, batch_size and early_stop_patience must be positive")
    torch.manual_seed(cfg.seed)
    model = CascadeDecoder(tensors.input_dim, tensors.go_features.shape[1], cfg.hidden_dims, cfg.dropout).to(device)
    selections, history = {}, []
    for stage in ("effect", "direction"):
        selections[stage], records = fit_stage(
            getattr(model, f"{stage}_stage"), stage, tensors, views, cfg, run, quiet,
            log_prefix, diagnostics=diagnostics,
        )
        history.extend(records)
    threshold = selections["effect"]["threshold"]
    splits = (["train"] if diagnostics else []) + ["val"] + (["test"] if evaluate_test else [])
    metrics = {s: evaluate_cascade(model, tensors, views[s], threshold, cfg.batch_size)[0] for s in splits}
    best = {"architecture": CHECKPOINT_FORMAT, "macro_f1": metrics["val"]["macro_f1"],
            "loss": cfg.cascade_loss, "cascade_margin": cfg.cascade_margin,
            "score_interpretation": "Uncalibrated sigmoid decision scores; hinge is not a probability loss",
            "effect_threshold": threshold, "direction_threshold": 0.5,
            "selection": selections, "metrics": metrics}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Unique run directory keeps old hinge checkpoints and all cascade trials.
        import tempfile
        destination = Path(tempfile.mkdtemp(prefix=f"cascade_{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}_", dir=out))
        checkpoint = destination / "decoder.pt"
        torch.save({"format": CHECKPOINT_FORMAT, "model_spec": model.spec,
                    "loss_spec": {"kind": cfg.cascade_loss, "margin": cfg.cascade_margin,
                                  "margin_space": "scaled_signed_logit"},
                    "model_state_dict": model.state_dict(), "config": vars(cfg),
                    "effect_threshold": threshold, "direction_threshold": 0.5,
                    "go_vocab": go_vocab, "go_vectors": tensors.go_features.detach().cpu(),
                    "go_artifact_sha256": tensors.go_artifact_sha256,
                    "go_metadata": tensors.go_metadata,
                    "selection": selections}, checkpoint)
        best["checkpoint"] = str(checkpoint)
        (destination / "metrics.json").write_text(json.dumps(best, indent=2, allow_nan=False) + "\n")
        (destination / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        if not quiet:
            print(f"[DONE] cascade checkpoint -> {checkpoint}")
    if run is not None:
        run.summary.update({f"{log_prefix}/{s}_macro_f1": metrics[s]["macro_f1"] for s in splits})
    if not quiet:
        print("[DONE] cascade " + "; ".join(f"{s} macro-F1={metrics[s]['macro_f1']:.4f}" for s in splits))
    return best


def load_cascade(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a GO scalar cascade checkpoint")
    # Old serialized configs predate the loss selector and always used BCE.
    ckpt.setdefault("config", {}).setdefault("cascade_loss", "bce")
    ckpt["config"].setdefault("cascade_margin", 1.0)
    model = CascadeDecoder(**ckpt["model_spec"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def attach_checkpoint_go(tensors, go_vocab, ckpt):
    saved = ckpt["go_vocab"]
    missing = sorted(set(go_vocab) - set(saved))
    if missing:
        raise ValueError(f"Checkpoint has no GO vector for {len(missing)} term(s): {missing[:10]}")
    ids = [saved[g] for g in sorted(go_vocab, key=go_vocab.get)]
    tensors.go_features = ckpt["go_vectors"][ids].to(tensors.wt_feat.device)
    tensors.go_artifact_sha256 = ckpt["go_artifact_sha256"]
    tensors.go_metadata = ckpt.get("go_metadata")


def dump_scores(tensors, split, scores, path):
    frame = tensors.df.loc[tensors.df["split"] == split].copy().reset_index(drop=True)
    frame["pred"] = [LABELS[int(i)] for i in scores["prediction"]]
    frame["routed_to_direction"] = scores["routed"]
    frame["correct"] = frame["label"] == frame["pred"]
    for stage in ("effect", "direction"):
        for key, value in scores[stage].items():
            # Evaluation computes direction on all rows for diagnostic metrics;
            # the row dump only exposes actual routed direction predictions.
            frame[f"{stage}_{key}"] = value if stage == "effect" else np.where(scores["routed"], value, np.nan)
    for i, label in enumerate(LABELS):
        frame[f"p_{label.lower()}"] = scores["probabilities"][:, i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def evaluate_checkpoint(args):
    from phase_2_decoder.config import pick_device
    from phase_3_variant_decoder.data.variant_dataset import prepare
    from phase_3_variant_decoder.data.pair_dataset import build_pair_tensors

    device = torch.device(pick_device(args.cuda))
    model, ckpt = load_cascade(args.checkpoint, device)
    cfg = VariantDecoderConfig()
    for key, value in ckpt["config"].items():
        setattr(cfg, key, value)
    for key in ("global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "dataset_dir", "sequences_json"):
        if getattr(args, key, None) is not None:
            setattr(cfg, key, getattr(args, key))
    # GO vectors are self-contained in the checkpoint; original file may move.
    cfg.architecture = "legacy"
    cfg.validate()
    df, go_vocab, report = prepare(cfg)
    print(report.render())
    tensors = build_pair_tensors(df, cfg).to(device)
    if tensors.input_dim != model.spec["protein_dim"]:
        raise ValueError("Protein input dimension differs from checkpoint")
    attach_checkpoint_go(tensors, go_vocab, ckpt)
    threshold = ckpt["effect_threshold"]
    if args.research_thresholds:
        view = tensors.split_view("val")
        _training_rows(view, "effect")
        p = stage_outputs(model.effect_stage, tensors, view)["probability"]
        threshold, _ = select_effect_threshold(view["label_idx"].cpu().numpy() != 2, p)
    out = {"architecture": CHECKPOINT_FORMAT, "effect_threshold": threshold, "direction_threshold": 0.5}
    for split in ("val", "test"):
        view = tensors.split_view(split)
        if not view["n"]:
            raise ValueError(f"Empty {split} split")
        metrics, scores = evaluate_cascade(model, tensors, view, threshold)
        out[split] = metrics
        print(f"[{split}] cascade macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
        if args.dump_scores:
            path = Path(args.dump_scores)
            dump_scores(tensors, split, scores, path.with_name(f"{path.stem}_{split}{path.suffix or '.csv'}"))
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return out
