"""Frozen-encoder three-class Concat decoder, separate from the cascade."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase_3_variant_decoder.config import LABELS
from phase_3_variant_decoder.data.pair_dataset import iterate_batches
from phase_3_variant_decoder.input_controls import class_weights, classification_metrics, save_predictions

MODES = ("concat",)
FORMAT = "contvar_direct_three_class_pair_v1"
INPUTS = {
    "concat": "[z_WT; z_VAR; GO]",
}


class DirectPairDecoder(nn.Module):
    """Shared WT/VAR projection followed by joint Concat classification.

    Shared projection has no dropout: differences cannot be manufactured by
    independent dropout masks. Dropout acts only inside the joint classifier.
    """
    def __init__(self, mode, protein_dim, go_dim, projection_dim=128,
                 hidden_dims=(128, 64), dropout=.2, normalization=None):
        super().__init__()
        if mode not in MODES or min(protein_dim, go_dim, projection_dim) < 1:
            raise ValueError("Invalid mode or input/projection dimensions")
        if not hidden_dims or any(d < 1 for d in hidden_dims) or not 0 <= dropout < 1:
            raise ValueError("Invalid hidden dimensions or dropout")
        self.mode = mode
        self.spec = dict(mode=mode, protein_dim=protein_dim, go_dim=go_dim,
                         projection_dim=projection_dim, hidden_dims=list(hidden_dims), dropout=dropout)
        for name, dim, default in (("protein_mean", protein_dim, 0), ("protein_std", protein_dim, 1),
                                   ("go_mean", go_dim, 0), ("go_std", go_dim, 1)):
            value = normalization[name] if normalization is not None else torch.full((dim,), float(default))
            self.register_buffer(name, value.detach().clone())
        self.projection = nn.Sequential(nn.Linear(protein_dim, projection_dim),
                                        nn.LayerNorm(projection_dim), nn.SiLU())
        dim = 2 * projection_dim + go_dim
        layers = []
        for width in hidden_dims:
            layers.extend([nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)])
            dim = width
        self.classifier = nn.Sequential(*layers, nn.Linear(dim, len(LABELS)))

    def features(self, wt, var, go):
        if wt.ndim != 2 or wt.shape != var.shape or go.ndim != 2 or len(go) != len(wt):
            raise ValueError("Expected matching WT/VAR [B,D] and GO [B,G]")
        w = self.projection((wt - self.protein_mean) / self.protein_std)
        v = self.projection((var - self.protein_mean) / self.protein_std)
        g = (go - self.go_mean) / self.go_std
        return torch.cat([w, v, g], dim=-1)

    def forward(self, wt, var, go):
        return self.classifier(self.features(wt, var, go))


def batch_logits(model, tensors, view, indices):
    idx = indices.to(tensors.wt_feat.device)
    inputs = tensors.gather_with_go(view["wt_row"][idx], view["var_row"][idx], view["go_idx"][idx])
    return model(*inputs)


@torch.no_grad()
def evaluate(model, tensors, view, weights, batch_size):
    model.eval()
    probabilities, loss_sum, nll_sum = [], 0., 0.
    if not view["n"]:
        raise ValueError("Cannot evaluate an empty partition")
    for idx in iterate_batches(view["n"], batch_size, False):
        logits = batch_logits(model, tensors, view, idx)
        y = view["label_idx"][idx.to(logits.device)]
        ce = F.cross_entropy(logits, y, reduction="none")
        if not torch.isfinite(logits).all() or not torch.isfinite(ce).all():
            raise FloatingPointError("Non-finite three-class decoder output")
        loss_sum += float((ce * weights[y]).sum())
        nll_sum += float(ce.sum())
        probabilities.append(logits.softmax(-1).cpu().numpy())
    probs = np.concatenate(probabilities)
    y = view["label_idx"].cpu().numpy()
    metrics = classification_metrics(y, probs.argmax(1))
    metrics.update(eval_loss=loss_sum / len(y), nll=nll_sum / len(y))
    metrics.update({f"{label.lower()}_support": int((y == i).sum()) for i, label in enumerate(LABELS)})
    return metrics, probs


def load_decoder(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a direct-three-class pair checkpoint")
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Checkpoint label order differs from LOF/GOF/NEUTRAL")
    model = DirectPairDecoder(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def train_decoder(mode, cfg, tensors, views, normalization, root, *, on_epoch=None, quiet=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    device = tensors.wt_feat.device
    model = DirectPairDecoder(mode, tensors.input_dim, tensors.go_features.shape[1],
                              cfg.projection_dim, cfg.hidden_dims, cfg.dropout, normalization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    counts, weights = class_weights(views["train"]["label_idx"], cfg.balance_classes)
    history, best_f1, best_epoch, stale = [], -1., 0, 0
    checkpoint_path = root / "decoder.pt"
    go_vocab = dict(tensors.df[["matched_go_id", "go_idx"]].drop_duplicates().itertuples(index=False, name=None))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.
        for idx in iterate_batches(views["train"]["n"], cfg.batch_size, True, generator):
            y = views["train"]["label_idx"][idx.to(device)]
            ce = F.cross_entropy(batch_logits(model, tensors, views["train"], idx), y, reduction="none")
            loss = (ce * weights[y]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {mode} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
        train_metric, _ = evaluate(model, tensors, views["train"], weights, cfg.batch_size)
        val_metric, _ = evaluate(model, tensors, views["val"], weights, cfg.batch_size)
        row = dict(epoch=epoch, train_loss=total_loss / views["train"]["n"],
                   train_eval_loss=train_metric["eval_loss"], val_eval_loss=val_metric["eval_loss"],
                   train_macro_f1=train_metric["macro_f1"], val_macro_f1=val_metric["macro_f1"],
                   f1_gap=train_metric["macro_f1"] - val_metric["macro_f1"], lr=cfg.lr)
        history.append(row)
        if val_metric["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_metric["macro_f1"], epoch, 0
            torch.save(dict(format=FORMAT, model_spec=model.spec,
                            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            labels=list(LABELS), selection=dict(epoch=epoch, val_macro_f1=best_f1),
                            class_counts=counts.tolist(), class_weights=weights.tolist(),
                            decision="softmax argmax; no threshold", encoder_frozen=True,
                            go_vocab=go_vocab, go_vectors=tensors.go_features.detach().cpu(),
                            go_artifact_sha256=tensors.go_artifact_sha256), checkpoint_path)
        else:
            stale += 1
        if on_epoch:
            on_epoch(mode, row)
        if not quiet:
            print(f"[{mode} {epoch:03d}] train-F1={row['train_macro_f1']:.4f}; "
                  f"val-F1={row['val_macro_f1']:.4f}; loss={row['train_loss']:.4f}", flush=True)
        if stale >= cfg.early_stop_patience:
            break
    model, _ = load_decoder(checkpoint_path, device)
    result = dict(parameter_count=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                  epochs_run=len(history), class_counts=counts.tolist(), class_weights=weights.tolist(),
                  checkpoint=str(checkpoint_path), metrics={})
    for split in ("train", "val", "test"):
        metrics, probs = evaluate(model, tensors, views[split], weights, cfg.batch_size)
        result["metrics"][split] = metrics
        save_predictions(tensors, split, probs.argmax(1), root / f"{split}_predictions.csv", probs)
    (root / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    return result, history
