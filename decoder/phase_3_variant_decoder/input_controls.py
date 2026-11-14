"""Frozen-feature input controls; independent of the cascade training code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase_3_variant_decoder.config import LABELS
from phase_3_variant_decoder.data.pair_dataset import iterate_batches
from phase_3_variant_decoder.metrics import metrics_from_confusion


METHODS = {
    "go_lookup": "GO-only lookup",
    "go_only": "GO-only MLP",
    "wt_go": "WT + GO MLP",
    "mutation_go": "Mutation + GO MLP",
    "pair": "Pair-MLP",
}
LEARNED = ("go_only", "wt_go", "mutation_go", "pair")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MUTATION_DIM = 2 * (len(AMINO_ACIDS) + 1) + 1
INPUTS = {
    "go_lookup": "GO ID; train-label majority per GO; unseen GO uses global train majority",
    "go_only": "GO embedding only",
    "wt_go": "WT embedding + GO embedding; no mutant embedding",
    "mutation_go": "WT/VAR amino-acid one-hot + position / max_seq_len + GO embedding; no protein embeddings",
    "pair": "WT + VAR + (VAR - WT) + abs(VAR - WT) + GO embedding",
}
EXPLANATIONS_TR = {
    "go_lookup": "GO başına train'deki en sık etiketi seçer. Neural training yoktur.",
    "go_only": "Yalnız GO embeddinginden etiketi öğrenir; protein veya mutasyon girdisi almaz.",
    "wt_go": "GO'ya WT bağlamını ekler. Local/both modunda mutasyon pozisyonunun WT temsili de vardır.",
    "mutation_go": "Yalnız aminoasit değişimi, pozisyon ve GO kullanır; basit mutasyon ipuçlarını kontrol eder.",
    "pair": "WT ve mutantı farklarıyla birlikte karşılaştırır. WT + GO'ya göre kazanç, ek girdilerin faydasını sınar.",
}
CHECKPOINT_FORMAT = "contvar_input_control_mlp_v1"


def attach_mutation_features(tensors, views, max_seq_len):
    """Label-independent mutation identity/position, not an encoder output."""
    lookup = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for split, view in views.items():
        frame = tensors.df.loc[tensors.df.split == split]
        wt = [lookup.get(aa, len(AMINO_ACIDS)) for aa in frame.wt_aa]
        var = [lookup.get(aa, len(AMINO_ACIDS)) for aa in frame.mut_aa]
        identity = np.eye(len(AMINO_ACIDS) + 1, dtype=np.float32)
        features = np.concatenate([identity[wt], identity[var], frame.pos.to_numpy()[:, None] / max_seq_len], axis=1)
        view["mutation_features"] = torch.as_tensor(features, dtype=torch.float32, device=tensors.wt_feat.device)


def train_moments(features, indices, chunk_size=2048):
    """Train-unique feature rows, chunked on device; no validation/test fitting."""
    indices = torch.unique(indices)
    if not len(indices):
        raise ValueError("Cannot fit normalization on an empty training partition")
    total = torch.zeros(features.shape[1], dtype=torch.float64, device=features.device)
    squares = torch.zeros_like(total)
    for idx in indices.split(chunk_size):
        batch = features[idx].double()
        if not torch.isfinite(batch).all():
            raise ValueError("Non-finite training embeddings")
        total += batch.sum(0)
        squares += batch.square().sum(0)
    mean = total / len(indices)
    std = (squares / len(indices) - mean.square()).clamp_min(0).sqrt()
    # Constant axes retain unit scale; held-out values must not divide by zero.
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


def fit_normalization(tensors, train):
    # WT-only statistics ensure the WT control's preprocessing sees no VAR data.
    pm, ps = train_moments(tensors.wt_feat, train["wt_row"])
    gm, gs = train_moments(tensors.go_features, train["go_idx"])
    return dict(protein_mean=pm, protein_std=ps, go_mean=gm, go_std=gs)


class InputControlMLP(nn.Module):
    """Three logits in LABELS order. Restricted arms never read excluded inputs."""

    def __init__(self, mode, protein_dim, go_dim, hidden_dims, dropout, normalization=None):
        super().__init__()
        if mode not in LEARNED:
            raise ValueError(f"Unknown input control: {mode}")
        self.spec = dict(mode=mode, protein_dim=protein_dim, go_dim=go_dim,
                         hidden_dims=list(hidden_dims), dropout=dropout)
        self.mode = mode
        for name, dim, default in (("protein_mean", protein_dim, 0), ("protein_std", protein_dim, 1),
                                   ("go_mean", go_dim, 0), ("go_std", go_dim, 1)):
            value = normalization[name] if normalization is not None else torch.full((dim,), float(default))
            self.register_buffer(name, value.detach().clone())
        dim = go_dim + {"go_only": 0, "wt_go": protein_dim, "mutation_go": MUTATION_DIM, "pair": 4 * protein_dim}[mode]
        layers = []
        for width in hidden_dims:
            layers.extend([nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)])
            dim = width
        layers.append(nn.Linear(dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, go, wt=None, var=None, mutation=None):
        g = (go - self.go_mean) / self.go_std
        if self.mode == "go_only":
            x = g
        elif self.mode == "mutation_go":
            x = torch.cat([mutation, g], -1)
        else:
            w = (wt - self.protein_mean) / self.protein_std
            if self.mode == "wt_go":
                x = torch.cat([w, g], -1)
            else:
                v = (var - self.protein_mean) / self.protein_std
                delta = v - w
                x = torch.cat([w, v, delta, delta.abs(), g], -1)
        return self.network(x)


def batch_logits(model, tensors, view, indices):
    idx = indices.to(tensors.wt_feat.device)
    go = tensors.go_features[view["go_idx"][idx]]
    wt = tensors.wt_feat[view["wt_row"][idx]] if model.mode in ("wt_go", "pair") else None
    var = tensors.var_feat[view["var_row"][idx]] if model.mode == "pair" else None
    mutation = view["mutation_features"][idx] if model.mode == "mutation_go" else None
    return model(go, wt, var, mutation)


@torch.no_grad()
def predict(model, tensors, view, batch_size):
    model.eval()
    values = []
    for idx in iterate_batches(view["n"], batch_size, False):
        logits = batch_logits(model, tensors, view, idx)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model output; inspect input embeddings and normalization")
        values.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(values)


def classification_metrics(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty aligned labels and predictions")
    if not np.isin(y, range(3)).all() or not np.isin(pred, range(3)).all():
        raise ValueError("Expected LOF / GOF / NEUTRAL indices")
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    return dict(n=len(y), confusion=cm.tolist(), **metrics_from_confusion(cm))


@torch.no_grad()
def evaluate_objective(model, tensors, view, weights, batch_size):
    """Clean train/val loss and F1 with dropout disabled and identical weighting."""
    model.eval()
    total, predictions = 0., []
    for idx in iterate_batches(view["n"], batch_size, False):
        logits = batch_logits(model, tensors, view, idx)
        y = view["label_idx"][idx.to(logits.device)]
        loss = F.cross_entropy(logits, y, reduction="none") * weights[y]
        if not torch.isfinite(loss).all():
            raise ValueError("Non-finite evaluation objective")
        total += float(loss.sum())
        predictions.append(logits.argmax(-1).cpu().numpy())
    metrics = classification_metrics(view["label_idx"].cpu().numpy(), np.concatenate(predictions))
    return {**metrics, "eval_loss": total / view["n"]}


def class_weights(labels, balanced=True):
    counts = torch.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("Each of LOF, GOF and NEUTRAL must occur in training")
    weights = len(labels) / (3 * counts.float()) if balanced else torch.ones_like(counts, dtype=torch.float32)
    return counts, weights


def save_predictions(tensors, split, pred, path, probabilities=None):
    fields = ["protein_id", "snp", "matched_go_id", "label", "seen_in_train"]
    frame = tensors.df.loc[tensors.df.split == split, fields].copy()
    frame.insert(0, "filtered_row_index", frame.index)
    frame["predicted_label"] = np.asarray(LABELS)[pred]
    if probabilities is not None:
        for i, label in enumerate(LABELS):
            frame[f"uncalibrated_softmax_{label.lower()}"] = probabilities[:, i]
    frame.to_csv(path, index=False)


def load_input_control(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected an input-control checkpoint")
    model = InputControlMLP(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def train_control(mode, cfg, tensors, views, normalization, root, *, on_epoch=None, quiet=False,
                  evaluate_test=True, diagnostics=False):
    """Train an input-control model; evaluation and diagnostics are explicit options."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)  # Same shuffle per arm, independent of dropout.
    device = tensors.wt_feat.device
    model = InputControlMLP(mode, tensors.input_dim, tensors.go_features.shape[1],
                            cfg.hidden_dims, cfg.dropout, normalization).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    counts, weights = class_weights(views["train"]["label_idx"], cfg.input_balance_classes)
    val_y = views["val"]["label_idx"].cpu().numpy()
    history, best_f1, best_epoch, stale = [], -1., 0, 0
    checkpoint_path = root / "decoder.pt"
    for epoch in range(1, cfg.epochs + 1):
        epoch_lr = opt.param_groups[0]["lr"]
        model.train()
        total_loss = 0.
        for idx in iterate_batches(views["train"]["n"], cfg.batch_size, True, generator):
            y = views["train"]["label_idx"][idx.to(device)]
            # Fixed sample mean, not the batch-dependent sum-of-weights reduction.
            loss = (F.cross_entropy(batch_logits(model, tensors, views["train"], idx), y,
                                    reduction="none") * weights[y]).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in {mode}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        val_probs = predict(model, tensors, views["val"], cfg.batch_size)
        metric = classification_metrics(val_y, val_probs.argmax(1))
        row = dict(epoch=epoch, train_loss=total_loss / views["train"]["n"], val_macro_f1=metric["macro_f1"])
        if diagnostics:
            tr = evaluate_objective(model, tensors, views["train"], weights, cfg.batch_size)
            va = evaluate_objective(model, tensors, views["val"], weights, cfg.batch_size)
            row.update(train_eval_loss=tr["eval_loss"], val_eval_loss=va["eval_loss"],
                       train_macro_f1=tr["macro_f1"], f1_gap=tr["macro_f1"] - metric["macro_f1"], lr=epoch_lr)
        history.append(row)
        if metric["macro_f1"] > best_f1:  # Earliest epoch wins exact ties.
            best_f1, best_epoch, stale = metric["macro_f1"], epoch, 0
            torch.save(dict(format=CHECKPOINT_FORMAT, model_spec=model.spec,
                            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            labels=list(LABELS), selection=dict(epoch=epoch, val_macro_f1=best_f1),
                            class_counts=counts.tolist(), class_weights=weights.tolist(),
                            go_artifact_sha256=tensors.go_artifact_sha256), checkpoint_path)
        else:
            stale += 1
        if on_epoch:
            on_epoch(mode, row)
        if not quiet:
            print(f"[{mode} {epoch:03d}] loss={row['train_loss']:.4f} val_macroF1={row['val_macro_f1']:.4f}", flush=True)
        if stale >= cfg.early_stop_patience:
            break
    model, _ = load_input_control(checkpoint_path, device)
    result = dict(parameter_count=sum(p.numel() for p in model.parameters()),
                  best_epoch=best_epoch, epochs_run=len(history), class_counts=counts.tolist(),
                  class_weights=weights.tolist(), checkpoint=str(checkpoint_path), metrics={})
    if diagnostics:
        result["metrics"]["train"] = evaluate_objective(model, tensors, views["train"], weights, cfg.batch_size)
    for split in (("val", "test") if evaluate_test else ("val",)):
        probs = predict(model, tensors, views[split], cfg.batch_size)
        pred = probs.argmax(1)
        result["metrics"][split] = classification_metrics(views[split]["label_idx"].cpu().numpy(), pred)
        if diagnostics:
            result["metrics"][split]["eval_loss"] = evaluate_objective(model, tensors, views[split], weights, cfg.batch_size)["eval_loss"]
        save_predictions(tensors, split, pred, root / f"{split}_predictions.csv", probs)
    (root / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    return result, history
