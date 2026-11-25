"""Fit one approved condition; classification loss only and validation selection."""
from __future__ import annotations

import json
import random
import time

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
import torch
from torch.utils.data import DataLoader

from .data import model_inputs
from .model import PairedGODecoder, classification_loss, load_decoder
from .util import LABELS, utc_now, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics(labels, prediction, scores, head):
    labels, prediction, scores = np.asarray(labels), np.asarray(prediction), np.asarray(scores)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1, 2], zero_division=0)
    result = dict(n=len(labels), macro_f1=float(f1.mean()), accuracy=float((labels == prediction).mean()),
                  balanced_accuracy=float(recall.mean()),
                  mcc=float(matthews_corrcoef(labels, prediction))
                  if len(np.unique(np.concatenate([labels, prediction]))) > 1 else 0.,
                  confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist())
    for i, name in enumerate(LABELS):
        target = labels == i
        result[name.lower()] = dict(precision=float(precision[i]), recall=float(recall[i]),
            f1=float(f1[i]), support=int(support[i]),
            auprc=float(average_precision_score(target, scores[:, i])) if target.any() else None,
            auroc=float(roc_auc_score(target, scores[:, i])) if target.any() and (~target).any() else None)
    aps = [result[n.lower()]["auprc"] for n in LABELS if result[n.lower()]["auprc"] is not None]
    result["macro_auprc"] = float(np.mean(aps)) if aps else None
    if head == "parallel":
        result["both_heads_positive_rate"] = float(((scores[:, 0] >= .5) & (scores[:, 1] >= .5)).mean())
        result["score_semantics"] = "independent sigmoid heads plus auxiliary neutral score; not normalized class probabilities"
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    collected = {k: [] for k in ("label", "prediction", "scores", "row", "loss")}
    for batch in loader:
        output = model(**model_inputs(batch, device))
        labels = batch["label"].to(device)
        loss = classification_loss(output, labels, model.head_kind, reduction="none")
        for name, value in dict(label=labels, prediction=output["prediction"], scores=output["scores"],
                                row=batch["row"], loss=loss).items():
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Nonfinite evaluation {name}")
            collected[name].append(array)
    c = {k: np.concatenate(v) for k, v in collected.items()}

    def summarize(mask):
        out = metrics(c["label"][mask], c["prediction"][mask], c["scores"][mask], model.head_kind)
        out["classification_loss"] = float(c["loss"][mask].mean())
        return out

    result = summarize(np.ones(len(c["label"]), dtype=bool))
    frame = loader.dataset.frame.iloc[c["row"]].copy().reset_index(drop=True)
    for column in ("common_test", "seen_in_train", "multiple_labels_for_input"):
        mask = frame[column].astype(str).str.lower().eq("true").to_numpy()
        if mask.any():
            result[column] = summarize(mask)
        if column != "common_test" and (~mask).any():
            result["not_" + column] = summarize(~mask)
    if "is_combinatorial" in frame:
        combo = frame.is_combinatorial.astype(str).str.lower().eq("true").to_numpy()
        if combo.any():
            result["combinatorial"] = summarize(combo)
            if (~combo).any():
                result["single_substitution"] = summarize(~combo)
    for i, label in enumerate(LABELS):
        frame[f"score_{label.lower()}"] = c["scores"][:, i]
    frame["prediction"] = [LABELS[i] for i in c["prediction"]]
    frame["classification_loss"] = c["loss"]
    return result, frame


def atomic_torch_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fit(condition, datasets, device, destination, settings, provenance, resume=False):
    seed = settings["seed"]
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "config.json"
    config = dict(seed=seed, condition=condition, settings=settings, provenance=provenance,
                  labels=list(LABELS), encoder_frozen=True, anc2vec_frozen=True,
                  reverse_penalty=False, classification_only=True, class_balancing=False,
                  training_rows=len(datasets["train"]), validation_rows=len(datasets["val"]),
                  training_class_counts=np.bincount(datasets["train"].labels, minlength=3).tolist(),
                  checkpoint_selection="validation macro-F1 only; earliest epoch on ties",
                  parallel_decision="threshold .5; strongest positive head; neither -> NEUTRAL")
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Resume configuration/provenance mismatch: {destination}")
        if not resume:
            raise FileExistsError(destination)
        done = destination / "fit_results.json"
        if done.exists():
            print(f"RESUME_COMPLETE {condition['name']}", flush=True)
            return json.loads(done.read_text(encoding="utf-8"))
    else:
        write_json(config_path, config)
    seed_everything(seed)
    model = PairedGODecoder(input_dim=1280, go_dim=datasets["train"].store.go.dimension,
        width=settings["width"], heads=settings["attention_heads"], head=condition["head"],
        pair_mode=condition["pair_mode"], window_mode=condition["window_mode"],
        dropout=settings["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds, batch_size=settings["batch_size"], shuffle=s == "train", num_workers=0,
        generator=generator if s == "train" else None, pin_memory=str(device).startswith("cuda")) for s, ds in datasets.items()}
    history, best, stale, first_epoch, elapsed_before = [], -1., 0, 1, 0.
    if resume and (destination / "last.pt").exists():
        last = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
        if last["seed"] != seed:
            raise ValueError("Resume checkpoint seed mismatch")
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        history = json.loads((destination / "history.json").read_text(encoding="utf-8"))[:last["epoch"]]
        best, stale, first_epoch = last["best"], last["stale"], last["epoch"] + 1
        elapsed_before = last["elapsed_seconds"]
        torch.set_rng_state(last["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        generator.set_state(last["loader_rng"])
    started = time.monotonic()
    for epoch in range(first_epoch, settings["epochs"] + 1):
        if stale >= settings["patience"]:
            break
        model.train()
        loss_sum = 0.
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch, device))
            labels = batch["label"].to(device)
            loss = classification_loss(output, labels, model.head_kind)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        validation, _ = evaluate(model, loaders["val"], device)
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(datasets["train"]), validation=validation))
        write_json(destination / "history.json", history)
        if validation["macro_f1"] > best:
            best, stale = validation["macro_f1"], 0
            checkpoint = dict(format="contvar_mf_window7_v1", seed=seed, labels=list(LABELS), model_spec=model.spec,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                epoch=epoch, validation_macro_f1=best, condition=condition, settings=settings,
                provenance=provenance, encoder_frozen=True, reverse_penalty=False)
            atomic_torch_save(checkpoint, destination / "best.pt")
        else:
            stale += 1
        atomic_torch_save(dict(seed=seed, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), epoch=epoch, best=best, stale=stale,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            loader_rng=generator.get_state(), elapsed_seconds=elapsed_before + time.monotonic() - started), destination / "last.pt")
        print(f"RUN {condition['number']:02d}/48 {condition['name']} epoch={epoch} "
              f"loss={history[-1]['train_loss']:.4f} val_F1={validation['macro_f1']:.4f} "
              f"elapsed={elapsed_before+time.monotonic()-started:.1f}s", flush=True)
    model, checkpoint = load_decoder(destination / "best.pt", device)
    validation, prediction = evaluate(model, loaders["val"], device)
    write_predictions(prediction, destination / "val_predictions.tsv", datasets["val"].store)
    result = dict(seed=seed, condition=condition, best_epoch=checkpoint["epoch"], epochs_run=len(history),
        parameter_count=sum(p.numel() for p in model.parameters()),
        seconds=elapsed_before + time.monotonic() - started, validation=validation,
        completed=utc_now(), training_rows=len(datasets["train"]), validation_rows=len(datasets["val"]))
    write_json(destination / "fit_results.json", result)
    if getattr(datasets["train"].store, "allow_combinatorial", False):
        (destination / "last.pt").unlink(missing_ok=True)
    return result


def write_predictions(frame, path, store):
    if getattr(store, "allow_combinatorial", False):
        # Embedding keys and other source fields remain joinable by sample_id.
        columns = ["sample_id", "protein_id", "mutations", "matched_go_id", "label", "label_idx",
                   "is_combinatorial", "augmented", "common_test", "common_key", "seen_in_train",
                   "multiple_labels_for_input", "prediction", "score_lof", "score_gof", "score_neutral",
                   "classification_loss"]
        frame = frame[columns]
    frame.to_csv(path, sep="\t", index=False)
