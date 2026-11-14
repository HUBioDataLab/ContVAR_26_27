"""Matched Concat/cascade checkpoint inference and frozen mutation diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.audit_mutation_cohorts import audit_family
from phase_3_variant_decoder.cascade import attach_checkpoint_go, evaluate_cascade
from phase_3_variant_decoder.config import LABELS
from phase_3_variant_decoder.data.variant_dataset import load_sequences
from phase_3_variant_decoder.direct_pair_decoder import evaluate
from phase_3_variant_decoder.input_controls import classification_metrics
from phase_3_variant_decoder.mutation_sensitivity import easy_diagnostic, pair_diagnostic, signed_index

IDENTITY = ["protein_id", "matched_go_id", "snp", "label"]
PROTOCOL = ("Frozen raw-train EASY anti-majority cohort; covered original HARD row pairs primary; "
            "deduplicated/nonconflicting pairs supplementary; ties receive 0.5")


def validate_cascade_config(cfg, saved):
    """Never silently compare different filter/feature settings."""
    defaults = dict(max_seq_len=1022, drop_contradictory=False)
    for key in ("family", "embedding_type", "input_mode", "max_seq_len", "drop_contradictory"):
        value = saved.get(key, defaults.get(key))
        if value != getattr(cfg, key):
            raise ValueError(f"Concat/cascade configuration mismatch: {key}: {getattr(cfg, key)!r} != {value!r}")


def validate_go_alignment(concat, cascade, vocab):
    """Compare actual aligned vectors, not file ordering or path labels."""
    def aligned(checkpoint):
        missing = set(vocab) - set(checkpoint["go_vocab"])
        if missing:
            raise ValueError(f"Checkpoint lacks GO vectors for {len(missing)} current terms")
        indices = [checkpoint["go_vocab"][g] for g in sorted(vocab, key=vocab.get)]
        vectors = checkpoint["go_vectors"][indices].cpu()
        if vectors.ndim != 2 or not torch.isfinite(vectors).all():
            raise ValueError("Invalid checkpoint GO vectors")
        return vectors
    a, b = aligned(concat), aligned(cascade)
    if a.shape != b.shape or not torch.equal(a, b):
        raise ValueError("Checkpoints use different GO vectors; this is not a matched-input comparison")


def prepare_cohorts(cfg, frame, root):
    audit, source_test, raw_pairs, raw = audit_family(
        cfg.family, cfg.dataset_dir, load_sequences(cfg.sequences_json),
        cfg.max_seq_len, cfg.drop_contradictory)
    if cfg.family == "hard":
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(raw.loc[raw.split.eq(a), "protein_id"]) & set(raw.loc[raw.split.eq(b), "protein_id"])
            if overlap:
                raise ValueError(f"HARD is not protein-disjoint: {a}/{b}")
    covered = source_test.loc[source_test.exclusion_reason.eq("")].reset_index(drop=True)
    test = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    pd.testing.assert_frame_equal(test[IDENTITY], covered[IDENTITY], check_dtype=False)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source_test.to_csv(root / "source_test_row_audit.csv", index=False)
    raw_pairs.to_csv(root / "raw_pair_manifest.csv", index=False)
    (root / "cohort_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return dict(audit=audit, covered_test=covered, raw_pairs=raw_pairs,
                raw_train=raw.loc[raw.split.eq("train")],
                train=frame.loc[frame.split.eq("train")])


def diagnostics(scored, context, name, root, tolerance=1e-8):
    """Exact same row/pair membership for both architectures; no fabricated heads."""
    scored = scored.copy().reset_index(drop=True)
    covered = context["covered_test"]
    pd.testing.assert_frame_equal(scored[IDENTITY], covered[IDENTITY], check_dtype=False)
    scored["source_row"] = covered.source_row.to_numpy()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result = dict(protocol=PROTOCOL, decision_index="class_score_gof - class_score_lof",
                  score_interpretation="Uncalibrated decision index; not measured activity or severity")
    if context["audit"]["family"] == "easy":
        easy, scored = easy_diagnostic(context["train"], scored,
                                       reference_train=context["raw_train"], model_name=name)
        easy["classification"] = [r for r in easy["classification"] if r["model"] == name]
        if easy["audit"]["anti_majority_rows"] != context["audit"]["easy"]["raw_cohort_covered"]:
            raise ValueError("EASY frozen cohort membership changed")
        result["easy"] = easy
    for key, clean in (("pairs", False), ("clean_pairs", True)):
        summary, pairs, groups = pair_diagnostic(scored, tolerance, clean=clean, score_column="decision_index")
        result[key] = summary
        pairs.to_csv(root / f"{key}.csv", index=False)
        groups.to_csv(root / f"{key}_groups.csv", index=False)
        if not clean:
            expected = context["raw_pairs"].loc[context["raw_pairs"].model_covered]
            fields = ["protein_id", "matched_go_id", "lower_source_row", "higher_source_row", "lower_label", "higher_label"]
            pd.testing.assert_frame_equal(
                pairs[fields].sort_values(fields).reset_index(drop=True),
                expected[fields].sort_values(fields).reset_index(drop=True), check_dtype=False)
    scored.to_csv(root / "test_predictions.csv", index=False)
    return result


def evaluate_models(concat_model, concat_ckpt, cascade_model, cascade_ckpt,
                    tensors, views, vocab, cfg, context, root, tolerance=1e-8):
    threshold = float(cascade_ckpt["effect_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved cascade effect threshold")
    validate_cascade_config(cfg, cascade_ckpt["config"])
    validate_go_alignment(concat_ckpt, cascade_ckpt, vocab)
    models = {"concat": (concat_model, concat_ckpt), "cascade": (cascade_model, cascade_ckpt)}
    arms = {}
    for method, (model, checkpoint) in models.items():
        if model.spec["protein_dim"] != tensors.input_dim:
            raise ValueError(f"{method}: protein dimension mismatch")
        attach_checkpoint_go(tensors, vocab, checkpoint)
        if model.spec["go_dim"] != tensors.go_features.shape[1]:
            raise ValueError(f"{method}: GO dimension mismatch")
        model.to(tensors.wt_feat.device).eval()
        destination = Path(root) / method
        destination.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for split in ("train", "val", "test"):
            view = views[split]
            if not view["n"]:
                raise ValueError(f"Empty {split} partition")
            if method == "concat":
                weights = torch.as_tensor(checkpoint["class_weights"], dtype=torch.float32,
                                          device=tensors.wt_feat.device)
                metric, class_scores = evaluate(model, tensors, view, weights, cfg.batch_size)
                prediction = class_scores.argmax(1)
            else:
                _, out = evaluate_cascade(model, tensors, view, threshold, cfg.batch_size)
                prediction = out["prediction"]
                effect = np.asarray(out["effect"]["probability"], dtype=float)
                direction = np.asarray(out["direction"]["probability"], dtype=float)
                class_scores = np.column_stack((effect*direction, effect*(1-direction), 1-effect))
                metric = classification_metrics(view["label_idx"].cpu().numpy(), prediction)
            class_scores = np.asarray(class_scores, dtype=float)
            frame = tensors.df.loc[tensors.df.split.eq(split)].copy().reset_index(drop=True)
            frame["prediction"] = np.asarray(LABELS)[prediction]
            if (not np.isfinite(class_scores).all() or class_scores.shape != (len(frame), 3)
                    or (class_scores < 0).any() or not np.allclose(class_scores.sum(1), 1)):
                raise ValueError(f"{method}: invalid class decision scores")
            for i, label in enumerate(LABELS):
                metric[f"{label.lower()}_support"] = int(frame.label.eq(label).sum())
                frame[f"class_score_{label.lower()}"] = class_scores[:, i]
            # Preserve the previous cascade diagnostic's float64 formula/ties.
            frame["decision_index"] = (signed_index(effect, direction) if method == "cascade"
                                       else class_scores[:, 1] - class_scores[:, 0])
            metrics[split] = metric
            if split == "test":
                mutation = diagnostics(frame, context, "Concat" if method == "concat" else "Cascade",
                                       destination, tolerance)
            else:
                frame.to_csv(destination / f"{split}_predictions.csv", index=False)
        arms[method] = dict(metrics=metrics, mutation=mutation,
                            parameter_count=sum(p.numel() for p in model.parameters()),
                            selection=checkpoint.get("selection"),
                            decision="softmax argmax" if method == "concat" else "saved hard cascade routing",
                            effect_threshold=threshold if method == "cascade" else None,
                            direction_threshold=.5 if method == "cascade" else None,
                            go_artifact_sha256=checkpoint["go_artifact_sha256"])
    for key in ("pairs", "clean_pairs"):
        a, b = [arms[m]["mutation"][key]["ranking"] for m in ("concat", "cascade")]
        if [(r["n_pairs"], r["n_groups"]) for r in a] != [(r["n_pairs"], r["n_groups"]) for r in b]:
            raise ValueError("Model pair cohorts differ")
    return arms
