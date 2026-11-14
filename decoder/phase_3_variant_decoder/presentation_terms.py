"""Single source of English labels for Phase-3 figures, tables and W&B.

Machine identifiers remain unchanged for checkpoint/cache compatibility.
The history below describes the repository pipeline, not an audit of arbitrary
external H5 files: filenames alone do not prove checkpoint provenance.
"""

import textwrap
import csv
import json
from pathlib import Path

REPORTING_VERSION = "phase3-reporting-v3-readable-tables"
ENCODER_LABELS = {
    "raw_esm": "Pretrained ESM-2",
    "seq_phase0": "Phase 0 (GO)",
    "seq_phase1": "Phase 0 (GO) -> Phase 1 (DMS)",
    "concat_phase0": "[Phase 0 (GO)] + [pretrained ESM-2]",
    "concat_phase1": "[Phase 0 (GO) -> Phase 1 (DMS)] + [pretrained ESM-2]",
}
ENCODER_DEFINITIONS = {
    "raw_esm": "Pretrained ESM-2 without ContVAR-specific GO or DMS adaptation; not an untrained encoder.",
    "seq_phase0": "Pretrained ESM-2 followed by Phase 0 GO-semantic pretraining.",
    "seq_phase1": "Pretrained ESM-2 followed by Phase 0 GO-semantic pretraining, then Phase 1 DMS fine-tuning. This is cumulative training, not a DMS-only ablation.",
    "concat_phase0": "Feature concatenation of the Phase 0 encoder output and pretrained ESM-2 output; '+' denotes two input blocks.",
    "concat_phase1": "Feature concatenation of the cumulatively trained Phase 0 -> Phase 1 encoder output and pretrained ESM-2 output; '+' denotes two input blocks, not an additional training phase.",
}
INPUT_LABELS = {"global": "Global only", "local": "Local only", "both": "Global + Local (concat)"}


def encoder_label(embedding: str, *, multiline: bool = False) -> str:
    label = ENCODER_LABELS[embedding]
    return label.replace(" -> ", "\n-> ") if multiline else label


def configuration_metadata(config) -> dict:
    """Add explicit display fields without changing stored experiment IDs."""
    values = config if isinstance(config, dict) else vars(config)
    embedding = values.get("embedding_type", values.get("embedding"))
    mode = values.get("input_mode")
    fields = {
        "reporting_version": REPORTING_VERSION,
        "family": values.get("family", "unspecified"),
        "embedding_type": embedding,
        "encoder_label": ENCODER_LABELS.get(embedding, "Unspecified encoder"),
        "encoder_training_history": ENCODER_DEFINITIONS.get(embedding, "Not supplied"),
        "encoder_history_basis": "Repository pipeline definition; external artifact lineage is not verified by a filename.",
        "input_mode": mode,
        "input_label": INPUT_LABELS.get(mode, "Unspecified input"),
        "seed": values.get("seed", "unspecified"),
    }
    if "cascade_balance_classes" in values and values.get("architecture", "cascade") == "cascade":
        fields["class_weighting"] = "Train-derived inverse-frequency" if values["cascade_balance_classes"] else "Unweighted"
        fields["cascade_loss"] = values.get("cascade_loss", "bce")
        fields["cascade_margin"] = values.get("cascade_margin", 1.0)
        fields["score_interpretation"] = "Uncalibrated sigmoid decision score"
    return fields


def configuration_text(config) -> str:
    meta = configuration_metadata(config)
    return (f"Split: {meta['family']} | Encoder: {meta['encoder_label']} | "
            f"Input: {meta['input_label']} | Seed: {meta['seed']}")


def save_figure(fig, path, config=None, *, description: str | None = None, context: str | None = None) -> None:
    """Include configuration in exported figures as well as W&B metadata."""
    notes = []
    if config is not None:
        notes.append(configuration_text(config))
    if context:
        notes.append(context)
    description = description or FIGURE_NOTES_TR.get(Path(path).name)
    if description:
        notes.append(description)
    if notes:
        note = "\n\n".join("\n".join(textwrap.wrap(part, width=125)) for part in notes)
        fig.text(.025, -.035, note, ha="left", va="top", fontsize=10.5, color="#253b53",
                 bbox={"facecolor": "#f1f5f9", "edgecolor": "#d5deea", "pad": 8})
    fig.savefig(path, dpi=200, bbox_inches="tight")


def finalize_report(artifacts: dict[str, Path], config=None) -> dict[str, str]:
    """Carry figure context into standalone CSV/Markdown/JSON exports too."""
    if config is not None:
        meta = configuration_metadata(config)
        # A weighting comparison contains both objectives, so generic config
        # weighting must not be mistaken for the condition of every row.
        display = {key: meta[key] for key in (
            "family", "embedding_type", "encoder_label", "encoder_training_history",
            "input_mode", "input_label", "seed", "reporting_version")}
        for key, path in list(artifacts.items()):
            if key == "summary_csv":
                with Path(path).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    rows = [{**display, **row} for row in rows]
                    with Path(path).open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            if key == "summary_markdown":
                with Path(path).open("a", encoding="utf-8") as handle:
                    handle.write("\n## Configuration and metric definitions\n\n" + configuration_text(config) + "\n\n")
                    handle.write("Training history follows the repository pipeline; external checkpoint provenance must be verified.\n\n")
                    handle.write("\n".join(f"- **{term}:** {definition}" for term, definition in METRIC_DEFINITIONS.items()) + "\n")
        manifest = Path(artifacts["summary_csv"]).parent / "report_metadata.json"
        manifest.write_text(json.dumps({"configuration": meta, "metric_definitions": METRIC_DEFINITIONS}, indent=2) + "\n", encoding="utf-8")
        artifacts["report_metadata"] = manifest
    return {key: str(path) for key, path in artifacts.items()}


def log_reporting_guide(run, wandb, configs: list[dict]) -> None:
    """Keep meanings visible in each run without renaming historical metrics."""
    rows = [configuration_metadata(cfg) for cfg in configs]
    columns = sorted(set().union(*(row.keys() for row in rows)))
    run.log({
        "00_Overview/Configuration_Definitions": wandb.Table(
            columns=columns, data=[[row.get(key) for key in columns] for row in rows]),
        "00_Overview/Metric_Definitions": wandb.Table(
            columns=["term", "definition"], data=[list(item) for item in METRIC_DEFINITIONS.items()]),
    })

METRIC_DEFINITIONS = {
    "Final three-class macro-F1":
        "Arithmetic mean of LOF, GOF and NEUTRAL F1 on all evaluation rows; each class has equal weight. Not accuracy.",
    "Per-class F1":
        "Harmonic mean of precision and recall for one class, treated one-versus-rest.",
    "Accuracy": "Correct final predictions divided by all evaluation rows.",
    "Effect-detection macro-F1":
        "Binary macro-F1: EFFECT (LOF or GOF) versus NEUTRAL, on all evaluation rows.",
    "Effect-direction macro-F1":
        "Binary LOF-versus-GOF macro-F1 on true LOF/GOF rows, irrespective of effect-gate routing. Excludes true NEUTRAL rows.",
    "Effect false-negative rate":
        "True LOF/GOF rows predicted NEUTRAL, divided by all true LOF/GOF rows.",
    "NEUTRAL false-positive rate":
        "True NEUTRAL rows routed to the direction stage (final LOF/GOF), divided by all true NEUTRAL rows.",
    "GO terms unseen in training":
        "Evaluation rows whose target GO ID does not occur in the filtered decoder-training partition. This does not mean unseen by Anc2vec, GO-semantic encoder pretraining, or the ontology itself.",
    "Global input": "Mean-pooled protein-level ESM representation over the retained sequence (up to 1,022 residues); not attention pooling.",
    "Local input":
        "ESM representation of the single mutated residue position. Not a sequence-window or 3D-pocket representation.",
    "Both / combined input": "Concatenation of global and mutation-site residue embeddings.",
    "Phase 0": "Sequence encoder after GO semantic pretraining.",
    "Phase 0 -> Phase 1": ENCODER_DEFINITIONS["seq_phase1"],
    "Pretrained ESM-2": ENCODER_DEFINITIONS["raw_esm"],
    "Concatenated encoder sources": "concat_phase0/concat_phase1 append pretrained ESM-2 features to adapted encoder features; distinct from input_mode=both (Global + Local).",
    "Training history versus decoder stage": "Phase 0/Phase 1 describe encoder training history. Effect detection/direction describe the two decoder stages. They are separate axes.",
    "F1 aggregation": "Macro-F1 is calculated from pooled evaluation-row predictions over a fixed class set; it is not an average of per-protein or per-GO F1 scores. Undefined class F1 is assigned zero.",
    "Easy versus hard": "Easy permits protein overlap across partitions; hard uses protein-disjoint partitions. Identifier separation does not establish sequence-homology independence. Evaluation populations and label distributions can differ.",
    "Validation versus test":
        "Validation metrics guide checkpoint/threshold selection. Test metrics describe the selected model and must not guide selection.",
}

ARTIFACT_CAPTIONS = {
    "input_ablation":
        "Global-only versus Local-only versus Global + Local concatenation. The decoder, hard split, cumulative Phase 0 -> Phase 1 encoder and primary seed are fixed. Local denotes the single mutated-residue position, not a sequence window or 3D pocket.",
    "encoder_ablation":
        "Pretrained ESM-2 versus Phase 0 (GO) versus cumulative Phase 0 (GO) -> Phase 1 (DMS). This does not include a DMS-only encoder. Hard split, Global + Local input and primary seed are fixed; differences also reflect encoder training history.",
    "split_generalization":
        "Easy versus protein-disjoint hard split. The cumulative Phase 0 -> Phase 1 encoder, Global + Local input and primary seed are fixed. Evaluation populations can differ; the gap alone does not quantify memorization or homology-independent generalization.",
    "seed_stability":
        "Repeated training seeds for the reference configuration. Error summaries describe seed variation, not a confidence interval for biological generalization.",
    "cascade_diagnostics":
        "Effect: binary on all test rows. Direction: binary on true LOF/GOF rows irrespective of routing. Final: three-class on all test rows. Routing errors use their respective true-class denominators. Seen/unseen refers only to target GO IDs in decoder training, not to ontology or encoder-pretraining coverage.",
    "overview": "Diagnostic ablations of input representations, encoder training stages, split families and random seeds; final scores are three-class macro-F1.",
    "error_breakdown":
        "Binary stage metrics and final three-class metrics use different tasks/populations. Effect false negatives are conditional on true effects; NEUTRAL false positives are conditional on true NEUTRAL rows.",
    "training_curves":
        "Training BCE loss and validation binary macro-F1 by epoch and cascade stage. Weighted and unweighted BCE values use different loss scales; compare predictive metrics directly.",
    "learned_training_curves":
        "Training BCE and validation binary macro-F1 by epoch, separately for effect detection and LOF/GOF direction. Compare validation-tuned routing with a learned radius at fixed probability threshold 0.50. Validation selects checkpoints; test does not select the boundary.",
    "test_metrics":
        "End-to-end three-class test performance and class-wise F1. Macro-F1 gives LOF, GOF and NEUTRAL equal weight; accuracy weights rows equally.",
    "confusion_matrices":
        "Prediction-count table: True class names the actual class, Actual is its support, and Predicted LOF/GOF/NEUTRAL columns show the assigned counts. Green cells are correct; pink cells are errors. No row-normalized percentages are displayed.",
    "confusions":
        "Prediction-count table: True class names the actual class, Actual is its support, and Predicted LOF/GOF/NEUTRAL columns show the assigned counts. Green cells are correct; pink cells are errors. No row-normalized percentages are displayed.",
    "threshold_sweep":
        "Validation-only sweep with model weights fixed. Binary effect macro-F1 and final three-class macro-F1 are different selection objectives. Test labels are not used to select thresholds.",
    "comparison":
        "Final three-class macro-F1, per-class test F1 and effect-gate routing errors. Effect false-negative rate is conditional on true LOF/GOF; NEUTRAL false-positive rate is conditional on true NEUTRAL.",
}

SHORT_DESCRIPTIONS_TR = {
    "input_ablation": "Global, Local ve Global + Local girdilerini karşılaştırıyoruz. Decoder, split, encoder geçmişi ve seed sabit. Local tek mutasyon pozisyonunun embedding'idir. F1 ne kadar yüksekse o kadar iyi; F1, Accuracy değildir.",
    "encoder_ablation": "Aynı decoder ve input ile encoder eğitim geçmişlerini karşılaştırıyoruz. Phase 0 -> Phase 1, GO eğitiminden sonra DMS ile devam edildiğini gösterir; yalnız DMS eğitimi değildir. Tek seed farkı kesin üstünlük kanıtı değildir.",
    "split_generalization": "Ayarlar aynı, değerlendirme grupları farklı: Easy protein örtüşmesine izin verir, Hard protein ID'lerini ayırır. İki skorun farkı tek başına ezberleme miktarını göstermez.",
    "seed_stability": "Aynı konfigürasyon farklı decoder eğitim seed'leriyle tekrarlandı. Kesikli çizgi test ortalamasıdır. SD tekrarlar arasındaki değişkenliktir; tek seed ile SD hesaplanamaz.",
    "cascade_diagnostics": "Effect: tüm örneklerde etki var mı? Direction: yalnız gerçek LOF/GOF örneklerinde yön ne? Final: tüm örneklerde üç sınıf. Farklı görevler olduğu için F1'lar doğrudan aynı başarı ölçüsü değildir. GO grupları decoder train'e göredir.",
    "overview": "Input, encoder geçmişi, split ve seed ayrı karşılaştırmalardır. VAL seçim, TEST son değerlendirme içindir. Macro-F1 üç sınıfa eşit ağırlık verir; bir accuracy yüzdesi değildir.",
    "error_breakdown": "Effect false-negative rate: gerçek etkili örneklerin kaçırılan oranı. NEUTRAL false-positive rate: gerçek NEUTRAL örneklerin yanlışlıkla etkili sayılan oranı. Hata oranında düşük, F1'da yüksek daha iyidir.",
    "training_curves": "Üst sıra train loss, alt sıra validation macro-F1 gösterir. Effect ve Direction ayrı görevlerdir. Weighted ve Unweighted loss ölçekleri farklı olduğundan başarı kıyasını F1 üzerinden yapıyoruz.",
    "learned_training_curves": "Effect karar sınırı için iki eğitim yaklaşımını karşılaştırıyoruz. Train loss öğrenmeyi, validation F1 checkpoint seçimini gösterir; test verisi seçimde kullanılmaz.",
    "test_metrics": "Macro-F1 üç sınıfın F1 ortalamasıdır. Accuracy toplam doğru tahmin oranıdır. LOF, GOF ve NEUTRAL F1 ayrı gösterilir; aynı sayı farklı görevler için kullanılmıyor.",
    "confusion_matrices": "True class gerçek sınıftır. Actual o sınıftaki örnek sayısı; Predicted sütunları modelin verdiği cevaplardır. Yeşil hücre doğru, pembe hücre yanlış tahmin sayısıdır. Bu tabloda satır yüzdesi yoktur.",
    "confusions": "True class gerçek sınıftır. Actual o sınıftaki örnek sayısı; Predicted sütunları modelin verdiği cevaplardır. Yeşil hücre doğru, pembe hücre yanlış tahmin sayısıdır. Bu tabloda satır yüzdesi yoktur.",
    "threshold_sweep": "Ağırlıklar sabitken yalnız effect threshold değişiyor. Eğriler validation sonuçlarıdır; threshold testten seçilmez. Hata oranında düşük, F1'da yüksek daha iyidir.",
    "comparison": "Sol: final Macro-F1. Orta: sınıf F1'ları. Sağ: effect gate hata oranları. F1'daki artış iyi, hata oranındaki artış kötüdür; hata oranlarının paydaları farklıdır.",
}

FIGURE_NOTES_TR = {
    filename: SHORT_DESCRIPTIONS_TR[key] for filename, key in {
        "01_input_ablation_f1.png": "input_ablation", "02_encoder_ablation_f1.png": "encoder_ablation",
        "03_split_generalization_f1.png": "split_generalization", "04_seed_stability_f1.png": "seed_stability",
        "05_cascade_diagnostics.png": "cascade_diagnostics", "diagnostic_overview.png": "overview",
        "diagnostic_error_breakdown.png": "error_breakdown", "training_curves.png": "training_curves",
        "test_metrics.png": "test_metrics", "architecture_test_metrics.png": "test_metrics",
        "learned_threshold_training_curves.png": "learned_training_curves",
        "learned_threshold_comparison.png": "comparison", "threshold_comparison.png": "comparison",
        "validation_threshold_sweep.png": "threshold_sweep",
    }.items()
}


def image_caption(artifact_key: str) -> str:
    return SHORT_DESCRIPTIONS_TR[artifact_key] + "\n\nTechnical definition: " + ARTIFACT_CAPTIONS[artifact_key]


def contextual_caption(artifact_key: str, config) -> str:
    return configuration_text(config) + ". " + image_caption(artifact_key)
