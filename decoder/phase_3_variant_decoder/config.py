"""
Phase 3 variant decoder configuration.

Design: phase_3_variant_decoder/DESIGN.md

Differences from phase 2: there is no aspect here (a single multi-aspect decoder,
DESIGN §3.3) and no checkpoint is loaded (the decoder is initialised from scratch).
Because the input is a [global | local] pair, each embedding type needs two H5 files.
"""

import os
import math
from pathlib import Path

# Same naming as phase_2_decoder/config.py:12 — kept deliberately so grid scripts
# and reporting do not run into name mismatches.
EMBEDDING_TYPES = ("raw_esm", "seq_phase0", "seq_phase1", "concat_phase0", "concat_phase1")

# concat types are paired as [primary | raw_esm]; no extra ESM forward pass needed.
SECONDARY_REQUIRED = ("concat_phase0", "concat_phase1")

FAMILIES = ("easy", "hard")

# Input ablation (DESIGN §3.1). Phase 3's central claim is that the local embedding
# carries the variant signal — the only way to test it is to remove the local half
# and compare.
#   both   [global | local]   default behaviour
#   global [global]           without local — the control for the claim
#   local  [local]            without global; NOT generated for sweeps, kept only
#                             for manual sanity checks
INPUT_MODES = ("both", "global", "local")
SWEEP_INPUT_MODES = ("both", "global")

# How the Effect gate is selected during training.  ``validation`` preserves the
# original behaviour: tune an external probability threshold on validation
# macro-F1 every epoch.  ``learned_radius`` keeps the probability threshold at
# 0.5, so the trainable radius inside BinaryScoreStage is the decision boundary.
EFFECT_THRESHOLD_STRATEGIES = ("validation", "learned_radius")
CASCADE_LOSS_TYPES = ("hinge", "squared_hinge", "bce")

# DESIGN §4 — this class order is used everywhere; do not confuse it with go_idx.
LABELS = ("LOF", "GOF", "NEUTRAL")
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_default_dataset_dir() -> str:
    """Prefer the explicit two-stage cascade split artifact.

    The environment override is useful on the server, where datasets live on the
    large data disk rather than inside the repository checkout.
    """
    override = os.environ.get("PHASE3_DATASET_DIR")
    if override:
        return override
    candidates = (
        REPO_ROOT.parent / "splits_cascade",
        REPO_ROOT / "splits_cascade",
        REPO_ROOT / "data" / "processed" / "splits_cascade",
    )
    found = next((path for path in candidates if path.is_dir()), None)
    return str(found or candidates[-1])


def resolve_default_go_embeddings() -> str | None:
    """Find the selected local Anc2vec artifact without depending on cwd."""
    override = os.environ.get("PHASE3_GO_EMBEDDINGS")
    if override:
        return override
    filename = "go_terms_embeddings_anc2vec.npz"
    candidates = (
        REPO_ROOT / filename,
        REPO_ROOT.parent / filename,
        REPO_ROOT / "data" / "processed" / "anc2vec" / filename,
    )
    return next((str(path) for path in candidates if path.is_file()), None)

# WandB project — SINGLE SOURCE OF TRUTH. make_sweeps, run_sweeps, apply_best,
# report and train all read it from here. Change it once and everything follows;
# otherwise sweep runs land in one project and apply_best runs in another.
# Can also be overridden per shell: PHASE3_WANDB_PROJECT=...
WANDB_PROJECT = os.environ.get("PHASE3_WANDB_PROJECT", "contvar_gridsearch")
WANDB_ENTITY = os.environ.get("PHASE3_WANDB_ENTITY") or None

# ESM-2 650M truncation limit. DESIGN §2.3: rows with pos > MAX_SEQ_LEN are dropped,
# because no residue embedding is produced for a position outside the truncation.
MAX_SEQ_LEN = 1022
ESM_DIM = 1280


class VariantDecoderConfig:

    def __init__(self, family: str = "hard", embedding_type: str = "raw_esm",
                 input_mode: str = "both"):
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family!r} (choices: {FAMILIES})")
        if embedding_type not in EMBEDDING_TYPES:
            raise ValueError(
                f"Unknown embedding_type: {embedding_type!r} (choices: {EMBEDDING_TYPES})"
            )
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode: {input_mode!r} (choices: {INPUT_MODES})")

        self.family = family
        self.embedding_type = embedding_type
        self.input_mode = input_mode

        # ── Data ─────────────────────────────────────────────────────
        self.dataset_dir = resolve_default_dataset_dir()
        self.sequences_json = "/media/ubuntu/8TB/mfurkan/contvar_data/raw/uniprot_dict.json"
        self.max_seq_len = MAX_SEQ_LEN
        # DESIGN §2.3 optional filter: 125 rows with contradictory labels for the
        # same (protein, snp, GO).
        self.drop_contradictory = False

        # ── Embedding H5 files ───────────────────────────────────────
        # Two files per type: global (protein / protein_snp) + local (protein_pos / protein_snp)
        self.global_h5 = None
        self.local_h5 = None
        self.global_h5_secondary = None   # concat_* only
        self.local_h5_secondary = None

        # ── Decoder architecture ─────────────────────────────────────
        # FFNDecoder is reused as-is from phase_2_decoder/models/ffn_decoder.py.
        self.hidden_dims = [512, 1024, 512]
        self.dropout = 0.3
        # Programmatic legacy sweep callers retain their original behavior.
        # The train CLI explicitly defaults to the new cascade architecture.
        self.architecture = "legacy"
        self.go_embeddings = resolve_default_go_embeddings()
        self.allow_go_proxies = False
        self.cascade_balance_classes = True
        self.cascade_loss = "hinge"
        # Margin in the SCALED signed decision-logit space; not cfg.margin,
        # which belongs to the legacy three-class delta hinge experiment.
        self.cascade_margin = 1.0
        self.effect_threshold_strategy = "validation"

        # ── Loss (DESIGN §4, single-margin revision) ──────────────────
        # One shared margin for all three classes: LOF/GOF want |d| >= m (on
        # their own side), NEUTRAL wants |d| <= m. There used to be a separate
        # `margin_neutral` (m_n < m); removed at the user's request.
        self.margin = 0.3          # m
        self.w_lof = 1.0
        self.w_gof = 1.0
        self.w_neutral = 1.0

        # ── Training ─────────────────────────────────────────────────
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 256
        self.epochs = 80
        self.early_stop_patience = 15
        self.seed = 42
        self.cuda = None

        # GOA pre-training is deferred (DESIGN §7). The flag exists already so the
        # code path is ready when it is enabled and the ablation is one argument away.
        self.stage1 = "none"

        # ── Weights & Biases ─────────────────────────────────────────
        self.wandb_project = WANDB_PROJECT
        self.wandb_entity = WANDB_ENTITY
        self.wandb_run_name = f"phase3-{family}-{embedding_type}-{input_mode}"
        self.wandb_enabled = True
        self.wandb_tags: list[str] = ["phase3"]
        # Where the run came from — goes into results.csv and wandb metadata.
        self.source = "train"      # train | sweep | apply_best
        # Which make_sweeps.py preset this run belongs to ("lite" | "full" | "").
        # results.csv is a single cumulative file shared by every batch of runs
        # ever done; without this, separating one search's rows from another's
        # (e.g. a quick "lite" scan vs. the final "full" search) requires guessing
        # from timestamps. Rows written before this field existed are "" — filter
        # those by date instead (report.py --since).
        self.preset = ""
        self.sweep_id = ""

    def wandb_metadata(self) -> tuple[list[str], dict]:
        """Filterable wandb tags + config metadata (R3)."""
        from phase_3_variant_decoder.presentation_terms import configuration_metadata
        tags = [
            "phase3",
            f"fam:{self.family}",
            f"emb:{self.embedding_type}",
            f"mode:{self.input_mode}",
            f"src:{self.source}",
            *[t for t in self.wandb_tags if t != "phase3"],
        ]
        meta = {
            "architecture": self.architecture,
            "family": self.family,
            "embedding_type": self.embedding_type,
            "input_mode": self.input_mode,
            "source": self.source,
            "sweep_id": self.sweep_id,
            "input_dim": self.input_dim,
            **configuration_metadata(self),
        }
        return tags, meta

    @property
    def input_dim(self) -> int:
        """Decoder input width.

                        raw/seq_*   concat_*
            both          2560        5120
            global        1280        2560
            local         1280        2560
        """
        blocks = 2 if self.embedding_type in SECONDARY_REQUIRED else 1   # primary (+ raw)
        parts = 2 if self.input_mode == "both" else 1                    # global (+ local)
        return ESM_DIM * blocks * parts

    def validate(self):
        if self.cascade_loss not in CASCADE_LOSS_TYPES:
            raise ValueError(f"Unknown cascade_loss: {self.cascade_loss}")
        if not math.isfinite(self.cascade_margin) or self.cascade_margin <= 0:
            raise ValueError("cascade_margin must be finite and positive")
        if self.architecture not in ("legacy", "cascade"):
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.effect_threshold_strategy not in EFFECT_THRESHOLD_STRATEGIES:
            raise ValueError(
                "Unknown effect_threshold_strategy: "
                f"{self.effect_threshold_strategy!r} "
                f"(choices: {EFFECT_THRESHOLD_STRATEGIES})"
            )
        if self.architecture == "cascade":
            if not self.go_embeddings:
                raise ValueError(
                    "Cascade requires a GO embedding NPZ. Place "
                    "go_terms_embeddings_anc2vec.npz in a discovered location, set "
                    "PHASE3_GO_EMBEDDINGS, or pass --go_embeddings; see docs/CASCADE.md"
                )
            if not Path(self.go_embeddings).is_file():
                raise FileNotFoundError(f"GO embeddings not found: {self.go_embeddings}")
        missing = [n for n in ("global_h5", "local_h5") if getattr(self, n) is None]
        if self.embedding_type in SECONDARY_REQUIRED:
            missing += [
                n for n in ("global_h5_secondary", "local_h5_secondary")
                if getattr(self, n) is None
            ]
        if missing:
            raise ValueError(
                f"embedding_type={self.embedding_type!r} is missing H5 path(s): "
                f"{', '.join(missing)}"
            )
        for name in ("global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary"):
            path = getattr(self, name)
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")
