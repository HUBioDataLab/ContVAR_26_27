"""Repository-local defaults and the frozen-feature input contract.

Large inputs remain in the repository's existing ``data/`` directory. The
legacy ``data/go/`` location is supported for the original encoder package.
No paths depend on the shell's current working directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECODER_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_SOURCE_DIR = DEFAULT_DATA_ROOT / "splits"
DEFAULT_PREPARED_DIR = DEFAULT_DATA_ROOT / "prepared" / "new_decoder"
DEFAULT_RUNS_DIR = DECODER_ROOT / "runs"
LABELS = ("LOF", "GOF", "NEUTRAL")
FAMILIES = ("easy", "hard")
SPLITS = ("train", "val", "test")
FEATURE_DIM = 1280
MAX_SEQUENCE_LENGTH = 1022
PARALLEL_THRESHOLD = 0.5
CHECKPOINT_FORMAT = "contvar_go_decoder_starter_v1"


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Share identical optimizer/model defaults between single and suite CLIs."""
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")


def validate_training_arguments(args: argparse.Namespace) -> None:
    """Reject invalid options before creating output directories or loading data."""
    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.threads,
        args.width,
        args.attention_heads,
    )
    if min(positive) < 1 or args.width % args.attention_heads:
        raise ValueError(
            "Positive training limits and width divisible by attention heads are required"
        )
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    value = getattr(args, "lambda_reverse", 0.0)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Reverse loss weight must be finite and nonnegative")


@dataclass(frozen=True)
class DataPaths:
    """Resolve two pooled protein scales, GO vectors and WT sequences."""

    root: Path = DEFAULT_DATA_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def global_h5(self) -> Path:
        return self.root / "embeddings/phase3_seq_phase1_global.h5"

    @property
    def mutation_h5(self) -> Path:
        return self.root / "embeddings/phase3_seq_phase1_local.h5"

    @property
    def go_embeddings(self) -> Path:
        repository_path = self.root / "embeddings/go_terms_embeddings_anc2vec.npz"
        legacy_path = self.root / "go/go_terms_embeddings_anc2vec.npz"
        if repository_path.is_file() or not legacy_path.is_file():
            return repository_path
        return legacy_path

    @property
    def sequences_json(self) -> Path:
        return self.root / "sequences/uniprot_dict.json"
