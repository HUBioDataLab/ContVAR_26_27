"""Small file/provenance helpers for the isolated MF/window-7 suite."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE.parent / "New_Decoder_Data_with_local_Context"
LABELS = ("LOF", "GOF", "NEUTRAL")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_conditions(cohorts=("old", "full")):
    """Apply the same decoder matrix to each feature source and split family."""
    result = []
    basic = [("parallel", "score_difference"), ("multiclass", "score_difference"),
             ("parallel", "joint"), ("multiclass", "joint")]
    for cohort in cohorts:
        for phase in ("none", "window"):
            for family in ("easy", "hard"):
                conditions = [("none", h, p) for h, p in basic] if phase == "none" else (
                    [("mean", h, p) for h, p in basic] +
                    [(w, h, "joint") for w in ("pair_mean", "esim") for h in ("parallel", "multiclass")])
                for window_mode, head, pair_mode in conditions:
                    result.append(dict(number=len(result) + 1, cohort=cohort, family=family,
                                       window_mode=window_mode, head=head, pair_mode=pair_mode,
                                       name=f"{cohort}/{family}/{window_mode}_{head}_{pair_mode}"))
    return result
