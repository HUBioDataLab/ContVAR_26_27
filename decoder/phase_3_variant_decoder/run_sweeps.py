"""
Sweep orchestrator — create, inspect, run, collect.

Design: phase_3_variant_decoder/DESIGN.md §5.3

A single entry point instead of typing `wandb sweep` + `wandb agent` twenty times.
Created sweeps are recorded in `runs/sweeps.json`; the later commands read from
there, so sweep IDs never have to be carried around by hand.

    # 1) create 20 sweeps (no GPU, a few seconds)
    python -m phase_3_variant_decoder.run_sweeps create --preset lite

    # 2) check progress
    python -m phase_3_variant_decoder.run_sweeps status

    # 3) run them all sequentially (foreground, prints progress)
    python -m phase_3_variant_decoder.run_sweeps run --cuda 3

    # 4) persist the best trial of each finished sweep
    python -m phase_3_variant_decoder.run_sweeps collect

`run` finishes the 20 sweeps one by one; if interrupted, running it again skips the
completed ones and continues where it left off.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, SWEEP_INPUT_MODES, WANDB_ENTITY, WANDB_PROJECT,
)

REGISTRY = Path("phase_3_variant_decoder/runs/sweeps.json")


# ── registry ─────────────────────────────────────────────────────────────────

def load_registry(path) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_registry(entries: list[dict], path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _key(e: dict) -> str:
    return f"{e['family']}/{e['embedding']}/{e['mode']}"


def _project_of(e: dict) -> str:
    """wandb project, parsed out of the stored `path` ("<entity>/<project>/<id>").

    BUG FOUND & FIXED HERE: dedup in cmd_create used to key on (family, embedding,
    mode, preset) alone. That collides across design iterations that reuse the
    same preset name ("full", "lite") in a *different* project — exactly what
    happened when the single-margin revision's sweeps were created: `create
    --preset full` was run twice against two different `--project` values before
    the registry existed for that project, producing TWO independent empty
    sweeps per architecture (40 sweep_ids for 20 architectures), on top of the
    OLD two-margin `full` sweeps already sitting in the registry from weeks
    earlier — none of which `create`'s old key could tell apart. `run --preset
    full` then queried the wandb API for the state of all ~60 matching entries
    serially before starting a single trial: network-bound, near-zero CPU, no
    child process yet, and no run visible — indistinguishable from "stuck".
    Fix: fold `project` into the identity everywhere a registry entry is
    matched, deduped, or filtered.
    """
    parts = e.get("path", "").split("/")
    return parts[1] if len(parts) >= 3 else ""


# ── create ───────────────────────────────────────────────────────────────────

def cmd_create(args):
    import wandb

    sweep_dir = Path(args.sweep_dir) / args.preset
    if not sweep_dir.is_dir():
        raise SystemExit(
            f"[ERROR] {sweep_dir} does not exist. Generate the YAMLs first:\n"
            f"       python -m phase_3_variant_decoder.make_sweeps --preset {args.preset}"
        )

    entries = load_registry(args.registry)
    existing = {(_key(e), e["preset"], _project_of(e)) for e in entries}
    created = skipped = 0

    for mode in args.input_modes:
        for family in args.families:
            for emb in args.embeddings:
                path = sweep_dir / f"{family}_{emb}_{mode}.yaml"
                if not path.exists():
                    print(f"  [SKIP] no YAML: {path.name}")
                    continue
                tag = f"{family}/{emb}/{mode}"
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                project = cfg.get("project", args.project)
                if (tag, args.preset, project) in existing and not args.force:
                    print(f"  [HAVE] {tag:<40s} ({project}) already registered "
                          f"(use --force to recreate)")
                    skipped += 1
                    continue

                sweep_id = wandb.sweep(cfg, project=project, entity=args.entity)
                entity = args.entity or wandb.Api().default_entity
                entries.append({
                    "family": family, "embedding": emb, "mode": mode,
                    "preset": args.preset,
                    "sweep_id": sweep_id,
                    "path": f"{entity}/{cfg.get('project', args.project)}/{sweep_id}",
                    "run_cap": cfg.get("run_cap"),
                    "yaml": str(path),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                save_registry(entries, args.registry)   # write each step — nothing lost if interrupted
                print(f"  [NEW ] {tag:<40s} {sweep_id}")
                created += 1

    print(f"\n[DONE] {created} new · {skipped} existing · registry -> {args.registry}")


# ── status ───────────────────────────────────────────────────────────────────

def _sweep_state(path: str) -> tuple[int, int, float | None]:
    """(finished trials, total trials, best metric)"""
    import wandb

    try:
        sweep = wandb.Api().sweep(path)
    except Exception:
        return -1, -1, None
    runs = list(sweep.runs)
    done = sum(1 for r in runs if r.state == "finished")
    best = None
    for r in runs:
        v = r.summary.get("best_macro_f1", r.summary.get("val_macro_f1"))
        if isinstance(v, (int, float)) and (best is None or v > best):
            best = float(v)
    return done, len(runs), best


def cmd_status(args):
    entries = [e for e in load_registry(args.registry)
               if (not args.preset or e["preset"] == args.preset)
               and (not args.project or _project_of(e) == args.project)]
    if not entries:
        raise SystemExit("[ERROR] registry is empty — run `create` first")

    print(f"{'family':<6s} {'embedding':<15s} {'mode':<7s} {'preset':<6s} "
          f"{'trials':>10s} {'best':>9s}  sweep_id")
    print("─" * 88)
    pending = 0
    for e in entries:
        if args.no_remote:
            done, total, best = -1, -1, None
        else:
            done, total, best = _sweep_state(e["path"])
        cap = e.get("run_cap") or "?"
        prog = "—" if done < 0 else f"{done}/{cap}"
        if done < 0 or (isinstance(cap, int) and done < cap):
            pending += 1
        print(f"{e['family']:<5s} {e['embedding']:<15s} {e['mode']:<7s} "
              f"{e['preset']:<6s} {prog:>10s} "
              f"{('—' if best is None else f'{best:.4f}'):>9s}  {e['sweep_id']}")
    print(f"\n  {len(entries)} sweeps · {pending} incomplete")


# ── launch ───────────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run the sweeps SEQUENTIALLY — in the foreground, printing progress.

    No parallelism, no tmux orchestration: you start it in your own pane and it
    finishes the 20 sweeps one at a time. Same pattern as build_pair_embeddings.
    """
    entries = [e for e in load_registry(args.registry)
               if (not args.preset or e["preset"] == args.preset)
               and (not args.project or _project_of(e) == args.project)]
    if not entries:
        raise SystemExit("[ERROR] registry is empty — run `create` first")

    if not args.include_done:
        pending = []
        for e in entries:
            done, _, _ = _sweep_state(e["path"])
            cap = e.get("run_cap")
            if done < 0 or not isinstance(cap, int) or done < cap:
                pending.append(e)
        if len(pending) < len(entries):
            print(f"[INFO] skipped {len(entries) - len(pending)} completed sweeps "
                  f"(use --include_done to include them)")
        entries = pending
    if not entries:
        print("[DONE] no incomplete sweeps.")
        return

    env = dict(os.environ)
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # Cap CPU usage. Two layers, since either alone can be silently ignored:
    #   (1) thread-pool env vars — numpy/torch's BLAS/OMP backends read these
    #       ONLY at process start, so they must be set before the child's
    #       interpreter (and therefore torch) is imported, i.e. via env= on
    #       the subprocess, not torch.set_num_threads() inside sweep_train.py.
    #   (2) taskset — an OS-level affinity cap that holds even if some library
    #       ignores the env vars above.
    cpu_cmd = []
    if args.cpus is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(args.cpus)
        if shutil.which("taskset"):
            cpu_cmd = ["taskset", "-c", f"0-{args.cpus - 1}"]
        else:
            print("[WARN] taskset not found — only thread-pool env vars will cap CPU usage")

    total = len(entries)
    print(f"[INFO] running {total} sweeps sequentially"
          + (f" · GPU {args.cuda}" if args.cuda is not None else "")
          + (f" · capped at {args.cpus} CPUs" if args.cpus is not None else ""))
    print(f"[INFO] start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    t0 = time.time()
    failed = []
    for i, e in enumerate(entries, 1):
        head = f"[{i}/{total}] {_key(e)}  ·  {e['sweep_id']}"
        print("=" * 78)
        print(f"  {head}   {time.strftime('%H:%M:%S')}")
        print("=" * 78, flush=True)
        rc = subprocess.run(cpu_cmd + ["wandb", "agent", e["path"]], env=env).returncode
        if rc != 0:
            failed.append((_key(e), rc))
            print(f"\n[WARN] {_key(e)} exited with code {rc} — moving to the next\n")
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(f"\n[{i}/{total}] done · elapsed {elapsed/3600:.1f} h · "
              f"eta {eta/3600:.1f} h\n", flush=True)

    print("=" * 78)
    print(f"[DONE] {total} sweeps · {(time.time()-t0)/3600:.1f} hours total")
    if failed:
        print(f"  {len(failed)} failed: " + ", ".join(f"{k}(rc={r})" for k, r in failed))

    if args.no_collect:
        print("\n  --no_collect passed — skipping the automatic long retrain.")
        print("  next: python -m phase_3_variant_decoder.run_sweeps collect")
        print("        python -m phase_3_variant_decoder.report")
        return

    # A sweep trial only trains for preset['epochs'] (60 for "full") — many
    # architectures' best_epoch lands close to that ceiling (still improving,
    # not plateaued/early-stopped), so the sweep-time checkpoint likely
    # undersells the config. Automatically retrain each architecture's winner
    # for longer (collect's own --epochs default, currently 120) right after
    # the sweep finishes, instead of requiring a separate manual step.
    print(f"\n[INFO] auto-collect: retraining each architecture's best trial "
          f"for {args.collect_epochs} epochs (pass --no_collect to skip this)\n")
    collect_args = argparse.Namespace(
        registry=args.registry, preset=args.preset, project=args.project,
        cuda=(args.cuda or "0"), force=False, dry_run=False,
        epochs=args.collect_epochs, patience=args.collect_patience,
    )
    cmd_collect(collect_args)


# ── collect ──────────────────────────────────────────────────────────────────

def cmd_collect(args):
    entries = [e for e in load_registry(args.registry)
               if (not args.preset or e["preset"] == args.preset)
               and (not args.project or _project_of(e) == args.project)]
    if not entries:
        raise SystemExit("[ERROR] registry is empty")

    for e in entries:
        done, _, best = _sweep_state(e["path"])
        cap = e.get("run_cap")
        if not args.force and (done <= 0 or (isinstance(cap, int) and done < cap)):
            print(f"  [SKIP] {_key(e):<40s} incomplete ({done}/{cap})")
            continue
        cmd = [sys.executable, "-m", "phase_3_variant_decoder.apply_best",
               "--sweep_id", e["path"],
               "--family", e["family"], "--embedding", e["embedding"],
               "--input_mode", e["mode"], "--cuda", args.cuda,
               "--epochs", str(args.epochs), "--patience", str(args.patience)]
        print(f"\n  [COLLECT] {_key(e)}  (best val {best:.4f}, retraining {args.epochs} epochs)" if best
              else f"\n  [COLLECT] {_key(e)}  (retraining {args.epochs} epochs)")
        if args.dry_run:
            print("          " + " ".join(cmd))
            continue
        subprocess.run(cmd, check=False)

    print("\n[DONE] for the report: python -m phase_3_variant_decoder.report")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 3 sweep orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--registry", default=str(REGISTRY),
                        help="sweep registry file (JSON)")
        sp.add_argument("--preset", default=None, help="lite | full (empty = all)")
        sp.add_argument("--families", nargs="+", default=list(FAMILIES))
        sp.add_argument("--embeddings", nargs="+", default=list(EMBEDDING_TYPES))
        sp.add_argument("--input_modes", nargs="+", default=list(SWEEP_INPUT_MODES))

    c = sub.add_parser("create", help="create sweeps from the YAMLs and register them")
    common(c)
    c.set_defaults(preset="lite")
    c.add_argument("--sweep_dir", default="phase_3_variant_decoder/sweeps")
    c.add_argument("--project", default=WANDB_PROJECT)
    c.add_argument("--entity", default=WANDB_ENTITY)
    c.add_argument("--force", action="store_true", help="recreate even if already registered")
    c.set_defaults(func=cmd_create)

    s = sub.add_parser("status", help="sweep progress")
    common(s)
    s.add_argument("--project", default=None,
                   help="only this wandb project (recommended once >1 design "
                        "iteration shares the registry — see _project_of's docstring)")
    s.add_argument("--no_remote", action="store_true", help="do not query wandb, just print the registry")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("run", help="run the sweeps sequentially (foreground)")
    common(r)
    r.add_argument("--project", default=None,
                   help="only this wandb project (recommended once >1 design "
                        "iteration shares the registry — see _project_of's docstring)")
    r.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 3")
    r.add_argument("--include_done", action="store_true",
                   help="also re-run completed sweeps")
    r.add_argument("--cpus", type=int, default=None,
                   help="cap each trial to N CPUs (env thread limits + taskset)")
    r.add_argument("--no_collect", action="store_true",
                   help="skip the automatic long retrain of each architecture's winner "
                        "that normally runs right after all sweeps finish")
    r.add_argument("--collect_epochs", type=int, default=120,
                   help="epoch budget for the automatic post-sweep retrain (a sweep trial "
                        "itself only gets the preset's epochs, e.g. 60 for 'full' — many "
                        "architectures' best_epoch lands close to that ceiling, still "
                        "improving, so the final checkpoint is retrained longer)")
    r.add_argument("--collect_patience", type=int, default=20,
                   help="early-stop patience for the automatic post-sweep retrain")
    r.set_defaults(func=cmd_run)

    co = sub.add_parser("collect", help="persist the best trial of finished sweeps")
    common(co)
    co.add_argument("--project", default=None,
                    help="only this wandb project (see _project_of's docstring)")
    co.add_argument("--epochs", type=int, default=120,
                    help="epoch budget for the retrain (passed to apply_best.py)")
    co.add_argument("--patience", type=int, default=20,
                    help="early-stop patience for the retrain (passed to apply_best.py)")
    co.add_argument("--cuda", default="0")
    co.add_argument("--force", action="store_true", help="collect incomplete sweeps too")
    co.add_argument("--dry_run", action="store_true")
    co.set_defaults(func=cmd_collect)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
