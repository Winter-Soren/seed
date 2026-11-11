#!/usr/bin/env python3
"""Run one tracked, resumable Seed evolution experiment.

Every run gets its own folder under runs/ containing:
  config.json        everything needed to reproduce it (args, world/brain config, git commit, machine)
  metrics.jsonl      one line per generation (append-only)
  checkpoint.npz     latest evolution state (atomic write; used by --resume)
  genome_final.npy   the final mean genome
  summary.json       final evaluation in unseen worlds: learning vs frozen (the lesion test)

Examples
  python scripts/run_experiment.py --name smoke --quick
  python scripts/run_experiment.py --name A-blank --start blank --generations 200
  python scripts/run_experiment.py --name B-bio --start biology --sigma 0.05 --lr 0.02
  python scripts/run_experiment.py --name A-blank --generations 200 --max-train-seconds 7200
  python scripts/run_experiment.py --resume runs/20261112-101500_A-blank --max-train-seconds 7200
  python scripts/run_experiment.py --name dense --cfg n_items=32 --cfg decay=0.002
"""
import argparse, datetime, hashlib, json, math, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from seed.seed import (Cfg, init_genome, biology_prior, make_evaluator, es_init, es_step,
                       format_row, test_genome, genome_size)


def git_info():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def genome_hash(x):
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def parse_cfg(pairs):
    base = Cfg()._asdict()
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if k not in base:
            raise SystemExit(f"unknown Cfg field {k!r}; fields: {list(base)}")
        out[k] = type(base[k])(v)
    return out


def start_genome(spec, c):
    if spec == "blank":
        return None
    if spec == "biology":
        return ravel_pytree(biology_prior(c))[0]
    if spec.startswith("file:"):
        return jnp.asarray(np.load(spec[5:]))
    raise SystemExit("--start must be blank, biology or file:<path.npy>")


def save_checkpoint(run_dir, state):
    tmp = run_dir / "checkpoint.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, mu=np.asarray(state["mu"]), m=np.asarray(state["m"]), v=np.asarray(state["v"]),
                 t=np.int64(state["t"]), key=np.asarray(state["key"]))
    os.replace(tmp, run_dir / "checkpoint.npz")


def load_checkpoint(run_dir):
    z = np.load(run_dir / "checkpoint.npz")
    return dict(mu=jnp.asarray(z["mu"]), m=jnp.asarray(z["m"]), v=jnp.asarray(z["v"]),
                t=int(z["t"]), key=jnp.asarray(z["key"], dtype=jnp.uint32))


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--start", default="blank", help="blank | biology | file:<genome.npy>")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--worlds", type=int, default=8, help="worlds per genome per generation (half each rule)")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plastic", action="store_true", help="evolve with all plasticity disabled (control)")
    ap.add_argument("--eval-worlds", type=int, default=128)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--max-train-seconds", type=float, default=None,
                    help="per-invocation training limit; stop after a complete generation and checkpoint "
                         "(excludes setup/evaluation; repeat on --resume)")
    ap.add_argument("--cfg", action="append", help="override a Cfg field, e.g. --cfg n_items=32")
    ap.add_argument("--hypothesis", default="", help="one line: what this run tests")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--resume", default=None, help="path to an existing run folder")
    ap.add_argument("--allow-dirty", action="store_true", help="run even with uncommitted code changes")
    a = ap.parse_args()
    if a.max_train_seconds is not None and (not math.isfinite(a.max_train_seconds) or a.max_train_seconds <= 0):
        ap.error("--max-train-seconds must be positive and finite")

    git = git_info()
    if git["dirty"] and not (a.allow_dirty or (a.quick and not a.resume)):
        raise SystemExit("Uncommitted code changes. Commit first (every run must map to a commit), "
                         "or pass --allow-dirty and say why in the experiment log.")
    if a.resume:
        run_dir = Path(a.resume)
        config = json.loads((run_dir / "config.json").read_text())
        args = config["args"]
        config.setdefault("resumes", []).append(dict(at=datetime.datetime.now().isoformat(), git=git,
                                                     max_train_seconds=a.max_train_seconds))
    else:
        args = vars(a).copy()
        if a.quick:
            args.update(generations=3, pop=8, worlds=2, eval_worlds=4, checkpoint_every=1)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        config = dict(args=args, cfg_overrides=parse_cfg(a.cfg), git=git,
                      created=datetime.datetime.now().isoformat(),
                      machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                                   jax=jax.__version__, devices=[str(d) for d in jax.devices()]))

    c = Cfg(**config["cfg_overrides"])
    config["cfg"] = c._asdict()
    config["genome_size"] = genome_size(c)
    plastic = not args["no_plastic"]
    evaluate, _ = make_evaluator(c, args["worlds"], plastic)

    if (run_dir / "checkpoint.npz").exists():
        state = load_checkpoint(run_dir)
        print(f"resuming {run_dir.name} at generation {state['t']}")
    else:
        mean = start_genome(args["start"], c)
        state = es_init(c, mean, args["seed"])
        config["start_genome_hash"] = genome_hash(state["mu"])
        np.save(run_dir / "genome_start.npy", np.asarray(state["mu"]))
    write_json(run_dir / "config.json", config)

    summary_path = run_dir / "summary.json"
    status, t0 = "running", time.monotonic()
    write_json(summary_path, dict(status=status, generation=state["t"]))
    try:
        with open(run_dir / "metrics.jsonl", "a") as mf:
            while state["t"] < args["generations"]:
                g0 = time.monotonic()
                state, row = es_step(state, evaluate, args["pop"], args["worlds"], args["sigma"], args["lr"])
                row["seconds"] = round(time.monotonic() - g0, 3)
                mf.write(json.dumps(row) + "\n"); mf.flush()
                if row["gen"] % 10 == 0 or state["t"] == args["generations"]:
                    print(format_row(row) + f" | {row['seconds']:.1f}s/gen", flush=True)
                if state["t"] % args["checkpoint_every"] == 0:
                    save_checkpoint(run_dir, state)
                if (a.max_train_seconds is not None and state["t"] < args["generations"]
                        and time.monotonic() - t0 >= a.max_train_seconds):
                    status = "interrupted"
                    print("training time limit reached; checkpointing, resume with --resume", run_dir)
                    break
        if state["t"] >= args["generations"]:
            status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        print("interrupted; checkpoint saved, resume with --resume", run_dir)
    save_checkpoint(run_dir, state)

    summary = dict(status=status, generation=state["t"], train_minutes=round((time.monotonic() - t0) / 60, 2),
                   final_genome_hash=genome_hash(state["mu"]))
    if status == "finished":
        np.save(run_dir / "genome_final.npy", np.asarray(state["mu"]))
        n = args["eval_worlds"]
        start = jnp.asarray(np.load(run_dir / "genome_start.npy"))
        summary["eval_worlds"] = n
        summary["final_learning"] = test_genome(state["mu"], c, n_worlds=n, plastic=True)
        summary["final_frozen"] = test_genome(state["mu"], c, n_worlds=n, plastic=False)
        summary["start_learning"] = test_genome(start, c, n_worlds=n, plastic=True)
        for k in ("final_learning", "final_frozen", "start_learning"):
            r = summary[k]
            print(f"{k:15s} fitness {r['fitness']:6.2f} | acc {r['acc_first']:.2f} -> {r['acc_second']:.2f}")
    write_json(summary_path, summary)
    print("run folder:", run_dir)


if __name__ == "__main__":
    main()
