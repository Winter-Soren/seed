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
import argparse, datetime, hashlib, io, json, math, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from seed.seed import (Cfg, init_genome, biology_prior, make_evaluator, es_init, es_step,
                       format_row, test_genome, genome_size)
from seed.structural import MODE, INTERVAL, PRUNE_FRACTION, STREAM_TAG, StructuralValidationError
from seed import direct, paired
from scripts import broader_eval, reversal_eval, eval_v3, eval_v4, eval_v5

DIRECT_PROTOCOL = "direct-v1"
PAIRED_PROTOCOL = "paired-train-v1"
ECOLOGY_LINEAGE_PROTOCOL = "ecology-lineage-v1"
PAIRED_OBJECTIVES = ("paired-mean", "paired-min")
V4_MAX_SECONDS = 1740
V5_MAX_SECONDS = 1140
V4_SOURCES = (
    dict(label="A0", controller="rules-v0", seed=0, run="runs/20261112-023245_E103-A-blank-s0",
         sha256="477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"),
    dict(label="A1", controller="rules-v0", seed=1, run="runs/20261112-050319_E103-A-blank-s1",
         sha256="35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899"),
    dict(label="A2", controller="rules-v0", seed=2, run="runs/20261112-102755_E103-A-blank-s2",
         sha256="f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6"),
    dict(label="direct0", controller=direct.FORMAT, seed=0, run="runs/20261119-033803_E113-direct-s0",
         sha256="0a47a3dbbb7482a42ef069cf6b54b44e511ffaf2d169d92914c10fdf75a4bb61"),
    dict(label="direct1", controller=direct.FORMAT, seed=1, run="runs/20261119-040602_E113-direct-s1",
         sha256="0e77c3083a88c8216e57bff0667b5f594542447c14f6ae7338ec0e090b9d7d3b"),
    dict(label="direct2", controller=direct.FORMAT, seed=2, run="runs/20261119-041609_E113-direct-s2",
         sha256="473124dfe1c2a0d77a9a4b7cdde5cb36c41f61b0664e17ede86ae88502971bbd"),
)
V4_REFERENCES = ("fixed_A", "fixed_B", "mixture", "random_walk", "greedy")


def git_info():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise SystemExit(f"Cannot verify Git state: {error}") from error
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain", "--untracked-files=no")
                           or run("ls-files", "--others", "--exclude-standard", "--", "seed/", "scripts/")))


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


def validate_eval_protocol(args, cfg_overrides):
    protocol = args.get("eval_protocol", "v1")
    if protocol not in ("v1", "v2", "structural-v1", broader_eval.PROTOCOL, DIRECT_PROTOCOL, PAIRED_PROTOCOL):
        raise SystemExit(f"Unknown evaluation protocol: {protocol!r}")
    if args.get("objective") is not None and protocol != PAIRED_PROTOCOL:
        raise SystemExit("--objective is only valid for paired-train-v1")
    controller = args.get("controller") or "rules-v0"
    if controller not in ("rules-v0", direct.FORMAT):
        raise SystemExit(f"Unknown controller: {controller!r}")
    if (controller == direct.FORMAT) != (protocol == DIRECT_PROTOCOL):
        raise SystemExit("direct-rnn-v1 requires --eval-protocol direct-v1 and vice versa")
    structural = args.get("structural_plasticity", "off")
    if structural not in ("off", MODE):
        raise SystemExit(f"Unknown structural mode: {structural!r}")
    if structural != "off" and args["no_plastic"]:
        raise SystemExit("structural training conflicts with --no-plastic")
    if (structural != "off") != (protocol == "structural-v1"):
        raise SystemExit("structural mode requires --eval-protocol structural-v1 and vice versa")
    if protocol == "v2" and (args["generations"] != 0 or not args["start"].startswith("file:")
                             or args["eval_worlds"] != 128 or cfg_overrides
                             or args.get("quick") or args["no_plastic"]):
        raise SystemExit("EVAL v2 requires a saved file: genome, zero generations, 128 evaluation worlds, "
                         "default Cfg, and no --quick or --no-plastic")
    if protocol == DIRECT_PROTOCOL:
        c = Cfg(**cfg_overrides)
        split = args.get("condition_split") or "final"
        if (not args["no_plastic"] or args["start"] != "blank" or args.get("quick")
                or structural != "off" or split not in ("development", "final")):
            raise SystemExit("direct-v1 requires blank start, --no-plastic, no quick/structural changes, "
                             "and a final or development split")
        if (any(not math.isfinite(value) for value in c) or c.G < 1 or c.M < 1 or c.R < 0
                or c.H < 1 or c.n_htypes < 1 or c.H % c.n_htypes or c.T < 4 or c.T % 2
                or not 0 <= 2*c.n_items <= c.G*c.G or args["pop"] < 2 or args["pop"] % 2
                or args["worlds"] < 1 or args["checkpoint_every"] < 1
                or not math.isfinite(args["sigma"]) or args["sigma"] <= 0
                or not math.isfinite(args["lr"]) or args["lr"] <= 0):
            raise SystemExit("direct-v1 requires finite, valid world and optimizer settings")
        if split == "final":
            if (cfg_overrides or c != Cfg() or args["eval_worlds"] != 256
                    or args["generations"] != 200 or args["seed"] not in (0, 1, 2)
                    or args["pop"] != 128 or args["worlds"] != 8
                    or args["sigma"] != .1 or args["lr"] != .1):
                raise SystemExit("direct-v1 final fixes default Cfg, 256 evaluation worlds, 200 generations, "
                                 "seed 0/1/2, population 128, eight training worlds, sigma/lr 0.1/0.1")
        elif (c.T > 30 or c.M > 2 or c.H > 6 or c.R > 1 or c.G > 8
              or args["pop"] > 8 or args["worlds"] > 4 or not 0 <= args["generations"] <= 4
              or not 2 <= args["eval_worlds"] <= 4):
            raise SystemExit("direct-v1 development requires tiny fixtures: T<=30, M<=2, H<=6, R<=1, "
                             "G<=8, population<=8, worlds<=4, generations<=4, evaluation worlds 2..4")
    if protocol == PAIRED_PROTOCOL:
        validate_paired_args(args, Cfg(**cfg_overrides), cfg_overrides)
    return protocol


def validate_paired_args(args, c, overrides):
    split = args.get("condition_split") or "final"
    if (args.get("objective") not in PAIRED_OBJECTIVES or args.get("controller", "rules-v0") != "rules-v0"
            or args["no_plastic"] or args.get("quick") or args.get("allow_dirty")
            or (args.get("structural_plasticity") or "off") != "off" or not args["start"].startswith("file:")
            or split not in ("final", "development")):
        raise SystemExit("paired-train-v1 requires an explicit paired objective, saved rules-v0 source, "
                         "both learning rules, clean code, and no quick or structural changes")
    if (any(not math.isfinite(value) for value in c) or c.G < 1 or c.M < 1 or c.R < 0
            or c.H < 1 or c.n_htypes != 3 or c.H % c.n_htypes or c.T < 4 or c.T % 2
            or c.micro < 1 or min(c.e0, c.e_max, c.wmax) <= 0 or c.e0 > c.e_max or min(c.decay, c.gain) < 0
            or not 0 <= 2*c.n_items <= c.G*c.G or args["pop"] < 2 or args["pop"] % 2
            or args["worlds"] < 1 or args["checkpoint_every"] < 1
            or not math.isfinite(args["sigma"]) or args["sigma"] <= 0
            or not math.isfinite(args["lr"]) or args["lr"] <= 0):
        raise SystemExit("paired-train-v1 requires finite valid settings and the 155-number genome format")
    if split == "final":
        if (overrides or c != Cfg() or args["generations"] != 50 or args["seed"] not in (1000, 1001, 1002)
                or (args["pop"], args["worlds"], args["sigma"], args["lr"], args["checkpoint_every"], args["eval_worlds"])
                != (64, 4, .1, .1, 10, 256)):
            raise SystemExit("paired-train-v1 final fixes 50 generations, seeds 1000/1001/1002, population 64, "
                             "four base-world pairs, sigma/lr 0.1, checkpoint 10, default Cfg and deferred evaluation")
        arm = args["objective"].removeprefix("paired-")
        if args["name"] != f"E115-{arm}-s{args['seed'] - 1000}":
            raise SystemExit("paired-train-v1 final requires its registered canonical run name")
    elif (c.T > 30 or c.M > 2 or c.H > 6 or c.R > 1 or c.G > 8 or args["seed"] != 41
          or args["pop"] > 4 or args["worlds"] > 4 or not 1 <= args["generations"] <= 4
          or not 2 <= args["eval_worlds"] <= 4):
        raise SystemExit("paired-train-v1 development requires tiny fixtures, seed 41, at most four generations/population/worlds")
    if not args["name"] or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in args["name"]):
        raise SystemExit("paired-train-v1 run names require letters, numbers, dots, hyphens or underscores")


def paired_source(args, c):
    path = (ROOT / args["start"][5:]).resolve()
    data = path.read_bytes()
    flat = diagnostic_genome(data, c, protocol=PAIRED_PROTOCOL)
    digest = hashlib.sha256(data).hexdigest()
    if args["condition_split"] == "final":
        source = V4_SOURCES[args["seed"] - 1000]
        if (path != (ROOT / source["run"] / "genome_final.npy").resolve()
                or digest != source["sha256"]):
            raise SystemExit("paired-train-v1 requires the pinned original A genome matching its seed")
    try:
        recorded_path = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        recorded_path = str(path)
    metadata = dict(path=recorded_path, sha256=digest, genome_hash=genome_hash(flat), controller="rules-v0")
    if args["condition_split"] == "final":
        provenance = load_saved_sources(c, (source,))[source["label"]]["provenance"]
        metadata.update(source_run=source["run"], source_config_sha256=provenance["config_sha256"],
                        source_summary_sha256=provenance["summary_sha256"], source_commits=provenance["source_commits"])
    return flat, data, metadata


def paired_initial_optimizer(c, flat, seed):
    state = es_init(c, flat, seed)
    if state["t"] != 0 or np.any(state["m"]) or np.any(state["v"]):
        raise ValueError("paired training must start with fresh zero Adam moments")
    return dict(method="es_init", generation=0, adam_m_zero=True, adam_v_zero=True,
                key=np.asarray(state["key"]).tolist(), l2=.005)


def diagnostic_genome(data, c, *, protocol="EVAL v2"):
    mean = np.load(io.BytesIO(data), allow_pickle=False)
    if (not isinstance(mean, np.ndarray) or mean.shape != (genome_size(c),)
            or not np.issubdtype(mean.dtype, np.floating)):
        raise SystemExit(f"{protocol} requires a flat floating-point genome of the expected size")
    mean = np.asarray(mean, dtype=np.float32)
    if not np.isfinite(mean).all():
        raise SystemExit(f"{protocol} requires a finite genome")
    return mean


def save_checkpoint(run_dir, state):
    tmp = run_dir / "checkpoint.tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, mu=np.asarray(state["mu"]), m=np.asarray(state["m"]), v=np.asarray(state["v"]),
                 t=np.int64(state["t"]), key=np.asarray(state["key"]))
    os.replace(tmp, run_dir / "checkpoint.npz")


def validate_es_state(state, size, controller):
    for field in ("mu", "m", "v"):
        value = np.asarray(state[field])
        if (value.shape != (size,) or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()):
            raise ValueError(f"{controller} {field} must be a finite floating-point vector of the expected size")
    key, generation = np.asarray(state["key"]), np.asarray(state["t"])
    if (key.shape != (2,) or key.dtype != np.uint32 or generation.shape != ()
            or not np.issubdtype(generation.dtype, np.integer) or int(generation) < 0
            or np.any(np.asarray(state["v"]) < 0)):
        raise ValueError(f"{controller} requires a uint32 RNG key, nonnegative integer generation and Adam variance")


def validate_direct_state(state, c):
    validate_es_state(state, direct.genome_size(c), direct.FORMAT)


def load_checkpoint(run_dir, *, direct_cfg=None, paired_cfg=None):
    with np.load(run_dir / "checkpoint.npz", allow_pickle=False) as z:
        if direct_cfg is not None:
            try:
                validate_direct_state(z, direct_cfg)
            except (ValueError, KeyError) as error:
                raise SystemExit(f"Cannot resume: {error}") from error
        if paired_cfg is not None:
            try:
                validate_es_state(z, genome_size(paired_cfg), PAIRED_PROTOCOL)
            except (ValueError, KeyError) as error:
                raise SystemExit(f"Cannot resume: {error}") from error
        return dict(mu=jnp.asarray(z["mu"]), m=jnp.asarray(z["m"]), v=jnp.asarray(z["v"]),
                    t=int(z["t"]), key=jnp.asarray(z["key"], dtype=jnp.uint32))


def direct_weight_stats(flat, c):
    weights = np.asarray(flat)
    return dict(max_abs=float(np.abs(weights).max()), fraction_abs_above_wmax=float((np.abs(weights) > c.wmax).mean()),
                comparison_wmax=c.wmax, clipping=False)


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def run_broader(a, git):
    c = Cfg(**parse_cfg(a.cfg))
    broader_eval.validate_request(a, c)
    flat, source_bytes, provenance = broader_eval.load_saved_genome(a.start, c)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(name=broader_eval.PROTOCOL, split=a.condition_split,
                    seed=broader_eval.SPLIT_SEEDS[a.condition_split], n_worlds=a.eval_worlds,
                    rule_counts={"0": (a.eval_worlds + 1) // 2, "1": a.eval_worlds // 2},
                    conditions=broader_eval.CONDITIONS,
                    plasticity={"learning": True, "frozen": False})
    config = dict(args=vars(a).copy(), cfg_overrides=parse_cfg(a.cfg), cfg=c._asdict(), git=git,
                  created=datetime.datetime.now().isoformat(), eval_protocol=protocol,
                  input_genome=provenance, start_genome_hash=genome_hash(flat),
                  genome_size=flat.size,
                  machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                               jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
    write_json(run_dir / "config.json", config)
    (run_dir / "genome_start.npy").write_bytes(source_bytes)
    summary = dict(status="running", generation=0, eval_protocol=broader_eval.PROTOCOL,
                   condition_split=a.condition_split, eval_seed=protocol["seed"],
                   eval_worlds=a.eval_worlds, input_genome=provenance,
                   final_genome_hash=genome_hash(flat))
    write_json(run_dir / "summary.json", summary)
    started = time.monotonic()
    try:
        summary["condition_evaluations"] = broader_eval.evaluate_conditions(
            jnp.asarray(flat), c, a.eval_worlds, a.condition_split)
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}", eval_seconds=time.monotonic() - started)
        write_json(run_dir / "summary.json", summary)
        raise
    summary.update(status="finished", eval_seconds=time.monotonic() - started)
    (run_dir / "genome_final.npy").write_bytes(source_bytes)
    write_json(run_dir / "summary.json", summary)
    print(f"{broader_eval.PROTOCOL} {a.condition_split}: {a.eval_worlds} worlds; {summary['eval_seconds']:.3f}s")
    for label, row in summary["condition_evaluations"].items():
        print(f"{label:10s} learning {row['learning']['fitness']:.6f} frozen {row['frozen']['fitness']:.6f} gap {row['learning_minus_frozen_fitness']:.6f}")
    print("run folder:", run_dir)


def run_reversal(a, git):
    c = Cfg(**parse_cfg(a.cfg))
    reversal_eval.validate_request(a, c)
    source = Path(a.start[5:]).read_bytes()
    flat = diagnostic_genome(source, c, protocol="REVERSAL v1")
    source_hash = hashlib.sha256(source).hexdigest()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(args=vars(a).copy(), cfg=c._asdict(), cfg_overrides=parse_cfg(a.cfg), git=git,
                  created=datetime.datetime.now().isoformat(), eval_protocol=reversal_eval.PROTOCOL,
                  eval_seed=reversal_eval.SPLIT_SEEDS[a.condition_split], source_genome_sha256=source_hash,
                  boundary=c.T//2, recovery_window=reversal_eval.WINDOW,
                  recovery_min_encounters=reversal_eval.MIN_ENCOUNTERS,
                  recovery_accuracy=reversal_eval.ACCURACY, genome_size=int(flat.size),
                  machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                               jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
    write_json(run_dir / "config.json", config)
    (run_dir / "genome_start.npy").write_bytes(source)
    summary = dict(status="running", generation=0, eval_protocol=reversal_eval.PROTOCOL,
                   condition_split=a.condition_split, eval_seed=config["eval_seed"], eval_worlds=a.eval_worlds,
                   boundary=c.T//2, source_genome_sha256=source_hash, final_genome_hash=genome_hash(flat))
    write_json(run_dir / "summary.json", summary)
    started = time.monotonic()
    try:
        summary["evaluations"] = reversal_eval.evaluate_conditions(flat, c, a.eval_worlds, a.condition_split)
        ev = summary["evaluations"]
        fitness = lambda name: ev[name]["metrics"]["post_fitness"]
        summary["post_fitness_differences"] = dict(
            noflip_plasticity=fitness("noflip_learning")-fitness("noflip_frozen"),
            flip_plasticity=fitness("flip_learning")-fitness("flip_frozen"),
            reversal_learning=fitness("flip_learning")-fitness("noflip_learning"),
            reversal_frozen=fitness("flip_frozen")-fitness("noflip_frozen"))
        json.dumps(summary, allow_nan=False)
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}", eval_seconds=time.monotonic()-started)
        write_json(run_dir / "summary.json", summary)
        raise
    summary.update(status="finished", eval_seconds=time.monotonic()-started)
    (run_dir / "genome_final.npy").write_bytes(source)
    write_json(run_dir / "summary.json", summary)
    for name, row in summary["evaluations"].items():
        metric = row["metrics"]
        print(f"{name}: post fitness {metric['post_fitness']}, window hits {metric['recovered_agents']}/{metric['agents_at_risk']} at risk")
    print("run folder:", run_dir)



def run_v3(a, git):
    c = Cfg(**parse_cfg(a.cfg))
    eval_v3.validate_request(a, c)
    source = Path(a.start[5:]).read_bytes()
    flat = diagnostic_genome(source, c, protocol="EVAL v3")
    source_hash = hashlib.sha256(source).hexdigest()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(args=vars(a).copy(), cfg=c._asdict(), cfg_overrides={}, git=git,
                  created=datetime.datetime.now().isoformat(), eval_protocol=eval_v3.PROTOCOL,
                  eval_seed=eval_v3.EVAL_SEED, eval_worlds=a.eval_worlds,
                  source_genome_sha256=source_hash, boundary=150, genome_size=int(flat.size),
                  machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                               jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
    write_json(run_dir / "config.json", config)
    (run_dir / "genome_start.npy").write_bytes(source)
    summary = dict(status="running", generation=0, eval_protocol=eval_v3.PROTOCOL,
                   eval_seed=eval_v3.EVAL_SEED, eval_worlds=a.eval_worlds, boundary=150,
                   source_genome_sha256=source_hash, final_genome_hash=genome_hash(flat))
    write_json(run_dir / "summary.json", summary)
    started = time.monotonic()
    try:
        evaluations, raw = eval_v3.evaluate_conditions(flat, c, a.eval_worlds, eval_v3.EVAL_SEED)
        # Save the observations before serialization validation so failures retain their evidence.
        np.savez_compressed(run_dir / "events.npz", **raw)
        summary["evaluations"] = evaluations
        json.dumps(summary, allow_nan=False)
    except BaseException as error:
        summary.pop("evaluations", None)
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}", eval_seconds=time.monotonic()-started)
        write_json(run_dir / "summary.json", summary)
        raise
    summary.update(status="finished", eval_seconds=time.monotonic()-started)
    (run_dir / "genome_final.npy").write_bytes(source)
    write_json(run_dir / "summary.json", summary)
    for condition, controllers in evaluations.items():
        for controller, metric in controllers.items():
            print(f"{condition}/{controller}: fitness {metric['fitness']:.6f} +/- {metric['fitness_se']:.6f} SE")
    print("run folder:", run_dir)


def validate_v4_request(a, c):
    if (a.generations != 0 or a.start != "blank" or a.controller is not None or a.no_plastic
            or a.quick or a.resume or a.structural_plasticity not in (None, "off")
            or a.max_train_seconds is not None or a.allow_dirty or a.seed != 0 or getattr(a, "objective", None) is not None
            or (a.pop, a.worlds, a.sigma, a.lr, a.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("EVAL v4 fixes its controller manifest; no start/controller/training overrides or resume are allowed")
    if not a.name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in a.name):
        raise ValueError("EVAL v4 run names require letters, numbers, dots, hyphens or underscores")
    if a.condition_split == "final":
        if a.cfg or c != Cfg() or a.eval_worlds != eval_v4.N_WORLDS:
            raise ValueError("EVAL v4 final fixes default Cfg and 256 worlds, with no Cfg overrides")
    elif a.condition_split == "development":
        if (any(not math.isfinite(value) for value in c) or not 4 <= c.T <= 30 or c.T % 2
                or not 2 <= c.M <= 4 or c.M % 2 or not 1 <= c.H <= 6 or c.n_htypes < 1 or c.H % c.n_htypes
                or not 0 <= c.R <= 1 or not 1 <= c.G <= 8 or not 0 <= 2*c.n_items <= c.G*c.G
                or not 2 <= a.eval_worlds <= 4):
            raise ValueError("EVAL v4 development requires tiny valid even-lifetime/family fixtures and 2..4 worlds")
    else:
        raise ValueError("EVAL v4 requires a final or development split")


def load_v4_sources(c):
    if tuple(source["label"] for source in V4_SOURCES) != ("A0", "A1", "A2", "direct0", "direct1", "direct2"):
        raise ValueError("EVAL v4 source manifest must contain all six controllers in registered order")
    return load_saved_sources(c, V4_SOURCES)


def load_saved_sources(c, manifest):
    loaded = {}
    for source in manifest:
        folder = ROOT / source["run"]
        data = (folder / "genome_final.npy").read_bytes()
        config_data, summary_data = (folder / "config.json").read_bytes(), (folder / "summary.json").read_bytes()
        config, summary = json.loads(config_data), json.loads(summary_data)
        args = config["args"]
        is_direct = source["controller"] == direct.FORMAT
        expected_size = direct.genome_size(c) if is_direct else genome_size(c)
        flat = np.load(io.BytesIO(data), allow_pickle=False)
        if (source["controller"] not in ("rules-v0", direct.FORMAT)
                or hashlib.sha256(data).hexdigest() != source["sha256"]
                or not isinstance(flat, np.ndarray) or flat.shape != (expected_size,)
                or not np.issubdtype(flat.dtype, np.floating) or not np.isfinite(flat).all()):
            raise ValueError(f"EVAL v4 source {source['label']} has invalid bytes, format, shape or values")
        if (config.get("cfg") != c._asdict() or config.get("genome_size") != expected_size
                or (args.get("controller") or "rules-v0") != source["controller"]
                or args.get("seed") != source["seed"] or args.get("start") != "blank"
                or args.get("generations") != 200 or args.get("no_plastic") is not is_direct
                or args.get("quick") or args.get("allow_dirty") or (args.get("structural_plasticity") or "off") != "off"
                or (args.get("pop"), args.get("worlds"), args.get("sigma"), args.get("lr")) != (128, 8, .1, .1)
                or summary.get("status") != "finished" or summary.get("generation") != 200
                or summary.get("final_genome_hash") != genome_hash(flat)):
            raise ValueError(f"EVAL v4 source {source['label']} has inconsistent completed-run provenance")
        if is_direct and (config.get("controller") != direct.FORMAT
                          or config.get("encoding", {}).get("format") != direct.FORMAT
                          or config.get("encoding", {}).get("weights") != expected_size
                          or args.get("eval_protocol") != DIRECT_PROTOCOL or args.get("condition_split") != "final"):
            raise ValueError(f"EVAL v4 source {source['label']} has inconsistent direct format metadata")
        provenance = [config.get("git", {})] + [entry.get("git", {}) for entry in config.get("resumes", [])]
        for git in provenance:
            commit = git.get("commit", "")
            if (git.get("dirty") is not False or len(commit) != 40
                    or any(ch not in "0123456789abcdef" for ch in commit)):
                raise ValueError(f"EVAL v4 source {source['label']} lacks clean committed provenance")
        loaded[source["label"]] = dict(flat=np.asarray(flat, dtype=np.float32), data=data,
            provenance=dict(source, genome_hash=genome_hash(flat), genome_size=expected_size,
                            config_sha256=hashlib.sha256(config_data).hexdigest(),
                            summary_sha256=hashlib.sha256(summary_data).hexdigest(), source_commits=provenance))
    return loaded


def load_v5_sources(c, split):
    originals = load_saved_sources(c, V4_SOURCES[:3])
    loaded = {}
    shared_budget = None
    for index in range(3):
        original = originals[f"A{index}"]
        loaded[f"start{index}"] = dict(original, provenance=dict(original["provenance"], label=f"start{index}"))
        for arm in ("mean", "min"):
            name = f"E115-{arm}-s{index}"
            folders = list((ROOT / "runs").glob(f"*_{name}"))
            if len(folders) != 1:
                raise ValueError(f"EVAL v5 requires exactly one registered {name} run, including failed attempts")
            folder = folders[0]
            config_data, summary_data = (folder / "config.json").read_bytes(), (folder / "summary.json").read_bytes()
            config, summary = json.loads(config_data), json.loads(summary_data)
            args = config["args"]
            if (args.get("eval_protocol") != PAIRED_PROTOCOL or args.get("objective") != "paired-" + arm
                    or args.get("name") != name or args.get("condition_split") != split
                    or config.get("cfg") != c._asdict() or Cfg(**config["cfg_overrides"]) != c
                    or config.get("controller") != "rules-v0"
                    or config.get("genome_size") != 155 or config.get("objective") != args["objective"]
                    or config.get("eval_protocol") != PAIRED_PROTOCOL or config.get("evaluation_status") != "deferred"):
                raise ValueError(f"EVAL v5 source {name} has inconsistent training metadata")
            validate_paired_args(args, c, config["cfg_overrides"])
            budget = tuple(args[key] for key in ("generations", "pop", "worlds", "sigma", "lr", "checkpoint_every", "eval_worlds"))
            if shared_budget is None:
                shared_budget = budget
            if budget != shared_budget or (split == "final" and args["seed"] != 1000 + index):
                raise ValueError("EVAL v5 continuation budgets and mapped seeds must match")
            source_flat, source_bytes, source_metadata = paired_source(args, c)
            if (source_bytes != original["data"] or config.get("input_genome") != source_metadata
                    or (folder / "genome_start.npy").read_bytes() != source_bytes
                    or config.get("start_genome_hash") != genome_hash(source_flat)
                    or config.get("initial_optimizer") != paired_initial_optimizer(c, source_flat, args["seed"])):
                raise ValueError(f"EVAL v5 source {name} does not preserve its original A input and fresh optimizer")
            data = (folder / "genome_final.npy").read_bytes()
            flat = diagnostic_genome(data, c, protocol="EVAL v5")
            with np.load(folder / "checkpoint.npz", allow_pickle=False) as checkpoint:
                validate_es_state(checkpoint, 155, PAIRED_PROTOCOL)
                if int(checkpoint["t"]) != args["generations"] or not np.array_equal(checkpoint["mu"], flat):
                    raise ValueError(f"EVAL v5 source {name} checkpoint disagrees with its completed genome")
            metrics = (folder / "metrics.jsonl").read_text()
            history = [json.loads(line) for line in metrics.splitlines()]
            if (not metrics.endswith("\n") or len(history) != args["generations"]
                    or any(row.get("gen") != i or any(not math.isfinite(value) for value in row.values())
                           for i, row in enumerate(history))
                    or summary.get("status") != "finished" or summary.get("generation") != args["generations"]
                    or summary.get("training_complete") is not True or summary.get("evaluation_status") != "deferred"
                    or summary.get("objective") != args["objective"] or summary.get("final_genome_hash") != genome_hash(flat)
                    or summary.get("final_genome_sha256") != hashlib.sha256(data).hexdigest()
                    or (folder / "events.npz").exists()):
                raise ValueError(f"EVAL v5 source {name} has incomplete training or unexpected evaluation evidence")
            provenance = [config.get("git", {})] + [entry.get("git", {}) for entry in config.get("resumes", [])]
            if any(git.get("dirty") is not False or len(git.get("commit", "")) != 40
                   or any(ch not in "0123456789abcdef" for ch in git.get("commit", "")) for git in provenance):
                raise ValueError(f"EVAL v5 source {name} lacks clean committed provenance")
            label = arm + str(index)
            loaded[label] = dict(flat=flat, data=data, provenance=dict(
                label=label, controller="rules-v0", seed=index, training_seed=args["seed"], objective=args["objective"],
                run=folder.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest(),
                genome_hash=genome_hash(flat), genome_size=155, input_genome=source_metadata,
                config_sha256=hashlib.sha256(config_data).hexdigest(), summary_sha256=hashlib.sha256(summary_data).hexdigest(),
                source_commits=provenance))
    return loaded


def run_v4(a, git):
    run_paired_diagnostic(a, git, eval_v4)


def run_paired_diagnostic(a, git, module):
    started = time.monotonic()
    c = Cfg(**parse_cfg(a.cfg))
    validate_v4_request(a, c)
    sources = load_v4_sources(c) if module is eval_v4 else load_v5_sources(c, a.condition_split)
    title = "EVAL " + module.PROTOCOL
    expected_panels = len(sources) + len(V4_REFERENCES)
    if any((ROOT / "runs").glob(f"*_{a.name}")):
        raise ValueError(f"An {title} attempt with this name already exists; preserve it and preregister a new attempt")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "inputs").mkdir()
    seed = module.SPLIT_SEEDS[a.condition_split]
    manifest = [dict(value["provenance"], copied_genome=f"inputs/{label}.npy") for label, value in sources.items()]
    config = dict(args=vars(a).copy(), cfg=c._asdict(), git=git, created=datetime.datetime.now().isoformat(),
                  eval_protocol=module.PROTOCOL, eval_seed=seed, eval_worlds=a.eval_worlds,
                  input_genomes=manifest, reference_policies=V4_REFERENCES,
                  internal_seconds_limit=V4_MAX_SECONDS if module is eval_v4 else V5_MAX_SECONDS,
                  machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                               jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
    write_json(run_dir / "config.json", config)
    for label, value in sources.items():
        with (run_dir / "inputs" / f"{label}.npy").open("xb") as copied:
            copied.write(value["data"])
    summary = dict(status="running", generation=0, eval_protocol=module.PROTOCOL, condition_split=a.condition_split,
                   eval_seed=seed, eval_worlds=a.eval_worlds, expected_panels=expected_panels, completed_panels=0, panels={})
    write_json(run_dir / "summary.json", summary)
    raw_by_panel = {}

    def check_deadline():
        if time.monotonic() - started >= (V4_MAX_SECONDS if module is eval_v4 else V5_MAX_SECONDS):
            raise TimeoutError(f"{title} internal time limit reached; completed panels and conditions are preserved")

    try:
        panels = [(label, value["provenance"]["controller"]) for label, value in sources.items()] + [(label, label) for label in V4_REFERENCES]
        for label, controller in panels:
            check_deadline()
            panel_start = time.monotonic()
            panel_dir = run_dir / label
            panel_dir.mkdir()
            entry = dict(status="running", controller=controller, conditions={}, raw_files={})
            summary["panels"][label] = entry
            write_json(run_dir / "summary.json", summary)

            def persist_condition(condition, metrics, arrays):
                if condition not in module.CONDITIONS or condition in entry["conditions"]:
                    raise ValueError(f"Unexpected or duplicate {title} condition")
                path = panel_dir / f"{condition}.npz"
                with path.open("xb") as output:
                    np.savez_compressed(output, **arrays)
                entry["raw_files"][condition] = f"{label}/{path.name}"
                json.dumps(metrics, allow_nan=False)
                entry["conditions"][condition] = metrics
                summary["eval_seconds"] = time.monotonic()-started
                write_json(run_dir / "summary.json", summary)
                check_deadline()

            flat = sources[label]["flat"] if label in sources else None
            metrics, raw = module.evaluate_controller(flat, c, a.eval_worlds, seed,
                                                     controller=controller, on_condition=persist_condition)
            if set(entry["conditions"]) != set(module.CONDITIONS) or metrics["conditions"] != entry["conditions"]:
                raise ValueError(f"{title} controller returned incomplete or inconsistent condition records")
            json.dumps(metrics, allow_nan=False)
            with (panel_dir / "paired.npz").open("xb") as output:
                np.savez_compressed(output, **{key: value for key, value in raw.items() if key.startswith("paired_")})
            entry.update(metrics, status="finished", panel_seconds=time.monotonic()-panel_start)
            entry["raw_files"]["paired"] = f"{label}/paired.npz"
            raw_by_panel[label] = raw
            summary["completed_panels"] += 1
            if summary["completed_panels"] == 1:
                summary["first_panel_seconds"] = entry["panel_seconds"]
            summary["eval_seconds"] = time.monotonic()-started
            write_json(run_dir / "summary.json", summary)
            print(f"{title} {label}: {entry['panel_seconds']:.3f}s, {summary['completed_panels']}/{expected_panels} panels complete", flush=True)
        check_deadline()
        comparison = module.compare_panels(raw_by_panel)
        if module is eval_v4:
            summary["contrasts"] = comparison
        else:
            summary.update(comparison, artifact_validity="passed")
        json.dumps(summary, allow_nan=False)
    except BaseException as error:
        summary.pop("contrasts", None)
        summary.pop("gate", None)
        summary.pop("artifact_validity", None)
        summary.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed",
                       error=f"{type(error).__name__}: {error}", eval_seconds=time.monotonic()-started)
        for panel in summary["panels"].values():
            if panel["status"] == "running":
                panel["status"] = summary["status"]
        write_json(run_dir / "summary.json", summary)
        raise
    summary.update(status="finished", eval_seconds=time.monotonic()-started)
    write_json(run_dir / "summary.json", summary)
    print("run folder:", run_dir)


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
    ap.add_argument("--controller", choices=("rules-v0", direct.FORMAT), default=None,
                    help="rules-v0 (default) or directly encoded fixed-weight recurrent controller; inherited on resume")
    ap.add_argument("--objective", choices=PAIRED_OBJECTIVES, default=None,
                    help="paired-train-v1 only; inherited and immutable on resume")
    ap.add_argument("--structural-plasticity", choices=("off", MODE), default=None,
                    help="default off; fixed bounded activity-driven growth/pruning")
    ap.add_argument("--eval-worlds", type=int, default=128)
    ap.add_argument("--eval-protocol", choices=("v1", "v2", "structural-v1", broader_eval.PROTOCOL, reversal_eval.PROTOCOL, eval_v3.PROTOCOL, eval_v4.PROTOCOL, eval_v5.PROTOCOL, DIRECT_PROTOCOL, PAIRED_PROTOCOL, ECOLOGY_LINEAGE_PROTOCOL), default=None,
                    help="versioned training/diagnostics; ecology-lineage-v1 is a development-only mechanism fixture")
    ap.add_argument("--memory-condition", choices=("shared", "empty"), default=None,
                    help="ecology-lineage-v1 selected trace; both paired memory conditions are retained")
    ap.add_argument("--condition-split", choices=tuple(broader_eval.SPLIT_SEEDS), default=None,
                    help="versioned training/diagnostic protocols; final fixes the registered configuration")
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
    explicit = {"--" + field.replace("_", "-") for field in vars(a) for word in sys.argv[1:]
                if word.startswith("--") and ("--" + field.replace("_", "-")).startswith(word.split("=", 1)[0])}
    if a.max_train_seconds is not None and (not math.isfinite(a.max_train_seconds) or a.max_train_seconds <= 0):
        ap.error("--max-train-seconds must be positive and finite")

    git = git_info()
    if git["dirty"] and not (a.allow_dirty or (a.quick and not a.resume)):
        raise SystemExit("Uncommitted code changes. Commit first (every run must map to a commit), "
                         "or pass --allow-dirty and say why in the experiment log.")
    if a.memory_condition is not None and a.eval_protocol != ECOLOGY_LINEAGE_PROTOCOL:
        ap.error("--memory-condition is only valid for ecology-lineage-v1")
    if a.eval_protocol == ECOLOGY_LINEAGE_PROTOCOL:
        from scripts import ecology_lineage
        try:
            ecology_lineage.run_tracked(a, git, ROOT)
        except ValueError as error:
            ap.error(str(error))
        return
    if a.objective is not None and not a.resume and a.eval_protocol != PAIRED_PROTOCOL:
        ap.error("--objective is only valid for paired-train-v1")
    if a.eval_protocol in (eval_v4.PROTOCOL, eval_v5.PROTOCOL):
        a.condition_split = a.condition_split or "final"
        try:
            run_paired_diagnostic(a, git, eval_v4 if a.eval_protocol == eval_v4.PROTOCOL else eval_v5)
        except ValueError as error:
            ap.error(str(error))
        return
    if a.eval_protocol in (eval_v3.PROTOCOL, reversal_eval.PROTOCOL) and a.controller == direct.FORMAT:
        ap.error("saved v3/reversal-v1 diagnostic CLI accepts rules-v0 only; direct-v1 evaluates its trained controller")
    if a.eval_protocol == eval_v3.PROTOCOL:
        a.structural_plasticity = a.structural_plasticity or "off"
        try:
            run_v3(a, git)
        except ValueError as error:
            ap.error(str(error))
        return
    if a.eval_protocol == reversal_eval.PROTOCOL:
        a.condition_split = a.condition_split or "development"
        a.structural_plasticity = a.structural_plasticity or "off"
        try:
            run_reversal(a, git)
        except ValueError as error:
            ap.error(str(error))
        return
    if a.resume:
        run_dir = Path(a.resume)
        config = json.loads((run_dir / "config.json").read_text())
        args = config["args"]
        if args.get("eval_protocol") == ECOLOGY_LINEAGE_PROTOCOL:
            ap.error("Development lineage attempts cannot resume; preserve all evidence and use a new run name")
        if args.get("eval_protocol") == PAIRED_PROTOCOL:
            if a.cfg or a.quick or a.allow_dirty:
                ap.error("Cannot change paired training configuration, use quick mode or dirty code on resume")
            for field in ("name", "start", "generations", "pop", "worlds", "sigma", "lr", "seed",
                          "no_plastic", "controller", "objective", "checkpoint_every", "eval_worlds"):
                if "--" + field.replace("_", "-") in explicit and getattr(a, field) != args.get(field):
                    ap.error(f"Cannot change paired training {field} on resume")
        elif a.objective is not None:
            ap.error("--objective is only valid for paired-train-v1")
        saved_controller = args.get("controller") or "rules-v0"
        if a.controller is not None and a.controller != saved_controller:
            raise SystemExit("Cannot change controller on resume")
        if saved_controller == direct.FORMAT and (a.cfg or a.quick):
            ap.error("Cannot change direct-v1 configuration or use --quick on resume")
        if args.get("eval_protocol") == eval_v3.PROTOCOL:
            ap.error("EVAL v3 panels cannot resume; preserve this folder and start a new run")
        if args.get("eval_protocol") in (eval_v4.PROTOCOL, eval_v5.PROTOCOL):
            ap.error("Paired diagnostic attempts cannot resume; preserve all evidence and preregister a new attempt")
        if args.get("eval_protocol") == reversal_eval.PROTOCOL:
            ap.error("reversal-v1 does not resume; preserve the folder and use a new run name")
        if a.eval_protocol is not None and a.eval_protocol != args.get("eval_protocol", "v1"):
            raise SystemExit("Cannot change evaluation protocol on resume")
        if args.get("eval_protocol", "v1") == broader_eval.PROTOCOL:
            ap.error("broader-v1 panels cannot be resumed; preserve this folder and start a new run")
        if a.condition_split is not None:
            if args.get("eval_protocol") not in (DIRECT_PROTOCOL, PAIRED_PROTOCOL) or a.condition_split != args.get("condition_split", "final"):
                ap.error("Cannot change condition split on resume")
        if a.structural_plasticity is not None and a.structural_plasticity != args.get("structural_plasticity", "off"):
            raise SystemExit("Cannot change structural mode on resume")
        if a.no_plastic and args.get("structural_plasticity", "off") != "off":
            ap.error("structural training conflicts with --no-plastic on resume")
        protocol = validate_eval_protocol(args, config["cfg_overrides"])
        config.setdefault("resumes", []).append(dict(at=datetime.datetime.now().isoformat(), git=git,
                                                     max_train_seconds=a.max_train_seconds))
    else:
        args = vars(a).copy()
        args["eval_protocol"] = a.eval_protocol or "v1"
        args["structural_plasticity"] = a.structural_plasticity or "off"
        args["controller"] = a.controller or "rules-v0"
        if args["eval_protocol"] in (DIRECT_PROTOCOL, PAIRED_PROTOCOL):
            args["condition_split"] = a.condition_split or "final"
        if a.condition_split is not None and args["eval_protocol"] not in (broader_eval.PROTOCOL, DIRECT_PROTOCOL, PAIRED_PROTOCOL):
            ap.error("--condition-split requires a versioned training or diagnostic protocol")
        if a.quick:
            args.update(generations=3, pop=8, worlds=2, eval_worlds=4, checkpoint_every=1)
        overrides = parse_cfg(a.cfg)
        protocol = validate_eval_protocol(args, overrides)
        if protocol == broader_eval.PROTOCOL:
            a.structural_plasticity = args["structural_plasticity"]
            a.condition_split = a.condition_split or "development"
            try:
                run_broader(a, git)
            except ValueError as error:
                ap.error(str(error))
            return
        if protocol == "v2":
            source_bytes = Path(args["start"][5:]).read_bytes()
            file_mean = diagnostic_genome(source_bytes, Cfg())
        if protocol == PAIRED_PROTOCOL:
            file_mean, source_bytes, source_metadata = paired_source(args, Cfg(**overrides))
            if any((ROOT / "runs").glob(f"*_{a.name}")):
                ap.error("A paired training attempt with this name already exists; preserve it and use its resume path")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = ROOT / "runs" / f"{stamp}_{a.name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        config = dict(args=args, cfg_overrides=overrides, git=git,
                      created=datetime.datetime.now().isoformat(),
                      machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                                   jax=jax.__version__, devices=[str(d) for d in jax.devices()]))
        if protocol == "v2":
            config.update(eval_protocol="v2", eval_seed=123, eval_worlds=128,
                          source_genome_sha256=hashlib.sha256(source_bytes).hexdigest())
        if protocol == PAIRED_PROTOCOL:
            config.update(controller="rules-v0", eval_protocol=PAIRED_PROTOCOL, objective=args["objective"],
                          input_genome=source_metadata, evaluation_status="deferred",
                          initial_optimizer=paired_initial_optimizer(Cfg(**overrides), file_mean, args["seed"]))
        if protocol == "structural-v1":
            config["structural_rule"] = dict(name=MODE, interval=INTERVAL, prune_fraction=PRUNE_FRACTION,
                                             extra_fanin=1, stream_tag=STREAM_TAG, timing="after brain tick; previous activity")
            if args["start"].startswith("file:"):
                source = Path(args["start"][5:]).resolve()
                config["input_genome"] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())

    c = Cfg(**config["cfg_overrides"])
    is_direct = protocol == DIRECT_PROTOCOL
    is_paired = protocol == PAIRED_PROTOCOL
    if is_paired and a.resume:
        file_mean, source_bytes, source_metadata = paired_source(args, c)
        if (config.get("cfg") != c._asdict() or config.get("controller") != "rules-v0"
                or config.get("genome_size") != 155 or config.get("eval_protocol") != PAIRED_PROTOCOL
                or config.get("objective") != args["objective"] or config.get("evaluation_status") != "deferred"
                or config.get("input_genome") != source_metadata
                or config.get("initial_optimizer") != paired_initial_optimizer(c, file_mean, args["seed"])
                or config.get("start_genome_hash") != genome_hash(file_mean)
                or (run_dir / "genome_start.npy").read_bytes() != source_bytes):
            raise SystemExit("Cannot resume: paired source bytes, controller, configuration or initial optimizer metadata changed")
        prior = run_dir / "summary.json"
        if ((run_dir / "genome_final.npy").exists() or (run_dir / "events.npz").exists()
                or (prior.exists() and json.loads(prior.read_text()).get("status") == "finished")):
            raise SystemExit("Cannot resume completed or evaluated paired training; preserve its final artifacts")
    if is_direct:
        count = direct.genome_size(c)
        eval_seed = 41 if args["condition_split"] == "development" else eval_v3.EVAL_SEED
        encoding = dict(format=direct.FORMAT, weights=count,
                        order="row-major non-sensor postsynaptic rows, excluding diagonal",
                        init_std=1/math.sqrt(count/(c.H+6)),
                        lifetime_weight_updates=False, weight_bounds=None, development_noise=False)
        if a.resume:
            if (config.get("controller") != direct.FORMAT or config.get("genome_size") != count
                    or config.get("encoding") != encoding or config.get("eval_seed") != eval_seed
                    or config.get("eval_worlds") != args["eval_worlds"]):
                raise SystemExit("Cannot resume: direct controller format, genome size or evaluation metadata changed")
            start = np.load(run_dir / "genome_start.npy", allow_pickle=False)
            if (start.shape != (count,) or not np.issubdtype(start.dtype, np.floating)
                    or not np.isfinite(start).all() or genome_hash(start) != config["start_genome_hash"]):
                raise SystemExit("Cannot resume: direct starting genome is invalid or changed")
            if config.get("initial_weights") != direct_weight_stats(start, c):
                raise SystemExit("Cannot resume: direct starting-weight diagnostics changed")
            prior_summary = run_dir / "summary.json"
            if ((run_dir / "events.npz").exists()
                    or (prior_summary.exists() and json.loads(prior_summary.read_text()).get("status") == "finished")):
                raise SystemExit("Cannot resume a finished or evaluated direct-v1 run; preserve its final artifacts")
        else:
            config.update(controller=direct.FORMAT, eval_protocol=DIRECT_PROTOCOL,
                          eval_seed=eval_seed, eval_worlds=args["eval_worlds"],
                          encoding=encoding)
    config["cfg"] = c._asdict()
    config["genome_size"] = direct.genome_size(c) if is_direct else genome_size(c)
    if protocol == "v2" and a.resume:
        source_bytes = (run_dir / "genome_start.npy").read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != config.get("source_genome_sha256"):
            raise SystemExit("Cannot resume EVAL v2: saved source genome SHA-256 mismatch")
        file_mean = diagnostic_genome(source_bytes, c)
    plastic = not args["no_plastic"]
    structural = args.get("structural_plasticity", "off") != "off"
    if is_paired:
        evaluate, _ = paired.make_evaluator(c, args["worlds"], args["objective"])
    elif is_direct:
        evaluate, _ = direct.make_evaluator(c, args["worlds"])
    else:
        evaluate, _ = make_evaluator(c, args["worlds"], plastic, structural=structural)

    if (run_dir / "checkpoint.npz").exists():
        state = load_checkpoint(run_dir, direct_cfg=c if is_direct else None, paired_cfg=c if is_paired else None)
        if is_paired and state["t"] >= args["generations"]:
            raise SystemExit("Cannot resume completed paired training; preserve its final checkpoint")
        print(f"resuming {run_dir.name} at generation {state['t']}")
    else:
        if is_direct:
            _, initial_key = jax.random.split(jax.random.PRNGKey(args["seed"]))
            mean = direct.init_genome(c, initial_key)
        else:
            mean = file_mean if protocol == "v2" or is_paired else start_genome(args["start"], c)
        state = es_init(c, mean, args["seed"])
        if not a.resume:
            config["start_genome_hash"] = genome_hash(state["mu"])
            if protocol == "v2" or is_paired:
                (run_dir / "genome_start.npy").write_bytes(source_bytes)
            else:
                np.save(run_dir / "genome_start.npy", np.asarray(state["mu"]))
        elif genome_hash(state["mu"]) != config["start_genome_hash"]:
            raise SystemExit("Cannot recover without checkpoint: starting genome mismatch")
    if is_direct:
        validate_direct_state(state, c)
        if not a.resume:
            config["initial_weights"] = direct_weight_stats(state["mu"], c)
    if is_paired:
        validate_es_state(state, 155, PAIRED_PROTOCOL)
    if protocol == "v2" and (genome_hash(file_mean) != config["start_genome_hash"]
                             or not np.array_equal(np.asarray(state["mu"]), file_mean)):
        raise SystemExit("Cannot resume EVAL v2: checkpoint differs from saved source genome")

    status, t0 = "running", time.monotonic()
    metrics_path = run_dir / "metrics.jsonl"
    metrics_text = metrics_path.read_text() if metrics_path.exists() else ""
    if metrics_text and not metrics_text.endswith("\n"):
        raise SystemExit("Cannot resume: metrics end with an incomplete record")
    try:
        history = [json.loads(line) for line in metrics_text.splitlines()]
    except json.JSONDecodeError as error:
        raise SystemExit(f"Cannot resume: invalid metrics JSON: {error}") from error
    if any(not isinstance(row, dict) or type(row.get("gen")) is not int or row["gen"] != i
           for i, row in enumerate(history)):
        raise SystemExit("Cannot resume: metrics generations must be contiguous from zero")
    if not 0 <= state["t"] <= len(history) <= args["generations"]:
        raise SystemExit("Cannot resume: checkpoint generation, metrics count, and target are inconsistent")
    if is_paired and any(not math.isfinite(value) for row in history for value in row.values()):
        raise SystemExit("Cannot resume: paired training history contains nonfinite metrics")

    checkpoint_generation = state["t"]
    # Verify the entire flushed tail before advancing the checkpoint or touching existing records.
    for previous in history[checkpoint_generation:]:
        if a.max_train_seconds is not None and time.monotonic() - t0 >= a.max_train_seconds:
            raise SystemExit("Training limit reached during metrics recovery; existing artifacts unchanged")
        recovered, row = es_step(state, evaluate, args["pop"], args["worlds"], args["sigma"], args["lr"])
        if is_direct:
            validate_direct_state(recovered, c)
        if is_paired:
            validate_es_state(recovered, 155, PAIRED_PROTOCOL)
        if {k: v for k, v in previous.items() if k != "seconds"} != row:
            raise SystemExit(f"Cannot resume: metric mismatch at generation {previous['gen']}; existing artifacts unchanged")
        state = recovered
    if a.resume:
        config["resumes"][-1]["replayed_generations"] = state["t"] - checkpoint_generation
    write_json(run_dir / "config.json", config)

    summary_path = run_dir / "summary.json"
    write_json(summary_path, dict(status=status, generation=state["t"]))
    try:
        with open(metrics_path, "a") as mf:
            while state["t"] < args["generations"]:
                if a.max_train_seconds is not None and time.monotonic() - t0 >= a.max_train_seconds:
                    status = "interrupted"
                    print("training time limit reached; checkpointing, resume with --resume", run_dir)
                    break
                g0 = time.monotonic()
                candidate, row = es_step(state, evaluate, args["pop"], args["worlds"], args["sigma"], args["lr"])
                if is_direct or is_paired:
                    validate_es_state(candidate, direct.genome_size(c) if is_direct else 155, args["controller"])
                    if not all(math.isfinite(value) for value in row.values()):
                        raise ValueError(f"Nonfinite {protocol} training metrics")
                state = candidate
                jax.block_until_ready(state)
                row["seconds"] = round(time.monotonic() - g0, 3)
                mf.write(json.dumps(row) + "\n"); mf.flush()
                if row["gen"] % 10 == 0 or state["t"] == args["generations"]:
                    print(format_row(row) + f" | {row['seconds']:.1f}s/gen", flush=True)
                if state["t"] % args["checkpoint_every"] == 0:
                    save_checkpoint(run_dir, state)
        if state["t"] >= args["generations"]:
            status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        print("interrupted; checkpoint saved, resume with --resume", run_dir)
    except StructuralValidationError as error:
        save_checkpoint(run_dir, state)
        write_json(summary_path, dict(status="failed", generation=state["t"], error=str(error)))
        raise
    except Exception as error:
        if not (is_direct or is_paired):
            raise
        save_checkpoint(run_dir, state)
        write_json(summary_path, dict(status="failed", generation=state["t"], controller=args["controller"],
                                     eval_protocol=protocol, error=f"{type(error).__name__}: {error}"))
        raise
    save_checkpoint(run_dir, state)

    summary = dict(status=status, generation=state["t"], train_minutes=round((time.monotonic() - t0) / 60, 2),
                   final_genome_hash=genome_hash(state["mu"]))
    if is_direct:
        summary.update(controller=direct.FORMAT, eval_protocol=DIRECT_PROTOCOL,
                       condition_split=args["condition_split"], eval_seed=eval_seed,
                       eval_worlds=args["eval_worlds"], initial_weights=config["initial_weights"],
                       final_weights=direct_weight_stats(state["mu"], c))
    if is_paired:
        summary.update(controller="rules-v0", eval_protocol=PAIRED_PROTOCOL, objective=args["objective"],
                       condition_split=args["condition_split"], training_complete=status == "finished",
                       evaluation_status="deferred")
    if status == "finished":
        np.save(run_dir / "genome_final.npy", np.asarray(state["mu"]))
        n = args["eval_worlds"]
        summary["eval_worlds"] = n
        if is_paired:
            summary["final_genome_sha256"] = hashlib.sha256((run_dir / "genome_final.npy").read_bytes()).hexdigest()
            evaluations = {}
        elif is_direct:
            eval_start = time.monotonic()
            try:
                panels, raw = eval_v3.evaluate_conditions(state["mu"], c, n, eval_seed,
                                                          controller=direct.FORMAT)
                np.savez_compressed(run_dir / "events.npz", **raw)
                summary.update(evaluations=panels, final_direct=panels["stationary"]["genome"],
                               eval_seconds=time.monotonic()-eval_start)
                json.dumps(summary, allow_nan=False)
            except BaseException as error:
                summary.pop("evaluations", None)
                summary.pop("final_direct", None)
                summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                               error=f"{type(error).__name__}: {error}", eval_seconds=time.monotonic()-eval_start)
                write_json(summary_path, summary)
                raise
            for condition, controllers in panels.items():
                metric = controllers["genome"]
                print(f"{condition}/direct: fitness {metric['fitness']:.6f} +/- {metric['fitness_se']:.6f} SE")
            evaluations = {}
        elif protocol == "v2":
            summary.update(eval_protocol="v2", eval_seed=123, evaluations={})
            for name, hebbian, predictor in (("both", True, True), ("hebbian_only", True, False),
                                             ("frozen", False, False)):
                result = test_genome(state["mu"], c, n_worlds=n, seed=123,
                                     plastic=hebbian, predictor_plastic=predictor)
                if not all(math.isfinite(value) for value in result.values()):
                    raise ValueError(f"Nonfinite EVAL v2 result in {name}")
                summary["evaluations"][name] = result
            evaluations = summary["evaluations"]
        elif protocol == "structural-v1":
            summary["eval_protocol"] = "structural-v1"
            summary["conditions"], summary["condition_seconds"] = {}, {}
            try:
                for label, kwargs in (("Weight-only", dict(plastic=True)),
                                      ("Weight+structure", dict(plastic=True, structural=True)),
                                      ("Frozen", dict(plastic=False, structural=True))):
                    eval_start = time.perf_counter()
                    summary["conditions"][label] = test_genome(state["mu"], c, n_worlds=n, **kwargs)
                    summary["condition_seconds"][label] = round(time.perf_counter() - eval_start, 6)
            except Exception as error:
                summary.update(status="failed", error=str(error))
                write_json(summary_path, summary)
                raise
            evaluations = summary["conditions"]
        else:
            start = jnp.asarray(np.load(run_dir / "genome_start.npy"))
            summary["final_learning"] = test_genome(state["mu"], c, n_worlds=n, plastic=True)
            summary["final_frozen"] = test_genome(state["mu"], c, n_worlds=n, plastic=False)
            summary["start_learning"] = test_genome(start, c, n_worlds=n, plastic=True)
            evaluations = {k: summary[k] for k in ("final_learning", "final_frozen", "start_learning")}
        for k, r in evaluations.items():
            print(f"{k:15s} fitness {r['fitness']:6.2f} | acc {r['acc_first']:.2f} -> {r['acc_second']:.2f}")
    write_json(summary_path, summary)
    print("run folder:", run_dir)


if __name__ == "__main__":
    main()
