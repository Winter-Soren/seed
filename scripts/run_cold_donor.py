"""Immutable, streamed runner for the preregistered E117 cold donor comparison."""
from dataclasses import asdict
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed.ecology import EcologyV1
from seed.seed import Cfg, init_genome
from scripts import cold_donor

PROTOCOL = "cold-donor-v1"
MAX_SECONDS = {"development": 600, "final": 9900}
ECOLOGY = EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02)
KEY_ORDER = ("donor_world", "donor_development", "target_world", "target_development")
SOURCES = (
    dict(label="A0", training_seed=0, path="runs/20261112-023245_E103-A-blank-s0/genome_final.npy",
         sha256="477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"),
    dict(label="A1", training_seed=1, path="runs/20261112-050319_E103-A-blank-s1/genome_final.npy",
         sha256="35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899"),
    dict(label="A2", training_seed=2, path="runs/20261112-102755_E103-A-blank-s2/genome_final.npy",
         sha256="f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6"),
)
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("cold-donor-v1 accepts only its registered identity, source and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("cold-donor-v1 requires zero generations and the fixed registered configuration; "
                         "no training, persistence, quick, dirty, resume or scientific overrides")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("cold-donor-v1 requires clean committed source provenance")
    if args.condition_split == "development":
        source = SOURCES[0]
        expected = ("E117-benchmark", 42, 4)
    else:
        candidates = [item for item in SOURCES if args.name == "E117-" + item["label"]]
        if len(candidates) != 1:
            raise ValueError("final cold-donor-v1 name must be E117-A0, E117-A1 or E117-A2")
        source = candidates[0]
        expected = ("E117-" + source["label"], 2026112101 + source["training_seed"], 256)
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("cold-donor-v1 name, seed and family count must match the registered split")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this cold-donor attempt already exists; preserve it and preregister any recovery")
    expected_path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != expected_path:
        raise ValueError("cold-donor-v1 requires the matching original E103 genome path")
    content = expected_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("original E103 genome NPY hash does not match its registered source")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original E103 genome must be 155 finite float32 values")
    return dict(source), content, flat


def _write_json(path, data):
    # Encode before opening: invalid measurements must not leave a partial JSON artifact.
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_status(folder, data):
    # Only the explicitly operational status file changes; completed evidence is exclusive-create.
    pending = folder / "status.tmp.json"
    _write_json(pending, data)
    os.replace(pending, folder / "status.json")


def _save_family(folder, index, row, c, keys, rule):
    cold_donor.validate_family(row, c)
    arrays = {key: np.asarray(value) for key, value in row.items()}
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("cold-donor raw arrays must contain finite numeric values")
    if not np.array_equal(arrays["keys"], keys) or int(arrays["rule"]) != rule:
        raise ValueError("family result does not match its assigned keys and rule")
    relative = f"families/family_{index:04d}.npz"
    path = folder / relative
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(index=index, path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_tracked(args, git, root, *, explicit=None):
    """Run one fixed identity once; a partial attempt never receives a gate decision."""
    started = time.monotonic()
    root = Path(root)
    source, content, flat = _validate(args, git, root, explicit)
    c = Cfg()
    limit = MAX_SECONDS[args.condition_split]
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
        ecology=dict(version=ECOLOGY.version, **asdict(ECOLOGY)), persistence="off", plastic=True,
        controller="rules-v0", genome_size=155, seed=args.seed, eval_seed=args.seed,
        eval_worlds=args.eval_worlds, expected_families=args.eval_worlds, key_order=list(KEY_ORDER),
        cold_start=c.T // 2, rule_assignment="family_index modulo 2; donor and recipient share rule",
        max_seconds=limit, scientific_result=args.condition_split == "final",
        purpose="paired inherited-versus-fresh recipient weights after a cold donor life",
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
        runtime=dict(python=platform.python_version(), platform=platform.system(),
                     platform_release=platform.release(), machine=platform.machine(), numpy=np.__version__,
                     jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib")))
    _write_json(folder / "config.json", config)
    with (folder / "genome_start.npy").open("xb") as stream:
        stream.write(content)
    rows, artifacts, used_keys = [], [], set()

    def status(state):
        return dict(status=state, completed_families=len(artifacts), expected_families=args.eval_worlds,
                    elapsed_seconds=time.monotonic() - started)

    def summary(state):
        return dict(**status(state), protocol=PROTOCOL, condition_split=args.condition_split,
                    generation=0, source_label=source["label"], eval_seed=args.seed, eval_worlds=args.eval_worlds,
                    persistence="off", scientific_result=False, artifact_validity="incomplete",
                    gate_status="undecided", raw_files=list(artifacts))

    def check_deadline():
        if time.monotonic() - started >= limit:
            raise TimeoutError(f"cold-donor-v1 exceeded its {limit}-second process allowance")

    _write_status(folder, status("running"))
    try:
        _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(42)))
        genome = unravel(jnp.asarray(flat))
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                check_deadline()
                keys = np.asarray(cold_donor.family_keys(args.seed, index))
                if keys.shape != (4, 2) or keys.dtype != np.uint32:
                    raise ValueError("family keys must be four uint32 PRNG keys")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != 4 or used_keys.intersection(tokens):
                    raise ValueError("donor and recipient keys must be distinct across all family roles")
                used_keys.update(tokens)
                rule = index % 2
                row = cold_donor.evaluate_family(genome, keys, rule, c, ECOLOGY)
                artifact = _save_family(folder, index, row, c, keys, rule)
                # Once saved, the family is retained even if a later summary or budget check fails.
                artifacts.append(artifact)
                rows.append(row)
                observation = dict(cold_donor.summarize([row]))
                observation.pop("gate", None)
                record = dict(family=index, rule=rule, keys=keys.tolist(), raw_file=artifact,
                              elapsed_seconds=time.monotonic() - started, measurements=observation)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                print(f"cold-donor family {index + 1}/{args.eval_worlds} saved", flush=True)
                check_deadline()
        measurements = dict(cold_donor.summarize(rows))
        gate = measurements.pop("gate")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     scientific_result=args.condition_split == "final", gate_status="not_applicable")
        if args.condition_split == "final":
            if set(gate) != {"passed", "latency_passed", "survival_passed"} or any(type(v) is not bool for v in gate.values()):
                raise ValueError("a completed final study requires a defined boolean gate")
            final.update(gate=gate, gate_status="passed" if gate["passed"] else "failed")
        check_deadline()
        _write_json(folder / "summary.json", final)
    except BaseException as error:
        state = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        partial = summary(state)
        partial["error"] = f"{type(error).__name__}: {error}"
        if not (folder / "summary.json").exists():
            _write_json(folder / "summary.json", partial)
        _write_status(folder, status(state))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder
