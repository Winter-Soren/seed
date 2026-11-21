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
import subprocess
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
RECOVERY_NAME = "E117-A2-recovery"
RECOVERY_PARENT = "runs/20261121-174842_E117-A2"
RECOVERY_COMMIT = "95a254bcbe5c35c5f8ec343e056c579d562645ac"
RECOVERY_GUARD = "results/diagnostics/E117-A2-attempt2-guard.jsonl"
RECOVERY_FAMILIES = 181
RECOVERY_PARENT_SECONDS = 400.7536469
RECOVERY_MAX_SECONDS = 9499
RECOVERY_HASHES = dict(
    config="1d6f834be2e7497feb1181df66e15a9c2629dc96622e5056449466d28ae60232",
    status="e7a4d6b53fb5e80f0fcb5db245fe8923cfeee030f107357d50d8582f986dca8d",
    metrics="b14eae112dcdbc227a73b8d141fbe31c007c9e4f8613aeb052035fc6acb33ed2",
    guard="d12ecdd94f49e2afec4e3f39c752ac359a250c81c31f145a0917ad6a88b2d213")


def _validate(args, git, root, explicit):
    recovery = args.name == RECOVERY_NAME
    allowed = ALLOWED_FLAGS | ({"--resume"} if recovery else set())
    if explicit is not None and set(explicit) - allowed:
        raise ValueError("cold-donor-v1 accepts only its registered identity, source and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("cold-donor-v1 requires zero generations and the fixed registered configuration; "
                         "no training, persistence, quick, dirty or scientific overrides")
    if (bool(args.resume) != recovery or recovery and (args.condition_split != "final"
            or (root / args.resume).resolve() != (root / RECOVERY_PARENT).resolve())):
        raise ValueError("only the registered E117-A2-recovery may resume its exact interrupted parent")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("cold-donor-v1 requires clean committed source provenance")
    if args.condition_split == "development":
        source = SOURCES[0]
        expected = ("E117-benchmark", 42, 4)
    else:
        candidates = [item for item in SOURCES if args.name == "E117-" + item["label"]
                      or recovery and item["label"] == "A2"]
        if len(candidates) != 1:
            raise ValueError("final cold-donor-v1 name must be E117-A0, E117-A1 or E117-A2")
        source = candidates[0]
        expected = (RECOVERY_NAME if recovery else "E117-" + source["label"],
                    2026112101 + source["training_seed"], 256)
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


def _write_bytes(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


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


def _runtime():
    return dict(python=platform.python_version(), platform=platform.system(),
                platform_release=platform.release(), machine=platform.machine(), numpy=np.__version__,
                jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib"))


def _scientific_source(root, commit):
    try:
        result = subprocess.run(["git", "rev-parse", f"{commit}:seed", f"{commit}:scripts/cold_donor.py"],
                                cwd=root, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("cannot verify the recovery's unchanged scientific source") from error
    hashes = result.stdout.splitlines()
    if len(hashes) != 2 or any(len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value) for value in hashes):
        raise ValueError("invalid scientific source object fingerprints")
    return dict(seed=hashes[0], cold_donor=hashes[1])


def _load_recovery(root, args, git, source, genome_bytes, c, runtime):
    """Check the fixed interrupted prefix without calculating or exposing outcomes."""
    parent = root / RECOVERY_PARENT
    expected_entries = {"config.json", "status.json", "metrics.jsonl", "genome_start.npy", "families"}
    if {path.name for path in parent.iterdir()} != expected_entries:
        raise ValueError("interrupted parent must have only its registered incomplete artifacts and no summary")
    payloads = {name: (parent / (name + extension)).read_bytes()
                for name, extension in (("config", ".json"), ("status", ".json"), ("metrics", ".jsonl"))}
    payloads["guard"] = (root / RECOVERY_GUARD).read_bytes()
    for name, expected_hash in RECOVERY_HASHES.items():
        if hashlib.sha256(payloads[name]).hexdigest() != expected_hash:
            raise ValueError(f"interrupted parent {name} bytes differ from the registered recovery fingerprint")
    original, status = json.loads(payloads["config"]), json.loads(payloads["status"])
    if original.get("git", {}).get("commit") != RECOVERY_COMMIT or original["git"].get("dirty") is not False:
        raise ValueError("interrupted parent did not use the required clean source commit")
    expected_args = dict(vars(args), name="E117-A2", resume=None, start="file:" + source["path"])
    if original.get("args") != expected_args:
        raise ValueError("recovery settings must exactly match the interrupted parent")
    expected_config = dict(protocol=PROTOCOL, source=source, condition_split="final", cfg=c._asdict(),
        cfg_overrides={}, ecology=dict(version=ECOLOGY.version, **asdict(ECOLOGY)), persistence="off", plastic=True,
        controller="rules-v0", genome_size=155, seed=2026112103, eval_seed=2026112103, eval_worlds=256,
        expected_families=256, key_order=list(KEY_ORDER), cold_start=c.T // 2, max_seconds=9900,
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=runtime)
    if any(original.get(key) != value for key, value in expected_config.items()):
        raise ValueError("recovery configuration or runtime differs from the interrupted parent")
    if (parent / "genome_start.npy").read_bytes() != genome_bytes:
        raise ValueError("interrupted parent's genome differs from the pinned original A2 bytes")
    if (status.get("status"), status.get("completed_families"), status.get("expected_families")) != ("running", RECOVERY_FAMILIES, 256):
        raise ValueError("parent operational status does not match its interrupted durable prefix")
    guard = [json.loads(line) for line in payloads["guard"].splitlines()]
    stopped = guard[-1] if guard else {}
    if (stopped.get("event") != "stopped" or stopped.get("processStarted") is not True
            or stopped.get("wallSeconds") != RECOVERY_PARENT_SECONDS or stopped.get("error", "").strip() != "Call cancelled"):
        raise ValueError("recovery requires the registered telemetry interruption and process wall charge")
    original_source = _scientific_source(root, RECOVERY_COMMIT)
    if _scientific_source(root, git["commit"]) != original_source:
        raise ValueError("recovery may change the wrapper only; engine and measurement source must be identical")
    metrics = payloads["metrics"]
    lines = metrics.splitlines()
    names = {f"family_{index:04d}.npz" for index in range(RECOVERY_FAMILIES)}
    if (not metrics.endswith(b"\n") or len(lines) != RECOVERY_FAMILIES
            or {path.name for path in (parent / "families").iterdir()} != names):
        raise ValueError("parent must have exactly 181 complete matching metric and raw records")
    rows, artifacts, used_keys = [], [], set()
    last_elapsed = 0.
    for index, line in enumerate(lines):
        record = json.loads(line)
        keys = np.asarray(cold_donor.family_keys(args.seed, index))
        elapsed = record.get("elapsed_seconds")
        if (record.get("family") != index or record.get("rule") != index % 2 or record.get("keys") != keys.tolist()
                or not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < last_elapsed):
            raise ValueError("interrupted metric identities, keys, rule or time ordering are invalid")
        last_elapsed = elapsed
        path = f"families/family_{index:04d}.npz"
        raw_bytes = (parent / path).read_bytes()
        artifact = dict(index=index, path=path, sha256=hashlib.sha256(raw_bytes).hexdigest())
        if record.get("raw_file") != artifact:
            raise ValueError("interrupted family bytes do not match their durable metric fingerprint")
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as arrays:
            if len(arrays.files) != len(set(arrays.files)):
                raise ValueError("interrupted family archive has duplicate array names")
            row = {key: arrays[key] for key in arrays.files}
        cold_donor.validate_family(row, c)
        if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in row.values()):
            raise ValueError("interrupted raw arrays must be finite and numeric")
        if not np.array_equal(row["keys"], keys) or int(row["rule"]) != index % 2:
            raise ValueError("interrupted family has the wrong registered keys or rule")
        tokens = [key.tobytes() for key in keys]
        if keys.shape != (4, 2) or keys.dtype != np.uint32 or len(set(tokens)) != 4 or used_keys.intersection(tokens):
            raise ValueError("interrupted families must have distinct registered keys")
        used_keys.update(tokens)
        rows.append(row)
        artifacts.append(artifact)
    parent_status_elapsed = status.get("elapsed_seconds")
    if (not isinstance(parent_status_elapsed, (int, float)) or not np.isfinite(parent_status_elapsed)
            or not last_elapsed <= parent_status_elapsed <= RECOVERY_PARENT_SECONDS):
        raise ValueError("interrupted operational time ledger is inconsistent")
    provenance = dict(version="cold-donor-recovery-v1", parent_run=RECOVERY_PARENT,
        parent_source_commit=RECOVERY_COMMIT, parent_config_sha256=RECOVERY_HASHES["config"],
        parent_status_sha256=RECOVERY_HASHES["status"], parent_metrics_sha256=RECOVERY_HASHES["metrics"],
        parent_genome_sha256=source["sha256"], guard_path=RECOVERY_GUARD, guard_sha256=RECOVERY_HASHES["guard"],
        completed_families=RECOVERY_FAMILIES, first_new_family=RECOVERY_FAMILIES,
        parent_elapsed_seconds=RECOVERY_PARENT_SECONDS, parent_last_metric_elapsed_seconds=last_elapsed,
        scientific_source=original_source, copied_raw_files=artifacts)
    return dict(provenance=provenance, payloads=payloads, rows=rows, artifacts=artifacts, used_keys=used_keys)


def run_tracked(args, git, root, *, explicit=None):
    """Run one fixed identity once; a partial attempt never receives a gate decision."""
    started = time.monotonic()
    root = Path(root)
    source, content, flat = _validate(args, git, root, explicit)
    c = Cfg()
    runtime = _runtime()
    recovery = _load_recovery(root, args, git, source, content, c, runtime) if args.resume else None
    parent_elapsed = RECOVERY_PARENT_SECONDS if recovery else 0.
    limit = RECOVERY_MAX_SECONDS if recovery else MAX_SECONDS[args.condition_split]
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    if recovery:
        recorded_args["resume"] = RECOVERY_PARENT
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
        ecology=dict(version=ECOLOGY.version, **asdict(ECOLOGY)), persistence="off", plastic=True,
        controller="rules-v0", genome_size=155, seed=args.seed, eval_seed=args.seed,
        eval_worlds=args.eval_worlds, expected_families=args.eval_worlds, key_order=list(KEY_ORDER),
        cold_start=c.T // 2, rule_assignment="family_index modulo 2; donor and recipient share rule",
        max_seconds=limit, scientific_result=args.condition_split == "final",
        purpose="paired inherited-versus-fresh recipient weights after a cold donor life",
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
        runtime=runtime)
    if recovery:
        config.update(recovery=recovery["provenance"], cumulative_max_seconds=MAX_SECONDS["final"])
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    rows, artifacts, used_keys = [], [], set()

    def status(state):
        elapsed = time.monotonic() - started
        result = dict(status=state, completed_families=len(artifacts), expected_families=args.eval_worlds,
                      elapsed_seconds=parent_elapsed + elapsed)
        if recovery:
            result["invocation_elapsed_seconds"] = elapsed
        return result

    def summary(state):
        result = dict(**status(state), protocol=PROTOCOL, condition_split=args.condition_split,
                      generation=0, source_label=source["label"], eval_seed=args.seed, eval_worlds=args.eval_worlds,
                      persistence="off", scientific_result=False, artifact_validity="incomplete",
                      gate_status="undecided", raw_files=list(artifacts))
        if recovery:
            result.update(recovery=recovery["provenance"], segments=[
                dict(kind="preserved_parent", source_commit=RECOVERY_COMMIT, first_family=0,
                     stop_family=min(len(artifacts), RECOVERY_FAMILIES), elapsed_seconds=parent_elapsed,
                     timing_basis="supervisor_wall"),
                dict(kind="recovery", source_commit=git["commit"], first_family=RECOVERY_FAMILIES,
                     stop_family=max(len(artifacts), RECOVERY_FAMILIES),
                     elapsed_seconds=result["invocation_elapsed_seconds"], timing_basis="runner_monotonic")])
        return result

    def check_deadline():
        elapsed = time.monotonic() - started
        if elapsed >= limit or recovery and parent_elapsed + elapsed >= MAX_SECONDS["final"]:
            raise TimeoutError(f"cold-donor-v1 exceeded its {limit}-second process allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("xb") as stream:
            if recovery:
                (folder / "parent").mkdir()
                for name, suffix in (("config", ".json"), ("status", ".json"), ("guard", ".jsonl")):
                    _write_bytes(folder / "parent" / (name + suffix), recovery["payloads"][name])
                for artifact, row in zip(recovery["artifacts"], recovery["rows"]):
                    raw = (root / RECOVERY_PARENT / artifact["path"]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                        raise ValueError("interrupted family changed after recovery validation")
                    _write_bytes(folder / artifact["path"], raw)
                    artifacts.append(artifact)
                    rows.append(row)
                stream.write(recovery["payloads"]["metrics"])
                stream.flush()
                os.fsync(stream.fileno())
                used_keys.update(recovery["used_keys"])
                _write_status(folder, status("running"))
                print(f"preserved {RECOVERY_FAMILIES} families; evaluating only the remaining "
                      f"{args.eval_worlds - RECOVERY_FAMILIES}", flush=True)
            _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(42)))
            genome = unravel(jnp.asarray(flat))
            for index in range(len(artifacts), args.eval_worlds):
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
                elapsed = time.monotonic() - started
                record = dict(family=index, rule=rule, keys=keys.tolist(), raw_file=artifact,
                              elapsed_seconds=parent_elapsed + elapsed, measurements=observation)
                if recovery:
                    record.update(source_commit=git["commit"], invocation_elapsed_seconds=elapsed)
                stream.write((json.dumps(record, allow_nan=False) + "\n").encode("utf-8"))
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
