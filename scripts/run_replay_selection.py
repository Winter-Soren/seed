"""Tracked E134 evolution and separately sealed parent/descendant panels."""

import datetime
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from scripts import social_observer_evolution as observer
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays, _save_genome, _state_arrays
from seed import replay_selection as engine, seed


PROTOCOL = "replay-selection-v1"
TRAIN_SEED = 2026121300
EVAL_SEED = 2026121400
MAX_SECONDS = {"development-training": 180, "development-evaluation": 120, "training": 450, "evaluation": 300}
SETTINGS = (16, 4, .05, .02)
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds", "--hypothesis"}
SOURCE_PATHS = ("seed/seed.py", "seed/encounter_diagnostic.py", "seed/replay_selection.py",
                "scripts/social_observer_evolution.py", "scripts/run_replay_selection.py",
                "scripts/run_experiment.py", "scripts/run_cold_donor.py", "scripts/run_shelter_evolution.py")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _ref(folder, relative):
    data = (folder / relative).read_bytes()
    return dict(path=relative, sha256=_sha(data), bytes=len(data))


def _save(folder, relative, arrays):
    return dict(_save_arrays(folder, relative, arrays), bytes=(folder / relative).stat().st_size)


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("replay selection permits only registered identity and evolution flags")
    if (args.eval_protocol != PROTOCOL or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or args.checkpoint_every != 10
            or (args.pop, args.worlds, args.sigma, args.lr) != SETTINGS):
        raise ValueError("replay selection fixes the native controller and ES settings")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("replay selection requires clean committed source")
    requests = {
        "E134-benchmark": ("development", "development-training", 0, TRAIN_SEED, 2, 0),
        "E134-benchmark-eval": ("development", "development-evaluation", 0, EVAL_SEED, 0, 4),
    }
    for index in range(3):
        requests[f"E134-A{index}-train"] = ("final", "training", index, TRAIN_SEED + index + 1, 10, 0)
        requests[f"E134-A{index}"] = ("final", "evaluation", index, EVAL_SEED + index + 1, 0, 128)
    if args.name not in requests:
        raise ValueError("unknown replay-selection attempt")
    split, mode, index, master, attempts, families = requests[args.name]
    if ((args.condition_split, args.seed, args.generations, args.eval_worlds) != (split, master, attempts, families)
            or any((root / "runs").glob("*_" + args.name))):
        raise ValueError("replay selection identity differs from registration or already exists")
    source = SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("replay selection requires the matching original E103 parent")
    content = (root / source["path"]).read_bytes()
    if _sha(content) != source["sha256"]:
        raise ValueError("saved E103 parent disagrees with its pinned digest")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if (flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all()
            or engine.PROTOCOL != PROTOCOL or engine.CFG != seed.Cfg(M=1)):
        raise ValueError("replay selection requires the original 155-number native genome")
    return mode, index, dict(source), content, flat


def evaluation_assignments(master, count):
    base = jax.random.fold_in(jax.random.PRNGKey(master), 0xE134)
    manifest_key, family_key = jax.random.split(base, 2)
    if count == 4:
        rows = np.asarray([(8, 16, 0), (16, 32, 0), (8, 32, 1), (16, 16, 1)], np.int32)
    elif count == 128:
        rows = np.repeat(np.asarray(list(product((8, 16), (16, 32), (0, 1))), np.int32), 16, axis=0)
        rows = rows[np.asarray(jax.random.permutation(manifest_key, count))]
    else:
        raise ValueError("evaluation requires four development or 128 independent held-out families")
    pre, gap, sham = rows.T
    keys = np.asarray(jax.vmap(lambda i: jax.random.split(jax.random.fold_in(family_key, i), len(KEY_ROLES)))(
        np.arange(count, dtype=np.uint32)))
    observer_keys(set(), keys, "evaluation family role")
    return dict(family_keys=keys, pre_contacts=pre, gap=gap, sham_rule=sham, initial_rules=np.arange(2, dtype=np.int32))


def observer_keys(seen, keys, label):
    values = np.asarray(keys)
    if values.dtype != np.uint32 or values.shape[-1:] != (2,):
        raise ValueError(f"invalid {label} keys")
    tokens = [row.tobytes() for row in values.reshape(-1, 2)]
    if len(set(tokens)) != len(tokens) or seen.intersection(tokens):
        raise ValueError(f"reused {label} keys")
    seen.update(tokens)


def _endpoint_receipts(root, source_code, commit, *, development=False):
    receipts, endpoints = [], []
    attempts = 2 if development else 10
    for index in range(1 if development else 3):
        name = "E134-benchmark" if development else f"E134-A{index}-train"
        paths = list((root / "runs").glob("*_" + name))
        if len(paths) != 1:
            raise ValueError("all three final training endpoints must be pinned before evaluation")
        folder = paths[0]
        config_bytes = (folder / "config.json").read_bytes()
        summary_bytes = (folder / "summary.json").read_bytes()
        config, summary = json.loads(config_bytes), json.loads(summary_bytes)
        endpoint = (folder / "genome_final.npy").read_bytes()
        endpoint_sha = _sha(endpoint)
        endpoint_flat = np.load(io.BytesIO(endpoint), allow_pickle=False)
        if (config.get("protocol") != PROTOCOL or config.get("mode") != ("development-training" if development else "training")
                or config.get("seed_index") != index or config.get("training_seed") != TRAIN_SEED + (0 if development else index + 1)
                or config.get("source") != SOURCES[index] or config.get("source_code") != source_code
                or config.get("git", {}).get("commit") != commit or config.get("git", {}).get("dirty") is not False
                or summary.get("status") != "finished" or summary.get("completed_attempts") != attempts
                or summary.get("completed_families") != 0 or len(summary.get("attempts", ())) != attempts + 1
                or len(summary.get("samples", ())) != attempts or summary.get("endpoint_sha256") != endpoint_sha):
            raise ValueError("a final training endpoint is incomplete or differs from its freeze receipt")
        checkpoint = _ref(folder, f"generations/attempt_{attempts:04d}.npz")
        if checkpoint != summary["attempts"][-1]:
            raise ValueError("training endpoint does not refer to the intact final attempted round")
        with np.load(folder / checkpoint["path"], allow_pickle=False) as saved:
            if (endpoint_flat.shape != (155,) or endpoint_flat.dtype != np.float32
                    or not np.isfinite(endpoint_flat).all() or saved["mu"].tobytes() != endpoint_flat.tobytes()
                    or int(saved["t"]) != summary["optimizer_updates"]):
                raise ValueError("frozen endpoint is not the last registered attempted mean")
        receipts.append(dict(seed_index=index, run=folder.relative_to(root).as_posix(), config_sha256=_sha(config_bytes),
            summary_sha256=_sha(summary_bytes), checkpoint_sha256=checkpoint["sha256"], endpoint_sha256=endpoint_sha))
        endpoints.append(endpoint)
    return receipts, endpoints


def _journal(stream, event):
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_tracked(args, git, root, *, explicit=None):
    started, root = time.monotonic(), Path(root).resolve()
    mode, index, source, parent_bytes, parent = _validate(args, git, root, explicit)
    source_code = {name: _sha((root / name).read_bytes()) for name in SOURCE_PATHS}
    endpoint_bytes, receipts, assignments = None, [], None
    training_seed = args.seed if args.generations else None
    evaluation_seed = args.seed if args.eval_worlds else None
    if evaluation_seed is not None:
        receipts, endpoints = _endpoint_receipts(root, source_code, git["commit"], development=mode == "development-evaluation")
        endpoint_bytes = endpoints[index]
        assignments = evaluation_assignments(evaluation_seed, args.eval_worlds)

    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    if args.generations:
        (folder / "generations").mkdir()
    if args.eval_worlds:
        (folder / "families").mkdir()
    manifest = _save(folder, "assignments.npz", assignments) if assignments is not None else None
    config = dict(protocol=PROTOCOL, mode=mode, args=dict(vars(args), start="file:"+source["path"]), git=dict(git),
        source=source, seed_index=index, training_seed=training_seed, evaluation_seed=evaluation_seed,
        expected_attempts=args.generations, expected_families=args.eval_worlds, source_code=source_code, l2=.005,
        cfg=engine.CFG._asdict(), train_receipts=receipts, assignments=manifest, key_roles=list(KEY_ROLES),
        endpoint_sha256=_sha(endpoint_bytes) if endpoint_bytes is not None else None,
        max_seconds=MAX_SECONDS[mode], scientific_result=False, runtime=_runtime())
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", parent_bytes)
    setup = time.monotonic() - started
    attempts, samples, raw_files = [], [], []
    attempt_walls, family_walls = [], []
    endpoint_sha, updates = None, 0

    def status(value):
        return dict(status=value, protocol=PROTOCOL, mode=mode, seed_index=index, elapsed_seconds=time.monotonic()-started,
            setup_seconds=setup, completed_attempts=len(samples), completed_families=len(raw_files),
            expected_attempts=args.generations, expected_families=args.eval_worlds, optimizer_updates=updates,
            attempt_wall_seconds=attempt_walls, family_wall_seconds=family_walls)

    def summary(value):
        return dict(**status(value), scientific_result=False, gate_status="undecided", endpoint_sha256=endpoint_sha,
            attempts=attempts, samples=samples, raw_files=raw_files)

    def deadline():
        if time.monotonic() - started >= MAX_SECONDS[mode]:
            raise TimeoutError("replay selection exceeded its registered process allowance")

    _write_status(folder, status("running"))
    seen = set()
    if assignments is not None:
        observer_keys(seen, assignments["family_keys"], "evaluation family role")
    try:
        if args.generations:
            evaluate = engine.make_evaluator()
            state = seed.es_init(engine.CFG, mean=parent, seed=training_seed)
            attempts.append(_save(folder, "generations/attempt_0000.npz", _state_arrays(state, 0)))
            with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
                for ordinal in range(1, args.generations+1):
                    deadline()
                    since = time.monotonic()
                    record = {}

                    def recorded(candidates, world_keys, birth_keys):
                        observer_keys(seen, world_keys, "training root")
                        if not np.array_equal(birth_keys[:args.pop//2], birth_keys[args.pop//2:]):
                            raise ValueError("antithetic candidates must share development randomness")
                        observer_keys(seen, birth_keys[:args.pop//2], "training birth")
                        batch = engine.training_batch(world_keys)
                        observer_keys(seen, batch["role_keys"], "training family role")
                        observer_keys(seen, batch["schedule_key"], "training schedule")
                        result = evaluate(candidates, batch, birth_keys)
                        bounds = np.asarray(result["exposure_energy_bounds"])
                        if (not np.asarray(result["survived"]).all() or bounds.shape != (args.pop, args.worlds, 2)
                                or not np.isfinite(bounds).all() or np.any(bounds[..., 0] <= 0)
                                or np.any(bounds[..., 1] >= engine.CFG.e_max)):
                            raise ValueError("forced exposure physics must preserve every training life without clipping")
                        record.update(candidate_genomes=candidates, world_keys=world_keys, development_keys=birth_keys,
                            **{"batch_"+name: value for name, value in batch.items()}, **result)
                        return result

                    next_key, noise_key, _, _ = jax.random.split(state["key"], 4)
                    half = jax.random.normal(noise_key, (args.pop//2, parent.size))
                    before = state
                    state, metrics = observer.es_step(state, recorded, *SETTINGS, l2=.005)
                    arrays = _state_arrays(state, int(state["t"]))
                    if not np.array_equal(arrays["key"], next_key) or (metrics["all_tied"] and any(
                            not np.array_equal(arrays[name], before[name]) for name in ("mu", "m", "v", "t"))):
                        raise ValueError("ES RNG or all-tied transition violated its contract")
                    record.update(eps=np.concatenate((half, -half)),
                        rank_weights=observer.average_centered_ranks(record["fitness"].mean(1)))
                    samples.append(_save(folder, f"generations/samples_{ordinal:04d}.npz", record))
                    attempts.append(_save(folder, f"generations/attempt_{ordinal:04d}.npz", arrays))
                    updates = int(arrays["t"])
                    _journal(stream, dict(attempt=ordinal, checkpoint=attempts[-1], sample=samples[-1], optimizer_updates=updates,
                        distinct_scores=metrics["distinct_scores"], objective_mean=metrics["fit_mean"],
                        objective_best=metrics["fit_best"], changed_parameters=int(np.count_nonzero(arrays["mu"] != before["mu"]))))
                    _write_status(folder, status("running"))
                    attempt_walls.append(time.monotonic()-since)
                    print(f"E134 attempt {ordinal}/{args.generations} saved", flush=True)
                endpoint = np.asarray(state["mu"])
                _save_genome(folder / "genome_final.npy", endpoint)
        else:
            _write_bytes(folder / "genome_final.npy", endpoint_bytes)
            endpoint = np.load(io.BytesIO(endpoint_bytes), allow_pickle=False)
        endpoint_sha = _ref(folder, "genome_final.npy")["sha256"]

        if assignments is not None:
            with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
                for family, keys in enumerate(assignments["family_keys"]):
                    deadline()
                    since = time.monotonic()
                    pre, gap, sham = (int(assignments[name][family]) for name in ("pre_contacts", "gap", "sham_rule"))
                    row = engine.evaluate_family(parent, endpoint, keys, family, pre, gap, sham)
                    raw_files.append(_save(folder, f"families/family_{family:04d}.npz", row))
                    _journal(stream, dict(index=family, raw_file=raw_files[-1]))
                    _write_status(folder, status("running"))
                    family_walls.append(time.monotonic()-since)
                    print(f"E134 family {family+1}/{args.eval_worlds} saved", flush=True)
        deadline()
        _write_json(folder / "summary.json", summary("finished"))
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        partial = dict(summary(value), error=f"{type(error).__name__}: {error}")
        _write_json(folder / "summary.json", partial)
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder, flush=True)
    return folder
