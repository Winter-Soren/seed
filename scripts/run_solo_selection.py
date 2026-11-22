"""Tracked native selection on free first-item foraging versus a focal proxy."""

import datetime
import io
import json
from pathlib import Path
import time

import jax
import numpy as np

from scripts import social_observer_evolution as observer
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_replay_selection import _journal, _ref, _save, _sha, observer_keys
from scripts.run_shelter_evolution import _save_genome, _state_arrays
from seed import background_selection as final_engine, seed, solo_selection as engine


PROTOCOL = "solo-selection-v1"
TRAIN_SEED = 2026121800
EVAL_SEED = 2026121900
VIEWS = ("background", "solo")
DEVELOPMENT_TRAINING_NAMES = ("E138-benchmark-background", "E138-benchmark-solo")
DEVELOPMENT_EVAL_NAME = "E138-benchmark-eval"
SETTINGS = (16, 4, .05, .02)
MAX_SECONDS = {"development-training": 140, "development-evaluation": 110, "training": 215, "evaluation": 295}
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds", "--hypothesis"}
SOURCE_PATHS = ("seed/seed.py", "seed/encounter_diagnostic.py", "seed/replay_selection.py", "seed/scene_transfer.py",
                "seed/background_selection.py", "seed/solo_selection.py", "scripts/social_observer_evolution.py", "scripts/run_replay_selection.py",
                "scripts/run_solo_selection.py", "scripts/run_experiment.py", "scripts/run_cold_donor.py",
                "scripts/run_shelter_evolution.py")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe", "solo_layout", "solo_action")


def _requests():
    requests = {name: ("development", "development-training", 0, view, TRAIN_SEED, 2, 0)
                for view, name in zip(VIEWS, DEVELOPMENT_TRAINING_NAMES)}
    requests[DEVELOPMENT_EVAL_NAME] = ("development", "development-evaluation", 0, None, EVAL_SEED, 0, 4)
    for index in range(3):
        for view in VIEWS:
            requests[f"E138-A{index}-{view}"] = ("final", "training", index, view, TRAIN_SEED+index+1, 10, 0)
        requests[f"E138-A{index}"] = ("final", "evaluation", index, None, EVAL_SEED+index+1, 0, 128)
    return requests


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("solo selection permits only its registered identity and ES flags")
    if (args.eval_protocol != PROTOCOL or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or args.checkpoint_every != 10
            or (args.pop, args.worlds, args.sigma, args.lr) != SETTINGS):
        raise ValueError("solo selection fixes native genome and ES settings")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("solo selection requires clean committed source")
    requests = _requests()
    if args.name not in requests:
        raise ValueError("unknown solo selection attempt")
    split, mode, index, view, master, attempts, families = requests[args.name]
    if ((args.condition_split, args.seed, args.generations, args.eval_worlds) != (split, master, attempts, families)
            or any((root / "runs").glob("*_"+args.name))):
        raise ValueError("solo selection identity differs from registration or already exists")
    source = SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("solo selection requires its original E103 parent")
    content = (root / source["path"]).read_bytes()
    if _sha(content) != source["sha256"]:
        raise ValueError("original E103 parent differs from the pinned digest")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if (flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all()
            or engine.PROTOCOL != PROTOCOL or engine.CFG != seed.Cfg(M=1)):
        raise ValueError("solo selection requires the native finite 155-number genome")
    return mode, index, view, dict(source), content, flat


def evaluation_assignments(master, count):
    if count not in (4, 128):
        raise ValueError("the only evaluation counts are four development or 128 final families")
    base = jax.random.fold_in(jax.random.PRNGKey(master), 0xE138)
    keys = np.asarray(jax.vmap(lambda i: jax.random.split(jax.random.fold_in(base, i), len(KEY_ROLES)))(
        np.arange(count, dtype=np.uint32)))
    observer_keys(set(), keys, "evaluation family role")
    pre, gap = np.tile(np.asarray([(8, 16), (16, 32), (8, 32), (16, 16)], np.int32), (count//4, 1)).T
    return dict(family_keys=keys, pre_contacts=pre, gap=gap, initial_rules=np.arange(2, dtype=np.int32))


def _endpoint_receipts(root, source_code, commit, *, development=False):
    receipts, endpoints = [], []
    attempts = 2 if development else 10
    train_mode = "development-training" if development else "training"
    for index in range(1 if development else 3):
        pair = []
        for view in VIEWS:
            name = DEVELOPMENT_TRAINING_NAMES[VIEWS.index(view)] if development else f"E138-A{index}-{view}"
            paths = list((root / "runs").glob("*_"+name))
            if len(paths) != 1:
                raise ValueError("paired training endpoints for every assigned lineage are required before evaluation")
            folder = paths[0]
            config_bytes = (folder / "config.json").read_bytes()
            summary_bytes = (folder / "summary.json").read_bytes()
            config, summary = json.loads(config_bytes), json.loads(summary_bytes)
            endpoint = (folder / "genome_final.npy").read_bytes()
            endpoint_sha = _sha(endpoint)
            flat = np.load(io.BytesIO(endpoint), allow_pickle=False)
            if (config.get("protocol") != PROTOCOL or config.get("mode") != train_mode or config.get("view") != view
                    or config.get("seed_index") != index or config.get("training_seed") != TRAIN_SEED+(0 if development else index+1)
                    or config.get("source") != SOURCES[index] or config.get("source_code") != source_code
                    or config.get("git", {}).get("commit") != commit or config.get("git", {}).get("dirty") is not False
                    or summary.get("status") != "finished" or summary.get("completed_attempts") != attempts
                    or summary.get("completed_families") != 0 or len(summary.get("attempts", ())) != attempts+1
                    or len(summary.get("samples", ())) != attempts or summary.get("endpoint_sha256") != endpoint_sha):
                raise ValueError("a training endpoint disagrees with its complete freeze receipt")
            checkpoint = _ref(folder, f"generations/attempt_{attempts:04d}.npz")
            if checkpoint != summary["attempts"][-1]:
                raise ValueError("training endpoint lacks its complete last-attempt state")
            with np.load(folder / checkpoint["path"], allow_pickle=False) as saved:
                if (flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all()
                        or saved["mu"].tobytes() != flat.tobytes() or int(saved["t"]) != summary["optimizer_updates"]):
                    raise ValueError("frozen endpoint is not the last registered ES mean")
            receipts.append(dict(seed_index=index, view=view, run=folder.relative_to(root).as_posix(),
                config_sha256=_sha(config_bytes), summary_sha256=_sha(summary_bytes),
                checkpoint_sha256=checkpoint["sha256"], endpoint_sha256=endpoint_sha))
            pair.append(endpoint)
        endpoints.append(pair)
    return receipts, endpoints


def _verify_final_release(root, receipts, commit):
    reference = json.loads((root / ".tmp/E138-offline-reference.json").read_text(encoding="utf-8-sig"))
    manifest_bytes = (root / ".tmp/E138-endpoints.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (reference.get("finalPanelsReleased") is not True
            or reference.get("releasedSource") != commit
            or reference.get("endpointManifestSha256") != _sha(manifest_bytes)
            or manifest.get("git_commit") != commit
            or manifest.get("attempt") != 10
            or manifest.get("endpoints") != receipts):
        raise ValueError("all six attempt-ten endpoint bytes must match the sealed final release")


def _verify_training_release(root, commit):
    reference = json.loads((root / ".tmp/E138-offline-reference.json").read_text(encoding="utf-8-sig"))
    if reference.get("finalTrainingReleased") is not True or reference.get("releasedSource") != commit:
        raise ValueError("final training requires an admitted matched development forecast")


def run_tracked(args, git, root, *, explicit=None):
    started, root = time.monotonic(), Path(root).resolve()
    mode, index, view, source, parent_bytes, parent = _validate(args, git, root, explicit)
    if mode == "training":
        _verify_training_release(root, git["commit"])
    source_code = {name: _sha((root / name).read_bytes()) for name in SOURCE_PATHS}
    pair, receipts, assignments = None, [], None
    training_seed = args.seed if args.generations else None
    evaluation_seed = args.seed if args.eval_worlds else None
    if evaluation_seed is not None:
        development = mode == "development-evaluation"
        receipts, endpoints = _endpoint_receipts(root, source_code, git["commit"], development=development)
        if not development:
            _verify_final_release(root, receipts, git["commit"])
        pair = endpoints[index]
        assignments = evaluation_assignments(evaluation_seed, args.eval_worlds)

    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    if args.generations:
        (folder / "generations").mkdir()
    if args.eval_worlds:
        (folder / "families").mkdir()
    manifest = _save(folder, "assignments.npz", assignments) if assignments is not None else None
    endpoint_sha = [_sha(content) for content in pair] if pair is not None else None
    config = dict(protocol=PROTOCOL, mode=mode, view=view, args=dict(vars(args), start="file:"+source["path"]), git=dict(git),
        source=source, seed_index=index, training_seed=training_seed, evaluation_seed=evaluation_seed,
        expected_attempts=args.generations, expected_families=args.eval_worlds, source_code=source_code, l2=.005,
        cfg=engine.CFG._asdict(), train_receipts=receipts, assignments=manifest, key_roles=list(KEY_ROLES),
        genotype_order=["parent", "background", "solo"],
        endpoint_sha256=endpoint_sha, max_seconds=MAX_SECONDS[mode], scientific_result=False, runtime=_runtime())
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", parent_bytes)
    setup = time.monotonic()-started
    attempts, samples, raw_files = [], [], []
    attempt_walls, family_walls = [], []
    updates = 0

    def status(value):
        return dict(status=value, protocol=PROTOCOL, mode=mode, view=view, seed_index=index, elapsed_seconds=time.monotonic()-started,
            setup_seconds=setup, completed_attempts=len(samples), completed_families=len(raw_files),
            expected_attempts=args.generations, expected_families=args.eval_worlds, optimizer_updates=updates,
            attempt_wall_seconds=attempt_walls, family_wall_seconds=family_walls)

    def summary(value):
        return dict(**status(value), scientific_result=False, gate_status="undecided", endpoint_sha256=endpoint_sha,
            attempts=attempts, samples=samples, raw_files=raw_files)

    def deadline():
        if time.monotonic()-started >= config["max_seconds"]:
            raise TimeoutError("solo selection exceeded its registered process allowance")

    _write_status(folder, status("running"))
    seen = set()
    try:
        if args.generations:
            evaluate = engine.make_evaluator(view)
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
                            raise ValueError("antithetic candidates do not share birth keys")
                        observer_keys(seen, birth_keys[:args.pop//2], "training birth")
                        batch = engine.training_batch(world_keys)
                        observer_keys(seen, batch["role_keys"], "training family role")
                        observer_keys(seen, batch["schedule_key"], "training schedule")
                        result = evaluate(candidates, batch, birth_keys)
                        both = np.asarray(result["view_fitness"])
                        if (both.shape != (args.pop, args.worlds, 2) or not np.isfinite(both).all()
                                or not np.array_equal(result["fitness"], both[..., VIEWS.index(view)])):
                            raise ValueError("only the assigned scalar may enter ES after both views are computed")
                        bounds = np.asarray(result["exposure_energy_bounds"])
                        if (not np.asarray(result["survived"]).all() or bounds.shape != (args.pop,args.worlds,2)
                                or not np.isfinite(bounds).all() or np.any(bounds[...,0] <= 0)
                                or np.any(bounds[...,1] >= engine.CFG.e_max)):
                            raise ValueError("forced training exposure must keep every body alive without clipping")
                        record.update(candidate_genomes=candidates, world_keys=world_keys, development_keys=birth_keys,
                            **{"batch_"+name: value for name,value in batch.items()}, **result)
                        return result

                    next_key, noise_key, _, _ = jax.random.split(state["key"], 4)
                    half = jax.random.normal(noise_key, (args.pop//2,parent.size))
                    before = state
                    state, metrics = observer.es_step(state, recorded, *SETTINGS, l2=.005)
                    arrays = _state_arrays(state, int(state["t"]))
                    if not np.array_equal(arrays["key"], next_key) or (metrics["all_tied"] and any(
                            not np.array_equal(arrays[name], before[name]) for name in ("mu", "m", "v", "t"))):
                        raise ValueError("invalid RNG or all-tied ES transition")
                    objective_scores = record["fitness"].mean(1)
                    solo_scores = np.asarray(record["view_fitness"][..., 1].mean(1))
                    record.update(eps=np.concatenate((half, -half)),
                                  rank_weights=observer.average_centered_ranks(objective_scores))
                    samples.append(_save(folder, f"generations/samples_{ordinal:04d}.npz", record))
                    attempts.append(_save(folder, f"generations/attempt_{ordinal:04d}.npz", arrays))
                    updates = int(arrays["t"])
                    _journal(stream, dict(attempt=ordinal, checkpoint=attempts[-1], sample=samples[-1], optimizer_updates=updates,
                        distinct_scores=metrics["distinct_scores"], objective_mean=metrics["fit_mean"], objective_best=metrics["fit_best"],
                        solo_distinct_scores=int(np.unique(solo_scores).size),
                        solo_separated_antithetic_pairs=int(np.count_nonzero(solo_scores[:8] != solo_scores[8:])),
                        changed_parameters=int(np.count_nonzero(arrays["mu"] != before["mu"]))))
                    _write_status(folder, status("running"))
                    attempt_walls.append(time.monotonic()-since)
                    print(f"E138 {view} attempt {ordinal}/{args.generations} saved", flush=True)
                _save_genome(folder / "genome_final.npy", state["mu"])
            endpoint_sha = _ref(folder, "genome_final.npy")["sha256"]
        else:
            assert assignments is not None and pair is not None
            for label, content in zip(VIEWS, pair):
                _write_bytes(folder / f"genome_{label}.npy", content)
            background, solo = (np.load(io.BytesIO(content), allow_pickle=False) for content in pair)
            with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
                for family in range(args.eval_worlds):
                    deadline()
                    since = time.monotonic()
                    pre, gap = (int(assignments[name][family]) for name in ("pre_contacts", "gap"))
                    row = final_engine.evaluate_family(parent, background, solo, assignments["family_keys"][family], family, pre, gap)
                    raw_files.append(_save(folder, f"families/family_{family:04d}.npz", row))
                    _journal(stream, dict(index=family, raw_file=raw_files[-1]))
                    _write_status(folder, status("running"))
                    family_walls.append(time.monotonic()-since)
                    print(f"E138 family {family+1}/{args.eval_worlds} saved", flush=True)
        deadline()
        _write_json(folder / "summary.json", summary("finished"))
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        _write_json(folder / "summary.json", dict(summary(value), error=f"{type(error).__name__}: {error}"))
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder, flush=True)
    return folder
