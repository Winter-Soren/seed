"""E126 immutable segmented reversal training and locked paired final evaluation."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

from seed import direct
from seed.seed import Cfg
from scripts import reversal_evolution as engine
from scripts.run_cold_donor import _runtime, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays, _save_genome


PROTOCOL = "reversal-evolution-v1"
TOTAL_ATTEMPTS = 200
SEGMENT_ATTEMPTS = 50
DEVELOPMENT_ATTEMPTS = 3
FINAL_WORLDS = 128
POP, PAIRS, SIGMA, LR, L2 = 128, 4, .1, .1, .005
SOFT_SECONDS = {"development": 780, "final": 3000}
HARD_SECONDS = {"development": 900, "final": 3600}
CONTROLLERS = {"plastic": "rules-v0", "direct": direct.FORMAT}
DEV_TAGS = {"plastic": 0xE12600, "direct": 0xE12601}
STATE_FIELDS = ("mu", "m", "v", "t", "key")
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed", "--controller",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds",
                 "--checkpoint-every", "--max-train-seconds", "--resume", "--hypothesis"}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _artifact(folder, relative):
    return dict(path=relative, sha256=_sha((folder / relative).read_bytes()))


def _read_artifact(folder, artifact):
    path = (folder / artifact["path"]).resolve()
    if not path.is_relative_to(folder.resolve()) or path == folder.resolve():
        raise ValueError("artifact path leaves its immutable segment")
    raw = path.read_bytes()
    if _sha(raw) != artifact["sha256"]:
        raise ValueError("immutable segment artifact hash differs")
    return raw


def _run_path(root, relative):
    folder = (root / relative).resolve()
    if folder.parent != (root / "runs").resolve():
        raise ValueError("a source segment must be a direct child of runs/")
    return folder


def _state(state, attempt, size, prefix=""):
    value = {field: np.asarray(state[prefix + field]).copy() for field in STATE_FIELDS}
    if (any(value[field].shape != (size,) or value[field].dtype != np.float32
            or not np.isfinite(value[field]).all() for field in ("mu", "m", "v"))
            or value["t"].shape != () or value["t"].dtype.kind not in "iu"
            or not 0 <= int(value["t"]) <= attempt or np.any(value["v"] < 0)
            or value["key"].shape != (2,) or value["key"].dtype != np.uint32):
        raise ValueError("checkpoint requires finite registered vectors, variance, update counter and RNG key")
    return dict(**value, attempted_round=np.asarray(attempt, np.int64))


def _device_state(arrays):
    return {field: int(arrays[field]) if field == "t" else jnp.asarray(arrays[field]) for field in STATE_FIELDS}


def _register_keys(seen, sampling):
    world, births = np.asarray(sampling["world_keys"]), np.asarray(sampling["development_keys"])
    assignment = np.asarray(sampling["assignment_key"])
    if (world.shape != (PAIRS, 2) or world.dtype != np.uint32
            or births.shape != (POP, PAIRS, 2) or births.dtype != np.uint32
            or assignment.shape != (2,) or assignment.dtype != np.uint32
            or not np.array_equal(births[:POP // 2], births[POP // 2:])):
        raise ValueError("each round must keep four worlds and shared antithetic development keys")
    values = np.concatenate((world, births[:POP // 2].reshape(-1, 2),
                             assignment.reshape(1, 2)))
    tokens = [value.tobytes() for value in values]
    if len(set(tokens)) != len(tokens) or seen.intersection(tokens):
        raise ValueError("training attempted rounds reused world, birth or schedule keys")
    seen.update(tokens)


def _provenance(root):
    paths = list((root / "seed").glob("*.py")) + [root / "scripts" / name for name in
        ("reversal_evolution.py", "run_reversal_evolution.py", "social_observer_evolution.py",
         "social_reversal.py", "social_learning.py",
         "run_cold_donor.py", "run_shelter_evolution.py")]
    prng = dict(implementation=str(jax.config.jax_default_prng_impl),
                threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
                enable_x64=bool(jax.config.jax_enable_x64), backend=jax.default_backend())
    return dict(runtime=_runtime(), prng=prng, code={path.relative_to(root).as_posix(): _sha(path.read_bytes())
                                         for path in sorted(paths)})


def _settings(provenance):
    return dict(protocol=PROTOCOL, cfg=Cfg(M=1)._asdict(), pop=POP, base_world_pairs=PAIRS,
        actual_lifetimes=2 * PAIRS, sigma=SIGMA, lr=LR, l2=L2, total_attempts=TOTAL_ATTEMPTS,
        flip_steps=[100, 125, 150, 175], objective="mean of individual paired-rule minimum late food-minus-poison",
        late_window=[50, 100], tie_rule="average centered ranks; all ties consume attempt without optimizer update",
        **provenance)


def _identity(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("reversal evolution accepts only its registered settings and immutable continuation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in HARD_SECONDS
            or args.quick or args.allow_dirty or args.cfg or args.no_plastic
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.start != "blank" or args.eval_worlds != FINAL_WORLDS or args.checkpoint_every != 10):
        raise ValueError("reversal evolution fixes blank starts, world geometry and scientific settings")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str)
            or not re.fullmatch("[0-9a-f]{40}", commit)):
        raise ValueError("reversal evolution requires clean committed code provenance")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this attempt exists; continuation must use a new registered segment name")
    match = re.fullmatch(r"E126-eval-s([012])", args.name)
    if match:
        index = int(match[1])
        if (args.condition_split != "final" or args.seed != 2026120501 + index or args.generations != 0
                or args.resume or args.controller is not None or args.max_train_seconds is not None
                or (args.pop, args.worlds, args.sigma, args.lr) != (128, 8, .1, .1)):
            raise ValueError("paired final evaluation requires its exact separate evaluation command")
        return dict(stage="evaluation", seed_index=index)
    development = args.condition_split == "development"
    pattern = r"E126-benchmark-(plastic|direct)" if development else r"(E126-(plastic|direct)-s([012]))-part(\d{3})"
    match = re.fullmatch(pattern, args.name)
    if not match:
        raise ValueError("training requires a registered architecture benchmark or numbered target segment")
    architecture = match[1] if development else match[2]
    index, part = (None, 1) if development else (int(match[3]), int(match[4]))
    target = args.name if development else match[1]
    expected_seed = 2026120400 if development else 2026120401 + index
    expected_attempts = DEVELOPMENT_ATTEMPTS if development else TOTAL_ATTEMPTS
    if (args.seed != expected_seed or args.generations != expected_attempts or part < 1
            or args.controller != CONTROLLERS[architecture]
            or (args.pop, args.worlds, args.sigma, args.lr) != (POP, PAIRS, SIGMA, LR)
            or args.max_train_seconds != SOFT_SECONDS[args.condition_split]
            or bool(args.resume) != (part > 1) or development and args.resume):
        raise ValueError("training seed, architecture, exposure, segment or time allowance differs from registration")
    return dict(stage="training", architecture=architecture, controller=CONTROLLERS[architecture],
                seed_index=index, part=part, target=target, expected_attempts=expected_attempts)


def _segments(root, target):
    result = {}
    for folder in (root / "runs").glob("*_" + target + "-part*"):
        match = re.fullmatch(re.escape(target) + r"-part(\d{3})", folder.name.split("_", 1)[-1])
        if not match or int(match[1]) in result:
            raise ValueError("target has an ambiguous or malformed segment chain")
        result[int(match[1])] = folder
    if result and sorted(result) != list(range(1, max(result) + 1)):
        raise ValueError("target segments must form one consecutive immutable chain")
    return result


def _load_chain(root, target, settings):
    segments = _segments(root, target)
    if not segments:
        raise ValueError("target has no registered training segment")
    previous, state, attempt, seen, founder = None, None, 0, set(), None
    source_controller, source_seed = None, None
    for part, folder in sorted(segments.items()):
        raw_config = (folder / "config.json").read_bytes()
        config = json.loads(raw_config)
        if (config.get("protocol") != PROTOCOL or config.get("stage") != "training"
                or config.get("condition_split") != "final" or config.get("target") != target
                or config.get("part") != part or config.get("start_attempt") != attempt
                or config.get("expected_attempts") != TOTAL_ATTEMPTS or config.get("settings") != settings
                or config.get("git", {}).get("dirty") is not False or config.get("parent") != previous):
            raise ValueError("source chain changed its protocol, settings, provenance or predecessor")
        if part == 1:
            source_controller, source_seed = config["controller"], config["seed"]
            founder = config["founder"]
            if (founder.get("folder") != folder.relative_to(root).as_posix()
                    or founder.get("master_seed") != source_seed
                    or founder.get("development_stream_tag") is not None
                    or founder.get("checkpoint") != config["start_checkpoint"]):
                raise ValueError("target founder must be its original fresh untagged initialization")
            initial_genome = np.load(io.BytesIO(_read_artifact(folder, founder["genome"])), allow_pickle=False)
        if config.get("controller") != source_controller or config.get("seed") != source_seed or config.get("founder") != founder:
            raise ValueError("target controller, master seed or initializer changed between segments")
        size = config["genome_size"]
        start_artifact = config["start_checkpoint"]
        with np.load(io.BytesIO(_read_artifact(folder, start_artifact)), allow_pickle=False) as saved:
            initial = _state(saved, attempt, size)
            if int(saved["attempted_round"]) != attempt:
                raise ValueError("segment start checkpoint has the wrong attempted round")
        if part == 1 and (not np.array_equal(initial["mu"], initial_genome)
                         or np.any(initial["m"]) or np.any(initial["v"]) or int(initial["t"]) != 0):
            raise ValueError("target founder genome and zero optimizer state disagree")
        if state is not None and any(not np.array_equal(initial[field], state[field]) for field in STATE_FIELDS):
            raise ValueError("continuation did not retain exact optimizer state")
        state = initial
        journal = (folder / "metrics.jsonl").read_bytes() if (folder / "metrics.jsonl").exists() else b""
        lines = journal.splitlines(keepends=True)
        complete = lines if not lines or lines[-1].endswith(b"\n") else lines[:-1]
        last_artifact = start_artifact
        for line in complete:
            record = json.loads(line)
            if record.get("attempt") != attempt + 1:
                raise ValueError("completed attempt journal has a gap or repeated attempt")
            raw = _read_artifact(folder, record["samples"])
            with np.load(io.BytesIO(raw), allow_pickle=False) as sample:
                next_state = _state(sample, attempt + 1, size, "next_")
                if (int(sample["attempted_round"]) != attempt + 1
                        or int(next_state["t"]) != int(state["t"]) + int(record["informative"])
                        or int(record["optimizer_updates"]) != int(next_state["t"])):
                    raise ValueError("attempt counter and informative optimizer counter disagree")
                _register_keys(seen, sample)
            state, attempt, last_artifact = next_state, attempt + 1, record["samples"]
        if attempt - config["start_attempt"] > SEGMENT_ATTEMPTS or attempt > TOTAL_ATTEMPTS:
            raise ValueError("source segment exceeded the fixed attempted-round budget")
        previous = dict(folder=folder.relative_to(root).as_posix(), config_sha256=_sha(raw_config),
            journal_sha256=_sha(journal), completed_attempt=attempt, terminal_state=last_artifact,
            terminal_state_kind="checkpoint" if last_artifact == start_artifact else "samples",
            ignored_unflushed_journal_bytes=len(journal) - sum(len(line) for line in complete))
    return dict(state=state, attempt=attempt, parent=previous, seen=seen, founder=founder,
                part=max(segments), controller=source_controller, seed=source_seed, target=target)


def _sealed_target(root, chain):
    """A complete journal is recoverable; only a verified terminal seal releases evaluation."""
    folder = _run_path(root, chain["parent"]["folder"])
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    if (chain["attempt"] != TOTAL_ATTEMPTS or summary.get("status") != "target_finished"
            or summary.get("artifact_validity") != "passed" or summary.get("protocol") != PROTOCOL
            or summary.get("condition_split") != "final" or summary.get("target") != chain["target"]
            or summary.get("controller") != chain["controller"] or summary.get("training_seed") != chain["seed"]
            or summary.get("attempted_rounds") != TOTAL_ATTEMPTS or summary.get("expected_attempts") != TOTAL_ATTEMPTS
            or summary.get("optimizer_updates") != int(chain["state"]["t"])
            or chain["parent"]["ignored_unflushed_journal_bytes"] != 0):
        return None
    with np.load(io.BytesIO(_read_artifact(folder, summary["terminal_checkpoint"])), allow_pickle=False) as saved:
        terminal = _state(saved, TOTAL_ATTEMPTS, chain["state"]["mu"].size)
        if int(saved["attempted_round"]) != TOTAL_ATTEMPTS:
            raise ValueError("sealed terminal checkpoint has the wrong attempted-round counter")
    genome = np.load(io.BytesIO(_read_artifact(folder, summary["terminal_genome"])), allow_pickle=False)
    if (any(not np.array_equal(terminal[field], chain["state"][field]) for field in STATE_FIELDS)
            or genome.dtype != np.float32 or not np.array_equal(genome, terminal["mu"])):
        raise ValueError("sealed terminal artifacts disagree with the last completed journal state")
    return dict(summary=_artifact(folder, "summary.json"), terminal_checkpoint=summary["terminal_checkpoint"],
                terminal_genome=summary["terminal_genome"])


def _all_endpoints(root, settings):
    endpoints = {}
    for index in range(3):
        for architecture, controller in CONTROLLERS.items():
            target = f"E126-{architecture}-s{index}"
            chain = _load_chain(root, target, settings)
            chain["sealed"] = _sealed_target(root, chain)
            if (chain["attempt"] != TOTAL_ATTEMPTS or chain["controller"] != controller
                    or chain["seed"] != 2026120401 + index or chain["sealed"] is None):
                raise ValueError("all six sealed exact attempt-200 endpoints must exist before any held-out world opens")
            endpoints[target] = chain
    return endpoints


def _folder(root, name):
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + name)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _train(args, git, root, identity, settings):
    started = time.monotonic()
    development = args.condition_split == "development"
    target, controller = identity["target"], identity["controller"]
    size = 155 if controller == "rules-v0" else direct.genome_size(Cfg(M=1))
    parent, seen, founder = None, set(), None
    if args.resume:
        chain = _load_chain(root, target, settings)
        if (_run_path(root, args.resume) != root / chain["parent"]["folder"]
                or identity["part"] != chain["part"] + 1 or chain["attempt"] > TOTAL_ATTEMPTS
                or chain["controller"] != controller or chain["seed"] != args.seed):
            raise ValueError("continuation must extend the unique unfinished target tip")
        if chain["attempt"] == TOTAL_ATTEMPTS and _sealed_target(root, chain) is not None:
            raise ValueError("a sealed complete target cannot be extended or evolved again")
        state, attempt = _device_state(chain["state"]), chain["attempt"]
        parent, seen, founder = chain["parent"], chain["seen"], chain["founder"]
    else:
        if not development and _segments(root, target):
            raise ValueError("a target initializer already exists")
        state, attempt = engine.initialize_state(controller, args.seed, Cfg(M=1)), 0
        if development:
            state["key"] = jax.random.fold_in(state["key"], DEV_TAGS[identity["architecture"]])
        initial = _state(state, 0, size)
        if int(initial["t"]) != 0 or np.any(initial["m"]) or np.any(initial["v"]):
            raise ValueError("new targets must start with zero Adam moments and no updates")
    start_attempt = attempt
    folder = _folder(root, args.name)
    (folder / "samples").mkdir()
    (folder / "checkpoints").mkdir()
    start_checkpoint = _save_arrays(folder, "checkpoints/start.npz", _state(state, attempt, size))
    _save_genome(folder / "genome_start.npy", state["mu"])
    if founder is None:
        founder = dict(folder=folder.relative_to(root).as_posix(), genome=_artifact(folder, "genome_start.npy"),
                       checkpoint=start_checkpoint, master_seed=args.seed,
                       initialization="blank rules" if controller == "rules-v0" else "normal/sqrt(156)",
                       development_stream_tag=DEV_TAGS[identity["architecture"]] if development else None)
    config = dict(protocol=PROTOCOL, args=vars(args).copy(), git=dict(git), condition_split=args.condition_split,
        **identity, seed=args.seed, genome_size=size, settings=settings, parent=parent, founder=founder,
        start_checkpoint=start_checkpoint, start_attempt=start_attempt,
        max_segment_attempts=DEVELOPMENT_ATTEMPTS if development else SEGMENT_ATTEMPTS,
        soft_seconds=SOFT_SECONDS[args.condition_split], max_seconds=HARD_SECONDS[args.condition_split],
        sealing_recovery=bool(parent is not None and start_attempt == TOTAL_ATTEMPTS),
        scientific_result=False, evaluation="locked until all six target endpoints reach attempt 200")
    _write_json(folder / "config.json", config)
    records, checkpoints, durations = [], [], []
    setup_seconds = time.monotonic() - started

    def status(value):
        return dict(status=value, stage="training", attempted_rounds=attempt, optimizer_updates=int(state["t"]),
            start_attempt=start_attempt, completed_segment_attempts=attempt - start_attempt,
            expected_attempts=identity["expected_attempts"], elapsed_seconds=time.monotonic() - started,
            setup_seconds=setup_seconds, attempt_seconds=list(durations))

    def checkpoint():
        relative = f"checkpoints/attempt_{attempt:04d}.npz"
        if (folder / relative).exists():
            return _artifact(folder, relative)
        artifact = _save_arrays(folder, relative, _state(state, attempt, size))
        checkpoints.append(artifact)
        return artifact

    def finish(value, error=None):
        terminal = checkpoint()
        _save_genome(folder / "genome_terminal.npy", state["mu"])
        result = dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
            target=target, controller=controller, training_seed=args.seed, seed_index=identity["seed_index"],
            scientific_result=False, gate_status="undecided", artifact_validity="passed" if error is None else "incomplete",
            attempts=list(records), checkpoints=list(checkpoints), terminal_checkpoint=terminal,
            terminal_genome=_artifact(folder, "genome_terminal.npy"), parent=parent, founder=founder)
        if error is not None:
            result["error"] = f"{type(error).__name__}: {error}"
        _write_json(folder / "summary.json", result)
        _write_status(folder, status(value))

    _write_status(folder, status("training"))
    try:
        evaluate = engine.make_evaluator(controller, Cfg(M=1)) if attempt < identity["expected_attempts"] else None
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            limit = identity["expected_attempts"] if development else min(identity["expected_attempts"], start_attempt + SEGMENT_ATTEMPTS)
            while attempt < limit:
                elapsed = time.monotonic() - started
                bound = 1.5 * max(durations[1:] or durations) if durations else 0.
                if elapsed + bound + 120 >= SOFT_SECONDS[args.condition_split]:
                    break
                round_started = time.monotonic()
                sample = {key: np.asarray(value) for key, value in engine.sampling(state, POP, PAIRS, SIGMA).items()}
                if "half_eps" in sample or "candidates" in sample:
                    raise ValueError("sampling evidence must retain PRNG keys and digests, not deterministic noise tensors")
                _register_keys(seen, sample)
                result_holder = {}

                def recorded_evaluate(candidates, world_keys, development_keys):
                    if (result_holder or not np.array_equal(world_keys, sample["world_keys"])
                            or not np.array_equal(development_keys, sample["development_keys"])
                            or not np.array_equal(engine.candidate_hashes(candidates), sample["candidate_sha256"])):
                        raise ValueError("actual candidate or world sampling differs from the registered ES draw")
                    result = evaluate(candidates, world_keys, development_keys)
                    result_holder.update(engine.pack_evidence(result))
                    engine.validate_training_evidence(result_holder, sample["flip_steps"], Cfg(M=1))
                    return result

                next_state, metrics = engine.es_step(state, recorded_evaluate, POP, PAIRS, SIGMA, LR, L2)
                jax.block_until_ready(next_state)
                arrays = _state(next_state, attempt + 1, size)
                if not np.array_equal(arrays["key"], sample["next_key"]):
                    raise ValueError("attempt did not advance exactly one ES random-stream step")
                scores = np.asarray(result_holder["fitness"]).mean(1)
                informative = bool(np.any(scores != scores[0]))
                if int(arrays["t"]) != int(state["t"]) + int(informative):
                    raise ValueError("optimizer updates must count informative attempts only")
                if not informative and any(not np.array_equal(arrays[field], state[field]) for field in ("mu", "m", "v", "t")):
                    raise ValueError("all-tied round changed genome, moments or update counter")
                sample.update(result_holder)
                sample.update({"next_" + field: arrays[field] for field in STATE_FIELDS})
                sample.update(attempted_round=np.asarray(attempt + 1, np.int64),
                              rank_weights=np.asarray(engine.average_centered_ranks(scores)))
                artifact = _save_arrays(folder, f"samples/attempt_{attempt + 1:04d}.npz", sample)
                duration = time.monotonic() - round_started
                record = dict(metrics, attempt=attempt + 1, informative=informative,
                    optimizer_updates=int(arrays["t"]), samples=artifact, iteration_seconds=duration,
                    changed_parameters=int(np.count_nonzero(arrays["mu"] != np.asarray(state["mu"]))),
                    elapsed_seconds=time.monotonic() - started)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                state, attempt = next_state, attempt + 1
                records.append(record)
                durations.append(duration)
                if attempt % 10 == 0:
                    checkpoint()
                _write_status(folder, status("training"))
                print(f"{target} attempt {attempt}/{identity['expected_attempts']} updates {int(state['t'])}", flush=True)
                if time.monotonic() - started >= HARD_SECONDS[args.condition_split]:
                    raise TimeoutError("reversal training exceeded its immutable hard invocation cap")
        finish("target_finished" if attempt == identity["expected_attempts"] else "segment_finished")
    except BaseException as error:
        if not (folder / "summary.json").exists():
            finish("interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed", error)
        raise
    print("run folder:", folder)
    return folder


def _evaluate(args, git, root, identity, settings):
    started = time.monotonic()
    endpoints = _all_endpoints(root, settings)
    index = identity["seed_index"]
    plastic = endpoints[f"E126-plastic-s{index}"]["state"]["mu"]
    recurrent = endpoints[f"E126-direct-s{index}"]["state"]["mu"]
    sources = {name: dict(**value["parent"], sealed=value["sealed"]) for name, value in endpoints.items()}
    schedule = {name: np.asarray(value) for name, value in engine.evaluation_schedule(args.seed, FINAL_WORLDS).items()}
    if (set(schedule) != {"key", "permutation", "flip_steps"}
            or schedule["key"].shape != (2,) or schedule["key"].dtype != np.uint32
            or schedule["permutation"].shape != (FINAL_WORLDS,) or schedule["permutation"].dtype != np.int32
            or not np.array_equal(np.sort(schedule["permutation"]), np.arange(FINAL_WORLDS))
            or not np.array_equal(schedule["flip_steps"], np.repeat([100, 125, 150, 175], FINAL_WORLDS // 4)[schedule["permutation"]])):
        raise ValueError("held-out schedule must preserve equal independent base counts per flip time")
    folder = _folder(root, args.name)
    (folder / "worlds").mkdir()
    assignment = _save_arrays(folder, "assignments.npz", schedule)
    _save_genome(folder / "plastic_genome.npy", plastic)
    _save_genome(folder / "direct_genome.npy", recurrent)
    config = dict(protocol=PROTOCOL, args=vars(args).copy(), git=dict(git), **identity, condition_split="final",
        settings=settings, sources=sources, eval_seed=args.seed, expected_worlds=FINAL_WORLDS,
        assignments=assignment, max_seconds=HARD_SECONDS["final"], scientific_result=False,
        scientific_gate="requires all three complete paired panels and all six exact attempt-200 endpoints")
    _write_json(folder / "config.json", config)
    rows, artifacts, durations = [], [], []
    seen = set().union(*(value["seen"] for value in endpoints.values()))
    if schedule["key"].tobytes() in seen:
        raise ValueError("held-out schedule key overlaps training")
    seen.add(schedule["key"].tobytes())

    def status(value):
        return dict(status=value, stage="evaluation", completed_worlds=len(artifacts), expected_worlds=FINAL_WORLDS,
                    elapsed_seconds=time.monotonic() - started, world_seconds=list(durations))

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split="final", seed_index=index,
            evaluation_seed=args.seed, scientific_result=False, gate_status="undecided",
            artifact_validity="incomplete", sources=sources, raw_files=list(artifacts))

    _write_status(folder, status("evaluation"))
    try:
        with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
            for world in range(FINAL_WORLDS):
                if time.monotonic() - started >= HARD_SECONDS["final"]:
                    raise TimeoutError("paired evaluation exceeded its hard invocation allowance")
                world_started = time.monotonic()
                keys = np.asarray(engine.evaluation_keys(args.seed, world))
                tokens = [key.tobytes() for key in keys.reshape(-1, 2)]
                if keys.shape != (2, 2) or keys.dtype != np.uint32 or len(set(tokens)) != len(tokens) or seen.intersection(tokens):
                    raise ValueError("held-out base worlds must use distinct assigned keys")
                seen.update(tokens)
                flip = int(schedule["flip_steps"][world])
                row = engine.evaluate_world(plastic, recurrent, keys, world, flip, Cfg(M=1))
                projected = engine.summary_input(row)
                if (not np.array_equal(row["keys"], keys) or int(row["base_index"]) != world or int(row["flip_step"]) != flip
                        or not np.array_equal(row["plastic_genome_sha256"], engine.array_hash(plastic))
                        or not np.array_equal(row["direct_genome_sha256"], engine.array_hash(recurrent))):
                    raise ValueError("held-out world differs from its fixed assignment")
                artifact = dict(index=world, **_save_arrays(folder, f"worlds/world_{world:04d}.npz", row))
                artifacts.append(artifact)
                rows.append(projected)
                durations.append(time.monotonic() - world_started)
                stream.write(json.dumps(dict(world=world, flip_step=flip, keys=keys.tolist(), raw_file=artifact,
                    elapsed_seconds=time.monotonic() - started, world_seconds=durations[-1]), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("evaluation"))
                del row
        measurements = engine.summarize(rows)
        final = summary("finished")
        final.update(measurements=measurements, artifact_validity="passed")
        if time.monotonic() - started >= HARD_SECONDS["final"]:
            raise TimeoutError("paired evaluation exceeded its hard invocation allowance")
        _write_json(folder / "summary.json", final)
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        if not (folder / "summary.json").exists():
            partial = summary(value)
            partial["error"] = f"{type(error).__name__}: {error}"
            _write_json(folder / "summary.json", partial)
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder


def build_comparison(root):
    """Aggregate only complete frozen panels; never run or select another checkpoint."""
    root = Path(root)
    panels, manifests, common_sources, common_settings = [], [], None, None
    for index in range(3):
        folders = list((root / "runs").glob(f"*_E126-eval-s{index}"))
        if len(folders) != 1 or not (folders[0] / "summary.json").exists():
            return dict(status="undecided", reason="three complete paired panels are required")
        folder = folders[0]
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        if (config.get("protocol") != PROTOCOL or config.get("stage") != "evaluation"
                or config.get("git", {}).get("dirty") is not False
                or result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_worlds") != FINAL_WORLDS or result.get("expected_worlds") != FINAL_WORLDS
                or config.get("seed_index") != index or config.get("eval_seed") != 2026120501 + index
                or len(result.get("raw_files", [])) != FINAL_WORLDS):
            return dict(status="undecided", reason="a paired panel is incomplete or invalid")
        if common_settings is not None and config.get("settings") != common_settings:
            raise ValueError("paired panels changed their registered settings, runtime or scientific source")
        common_settings = config.get("settings")
        sources = config.get("sources", {})
        expected = {f"E126-{architecture}-s{seed_index}" for architecture in CONTROLLERS for seed_index in range(3)}
        if (set(sources) != expected or any(source.get("completed_attempt") != TOTAL_ATTEMPTS for source in sources.values())
                or common_sources is not None and sources != common_sources):
            raise ValueError("paired panels must freeze the same six complete target endpoints")
        for source in sources.values():
            if set(source.get("sealed", {})) != {"summary", "terminal_checkpoint", "terminal_genome"}:
                raise ValueError("paired panels require all six immutable terminal seals")
            for artifact in source["sealed"].values():
                _read_artifact(_run_path(root, source["folder"]), artifact)
        common_sources = sources
        for artifact in result["raw_files"]:
            _read_artifact(folder, artifact)
        panels.append(dict(seed_index=index, **result["measurements"]))
        manifests.append(dict(folder=folder.relative_to(root).as_posix(), config=_artifact(folder, "config.json"),
                              summary=_artifact(folder, "summary.json")))
    result = engine.aggregate(panels)
    return dict(status="complete", protocol=PROTOCOL, sources=common_sources, panels=manifests, **result)


def run_tracked(args, git, root, *, explicit=None):
    root = Path(root).resolve()
    identity = _identity(args, git, root, explicit)
    settings = _settings(_provenance(root))
    if identity["stage"] == "training":
        return _train(args, git, root, identity, settings)
    return _evaluate(args, git, root, identity, settings)
