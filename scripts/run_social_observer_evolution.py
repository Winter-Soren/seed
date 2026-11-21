"""E122 fixed-demonstrator observer evolution with immutable attempted rounds."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from seed import seed
from scripts import social_observer_evolution as observer
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays, _save_genome, _state_arrays


PROTOCOL = "social-observer-train-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds", "--hypothesis"}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("social-observer-train-v1 accepts only its registered run and evolution flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or args.checkpoint_every != 10
            or (args.pop, args.worlds, args.sigma, args.lr) != (16, 4, .05, .02)):
        raise ValueError("observer evolution requires its fixed solo observation and ES settings")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("observer evolution requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected, evaluation_seed = SOURCES[0], ("E122-benchmark", 49, 1, 4), 50
    else:
        selected = [item for item in SOURCES if args.name == "E122-" + item["label"]]
        if len(selected) != 1:
            raise ValueError("final observer evolution must be named E122-A0, E122-A1 or E122-A2")
        source = selected[0]
        index = source["training_seed"]
        expected = ("E122-" + source["label"], 2026112801 + index, 10, 128)
        evaluation_seed = 2026112901 + index
    if (args.name, args.seed, args.generations, args.eval_worlds) != expected:
        raise ValueError("observer identity, seeds and budgets differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this observer attempt exists; preserve it and register any recovery")
    source_path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != source_path:
        raise ValueError("observer evolution requires its matching original E103 genome")
    content = source_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("original E103 genome differs from its registered SHA256")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original genome must contain 155 finite float32 values")
    return dict(source), content, flat, evaluation_seed


def _register_keys(seen, keys, label):
    values = np.asarray(keys)
    if values.dtype != np.uint32 or values.shape[-1:] != (2,):
        raise ValueError(f"invalid {label} keys")
    tokens = [key.tobytes() for key in values.reshape(-1, 2)]
    if len(set(tokens)) != len(tokens) or seen.intersection(tokens):
        raise ValueError(f"reused {label} keys")
    seen.update(tokens)


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root)
    source, content, starting, evaluation_seed = _validate(args, git, root, explicit)
    teacher = starting.copy()
    c = seed.Cfg(M=1)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("generations", "teachers", "families"):
        (folder / name).mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, branch="O", args=recorded_args, git=dict(git),
        source=source, teacher_source=dict(source), condition_split=args.condition_split,
        cfg=c._asdict(), cfg_overrides={}, ecology=None, persistence="off", plastic=True,
        controller="rules-v0", genome_size=155, seed=args.seed, evaluation_seed=evaluation_seed,
        eval_worlds=args.eval_worlds, expected_families=args.eval_worlds, expected_attempts=args.generations,
        arms=list(observer.ARMS), controls=list(observer.CONTROLS), practice_steps=c.T // 2,
        watch_steps=c.T // 2, solo_steps=c.T, l2=.005, initial_optimizer="zero moments and update counter",
        objective="absolute observed first-food success over all assigned training families",
        selection="endpoint after all registered attempted rounds",
        tie_rule="average tied ranks; all-tied batch advances RNG and preserves mu, m, v and t",
        rule_assignment="family index modulo 2", teacher="fixed original genome for every attempted round",
        training_key_order=["teacher_world", "teacher_birth", "solo_world"],
        evaluation_key_order=["teacher_world", "teacher_birth", "solo_world", "pupil_birth"],
        max_seconds=MAX_SECONDS[args.condition_split], scientific_result=args.condition_split == "final",
        runtime=_runtime())
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    _write_bytes(folder / "teacher_genome.npy", content)
    attempt, updates, families = 0, 0, []
    attempts, family_files, trajectory, sampling, batch_holder = [], [], [], {}, {}
    seen_keys = set()

    def status(value):
        return dict(status=value, attempted_rounds=attempt, expected_attempts=args.generations,
                    optimizer_updates=updates, completed_families=len(family_files),
                    expected_families=args.eval_worlds, elapsed_seconds=time.monotonic() - started)

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
                    source_label=source["label"], training_seed=args.seed, evaluation_seed=evaluation_seed,
                    scientific_result=False, artifact_validity="incomplete", gate_status="undecided",
                    attempts=list(attempts), raw_files=list(family_files))

    def deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("observer evolution exceeded its registered process allowance")

    _write_status(folder, status("training"))
    try:
        evaluate = observer.make_evaluator(c)

        def recorded_evaluate(candidates, world_keys, development_keys):
            if batch_holder:
                raise ValueError("an attempted round must generate its teacher batch exactly once")
            if not np.array_equal(np.asarray(development_keys)[:args.pop // 2],
                                  np.asarray(development_keys)[args.pop // 2:]):
                raise ValueError("antithetic candidates must share pupil development keys")
            _register_keys(seen_keys, world_keys, "training base-world")
            _register_keys(seen_keys, development_keys[:args.pop // 2], "training pupil development")
            batch = observer.make_training_batch(teacher, world_keys, c)
            if (not np.array_equal(batch["teacher_genome"], starting)
                    or not np.array_equal(batch["world_keys"], np.asarray(world_keys))
                    or not np.array_equal(batch["rules"], np.arange(args.worlds) % 2)
                    or not np.array_equal(batch["solo_world_keys"], batch["role_keys"][:, 2])):
                raise ValueError("training movie source or assignment differs from its fixed teacher")
            _register_keys(seen_keys, batch["role_keys"], "training teacher and solo world")
            batch_holder.update(batch)
            result = evaluate(candidates, batch["movies"], batch["solo_world_keys"], development_keys, batch["rules"])
            sampling.update(candidate_genomes=np.asarray(candidates), world_keys=np.asarray(world_keys),
                            development_keys=np.asarray(development_keys),
                            **{name: np.asarray(value) for name, value in result.items()})
            return result

        state = seed.es_init(c, mean=starting, seed=args.seed)
        arrays = _state_arrays(state, 0)
        if (not np.array_equal(arrays["mu"], starting) or np.any(arrays["m"]) or np.any(arrays["v"])):
            raise ValueError("observer evolution must begin at its original genome with zero moments")
        attempts.append(_save_arrays(folder, "generations/attempt_0000.npz", arrays))
        trajectory.append(arrays["mu"].copy())
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for next_attempt in range(1, args.generations + 1):
                deadline()
                before = {name: np.asarray(value).copy() for name, value in state.items()}
                next_key, noise_key, _, _ = jax.random.split(state["key"], 4)
                half_noise = np.asarray(jax.random.normal(noise_key, (args.pop // 2, starting.size)))
                sampling.clear()
                batch_holder.clear()
                sampling["eps"] = np.concatenate((half_noise, -half_noise))
                state, record = observer.es_step(state, recorded_evaluate, args.pop, args.worlds, args.sigma, args.lr)
                scores = np.asarray(sampling["fitness"]).mean(1)
                sampling["rank_weights"] = np.asarray(observer.average_centered_ranks(scores))
                informative = bool(np.any(scores != scores[0]))
                arrays = _state_arrays(state, updates + int(informative))
                if not np.array_equal(arrays["key"], np.asarray(next_key)):
                    raise ValueError("attempted round did not advance the registered ES random stream")
                if not informative and any(not np.array_equal(arrays[name], before[name]) for name in ("mu", "m", "v", "t")):
                    raise ValueError("an all-tied batch changed its optimizer or genome")
                teacher_artifact = _save_arrays(folder, f"teachers/teacher_{next_attempt:04d}.npz", batch_holder)
                samples = _save_arrays(folder, f"generations/samples_{next_attempt:04d}.npz", sampling)
                checkpoint = _save_arrays(folder, f"generations/attempt_{next_attempt:04d}.npz", arrays)
                attempts.append(checkpoint)
                trajectory.append(arrays["mu"].copy())
                attempt, updates = next_attempt, int(arrays["t"])
                record = dict(record, attempt=attempt, informative=informative, optimizer_updates=updates,
                              checkpoint=checkpoint, samples=samples, teacher=teacher_artifact,
                              changed_parameters=int(np.count_nonzero(arrays["mu"] != before["mu"])),
                              elapsed_seconds=time.monotonic() - started)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("training"))
                print(f"observer attempt {attempt}/{args.generations}, informative updates {updates} saved", flush=True)
                deadline()
        evolved = np.asarray(state["mu"]).copy()
        _save_genome(folder / "genome_final.npy", evolved)
        _save_genome(folder / "genome_trajectory.npy", np.stack(trajectory))
        _write_status(folder, status("evaluating"))
        with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                deadline()
                keys = np.asarray(observer.family_keys(evaluation_seed, index))
                _register_keys(seen_keys, keys, "held-out family")
                row = observer.evaluate_family(teacher, starting, evolved, keys, index, c)
                observer.validate_family(row, c)
                if (not np.array_equal(row["keys"], keys) or int(row["family_index"]) != index
                        or int(row["rule"]) != index % 2 or int(row["yoked_swap"]) != (index // 2) % 2):
                    raise ValueError("observer family differs from its assigned keys, rule or controls")
                for name, flat in (("teacher_genome", teacher), ("starting_genome", starting), ("evolved_genome", evolved)):
                    if not np.array_equal(row[name], flat):
                        raise ValueError("evaluation genotype differs from its fixed assigned source")
                artifact = dict(index=index, **_save_arrays(folder, f"families/family_{index:04d}.npz", row))
                family_files.append(artifact)
                families.append(row)
                stream.write(json.dumps(dict(family=index, rule=index % 2, yoked_swap=(index // 2) % 2,
                    keys=keys.tolist(), raw_file=artifact, elapsed_seconds=time.monotonic() - started),
                    allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("evaluating"))
                print(f"observer evaluation {index + 1}/{args.eval_worlds} saved", flush=True)
                deadline()
        measurements = dict(observer.summarize(families))
        gate = measurements.pop("gate")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     scientific_result=args.condition_split == "final", gate_status="not_applicable",
                     changed_parameters=int(np.count_nonzero(evolved != starting)))
        if args.condition_split == "final":
            if set(gate) != {"passed", "first_food_passed", "survival_passed"} or any(type(value) is not bool for value in gate.values()):
                raise ValueError("completed observer evolution requires a defined boolean gate")
            final.update(gate=gate, gate_status="passed" if gate["passed"] else "failed")
        deadline()
        _write_json(folder / "summary.json", final)
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        partial = summary(value)
        partial["error"] = f"{type(error).__name__}: {error}"
        if not (folder / "summary.json").exists():
            _write_json(folder / "summary.json", partial)
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder
