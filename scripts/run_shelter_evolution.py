"""Tracked E118 shelter evolution with immutable generations and paired evaluation."""
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
import numpy as np

from seed import seed
from scripts import shelter_evolution
from scripts.run_cold_donor import SOURCES, _write_json, _write_status


PROTOCOL = "shelter-evolution-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds",
                 "--hypothesis"}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("shelter-evolution-v1 accepts only its registered run and evolution flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or args.checkpoint_every != 10
            or (args.pop, args.worlds, args.sigma, args.lr) != (16, 4, .05, .02)):
        raise ValueError("shelter-evolution-v1 requires its fixed ecology, persistence and evolution settings")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("shelter evolution requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected, evaluation_seed = SOURCES[0], ("E118-benchmark", 43, 1, 4), 44
    else:
        selected = [item for item in SOURCES if args.name == "E118-" + item["label"]]
        if len(selected) != 1:
            raise ValueError("final shelter evolution must be named E118-A0, E118-A1 or E118-A2")
        source = selected[0]
        index = source["training_seed"]
        expected = ("E118-" + source["label"], 2026112201 + index, 10, 128)
        evaluation_seed = 2026112301 + index
    if (args.name, args.seed, args.generations, args.eval_worlds) != expected:
        raise ValueError("shelter evolution identity, seeds and budgets must match the registered split")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this shelter evolution attempt exists; preserve it and register any recovery")
    source_path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != source_path:
        raise ValueError("shelter evolution requires the matching original E103 genome")
    content = source_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("original genome file differs from its registered SHA256")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("the original genome must contain 155 finite float32 values")
    return dict(source), content, flat, evaluation_seed


def _save_arrays(folder, relative, arrays):
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("saved evidence must contain finite numeric arrays")
    path = folder / relative
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _save_genome(path, genome):
    with path.open("xb") as stream:
        np.save(stream, np.asarray(genome, dtype=np.float32), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _state_arrays(state, generation):
    arrays = {name: np.asarray(value) for name, value in state.items()}
    if (set(arrays) != {"mu", "m", "v", "t", "key"}
            or any(arrays[name].shape != (155,) or arrays[name].dtype != np.float32
                   or not np.isfinite(arrays[name]).all() for name in ("mu", "m", "v"))
            or arrays["t"].shape != () or arrays["t"].dtype.kind not in "iu"
            or int(arrays["t"]) != generation or arrays["key"].shape != (2,)
            or arrays["key"].dtype != np.uint32):
        raise ValueError("evolution state has invalid arrays or generation")
    return arrays


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root)
    source, content, parent, evaluation_seed = _validate(args, git, root, explicit)
    c = seed.Cfg()
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "generations").mkdir()
    (folder / "families").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
                  condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
                  ecology=dict(version=shelter_evolution.ECOLOGY.version, **asdict(shelter_evolution.ECOLOGY)),
                  persistence=dict(version=shelter_evolution.PERSISTENCE.version, **asdict(shelter_evolution.PERSISTENCE)),
                  plastic=True, controller="rules-v0", genome_size=155,
                  seed=args.seed, evaluation_seed=evaluation_seed, eval_worlds=args.eval_worlds,
                  expected_families=args.eval_worlds, expected_generations=args.generations,
                  objective="301 minus first living shelter entry, averaged over outside births per world",
                  selection="fixed final mean after registered generations", l2=.005,
                  rule_assignment="world index modulo 2", inheritance="fresh lifetime weights at each birth",
                  max_seconds=MAX_SECONDS[args.condition_split], scientific_result=args.condition_split == "final",
                  runtime=dict(python=platform.python_version(), platform=platform.system(),
                               numpy=np.__version__, jax=importlib.metadata.version("jax"),
                               jaxlib=importlib.metadata.version("jaxlib")))
    _write_json(folder / "config.json", config)
    with (folder / "genome_start.npy").open("xb") as stream:
        stream.write(content)
    generation, families = 0, []
    generation_files, family_files = [], []
    trajectory = []
    sampling = {}

    def status(value):
        return dict(status=value, generation=generation, expected_generations=args.generations,
                    completed_families=len(family_files), expected_families=args.eval_worlds,
                    elapsed_seconds=time.monotonic() - started)

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
                    source_label=source["label"], training_seed=args.seed, evaluation_seed=evaluation_seed,
                    scientific_result=False, artifact_validity="incomplete", gate_status="undecided",
                    generations=list(generation_files), raw_files=list(family_files))

    def check_deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("shelter evolution exceeded its registered process allowance")

    _write_status(folder, status("training"))
    try:
        evaluate, _ = shelter_evolution.make_evaluator(c, args.worlds)

        def recorded_evaluate(candidates, world_keys, development_keys):
            result = evaluate(candidates, world_keys, development_keys)
            sampling.update(candidate_genomes=np.asarray(candidates), world_keys=np.asarray(world_keys),
                            development_keys=np.asarray(development_keys), fitness=np.asarray(result["fitness"]))
            return result

        state = seed.es_init(c, mean=parent, seed=args.seed)
        arrays = _state_arrays(state, 0)
        if not np.array_equal(arrays["mu"], parent):
            raise ValueError("evolution did not start from the registered original genome")
        generation_files.append(_save_arrays(folder, "generations/generation_0000.npz", arrays))
        trajectory.append(arrays["mu"].copy())
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for next_generation in range(1, args.generations + 1):
                check_deadline()
                previous = np.asarray(state["mu"]).copy()
                _, noise_key, _, _ = jax.random.split(state["key"], 4)
                half_noise = np.asarray(jax.random.normal(noise_key, (args.pop // 2, parent.size)))
                sampling.clear()
                sampling["eps"] = np.concatenate((half_noise, -half_noise))
                state, row = seed.es_step(state, recorded_evaluate, args.pop, args.worlds, args.sigma, args.lr)
                arrays = _state_arrays(state, next_generation)
                samples = _save_arrays(folder, f"generations/samples_{next_generation:04d}.npz", sampling)
                artifact = _save_arrays(folder, f"generations/generation_{next_generation:04d}.npz", arrays)
                generation_files.append(artifact)
                trajectory.append(arrays["mu"].copy())
                generation = next_generation
                row = dict(row, generation=generation, checkpoint=artifact, samples=samples,
                           changed_parameters=int(np.count_nonzero(arrays["mu"] != previous)),
                           l2_distance_from_parent=float(np.linalg.norm(arrays["mu"] - parent)),
                           elapsed_seconds=time.monotonic() - started)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("training"))
                print(f"shelter evolution generation {generation}/{args.generations} saved", flush=True)
                check_deadline()
        descendant = np.asarray(state["mu"]).copy()
        _save_genome(folder / "genome_final.npy", descendant)
        _save_genome(folder / "genome_trajectory.npy", np.stack(trajectory))
        _write_status(folder, status("evaluating"))
        seen_keys = set()
        with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                check_deadline()
                row = shelter_evolution.evaluate_pair(parent, descendant, evaluation_seed, index, c)
                shelter_evolution.validate_pair(row, c)
                keys = np.asarray(row["keys"])
                expected = np.asarray(shelter_evolution.evaluation_keys(evaluation_seed, index))
                if (keys.shape != (2, 2) or keys.dtype != np.uint32 or not np.array_equal(keys, expected)
                        or int(row["rule"]) != index % 2):
                    raise ValueError("evaluation family differs from its registered pairing")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != 2 or seen_keys.intersection(tokens):
                    raise ValueError("evaluation families must have distinct world and development keys")
                seen_keys.update(tokens)
                artifact = _save_arrays(folder, f"families/family_{index:04d}.npz", row)
                family_files.append(artifact)
                families.append(row)
                stream.write(json.dumps(dict(family=index, rule=index % 2, keys=keys.tolist(), raw_file=artifact,
                                             elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("evaluating"))
                print(f"shelter evolution evaluation {index + 1}/{args.eval_worlds} saved", flush=True)
        measurements = dict(shelter_evolution.summarize(families))
        gate = measurements.pop("gate")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     scientific_result=args.condition_split == "final", gate_status="not_applicable",
                     changed_parameters=int(np.count_nonzero(descendant != parent)),
                     l2_distance_from_parent=float(np.linalg.norm(descendant - parent)))
        if args.condition_split == "final":
            if set(gate) != {"passed", "latency_passed", "survival_passed"} or any(type(value) is not bool for value in gate.values()):
                raise ValueError("completed final shelter evolution requires a defined boolean gate")
            final.update(gate=gate, gate_status="passed" if gate["passed"] else "failed")
        check_deadline()
        _write_json(folder / "summary.json", final)
    except BaseException as error:
        state_name = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        partial = summary(state_name)
        partial["error"] = f"{type(error).__name__}: {error}"
        if not (folder / "summary.json").exists():
            _write_json(folder / "summary.json", partial)
        _write_status(folder, status(state_name))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder
