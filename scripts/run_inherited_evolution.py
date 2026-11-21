"""E119 continuation of saved ES states with a training-only inherited archive."""
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
import numpy as np

from seed import seed
from scripts import inherited_evolution
from scripts.run_cold_donor import _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays, _save_genome, _state_arrays


PROTOCOL = "inherited-evolution-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
SOURCE_COMMIT = "cd08b59c02994c048533a539ae65d580369bb29c"
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--pop", "--worlds", "--sigma", "--lr", "--eval-worlds",
                 "--hypothesis"}
SOURCES = (
    dict(label="A0", run="runs/20261121-182237_E118-A0", initial_generation=10,
         sha256="c56d774ddb2130bfb9fc07a5f143ab29ae701ffbecf4d7130128881a5470f3e4",
         checkpoint_sha256="2e1bea556060085e34c2dc7cf027f639d3f36b4ee592c64b1135e8c09907a4d2",
         config_sha256="7d7645e791440e6603d9f35cf043abf17701e597ad6f2592b7b76430de6b597f",
         status_sha256="110910637b4b0476bc8ef6bf65153e515f033dd1e602dff911ffb1b025a86fe1"),
    dict(label="A1", run="runs/20261121-182511_E118-A1", initial_generation=10,
         sha256="4d2f766a5435b809165629b47fd972b3a1160a9ee2fa97c13b214a6f0712c807",
         checkpoint_sha256="1984cacb45e87a3fdc29bb03bb83a424873f333619008086ca98b0c768a2e78e",
         config_sha256="038e3ca3308f213476a1c09bf565978f76e72e9c170d29b9bc507eea5e2e9bcb",
         status_sha256="d5f57b44e785789098004a43ab6b6c1ce7e92bb32bc199e21a1fcd046bf1b254"),
    dict(label="A2", run="runs/20261121-182806_E118-A2", initial_generation=10,
         sha256="88592fec7002c64bb51df24b31ba2cc99a3e611397eef3ebfd78e807d5ad203e",
         checkpoint_sha256="d0c8ce66eba91dba04151b70347ba7eae084ee9e35616ed2dd43d0a81883a740",
         config_sha256="8b4f2496ec8d8f1a240fc6a76572d42655be4496646e19b30cb4329940d69ce1",
         status_sha256="8b267b339284fbd33cb52eeb36921bee7a7b1a54b7c6d0c376ee7091f67d3f06"),
)
DEVELOPMENT_SOURCE = dict(
    label="A0", run="runs/20261121-182049_E118-benchmark", initial_generation=1,
    sha256="0406de7317b3bf89a580f6803fa207eede85f091176fd423e443961e373867ad",
    checkpoint_sha256="9c8fa76b3b2b3da22707199cf00988621d873f5901dcd45c7115354b87286cde",
    config_sha256="fe073e1d9c840cc5e460f657338b705ae584182177a04184bceada035e02f998",
    status_sha256="e6432ea822ea4d59682f729e741c73e1100ccc8aa1bca281e52298ee46728954")


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("inherited-evolution-v1 accepts only the registered run and evolution flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or args.checkpoint_every != 10
            or (args.pop, args.worlds, args.sigma, args.lr) != (16, 4, .05, .02)):
        raise ValueError("inherited evolution requires its fixed ecology, persistence and continuation settings")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("inherited evolution requires clean committed provenance")
    if args.condition_split == "development":
        source, expected, evaluation_seed = DEVELOPMENT_SOURCE, ("E119-benchmark", 45, 1, 4), 46
    else:
        selected = [(index, item) for index, item in enumerate(SOURCES) if args.name == "E119-" + item["label"]]
        if len(selected) != 1:
            raise ValueError("final inherited evolution must be named E119-A0, E119-A1 or E119-A2")
        index, source = selected[0]
        expected = ("E119-" + source["label"], 2026112401 + index, 10, 128)
        evaluation_seed = 2026112501 + index
    if (args.name, args.seed, args.generations, args.eval_worlds) != expected:
        raise ValueError("inherited evolution identity, seeds and budgets differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this inherited evolution attempt exists; preserve it and register any recovery")
    source = dict(source, path=source["run"] + "/genome_final.npy",
                  checkpoint=source["run"] + f"/generations/generation_{source['initial_generation']:04d}.npz",
                  commit=SOURCE_COMMIT)
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("inherited evolution requires the matching E118 final genome")
    payloads = {"genome": (root / source["path"]).read_bytes(),
                "checkpoint": (root / source["checkpoint"]).read_bytes(),
                "config": (root / source["run"] / "config.json").read_bytes(),
                "status": (root / source["run"] / "status.json").read_bytes()}
    for name, content in payloads.items():
        expected_hash = source["sha256" if name == "genome" else name + "_sha256"]
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"E118 source {name} differs from its registered SHA256")
    config, status = json.loads(payloads["config"]), json.loads(payloads["status"])
    if (config["protocol"] != "shelter-evolution-v1" or config["git"]["commit"] != SOURCE_COMMIT
            or config["git"]["dirty"] is not False or status["status"] != "finished"
            or status["generation"] != source["initial_generation"]
            or config["cfg"] != seed.Cfg()._asdict()):
        raise ValueError("E118 source must be the completed registered configuration")
    founder = np.load(io.BytesIO(payloads["genome"]), allow_pickle=False)
    with np.load(io.BytesIO(payloads["checkpoint"]), allow_pickle=False) as archive:
        state = _state_arrays(dict(archive), source["initial_generation"])
    if (founder.shape != (155,) or founder.dtype != np.float32 or not np.isfinite(founder).all()
            or not np.array_equal(state["mu"], founder) or np.any(state["v"] < 0)):
        raise ValueError("E118 terminal genome and optimizer mean must agree exactly")
    return source, payloads, state, evaluation_seed


def _register_keys(seen, keys, label):
    keys = np.asarray(keys)
    if keys.dtype != np.uint32 or keys.shape[-1:] != (2,):
        raise ValueError(f"invalid {label} random keys")
    tokens = [key.tobytes() for key in keys.reshape(-1, 2)]
    if len(set(tokens)) != len(tokens) or seen.intersection(tokens):
        raise ValueError(f"reused {label} random key")
    seen.update(tokens)


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root)
    source, payloads, state, evaluation_seed = _validate(args, git, root, explicit)
    founder = state["mu"].copy()
    state = {name: jnp.asarray(value) for name, value in state.items()}
    state["t"] = int(state["t"])
    c = seed.Cfg()
    archive = inherited_evolution.empty_archive(c)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("generations", "donors", "families"):
        (folder / name).mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
                  condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
                  ecology=dict(version=inherited_evolution.ECOLOGY.version, **asdict(inherited_evolution.ECOLOGY)),
                  persistence=dict(version=inherited_evolution.PERSISTENCE.version, **asdict(inherited_evolution.PERSISTENCE)),
                  plastic=True, controller="rules-v0", genome_size=155, seed=args.seed,
                  donor_seed=args.seed, evaluation_seed=evaluation_seed, eval_worlds=args.eval_worlds,
                  expected_families=args.eval_worlds, expected_generations=args.generations,
                  initial_generation=source["initial_generation"], l2=.005,
                  objective="301 minus first living shelter entry, averaged over outside births per world",
                  selection="fixed final mean after registered additional generations",
                  inheritance="cumulative masked sensor-to-output donor deltas before every ES update",
                  rule_assignment="world index modulo 2", evaluation_arms=list(inherited_evolution.ARMS),
                  max_seconds=MAX_SECONDS[args.condition_split], scientific_result=args.condition_split == "final",
                  runtime=dict(python=platform.python_version(), platform=platform.system(),
                               numpy=np.__version__, jax=jax.__version__,
                               jaxlib=importlib.metadata.version("jaxlib")))
    _write_json(folder / "config.json", config)
    for filename, content in (("genome_start.npy", payloads["genome"]),
                              ("checkpoint_start.npz", payloads["checkpoint"])):
        with (folder / filename).open("xb") as stream:
            stream.write(content)
    generation, families = 0, []
    generation_files, archive_files, donor_files, family_files = [], [], [], []
    trajectory, archive_trajectory, sampling, seen_keys = [], [], {}, set()

    def status(value):
        return dict(status=value, generation=generation, optimizer_generation=source["initial_generation"] + generation,
                    expected_generations=args.generations, completed_families=len(family_files),
                    expected_families=args.eval_worlds, elapsed_seconds=time.monotonic() - started)

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
                    source_label=source["label"], donor_seed=args.seed, evaluation_seed=evaluation_seed,
                    scientific_result=False, artifact_validity="incomplete", gate_status="undecided",
                    generations=list(generation_files), archives=list(archive_files),
                    donors=list(donor_files), raw_files=list(family_files))

    def check_deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("inherited evolution exceeded its registered process allowance")

    _write_status(folder, status("training"))
    try:
        evaluate, _ = inherited_evolution.make_evaluator(c, args.worlds)

        def recorded_evaluate(candidates, world_keys, development_keys):
            if not np.array_equal(np.asarray(development_keys)[:args.pop // 2],
                                  np.asarray(development_keys)[args.pop // 2:]):
                raise ValueError("antithetic candidate development keys differ")
            _register_keys(seen_keys, world_keys, "training world")
            _register_keys(seen_keys, development_keys[:args.pop // 2], "training development")
            result = evaluate(candidates, world_keys, development_keys, archive)
            sampling.update(candidate_genomes=np.asarray(candidates), world_keys=np.asarray(world_keys),
                            development_keys=np.asarray(development_keys), archive=np.asarray(archive).copy(),
                            fitness=np.asarray(result["fitness"]),
                            **{name: np.asarray(result[name]) for name in
                               ("candidate_birth", "candidate_initial", "candidate_mask")})
            return result

        arrays = _state_arrays(state, source["initial_generation"])
        generation_files.append(_save_arrays(folder, "generations/generation_0000.npz", arrays))
        archive_files.append(_save_arrays(folder, "generations/archive_0000.npz", dict(archive=archive)))
        trajectory.append(arrays["mu"].copy())
        archive_trajectory.append(np.asarray(archive).copy())
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for next_generation in range(1, args.generations + 1):
                check_deadline()
                previous = np.asarray(state["mu"]).copy()
                previous_archive = np.asarray(archive).copy()
                donor = inherited_evolution.donor_update(previous, archive, args.seed, next_generation - 1, c, worlds=args.worlds)
                inherited_evolution.validate_donors(donor, c)
                _register_keys(seen_keys, donor["keys"], "donor")
                expected_keys = inherited_evolution.donor_keys(args.seed, next_generation - 1, args.worlds)
                if (not np.array_equal(donor["keys"], expected_keys)
                        or not np.array_equal(donor["archive_before"], archive)):
                    raise ValueError("donor update differs from its assigned archive or keys")
                artifact = _save_arrays(folder, f"donors/donor_{next_generation:04d}.npz", donor)
                donor_files.append(artifact)
                archive = np.asarray(donor["archive_after"]).copy()
                check_deadline()
                _, noise_key, _, _ = jax.random.split(state["key"], 4)
                half_noise = np.asarray(jax.random.normal(noise_key, (args.pop // 2, founder.size)))
                sampling.clear()
                sampling["eps"] = np.concatenate((half_noise, -half_noise))
                state, row = seed.es_step(state, recorded_evaluate, args.pop, args.worlds, args.sigma, args.lr)
                arrays = _state_arrays(state, source["initial_generation"] + next_generation)
                samples = _save_arrays(folder, f"generations/samples_{next_generation:04d}.npz", sampling)
                checkpoint = _save_arrays(folder, f"generations/generation_{next_generation:04d}.npz", arrays)
                memory = _save_arrays(folder, f"generations/archive_{next_generation:04d}.npz", dict(archive=archive))
                generation_files.append(checkpoint)
                archive_files.append(memory)
                trajectory.append(arrays["mu"].copy())
                archive_trajectory.append(archive.copy())
                generation = next_generation
                row = dict(row, generation=generation, checkpoint=checkpoint, samples=samples,
                           donor=artifact, archive=memory,
                           changed_parameters=int(np.count_nonzero(arrays["mu"] != previous)),
                           l2_distance_from_founder=float(np.linalg.norm(arrays["mu"] - founder)),
                           changed_archive_entries=int(np.count_nonzero(archive != previous_archive)),
                           elapsed_seconds=time.monotonic() - started)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("training"))
                print(f"inherited evolution generation {generation}/{args.generations} saved", flush=True)
                check_deadline()
        descendant = np.asarray(state["mu"]).copy()
        _save_genome(folder / "genome_final.npy", descendant)
        _save_genome(folder / "genome_trajectory.npy", np.stack(trajectory))
        _save_genome(folder / "archive_trajectory.npy", np.stack(archive_trajectory))
        arms = dict(founder_empty=(founder, archive_trajectory[0]),
                    parent_previous=(previous, previous_archive), parent_latest=(previous, archive),
                    child_previous=(descendant, previous_archive), child_latest=(descendant, archive),
                    child_empty=(descendant, archive_trajectory[0]))
        _write_status(folder, status("evaluating"))
        with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                check_deadline()
                row = inherited_evolution.evaluate_world(arms, evaluation_seed, index, c)
                inherited_evolution.validate_world(row, c)
                keys = np.asarray(row["keys"])
                if (not np.array_equal(keys, inherited_evolution.evaluation_keys(evaluation_seed, index))
                        or int(row["rule"]) != index % 2):
                    raise ValueError("evaluation world differs from registered assignment")
                _register_keys(seen_keys, keys, "evaluation")
                artifact = _save_arrays(folder, f"families/family_{index:04d}.npz", row)
                family_files.append(artifact)
                families.append(row)
                stream.write(json.dumps(dict(family=index, rule=index % 2, keys=keys.tolist(), raw_file=artifact,
                                             elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("evaluating"))
                print(f"inherited evolution evaluation {index + 1}/{args.eval_worlds} saved", flush=True)
        measurements = dict(inherited_evolution.summarize(families))
        gate = measurements.pop("gate")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     scientific_result=args.condition_split == "final", gate_status="not_applicable",
                     changed_parameters=int(np.count_nonzero(descendant != founder)),
                     l2_distance_from_founder=float(np.linalg.norm(descendant - founder)))
        if args.condition_split == "final":
            if set(gate) != {"passed", "latency_passed", "survival_passed"} or any(type(value) is not bool for value in gate.values()):
                raise ValueError("completed inherited evolution requires a defined boolean gate")
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
