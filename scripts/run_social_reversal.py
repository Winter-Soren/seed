"""E125 saved-controller transfer to hidden randomized rule reversal."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed import direct
from seed.seed import Cfg
from scripts import social_reversal, run_social_transmission as calibration
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "reversal-random-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
ARMS = ("observed", "belief_reset", "birth_parent", "direct_baseline")
CONDITIONS = ("stationary", "reversal")
PRIMARY = "observed"
FLIP_STEPS = (100, 125, 150, 175)
KEY_ORDER = ("teacher_world", "teacher_birth", "solo_world", "pupil_birth")
GATE_FIELDS = {"passed", "late_level_passed", "late_chance_passed", "recovery_passed",
               "accuracy_noninferiority_passed", "survival_noninferiority_passed"}
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}
DIRECT_SOURCES = (
    dict(label="direct0", training_seed=0, path="runs/20261119-033803_E113-direct-s0/genome_final.npy",
         sha256="0a47a3dbbb7482a42ef069cf6b54b44e511ffaf2d169d92914c10fdf75a4bb61"),
    dict(label="direct1", training_seed=1, path="runs/20261119-040602_E113-direct-s1/genome_final.npy",
         sha256="0e77c3083a88c8216e57bff0667b5f594542447c14f6ae7338ec0e090b9d7d3b"),
    dict(label="direct2", training_seed=2, path="runs/20261119-041609_E113-direct-s2/genome_final.npy",
         sha256="473124dfe1c2a0d77a9a4b7cdde5cb36c41f61b0664e17ede86ae88502971bbd"),
)


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("reversal-random-v1 accepts only its registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("random reversal fixes saved controllers, calibration and zero training generations")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("random reversal requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected = SOURCES[0], ("E125-benchmark", 2026120300, 4)
    else:
        matches = [item for item in SOURCES if args.name == "E125-" + item["label"]]
        if len(matches) != 1:
            raise ValueError("final reversal must be named E125-A0, E125-A1 or E125-A2")
        source = matches[0]
        expected = ("E125-" + source["label"], 2026120301 + source["training_seed"], 128)
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("reversal identity, seed and family count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this reversal attempt exists; preserve it and preregister any recovery")
    path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != path:
        raise ValueError("reversal requires the matching original E103 genome")
    content = calibration._read_pinned(path, source["sha256"])
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    baseline = DIRECT_SOURCES[source["training_seed"]]
    direct_bytes = calibration._read_pinned(root / baseline["path"], baseline["sha256"])
    direct_flat = np.load(io.BytesIO(direct_bytes), allow_pickle=False)
    if (flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all()
            or direct_flat.shape != (direct.genome_size(Cfg(M=1)),)
            or direct_flat.dtype != np.float32 or not np.isfinite(direct_flat).all()):
        raise ValueError("saved original and direct genomes must retain their registered finite float32 formats")
    if tuple(social_reversal.ARMS) != ARMS or tuple(social_reversal.CONDITIONS) != CONDITIONS:
        raise ValueError("reversal backend panels differ from registration")
    return dict(source), content, flat, dict(baseline), direct_bytes, direct_flat


def _assignments(master, count):
    manifest = {name: np.asarray(value) for name, value in social_reversal.assignments_manifest(master, count).items()}
    if (set(manifest) != {"key", "permutation", "table"}
            or manifest["key"].shape != (2,) or manifest["key"].dtype != np.uint32
            or manifest["permutation"].shape != (count,) or manifest["permutation"].dtype != np.int32
            or manifest["table"].shape != (count, 2) or manifest["table"].dtype != np.int32
            or not np.array_equal(np.sort(manifest["permutation"]), np.arange(count))
            or not np.array_equal(manifest["key"], social_reversal.assignment_key(master))):
        raise ValueError("randomized schedule must retain its dedicated key, permutation and full assignment table")
    base = np.asarray([(0, 100), (1, 125), (0, 150), (1, 175)] if count == 4 else
                      [(rule, step) for rule in (0, 1) for step in FLIP_STEPS for _ in range(16)], np.int32)
    if count not in (4, 128) or not np.array_equal(manifest["table"], base[manifest["permutation"]]):
        raise ValueError("schedule must preserve the preregistered balanced rule-by-time cells")
    return manifest


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root).resolve()
    source, content, flat, baseline, direct_bytes, direct_flat = _validate(args, git, root, explicit)
    calibration_source, selection_bytes, selection, selected_lambda, used_keys = calibration._load_calibration(
        root, source, args.condition_split)
    c = Cfg(M=1)
    direct_weights = np.asarray(direct.decode(direct_flat, c))
    schedule = _assignments(args.seed, args.eval_worlds)
    if schedule["key"].tobytes() in used_keys:
        raise ValueError("schedule and calibration require independent random keys")
    used_keys.add(schedule["key"].tobytes())
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "calibration").mkdir()
    (folder / "families").mkdir()
    assignments = _save_arrays(folder, "assignments.npz", schedule)
    frozen = dict(path="calibration/selection.json", sha256=calibration_source["selection"]["sha256"])
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source, direct_source=baseline,
        calibration_source=calibration_source, frozen_calibration=frozen, selected_lambda=selected_lambda,
        assignments=assignments, assignment_columns=["rule", "flip_step"], assignment_stream_tag=0xE125,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={}, ecology=None, persistence="off",
        generation=0, seed=args.seed, eval_seed=args.seed, eval_worlds=args.eval_worlds,
        expected_families=args.eval_worlds, arms=list(ARMS), conditions=list(CONDITIONS), primary=PRIMARY,
        key_order=list(KEY_ORDER), flip_steps=list(FLIP_STEPS), practice_steps=150, watch_steps=150, solo_steps=300,
        windows=dict(pre=[-50, 0], early=[0, 50], late=[50, 100]), max_seconds=MAX_SECONDS[args.condition_split],
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        mechanism="unchanged original genome and E123 calibrated fixed belief readout; hidden world reversal only",
        primary_preparation="150 passive frames from a fresh original teacher; q remains fixed throughout solo life",
        fair_birth_panel=["birth_parent", "direct_baseline"], selected_neutral_birth_reuses="birth_parent",
        comparison_limit="prepared versus unprepared direct controller is descriptive unequal information",
        formal_ladder_status="not_decided", scientific_result=args.condition_split == "final")
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    _write_bytes(folder / "direct_genome.npy", direct_bytes)
    _write_bytes(folder / frozen["path"], selection_bytes)
    rows, artifacts, family_seconds = [], [], []
    setup_seconds = time.monotonic() - started

    def status(value):
        return dict(status=value, completed_families=len(artifacts), expected_families=args.eval_worlds,
                    elapsed_seconds=time.monotonic() - started, setup_seconds=setup_seconds,
                    family_seconds=list(family_seconds))

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
            generation=0, source_label=source["label"], eval_seed=args.seed, eval_worlds=args.eval_worlds,
            scientific_result=False, artifact_validity="incomplete", gate_status="undecided", primary=PRIMARY,
            formal_ladder_status="not_decided", selected_lambda=selected_lambda, calibration_source=calibration_source,
            frozen_calibration=frozen, assignments=assignments, raw_files=list(artifacts))

    def check_deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("random reversal exceeded its registered invocation allowance")

    def check_frozen():
        calibration._read_pinned(folder / frozen["path"], frozen["sha256"])
        calibration._read_pinned(root / calibration_source["selection"]["path"], frozen["sha256"])
        calibration._read_pinned(folder / assignments["path"], assignments["sha256"])

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index, (rule, flip_step) in enumerate(schedule["table"]):
                family_started = time.monotonic()
                check_deadline()
                check_frozen()
                keys = np.asarray(social_reversal.family_keys(args.seed, index))
                if keys.shape != (4, 2) or keys.dtype != np.uint32:
                    raise ValueError("reversal families require four uint32 random keys")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != 4 or used_keys.intersection(tokens):
                    raise ValueError("reversal worlds, births, schedule and calibration must have disjoint keys")
                used_keys.update(tokens)
                row = social_reversal.evaluate_family(flat, direct_weights, selected_lambda, keys, index,
                                                      int(rule), int(flip_step), c)
                if (not np.array_equal(row["keys"], keys) or int(row["family_index"]) != index
                        or int(row["rule"]) != int(rule) or int(row["flip_step"]) != int(flip_step)
                        or float(row["lambda_value"]) != selected_lambda
                        or not np.array_equal(row["original_genome"], flat)
                        or not np.array_equal(row["direct_weights"], direct_weights)):
                    raise ValueError("reversal family changed its assigned world, boundary, calibration or sources")
                row["direct_genome"] = direct_flat.copy()
                row["calibration_sha256"] = np.frombuffer(bytes.fromhex(frozen["sha256"]), dtype=np.uint8).copy()
                row["assignments_sha256"] = np.frombuffer(bytes.fromhex(assignments["sha256"]), dtype=np.uint8).copy()
                projected = social_reversal.summary_input(row)
                artifact = dict(index=index, **_save_arrays(folder, f"families/family_{index:04d}.npz", row))
                artifacts.append(artifact)
                rows.append(projected)
                family_seconds.append(time.monotonic() - family_started)
                stream.write(json.dumps(dict(family=index, rule=int(rule), flip_step=int(flip_step),
                    keys=keys.tolist(), raw_file=artifact, calibration_sha256=frozen["sha256"],
                    assignments_sha256=assignments["sha256"], family_seconds=family_seconds[-1],
                    elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                print(f"random reversal family {index + 1}/{args.eval_worlds} saved", flush=True)
                del row
                check_deadline()
        measurements = dict(social_reversal.summarize(rows))
        gate = measurements.pop("gate")
        if (measurements.get("families") != args.eval_worlds or measurements.get("primary") != PRIMARY
                or set(measurements.get("arms", {})) != set(ARMS) or set(gate) != GATE_FIELDS
                or any(type(value) is not bool for value in gate.values())
                or gate["passed"] != all(value for name, value in gate.items() if name != "passed")):
            raise ValueError("random reversal requires all panels and its complete intersection gate")
        for arm in measurements["arms"].values():
            if (set(arm.get("conditions", {})) != set(CONDITIONS)
                    or any(set(condition.get("windows", {})) != {"pre", "early", "late"}
                           for condition in arm["conditions"].values())):
                raise ValueError("every reversal controller must retain both worlds and all three windows")
        check_frozen()
        for artifact in [calibration_source["config"], *calibration_source["raw_files"]]:
            calibration._read_pinned(root / artifact["path"], artifact["sha256"])
        for artifact in (source, baseline):
            calibration._read_pinned(root / artifact["path"], artifact["sha256"])
        if ((folder / "genome_start.npy").read_bytes() != content
                or (folder / "direct_genome.npy").read_bytes() != direct_bytes
                or not np.array_equal(flat, np.load(io.BytesIO(content), allow_pickle=False))
                or not np.array_equal(direct_flat, np.load(io.BytesIO(direct_bytes), allow_pickle=False))
                or not np.array_equal(direct_weights, np.asarray(direct.decode(direct_flat, c)))):
            raise ValueError("reversal changed a pinned genome or decoded baseline")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
            scientific_result=args.condition_split == "final", gate_status="not_applicable",
            safeguards=dict(world_keys_disjoint=True, calibration_source_unchanged=True,
                            sources_unchanged=True, balanced_schedule_preserved=True, paired_traces_validated=True))
        if args.condition_split == "final":
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
