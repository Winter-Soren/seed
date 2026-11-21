"""E124 immutable public-movie transmission with frozen E123 calibration."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed.seed import Cfg
from scripts import social_transmission
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays
from scripts.run_social_belief import _validate_selection


PROTOCOL = "social-transmission-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
ARMS = ("predecessor", "repeated_founder", "yoked", "belief_reset")
CONTROLS = ARMS[1:]
STAGES = ("founder", "c1", "c2", "c3")
KEY_ORDER = ("solo_world", "pupil_birth")
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}
# E123 training calibration only; each selection digest pins its complete raw calibration manifest.
CALIBRATION_SOURCES = {
    "E123-benchmark": dict(
        folder="runs/20261121-223900_E123-benchmark",
        commit="b7a6dfc91b08b8cc44dd66510752e8f6b8b5e758",
        config_sha256="c4c0093600edbdb35c8b330d72a2b61d938b4fb756a2947e78316c116547a15c",
        selection_sha256="a79e663b8889067defe53c16c2e649ae398027426f61ebf3b2cfe0cb35200bdc"),
    "E123-A0": dict(
        folder="runs/20261121-224042_E123-A0",
        commit="b7a6dfc91b08b8cc44dd66510752e8f6b8b5e758",
        config_sha256="029a7f5ce0354da59d747358ba8000128673f25bad7dd97b32a552dc73cecccb",
        selection_sha256="b1d49b1fcc42e6a2bba6ea190106cadc216319b1ebab29750c10d5f365742c02"),
    "E123-A1": dict(
        folder="runs/20261121-224439_E123-A1",
        commit="b7a6dfc91b08b8cc44dd66510752e8f6b8b5e758",
        config_sha256="8a03cf06c3615d0d8d08f57d2fcc4a9f4bfa1ad00ff9697f37b387e3f4f2d88b",
        selection_sha256="3fe92f620be735d9a3843faf3cd651e7dfa5498e588b89fb87dbe51ae39027d9"),
    "E123-A2": dict(
        folder="runs/20261121-224833_E123-A2",
        commit="b7a6dfc91b08b8cc44dd66510752e8f6b8b5e758",
        config_sha256="7a3aa382b8e7af57f4507e768905c6e8b49abc836c632097653d13c5279bef8a",
        selection_sha256="63feeacae1fd6f3766e049914e64a975827f76d175801050a465b27328e00425"),
}


def _read_pinned(path, digest):
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("source artifact requires a registered full SHA256")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"source artifact differs from registration: {path.name}")
    return content


def _load_calibration(root, source, split):
    name = "E123-benchmark" if split == "development" else "E123-" + source["label"]
    pin = CALIBRATION_SOURCES.get(name)
    if not pin or set(pin) != {"folder", "commit", "config_sha256", "selection_sha256"}:
        raise ValueError("E123 calibration source is not preregistered for this lineage and split")
    parent = (root / pin["folder"]).resolve()
    if parent.parent != (root / "runs").resolve() or not parent.name.endswith("_" + name):
        raise ValueError("calibration source must be the registered E123 run folder")
    config_bytes = _read_pinned(parent / "config.json", pin["config_sha256"])
    config = json.loads(config_bytes)
    expected = 4 if split == "development" else 32
    seed = 51 if split == "development" else 2026113001 + source["training_seed"]
    if (config.get("protocol") != "social-belief-readout-v1" or config.get("condition_split") != split
            or config.get("source") != source or config.get("calibration_seed") != seed
            or config.get("expected_calibration_families") != expected
            or config.get("calibration_worlds") != expected or config.get("cfg") != Cfg(M=1)._asdict()
            or config.get("generation") != 0 or config.get("args", {}).get("name") != name
            or config.get("git", {}).get("dirty") is not False
            or config.get("git", {}).get("commit") != pin["commit"]
            or not isinstance(pin["commit"], str) or len(pin["commit"]) != 40
            or any(ch not in "0123456789abcdef" for ch in pin["commit"])):
        raise ValueError("E123 calibration provenance differs from the registered original lineage")
    selection_bytes = _read_pinned(parent / "calibration/selection.json", pin["selection_sha256"])
    selection = json.loads(selection_bytes)
    if (selection.get("status") != "frozen" or selection.get("calibration_seed") != seed
            or selection.get("completed_families") != expected or selection.get("expected_families") != expected):
        raise ValueError("source calibration is incomplete or unfrozen")
    selected_lambda = _validate_selection(selection, expected)
    artifacts = selection.get("raw_files", [])
    if len(artifacts) != expected:
        raise ValueError("source calibration must retain all assigned raw records")
    used_keys, raw_files = set(), []
    for index, artifact in enumerate(artifacts):
        relative = f"calibration/family_{index:04d}.npz"
        if set(artifact) != {"index", "path", "sha256"} or artifact["index"] != index or artifact["path"] != relative:
            raise ValueError("calibration evidence must be complete, ordered and confined to calibration/")
        raw_path = parent / relative
        raw = _read_pinned(raw_path, artifact["sha256"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            keys = arrays["keys"]
            if keys.shape != (4, 2) or keys.dtype != np.uint32:
                raise ValueError("source calibration contains malformed assignment keys")
            tokens = [key.tobytes() for key in keys]
            if len(set(tokens)) != 4 or used_keys.intersection(tokens):
                raise ValueError("source calibration reused an assigned random key")
            used_keys.update(tokens)
        raw_files.append(dict(index=index, path=raw_path.relative_to(root).as_posix(), sha256=artifact["sha256"]))
    provenance = dict(name=name, folder=parent.relative_to(root).as_posix(), commit=pin["commit"],
        calibration_seed=seed, expected_calibration_families=expected,
        config=dict(path=(parent / "config.json").relative_to(root).as_posix(), sha256=pin["config_sha256"]),
        selection=dict(path=(parent / "calibration/selection.json").relative_to(root).as_posix(),
                       sha256=pin["selection_sha256"]), raw_files=raw_files)
    return provenance, selection_bytes, selection, selected_lambda, used_keys


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("social-transmission-v1 accepts only its registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("transmission fixes the original genome, calibrated scalar and zero ES generations")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("transmission requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected = SOURCES[0], ("E124-benchmark", 2026120200, 4)
    else:
        selected = [item for item in SOURCES if args.name == "E124-" + item["label"]]
        if len(selected) != 1:
            raise ValueError("final transmission must be named E124-A0, E124-A1 or E124-A2")
        source = selected[0]
        expected = ("E124-" + source["label"], 2026120201 + source["training_seed"], 128)
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("transmission name, seed and chain count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this transmission attempt exists; preserve it and preregister any recovery")
    path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != path:
        raise ValueError("transmission requires the matching original E103 genome")
    content = _read_pinned(path, source["sha256"])
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original genome must contain 155 finite float32 values")
    if tuple(social_transmission.ARMS) != ARMS:
        raise ValueError("transmission backend controls differ from registration")
    return dict(source), content, flat


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root).resolve()
    source, content, flat = _validate(args, git, root, explicit)
    calibration_source, selection_bytes, selection, selected_lambda, used_keys = _load_calibration(root, source, args.condition_split)
    c = Cfg(M=1)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "calibration").mkdir()
    (folder / "chains").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    frozen = dict(path="calibration/selection.json", sha256=calibration_source["selection"]["sha256"])
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
        calibration_source=calibration_source, frozen_calibration=frozen, selected_lambda=selected_lambda,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
        ecology=None, persistence="off", plastic=True, controller="rules-v0", genome_size=155,
        generation=0, seed=args.seed, eval_seed=args.seed, eval_worlds=args.eval_worlds,
        expected_chains=args.eval_worlds, cohorts=3, stages=list(STAGES), key_order=list(KEY_ORDER),
        arms=list(ARMS), controls=list(CONTROLS), founder_steps=c.T, watch_steps=c.T // 2, solo_steps=c.T,
        max_seconds=MAX_SECONDS[args.condition_split], raw_schema=PROTOCOL,
        raw_pattern="chains/chain_{index:04d}.npz", runtime=_runtime(),
        mechanism="fixed E103 genome plus engineered E123 calibrated public-contact belief and action readout",
        transmission="only last 150 public frames of predecessor arm; no inherited internal state or selection",
        rule_assignment="chain index modulo 2; same rule throughout a chain; independent births and layouts",
        yoked_assignment="item channels swapped when (chain index // 2) modulo 2 is one",
        statistical_unit="independent chain; never pooled cohorts or movie frames",
        retention="each cohort first-food gain >2 paired SE versus yoked and reset; survival lower2SE >-0.10 versus all controls and c3 versus c1",
        improvement="retention passes and c3 first-food gain >2 paired SE versus c1 and repeated founder; same survival bounds",
        scientific_result=args.condition_split == "final")
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    _write_bytes(folder / frozen["path"], selection_bytes)
    rows, artifacts, chain_seconds = [], [], []
    setup_seconds = time.monotonic() - started

    def status(value):
        return dict(status=value, phase="evaluation", completed_chains=len(artifacts),
                    expected_chains=args.eval_worlds, elapsed_seconds=time.monotonic() - started,
                    setup_seconds=setup_seconds, chain_seconds=list(chain_seconds))

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
            generation=0, source_label=source["label"], eval_seed=args.seed, eval_worlds=args.eval_worlds,
            scientific_result=False, artifact_validity="incomplete", retained_efficacy_status="undecided",
            cumulative_improvement_status="undecided", selected_lambda=selected_lambda,
            calibration_source=calibration_source, frozen_calibration=frozen, raw_files=list(artifacts))

    def check_deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("transmission exceeded its registered invocation allowance")

    def check_frozen():
        _read_pinned(folder / frozen["path"], frozen["sha256"])
        _read_pinned(root / calibration_source["selection"]["path"], frozen["sha256"])

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                chain_started = time.monotonic()
                check_deadline()
                check_frozen()
                keys = np.asarray(social_transmission.chain_keys(args.seed, index))
                if keys.shape != (4, 2, 2) or keys.dtype != np.uint32:
                    raise ValueError("transmission chains require eight uint32 random keys")
                tokens = [key.tobytes() for key in keys.reshape(8, 2)]
                if len(set(tokens)) != 8 or used_keys.intersection(tokens):
                    raise ValueError("transmission stages and calibration must have disjoint random keys")
                used_keys.update(tokens)
                row = social_transmission.evaluate_chain(flat, keys, index, selected_lambda, c)
                if (not np.array_equal(row["keys"], keys) or int(row["chain_index"]) != index
                        or int(row["rule"]) != index % 2 or int(row["yoked_swap"]) != (index // 2) % 2
                        or float(row["lambda_value"]) != selected_lambda
                        or not np.array_equal(row["original_genome"], flat)):
                    raise ValueError("transmission chain changed its assigned keys, rule, movie control or calibration")
                row["calibration_sha256"] = np.frombuffer(bytes.fromhex(frozen["sha256"]), dtype=np.uint8).copy()
                projected = social_transmission.summary_input(row)
                artifact = dict(index=index, **_save_arrays(folder, f"chains/chain_{index:04d}.npz", row))
                artifacts.append(artifact)
                rows.append(projected)
                chain_seconds.append(time.monotonic() - chain_started)
                stream.write(json.dumps(dict(chain=index, rule=index % 2, yoked_swap=(index // 2) % 2,
                    keys=keys.tolist(), raw_file=artifact, calibration_sha256=frozen["sha256"],
                    chain_seconds=chain_seconds[-1],
                    elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                print(f"transmission chain {index + 1}/{args.eval_worlds} saved", flush=True)
                del row
                check_deadline()
        measurements = dict(social_transmission.summarize(rows))
        retained = measurements.pop("retained_efficacy")
        improved = measurements.pop("cumulative_improvement")
        measurements.pop("gate", None)
        if set(measurements.get("cohorts", {})) != {"1", "2", "3"}:
            raise ValueError("transmission must report all three cohorts")
        for cohort in measurements["cohorts"].values():
            if set(cohort.get("arms", {})) != set(ARMS) or set(cohort.get("contrasts", {})) != set(CONTROLS):
                raise ValueError("transmission must report all four arms and three matched contrasts per cohort")
        fields = {"passed", "first_food_passed", "survival_passed"}
        if (any(set(gate) != fields or any(type(value) is not bool for value in gate.values())
                for gate in (retained, improved)) or improved["passed"] and not retained["passed"]):
            raise ValueError("retention and additional improvement require separate, ordered boolean decisions")
        if (retained["passed"] != (retained["first_food_passed"] and retained["survival_passed"])
                or improved["passed"] != (retained["passed"] and improved["first_food_passed"] and improved["survival_passed"])):
            raise ValueError("transmission verdicts disagree with their efficacy and survival gates")
        check_frozen()
        for artifact in [calibration_source["config"], *calibration_source["raw_files"]]:
            _read_pinned(root / artifact["path"], artifact["sha256"])
        if ((folder / "genome_start.npy").read_bytes() != content
                or not np.array_equal(flat, np.load(io.BytesIO(content), allow_pickle=False))):
            raise ValueError("transmission changed the frozen original genotype")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
            scientific_result=args.condition_split == "final", retained_efficacy_status="not_applicable",
            cumulative_improvement_status="not_applicable",
            safeguards=dict(world_keys_disjoint=True, calibration_source_unchanged=True,
                            original_genome_unchanged=True, three_cohort_four_arm_evidence_validated=True))
        if args.condition_split == "final":
            final.update(retained_efficacy=retained, cumulative_improvement=improved,
                retained_efficacy_status="passed" if retained["passed"] else "failed",
                cumulative_improvement_status="passed" if improved["passed"] else "failed")
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
