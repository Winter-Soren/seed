"""E123 immutable calibration followed by held-out engineered belief readout."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed.seed import Cfg
from scripts import social_belief
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "social-belief-readout-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
CALIBRATION_MAX_SECONDS = 900
KEY_ORDER = ("teacher_world", "teacher_birth", "solo_world", "pupil_birth")
ARMS = ("observed", "yoked", "scene_only", "no_demo", "belief_reset", "birth_parent")
CONTROLS = ARMS[1:]
LAMBDA_GRID = (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("social-belief-readout-v1 accepts only its registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("belief readout fixes original genomes, zero ES generations and its calibration design")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("belief readout requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected, calibration_worlds, eval_seed = SOURCES[0], ("E123-benchmark", 51, 4), 4, 52
    else:
        selected = [item for item in SOURCES if args.name == "E123-" + item["label"]]
        if len(selected) != 1:
            raise ValueError("final belief-readout panels must be named E123-A0, E123-A1 or E123-A2")
        source = selected[0]
        expected = ("E123-" + source["label"], 2026113001 + source["training_seed"], 128)
        calibration_worlds, eval_seed = 32, 2026120101 + source["training_seed"]
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("belief-readout identity, calibration seed and family count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this belief-readout attempt exists; preserve it and preregister any recovery")
    source_path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != source_path:
        raise ValueError("belief readout requires the matching original E103 genome")
    content = source_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("original E103 genome differs from its registered SHA256")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original genome must contain 155 finite float32 values")
    if tuple(social_belief.ARMS) != ARMS or tuple(social_belief.LAMBDA_GRID) != LAMBDA_GRID:
        raise ValueError("backend arms or calibration grid differ from registration")
    return dict(source), content, flat, calibration_worlds, eval_seed


def _validate_selection(selection, expected):
    """Reject incomplete or inconsistent fitting records before final worlds exist."""
    if selection.get("families") != expected or type(selection.get("contacts")) is not int:
        raise ValueError("calibration must retain every assigned family and its contact coverage")
    contacts = selection["contacts"]
    if not 0 <= contacts <= expected:
        raise ValueError("invalid calibration contact count")
    losses = selection.get("losses", [])
    if len(losses) != len(LAMBDA_GRID) or [row.get("lambda") for row in losses] != list(LAMBDA_GRID):
        raise ValueError("calibration must save all nine losses in the registered grid order")
    values = [row.get("mean_log_loss") for row in losses]
    if contacts:
        if any(value is None or not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("contact calibration requires nine finite nonnegative losses")
        winner = min(zip(LAMBDA_GRID, values), key=lambda pair: (pair[1], abs(pair[0]), -pair[0]))[0]
    else:
        if any(value is not None for value in values):
            raise ValueError("missing contacts cannot supply an invented calibration target")
        winner = 0.
    if selection.get("selected_lambda") != winner:
        raise ValueError("selected calibration scalar does not follow the registered loss and tie rule")
    for field in ("contact_coverage", "vote_coverage"):
        metric = selection.get(field, {})
        if (metric.get("mean") is None or metric.get("se") is None
                or not np.isfinite([metric["mean"], metric["se"]]).all()
                or not 0 <= metric["mean"] <= 1 or metric["se"] < 0):
            raise ValueError("calibration coverage must include finite family means and standard errors")
    if selection["contact_coverage"]["mean"] != contacts / expected:
        raise ValueError("calibration contact coverage disagrees with its denominator")
    return float(winner)


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root)
    source, content, flat, calibration_worlds, eval_seed = _validate(args, git, root, explicit)
    c = Cfg(M=1)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "calibration").mkdir()
    (folder / "families").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
        ecology=None, persistence="off", plastic=True, controller="rules-v0", genome_size=155,
        generation=0, seed=args.seed, calibration_seed=args.seed, eval_seed=eval_seed,
        calibration_worlds=calibration_worlds, eval_worlds=args.eval_worlds,
        expected_calibration_families=calibration_worlds, expected_families=args.eval_worlds,
        key_order=list(KEY_ORDER), arms=list(ARMS), controls=list(CONTROLS),
        practice_steps=c.T // 2, watch_steps=c.T // 2, solo_steps=c.T,
        lambda_grid=list(LAMBDA_GRID), calibration_max_seconds=CALIBRATION_MAX_SECONDS,
        max_seconds=MAX_SECONDS[args.condition_split], raw_schema=PROTOCOL,
        calibration_raw_pattern="calibration/family_{index:04d}.npz",
        raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        mechanism="engineered public-contact belief and fixed local action readout; scalar calibrated from own outcomes",
        calibration="one unassisted own contact per assigned family; no contact supplies no target",
        calibration_input_fields=["z", "votes", "contact_color", "contact_outcome"],
        calibration_tie_rule="minimum log loss, then smallest absolute lambda, then positive lambda",
        belief="q=sigmoid(lambda*z); z=(visible_A_contacts-visible_B_contacts)/max(1,total_contacts)",
        readout="add 3*(2*q-1)*(A_destination-B_destination) to each ordinary action logit",
        rule_assignment="family index modulo 2; teacher and solo share rule; stages use disjoint keys",
        yoked_assignment="item channels swapped when (family index // 2) modulo 2 is one",
        scientific_result=args.condition_split == "final")
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    rows, calibration_rows, artifacts, calibration_artifacts, used_keys = [], [], [], [], set()
    phase, selection, frozen = "calibration", None, None
    calibration_seconds, evaluation_started = None, None

    def status(value):
        now = time.monotonic()
        return dict(status=value, phase=phase, completed_calibration_families=len(calibration_artifacts),
                    expected_calibration_families=calibration_worlds,
                    completed_families=len(artifacts), expected_families=args.eval_worlds,
                    calibration_seconds=now - started if calibration_seconds is None else calibration_seconds,
                    evaluation_seconds=0. if evaluation_started is None else now - evaluation_started,
                    elapsed_seconds=now - started)

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
                    generation=0, source_label=source["label"], calibration_seed=args.seed, eval_seed=eval_seed,
                    eval_worlds=args.eval_worlds, scientific_result=False, artifact_validity="incomplete",
                    gate_status="undecided", calibration=selection, frozen_calibration=frozen,
                    calibration_raw_files=list(calibration_artifacts), raw_files=list(artifacts))

    def check_deadline():
        elapsed = time.monotonic() - started
        if (elapsed >= MAX_SECONDS[args.condition_split]
                or phase == "calibration" and elapsed >= CALIBRATION_MAX_SECONDS):
            raise TimeoutError("belief readout exceeded its registered calibration or invocation allowance")

    def assigned_keys(master, index):
        keys = np.asarray(social_belief.family_keys(master, index))
        if keys.shape != (4, 2) or keys.dtype != np.uint32:
            raise ValueError("belief families require four uint32 random keys")
        tokens = [key.tobytes() for key in keys]
        if len(set(tokens)) != 4 or used_keys.intersection(tokens):
            raise ValueError("calibration and evaluation must have disjoint teacher/pupil random keys")
        used_keys.update(tokens)
        return keys

    def save_family(row, keys, index, stream, calibration=False):
        if (not np.array_equal(row["keys"], keys) or int(row["family_index"]) != index
                or int(row["rule"]) != index % 2
                or not calibration and int(row["yoked_swap"]) != (index // 2) % 2):
            raise ValueError("belief family differs from its assigned keys, rule or movie control")
        directory = "calibration" if calibration else "families"
        artifact = dict(index=index, **_save_arrays(folder, f"{directory}/family_{index:04d}.npz", row))
        (calibration_artifacts if calibration else artifacts).append(artifact)
        record = dict(phase=phase, family=index, rule=index % 2,
            keys=keys.tolist(), raw_file=artifact, elapsed_seconds=time.monotonic() - started)
        if not calibration:
            record["calibration_sha256"] = frozen["sha256"]
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        _write_status(folder, status("running"))

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(calibration_worlds):
                check_deadline()
                keys = assigned_keys(args.seed, index)
                row = social_belief.calibration_family(flat, keys, index, c)
                social_belief.validate_calibration_family(row, c)
                save_family(row, keys, index, stream, calibration=True)
                projected = social_belief.calibration_input(row)
                if set(projected) != {"z", "votes", "contact_color", "contact_outcome"}:
                    raise ValueError("calibration fitter must receive only public evidence and the pupil's own contact")
                calibration_rows.append(projected)
                print(f"belief calibration family {index + 1}/{calibration_worlds} saved", flush=True)
                check_deadline()
            fitted = dict(social_belief.calibrate(calibration_rows))
            selected_lambda = _validate_selection(fitted, calibration_worlds)
            check_deadline()
            selection = dict(**fitted, status="frozen", calibration_seed=args.seed,
                completed_families=calibration_worlds, expected_families=calibration_worlds,
                elapsed_seconds=time.monotonic() - started, raw_files=list(calibration_artifacts))
            selection_path = folder / "calibration" / "selection.json"
            _write_json(selection_path, selection)
            frozen = dict(path="calibration/selection.json", sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
            check_deadline()
            calibration_seconds = time.monotonic() - started
            stream.write(json.dumps(dict(phase="calibration_frozen", **frozen,
                selected_lambda=selected_lambda, elapsed_seconds=calibration_seconds), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            phase = "evaluation"
            evaluation_started = time.monotonic()
            _write_status(folder, status("running"))
            for index in range(args.eval_worlds):
                check_deadline()
                if hashlib.sha256(selection_path.read_bytes()).hexdigest() != frozen["sha256"]:
                    raise ValueError("frozen calibration changed before or during held-out evaluation")
                keys = assigned_keys(eval_seed, index)
                row = social_belief.evaluate_family(flat, keys, index, selected_lambda, c)
                social_belief.validate_family(row, c)
                if float(row["lambda_value"]) != selected_lambda:
                    raise ValueError("held-out family did not use the frozen calibration scalar")
                row["calibration_sha256"] = np.frombuffer(bytes.fromhex(frozen["sha256"]), dtype=np.uint8).copy()
                save_family(row, keys, index, stream)
                rows.append(row)
                print(f"belief evaluation family {index + 1}/{args.eval_worlds} saved", flush=True)
                check_deadline()
        measurements = dict(social_belief.summarize(rows))
        gate = measurements.pop("gate")
        if set(measurements.get("arms", {})) != set(ARMS) or set(measurements.get("contrasts", {})) != set(CONTROLS):
            raise ValueError("belief evaluation must report all six arms and five paired comparisons")
        if (hashlib.sha256(selection_path.read_bytes()).hexdigest() != frozen["sha256"]
                or (folder / "genome_start.npy").read_bytes() != content
                or not np.array_equal(flat, np.load(io.BytesIO(content), allow_pickle=False))):
            raise ValueError("calibration or original genome artifact changed during evaluation")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
            scientific_result=args.condition_split == "final", gate_status="not_applicable",
            safeguards=dict(world_keys_disjoint=True, calibration_frozen_before_evaluation=True,
                            original_genome_unchanged=True, six_arm_evidence_validated=True))
        if args.condition_split == "final":
            if (set(gate) != {"passed", "first_food_passed", "survival_passed"}
                    or any(type(value) is not bool for value in gate.values())):
                raise ValueError("complete belief evaluation requires a defined boolean intersection gate")
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
