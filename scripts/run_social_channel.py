"""Tracked E121 public-movie positive control using immutable per-family evidence."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed.seed import Cfg
from scripts import social_channel
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "social-channel-v1"
MAX_SECONDS = {"development": 900, "final": 3600}
KEY_ORDER = ("teacher_world", "teacher_birth")
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("social-channel-v1 accepts only its registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg
            or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.structural_plasticity not in (None, "off") or args.objective is not None
            or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("social-channel-v1 requires its fixed teacher and deterministic decoder configuration")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("channel evaluation requires clean committed source provenance")
    if args.condition_split == "development":
        source, expected = SOURCES[0], ("E121-benchmark", 48, 4)
    else:
        selected = [source for source in SOURCES if args.name == "E121-" + source["label"]]
        if len(selected) != 1:
            raise ValueError("final channel study must be named E121-A0, E121-A1 or E121-A2")
        source = selected[0]
        expected = ("E121-" + source["label"], 2026112701 + source["training_seed"], 128)
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("channel identity, seed and family count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this channel attempt exists; preserve it and register any recovery")
    source_path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != source_path:
        raise ValueError("channel study requires the matching original E103 genome")
    content = source_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("original E103 genome differs from its registered SHA256")
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original genome must contain 155 finite float32 values")
    return dict(source), content, flat


def run_tracked(args, git, root, *, explicit=None):
    started = time.monotonic()
    root = Path(root)
    source, content, flat = _validate(args, git, root, explicit)
    c = Cfg(M=1)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source,
        condition_split=args.condition_split, cfg=c._asdict(), cfg_overrides={},
        ecology=None, persistence="off", plastic=True, controller="rules-v0", genome_size=155,
        generation=0, seed=args.seed, eval_seed=args.seed, eval_worlds=args.eval_worlds,
        expected_families=args.eval_worlds, key_order=list(KEY_ORDER),
        arms=list(social_channel.ARMS), methods=list(social_channel.METHODS),
        practice_steps=c.T // 2, watch_steps=c.T // 2,
        rule_assignment="family index modulo 2",
        yoked_assignment="item channels swapped when (family index // 2) modulo 2 is one",
        decoder="majority contacted visible item color; tie or no vote scores one half",
        track="consecutive teacher offsets infer movement with the lagged external camera",
        scene="unique minimum normalized overlapping item mismatch; consecutive camera shifts infer contact",
        gate=dict(minimum_coverage=.5, minimum_accuracy=.6, standard_errors=2., all_seeds=3),
        max_seconds=MAX_SECONDS[args.condition_split], scientific_result=args.condition_split == "final",
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=_runtime())
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", content)
    rows, artifacts, used_keys = [], [], set()

    def status(value):
        return dict(status=value, completed_families=len(artifacts), expected_families=args.eval_worlds,
                    elapsed_seconds=time.monotonic() - started)

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
                    generation=0, source_label=source["label"], eval_seed=args.seed,
                    eval_worlds=args.eval_worlds, scientific_result=False, artifact_validity="incomplete",
                    gate_status="undecided", raw_files=list(artifacts))

    def check_deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("channel evaluation exceeded its registered process allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                check_deadline()
                keys = np.asarray(social_channel.family_keys(args.seed, index))
                if keys.shape != (2, 2) or keys.dtype != np.uint32:
                    raise ValueError("channel family requires two uint32 random keys")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != 2 or used_keys.intersection(tokens):
                    raise ValueError("teacher random keys must be unique across all families")
                used_keys.update(tokens)
                row = social_channel.evaluate_family(flat, keys, index, c)
                social_channel.validate_family(row, c)
                if (not np.array_equal(row["keys"], keys) or int(row["family_index"]) != index
                        or int(row["rule"]) != index % 2 or int(row["yoked_swap"]) != (index // 2) % 2):
                    raise ValueError("channel family differs from its assigned keys or controls")
                artifact = dict(index=index, **_save_arrays(folder, f"families/family_{index:04d}.npz", row))
                artifacts.append(artifact)
                rows.append(row)
                stream.write(json.dumps(dict(family=index, rule=index % 2, yoked_swap=(index // 2) % 2,
                    keys=keys.tolist(), raw_file=artifact, elapsed_seconds=time.monotonic() - started),
                    allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                print(f"social channel family {index + 1}/{args.eval_worlds} saved", flush=True)
                check_deadline()
        measurements = dict(social_channel.summarize(rows))
        gate = measurements.pop("gate")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     scientific_result=args.condition_split == "final", gate_status="not_applicable")
        if args.condition_split == "final":
            required = {"passed", "coverage_passed", "accuracy_passed", "chance_passed", "yoked_passed"}
            if set(gate) != required or any(type(value) is not bool for value in gate.values()):
                raise ValueError("a completed channel study requires a defined boolean gate")
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
