"""E128 immutable retention diagnostics from original E103/E113 controllers."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed import direct, retention_diagnostic as engine
from seed.seed import Cfg
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_encounter_diagnostic import _array_sha, _read_pinned
from scripts.run_shelter_evolution import _save_arrays
from scripts.run_social_reversal import DIRECT_SOURCES


PROTOCOL = "retention-diagnostic-v1"
MAX_SECONDS = {"development": 120, "final": 300}
ARMS = ("plastic_stationary", "plastic_reversal", "direct_stationary", "direct_reversal")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "retention")
RETENTION_TICKS = (0, 1, 33)
LOCAL_FIELDS = ("p_food", "p_poison", "approach", "other", "S")
GRID_FIELDS = ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")
AUDIT_PATH = "results/diagnostics/E128-independent-audit.json"
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}


def _estimate(values, *, bound=False):
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("retention uncertainty needs at least two finite independent families")
    result = dict(mean=float(values.mean()), se=float(values.std(ddof=1) / np.sqrt(len(values))), n=len(values))
    if bound:
        result["lower_2se"] = result["mean"] - 2 * result["se"]
    return result


def _advantage(value, minimum=0.):
    return value["mean"] >= minimum and value["mean"] > 2 * value["se"]


def summarize(rows):
    """Average both rules within family before every uncertainty and paired contrast."""
    n = len(rows)
    if n < 2 or [int(row["base_index"]) for row in rows] != list(range(n)):
        raise ValueError("retention summaries require the complete ordered family panel")
    values, arms = {}, {}
    shapes = {"pre_old_by_rule": (2,), "pre_local_by_rule": (2, 5),
              "local_by_rule": (2, 3, 5), "grid_by_rule": (2, 3, 7)}
    for arm in ARMS:
        by_rule = {}
        for name, shape in shapes.items():
            array = np.asarray([row[f"{arm}_{name}"] for row in rows], np.float64)
            if array.shape != (n, *shape) or not np.isfinite(array).all():
                raise ValueError("retention compact evidence has invalid shape or nonfinite values")
            by_rule[name] = array
        values[arm] = by_rule
        paired = {name: array.mean(axis=1) for name, array in by_rule.items()}
        arms[arm] = dict(pre_old=_estimate(paired["pre_old_by_rule"]),
            pre_local={name: _estimate(paired["pre_local_by_rule"][:, j]) for j, name in enumerate(LOCAL_FIELDS)},
            by_delay={str(t): dict(
                local={name: _estimate(paired["local_by_rule"][:, i, j]) for j, name in enumerate(LOCAL_FIELDS)},
                grid={name: _estimate(paired["grid_by_rule"][:, i, j]) for j, name in enumerate(GRID_FIELDS)})
                for i, t in enumerate(RETENTION_TICKS)})
    architectures, by_initial_rule = {}, {"0": {}, "1": {}}
    for architecture in ("plastic", "direct"):
        reversal, stationary = values[architecture + "_reversal"], values[architecture + "_stationary"]
        local = reversal["local_by_rule"][:, :, :, 4].mean(axis=1)
        contrast = local - stationary["local_by_rule"][:, :, :, 4].mean(axis=1)
        food = reversal["grid_by_rule"][:, :, :, 0].mean(axis=1)
        grid_contrast = food - stationary["grid_by_rule"][:, :, :, 0].mean(axis=1)
        preference = reversal["grid_by_rule"][:, :, :, 3].mean(axis=1)
        local_result = dict(by_delay={str(t): dict(signed_preference=_estimate(local[:, i]),
            reversal_minus_stationary=_estimate(contrast[:, i])) for i, t in enumerate(RETENTION_TICKS)},
            change_33_minus_0=_estimate(local[:, 2] - local[:, 0], bound=True))
        grid_result = dict(by_delay={str(t): dict(first_food=_estimate(food[:, i]),
            food_minus_poison=_estimate(preference[:, i]), reversal_minus_stationary=_estimate(grid_contrast[:, i]))
            for i, t in enumerate(RETENTION_TICKS)},
            change_33_minus_0=_estimate(food[:, 2] - food[:, 0], bound=True))
        local_checks = {**{f"S{t}": _advantage(local_result["by_delay"][str(t)]["signed_preference"], .1) for t in (0, 33)},
                        **{f"C{t}": _advantage(local_result["by_delay"][str(t)]["reversal_minus_stationary"], .1) for t in (0, 33)},
                        "retention": local_result["change_33_minus_0"]["lower_2se"] > -.05}
        grid_checks = {**{f"Y{t}": _advantage(grid_result["by_delay"][str(t)]["food_minus_poison"]) for t in (0, 33)},
                       "C33": _advantage(grid_result["by_delay"]["33"]["reversal_minus_stationary"], .1),
                       "retention": grid_result["change_33_minus_0"]["lower_2se"] > -.05}
        for result, checks in ((local_result, local_checks), (grid_result, grid_checks)):
            result.update(criteria=checks, passed=all(checks.values()),
                          failed_criteria=[name for name, passed in checks.items() if not passed])
        architectures[architecture] = dict(local=local_result, grid=grid_result,
            combined_passed=local_result["passed"] and grid_result["passed"])
        for rule in (0, 1):
            by_initial_rule[str(rule)][architecture] = dict(pre_old=_estimate(reversal["pre_old_by_rule"][:, rule]),
                by_delay={str(t): dict(
                    S=_estimate(reversal["local_by_rule"][:, rule, i, 4]),
                    C=_estimate(reversal["local_by_rule"][:, rule, i, 4] - stationary["local_by_rule"][:, rule, i, 4]),
                    G=_estimate(reversal["grid_by_rule"][:, rule, i, 0]),
                    Y=_estimate(reversal["grid_by_rule"][:, rule, i, 3]),
                    grid_difference=_estimate(reversal["grid_by_rule"][:, rule, i, 0] - stationary["grid_by_rule"][:, rule, i, 0]))
                    for i, t in enumerate(RETENTION_TICKS)})
    return dict(families=n, initial_rule_replays=2*n, retention_ticks=list(RETENTION_TICKS),
                arms=arms, architectures=architectures, by_initial_rule=by_initial_rule)


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("retention replication requires all three registered lineages")
    decisions = {}
    for architecture in ("plastic", "direct"):
        local, combined = [], []
        for panel in panels:
            evidence = panel["measurements"]["architectures"][architecture]
            passed = [evidence[part]["passed"] for part in ("local", "grid")]
            if any(type(value) is not bool for value in (*passed, evidence["combined_passed"])):
                raise ValueError("complete retention panels need defined decisions")
            if evidence["combined_passed"] != all(passed):
                raise ValueError("retention combined decision must use the same lineage's local and grid gates")
            if passed[0]:
                local.append(panel["index"])
                if passed[1]:
                    combined.append(panel["index"])
        decisions[architecture] = {name: dict(count=len(indices), seeds=indices, passed=len(indices) == 3)
                                   for name, indices in (("local", local), ("combined", combined))}
    return decisions


def _source_code(root):
    names = ("seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py", "seed/retention_diagnostic.py",
             "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("retention diagnostics accept only registered source and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS or args.generations != 0
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("retention diagnostics fix original controllers and zero training generations")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("retention diagnostics require clean committed source provenance")
    matches = [i for i in range(3) if args.name == f"E128-A{i}"]
    if args.condition_split == "development":
        index, expected = 0, ("E128-benchmark", 2026120700, 4)
    elif len(matches) == 1:
        index = matches[0]
        expected = (f"E128-A{index}", 2026120701 + index, 128)
    else:
        raise ValueError("final retention panels must be E128-A0, E128-A1 or E128-A2")
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("retention name, seed and family count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this retention attempt exists; preserve it and preregister any recovery")
    source, baseline = SOURCES[index], DIRECT_SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("retention diagnostics require the matching original E103 genome")
    original, recurrent = (_read_pinned(root / item["path"], item["sha256"]) for item in (source, baseline))
    plastic, fixed = (np.load(io.BytesIO(raw), allow_pickle=False) for raw in (original, recurrent))
    if (plastic.shape != (155,) or fixed.shape != (direct.genome_size(Cfg(M=1)),)
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (plastic, fixed))):
        raise ValueError("retention source controllers must retain their finite float32 encodings")
    if (engine.PROTOCOL != PROTOCOL or tuple(engine.ARMS) != ARMS or tuple(engine.KEY_ROLES) != KEY_ROLES
            or tuple(engine.RETENTION_TICKS) != RETENTION_TICKS or engine.GRID_STEPS != 50
            or tuple(engine.LOCAL_FIELDS) != LOCAL_FIELDS or tuple(engine.GRID_FIELDS) != GRID_FIELDS):
        raise ValueError("retention engine differs from its frozen design")
    return source, baseline, original, recurrent, plastic, fixed


def run_tracked(args, git, root, *, explicit=None):
    started, root = time.monotonic(), Path(root).resolve()
    source, baseline, original, recurrent, plastic, fixed = _validate(args, git, root, explicit)
    c = Cfg(M=1)
    schedule = {name: np.asarray(value) for name, value in engine.assignments_manifest(args.seed, args.eval_worlds).items()}
    n = args.eval_worlds
    if (set(schedule) != {"key", "permutation", "pre_contacts"} or schedule["key"].shape != (2,)
            or schedule["key"].dtype != np.uint32 or schedule["permutation"].shape != (n,)
            or schedule["permutation"].dtype != np.int32 or schedule["pre_contacts"].shape != (n,)
            or schedule["pre_contacts"].dtype != np.int32
            or not np.array_equal(np.sort(schedule["permutation"]), np.arange(n))
            or not np.array_equal(schedule["pre_contacts"], np.repeat([8, 16], n // 2)[schedule["permutation"]])):
        raise ValueError("retention assignments must balance registered eight/sixteen-contact boundaries")
    source_code = _source_code(root)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    assignments = _save_arrays(folder, "assignments.npz", schedule)
    recorded_args = dict(vars(args), start="file:" + source["path"])
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source, direct_source=baseline,
        cfg=c._asdict(), cfg_overrides={}, condition_split=args.condition_split, generation=0, eval_seed=args.seed,
        expected_families=n, statistical_unit="independent paired base family", assignments=assignments,
        assignment_stream_tag=0xE128, arms=list(ARMS), key_roles=list(KEY_ROLES), initial_rules=[0, 1],
        pre_contact_choices=[8, 16], post_contacts=16, retention_ticks=list(RETENTION_TICKS),
        grid_steps=50, probe_placements=12, source_code=source_code, raw_schema=PROTOCOL,
        raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        max_seconds=MAX_SECONDS[args.condition_split], scientific_result=False)
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "plastic_genome.npy", original)
    _write_bytes(folder / "direct_genome.npy", recurrent)
    rows, artifacts, durations, walls, used = [], [], [], [], {schedule["key"].tobytes()}
    setup = time.monotonic() - started

    def status(value):
        if (len(walls) > len(artifacts) or any(not np.isfinite(t) or t < 0 for t in walls)
                or value == "finished" and len(walls) != n):
            raise ValueError("retention wall timings must preserve a nonnegative complete prefix")
        return dict(status=value, completed_families=len(artifacts), expected_families=n,
            elapsed_seconds=time.monotonic()-started, setup_seconds=setup, family_seconds=list(durations),
            family_wall_seconds=list(walls), family_pre_contacts=schedule["pre_contacts"][:len(artifacts)].tolist())

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
            source_label=source["label"], seed_index=source["training_seed"], evaluation_seed=args.seed,
            scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=list(artifacts))

    def deadline():
        if time.monotonic()-started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("retention diagnostic exceeded its registered invocation allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index, boundary in enumerate(schedule["pre_contacts"]):
                deadline()
                family_started = time.monotonic()
                keys = np.asarray(engine.evaluation_keys(args.seed, index))
                if keys.dtype != np.uint32 or keys.shape != (7, 2):
                    raise ValueError("retention keys must be an ordered seven-role uint32 table")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != 7 or used.intersection(tokens):
                    raise ValueError("retention family or assignment keys were reused")
                used.update(tokens)
                raw = engine.evaluate_world(plastic, fixed, keys, index, int(boundary), c=c)
                projected = engine.summary_input(raw)
                if (not np.array_equal(raw["keys"], keys) or int(raw["base_index"]) != index
                        or int(raw["pre_contacts"]) != boundary or not np.array_equal(raw["initial_rules"], [0, 1])
                        or not np.array_equal(raw["retention_ticks"], RETENTION_TICKS)
                        or not np.array_equal(raw["plastic_genome_sha256"], _array_sha(plastic))
                        or not np.array_equal(raw["direct_genome_sha256"], _array_sha(fixed))):
                    raise ValueError("retention evidence differs from its assigned family or frozen controllers")
                artifact = dict(index=index, **_save_arrays(folder, f"families/family_{index:04d}.npz", raw))
                artifacts.append(artifact)
                rows.append(projected)
                durations.append(time.monotonic()-family_started)
                stream.write(json.dumps(dict(family=index, pre_contacts=int(boundary), keys=keys.tolist(), raw_file=artifact,
                    family_seconds=durations[-1], elapsed_seconds=time.monotonic()-started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                walls.append(time.monotonic()-family_started)
                print(f"retention family {index + 1}/{n} saved", flush=True)
                del raw
                deadline()
        measurements = summarize(rows)
        for path, content, value in (("plastic_genome.npy", original, plastic), ("direct_genome.npy", recurrent, fixed)):
            if ((folder / path).read_bytes() != content
                    or value.tobytes() != np.load(io.BytesIO(content), allow_pickle=False).tobytes()):
                raise ValueError("retention evaluation changed a frozen source controller")
        for item in (source, baseline):
            _read_pinned(root / item["path"], item["sha256"])
        result = summary("finished")
        result.update(artifact_validity="passed", measurements=measurements,
                      gate_status="not_applicable" if args.condition_split == "development" else "undecided")
        deadline()
        _write_json(folder / "summary.json", result)
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        if not (folder / "summary.json").exists():
            _write_json(folder / "summary.json", dict(summary(value), error=f"{type(error).__name__}: {error}"))
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder


def build_comparison(root):
    """Release measurements only after all three complete panels and an independent audit."""
    root = Path(root).resolve()
    folders, configs = [], []
    for index in range(3):
        candidates = list((root / "runs").glob(f"*_E128-A{index}"))
        if len(candidates) != 1 or not (candidates[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete retention panels are required")
        folder = candidates[0]
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if state.get("status") != "finished" or state.get("completed_families") != 128 or state.get("expected_families") != 128:
            return dict(status="undecided", reason="a registered retention panel is incomplete")
        if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "final"
                or config.get("args", {}).get("name") != f"E128-A{index}"
                or config.get("source") != SOURCES[index] or config.get("direct_source") != DIRECT_SOURCES[index]
                or config.get("eval_seed") != 2026120701+index or config.get("expected_families") != 128
                or config.get("git", {}).get("dirty") is not False or config.get("retention_ticks") != list(RETENTION_TICKS)):
            raise ValueError("retention panel metadata differs from its registered source, seed or protocol")
        folders.append(folder)
        configs.append(config)
    path = root / AUDIT_PATH
    if not path.exists():
        return dict(status="undecided", reason="independent three-panel retention verification is pending")
    raw_audit = path.read_bytes()
    audit = json.loads(raw_audit)
    if audit.get("status") != "passed" or audit.get("protocol") != PROTOCOL or audit.get("mode") != "final":
        return dict(status="undecided", reason="independent final retention verification has not passed")
    registered, checked = audit.get("registered_source", ""), audit.get("checked_file_sha256", {})
    if (not isinstance(registered, str) or len(registered) != 40 or any(ch not in "0123456789abcdef" for ch in registered)
            or not isinstance(checked, dict)
            or [(p.get("index"), p.get("run")) for p in audit.get("panels", [])]
            != [(i, folder.relative_to(root).as_posix()) for i, folder in enumerate(folders)]):
        raise ValueError("retention audit source or selected-panel provenance is invalid")

    def verified(path, expected=None):
        digest = checked.get(path.relative_to(root).as_posix())
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("retention audit does not cover the current artifact manifest")
        return _read_pinned(path, digest)

    panels = []
    for index, (folder, config) in enumerate(zip(folders, configs)):
        for name in ("config.json", "status.json", "metrics.jsonl"):
            verified(folder / name)
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        for name, source in (("plastic_genome.npy", SOURCES[index]), ("direct_genome.npy", DIRECT_SOURCES[index])):
            verified(folder / name, source["sha256"])
            verified(root / source["path"], source["sha256"])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 128 or result.get("expected_families") != 128
                or len(result.get("raw_files", [])) != 128):
            raise ValueError("an audited retention panel is incomplete")
        for family, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{family:04d}.npz"
            if artifact.get("index") != family or artifact.get("path") != relative:
                raise ValueError("retention evidence must contain every assigned family exactly once")
            verified(folder / relative, artifact["sha256"])
        measurements = result["measurements"]
        if (measurements.get("families") != 128 or measurements.get("initial_rule_replays") != 256
                or measurements.get("retention_ticks") != list(RETENTION_TICKS)):
            raise ValueError("retention denominators or observation delays changed")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=measurements))
    decisions = _replicated_gates(panels)
    if audit.get("replicated_gates") != decisions:
        raise ValueError("runner and independent audit disagree on retention decisions")
    return dict(status="complete", protocol=PROTOCOL, scientific_result=True, registered_source=registered,
                audit=dict(path=AUDIT_PATH, sha256=hashlib.sha256(raw_audit).hexdigest()),
                panels=panels, replicated_gates=decisions)
