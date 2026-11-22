"""E133 immutable matched-history diagnostics on the original E103 controllers."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed import contact_gate as engine
from seed.seed import Cfg
from scripts.run_cold_donor import SOURCES, ALLOWED_FLAGS, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_encounter_diagnostic import _array_sha, _read_pinned
from scripts.run_retention_diagnostic import _estimate, _advantage
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "contact-gate-v1"
DEVELOPMENT_SEED = 2026121200
MAX_SECONDS = {"development": 220, "final": 600}
ARMS = ("native_stationary", "native_reversal", "contact_stationary", "contact_reversal",
        "sparse_stationary", "sparse_reversal", "reopened_stationary", "reopened_reversal")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "sparse")
APPARATUS = dict(pre_contacts=16, post_contacts=8, block_steps=4, delay_ticks=33, grid_steps=50, local_placements=12)
FIELDS = ("local_S", "solo_Y", "first_food", "first_poison", "no_contact", "exposure_survival", "solo_survival")
AUDIT_PATH = "results/diagnostics/E133-independent-audit.json"
SOURCE_PATHS = ("seed/seed.py", "seed/exploration.py", "seed/direct.py", "seed/encounter_diagnostic.py",
                "seed/contact_gate.py", "scripts/run_contact_gate.py", "scripts/run_cold_donor.py",
                "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py", "scripts/run_shelter_evolution.py")
COMPONENT_MINIMUMS = dict(pre_old=0., gate_immediate=.10, native_immediate=.10, gate_delayed=.10,
                        gate_solo_delayed=0., retention_advantage=.025, gate_local_vs_sparse=.05,
                        delay_ablation=.025, gate_solo_vs_native=.05, gate_solo_vs_sparse=.05)
SAFEGUARD_NAMES = ("acquisition_vs_native", "gate_own_retention", "stationary_local_native", "stationary_local_sparse",
                   "stationary_solo_native", "stationary_solo_sparse", "solo_survival_stationary_native",
                   "solo_survival_stationary_sparse", "solo_survival_reversal_native", "solo_survival_reversal_sparse")


def _gate(components, safeguards):
    checks = {name: _advantage(components[name], minimum) for name, minimum in COMPONENT_MINIMUMS.items()}
    checks.update({name: safeguards[name]["lower_2se"] > -.05 for name in SAFEGUARD_NAMES})
    return dict(criteria=checks, passed=all(checks.values()), failed_criteria=[name for name, passed in checks.items() if not passed])


def summarize(rows):
    """Collapse the two assigned rules within family before every paired estimate."""
    n = len(rows)
    if n < 2 or [int(row["base_index"]) for row in rows] != list(range(n)):
        raise ValueError("contact gates require every assigned independent family in order")
    pre = np.asarray([row["pre_old"] for row in rows], np.float64)
    overlap = np.asarray([row["sparse_feedback_overlap"] for row in rows], np.float64)
    if pre.shape != (n, 2) or not np.isfinite(pre).all() or np.any(np.abs(pre) > 1):
        raise ValueError("old preference must retain both rules and zero-weight dead assignments")
    if (overlap.shape != (n,) or not np.isfinite(overlap).all() or np.any((overlap < 0) | (overlap > 1))
            or np.any(overlap*8 != np.floor(overlap*8))):
        raise ValueError("sparse feedback overlap must count the eight predetermined block offsets")
    arrays = {}
    for arm in ARMS:
        values = {field: np.asarray([row["arms"][arm][field] for row in rows], np.float64) for field in FIELDS}
        if any(value.shape != (n, 2, 2) or not np.isfinite(value).all() for value in values.values()):
            raise ValueError("contact endpoints must retain rule and immediate/delayed axes for every birth")
        binary = ("first_food", "first_poison", "no_contact", "exposure_survival", "solo_survival")
        if (any(not np.isin(values[name], [0, 1]).all() for name in binary)
                or np.any(np.abs(values["local_S"]) > 1)
                or not np.array_equal(values["solo_Y"], values["first_food"]-values["first_poison"])
                or np.any(values["first_food"]+values["first_poison"]+values["no_contact"] != 1)
                or np.any(values["solo_survival"] > values["exposure_survival"])
                or np.any(values["exposure_survival"][:, :, 1] > values["exposure_survival"][:, :, 0])):
            raise ValueError("contact choice categories or absorbing exposure survival disagree")
        dead = values["exposure_survival"] == 0
        if any(np.any(values[name][dead] != 0) for name in ("local_S", "solo_Y", "first_food", "first_poison", "solo_survival")):
            raise ValueError("dead exposure states cannot supply a later choice score")
        arrays[arm] = {name: value.mean(axis=1) for name, value in values.items()}
    native, contact, sparse, reopened = (arrays[name+"_reversal"] for name in ("native", "contact", "sparse", "reopened"))
    change = lambda arm: arm["local_S"][:, 1]-arm["local_S"][:, 0]
    solo_native = contact["solo_Y"][:, 1]-native["solo_Y"][:, 1]
    solo_sparse = contact["solo_Y"][:, 1]-sparse["solo_Y"][:, 1]
    stationary_native = arrays["contact_stationary"]["solo_Y"][:, 1]-arrays["native_stationary"]["solo_Y"][:, 1]
    stationary_sparse = arrays["contact_stationary"]["solo_Y"][:, 1]-arrays["sparse_stationary"]["solo_Y"][:, 1]
    components = dict(pre_old=_estimate(pre.mean(axis=1)), gate_immediate=_estimate(contact["local_S"][:, 0]),
        native_immediate=_estimate(native["local_S"][:, 0]), gate_delayed=_estimate(contact["local_S"][:, 1]),
        gate_solo_delayed=_estimate(contact["solo_Y"][:, 1]), retention_advantage=_estimate(change(contact)-change(native)),
        gate_local_vs_sparse=_estimate(contact["local_S"][:, 1]-sparse["local_S"][:, 1]),
        delay_ablation=_estimate(contact["local_S"][:, 1]-reopened["local_S"][:, 1]),
        gate_solo_vs_native=_estimate(solo_native), gate_solo_vs_sparse=_estimate(solo_sparse),
        adjusted_solo_vs_native=_estimate(solo_native-stationary_native),
        adjusted_solo_vs_sparse=_estimate(solo_sparse-stationary_sparse), sparse_feedback_overlap=_estimate(overlap))
    safeguards = dict(acquisition_vs_native=_estimate(contact["local_S"][:, 0]-native["local_S"][:, 0], bound=True),
                      gate_own_retention=_estimate(change(contact), bound=True))
    for field, label in (("local_S", "local"), ("solo_Y", "solo")):
        for control in ("native", "sparse"):
            safeguards["stationary_"+label+"_"+control] = _estimate(
                arrays["contact_stationary"][field][:, 1]-arrays[control+"_stationary"][field][:, 1], bound=True)
    for history in ("stationary", "reversal"):
        for control in ("native", "sparse"):
            safeguards["solo_survival_"+history+"_"+control] = _estimate(
                arrays["contact_"+history]["solo_survival"][:, 1]-arrays[control+"_"+history]["solo_survival"][:, 1], bound=True)
    return dict(families=n, arms={arm: {name: [_estimate(value[:, tick]) for tick in (0, 1)]
        for name, value in fields.items()} for arm, fields in arrays.items()}, components=components,
        safeguards=safeguards, gate=_gate(components, safeguards))


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("contact gates require the three original lineages separately")
    qualifying = []
    for panel in panels:
        values = panel["measurements"]
        expected = _gate(values["components"], values["safeguards"])
        if values["gate"] != expected or type(values["gate"]["passed"]) is not bool:
            raise ValueError("contact gate disagrees with its paired components")
        if expected["passed"]:
            qualifying.append(panel["index"])
    return dict(contact_gate=dict(count=len(qualifying), seeds=qualifying, passed=len(qualifying) == 3))


def _source_code(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("contact gates accept only the registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS or args.generations != 0
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("contact gates fix the original controller and apparatus with zero training")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("contact gates require clean committed source")
    indices = [i for i in range(3) if args.name == f"E133-A{i}"]
    if args.condition_split == "development":
        index, expected = 0, ("E133-benchmark", DEVELOPMENT_SEED, 4)
    elif len(indices) == 1:
        index = indices[0]
        expected = (f"E133-A{index}", DEVELOPMENT_SEED+1+index, 64)
    else:
        raise ValueError("unknown contact-gate lineage; no retries or recovery names are registered")
    if (args.name, args.seed, args.eval_worlds) != expected or any((root / "runs").glob("*_" + args.name)):
        raise ValueError("contact-gate identity differs from registration or its immutable attempt already exists")
    source = SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("contact gates require the original matching E103 genome_final.npy")
    content = _read_pinned(root / source["path"], source["sha256"])
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("the saved controller must retain 155 finite float32 values")
    if (engine.PROTOCOL != PROTOCOL or tuple(engine.ARMS) != ARMS or tuple(engine.KEY_ROLES) != KEY_ROLES
            or engine.CFG != Cfg(M=1) or engine.ASSIGNMENT_TAG != 0xE133 or engine.APPARATUS != APPARATUS
            or tuple(engine.FIELDS) != FIELDS):
        raise ValueError("contact-gate backend differs from the registered one-creature apparatus")
    return index, source, content, flat


def run_tracked(args, git, root, *, explicit=None):
    started, root = time.monotonic(), Path(root).resolve()
    index, source, content, flat = _validate(args, git, root, explicit)
    n = args.eval_worlds
    assignments = {name: np.asarray(value) for name, value in engine.assignments_manifest(args.seed, n).items()}
    if (set(assignments) != {"family_keys", "initial_rules"} or assignments["family_keys"].shape != (n, len(KEY_ROLES), 2)
            or assignments["family_keys"].dtype != np.uint32 or assignments["initial_rules"].dtype != np.int32
            or not np.array_equal(assignments["initial_rules"], [0, 1])
            or len({key.tobytes() for key in assignments["family_keys"].reshape(-1, 2)}) != n*len(KEY_ROLES)):
        raise ValueError("contact gates require both rules and distinct family role streams")
    code = _source_code(root)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    manifest = _save_arrays(folder, "assignments.npz", assignments)
    config = dict(protocol=PROTOCOL, args=dict(vars(args), start="file:"+source["path"]), git=dict(git),
        source_genome=source, source_generations=[200], cfg=engine.CFG._asdict(), cfg_overrides={},
        condition_split=args.condition_split, generation=0, seed_index=index, eval_seed=args.seed, expected_families=n,
        statistical_unit="independent family averaging both initially assigned rule replays", assignments=manifest,
        arms=list(ARMS), key_roles=list(KEY_ROLES), initial_rules=[0, 1], assignment_tag=0xE133, apparatus=dict(APPARATUS),
        source_code=code, raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        max_seconds=MAX_SECONDS[args.condition_split], scientific_result=False)
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_final.npy", content)
    rows, artifacts, seconds, walls = [], [], [], []
    setup = time.monotonic()-started

    def status(value):
        if (len(walls) > len(artifacts) or any(not np.isfinite(t) or t < 0 for t in walls)
                or value == "finished" and len(walls) != n):
            raise ValueError("family wall timing must preserve the known completed prefix")
        return dict(status=value, completed_families=len(artifacts), expected_families=n, setup_seconds=setup,
            elapsed_seconds=time.monotonic()-started, family_seconds=list(seconds), family_wall_seconds=list(walls))

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split, seed_index=index,
            evaluation_seed=args.seed, scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=list(artifacts))

    def deadline():
        if time.monotonic()-started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("contact gates exceeded the registered invocation allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for family, keys in enumerate(assignments["family_keys"]):
                deadline()
                family_started = time.monotonic()
                if not np.array_equal(keys, np.asarray(engine.evaluation_keys(args.seed, family))):
                    raise ValueError("family roles differ from their registered key derivation")
                raw = engine.evaluate_world(flat, keys, family, c=engine.CFG)
                compact = engine.summary_input(raw)
                if (int(raw["base_index"]) != family or not np.array_equal(raw["keys"], keys)
                        or not np.array_equal(raw["initial_rules"], [0, 1]) or not np.array_equal(raw["genome_sha256"], _array_sha(flat))):
                    raise ValueError("raw contact-gate evidence differs from the assigned source or family")
                artifact = dict(index=family, **_save_arrays(folder, f"families/family_{family:04d}.npz", raw))
                artifacts.append(artifact)
                rows.append(compact)
                seconds.append(time.monotonic()-family_started)
                stream.write(json.dumps(dict(family=family, keys=keys.tolist(), raw_file=artifact,
                    family_seconds=seconds[-1], elapsed_seconds=time.monotonic()-started), allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                walls.append(time.monotonic()-family_started)
                print(f"contact-gate family {family+1}/{n} saved", flush=True)
                del raw
                deadline()
        measurements = summarize(rows)
        _read_pinned(root / source["path"], source["sha256"])
        if (folder / "genome_final.npy").read_bytes() != content or flat.tobytes() != np.load(io.BytesIO(content), allow_pickle=False).tobytes():
            raise ValueError("contact-gate evaluation changed its saved source genome")
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


def _same_measurements(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_measurements(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(_same_measurements(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return isinstance(right, (float, int)) and not isinstance(right, bool) and bool(np.isclose(left, right, rtol=1e-10, atol=1e-9))
    return type(left) is type(right) and left == right


def build_comparison(root):
    root, selected = Path(root).resolve(), []
    for index in range(3):
        folders = list((root / "runs").glob(f"*_E133-A{index}"))
        if len(folders) != 1 or not (folders[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete contact-gate panels are required")
        folder = folders[0]
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if state.get("status") != "finished" or state.get("completed_families") != 64 or state.get("expected_families") != 64:
            return dict(status="undecided", reason="a registered contact-gate panel is incomplete")
        if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "final"
                or config.get("args", {}).get("name") != f"E133-A{index}" or config.get("eval_seed") != DEVELOPMENT_SEED+1+index
                or config.get("source_genome") != SOURCES[index] or config.get("expected_families") != 64
                or config.get("apparatus") != APPARATUS or config.get("git", {}).get("dirty") is not False):
            raise ValueError("contact-gate panel metadata differs from its registered lineage")
        selected.append((folder, config))
    path = root / AUDIT_PATH
    if not path.exists():
        return dict(status="undecided", reason="independent joint contact-gate verification is pending")
    raw_audit = path.read_bytes()
    audit = json.loads(raw_audit)
    if audit.get("status") != "passed" or audit.get("protocol") != PROTOCOL or audit.get("mode") != "final":
        return dict(status="undecided", reason="independent final contact-gate verification has not passed")
    registered, checked = audit.get("registered_source", ""), audit.get("checked_file_sha256", {})
    if (not isinstance(registered, str) or len(registered) != 40 or any(c not in "0123456789abcdef" for c in registered)
            or not isinstance(checked, dict) or [(p.get("index"), p.get("run")) for p in audit.get("panels", [])]
            != [(i, folder.relative_to(root).as_posix()) for i, (folder, _) in enumerate(selected)]):
        raise ValueError("contact-gate audit source or exact panel selection is invalid")

    def verified(file, expected=None):
        digest = checked.get(file.relative_to(root).as_posix())
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("joint contact-gate audit does not cover current evidence")
        return _read_pinned(file, digest)

    panels = []
    for index, (folder, config) in enumerate(selected):
        if config["git"]["commit"] != registered:
            raise ValueError("contact-gate panels must use the independently registered source commit")
        for name in ("config.json", "status.json", "metrics.jsonl"):
            verified(folder / name)
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        verified(folder / "genome_final.npy", SOURCES[index]["sha256"])
        verified(root / SOURCES[index]["path"], SOURCES[index]["sha256"])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 64 or result.get("expected_families") != 64 or len(result.get("raw_files", [])) != 64):
            raise ValueError("an audited contact-gate panel is incomplete")
        for index_in_panel, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{index_in_panel:04d}.npz"
            if artifact.get("index") != index_in_panel or artifact.get("path") != relative:
                raise ValueError("contact-gate evidence must preserve every original family exactly once")
            verified(folder / relative, artifact["sha256"])
        values = result["measurements"]
        if values.get("families") != 64 or not _same_measurements(values, audit["panels"][index].get("measurements")):
            raise ValueError("contact-gate measurements differ from the independent reconstruction")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=values))
    gates = _replicated_gates(panels)
    if audit.get("replicated_gates") != gates:
        raise ValueError("runner and independent contact-gate decisions disagree")
    return dict(status="complete", protocol=PROTOCOL, scientific_result=True, registered_source=registered,
        audit=dict(path=AUDIT_PATH, sha256=hashlib.sha256(raw_audit).hexdigest()), panels=panels, replicated_gates=gates)
