"""E132 immutable, all-assigned voluntary-offer diagnostics on saved E103 genomes."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from seed import voluntary_offers as engine
from seed.seed import Cfg
from scripts.run_cold_donor import SOURCES, ALLOWED_FLAGS, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_encounter_diagnostic import _array_sha, _read_pinned
from scripts.run_retention_diagnostic import _estimate, _advantage
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "voluntary-offers-v1"
DEVELOPMENT_SEED = 2026121100
MAX_SECONDS = {"development": 180, "final": 500}
ARMS = ("near_stationary", "near_reversal", "sham_stationary", "sham_reversal")
KEY_ROLES = ("birth", "geometry", "exposure_action", "solo_world", "solo_action", "probe")
APPARATUS = dict(acquisition_blocks=8, continuation_blocks=8, block_steps=8,
                 split_tick=64, exposure_ticks=128, feedback_ticks=1, solo_steps=50, local_placements=12)
FIELDS = ("local_S", "both_colors", "solo_Y", "first_food", "first_poison", "no_contact",
          "exposure_survival", "readout_survival", "solo_survival", "intake_A", "intake_B", "realized_energy")
AUDIT_PATH = "results/diagnostics/E132-independent-audit.json"
SOURCE_PATHS = ("seed/seed.py", "seed/exploration.py", "seed/direct.py", "seed/encounter_diagnostic.py",
               "seed/voluntary_offers.py", "scripts/run_voluntary_offers.py", "scripts/run_cold_donor.py",
               "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py", "scripts/run_shelter_evolution.py")


def _gate(components, safeguards):
    checks = dict(pre_old_positive=_advantage(components["pre_old"]),
        near_reversal_local_positive=_advantage(components["near_reversal_local"]),
        both_colors_improve=_advantage(components["both_colors_gain"], .10),
        near_reversal_solo_positive=_advantage(components["near_reversal_solo"]),
        solo_gain=_advantage(components["solo_gain"], .05),
        adjusted_solo_gain=_advantage(components["adjusted_solo_gain"]),
        stationary_solo_noninferiority=safeguards["stationary_solo"]["lower_2se"] > -.05,
        reversal_survival_noninferiority=safeguards["reversal_exposure_survival"]["lower_2se"] > -.05,
        stationary_survival_noninferiority=safeguards["stationary_exposure_survival"]["lower_2se"] > -.05)
    return dict(criteria=checks, passed=all(checks.values()), failed_criteria=[key for key, passed in checks.items() if not passed])


def summarize(rows):
    n = len(rows)
    if n < 2 or [int(row["base_index"]) for row in rows] != list(range(n)):
        raise ValueError("offers require every assigned independent family in order")
    pre = np.asarray([row["pre_old_S_by_rule"] for row in rows], np.float64)
    if pre.shape != (n, 2) or not np.isfinite(pre).all() or np.any(np.abs(pre) > 1):
        raise ValueError("preference must retain both rules and zero-weight dead assignments")
    arrays = {}
    for arm in ARMS:
        values = {field: np.asarray([row[arm + "_" + field + "_by_rule"] for row in rows], np.float64) for field in FIELDS}
        if any(value.shape != (n, 2) or not np.isfinite(value).all() for value in values.values()):
            raise ValueError("offer endpoints must retain two rule replays for every birth")
        binary = ("both_colors", "first_food", "first_poison", "no_contact", "exposure_survival", "readout_survival", "solo_survival")
        if (any(not np.isin(values[name], [0, 1]).all() for name in binary)
                or np.any(np.abs(values["local_S"]) > 1)
                or not np.array_equal(values["solo_Y"], values["first_food"]-values["first_poison"])
                or np.any(values["first_food"]+values["first_poison"]+values["no_contact"] != 1)
                or np.any(values["readout_survival"] > values["exposure_survival"])
                or np.any(values["solo_survival"] > values["readout_survival"])):
            raise ValueError("offer choice categories or absorbing survival disagree")
        for name in ("intake_A", "intake_B"):
            if np.any(values[name] < 0) or np.any(values[name] > 8) or np.any(values[name] != np.floor(values[name])):
                raise ValueError("each color is offered only once in each of eight continuation blocks")
        if not np.array_equal(values["both_colors"], (values["intake_A"] > 0) & (values["intake_B"] > 0)):
            raise ValueError("both-color exposure must count actual intake including fatal contacts")
        dead = values["readout_survival"] == 0
        if any(np.any(values[name][dead] != 0) for name in ("local_S", "solo_Y", "first_food", "first_poison", "solo_survival")):
            raise ValueError("dead exposure states cannot supply a later choice score")
        arrays[arm] = values

    def measure(rule=None):
        collapse = lambda value: value.mean(axis=1) if rule is None else value[:, rule]
        family = {arm: {name: collapse(value) for name, value in fields.items()} for arm, fields in arrays.items()}
        near, sham = family["near_reversal"], family["sham_reversal"]
        near_stat, sham_stat = family["near_stationary"], family["sham_stationary"]
        solo_gain = near["solo_Y"]-sham["solo_Y"]
        stationary_gain = near_stat["solo_Y"]-sham_stat["solo_Y"]
        components = dict(pre_old=_estimate(collapse(pre)), near_reversal_local=_estimate(near["local_S"]),
            both_colors_gain=_estimate(near["both_colors"]-sham["both_colors"]),
            near_reversal_solo=_estimate(near["solo_Y"]), solo_gain=_estimate(solo_gain),
            adjusted_solo_gain=_estimate(solo_gain-stationary_gain))
        safeguards = dict(stationary_solo=_estimate(stationary_gain, bound=True),
            reversal_exposure_survival=_estimate(near["exposure_survival"]-sham["exposure_survival"], bound=True),
            stationary_exposure_survival=_estimate(near_stat["exposure_survival"]-sham_stat["exposure_survival"], bound=True))
        return dict(pre_old_local=_estimate(collapse(pre)),
            arms={arm: {name: _estimate(value) for name, value in fields.items()} for arm, fields in family.items()},
            components=components, safeguards=safeguards)

    pooled = measure()
    return dict(families=n, initial_rule_replays=2*n, continuation_lives=8*n, agents_per_rule=1,
                apparatus=dict(APPARATUS), **pooled, gate=_gate(pooled["components"], pooled["safeguards"]),
                by_initial_rule={str(rule): measure(rule) for rule in (0, 1)})


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("voluntary access requires the three original lineages separately")
    qualifying = []
    for panel in panels:
        values = panel["measurements"]
        expected = _gate(values["components"], values["safeguards"])
        if values["gate"] != expected or type(values["gate"]["passed"]) is not bool:
            raise ValueError("voluntary access gate disagrees with its paired components")
        if expected["passed"]:
            qualifying.append(panel["index"])
    return dict(voluntary_access=dict(count=len(qualifying), seeds=qualifying, passed=len(qualifying) == 3))


def _source_code(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("voluntary offers accepts only the registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS or args.generations != 0
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("voluntary offers fixes the original controller and apparatus with zero training")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("voluntary offers requires clean committed source")
    indices = [i for i in range(3) if args.name == f"E132-A{i}"]
    if args.condition_split == "development":
        index, expected = 0, ("E132-benchmark", DEVELOPMENT_SEED, 4)
    elif len(indices) == 1:
        index = indices[0]
        expected = (f"E132-A{index}", DEVELOPMENT_SEED+1+index, 64)
    else:
        raise ValueError("unknown voluntary-offer lineage; no retries or recovery names are registered")
    if (args.name, args.seed, args.eval_worlds) != expected or any((root / "runs").glob("*_" + args.name)):
        raise ValueError("offer identity differs from registration or its immutable attempt already exists")
    source = SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("offers require the original matching E103 genome_final.npy")
    content = _read_pinned(root / source["path"], source["sha256"])
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("the saved controller must retain 155 finite float32 values")
    if (engine.PROTOCOL != PROTOCOL or tuple(engine.ARMS) != ARMS or tuple(engine.KEY_ROLES) != KEY_ROLES
            or engine.CFG != Cfg(M=1) or engine.ASSIGNMENT_TAG != 0xE132
            or (engine.BLOCK_STEPS, engine.PRE_BLOCKS, engine.POST_BLOCKS, engine.GRID_STEPS) != (8, 8, 8, 50)):
        raise ValueError("offer backend differs from the registered one-creature apparatus")
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
        raise ValueError("offers require both rules and distinct family role streams")
    code = _source_code(root)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    manifest = _save_arrays(folder, "assignments.npz", assignments)
    config = dict(protocol=PROTOCOL, args=dict(vars(args), start="file:"+source["path"]), git=dict(git),
        source_genome=source, source_generations=[200], cfg=engine.CFG._asdict(), cfg_overrides={},
        condition_split=args.condition_split, generation=0, seed_index=index, eval_seed=args.seed, expected_families=n,
        statistical_unit="independent family averaging both initially assigned rule replays", assignments=manifest,
        arms=list(ARMS), key_roles=list(KEY_ROLES), initial_rules=[0, 1], assignment_tag=0xE132, apparatus=dict(APPARATUS),
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
            raise TimeoutError("voluntary offers exceeded its registered invocation allowance")

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
                    raise ValueError("raw offer evidence differs from the assigned source or family")
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
                print(f"voluntary-offer family {family+1}/{n} saved", flush=True)
                del raw
                deadline()
        measurements = summarize(rows)
        _read_pinned(root / source["path"], source["sha256"])
        if (folder / "genome_final.npy").read_bytes() != content or flat.tobytes() != np.load(io.BytesIO(content), allow_pickle=False).tobytes():
            raise ValueError("offer evaluation changed its saved source genome")
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
    root, selected = Path(root).resolve(), []
    for index in range(3):
        folders = list((root / "runs").glob(f"*_E132-A{index}"))
        if len(folders) != 1 or not (folders[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete voluntary-offer panels are required")
        folder = folders[0]
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if state.get("status") != "finished" or state.get("completed_families") != 64 or state.get("expected_families") != 64:
            return dict(status="undecided", reason="a registered offer panel is incomplete")
        if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "final"
                or config.get("args", {}).get("name") != f"E132-A{index}" or config.get("eval_seed") != DEVELOPMENT_SEED+1+index
                or config.get("source_genome") != SOURCES[index] or config.get("expected_families") != 64
                or config.get("apparatus") != APPARATUS or config.get("git", {}).get("dirty") is not False):
            raise ValueError("offer panel metadata differs from its registered lineage")
        selected.append((folder, config))
    path = root / AUDIT_PATH
    if not path.exists():
        return dict(status="undecided", reason="independent joint offer verification is pending")
    raw_audit = path.read_bytes()
    audit = json.loads(raw_audit)
    if audit.get("status") != "passed" or audit.get("protocol") != PROTOCOL or audit.get("mode") != "final":
        return dict(status="undecided", reason="independent final offer verification has not passed")
    registered, checked = audit.get("registered_source", ""), audit.get("checked_file_sha256", {})
    if (not isinstance(registered, str) or len(registered) != 40 or any(c not in "0123456789abcdef" for c in registered)
            or not isinstance(checked, dict) or [(p.get("index"), p.get("run")) for p in audit.get("panels", [])]
            != [(i, folder.relative_to(root).as_posix()) for i, (folder, _) in enumerate(selected)]):
        raise ValueError("offer audit source or exact panel selection is invalid")

    def verified(file, expected=None):
        digest = checked.get(file.relative_to(root).as_posix())
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("joint offer audit does not cover current evidence")
        return _read_pinned(file, digest)

    panels = []
    for index, (folder, config) in enumerate(selected):
        for name in ("config.json", "status.json", "metrics.jsonl"):
            verified(folder / name)
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        verified(folder / "genome_final.npy", SOURCES[index]["sha256"])
        verified(root / SOURCES[index]["path"], SOURCES[index]["sha256"])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 64 or result.get("expected_families") != 64 or len(result.get("raw_files", [])) != 64):
            raise ValueError("an audited offer panel is incomplete")
        for index_in_panel, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{index_in_panel:04d}.npz"
            if artifact.get("index") != index_in_panel or artifact.get("path") != relative:
                raise ValueError("offer evidence must preserve every original family exactly once")
            verified(folder / relative, artifact["sha256"])
        values = result["measurements"]
        if (values.get("families") != 64 or values.get("initial_rule_replays") != 128 or values.get("continuation_lives") != 512
                or values.get("agents_per_rule") != 1 or values.get("apparatus") != APPARATUS):
            raise ValueError("offer comparison changed its assigned denominator or apparatus")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=values))
    gates = _replicated_gates(panels)
    if audit.get("replicated_gates") != gates:
        raise ValueError("runner and independent offer gate disagree")
    return dict(status="complete", protocol=PROTOCOL, scientific_result=True, registered_source=registered,
        audit=dict(path=AUDIT_PATH, sha256=hashlib.sha256(raw_audit).hexdigest()), panels=panels, replicated_gates=gates)
