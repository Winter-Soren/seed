"""E130 immutable founder/descendant comparison in freely encountered reversal worlds."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from seed import founder_reversal as engine
from seed.seed import Cfg
from scripts.run_cold_donor import SOURCES as FINAL_SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_encounter_diagnostic import _array_sha, _read_pinned
from scripts.run_retention_diagnostic import _estimate, _advantage
from scripts.run_shelter_evolution import _save_arrays


PROTOCOL = "founder-reversal-v1"
STUDY_ID = "E130"
DEVELOPMENT_SEED = 2026120900
ROLLOUTS_FIELD = "genotype_history_rollouts"
MAX_SECONDS = {"development": 180, "final": 450}
ARMS = ("gen0_stationary", "gen0_reversal", "gen200_stationary", "gen200_reversal")
KEY_ROLES = ("world_action", "development")
BIN_EDGES = (0, 50, 100, 150, 200, 250, 300)
AUDIT_PATH = "results/diagnostics/E130-independent-audit.json"
START_SOURCES = tuple(dict(source, path=source["path"].replace("genome_final.npy", "genome_start.npy"), sha256=digest)
    for source, digest in zip(FINAL_SOURCES, (
        "621e5a735e408e487653f7ffc15561be773fbd1053af847ef1bb633fd5243e07",
        "3ec510cab0205915614a7df2d3b217eca201972897842592fcef419b215e5d99",
        "c62051a478c75f3211645968e5a15bb64bcfcdb8f61c366357020f4cff8294b9")))
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}
SOURCE_SLOTS = (("genome_start.npy", "source_start", "gen0_genome_sha256", START_SOURCES),
                ("genome_final.npy", "source_final", "gen200_genome_sha256", FINAL_SOURCES))
CONFIG_EXTRA = {"source_generations": [0, 200]}


def _gate(components, safeguards):
    checks = dict(descendant_pre_positive=_advantage(components["descendant_pre"]),
        descendant_late_positive=_advantage(components["descendant_reversal_late"]),
        descendant_recovers=_advantage(components["descendant_recovery"], .05),
        descendant_beats_founder=_advantage(components["descendant_vs_founder_late"], .05),
        recovery_interaction=_advantage(components["recovery_interaction"], .05),
        pre_noninferiority=safeguards["pre_Q"]["lower_2se"] > -.05,
        stationary_late_noninferiority=safeguards["stationary_late_Q"]["lower_2se"] > -.05,
        reversal_survival_noninferiority=safeguards["reversal_end_survival"]["lower_2se"] > -.05,
        stationary_survival_noninferiority=safeguards["stationary_end_survival"]["lower_2se"] > -.05)
    return dict(criteria=checks, passed=all(checks.values()), failed_criteria=[name for name, passed in checks.items() if not passed])


def _family_arrays(rows, arms):
    n = len(rows)
    if n < 2 or [int(row["base_index"]) for row in rows] != list(range(n)):
        raise ValueError("founder comparison requires every ordered independent family")
    arrays = {}
    for arm in arms:
        q = np.asarray([row[arm + "_Q_by_rule"] for row in rows])
        alive = np.asarray([row[arm + "_alive_end_by_rule"] for row in rows])
        survival = np.asarray([row[arm + "_final_survival_by_rule"] for row in rows])
        absent = np.asarray([row[arm + "_no_contact_by_rule"] for row in rows])
        if (q.shape != (n, 2, 6, 4) or not np.isin(q, [-1, 0, 1]).all()
                or alive.shape != q.shape or not np.isin(alive, [0, 1]).all()
                or absent.shape != q.shape or not np.array_equal(absent, q == 0)
                or survival.shape != (n, 2, 4) or not np.array_equal(survival, alive[:, :, -1])):
            raise ValueError("founder outcomes must retain both rules and all four assigned animals")
        arrays[arm] = dict(Q=q.astype(np.float64), first_food=(q == 1).astype(np.float64),
            first_poison=(q == -1).astype(np.float64), no_contact=(q == 0).astype(np.float64),
            survival=alive.astype(np.float64), end_survival=survival.astype(np.float64))
    for generation in {arm.rsplit("_", 1)[0] for arm in arms}:
        for field in ("Q", "survival"):
            if not np.array_equal(arrays[generation + "_stationary"][field][:, :, :3],
                                  arrays[generation + "_reversal"][field][:, :, :3]):
                raise ValueError("stationary and reversal histories must agree before the switch")

    return arrays


def _arm_summaries(arrays, rule=None):
    family, arms = {}, {}
    for arm, fields in arrays.items():
        family[arm] = {name: (value if rule is None else value[:, rule:rule+1]).mean(axis=(1, value.ndim-1))
                       for name, value in fields.items()}
        arms[arm] = dict(windows=[dict(index=i, start=BIN_EDGES[i], end=BIN_EDGES[i+1],
            **{name: _estimate(family[arm][name][:, i]) for name in ("Q", "first_food", "first_poison", "no_contact", "survival")})
            for i in range(6)], end_survival=_estimate(family[arm]["end_survival"]))
    return family, arms


def summarize(rows):
    arrays, n = _family_arrays(rows, ARMS), len(rows)

    def measure(rule=None):
        family, arms = _arm_summaries(arrays, rule)
        f_rev, f_stat = family["gen0_reversal"]["Q"], family["gen0_stationary"]["Q"]
        d_rev, d_stat = family["gen200_reversal"]["Q"], family["gen200_stationary"]["Q"]
        components = dict(descendant_pre=_estimate(d_rev[:, 2]), descendant_reversal_late=_estimate(d_rev[:, 5]),
            descendant_recovery=_estimate(d_rev[:, 5]-d_rev[:, 3]),
            descendant_vs_founder_late=_estimate(d_rev[:, 5]-f_rev[:, 5]),
            recovery_interaction=_estimate(((d_rev[:, 5]-d_rev[:, 3])-(d_stat[:, 5]-d_stat[:, 3]))
                                         - ((f_rev[:, 5]-f_rev[:, 3])-(f_stat[:, 5]-f_stat[:, 3]))))
        safeguards = dict(pre_Q=_estimate(d_rev[:, 2]-f_rev[:, 2], bound=True),
            stationary_late_Q=_estimate(d_stat[:, 5]-f_stat[:, 5], bound=True),
            reversal_end_survival=_estimate(family["gen200_reversal"]["end_survival"]-family["gen0_reversal"]["end_survival"], bound=True),
            stationary_end_survival=_estimate(family["gen200_stationary"]["end_survival"]-family["gen0_stationary"]["end_survival"], bound=True))
        return dict(arms=arms, components=components, safeguards=safeguards)

    pooled = measure()
    return dict(families=n, initial_rule_replays=2*n, genotype_history_rollouts=8*n, agents_per_rule=4,
        bin_edges=list(BIN_EDGES), flip_step=150, **pooled, gate=_gate(pooled["components"], pooled["safeguards"]),
        by_initial_rule={str(rule): measure(rule) for rule in (0, 1)})


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("evolutionary recovery requires three separate registered lineages")
    qualifying = []
    for panel in panels:
        measures = panel["measurements"]
        expected = _gate(measures["components"], measures["safeguards"])
        if measures["gate"] != expected or type(measures["gate"]["passed"]) is not bool:
            raise ValueError("founder comparison gate disagrees with its recorded components")
        if expected["passed"]:
            qualifying.append(panel["index"])
    return dict(evolutionary_recovery=dict(count=len(qualifying), seeds=qualifying, passed=len(qualifying) == 3))


def _source_code(root):
    names = ("seed/seed.py", "seed/founder_reversal.py", "scripts/run_founder_reversal.py",
             "scripts/run_retention_diagnostic.py", "scripts/run_encounter_diagnostic.py",
             "scripts/run_cold_donor.py", "scripts/run_shelter_evolution.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _validate(args, git, root, explicit, *, study=None):
    study = sys.modules[__name__] if study is None else study
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("founder comparison accepts only its registered identity and source flags")
    if (args.eval_protocol != study.PROTOCOL or args.condition_split not in study.MAX_SECONDS or args.generations != 0
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("founder comparison fixes original controllers, ordinary plasticity and zero new training")
    commit = git.get("commit", "")
    if git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("founder comparison requires clean committed code")
    indices = [i for i in range(3) if args.name == f"{study.STUDY_ID}-A{i}"]
    if args.condition_split == "development":
        index, expected = 0, (study.STUDY_ID + "-benchmark", study.DEVELOPMENT_SEED, 4)
    elif len(indices) == 1:
        index = indices[0]
        expected = (f"{study.STUDY_ID}-A{index}", study.DEVELOPMENT_SEED+1+index, 64)
    else:
        raise ValueError("final panels must match one of the three registered lineage names")
    if (args.name, args.seed, args.eval_worlds) != expected or any((root / "runs").glob("*_" + args.name)):
        raise ValueError("founder identity differs from registration or its immutable attempt already exists")
    source = study.SOURCE_SLOTS[0][3][index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("comparison requires its matching original E103 source file")
    originals = tuple(_read_pinned(root / sources[index]["path"], sources[index]["sha256"])
                      for _, _, _, sources in study.SOURCE_SLOTS)
    values = tuple(np.load(io.BytesIO(raw), allow_pickle=False) for raw in originals)
    if any(value.shape != (155,) or value.dtype != np.float32 or not np.isfinite(value).all() for value in values):
        raise ValueError("original genomes must be finite float32 rules-v0 arrays")
    backend = study.engine
    if (backend.PROTOCOL != study.PROTOCOL or tuple(backend.ARMS) != study.ARMS or tuple(backend.KEY_ROLES) != study.KEY_ROLES
            or tuple(backend.BIN_EDGES) != BIN_EDGES or backend.FLIP_STEP != 150 or backend.CFG != Cfg()):
        raise ValueError("founder engine differs from its registered ordinary world")
    return index, originals, values


def run_tracked(args, git, root, *, explicit=None, study=None):
    study = sys.modules[__name__] if study is None else study
    started, root = time.monotonic(), Path(root).resolve()
    index, originals, values = _validate(args, git, root, explicit, study=study)
    source, backend = study.SOURCE_SLOTS[0][3][index], study.engine
    n = args.eval_worlds
    assignment = {key: np.asarray(value) for key, value in backend.assignments_manifest(args.seed, n).items()}
    if (set(assignment) != {"family_keys", "initial_rules"} or assignment["family_keys"].shape != (n, len(study.KEY_ROLES), 2)
            or assignment["family_keys"].dtype != np.uint32 or assignment["initial_rules"].dtype != np.int32
            or not np.array_equal(assignment["initial_rules"], [0, 1])
            or len({key.tobytes() for key in assignment["family_keys"].reshape(-1, 2)}) != len(study.KEY_ROLES)*n):
        raise ValueError("founder families need unique paired world/development keys and both rules")
    code = study._source_code(root)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    manifest = _save_arrays(folder, "assignments.npz", assignment)
    config = dict(protocol=study.PROTOCOL, args=dict(vars(args), start="file:" + source["path"]), git=dict(git),
        **{field: sources[index] for _, field, _, sources in study.SOURCE_SLOTS},
        **study.CONFIG_EXTRA, cfg=Cfg()._asdict(), cfg_overrides={},
        condition_split=args.condition_split, generation=0, seed_index=index, eval_seed=args.seed, expected_families=n,
        statistical_unit="independent family averaging four original animals and both rules", assignments=manifest,
        arms=list(study.ARMS), key_roles=list(study.KEY_ROLES), initial_rules=[0, 1], bin_edges=list(BIN_EDGES), flip_step=150,
        source_code=code, raw_schema=study.PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        max_seconds=study.MAX_SECONDS[args.condition_split], scientific_result=False)
    _write_json(folder / "config.json", config)
    for (name, _, _, _), content in zip(study.SOURCE_SLOTS, originals):
        _write_bytes(folder / name, content)
    rows, artifacts, durations, walls = [], [], [], []
    policy_durations = []
    setup = time.monotonic()-started

    def status(value):
        if (len(walls) > len(artifacts) or any(not np.isfinite(t) or t < 0 for t in walls)
                or value == "finished" and len(walls) != n):
            raise ValueError("founder timing must preserve a complete nonnegative family prefix")
        result = dict(status=value, completed_families=len(artifacts), expected_families=n, elapsed_seconds=time.monotonic()-started,
                      setup_seconds=setup, family_seconds=list(durations), family_wall_seconds=list(walls))
        if hasattr(study, "POLICIES"):
            result.update(policy_order=list(study.POLICIES), family_policy_seconds=list(policy_durations))
        return result

    def summary(value):
        return dict(**status(value), protocol=study.PROTOCOL, condition_split=args.condition_split, seed_index=index,
            evaluation_seed=args.seed, scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=list(artifacts))

    def deadline():
        if time.monotonic()-started >= study.MAX_SECONDS[args.condition_split]:
            raise TimeoutError("founder comparison exceeded its registered invocation allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for family, keys in enumerate(assignment["family_keys"]):
                deadline()
                family_started = time.monotonic()
                if not np.array_equal(keys, np.asarray(backend.evaluation_keys(args.seed, family))):
                    raise ValueError("founder assignment differs from the registered key derivation")
                raw = backend.evaluate_world(*values, keys, family, c=backend.CFG)
                compact = backend.summary_input(raw)
                if (int(raw["base_index"]) != family or not np.array_equal(raw["keys"], keys)
                        or not np.array_equal(raw["initial_rules"], [0, 1]) or not np.array_equal(raw["bin_edges"], BIN_EDGES)
                        or int(raw["flip_step"]) != 150
                        or any(not np.array_equal(raw[field], _array_sha(value))
                               for (_, _, field, _), value in zip(study.SOURCE_SLOTS, values))):
                    raise ValueError("founder raw evidence differs from its assigned inputs")
                if hasattr(study, "POLICIES"):
                    timing = np.asarray(raw["policy_seconds"], np.float64)
                    if timing.shape != (len(study.POLICIES),) or not np.isfinite(timing).all() or np.any(timing <= 0):
                        raise ValueError("policy timing must retain every synchronized positive observation")
                artifact = dict(index=family, **_save_arrays(folder, f"families/family_{family:04d}.npz", raw))
                artifacts.append(artifact)
                if hasattr(study, "POLICIES"):
                    policy_durations.append(timing.tolist())
                rows.append(compact)
                durations.append(time.monotonic()-family_started)
                metric = dict(family=family, keys=keys.tolist(), raw_file=artifact,
                    family_seconds=durations[-1], elapsed_seconds=time.monotonic()-started)
                if hasattr(study, "POLICIES"):
                    metric["policy_seconds"] = timing.tolist()
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                walls.append(time.monotonic()-family_started)
                print(f"founder family {family+1}/{n} saved", flush=True)
                del raw
                deadline()
        measurements = study.summarize(rows)
        for (name, _, _, sources), content, value in zip(study.SOURCE_SLOTS, originals, values):
            item = sources[index]
            _read_pinned(root / item["path"], item["sha256"])
            if (folder / name).read_bytes() != content or value.tobytes() != np.load(io.BytesIO(content), allow_pickle=False).tobytes():
                raise ValueError("founder evaluation changed a source endpoint")
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


def build_comparison(root, *, study=None):
    study = sys.modules[__name__] if study is None else study
    root = Path(root).resolve()
    folders, configs = [], []
    for index in range(3):
        candidates = list((root / "runs").glob(f"*_{study.STUDY_ID}-A{index}"))
        if len(candidates) != 1 or not (candidates[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete founder panels are required")
        folder = candidates[0]
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if state.get("status") != "finished" or state.get("completed_families") != 64 or state.get("expected_families") != 64:
            return dict(status="undecided", reason="a registered founder panel is incomplete")
        if (config.get("protocol") != study.PROTOCOL or config.get("condition_split") != "final"
                or config.get("args", {}).get("name") != f"{study.STUDY_ID}-A{index}" or config.get("eval_seed") != study.DEVELOPMENT_SEED+1+index
                or any(config.get(field) != sources[index] for _, field, _, sources in study.SOURCE_SLOTS)
                or config.get("expected_families") != 64 or config.get("git", {}).get("dirty") is not False):
            raise ValueError("founder panel metadata differs from the registered lineage comparison")
        folders.append(folder)
        configs.append(config)
    path = root / study.AUDIT_PATH
    if not path.exists():
        return dict(status="undecided", reason="independent three-panel founder verification is pending")
    raw_audit = path.read_bytes()
    audit = json.loads(raw_audit)
    if audit.get("status") != "passed" or audit.get("protocol") != study.PROTOCOL or audit.get("mode") != "final":
        return dict(status="undecided", reason="independent final founder verification has not passed")
    registered, checked = audit.get("registered_source", ""), audit.get("checked_file_sha256", {})
    if (not isinstance(registered, str) or len(registered) != 40 or any(ch not in "0123456789abcdef" for ch in registered)
            or not isinstance(checked, dict) or [(p.get("index"), p.get("run")) for p in audit.get("panels", [])]
            != [(i, folder.relative_to(root).as_posix()) for i, folder in enumerate(folders)]):
        raise ValueError("founder audit source or panel selection is invalid")

    def verified(path, expected=None):
        digest = checked.get(path.relative_to(root).as_posix())
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("founder audit does not cover the current evidence")
        return _read_pinned(path, digest)

    panels = []
    for index, (folder, config) in enumerate(zip(folders, configs)):
        for name in ("config.json", "status.json", "metrics.jsonl"):
            verified(folder / name)
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        for name, _, _, sources in study.SOURCE_SLOTS:
            source = sources[index]
            verified(folder / name, source["sha256"])
            verified(root / source["path"], source["sha256"])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 64 or result.get("expected_families") != 64
                or len(result.get("raw_files", [])) != 64):
            raise ValueError("an audited founder panel is incomplete")
        for family, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{family:04d}.npz"
            if artifact.get("index") != family or artifact.get("path") != relative:
                raise ValueError("founder evidence must contain every assigned family exactly once")
            verified(folder / relative, artifact["sha256"])
        measures = result["measurements"]
        if (measures.get("families") != 64 or measures.get("initial_rule_replays") != 128
                or measures.get(study.ROLLOUTS_FIELD) != 64*len(study.ARMS)*2 or measures.get("agents_per_rule") != 4
                or measures.get("bin_edges") != list(BIN_EDGES) or measures.get("flip_step") != 150):
            raise ValueError("founder comparison changed its assigned denominators or windows")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=measures))
    decisions = study._replicated_gates(panels)
    if audit.get("replicated_gates") != decisions:
        raise ValueError("runner and independent founder audit disagree")
    extra = study._comparison_extra(root, audit, verified, decisions) if hasattr(study, "_comparison_extra") else {}
    return dict(status="complete", protocol=study.PROTOCOL, scientific_result=True, registered_source=registered,
        audit=dict(path=study.AUDIT_PATH, sha256=hashlib.sha256(raw_audit).hexdigest()), panels=panels, replicated_gates=decisions, **extra)
