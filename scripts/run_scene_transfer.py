"""Immutable E135 scene-transfer panels from pinned E103 and E134 genomes."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from scripts.run_cold_donor import ALLOWED_FLAGS, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_replay_selection import SOURCES, SOURCE_PATHS as E134_SOURCE_PATHS, _save, _journal, observer_keys
from seed import scene_transfer as engine, seed


PROTOCOL = "scene-transfer-v1"
MASTER_SEEDS = (2026121500, 2026121501, 2026121502, 2026121503)
KEY_ROLES = ("script", "birth", "chamber", "background", "probe_action", "continuation", "solo_layout", "solo_action")
ARMS, HISTORIES, VIEWS = ("parent", "descendant"), ("stationary", "reversal"), ("masked", "full")
MAX_SECONDS = {"development": 150, "final": 370}
SOURCE_PATHS = E134_SOURCE_PATHS + ("seed/scene_transfer.py", "scripts/run_scene_transfer.py")
AUDIT_PATH = "results/diagnostics/E135-independent-audit.json"
ENDPOINT_SOURCE = "e514d3fea404f2e765d0b065dd26ac407eab2695"
ENDPOINTS = (
    dict(seed_index=0, run="runs/20261122-214433_E134-A0-train",
         config_sha256="818cd175a86e6c814656e5aa0d933d996f568dea8803f2f0dad29bdfa30c8fb7",
         summary_sha256="b0494b82e94c0c9a7a872cc82886491873b0decb95a642712f65ab3657938a4c",
         checkpoint_sha256="a3a333f16010c2531c837d62228f5f0fa90c099a69a3c15188c0f06e9da39dcf",
         endpoint_sha256="936176dfea318a4ec698393e5b693877b8a1bd3dd4406db6d9c76778dfe76378"),
    dict(seed_index=1, run="runs/20261122-214515_E134-A1-train",
         config_sha256="101811f422b2a2772f60b763f951bcba84f94a00f0da98b1ce9ee3263f6fc0e6",
         summary_sha256="3abf0e5aae5e0fd4faa247948662700a5c764711ece9b481611a4fae2743f95a",
         checkpoint_sha256="db450a9064c87ac50581d9bd9fc224cf96446aae08672e93e365217393c57d81",
         endpoint_sha256="ed054c08356b30ea7ccd80f4346f09a4f07cd371729c3d86e7b4a14aa837f37a"),
    dict(seed_index=2, run="runs/20261122-214556_E134-A2-train",
         config_sha256="e30dea71dd52f7c1639b2676751ce6eed477e339152f39bfbbde7df5044f2a8f",
         summary_sha256="0e4faf98c274313598a2ab9d101f0d46ee1846f70873cf9127eab931a2050882",
         checkpoint_sha256="03553b4d9755ebcd694620f7de73186803d7d275a7ab67130cf2d4b3e7990f79",
         endpoint_sha256="bcf5ad20fd1e38c322bb459c3d694f64a7fe8ca018720a47c7942a49d2191e14"),
)
GATE_FIELDS = ("isolated_replicates", "attenuation_confirmed", "passed", "practical_loss_excluded",
               "advantage_erased", "partial_attenuation", "reference_deficit")


def _read_pinned(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"SHA256 mismatch for {path}")
    return content


def _genome(content):
    flat = np.load(io.BytesIO(content), allow_pickle=False)
    if flat.shape != (155,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("E135 requires a finite 155-number float32 native genome")
    return flat


def _array_sha(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8)


def _source_code(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def _endpoint_ref(index):
    return dict(path=ENDPOINTS[index]["run"] + "/genome_final.npy", sha256=ENDPOINTS[index]["endpoint_sha256"])


def _pinned_endpoints(root):
    """Verify all three pre-existing endpoint receipts before deriving any new keys."""
    endpoints = []
    for index, receipt in enumerate(ENDPOINTS):
        folder = root / receipt["run"]
        config = json.loads(_read_pinned(folder / "config.json", receipt["config_sha256"]))
        summary = json.loads(_read_pinned(folder / "summary.json", receipt["summary_sha256"]))
        checkpoint_bytes = _read_pinned(folder / "generations/attempt_0010.npz", receipt["checkpoint_sha256"])
        endpoint_bytes = _read_pinned(folder / "genome_final.npy", receipt["endpoint_sha256"])
        _genome(_read_pinned(root / SOURCES[index]["path"], SOURCES[index]["sha256"]))
        endpoint = _genome(endpoint_bytes)
        if (receipt["seed_index"] != index or config.get("protocol") != "replay-selection-v1"
                or config.get("mode") != "training" or config.get("seed_index") != index
                or config.get("source") != SOURCES[index] or config.get("expected_attempts") != 10
                or config.get("git", {}).get("commit") != ENDPOINT_SOURCE or config.get("git", {}).get("dirty") is not False
                or summary.get("status") != "finished" or summary.get("completed_attempts") != 10
                or summary.get("completed_families") != 0 or summary.get("endpoint_sha256") != receipt["endpoint_sha256"]):
            raise ValueError("E135 requires the complete, pinned attempt-ten E134 endpoints")
        with np.load(io.BytesIO(checkpoint_bytes), allow_pickle=False) as checkpoint:
            if (checkpoint["mu"].dtype != np.float32 or checkpoint["mu"].shape != (155,)
                    or checkpoint["mu"].tobytes() != endpoint.tobytes()
                    or int(checkpoint["t"]) != summary.get("optimizer_updates")):
                raise ValueError("E135 endpoint differs from the final registered E134 mean")
        endpoints.append(endpoint_bytes)
    return endpoints


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("scene transfer accepts only registered identity and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS or args.generations != 0
            or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("scene transfer fixes the native controller, apparatus and zero training")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("scene transfer requires clean committed source")
    requests = {"E135-dev": ("development", 0, MASTER_SEEDS[0], 4)}
    requests.update({f"E135-A{i}": ("final", i, MASTER_SEEDS[i + 1], 128) for i in range(3)})
    if args.name not in requests:
        raise ValueError("unknown scene-transfer attempt; retries require a new registration")
    mode, index, master, count = requests[args.name]
    if ((args.condition_split, args.seed, args.eval_worlds) != (mode, master, count)
            or any((root / "runs").glob("*_" + args.name))):
        raise ValueError("scene-transfer identity differs from registration or its immutable attempt exists")
    source = SOURCES[index]
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != (root / source["path"]).resolve():
        raise ValueError("scene transfer requires the matching original E103 parent")
    parent_bytes = _read_pinned(root / source["path"], source["sha256"])
    parent = _genome(parent_bytes)
    if (engine.PROTOCOL != PROTOCOL or tuple(engine.KEY_ROLES) != KEY_ROLES or tuple(engine.MASTER_SEEDS) != MASTER_SEEDS
            or tuple(engine.ARMS) != ARMS or tuple(engine.HISTORIES) != HISTORIES or tuple(engine.VIEWS) != VIEWS
            or engine.ASSIGNMENT_TAG != 0xE135 or engine.CFG != seed.Cfg(M=1)):
        raise ValueError("scene-transfer backend differs from the registered apparatus")
    return mode, index, dict(source), parent_bytes, parent


def evaluation_assignments(master, count):
    if count not in (4, 128) or (master == MASTER_SEEDS[0]) != (count == 4) or master not in MASTER_SEEDS:
        raise ValueError("E135 assignments require the fixed development or final panel")
    schedules = np.asarray([engine.schedule(i) for i in range(count)], np.int32)
    expected = np.tile(np.asarray([(8, 16), (16, 32), (8, 32), (16, 16)], np.int32), (count // 4, 1))
    keys = np.asarray([engine.family_keys(master, i) for i in range(count)])
    if schedules.shape != (count, 2) or not np.array_equal(schedules, expected) or keys.shape != (count, 8, 2):
        raise ValueError("E135 assignments differ from the fixed cold/warm schedule or role count")
    observer_keys(set(), keys, "E135 family role")
    return dict(family_keys=keys, pre_contacts=schedules[:, 0], gap=schedules[:, 1], initial_rules=np.arange(2, dtype=np.int32))


def run_tracked(args, git, root, *, explicit=None):
    started, root = time.monotonic(), Path(root).resolve()
    mode, index, source, parent_bytes, parent = _validate(args, git, root, explicit)
    endpoints = _pinned_endpoints(root)
    endpoint_bytes = endpoints[index]
    descendant = _genome(endpoint_bytes)
    source_code = _source_code(root)
    assignments = evaluation_assignments(args.seed, args.eval_worlds)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    manifest = _save(folder, "assignments.npz", assignments)
    config = dict(protocol=PROTOCOL, mode=mode, args=dict(vars(args), start="file:" + source["path"]), git=dict(git),
        source=source, endpoint=_endpoint_ref(index), endpoint_source=ENDPOINT_SOURCE,
        train_receipts=[dict(receipt) for receipt in ENDPOINTS], seed_index=index, evaluation_seed=args.seed,
        expected_families=args.eval_worlds, source_code=source_code, cfg=engine.CFG._asdict(), assignments=manifest,
        key_roles=list(KEY_ROLES), arms=list(ARMS), histories=list(HISTORIES), views=list(VIEWS),
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
        cold_families=[0, 1], warm_families=list(range(2, args.eval_worlds)),
        max_seconds=MAX_SECONDS[mode], scientific_result=False, runtime=_runtime())
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "genome_start.npy", parent_bytes)
    _write_bytes(folder / "genome_final.npy", endpoint_bytes)
    setup = time.monotonic() - started
    rows, raw_files, family_seconds, family_walls = [], [], [], []

    def status(value):
        return dict(status=value, protocol=PROTOCOL, mode=mode, seed_index=index,
            elapsed_seconds=time.monotonic() - started, setup_seconds=setup,
            completed_families=len(raw_files), expected_families=args.eval_worlds,
            family_seconds=list(family_seconds), family_wall_seconds=list(family_walls))

    def summary(value):
        return dict(**status(value), scientific_result=False, artifact_validity="incomplete",
                    gate_status="undecided", raw_files=list(raw_files))

    def deadline():
        if time.monotonic() - started >= MAX_SECONDS[mode]:
            raise TimeoutError("scene transfer exceeded its registered process allowance")

    _write_status(folder, status("running"))
    hashes = np.stack((_array_sha(parent), _array_sha(descendant)))
    try:
        with (folder / "evaluation.jsonl").open("x", encoding="utf-8") as stream:
            for family, keys in enumerate(assignments["family_keys"]):
                deadline()
                since = time.monotonic()
                pre, gap = (int(assignments[name][family]) for name in ("pre_contacts", "gap"))
                if not np.array_equal(keys, engine.family_keys(args.seed, family)):
                    raise ValueError("scene-transfer keys differ from the committed assignment")
                raw = engine.build_family(keys, family, parent, descendant)
                if (int(raw["base_index"]) != family or not np.array_equal(raw["keys"], keys)
                        or (int(raw["pre_contacts"]), int(raw["gap"])) != (pre, gap)
                        or not np.array_equal(raw["initial_rules"], [0, 1]) or not np.array_equal(raw["genome_sha256"], hashes)):
                    raise ValueError("raw scene-transfer evidence differs from its assigned family or genomes")
                compact = engine.summary_input(raw)
                raw_files.append(dict(index=family, **_save(folder, f"families/family_{family:04d}.npz", raw)))
                rows.append(compact)
                family_seconds.append(time.monotonic() - since)
                _journal(stream, dict(index=family, keys=keys.tolist(), pre_contacts=pre, gap=gap, raw_file=raw_files[-1],
                                     family_seconds=family_seconds[-1], elapsed_seconds=time.monotonic() - started))
                _write_status(folder, status("running"))
                family_walls.append(time.monotonic() - since)
                print(f"E135 family {family + 1}/{args.eval_worlds} saved", flush=True)
                del raw
                deadline()
        measurements = engine.summarize_rows(rows)
        decision = engine.gate(measurements)
        if set(decision) != set(GATE_FIELDS) or any(type(value) is not bool for value in decision.values()):
            raise ValueError("scene-transfer classification differs from the frozen decision schema")
        if (measurements.get("families") != args.eval_worlds or len(family_walls) != args.eval_worlds
                or not np.isfinite(family_walls).all() or any(wall < inner for wall, inner in zip(family_walls, family_seconds))):
            raise ValueError("scene-transfer completion must retain every assigned family and full wall timing")
        for path, expected in ((root / source["path"], parent_bytes), (root / _endpoint_ref(index)["path"], endpoint_bytes),
                               (folder / "genome_start.npy", parent_bytes), (folder / "genome_final.npy", endpoint_bytes)):
            if path.read_bytes() != expected:
                raise ValueError("scene transfer changed a saved parent or endpoint")
        if parent.tobytes() != _genome(parent_bytes).tobytes() or descendant.tobytes() != _genome(endpoint_bytes).tobytes():
            raise ValueError("scene transfer mutated a genome used by later families")
        result = summary("finished")
        result.update(artifact_validity="passed", measurements=measurements, gate=decision,
                      gate_status="not_applicable" if mode == "development" else "undecided")
        deadline()
        _write_json(folder / "summary.json", result)
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        if not (folder / "summary.json").exists():
            _write_json(folder / "summary.json", dict(summary(value), error=f"{type(error).__name__}: {error}"))
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder, flush=True)
    return folder


def _same_measurements(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_measurements(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(_same_measurements(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return isinstance(right, (float, int)) and not isinstance(right, bool) and bool(np.isclose(left, right, rtol=1e-10, atol=1e-9))
    return type(left) is type(right) and left == right


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("scene transfer needs all three original lineages separately")
    if any(set(panel["gate"]) != set(GATE_FIELDS) or any(type(value) is not bool for value in panel["gate"].values()) for panel in panels):
        raise ValueError("scene-transfer classifications differ from the registered schema")
    return {field: dict(count=sum(panel["gate"][field] for panel in panels),
                        seeds=[panel["index"] for panel in panels if panel["gate"][field]],
                        passed=all(panel["gate"][field] for panel in panels)) for field in GATE_FIELDS}


def build_comparison(root):
    root, selected = Path(root).resolve(), []
    for index in range(3):
        folders = list((root / "runs").glob(f"*_E135-A{index}"))
        if len(folders) != 1 or not (folders[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete scene-transfer panels are required")
        folder = folders[0]
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if state.get("status") != "finished" or state.get("completed_families") != 128 or state.get("expected_families") != 128:
            return dict(status="undecided", reason="a registered scene-transfer panel is incomplete")
        if (config.get("protocol") != PROTOCOL or config.get("mode") != "final"
                or config.get("args", {}).get("name") != f"E135-A{index}" or config.get("evaluation_seed") != MASTER_SEEDS[index + 1]
                or config.get("source") != SOURCES[index] or config.get("endpoint") != _endpoint_ref(index)
                or config.get("endpoint_source") != ENDPOINT_SOURCE or config.get("train_receipts") != list(ENDPOINTS)
                or config.get("expected_families") != 128 or config.get("git", {}).get("dirty") is not False):
            raise ValueError("scene-transfer metadata differs from the pinned lineage")
        selected.append((folder, config))
    path = root / AUDIT_PATH
    if not path.exists():
        return dict(status="undecided", reason="independent joint scene-transfer verification is pending")
    audit_bytes = path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "passed" or audit.get("protocol") != PROTOCOL or audit.get("mode") != "final":
        return dict(status="undecided", reason="independent scene-transfer verification has not passed")
    registered, checked = audit.get("registered_source", ""), audit.get("checked_file_sha256", {})
    admission, registered_code = audit.get("admission_commit", ""), audit.get("registered_source_code", {})
    if (not isinstance(registered, str) or len(registered) != 40 or any(ch not in "0123456789abcdef" for ch in registered)
            or not isinstance(admission, str) or len(admission) != 40 or any(ch not in "0123456789abcdef" for ch in admission)
            or not isinstance(registered_code, dict) or set(registered_code) != set(SOURCE_PATHS)
            or any(not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
                   for digest in registered_code.values())
            or not isinstance(checked, dict) or [(p.get("index"), p.get("run")) for p in audit.get("panels", [])]
            != [(i, folder.relative_to(root).as_posix()) for i, (folder, _) in enumerate(selected)]):
        raise ValueError("scene-transfer audit source or panel selection is invalid")

    def verified(file, expected=None):
        digest = checked.get(file.relative_to(root).as_posix())
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("scene-transfer audit does not cover current evidence")
        return _read_pinned(file, digest)

    panels = []
    for index, (folder, config) in enumerate(selected):
        if config["git"]["commit"] != admission or config.get("source_code") != registered_code:
            raise ValueError("scene-transfer panel differs from the audited admission commit or scientific source")
        for name in ("config.json", "status.json", "evaluation.jsonl"):
            verified(folder / name)
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        verified(folder / "genome_start.npy", SOURCES[index]["sha256"])
        verified(folder / "genome_final.npy", ENDPOINTS[index]["endpoint_sha256"])
        verified(root / SOURCES[index]["path"], SOURCES[index]["sha256"])
        for name, field in (("config.json", "config_sha256"), ("summary.json", "summary_sha256"),
                            ("generations/attempt_0010.npz", "checkpoint_sha256"), ("genome_final.npy", "endpoint_sha256")):
            verified(root / ENDPOINTS[index]["run"] / name, ENDPOINTS[index][field])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 128 or result.get("expected_families") != 128 or len(result.get("raw_files", [])) != 128):
            raise ValueError("an audited scene-transfer panel is incomplete")
        for family, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{family:04d}.npz"
            if artifact.get("index") != family or artifact.get("path") != relative:
                raise ValueError("scene-transfer evidence must preserve every family exactly once")
            content = verified(folder / relative, artifact["sha256"])
            if artifact.get("bytes") != len(content):
                raise ValueError("scene-transfer raw byte count differs from its archived evidence")
        values, decision = result["measurements"], result["gate"]
        independent = audit["panels"][index]
        if (values.get("families") != 128 or not _same_measurements(values, independent.get("measurements"))
                or not _same_measurements(decision, independent.get("gate"))):
            raise ValueError("scene-transfer measurements or decisions differ from independent reconstruction")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=values, gate=decision))
    gates = _replicated_gates(panels)
    if audit.get("replicated_gates") != gates:
        raise ValueError("scene-transfer replication differs from independent reconstruction")
    return dict(status="complete", protocol=PROTOCOL, scientific_result=True, registered_source=registered, admission_commit=admission,
                audit=dict(path=AUDIT_PATH, sha256=hashlib.sha256(audit_bytes).hexdigest()), panels=panels, replicated_gates=gates)
