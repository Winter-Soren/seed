"""E127 immutable matched diagnostics from original E103/E113 saved controllers."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from seed import direct, encounter_diagnostic as engine
from seed.seed import Cfg
from scripts.run_cold_donor import SOURCES, _runtime, _write_bytes, _write_json, _write_status
from scripts.run_shelter_evolution import _save_arrays
from scripts.run_social_reversal import DIRECT_SOURCES


PROTOCOL = "encounter-diagnostic-v1"
MAX_SECONDS = {"development": 180, "final": 3600}
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_frozen", "direct_stationary", "direct_reversal")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe")
AUDIT_PATH = "results/diagnostics/E127-independent-audit.json"
BENCHMARK_RETRY = "E127-benchmark-retry1"
BENCHMARK_RECOVERY = "E127-benchmark-recovery2"
A0_RECOVERY = "E127-A0-recovery1"
FINAL_NAMES = (A0_RECOVERY, "E127-A1", "E127-A2")
RECOVERY_PREDECESSORS = {
    A0_RECOVERY: dict(run="runs/20261122-050231_E127-A0",
        commit="5e6398e5d4f493afe428c3daaebe9bee3247ddfd",
        inventory_sha256="15ff3384e436471a370807d55b07e9a211de8856b8a27e1ced00b10f8b530022"),
    BENCHMARK_RECOVERY: dict(run="runs/20261122-045153_E127-benchmark-retry1",
        commit="d99bcc80957bb97dfc3d7d27e1ed2a80ee3a31a1",
        inventory_sha256="7768d9fd14af85a2fb61b010c572daca7bd05f4b1f34589b1ccc1902972e65fd"),
}
ALLOWED_FLAGS = {"--eval-protocol", "--condition-split", "--name", "--start", "--seed",
                 "--generations", "--eval-worlds", "--hypothesis"}


def _read_pinned(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("saved original controller differs from its registered SHA256")
    return raw


def _array_sha(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8).copy()


def _source_code(root):
    names = ("seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py",
             "scripts/run_encounter_diagnostic.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _file_inventory(folder):
    """Hash preserved evidence without decoding any behavioral arrays."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[path.relative_to(folder).as_posix()] = digest.hexdigest()
    return files


def _inventory_sha(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_assignments(folder, config, schedule):
    assignment = config.get("assignments", {})
    if assignment.get("path") != "assignments.npz":
        raise ValueError("predecessor assignment manifest differs from registration")
    raw = _read_pinned(folder / "assignments.npz", assignment.get("sha256"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as previous:
        if (set(previous.files) != set(schedule)
                or any(previous[name].dtype != value.dtype or not np.array_equal(previous[name], value)
                       for name, value in schedule.items())):
            raise ValueError("recovery must reuse the identical registered assignments")


def _assignments(master, count):
    data = {name: np.asarray(value) for name, value in engine.assignments_manifest(master, count).items()}
    if (set(data) != {"key", "permutation", "pre_contacts"}
            or data["key"].shape != (2,) or data["key"].dtype != np.uint32
            or data["permutation"].shape != (count,) or data["permutation"].dtype != np.int32
            or data["pre_contacts"].shape != (count,) or data["pre_contacts"].dtype != np.int32
            or not np.array_equal(np.sort(data["permutation"]), np.arange(count))
            or not np.array_equal(data["pre_contacts"], np.repeat([8, 16], count // 2)[data["permutation"]])):
        raise ValueError("encounter panel must balance its registered eight/sixteen-contact boundaries")
    return data


def _retry_predecessor(args, root, schedule):
    """Permit one development retry only while its empty failure remains intact."""
    if args.name != BENCHMARK_RETRY:
        return None
    candidates = list((root / "runs").glob("*_E127-benchmark"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise ValueError("benchmark retry requires one retained canonical failed benchmark")
    folder = candidates[0]
    required = {"config.json", "status.json", "summary.json", "metrics.jsonl", "assignments.npz",
                "plastic_genome.npy", "direct_genome.npy"}
    if any(not (folder / name).is_file() for name in required):
        raise ValueError("benchmark retry requires every original failure file to remain present")
    config, status, summary = (json.loads((folder / name).read_text(encoding="utf-8"))
                               for name in ("config.json", "status.json", "summary.json"))
    expected_args = vars(args).copy()
    expected_args.update(name="E127-benchmark", start="file:" + SOURCES[0]["path"],
                         hypothesis=config.get("args", {}).get("hypothesis"))
    expected = dict(protocol=PROTOCOL, condition_split="development", generation=0,
                    eval_seed=2026120600, expected_families=4, source=SOURCES[0], direct_source=DIRECT_SOURCES[0],
                    cfg=Cfg(M=1)._asdict(), cfg_overrides={}, arms=list(ARMS), key_roles=list(KEY_ROLES),
                    initial_rules=[0, 1], pre_contact_choices=[8, 16], post_contacts=16,
                    readout_contacts=[0, 2, 8, 16], grid_steps=50, probe_placements=12,
                    assignment_stream_tag=0xE127, scientific_result=False)
    previous_commit = config.get("git", {}).get("commit", "")
    if (any(config.get(key) != value for key, value in expected.items())
            or config.get("args") != expected_args or config.get("git", {}).get("dirty") is not False
            or not isinstance(previous_commit, str) or len(previous_commit) != 40
            or any(ch not in "0123456789abcdef" for ch in previous_commit)):
        raise ValueError("benchmark retry must preserve the original development settings and source pair")
    if (any(item.get("status") != "failed" or item.get("completed_families") != 0
            or item.get("expected_families") != 4 or item.get("family_seconds") != [] for item in (status, summary))
            or summary.get("raw_files") != [] or "measurements" in summary
            or summary.get("protocol") != PROTOCOL or summary.get("condition_split") != "development"
            or summary.get("evaluation_seed") != 2026120600 or summary.get("source_label") != "A0"
            or summary.get("seed_index") != 0 or summary.get("scientific_result") is not False
            or summary.get("artifact_validity") != "incomplete" or summary.get("gate_status") != "undecided"
            or not isinstance(summary.get("error"), str) or not summary["error"]
            or (folder / "metrics.jsonl").read_bytes() != b""
            or not (folder / "families").is_dir() or any((folder / "families").iterdir())
            or any(folder.rglob("family_*.npz"))):
        raise ValueError("benchmark retry requires a failed zero-family attempt with no saved outcome evidence")
    _verify_assignments(folder, config, schedule)
    _read_pinned(folder / "plastic_genome.npy", SOURCES[0]["sha256"])
    _read_pinned(folder / "direct_genome.npy", DIRECT_SOURCES[0]["sha256"])
    files = {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(folder.rglob("*")) if path.is_file()}
    return dict(run=folder.relative_to(root).as_posix(), files=files)


def _recovery_predecessor(args, root, schedule):
    """Verify the one registered historical panel using metadata and opaque hashes only."""
    if args.name not in RECOVERY_PREDECESSORS:
        return None
    final = args.name == A0_RECOVERY
    old_name, split, seed, expected, completed = (("E127-A0", "final", 2026120601, 128, 11)
        if final else (BENCHMARK_RETRY, "development", 2026120600, 4, 4))
    pin = RECOVERY_PREDECESSORS[args.name]
    candidates = list((root / "runs").glob("*_" + old_name))
    if (len(candidates) != 1 or candidates[0] != root / pin["run"]
            or not candidates[0].is_dir()):
        raise ValueError("recovery requires its unique, exact registered predecessor folder")
    folder = candidates[0]
    files = _file_inventory(folder)
    required = {"config.json", "status.json", "summary.json", "metrics.jsonl", "assignments.npz",
                "plastic_genome.npy", "direct_genome.npy"}
    required.update(f"families/family_{index:04d}.npz" for index in range(completed))
    if set(files) != required or _inventory_sha(files) != pin["inventory_sha256"]:
        raise ValueError("registered recovery predecessor inventory or SHA256 changed")
    config, status, summary = (json.loads((folder / name).read_text(encoding="utf-8"))
                               for name in ("config.json", "status.json", "summary.json"))
    old_args = vars(args).copy()
    old_args.update(name=old_name, start="file:" + SOURCES[0]["path"],
                    hypothesis=config.get("args", {}).get("hypothesis"))
    metadata = dict(protocol=PROTOCOL, condition_split=split, generation=0, eval_seed=seed,
        expected_families=expected, source=SOURCES[0], direct_source=DIRECT_SOURCES[0], cfg=Cfg(M=1)._asdict(),
        cfg_overrides={}, arms=list(ARMS), key_roles=list(KEY_ROLES), initial_rules=[0, 1],
        pre_contact_choices=[8, 16], post_contacts=16, readout_contacts=[0, 2, 8, 16], grid_steps=50,
        probe_placements=12, assignment_stream_tag=0xE127, scientific_result=False)
    if (any(config.get(key) != value for key, value in metadata.items()) or config.get("args") != old_args
            or config.get("git", {}).get("commit") != pin["commit"]
            or config.get("git", {}).get("dirty") is not False):
        raise ValueError("recovery predecessor settings, source commit or original controller pair changed")
    state = "failed" if final else "finished"
    if (any(item.get("status") != state or item.get("completed_families") != completed
            or item.get("expected_families") != expected for item in (status, summary))
            or summary.get("protocol") != PROTOCOL or summary.get("condition_split") != split
            or summary.get("evaluation_seed") != seed or summary.get("source_label") != "A0"
            or summary.get("seed_index") != 0 or summary.get("scientific_result") is not False
            or summary.get("artifact_validity") != ("incomplete" if final else "passed")
            or summary.get("gate_status") != ("undecided" if final else "not_applicable")
            or final and ("measurements" in summary or not isinstance(summary.get("error"), str) or not summary["error"])):
        raise ValueError("recovery predecessor is not the registered failed prefix or completed rehearsal")
    _verify_assignments(folder, config, schedule)
    for name, source in (("plastic_genome.npy", SOURCES[0]), ("direct_genome.npy", DIRECT_SOURCES[0])):
        if files[name] != source["sha256"]:
            raise ValueError("recovery predecessor changed an original source genome")
    journal = [json.loads(line) for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    durations = status.get("family_seconds")
    if (len(journal) != completed or len(summary.get("raw_files", [])) != completed
            or not isinstance(durations, list) or len(durations) != completed
            or summary.get("family_seconds") != durations):
        raise ValueError("predecessor journal or preserved family count differs from registration")
    for index, record in enumerate(journal):
        path = f"families/family_{index:04d}.npz"
        artifact = dict(index=index, path=path, sha256=files[path])
        keys = np.asarray(engine.evaluation_keys(seed, index)).tolist()
        if (record.get("family") != index or record.get("pre_contacts") != int(schedule["pre_contacts"][index])
                or record.get("keys") != keys or record.get("raw_file") != artifact
                or summary["raw_files"][index] != artifact or record.get("family_seconds") != durations[index]):
            raise ValueError("predecessor journal order, assignment or raw file digest changed")
    if not final:
        original_args = SimpleNamespace(**vars(args))
        original_args.name = BENCHMARK_RETRY
        if config.get("retry_of") != _retry_predecessor(original_args, root, schedule):
            raise ValueError("corrected rehearsal must preserve the original failed benchmark chain")
    return dict(run=pin["run"], files=files)


def _validate(args, git, root, explicit):
    if explicit is not None and set(explicit) - ALLOWED_FLAGS:
        raise ValueError("encounter diagnostics accept only the registered source and evaluation flags")
    if (args.eval_protocol != PROTOCOL or args.condition_split not in MAX_SECONDS
            or args.generations != 0 or args.quick or args.allow_dirty or args.resume or args.cfg or args.no_plastic
            or args.controller not in (None, "rules-v0") or args.structural_plasticity not in (None, "off")
            or args.objective is not None or args.memory_condition is not None or getattr(args, "persistence", False)
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.checkpoint_every) != (128, 8, .1, .1, 10)):
        raise ValueError("encounter diagnostics fix original controllers and zero training generations")
    commit = git.get("commit", "")
    if (git.get("dirty") is not False or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("encounter diagnostics require clean committed source provenance")
    if args.condition_split == "development":
        name = args.name if args.name in (BENCHMARK_RETRY, BENCHMARK_RECOVERY) else "E127-benchmark"
        source, expected = SOURCES[0], (name, 2026120600, 4)
    else:
        candidates = [source for source in SOURCES if args.name == "E127-" + source["label"]
                      or args.name == A0_RECOVERY and source["training_seed"] == 0]
        if len(candidates) != 1:
            raise ValueError("final encounter panels must be E127-A0, E127-A1 or E127-A2")
        source = candidates[0]
        expected = (A0_RECOVERY if args.name == A0_RECOVERY else "E127-" + source["label"],
                    2026120601 + source["training_seed"], 128)
    if (args.name, args.seed, args.eval_worlds) != expected:
        raise ValueError("encounter panel name, seed and family count differ from registration")
    if any((root / "runs").glob("*_" + args.name)):
        raise ValueError("this encounter attempt exists; preserve it and preregister any recovery")
    path = (root / source["path"]).resolve()
    if not args.start.startswith("file:") or (root / args.start[5:]).resolve() != path:
        raise ValueError("encounter diagnostics require the matching original E103 genome")
    baseline = DIRECT_SOURCES[source["training_seed"]]
    original = _read_pinned(path, source["sha256"])
    recurrent = _read_pinned(root / baseline["path"], baseline["sha256"])
    plastic_flat, direct_flat = (np.load(io.BytesIO(raw), allow_pickle=False) for raw in (original, recurrent))
    if (plastic_flat.shape != (155,) or direct_flat.shape != (direct.genome_size(Cfg(M=1)),)
            or any(flat.dtype != np.float32 or not np.isfinite(flat).all() for flat in (plastic_flat, direct_flat))):
        raise ValueError("source controllers must retain their finite float32 registered encodings")
    if (engine.PROTOCOL != PROTOCOL or tuple(engine.ARMS) != ARMS or tuple(engine.KEY_ROLES) != KEY_ROLES
            or engine.GRID_STEPS != 50 or tuple(engine.CONTACTS) != (0, 2, 8, 16)):
        raise ValueError("encounter engine differs from its frozen version or panels")
    return dict(source), original, plastic_flat, dict(baseline), recurrent, direct_flat


def run_tracked(args, git, root, *, explicit=None):
    """Save every assigned family; replicated decisions require a separate all-panel audit."""
    started = time.monotonic()
    root = Path(root).resolve()
    source, original, plastic, baseline, recurrent, direct_flat = _validate(args, git, root, explicit)
    c = Cfg(M=1)
    schedule = _assignments(args.seed, args.eval_worlds)
    retry_of = _retry_predecessor(args, root, schedule)
    recovery_of = _recovery_predecessor(args, root, schedule)
    code_source = _source_code(root)
    folder = root / "runs" / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.name)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "families").mkdir()
    assignments = _save_arrays(folder, "assignments.npz", schedule)
    recorded_args = vars(args).copy()
    recorded_args["start"] = "file:" + source["path"]
    config = dict(protocol=PROTOCOL, args=recorded_args, git=dict(git), source=source, direct_source=baseline,
        cfg=c._asdict(), cfg_overrides={}, condition_split=args.condition_split, generation=0,
        eval_seed=args.seed, expected_families=args.eval_worlds, statistical_unit="independent paired base family",
        assignments=assignments, assignment_stream_tag=0xE127, arms=list(ARMS), key_roles=list(KEY_ROLES),
        initial_rules=[0, 1], pre_contact_choices=[8, 16], post_contacts=16, readout_contacts=[0, 2, 8, 16],
        grid_steps=50, probe_placements=12, source_code=code_source,
        raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", runtime=_runtime(),
        max_seconds=MAX_SECONDS[args.condition_split], scientific_result=False)
    if retry_of is not None:
        config["retry_of"] = retry_of
    if recovery_of is not None:
        config["recovery_of"] = recovery_of
    _write_json(folder / "config.json", config)
    _write_bytes(folder / "plastic_genome.npy", original)
    _write_bytes(folder / "direct_genome.npy", recurrent)
    rows, artifacts, durations, wall_durations, used_keys = [], [], [], [], {schedule["key"].tobytes()}
    setup_seconds = time.monotonic() - started

    def status(value):
        if (len(wall_durations) > len(artifacts) or any(not np.isfinite(duration) or duration < 0 for duration in wall_durations)
                or value == "finished" and len(wall_durations) != len(artifacts)):
            raise ValueError("family wall timing must preserve its complete nonnegative prefix")
        return dict(status=value, completed_families=len(artifacts), expected_families=args.eval_worlds,
                    elapsed_seconds=time.monotonic() - started, setup_seconds=setup_seconds,
                    family_seconds=list(durations), family_wall_seconds=list(wall_durations),
                    family_pre_contacts=[int(value) for value in schedule["pre_contacts"][:len(artifacts)]])

    def summary(value):
        return dict(**status(value), protocol=PROTOCOL, condition_split=args.condition_split,
            source_label=source["label"], seed_index=source["training_seed"], evaluation_seed=args.seed,
            scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=list(artifacts))

    def deadline():
        if time.monotonic() - started >= MAX_SECONDS[args.condition_split]:
            raise TimeoutError("encounter diagnostic exceeded its registered invocation allowance")

    _write_status(folder, status("running"))
    try:
        with (folder / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.eval_worlds):
                deadline()
                family_started = time.monotonic()
                keys = np.asarray(engine.evaluation_keys(args.seed, index))
                if keys.dtype != np.uint32 or keys.shape != (6, 2):
                    raise ValueError("diagnostic assignment keys must be an ordered uint32 key table")
                tokens = [key.tobytes() for key in keys]
                if len(set(tokens)) != len(tokens) or used_keys.intersection(tokens):
                    raise ValueError("independent diagnostic families reused a random key")
                used_keys.update(tokens)
                pre_contacts = int(schedule["pre_contacts"][index])
                row = engine.evaluate_world(plastic, direct_flat, keys, index, pre_contacts, c=c)
                projected = engine.summary_input(row)
                if (not np.array_equal(row["keys"], keys) or int(row["base_index"]) != index
                        or int(row["pre_contacts"]) != pre_contacts or not np.array_equal(row["initial_rules"], [0, 1])
                        or not np.array_equal(row["plastic_genome_sha256"], _array_sha(plastic))
                        or not np.array_equal(row["direct_genome_sha256"], _array_sha(direct_flat))):
                    raise ValueError("diagnostic family differs from its registered assignment")
                artifact = dict(index=index, **_save_arrays(folder, f"families/family_{index:04d}.npz", row))
                artifacts.append(artifact)
                rows.append(projected)
                durations.append(time.monotonic() - family_started)
                stream.write(json.dumps(dict(family=index, pre_contacts=pre_contacts, keys=keys.tolist(), raw_file=artifact,
                    family_seconds=durations[-1], elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _write_status(folder, status("running"))
                wall_durations.append(time.monotonic() - family_started)
                print(f"encounter family {index + 1}/{args.eval_worlds} saved", flush=True)
                del row
                deadline()
        measurements = engine.summarize(rows)
        _read_pinned(root / source["path"], source["sha256"])
        _read_pinned(root / baseline["path"], baseline["sha256"])
        if ((folder / "plastic_genome.npy").read_bytes() != original or (folder / "direct_genome.npy").read_bytes() != recurrent
                or not np.array_equal(plastic, np.load(io.BytesIO(original), allow_pickle=False))
                or not np.array_equal(direct_flat, np.load(io.BytesIO(recurrent), allow_pickle=False))):
            raise ValueError("a diagnostic changed its frozen controller")
        if retry_of is not None and _retry_predecessor(args, root, schedule) != retry_of:
            raise ValueError("the original benchmark failure changed during its immutable retry")
        if recovery_of is not None and _recovery_predecessor(args, root, schedule) != recovery_of:
            raise ValueError("the registered predecessor changed during its immutable recovery")
        final = summary("finished")
        final.update(artifact_validity="passed", measurements=measurements,
                     gate_status="not_applicable" if args.condition_split == "development" else "undecided")
        deadline()
        _write_json(folder / "summary.json", final)
    except BaseException as error:
        value = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        if not (folder / "summary.json").exists():
            partial = summary(value)
            partial["error"] = f"{type(error).__name__}: {error}"
            _write_json(folder / "summary.json", partial)
        _write_status(folder, status(value))
        raise
    _write_status(folder, status("finished"))
    print("run folder:", folder)
    return folder


def _replicated_gates(panels):
    decisions = {}
    for architecture in ("plastic", "direct"):
        local, transfer, hebbian = [], [], []
        for panel in panels:
            index, measures = panel["index"], panel["measurements"]
            gate = measures["architectures"][architecture]
            if any(type(gate.get(name)) is not bool for name in ("local_pass", "transfer_pass")):
                raise ValueError("complete panels require defined local and transfer decisions")
            if gate["local_pass"]:
                local.append(index)
                if gate["transfer_pass"]:
                    transfer.append(index)
            if architecture == "plastic":
                dependency = measures["hebbian"].get("hebbian_local_pass")
                if type(dependency) is not bool:
                    raise ValueError("complete plastic panels require a defined Hebbian decision")
                if gate["local_pass"] and dependency:
                    hebbian.append(index)
        selected = dict(local=local, transfer=transfer)
        if architecture == "plastic":
            selected["hebbian_local"] = hebbian
        decisions[architecture] = {name: dict(count=len(indices), seeds=indices, passed=len(indices) >= 2)
                                   for name, indices in selected.items()}
    return decisions


def build_comparison(root):
    """Release replicated decisions only after all three panels and their independent audit."""
    root = Path(root).resolve()
    folders, configs = [], []
    for index, name in enumerate(FINAL_NAMES):
        candidates = list((root / "runs").glob("*_" + name))
        if len(candidates) != 1 or not (candidates[0] / "status.json").exists():
            return dict(status="undecided", reason="all three complete final panels are required")
        folder = candidates[0]
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if (status.get("status") != "finished" or status.get("completed_families") != 128
                or status.get("expected_families") != 128):
            return dict(status="undecided", reason="a registered final panel is incomplete")
        if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "final"
                or config.get("args", {}).get("name") != name
                or config.get("source") != SOURCES[index] or config.get("direct_source") != DIRECT_SOURCES[index]
                or config.get("eval_seed") != 2026120601 + index or config.get("expected_families") != 128
                or config.get("git", {}).get("dirty") is not False or config.get("grid_steps") != 50):
            raise ValueError("final diagnostic metadata differs from its fixed source, seed or protocol")
        folders.append(folder)
        configs.append(config)
    audit_path = root / AUDIT_PATH
    if not audit_path.exists():
        return dict(status="undecided", reason="independent three-panel verification is pending")
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if (audit.get("status") != "passed" or audit.get("protocol") != PROTOCOL or audit.get("mode") != "final"):
        return dict(status="undecided", reason="independent final verification has not passed")
    registered = audit.get("registered_source", "")
    checked = audit.get("checked_file_sha256", {})
    if (not isinstance(registered, str) or len(registered) != 40
            or any(ch not in "0123456789abcdef" for ch in registered) or not isinstance(checked, dict)):
        raise ValueError("independent verification lacks source and artifact provenance")
    selected = [(index, folder.relative_to(root).as_posix()) for index, folder in enumerate(folders)]
    if [(panel.get("index"), panel.get("run")) for panel in audit.get("panels", [])] != selected:
        raise ValueError("independent verification selected different final panels")

    def verified(path, expected=None):
        relative = path.relative_to(root).as_posix()
        digest = checked.get(relative)
        if not isinstance(digest, str) or len(digest) != 64 or expected is not None and digest != expected:
            raise ValueError("independent verification does not cover the current artifact manifest")
        return _read_pinned(path, digest)

    panels, excluded_runs = [], []
    for index, (folder, config) in enumerate(zip(folders, configs)):
        verified(folder / "config.json")
        verified(folder / "metrics.jsonl")
        verified(folder / "assignments.npz", config["assignments"]["sha256"])
        if index == 0:
            args = SimpleNamespace(**config["args"])
            receipt = _recovery_predecessor(args, root, _assignments(args.seed, args.eval_worlds))
            if config.get("recovery_of") != receipt:
                raise ValueError("final A0 recovery does not preserve its registered predecessor")
            for path, digest in receipt["files"].items():
                verified(root / receipt["run"] / path, digest)
            excluded_runs.append(receipt["run"])
        for source in (config["source"], config["direct_source"]):
            verified(root / source["path"], source["sha256"])
        verified(folder / "plastic_genome.npy", SOURCES[index]["sha256"])
        verified(folder / "direct_genome.npy", DIRECT_SOURCES[index]["sha256"])
        result = json.loads(verified(folder / "summary.json"))
        if (result.get("status") != "finished" or result.get("artifact_validity") != "passed"
                or result.get("completed_families") != 128 or result.get("expected_families") != 128
                or len(result.get("raw_files", [])) != 128):
            raise ValueError("an audited diagnostic panel is incomplete")
        for family, artifact in enumerate(result["raw_files"]):
            relative = f"families/family_{family:04d}.npz"
            if artifact.get("index") != family or artifact.get("path") != relative:
                raise ValueError("family evidence must include every assigned index exactly once")
            verified(folder / relative, artifact["sha256"])
        measurements = result["measurements"]
        if measurements.get("families") != 128 or measurements.get("initial_rule_replays") != 256:
            raise ValueError("diagnostic denominators must retain every assigned family and rule replay")
        panels.append(dict(index=index, run=folder.relative_to(root).as_posix(), measurements=measurements))
    decisions = _replicated_gates(panels)
    if audit.get("replicated_gates") != decisions:
        raise ValueError("runner and independent verifier disagree on replicated diagnostic decisions")
    return dict(status="complete", protocol=PROTOCOL, scientific_result=True, registered_source=registered,
                audit=dict(path=AUDIT_PATH, sha256=hashlib.sha256(audit_raw).hexdigest()),
                panels=panels, replicated_gates=decisions, excluded_partial_runs=excluded_runs)
