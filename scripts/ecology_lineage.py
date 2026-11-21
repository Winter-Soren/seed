"""Tracked, development-only entry for three fresh ecology/lineage cohorts.

CLI: run_experiment.py --eval-protocol ecology-lineage-v1 --quick
     --condition-split development --seed 41 --generations 0
     --name ecology-fixture --memory-condition shared --start blank

This engineering fixture fixes its schedule before execution. Advancement is not
a competence decision. Both shared and empty-memory traces are retained; selecting
a memory condition chooses which observations are highlighted. No final science
protocol, evolution, or automatic retry is enabled here.
"""
from dataclasses import asdict
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed.ecology import EcologyV1
from seed.persistence import PersistenceV1
from seed import lineage
from seed.seed import Cfg, init_genome

PROTOCOL = "ecology-lineage-v1"
DEVELOPMENT_SEED = 41
DEVELOPMENT_CFG = Cfg(G=6, M=2, T=12, R=1, H=3, n_items=4)
MAX_SECONDS = 120
MEMORY_CONDITIONS = {"shared": "inherited", "empty": "fresh"}
PHASES = ("base", "costly_movement", "cold_with_shelter")
ECOLOGIES = (EcologyV1(), EcologyV1(movement_energy_cost=.01),
             EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02))


def _trait_metadata(persistence):
    if persistence is None:
        return {}
    if not isinstance(persistence, PersistenceV1):
        raise ValueError("persistence must be None or a PersistenceV1 setting")
    return dict(persistence=dict(version=persistence.version, **asdict(persistence)))


def _keys():
    kg, kw, kd = jax.random.split(jax.random.PRNGKey(DEVELOPMENT_SEED), 3)
    return kg, jax.random.split(kw, 3), jax.random.split(kd, 3)


def _genome(flat):
    kg, _, _ = _keys()
    initial, unravel = ravel_pytree(init_genome(DEVELOPMENT_CFG, kg))
    if flat is None:
        flat = np.asarray(initial)
    flat = np.asarray(flat)
    if flat.shape != (155,) or not np.issubdtype(flat.dtype, np.floating) or not np.isfinite(flat).all():
        raise ValueError("development lineage requires a finite flat 155-number rule genome")
    flat = np.asarray(flat, np.float32)
    if not np.isfinite(flat).all():
        raise ValueError("genome is not finite at the runtime float32 precision")
    return flat, unravel(jnp.asarray(flat))


def _observations(raw):
    events = raw["events"]
    f, p, before, after = (np.asarray(events[name]) for name in ("food", "poison", "alive_before", "alive_after"))
    if any(x.shape != (12, 2) or not np.isin(x, (0, 1)).all() for x in (f, p, before, after)):
        raise ValueError("invalid development cohort event shapes or binary values")
    f, p, before, after = (x.astype(bool) for x in (f, p, before, after))
    if (not before[0].all() or np.any(f & p) or np.any((f | p) & ~before)
            or np.any(after & ~before) or not np.array_equal(before[1:], after[:-1])):
        raise ValueError("inconsistent development cohort events")
    returns = f.sum(0, dtype=np.int64)-p.sum(0, dtype=np.int64)
    if float(np.asarray(raw["fitness"])) != float(returns.mean()):
        raise ValueError("cohort fitness disagrees with original-agent events")
    row = dict(food=int(f.sum()), poison=int(p.sum()), fitness=float(returns.mean()),
               starting_agents=2, surviving_agents=int(after[-1].sum()),
               survival=float(after[-1].mean()), no_encounter_agents=int((~(f | p).any(0)).sum()))
    if "ecology" in raw:
        ecology = raw["ecology"]
        occupied, loss = np.asarray(ecology["sheltered"]), np.asarray(ecology["energy_lost"])
        if (occupied.shape != (12, 2) or loss.shape != (12, 2) or not np.isin(occupied, (0, 1)).all()
                or not np.isfinite(loss).all() or np.any(loss < 0) or np.any(occupied & ~before)
                or np.any(loss[~before] != 0)):
            raise ValueError("invalid shelter or extra-energy diagnostics")
        row.update(sheltered_agent_steps=int(occupied.sum()), extra_energy_lost=float(loss.sum()))
        if "initial_sheltered" in ecology:
            initially = np.asarray(ecology["initial_sheltered"])
            if initially.shape != (2,) or not np.isin(initially, (0, 1)).all():
                raise ValueError("invalid initial shelter occupancy")
            initially = initially.astype(bool)
            row.update(agents_initially_sheltered=int(initially.sum()),
                       agents_reaching_shelter=int((occupied.any(0) & ~initially).sum()),
                       exposed_agents_reaching_shelter_before_cold=int((occupied[:6].any(0) & ~initially).sum()))
    if "persistence" in raw:
        for name in ("active", "remembered", "progress_observed", "stalled", "tactic_attempted", "action_switched"):
            event = np.asarray(raw["persistence"][name])
            if event.shape != (12, 2) or not np.isin(event, (0, 1)).all() or np.any(event.astype(bool) & ~before):
                raise ValueError("invalid persistence activity diagnostics")
            row["persistence_"+name+"_agent_steps"] = int(event.sum())
    return row


def run_development_curriculum(flat_genome=None, *, seed=DEVELOPMENT_SEED,
                               memory_condition="shared", on_cohort=None, persistence=None):
    """Run the fixed seed-41 mechanism fixture; callback streams each complete pair.

    New agents and developmental keys are used in every cohort. Shared memory is
    the sensory-to-motor lineage archive; empty memory is the matched fresh arm.
    Both arms run regardless of the selected display condition. Without a callback,
    the returned ``cohorts`` retain all three compact paired traces.
    Optional persistence applies equally to both arms and resets each lifetime;
    these arms do not compare persistence enabled against disabled.
    """
    if type(seed) is not int or seed != DEVELOPMENT_SEED or memory_condition not in MEMORY_CONDITIONS:
        raise ValueError("this development fixture permits seed 41 and shared/empty memory only")
    trait = _trait_metadata(persistence)
    options = {} if persistence is None else dict(persistence=persistence)
    started = time.monotonic()
    flat, genome = _genome(flat_genome)
    _, wkeys, dkeys = _keys()
    records, observations = [], []

    def receive(row):
        index = len(observations)
        if index >= 3 or row["life"] != index:
            raise ValueError("unexpected or duplicate development cohort")
        if (not np.array_equal(row["world_key"], wkeys[index])
                or not np.array_equal(row["development_key"], dkeys[index])
                or not np.array_equal(row["agent_ids"], np.column_stack((np.full(2, index), np.arange(2))))
                or row["rule"] != 0):
            raise ValueError("development cohort keys, identities or rule changed")
        if (row.get("persistence") != trait.get("persistence")
                or any(("persistence" in row[key]) != (persistence is not None) for key in MEMORY_CONDITIONS.values())):
            raise ValueError("development cohort persistence setting or diagnostics changed")
        paired = {name: _observations(row[key]) for name, key in MEMORY_CONDITIONS.items()}
        observation = dict(cohort=index, phase=PHASES[index], memory_condition=memory_condition,
                           selected=paired[memory_condition], paired=paired)
        if on_cohort is None:
            records.append(row)
        else:
            on_cohort(row, observation)
        observations.append(observation)
        if time.monotonic()-started >= MAX_SECONDS:
            raise TimeoutError("development lineage deadline reached; completed cohorts remain preserved")

    result = lineage.collective_lives(genome, wkeys, dkeys, np.zeros(3, np.int32), DEVELOPMENT_CFG,
                                     ecologies=ECOLOGIES, plastic=True, on_life=receive, **options)
    if len(observations) != 3 or result.get("completed_lives") != 3:
        raise ValueError("lineage did not finish exactly three development cohorts")
    if result.get("persistence") != trait.get("persistence"):
        raise ValueError("final lineage persistence setting changed")
    return dict(protocol=PROTOCOL, purpose="development mechanism fixture; no scientific claim",
                scientific_result=False, seed=seed, memory_condition=memory_condition,
                schedule_independent_of_competence=True, observations=observations, cohorts=records,
                lineage=result, elapsed_seconds=time.monotonic()-started, **trait)


def load_development_archive(run_dir):
    """Load only a completed fixture's portable lineage checkpoint, without pickle.

    Pass the returned state to ``lineage.collective_lives(..., archive_state=state)``
    with the same genome/layout, matching persistence setting and new world/development
    keys. This loads saved lessons; it does not resume or modify the tracked fixture.
    """
    folder = Path(run_dir).resolve()

    def artifact(name):
        path = folder/name
        if not path.is_file() or path.resolve().parent != folder:
            raise ValueError(f"completed development archive requires its local {name}")
        return path

    config = json.loads(artifact("config.json").read_bytes())
    summary = json.loads(artifact("summary.json").read_bytes())
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("development fixture configuration and summary must be objects")
    if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "development"
            or config.get("cfg") != DEVELOPMENT_CFG._asdict() or config.get("scientific_result") is not False
            or summary.get("protocol") != PROTOCOL or summary.get("condition_split") != "development"
            or summary.get("status") != "finished" or summary.get("completed_cohorts") != 3
            or summary.get("expected_cohorts") != 3 or summary.get("scientific_result") is not False):
        raise ValueError("archive loading requires a finished three-cohort development fixture")
    if summary.get("final_memory") != dict(metadata="lineage_final.json", arrays="lineage_final.npz"):
        raise ValueError("completed fixture must reference its fixed local lineage_final artifacts")
    metadata = json.loads(artifact("lineage_final.json").read_bytes())
    if (not isinstance(metadata, dict) or metadata.get("protocol") != "lineage-v1"
            or metadata.get("mode") != "collective" or metadata.get("completed_lives") != 3):
        raise ValueError("final memory is not a completed collective lineage fixture")
    state = metadata.get("archive_state")
    if not isinstance(state, dict):
        raise ValueError("this fixture has no portable archive_state; legacy traces alone are not a checkpoint")
    expected_fields = {"protocol", "state_version", "method", "genome_sha256", "layout", "next_life",
                       "archive", "used_world_keys", "used_development_keys"}
    if "persistence" in state:
        setting = state["persistence"]
        if (not isinstance(setting, dict) or set(setting) != {"version", "memory_steps", "stall_steps"}
                or setting["version"] != PersistenceV1.version):
            raise ValueError("portable archive persistence setting is invalid")
        expected = _trait_metadata(PersistenceV1(memory_steps=setting["memory_steps"], stall_steps=setting["stall_steps"]))
        expected_fields.add("persistence")
    else:
        expected = {}
    for document in (config, summary, metadata, state):
        actual = {"persistence": document["persistence"]} if "persistence" in document else {}
        if actual != expected:
            raise ValueError("portable archive persistence setting disagrees with its fixture")
    layout = dict(sensors=49, motors=6, hidden=3, hidden_types=3, view_radius=1,
                  weight_bound=3., ecology_version="ecology-v1")
    digest = state.get("genome_sha256", "")
    if (set(state) != expected_fields or state.get("protocol") != "lineage-v1" or state.get("state_version") != 1
            or state.get("method") != "sensor-motor-mean-delta-v1" or state.get("next_life") != 3
            or state.get("layout") != layout or not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError("portable archive_state metadata does not match the development fixture")
    loaded = dict(state)
    with np.load(artifact("lineage_final.npz"), allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)):
            raise ValueError("checkpoint archive contains duplicate array indices")
        used_indices = set()
        for field, shape, dtype in (("archive", (6, 49), "float32"),
                                    ("used_world_keys", (3, 2), "uint32"),
                                    ("used_development_keys", (3, 2), "uint32")):
            reference = state[field]
            if not isinstance(reference, dict) or set(reference) != {"array", "shape", "dtype"}:
                raise ValueError("checkpoint arrays require indexed local references")
            key = reference["array"]
            if (not isinstance(key, str) or len(key) != 9 or not key.startswith("array")
                    or any(ch not in "0123456789" for ch in key[5:]) or key not in arrays.files or key in used_indices
                    or reference["shape"] != list(shape) or reference["dtype"] != dtype):
                raise ValueError("checkpoint array reference, dtype or shape is invalid")
            used_indices.add(key)
            value = arrays[key]
            if value.shape != shape or str(value.dtype) != dtype or not np.isfinite(value).all():
                raise ValueError("checkpoint array content disagrees with its indexed dtype or shape")
            if field == "archive" and np.any(np.abs(value) > DEVELOPMENT_CFG.wmax):
                raise ValueError("checkpoint archive exceeds its weight bound")
            if field != "archive" and len(np.unique(value, axis=0)) != 3:
                raise ValueError("checkpoint cohort keys must be distinct")
            loaded[field] = value.copy()
    return loaded


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_status(folder, data):
    pending = folder/"status.tmp.json"
    _write_json(pending, data)
    os.replace(pending, folder/"status.json")


def _bundle(folder, name, data):
    arrays = {}

    def pack(value):
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError("raw cohort arrays must be finite numeric values")
            key = f"array{len(arrays):04d}"
            arrays[key] = array
            return dict(array=key, shape=list(array.shape), dtype=str(array.dtype))
        return value

    metadata = pack(data)
    json.dumps(metadata, allow_nan=False)
    with (folder/(name+".npz")).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    _write_json(folder/(name+".json"), metadata)
    return dict(metadata=name+".json", arrays=name+".npz")


def run_tracked(args, git, root):
    """Runner adapter: immutable attempt name, streamed evidence, no resume/final path."""
    if (not args.quick or args.condition_split != "development" or args.seed != 41 or args.generations != 0
            or args.resume or args.cfg or args.no_plastic or args.controller not in (None, "rules-v0")
            or args.objective is not None or args.structural_plasticity not in (None, "off")
            or args.max_train_seconds is not None
            or (args.pop, args.worlds, args.sigma, args.lr, args.eval_worlds, args.checkpoint_every)
               != (128, 8, .1, .1, 128, 10)):
        raise ValueError("ecology-lineage-v1 requires --quick --condition-split development --seed 41 --generations 0; "
                         "no final science, resume, evolution or other configuration overrides are enabled")
    condition = args.memory_condition or "shared"
    persistence = PersistenceV1() if getattr(args, "persistence", False) else None
    trait = _trait_metadata(persistence)
    if condition not in MEMORY_CONDITIONS or not args.name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in args.name):
        raise ValueError("invalid development run name or memory condition")
    root = Path(root)
    if any((root/"runs").glob("*_"+args.name)):
        raise ValueError("a run with this name already exists; preserve it and use a new development name")
    if args.start == "blank":
        flat, _ = _genome(None)
        buffer = io.BytesIO()
        np.save(buffer, flat)
        source_bytes = buffer.getvalue()
        source = dict(kind="blank", seed=41)
    elif args.start.startswith("file:"):
        path = root/args.start[5:]
        source_bytes = path.read_bytes()
        flat, _ = _genome(np.load(io.BytesIO(source_bytes), allow_pickle=False))
        source = dict(kind="file", filename=path.name)
    else:
        raise ValueError("development lineage accepts blank or file:<155-number-genome.npy>")
    source["sha256"] = hashlib.sha256(source_bytes).hexdigest()
    folder = root/"runs"/(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+args.name)
    folder.mkdir(parents=True, exist_ok=False)
    recorded_args = vars(args).copy()
    if source["kind"] == "file":
        recorded_args["start"] = "file:genome_start.npy"
    config = dict(args=recorded_args, git=git, protocol=PROTOCOL, condition_split="development", cfg=DEVELOPMENT_CFG._asdict(),
        scientific_result=False, purpose="development mechanism fixture; not a registered evaluation",
        memory_condition=condition, selected_trace=MEMORY_CONDITIONS[condition], paired_traces_retained=True,
        source=source, genome_size=155, max_seconds=MAX_SECONDS, schedule_independent_of_competence=True,
        schedule=[dict(cohort=i, phase=phase, rule=0, ecology=dict(version=ecology.version, **asdict(ecology)))
                  for i, (phase, ecology) in enumerate(zip(PHASES, ECOLOGIES))], **trait)
    _write_json(folder/"config.json", config)
    with (folder/"genome_start.npy").open("xb") as output:
        output.write(source_bytes)
    history = []
    _write_status(folder, dict(status="running", completed_cohorts=0, expected_cohorts=3))
    try:
        with (folder/"metrics.jsonl").open("x", encoding="utf-8") as stream:
            def persist(row, observation):
                files = _bundle(folder, f"cohort_{row['life']:03d}", row)
                record = dict(observation, raw_files=files)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                history.append(record)
                _write_status(folder, dict(status="running", completed_cohorts=len(history), expected_cohorts=3))

            options = {} if persistence is None else dict(persistence=persistence)
            result = run_development_curriculum(flat, memory_condition=condition, on_cohort=persist, **options)
        final_files = _bundle(folder, "lineage_final", result["lineage"])
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        outcome = dict(status=status, protocol=PROTOCOL, condition_split="development", scientific_result=False,
                       generation=0, completed_cohorts=len(history), expected_cohorts=3, memory_condition=condition,
                       observations=history, error=f"{type(error).__name__}: {error}", **trait)
        _write_json(folder/"summary.json", outcome)
        _write_status(folder, dict(status=status, completed_cohorts=len(history), expected_cohorts=3))
        raise
    outcome = dict(status="finished", protocol=PROTOCOL, condition_split="development", scientific_result=False,
                   generation=0, completed_cohorts=3, expected_cohorts=3, memory_condition=condition,
                   schedule_independent_of_competence=True, observations=history, final_memory=final_files,
                   elapsed_seconds=result["elapsed_seconds"], **trait)
    _write_json(folder/"summary.json", outcome)
    _write_status(folder, dict(status="finished", completed_cohorts=3, expected_cohorts=3))
    print("development mechanism fixture only; no scientific gate or learning claim")
    print("run folder:", folder)
    return folder
