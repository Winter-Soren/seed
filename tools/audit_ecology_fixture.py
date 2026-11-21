"""Read-only audit of the completed ecology-transfer-smoke-a0 fixture.

Stdlib and NumPy only: no Seed/JAX import, simulation, optimizer or file writes.
Reads raw JSON/NPZ bundles only after summary/status say all three cohorts are
finished. The supplied terminal supervisor log establishes the process budget.
This checks engineering evidence; it never establishes a scientific result.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

NAME = "ecology-transfer-smoke-a0"
PROTOCOL = "ecology-lineage-v1"
ORIGINAL = "runs/20261112-023245_E103-A-blank-s0/genome_final.npy"
ORIGINAL_SHA = "477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"
CFG = dict(G=6, M=2, T=12, R=1, n_items=4, H=3, n_htypes=3,
           e0=1., decay=.003, gain=.25, e_max=2., micro=2, wmax=3.)
PHASES = ("base", "costly_movement", "cold_with_shelter")
COSTS = ((0., 0.), (.01, 0.), (.01, .02))
EVENTS = ("food", "poison", "alive_before", "alive_after")
ARMS = {"shared": "inherited", "empty": "fresh"}
D, O, N = 49, 6, 58


class AuditError(Exception):
    pass


class Incomplete(AuditError):
    pass


def require(condition, label):
    if not condition:
        raise AuditError(label)


def same(actual, expected, label):
    if isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.shape == expected.shape
                and np.array_equal(actual, expected), label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + " fields")
        for key, value in expected.items():
            same(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float, np.number)) and np.isfinite(actual)
                and math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-7), label)
    else:
        require(actual == expected, label)


def subset(actual, expected, label):
    require(isinstance(actual, dict) and all(key in actual for key in expected), label + " missing fields")
    same({key: actual[key] for key in expected}, expected, label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def observations(raw, label, movement, cold):
    require(set(raw["events"]) == set(EVENTS), label + " event fields")
    events = [np.asarray(raw["events"][key]) for key in EVENTS]
    require(all(value.shape == (12, 2) and np.isin(value, (0, 1)).all() for value in events), label + " binary events")
    food, poison, before, after = (value.astype(bool) for value in events)
    require(before[0].all() and not (food & poison).any() and not ((food | poison) & ~before).any()
            and not (after & ~before).any() and np.array_equal(before[1:], after[:-1]), label + " event/death accounting")
    returns = food.sum(0, dtype=np.int64)-poison.sum(0, dtype=np.int64)
    same(float(raw["fitness"]), float(returns.mean()), label + " fitness")
    same(float(raw["survived"]), float(after[-1].mean()), label + " survival")
    counts = np.array([[food[:6].sum(), poison[:6].sum()], [food[6:].sum(), poison[6:].sum()]])
    same(raw["counts"], counts, label + " half-life counts")
    ecology = raw["ecology"]
    require(set(ecology) == {"shelter", "initial_sheltered", "sheltered", "energy_lost"}, label + " ecology fields")
    shelter, initial, occupied, loss = (np.asarray(ecology[key]) for key in (
        "shelter", "initial_sheltered", "sheltered", "energy_lost"))
    require(shelter.shape == (6, 6) and np.isin(shelter, (0, 1)).all() and shelter.sum() == 1, label + " shelter geometry")
    require(initial.shape == (2,) and np.isin(initial, (0, 1)).all()
            and occupied.shape == loss.shape == (12, 2) and np.isin(occupied, (0, 1)).all()
            and np.isfinite(loss).all() and np.all(loss >= 0), label + " occupancy/loss shapes")
    initial, occupied = initial.astype(bool), occupied.astype(bool)
    require(not (occupied & ~before).any() and np.all(loss[~before] == 0), label + " dead-agent occupancy/energy loss")
    bound = movement * before + cold * (np.arange(12)[:, None] >= 6) * before * ~occupied
    require(np.all(loss <= bound + 1e-6), label + " loss exceeds registered pressure")
    if movement == cold == 0:
        require(np.all(loss == 0), label + " zero-pressure baseline loss")
    weights = raw["weights"]
    for key in ("birth_W_sha256", "initial_W_sha256", "terminal_W_sha256"):
        require(isinstance(weights[key], list) and len(weights[key]) == 2
                and all(isinstance(value, str) and len(value) == 64
                        and all(ch in "0123456789abcdef" for ch in value) for value in weights[key]), label + " " + key)
    same(weights["final_alive"], after[-1], label + " final weight-state eligibility")
    changed, magnitude = weights["changed_connections"], weights["mean_abs_change"]
    require(changed.shape == magnitude.shape == (2,) and np.isfinite(magnitude).all()
            and np.issubdtype(changed.dtype, np.integer) and np.all((changed >= 0) & (changed <= N*N))
            and np.all(magnitude >= 0), label + " weight-change diagnostics")
    row = dict(food=int(food.sum()), poison=int(poison.sum()), fitness=float(returns.mean()),
        starting_agents=2, surviving_agents=int(after[-1].sum()), survival=float(after[-1].mean()),
        no_encounter_agents=int((~(food | poison).any(0)).sum()), sheltered_agent_steps=int(occupied.sum()),
        extra_energy_lost=float(loss.sum()), agents_initially_sheltered=int(initial.sum()),
        agents_reaching_shelter=int((occupied.any(0) & ~initial).sum()),
        exposed_agents_reaching_shelter_before_cold=int((occupied[:6].any(0) & ~initial).sum()))
    deaths = before & ~after
    details = dict(deaths=int(deaths.sum()), death_steps=np.where(deaths.any(0), deaths.argmax(0), -1).tolist(),
        per_agent_return=returns.tolist(), shelter_cells=np.argwhere(shelter).tolist(),
        donor_terminal_alive=after[-1].tolist())
    return row, details


class Auditor:
    def __init__(self, root):
        self.root, self.hashes = root.resolve(), {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "artifact path leaves source checkout")
        return path

    def read(self, relative):
        path = self.path(relative)
        if not path.is_file():
            raise Incomplete("missing artifact: " + str(relative))
        data = path.read_bytes()
        key = path.relative_to(self.root).as_posix()
        digest = sha(data)
        require(key not in self.hashes or digest == self.hashes[key], "artifact changed during audit: " + key)
        self.hashes[key] = digest
        return data

    def json(self, relative):
        return json.loads(self.read(relative), parse_constant=lambda value: (_ for _ in ()).throw(AuditError("nonfinite JSON")))

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True, timeout=15).stdout.strip()

    def bundle(self, folder, mapping, expected_name):
        same(mapping, dict(metadata=expected_name + ".json", arrays=expected_name + ".npz"), "raw bundle mapping")
        metadata = self.json(folder + "/" + mapping["metadata"])
        with np.load(io.BytesIO(self.read(folder + "/" + mapping["arrays"])), allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "duplicate NPZ keys")
            arrays = {name: archive[name] for name in archive.files}
        require(set(arrays) == {f"array{i:04d}" for i in range(len(arrays))}, "noncanonical raw array keys")
        used = []
        def unpack(value):
            if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
                key = value["array"]
                require(key in arrays and key not in used, "missing/reused raw array descriptor")
                used.append(key)
                array = arrays[key]
                require(list(array.shape) == value["shape"] and str(array.dtype) == value["dtype"]
                        and array.dtype.kind in "biuf" and np.isfinite(array).all(), "raw shape/dtype/finiteness")
                return array
            if isinstance(value, dict):
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value
        result = unpack(metadata)
        require(set(used) == set(arrays), "unreferenced raw arrays")
        return result

    def audit(self, folder, source_commit):
        paths = list(self.path("runs").glob("*_" + NAME))
        require(len(paths) == 1 and paths[0].resolve() == self.path(folder), "require unique canonical fixture attempt")
        summary, status = self.json(folder + "/summary.json"), self.json(folder + "/status.json")
        if summary.get("status") != "finished" or status.get("status") != "finished":
            raise Incomplete("fixture is not finished; no partial raw data opened")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split="development", scientific_result=False,
            generation=0, completed_cohorts=3, expected_cohorts=3, memory_condition="shared",
            schedule_independent_of_competence=True), "summary")
        same(status, dict(status="finished", completed_cohorts=3, expected_cohorts=3), "terminal status")
        require(isinstance(summary.get("elapsed_seconds"), (int, float))
                and math.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] >= 0,
                "finite nonnegative reported fixture time")
        expected_files = {"config.json", "summary.json", "status.json", "genome_start.npy", "metrics.jsonl"}
        expected_files.update(f"{name}.{suffix}" for name in ("cohort_000", "cohort_001", "cohort_002", "lineage_final")
                              for suffix in ("json", "npz"))
        require({path.name for path in self.path(folder).iterdir()} == expected_files, "completed fixture artifact set")
        config = self.json(folder + "/config.json")
        subset(config, dict(protocol=PROTOCOL, condition_split="development", cfg=CFG, scientific_result=False,
            memory_condition="shared", selected_trace="inherited", paired_traces_retained=True, genome_size=155,
            max_seconds=120, schedule_independent_of_competence=True), "fixture configuration")
        subset(config["args"], dict(name=NAME, eval_protocol=PROTOCOL, quick=True, condition_split="development", seed=41,
            generations=0, memory_condition="shared", start="file:genome_start.npy", resume=None, no_plastic=False,
            allow_dirty=False, objective=None, max_train_seconds=None, pop=128, worlds=8, sigma=.1, lr=.1,
            eval_worlds=128, checkpoint_every=10), "fixture arguments")
        require(not config["args"].get("cfg") and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), "unregistered fixture override")
        git = config["git"]
        require(git.get("dirty") is False and isinstance(git.get("commit"), str)
                and len(git["commit"]) == 40, "source must be recorded clean")
        expected_trees = {name: self.git("rev-parse", source_commit + ":" + name) for name in ("seed", "scripts")}
        same({name: self.git("rev-parse", git["commit"] + ":" + name) for name in expected_trees}, expected_trees, "registered source trees")
        expected_schedule = [dict(cohort=i, phase=phase, rule=0,
            ecology=dict(version="ecology-v1", movement_energy_cost=COSTS[i][0],
                         seasonal_energy_cost=COSTS[i][1])) for i, phase in enumerate(PHASES)]
        same(config["schedule"], expected_schedule, "fixed pressure schedule")
        original, copied = self.read(ORIGINAL), self.read(folder + "/genome_start.npy")
        require(sha(original) == ORIGINAL_SHA and copied == original, "byte-identical original A0 genome")
        same(config["source"], dict(kind="file", filename="genome_final.npy", sha256=ORIGINAL_SHA), "source fingerprint")
        flat = np.load(io.BytesIO(copied), allow_pickle=False)
        require(flat.shape == (155,) and flat.dtype.kind == "f" and np.isfinite(flat).all(), "original genome shape/finiteness")
        text = self.read(folder + "/metrics.jsonl").decode("utf-8")
        require(text.endswith("\n"), "partial metrics line")
        history = [json.loads(line) for line in text.splitlines()]
        require(len(history) == 3, "exactly three completed cohort records")
        same(summary["observations"], history, "summary/history identity")
        final = self.bundle(folder, summary["final_memory"], "lineage_final")
        subset(final, dict(protocol="lineage-v1", mode="collective", config=CFG, lives=[], completed_lives=3), "final lineage")
        genome = final["genome"]
        shapes = dict(conn=(5, 5), hebb=(5, 5, 4), log_eta=(), log_plr=(), mod=(3,), wscale=(5, 5))
        require(set(genome) == set(shapes) and all(genome[key].shape == shape and genome[key].dtype == np.float32
                for key, shape in shapes.items()), "final genome schema")
        same(np.concatenate([genome[key].ravel() for key in sorted(genome)]), np.asarray(flat, np.float32), "unchanged final genome")
        digest = hashlib.sha256()
        for key in sorted(genome):
            value = genome[key]
            digest.update(key.encode() + value.dtype.str.encode() + str(value.shape).encode() + value.tobytes())
        checkpoint_metadata = dict(protocol="lineage-v1", state_version=1, method="sensor-motor-mean-delta-v1",
            genome_sha256=digest.hexdigest(), layout=dict(sensors=D, motors=O, hidden=3, hidden_types=3,
                view_radius=1, weight_bound=3., ecology_version="ecology-v1"))
        previous, world_keys, development_keys, checked = np.zeros((O, D), np.float32), [], [], []
        for i, record in enumerate(history):
            raw = self.bundle(folder, record["raw_files"], f"cohort_{i:03d}")
            subset(raw, dict(life=i, rule=0, ecology=expected_schedule[i]["ecology"]), f"cohort {i}")
            same(raw["agent_ids"], np.column_stack((np.full(2, i), np.arange(2))), f"cohort {i} newborn identities")
            for field, keys in (("world_key", world_keys), ("development_key", development_keys)):
                value = raw[field]
                require(value.shape == (2,) and value.dtype == np.uint32
                        and not any(np.array_equal(value, prior) for prior in keys), f"cohort {i} fresh {field}")
                keys.append(value)
            shared, fresh = raw["inherited"], raw["fresh"]
            same(shared["weights"]["birth_W_sha256"], fresh["weights"]["birth_W_sha256"], f"cohort {i} raw birth signatures")
            same(shared["ecology"]["shelter"], fresh["ecology"]["shelter"], f"cohort {i} matched shelter")
            same(shared["ecology"]["initial_sheltered"], fresh["ecology"]["initial_sheltered"], f"cohort {i} matched initial occupancy")
            paired, details = {}, {}
            for displayed, field in ARMS.items():
                paired[displayed], details[displayed] = observations(raw[field], f"cohort {i}/{displayed}", *COSTS[i])
            same(record, dict(cohort=i, phase=PHASES[i], memory_condition="shared", selected=paired["shared"],
                paired=paired, raw_files=record["raw_files"]), f"cohort {i} recomputed metrics")
            before, delta, mask, counts, after = (raw[key] for key in (
                "archive_before", "donor_delta", "donor_mask", "donor_counts", "archive_after"))
            require(before.shape == after.shape == counts.shape == (O, D) and delta.shape == mask.shape == (2, O, D)
                    and before.dtype == after.dtype == delta.dtype == np.float32
                    and np.isin(mask, (0, 1)).all() and np.all(np.abs(delta) <= 6)
                    and np.all(delta[mask == 0] == 0), f"cohort {i} donor fields/bounds")
            same(before, previous, f"cohort {i} archive chain")
            same(counts, mask.sum(0), f"cohort {i} donor counts including zero/dead donors")
            expected = np.empty((O, D), np.float32)
            for motor in range(O):
                for sensor in range(D):
                    total = np.float32(0.)
                    eligible = 0
                    for donor in range(2):
                        if mask[donor, motor, sensor]:
                            total += delta[donor, motor, sensor]
                            eligible += 1
                    expected[motor, sensor] = np.clip(before[motor, sensor] + total/np.float32(max(eligible, 1)), -3., 3.)
            same(after, expected, f"cohort {i} per-edge archive transition")
            overlap = np.count_nonzero((before != 0)[None] & (mask != 0), axis=(1, 2))
            changed = [a != b for a, b in zip(shared["weights"]["initial_W_sha256"], fresh["weights"]["initial_W_sha256"])]
            require(all(not difference or overlap[j] > 0 for j, difference in enumerate(changed)),
                    f"cohort {i} recipient signature changed without nonzero archive/mask overlap")
            require(np.all(shared["weights"]["changed_connections"] >= np.count_nonzero(delta, axis=(1, 2))),
                    f"cohort {i} donor changes exceed full-weight change count")
            if i == 0:
                require(not np.any(before) and not any(changed), "first cohort must receive an empty archive")
                for field in EVENTS:
                    same(shared["events"][field], fresh["events"][field], "first matched life events")
            expected_state = dict(checkpoint_metadata, archive=after, next_life=i+1,
                used_world_keys=np.asarray(world_keys), used_development_keys=np.asarray(development_keys))
            require(all(raw["archive_state"][key].dtype == np.uint32
                        for key in ("used_world_keys", "used_development_keys")), f"cohort {i} checkpoint key dtypes")
            same(raw["archive_state"], expected_state, f"cohort {i} portable archive checkpoint")
            checked.append(dict(cohort=i, phase=PHASES[i], paired=paired, details=details,
                archive_nonzero_before=int(np.count_nonzero(before)), archive_nonzero_after=int(np.count_nonzero(after)),
                archive_transition_changed_edges=int(np.count_nonzero(after-before)),
                donor_update_nonzero_edges=int(np.count_nonzero(delta.sum(0)/np.maximum(counts, 1))),
                recipient_overlap_edges=overlap.tolist(), recipient_initial_signature_changed=changed,
                fresh_initial_signature_matches_raw_birth=[a == b for a, b in zip(
                    fresh["weights"]["initial_W_sha256"], fresh["weights"]["birth_W_sha256"])],
                donor_zero_changes_in_existing_edges=int(np.count_nonzero((delta == 0) & (mask != 0))),
                archive_checkpoint_files=record["raw_files"]))
            previous = after
        same(final["final_archive"], previous, "final archive matches last transition")
        require(all(final["archive_state"][key].dtype == np.uint32
                    for key in ("used_world_keys", "used_development_keys")), "final checkpoint key dtypes")
        same(final["archive_state"], expected_state, "final portable checkpoint")
        return dict(run=folder, source_commit=git["commit"], expected_code_trees=expected_trees,
            original_genome_sha256=ORIGINAL_SHA, completed_cohorts=3, matched_cohort_pairs=3,
            rollout_lives=6, starting_agents_per_life=2,
            steps_per_life=12, cohorts=checked, final_archive_nonzero=int(np.count_nonzero(previous)),
            final_archive_checkpoint_files=summary["final_memory"], reported_elapsed_seconds=summary["elapsed_seconds"],
            total_deaths_by_arm={arm: sum(row["details"][arm]["deaths"] for row in checked) for arm in ARMS},
            transfer_detected_by_signatures=any(any(row["recipient_initial_signature_changed"]) for row in checked[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="isolated committed ecology checkout")
    parser.add_argument("--run", required=True, help="completed run path relative to that checkout")
    parser.add_argument("--source-commit", required=True, help="full registered ecology source SHA")
    parser.add_argument("--guard", type=Path, required=True, help="completed supervisor JSONL, not a live log")
    args = parser.parse_args()
    require(len(args.source_commit) == 40 and all(ch in "0123456789abcdef" for ch in args.source_commit), "full source SHA required")
    audit = Auditor(args.root)
    report = dict(status="not-audited", kind="engineering-fixture", scientific_result=False,
        helper_sha256=sha(Path(__file__).read_bytes()))
    try:
        guard_data = args.guard.read_bytes()
        guard = [json.loads(line) for line in guard_data.decode("utf-8-sig").splitlines()]
        if not guard or guard[-1].get("event") not in ("finished", "stopped"):
            raise Incomplete("supervisor is not terminal; no fixture arrays opened")
        terminal = guard[-1]
        report["guard"] = dict(sha256=sha(guard_data), terminal=terminal)
        result = audit.audit(args.run, args.source_commit)
        report.update(artifact_validity="passed", fixture=result)
        elapsed = terminal.get("wallSeconds")
        within_budget = isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and 0 <= elapsed <= 120
        timer_consistent = isinstance(elapsed, (int, float)) and result["reported_elapsed_seconds"] <= elapsed + .001
        process_pass = terminal.get("event") == "finished" and terminal.get("exitCode") == 0 and within_budget and timer_consistent
        report.update(status="passed" if process_pass else "failed", engineering_acceptance="passed" if process_pass else "failed",
            process_budget_seconds=120, process_budget_passed=process_pass, limitations=[
                "One fixed seed-41 software fixture, not three independent evolutionary seeds or an evaluation protocol.",
                "World/development keys and initial occupancy/shelter are paired; random streams and physics are not regenerated.",
                "Birth/initial/terminal weights are signatures, not full matrices: exact clipped recipient weights are not reconstructable.",
                "Archive transitions count every donor with an existing edge, including zero changes, without filtering on survival.",
                "A changed signature with archive overlap shows a detected transfer; it does not establish a beneficial learned behavior.",
                "Death counts are explicit. If no deaths occurred, this fixture does not demonstrate transfer after an observed death.",
                "The saved archive checkpoint is validated, but no resumed simulation was executed by this audit.",
                "Fitness, shelter occupancy and survival here do not support generalization or a scientific improvement claim."])
        if not process_pass:
            report["error"] = "terminal process outcome did not satisfy exit 0 within the registered 120-second full-process budget"
        code = 0 if process_pass else 1
    except Incomplete as error:
        report.update(status="incomplete", error=str(error))
        code = 2
    except (AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        code = 1
    report["checked_file_sha256"] = audit.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
