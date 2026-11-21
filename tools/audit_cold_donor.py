"""Read-only independent E117 audit using NumPy and the standard library only.

No simulator imports, runs, file writes, or partial-result inspection. Final mode
requires all three completed canonical runs before opening any measurements.
Exit 0 means artifact audit passed, not that the scientific gate passed; exit 2
means incomplete evidence, and exit 1 means an artifact discrepancy.
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


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cold-donor-v1"
ORIGINALS = (
    ("runs/20261112-023245_E103-A-blank-s0/genome_final.npy", "477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"),
    ("runs/20261112-050319_E103-A-blank-s1/genome_final.npy", "35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899"),
    ("runs/20261112-102755_E103-A-blank-s2/genome_final.npy", "f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6"),
)
CFG = dict(G=16, M=4, T=300, R=2, n_items=24, H=48, n_htypes=3,
           e0=1., decay=.003, gain=.25, e_max=2., micro=2, wmax=3.)
ECOLOGY = dict(version="ecology-v1", movement_energy_cost=.01, seasonal_energy_cost=.02)
KEY_ORDER = ["donor_world", "donor_development", "target_world", "target_development"]
ARMS = ("donor", "inherited", "fresh")
EVENTS = ("food", "poison", "alive_before", "alive_after", "sheltered")


class AuditError(Exception):
    pass


class Incomplete(AuditError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def same(actual, expected, label):
    """Check structure/counts exactly; allow roundoff only in real arithmetic."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": fields")
        for key, value in expected.items():
            same(actual[key], value, label + "." + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ": length")
        for index, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f"{label}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, (bool, np.bool_))
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def subset(actual, expected, label):
    require(isinstance(actual, dict) and set(expected).issubset(actual), label + ": missing fields")
    same({key: actual[key] for key in expected}, expected, label)


def world_stats(values):
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and len(values) >= 1 and np.isfinite(values).all(), "invalid family vector")
    mean = float(values.sum() / len(values))
    se = None if len(values) < 2 else math.sqrt(float(np.square(values - mean).sum()) / (len(values) * (len(values) - 1)))
    return dict(mean=mean, se=se)


def ratio_stats(numerators, denominators):
    """Pooled agent ratio, with uncertainty clustered at independent families."""
    numerators, denominators = np.asarray(numerators, float), np.asarray(denominators, float)
    require(numerators.ndim == 1 and numerators.shape == denominators.shape
            and len(numerators) >= 1 and np.isfinite(numerators).all()
            and np.isfinite(denominators).all() and np.all(denominators >= 0), "invalid ratio inputs")
    if denominators.sum() == 0:
        return dict(mean=None, se=None)
    mean = float(numerators.sum() / denominators.sum())
    influence = (numerators - mean * denominators) / denominators.mean()
    return dict(mean=mean, se=world_stats(influence)["se"])


def first_living_arrival(initial_sheltered, sheltered, alive_after):
    """Death competes with arrival; a fatal entry cannot be a successful visit."""
    initial_sheltered = np.asarray(initial_sheltered, bool)
    sheltered, alive_after = np.asarray(sheltered, bool), np.asarray(alive_after, bool)
    require(sheltered.ndim == 2 and sheltered.shape == alive_after.shape
            and initial_sheltered.shape == sheltered.shape[1:], "arrival shape mismatch")
    steps = len(sheltered)
    entered = sheltered & alive_after
    first = np.where(entered.any(axis=0), entered.argmax(axis=0) + 1, steps + 1)
    return np.where(initial_sheltered, 0, first)


def scientific_gate(latency, survival):
    latency_passed = latency["mean"] is not None and latency["mean"] > 2 * latency["se"]
    survival_passed = survival["mean"] >= -2 * survival["se"]
    return dict(passed=bool(latency_passed and survival_passed),
                latency_passed=bool(latency_passed), survival_passed=bool(survival_passed))


def typed(raw, name, shape, kind, *, binary=False):
    require(name in raw, "missing array " + name)
    value = raw[name]
    require(value.shape == shape and value.dtype.kind in kind, name + ": shape or dtype")
    require(np.isfinite(value).all(), name + ": nonfinite")
    if binary:
        require(np.isin(value, (0, 1)).all(), name + ": nonbinary")
    return value


def arm_events(raw, arm, steps, agents):
    prefix = arm + "_"
    events = {key: typed(raw, prefix + key, (steps, agents), "biu", binary=True).astype(bool)
              for key in EVENTS}
    food, poison = events["food"], events["poison"]
    before, after, sheltered = events["alive_before"], events["alive_after"], events["sheltered"]
    require(before[0].all() and np.array_equal(before[1:], after[:-1]), prefix + "birth/alive continuity")
    require(not (after & ~before).any(), prefix + "resurrection")
    require(not (food & poison).any() and not ((food | poison) & ~before).any(), prefix + "invalid eating")
    require(not (sheltered & ~before).any(), prefix + "dead agent marked sheltered")
    initial = typed(raw, prefix + "initial_sheltered", (agents,), "biu", binary=True).astype(bool)
    shelter = typed(raw, prefix + "shelter", (16, 16), "biu", binary=True).astype(bool)
    require(shelter.sum() == 16, prefix + "shelter area")
    # Verify one toroidal 4 by 4 shelter, not merely its area.
    rows, columns = np.where(shelter)
    candidates = [((np.arange(16)[:, None] - r) % 16 < 4)
                  & ((np.arange(16)[None, :] - c) % 16 < 4) for r, c in zip(rows, columns)]
    require(any(np.array_equal(shelter, candidate) for candidate in candidates), prefix + "shelter geometry")
    positions = typed(raw, prefix + "initial_position", (agents, 2), "iu")
    require(((positions >= 0) & (positions < 16)).all(), prefix + "position range")
    require(np.array_equal(initial, shelter[positions[:, 0], positions[:, 1]]), prefix + "spawn shelter classification")
    entry = first_living_arrival(initial, sheltered, after)
    claimed = typed(raw, prefix + "first_entry", (agents,), "iu")
    require(np.array_equal(entry, claimed), prefix + "arrival reconstruction")
    loss = typed(raw, prefix + "energy_lost", (steps, agents), "f")
    upper = .01 * before + .02 * (np.arange(steps)[:, None] >= steps // 2) * before * ~sheltered
    require((loss >= 0).all() and np.all(loss <= upper + 1e-6)
            and np.all(loss[~before] == 0), prefix + "ecology loss bound")
    fitness = float((food.astype(np.int64) - poison.astype(np.int64)).sum() / agents)
    require(math.isclose(float(typed(raw, prefix + "fitness", (), "f")), fitness, abs_tol=1e-6), prefix + "fitness")
    for suffix in ("birth_sha256", "initial_sha256", "terminal_sha256"):
        require(typed(raw, prefix + suffix, (agents, 32), "u").dtype == np.uint8, prefix + suffix + " hash dtype")
    changed = typed(raw, prefix + "changed_connections", (agents,), "iu")
    mean_change = typed(raw, prefix + "mean_abs_weight_change", (agents,), "f")
    require(((changed >= 0) & (changed <= 183 ** 2)).all() and (mean_change >= 0).all()
            and (mean_change <= 6.).all() and np.array_equal(changed == 0, mean_change == 0), prefix + "weight change bounds")
    changed_hash = np.any(raw[prefix + "initial_sha256"] != raw[prefix + "terminal_sha256"], axis=1)
    require(np.array_equal(changed != 0, changed_hash), prefix + "terminal weight fingerprint/change count")
    outside = ~initial
    arrived = outside & (entry <= steps)
    died_without_entry = outside & ~arrived & ~after[-1]
    alive_without_entry = outside & ~arrived & after[-1]
    require(int(outside.sum()) == int(arrived.sum() + died_without_entry.sum() + alive_without_entry.sum()),
            prefix + "outside-born competing outcomes")
    return dict(initial=initial, entry=entry, outside=outside, survival=after[-1],
                food=food, poison=poison, before=before, after=after, sheltered=sheltered,
                loss=loss, fitness=fitness, reached=arrived, died_without_entry=died_without_entry,
                alive_without_entry=alive_without_entry,
                arrived_before_cold=outside & (entry <= steps // 2),
                cold_exposure=before[steps // 2:] & ~sheltered[steps // 2:])


def family_arrays(raw, index):
    expected_fields = {"keys", "rule", "steps", "agents", "cold_start", "weight_bound",
                       "archive_before", "archive_after", "donor_delta", "donor_mask", "donor_counts",
                       "recipient_mask", "recipient_birth", "recipient_initial_delta"}
    expected_fields.update(arm + "_" + key for arm in ARMS for key in (*EVENTS,
        "initial_sheltered", "shelter", "initial_position", "first_entry", "energy_lost", "fitness",
        "birth_sha256", "initial_sha256", "terminal_sha256", "changed_connections", "mean_abs_weight_change"))
    require(set(raw) == expected_fields, "raw schema fields differ from cold-donor-v1")
    for key, expected in dict(rule=index % 2, steps=300, agents=4, cold_start=150).items():
        require(int(typed(raw, key, (), "iu")) == expected, key + ": fixed design")
    require(float(typed(raw, "weight_bound", (), "f")) == 3., "fixed weight bound")
    keys = typed(raw, "keys", (4, 2), "u")
    require(keys.dtype == np.uint32 and len({row.tobytes() for row in keys}) == 4, "independent family keys")
    result = {arm: arm_events(raw, arm, 300, 4) for arm in ARMS}
    for suffix in ("shelter", "initial_position", "initial_sheltered", "birth_sha256"):
        require(np.array_equal(raw["inherited_" + suffix], raw["fresh_" + suffix]), "target paired " + suffix)
    shape = (6, 129)
    before = typed(raw, "archive_before", shape, "f")
    after = typed(raw, "archive_after", shape, "f")
    delta = typed(raw, "donor_delta", (4, *shape), "f")
    mask = typed(raw, "donor_mask", (4, *shape), "bifu", binary=True)
    counts = typed(raw, "donor_counts", shape, "ifu")
    require(np.all(before == 0), "nonzero initial archive / cross-family leakage")
    require(np.all(delta[mask == 0] == 0) and np.all(np.abs(delta) <= 6.), "donor delta mask/bounds")
    require(np.array_equal(counts, mask.sum(axis=0)), "donor counts exclude dead or unchanged donors")
    # Retain float32 arithmetic so clipping and edgewise averaging are independently exact.
    expected = np.clip(delta.sum(axis=0) / np.maximum(counts, 1), -3., 3.)
    require(np.array_equal(after, expected), "archive is mean learned delta of every original donor")
    recipient_mask = typed(raw, "recipient_mask", (4, *shape), "bifu", binary=True)
    birth = typed(raw, "recipient_birth", (4, *shape), "f")
    observed_delta = typed(raw, "recipient_initial_delta", (4, *shape), "f")
    require(np.all(np.abs(birth) <= 3.) and np.all(birth[recipient_mask == 0] == 0), "recipient birth mask/bounds")
    inherited = birth.copy()
    overlay = after * recipient_mask
    np.copyto(inherited, np.clip(birth + overlay, -3., 3.), where=overlay != 0)
    require(np.array_equal(observed_delta, inherited - birth), "recipient clipped archive overlay")
    changed = np.any(observed_delta != 0, axis=(1, 2))
    different_hash = np.any(raw["inherited_initial_sha256"] != raw["fresh_initial_sha256"], axis=1)
    require(np.array_equal(changed, different_hash), "target weight fingerprints disagree with inheritance")
    if not changed.any():
        for suffix in (*EVENTS, "first_entry", "energy_lost", "fitness", "terminal_sha256"):
            require(np.array_equal(raw["inherited_" + suffix], raw["fresh_" + suffix]),
                    "zero-overlay paired trajectories differ: " + suffix)
    return result


def measurements(families, arrays):
    """Recalculate reported values directly from validated original-agent events."""
    count = len(families)
    outside = np.array([family["fresh"]["outside"].sum() for family in families])
    result = dict(families=count, agents_per_family=4, assigned_agents=count * 4,
                  outside_born_agents=int(outside.sum()),
                  rule_counts={str(rule): sum(int(raw["rule"]) == rule for raw in arrays) for rule in (0, 1)})
    for arm in ARMS:
        rows = [family[arm] for family in families]
        denominator = np.array([row["outside"].sum() for row in rows])
        result[arm] = dict(
            latency=ratio_stats([np.sum(row["entry"][row["outside"]]) for row in rows], denominator),
            shelter_arrival=ratio_stats([row["reached"].sum() for row in rows], denominator),
            survival=world_stats([row["survival"].sum() / 4 for row in rows]),
            fitness=world_stats([row["fitness"] for row in rows]),
            cold_sheltered_fraction=world_stats([
                np.count_nonzero(row["sheltered"][150:] & row["after"][150:]) / 600 for row in rows]),
            outdoor_cold_agent_steps=world_stats([row["cold_exposure"].sum() for row in rows]),
            extra_energy_lost=world_stats([row["loss"].sum(dtype=np.float64) for row in rows]),
            deaths=int(sum(np.count_nonzero(~row["survival"]) for row in rows)),
            cold_exposed_agents=int(sum(row["cold_exposure"].any(axis=0).sum() for row in rows)),
            cold_exposed_agent_steps=int(sum(row["cold_exposure"].sum() for row in rows)),
            living_arrivals=int(sum(row["reached"].sum() for row in rows)),
            outside_born_agents=int(denominator.sum()))
    benefit = [np.sum((family["fresh"]["entry"] - family["inherited"]["entry"])
                       [family["fresh"]["outside"]]) for family in families]
    result["latency_improvement"] = ratio_stats(benefit, outside)
    result["survival_difference"] = world_stats([
        (np.count_nonzero(family["inherited"]["survival"]) - np.count_nonzero(family["fresh"]["survival"])) / 4
        for family in families])
    result["archive"] = dict(
        changed_entries=world_stats([np.count_nonzero(raw["archive_after"]) for raw in arrays]),
        mean_abs_update=world_stats([np.abs(raw["archive_after"]).sum(dtype=np.float64) / (6 * 129) for raw in arrays]),
        recipients_changed=int(sum(np.any(raw["recipient_initial_delta"] != 0, axis=(1, 2)).sum() for raw in arrays)))
    return result


def competing_outcomes(families):
    """Descriptive decomposition; death is never removed from the primary score."""
    report = {}
    for arm in ARMS:
        rows = [family[arm] for family in families]
        report[arm] = {
            "born_inside": int(sum(row["initial"].sum() for row in rows)),
            "outside_living_arrival": int(sum(row["reached"].sum() for row in rows)),
            "outside_died_without_arrival": int(sum(row["died_without_entry"].sum() for row in rows)),
            "outside_survived_without_arrival": int(sum(row["alive_without_entry"].sum() for row in rows)),
            "outside_arrived_before_cold": int(sum(row["arrived_before_cold"].sum() for row in rows)),
            "outside_arrived_then_died": int(sum((row["reached"] & ~row["survival"]).sum() for row in rows)),
        }
    return report


class Auditor:
    def __init__(self, root, source):
        self.root = root.resolve()
        self.hashes = {}
        self.source = source
        self.expected_trees = self.code_trees(source)
        self.used_keys = set()

    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True,
                                text=True, check=True, timeout=15)
        return result.stdout.strip()

    def code_trees(self, commit):
        return {name: self.git("rev-parse", f"{commit}:{name}") for name in ("seed", "scripts")}

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "path leaves repository: " + str(relative))
        return path

    def read(self, relative):
        path = self.path(relative)
        if not path.is_file():
            raise Incomplete("missing file: " + str(relative))
        data = path.read_bytes()
        name = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        require(name not in self.hashes or self.hashes[name] == digest, "artifact changed: " + name)
        self.hashes[name] = digest
        return data

    def json(self, relative):
        def invalid(value):
            raise AuditError("nonfinite JSON in " + str(relative))
        return json.loads(self.read(relative), parse_constant=invalid)

    def arrays(self, relative):
        with np.load(io.BytesIO(self.read(relative)), allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "duplicate arrays in " + str(relative))
            return {key: archive[key] for key in archive.files}

    def completed_bundle(self, development=False):
        names = ["E117-benchmark"] if development else [f"E117-A{i}" for i in range(3)]
        expected = 4 if development else 256
        folders, incomplete = [], []
        # Only operational status and filenames are read until the ENTIRE bundle is ready.
        for name in names:
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, "duplicate canonical attempt: " + name)
            if not candidates:
                incomplete.append(name + " missing")
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            if not self.path(folder + "/status.json").is_file():
                incomplete.append(name + " missing status")
                continue
            status = self.json(folder + "/status.json")
            if status.get("status") != "finished" or status.get("completed_families") != expected:
                incomplete.append(name + " incomplete")
                continue
            same(status.get("expected_families"), expected, name + " planned families")
            files = {path.name for path in self.path(folder + "/families").glob("*.npz")}
            if files != {f"family_{index:04d}.npz" for index in range(expected)}:
                incomplete.append(name + " incomplete archive set")
                continue
            if not self.path(folder + "/summary.json").is_file():
                incomplete.append(name + " no terminal summary")
                continue
            folders.append(folder)
        if incomplete:
            raise Incomplete("; ".join(incomplete) + "; no measurements opened")
        return folders

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        source_path, digest = ORIGINALS[index]
        name = "E117-benchmark" if development else f"E117-A{index}"
        split = "development" if development else "final"
        seed = 42 if development else 2026112101 + index
        count = 4 if development else 256
        subset(config, dict(protocol=PROTOCOL, cfg=CFG, cfg_overrides={}, ecology=ECOLOGY,
            persistence="off", plastic=True, controller="rules-v0", genome_size=155,
            condition_split=split, seed=seed, eval_seed=seed, eval_worlds=count, expected_families=count,
            cold_start=150, key_order=KEY_ORDER, max_seconds=600 if development else 9900,
            scientific_result=not development, raw_schema=PROTOCOL,
            raw_pattern="families/family_{index:04d}.npz",
            rule_assignment="family_index modulo 2; donor and recipient share rule",
            source=dict(label=f"A{index}", path=source_path, sha256=digest, training_seed=index)), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=seed, eval_worlds=count, generations=0, start="file:" + source_path,
            quick=False, allow_dirty=False, no_plastic=False, pop=128, worlds=8,
            sigma=.1, lr=.1, checkpoint_every=10), name + " arguments")
        for field in ("resume", "cfg", "objective", "memory_condition", "max_train_seconds"):
            require(not config["args"].get(field), name + " forbidden argument " + field)
        require(config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off")
                and not config["args"].get("persistence"), name + " changed mechanism")
        provenance = config["git"]
        commit = provenance.get("commit", "")
        require(provenance.get("dirty") is False and len(commit) == 40
                and all(ch in "0123456789abcdef" for ch in commit), name + " dirty/malformed source")
        require(self.code_trees(commit) == self.expected_trees, name + " unregistered scientific source")
        source = self.read(source_path)
        require(hashlib.sha256(source).hexdigest() == digest, name + " original E103 source changed")
        require(self.read(folder + "/genome_start.npy") == source, name + " input genome bytes")
        flat = np.load(io.BytesIO(source), allow_pickle=False)
        require(flat.shape == (155,) and flat.dtype == np.float32 and np.isfinite(flat).all(), name + " genome format")
        return config

    def register_keys(self, keys, label):
        tokens = [key.tobytes() for key in keys]
        require(len(tokens) == 4 and len(set(tokens)) == 4 and not self.used_keys.intersection(tokens),
                label + " reused world/development key across families/seeds/splits")
        self.used_keys.update(tokens)

    def development_keys(self):
        """Check development isolation without opening its scores or event arrays."""
        folder = self.completed_bundle(development=True)[0]
        self.config(folder, 0, True)
        for index in range(4):
            relative = folder + f"/families/family_{index:04d}.npz"
            with np.load(io.BytesIO(self.read(relative)), allow_pickle=False) as archive:
                keys = archive["keys"]
                require(keys.shape == (4, 2) and keys.dtype == np.uint32, "development key schema")
                self.register_keys(keys, relative)
        return dict(run=folder, keys_checked=16, scores_opened=False)

    def run(self, folder, index, development=False):
        config = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count = 4 if development else 256
        subset(summary, dict(status="finished", protocol=PROTOCOL, completed_families=count,
            expected_families=count, condition_split="development" if development else "final",
            generation=0, source_label=f"A{index}", eval_seed=config["seed"], eval_worlds=count,
            persistence="off", scientific_result=not development, artifact_validity="passed"), folder + " summary")
        require(0 <= summary["elapsed_seconds"] < config["max_seconds"], folder + " completed after wall cap")
        require(len(summary["raw_files"]) == count, folder + " raw manifest size")
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == count, folder + " metrics length")
        families, arrays, elapsed = [], [], 0.
        for number in range(count):
            relative = f"families/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + relative)
            artifact = dict(index=number, path=relative, sha256=self.hashes[folder + "/" + relative])
            same(summary["raw_files"][number], artifact, relative + " manifest")
            family = family_arrays(raw, number)
            self.register_keys(raw["keys"], folder + "/" + relative)
            record = json.loads(lines[number])
            subset(record, dict(family=number, rule=number % 2, keys=raw["keys"].tolist(),
                                 raw_file=artifact), relative + " record")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], relative + " elapsed order")
            elapsed = record["elapsed_seconds"]
            same(record["measurements"], measurements([family], [raw]), relative + " family measurement")
            families.append(family)
            arrays.append(raw)
        recalculated = measurements(families, arrays)
        same(summary["measurements"], recalculated, folder + " pooled measurements")
        result = dict(run=folder, source_commit=config["git"]["commit"], measurements=recalculated,
                      competing_outcomes=competing_outcomes(families), elapsed_seconds=summary["elapsed_seconds"])
        if development:
            require("gate" not in summary and summary["gate_status"] == "not_applicable", "development must not decide science")
        else:
            gate = scientific_gate(recalculated["latency_improvement"], recalculated["survival_difference"])
            same(summary["gate"], gate, folder + " scientific gate")
            same(summary["gate_status"], "passed" if gate["passed"] else "failed", folder + " gate status")
            result["gate"] = gate
        return result


def self_check():
    same(world_stats([0, 2]), dict(mean=1., se=1.), "manual SE")
    same(ratio_stats([1, 3], [2, 2]), dict(mean=1., se=.5), "manual equal-denominator ratio")
    # Zero-denominator families remain clusters; ratio differs from mean family rates.
    same(ratio_stats([0, 2, 8], [0, 1, 2]),
         dict(mean=10/3, se=math.sqrt(16/27)), "manual unequal-denominator ratio")
    same(ratio_stats([0, 0], [0, 0]), dict(mean=None, se=None), "undefined ratio")
    inside = [True, False, False, False]
    sheltered = np.array([[False, True, True, False], [False, True, True, False]])
    after = np.array([[True, False, True, True], [True, False, False, True]])
    require(np.array_equal(first_living_arrival(inside, sheltered, after), [0, 3, 1, 3]),
            "inside spawn, fatal arrival, arrival then death, and nonarrival")
    same(scientific_gate(dict(mean=2., se=1.), dict(mean=0., se=0.)),
         dict(passed=False, latency_passed=False, survival_passed=True), "strict latency boundary")
    same(scientific_gate(dict(mean=2.001, se=1.), dict(mean=-2., se=1.)),
         dict(passed=True, latency_passed=True, survival_passed=True), "inclusive survival boundary")
    require(not scientific_gate(dict(mean=1., se=0.), dict(mean=-.001, se=0.))["passed"], "survival veto")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", help="full preregistered scientific source SHA")
    parser.add_argument("--development", action="store_true", help="audit the benchmark only; no scientific verdict")
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
        require(not any(name == "seed" or name.startswith(("seed.", "jax", "scripts.")) for name in sys.modules),
                "standalone audit imported engine code")
        print(json.dumps(dict(status="self-check-passed", scientific_inputs_opened=False)))
        return 0
    if not arguments.source_commit or len(arguments.source_commit) != 40 or any(
            ch not in "0123456789abcdef" for ch in arguments.source_commit):
        parser.error("provide --source-commit with the full registered SHA")
    report = dict(status="not-audited", protocol=PROTOCOL,
                  mode="development" if arguments.development else "final",
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    auditor = None
    try:
        auditor = Auditor(arguments.root, arguments.source_commit)
        folders = auditor.completed_bundle(arguments.development)
        report["registered_source"] = auditor.source
        report["registered_code_trees"] = auditor.expected_trees
        if not arguments.development:
            report["development_isolation"] = auditor.development_keys()
        report["runs"] = [auditor.run(folder, index, arguments.development) for index, folder in enumerate(folders)]
        if not arguments.development:
            report["scientific_gate_passed"] = all(run["gate"]["passed"] for run in report["runs"])
        report.update(status="passed", final_evaluation_audited=not arguments.development, limitations=[
            "Read-only event and archive reconstruction, not an independent physics or RNG replay.",
            "Random keys are unique and paired with fixed recorded masters; key derivation is not regenerated.",
            "Whole-weight pairing uses recorded SHA256 fingerprints; only the stored sensor-to-motor overlay is reconstructed here.",
            "Latency assigns 301 to death/nonarrival and can occur before cold; it is not survivor-only travel time.",
            "A passed comparison would show transfer in this ecology, not shelter-specific content, which requires a matched archive control.",
            "Errors cluster independent families for each fixed genome, not variation across evolution seeds.",
            "Resource guards and the daily compute ledger are checked separately by the coordinator."])
        code = 0
    except Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
