"""Read-only E118 reconstruction from saved optimizer states and world events.

The audit imports NumPy and the existing read-only arithmetic/file helpers. It
does not run a simulator. Final mode requires all three panels before reading
their outcomes. Exit 0 means valid evidence, independently of the scientific gate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_cold_donor as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "shelter-evolution-v1"
PERSISTENCE = dict(version="persistence-v1", memory_steps=4, stall_steps=2)
ARMS = ("parent", "descendant")
require, same, subset, typed = base.require, base.same, base.subset, base.typed


def close(actual, expected, label):
    require(np.shape(actual) == np.shape(expected) and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=2e-5, atol=2e-6), label + " numerical mismatch")


def adam_update(previous, samples):
    """Reconstruct ES selection, rank shaping and the published Adam update."""
    eps = typed(samples, "eps", (16, 155), "f")
    candidates = typed(samples, "candidate_genomes", (16, 155), "f")
    fitness = typed(samples, "fitness", (16, 4), "f")
    require(np.array_equal(eps[8:], -eps[:8]), "perturbations must be antithetic")
    require(np.all((fitness >= 0) & (fitness <= 300)), "training shelter score bounds")
    close(candidates, previous["mu"] + np.float32(.05) * eps, "sampled genomes")
    scores = fitness.mean(1, dtype=np.float32)
    ranks = np.argsort(np.argsort(scores, kind="stable"), kind="stable").astype(np.float32)
    weights = ranks / np.float32(15) - np.float32(.5)
    grad = (weights[:, None] * eps).sum(0) / np.float32(.8) - np.float32(.005) * previous["mu"]
    generation = int(previous["t"]) + 1
    m = np.float32(.9) * previous["m"] + np.float32(.1) * grad
    v = np.float32(.999) * previous["v"] + np.float32(.001) * grad ** 2
    mu = previous["mu"] + np.float32(.02) * (m / np.float32(1 - .9 ** generation)) / (
        np.sqrt(v / np.float32(1 - .999 ** generation)) + np.float32(1e-8))
    return dict(mu=mu, m=m, v=v), scores


def pair_events(raw, index, cfg=base.CFG):
    steps, agents, grid = cfg["T"], cfg["M"], cfg["G"]
    for name, expected in (("steps", steps), ("agents", agents), ("rule", index % 2)):
        value = typed(raw, name, (), "iu")
        require(int(value) == expected, name + " differs from registered family")
    keys = typed(raw, "keys", (2, 2), "u")
    require(keys.dtype == np.uint32, "evaluation keys must be uint32")
    position = typed(raw, "initial_position", (agents, 2), "iu")
    require(np.all((position >= 0) & (position < grid)), "birth position bounds")
    result = {}
    for arm in ARMS:
        prefix = arm + "_"
        events = {name: typed(raw, prefix + name, (steps, agents), "biu", binary=True).astype(bool)
                  for name in base.EVENTS}
        food, poison = events["food"], events["poison"]
        before, after, sheltered = events["alive_before"], events["alive_after"], events["sheltered"]
        require(before[0].all() and np.array_equal(before[1:], after[:-1])
                and not np.any(after & ~before) and not np.any(food & poison)
                and not np.any((food | poison | sheltered) & ~before), prefix + "event/death timing")
        shelter = typed(raw, prefix + "shelter", (grid, grid), "biu", binary=True).astype(bool)
        require(shelter.sum() == max(1, grid // 4) ** 2, prefix + "shelter size")
        initial = typed(raw, prefix + "initial_sheltered", (agents,), "biu", binary=True).astype(bool)
        require(np.array_equal(initial, shelter[position[:, 0], position[:, 1]]), prefix + "inside births")
        arrival = base.first_living_arrival(initial, sheltered, after)
        require(np.array_equal(typed(raw, prefix + "first_entry", (agents,), "iu"), arrival),
                prefix + "first living entry")
        fitness = float((food.sum(0, dtype=np.int64) - poison.sum(0, dtype=np.int64)).mean())
        require(float(typed(raw, prefix + "fitness", (), "iuf")) == fitness, prefix + "food fitness")
        lost = typed(raw, prefix + "energy_lost", (steps, agents), "f")
        require(np.all(lost >= 0) and np.all(lost[~before] == 0), prefix + "energy loss timing")
        connections = typed(raw, prefix + "birth_connections", (agents,), "iu")
        require(np.all(connections >= 0), prefix + "connection count")
        hashes = typed(raw, prefix + "birth_sha256", (agents, 32), "u")
        require(hashes.dtype == np.uint8, prefix + "birth fingerprints")
        flags = {name: typed(raw, prefix + "persistence_" + name, (steps, agents), "biu", binary=True).astype(bool)
                 for name in ("active", "remembered", "tactic_attempted")}
        require(all(not np.any(flag & ~before) and not flag[:steps // 2].any() for flag in flags.values())
                and not np.any(flags["remembered"] & ~flags["active"])
                and not np.any(flags["tactic_attempted"] & ~flags["active"]), prefix + "persistence timing")
        outside = ~initial
        result[arm] = dict(initial=initial, arrival=arrival, latency_sum=int(arrival[outside].sum()),
            outside=int(outside.sum()), survival=float(after[-1].mean()), fitness=fitness,
            arrivals=int(((arrival <= steps) & outside).sum()),
            before_cold=int(((arrival <= steps // 2) & outside).sum()),
            cold_occupancy=float((sheltered[steps // 2:] & after[steps // 2:]).mean()),
            deaths=int((~after[-1]).sum()), connections=float(connections.mean()),
            died_before_arrival=int((outside & (arrival > steps) & ~after[-1]).sum()))
    for field in ("shelter", "initial_sheltered"):
        require(np.array_equal(raw["parent_" + field], raw["descendant_" + field]), "paired " + field)
    return result


def measurements(rows, agents=4):
    counts = np.array([row["parent"]["outside"] for row in rows])
    by_arm = {}
    for arm in ARMS:
        values = [row[arm] for row in rows]
        by_arm[arm] = dict(
            latency=base.ratio_stats([v["latency_sum"] for v in values], counts),
            survival=base.world_stats([v["survival"] for v in values]),
            fitness=base.world_stats([v["fitness"] for v in values]),
            living_arrivals=sum(v["arrivals"] for v in values),
            before_cold_arrivals=sum(v["before_cold"] for v in values),
            cold_sheltered_fraction=base.world_stats([v["cold_occupancy"] for v in values]),
            deaths=sum(v["deaths"] for v in values),
            birth_connections=base.world_stats([v["connections"] for v in values]))
    latency = base.ratio_stats([r["parent"]["latency_sum"] - r["descendant"]["latency_sum"] for r in rows], counts)
    survival = base.world_stats([r["descendant"]["survival"] - r["parent"]["survival"] for r in rows])
    return dict(families=len(rows), agents_per_family=agents, assigned_agents=len(rows) * agents,
                outside_born_agents=int(counts.sum()), rule_counts={str(i): sum(j % 2 == i for j in range(len(rows))) for i in (0, 1)},
                **by_arm, latency_improvement=latency, survival_difference=survival)


class Auditor(base.Auditor):
    def register_keys(self, keys, label):
        keys = np.asarray(keys)
        require(keys.ndim == 2 and keys.shape[1] == 2 and keys.dtype == np.uint32, label + " key shape")
        tokens = [key.tobytes() for key in keys]
        require(len(set(tokens)) == len(tokens) and not self.used_keys.intersection(tokens), label + " reused keys")
        self.used_keys.update(tokens)

    def completed_bundle(self, development=False):
        folders, pending = [], []
        count, generations = (4, 1) if development else (128, 10)
        names = ["E118-benchmark"] if development else [f"E118-A{i}" for i in range(3)]
        for name in names:
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, name + " duplicate canonical attempts")
            if not candidates or not (candidates[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            expected = dict(status="finished", generation=generations, expected_generations=generations,
                            completed_families=count, expected_families=count)
            if any(status.get(k) != v for k, v in expected.items()):
                pending.append(name)
                continue
            files = {p.name for p in self.path(folder + "/families").glob("*.npz")}
            checkpoints = {p.name for p in self.path(folder + "/generations").glob("*.npz")}
            if (files != {f"family_{j:04d}.npz" for j in range(count)}
                    or checkpoints != {f"generation_{j:04d}.npz" for j in range(generations + 1)}
                    | {f"samples_{j:04d}.npz" for j in range(1, generations + 1)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
                continue
            folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; outcomes remain unopened")
        return folders

    def manifest(self, folder, relative, saved):
        arrays = self.arrays(folder + "/" + relative)
        same(saved, dict(path=relative, sha256=self.hashes[folder + "/" + relative]), relative + " manifest")
        return arrays

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, generations, training, evaluation = (4, 1, 43, 44) if development else (128, 10, 2026112201 + index, 2026112301 + index)
        name = "E118-benchmark" if development else f"E118-A{index}"
        split = "development" if development else "final"
        source_path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, cfg=base.CFG, cfg_overrides={}, ecology=base.ECOLOGY,
            persistence=PERSISTENCE, plastic=True, controller="rules-v0", genome_size=155,
            condition_split=split, seed=training, evaluation_seed=evaluation, eval_worlds=count,
            expected_families=count, expected_generations=generations, l2=.005,
            selection="fixed final mean after registered generations", inheritance="fresh lifetime weights at each birth",
            max_seconds=900 if development else 3600, scientific_result=not development), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=training, generations=generations, eval_worlds=count, pop=16, worlds=4, sigma=.05, lr=.02,
            start="file:" + source_path, checkpoint_every=10, no_plastic=False, allow_dirty=False,
            quick=False, resume=None, cfg=None, objective=None, memory_condition=None,
            max_train_seconds=None), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), name + " mechanism flags")
        subset(config["source"], dict(label=f"A{index}", training_seed=index, path=source_path, sha256=digest), name + " original")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40, name + " clean source")
        require(self.code_trees(provenance["commit"]) == self.expected_trees, name + " scientific source trees")
        original = self.read(source_path)
        require(hashlib.sha256(original).hexdigest() == digest
                and self.read(folder + "/genome_start.npy") == original, name + " original genome bytes")
        parent = np.load(io.BytesIO(original), allow_pickle=False)
        require(parent.shape == (155,) and parent.dtype == np.float32 and np.isfinite(parent).all(), name + " parent")
        return config, parent

    def state(self, arrays, generation):
        require(set(arrays) == {"mu", "m", "v", "t", "key"}, "optimizer state fields")
        for name in ("mu", "m", "v"):
            require(typed(arrays, name, (155,), "f").dtype == np.float32, name + " must be float32")
        require(np.all(arrays["v"] >= 0), "negative Adam second moment")
        require(int(typed(arrays, "t", (), "iu")) == generation, "generation mismatch")
        require(typed(arrays, "key", (2,), "u").dtype == np.uint32, "state RNG dtype")
        return arrays

    def evolution(self, folder, config, summary, parent):
        generations = config["expected_generations"]
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == generations and len(summary["generations"]) == generations + 1, "generation accounting")
        trajectory = np.load(io.BytesIO(self.read(folder + "/genome_trajectory.npy")), allow_pickle=False)
        require(trajectory.shape == (generations + 1, 155) and trajectory.dtype == np.float32, "trajectory shape/dtype")
        previous = self.state(self.manifest(folder, "generations/generation_0000.npz", summary["generations"][0]), 0)
        require(np.array_equal(previous["mu"], parent) and not previous["m"].any() and not previous["v"].any(), "fresh original optimizer")
        require(np.array_equal(trajectory[0], parent), "trajectory original")
        elapsed, records, state_keys = 0., [], {previous["key"].tobytes()}
        for generation, line in enumerate(lines, 1):
            row = json.loads(line)
            same(row["gen"], generation - 1, "zero-based generation")
            same(row["generation"], generation, "generation")
            same(row["checkpoint"], summary["generations"][generation], "checkpoint link")
            current = self.state(self.manifest(folder, f"generations/generation_{generation:04d}.npz", row["checkpoint"]), generation)
            require(np.array_equal(trajectory[generation], current["mu"]), "trajectory/checkpoint identity")
            require(current["key"].tobytes() not in state_keys, "reused optimizer key")
            state_keys.add(current["key"].tobytes())
            samples = self.manifest(folder, f"generations/samples_{generation:04d}.npz", row["samples"])
            world_keys = typed(samples, "world_keys", (4, 2), "u")
            development_keys = typed(samples, "development_keys", (16, 4, 2), "u")
            require(world_keys.dtype == development_keys.dtype == np.uint32
                    and np.array_equal(development_keys[:8], development_keys[8:]), "antithetic development pairing")
            self.register_keys(np.concatenate((world_keys, development_keys[:8].reshape(-1, 2))), "training sampling")
            expected, scores = adam_update(previous, samples)
            for name, value in expected.items():
                close(current[name], value, "generation " + str(generation) + " " + name)
            for name, value in (("fit_mean", scores.mean()), ("fit_best", scores.max()), ("fit_std", scores.std())):
                close(row[name], value, "training " + name)
            for name in ("acc_first", "acc_second", "survived"):
                require(np.isfinite(row[name]) and 0 <= row[name] <= 1, name + " bounds")
            changed = int(np.count_nonzero(current["mu"] != previous["mu"]))
            require(changed > 0 and row["changed_parameters"] == changed, "genome did not change as recorded")
            close(row["l2_distance_from_parent"], float(np.linalg.norm(current["mu"] - parent)), "distance from parent")
            require(elapsed <= row["elapsed_seconds"] <= summary["elapsed_seconds"], "training elapsed order")
            elapsed = row["elapsed_seconds"]
            records.append(dict(generation=generation, changed_parameters=changed,
                genome_sha256=hashlib.sha256(current["mu"].tobytes()).hexdigest(),
                maximum_adam_roundoff=max(float(np.max(np.abs(current[k] - v))) for k, v in expected.items())))
            previous = current
        final_bytes = self.read(folder + "/genome_final.npy")
        final = np.load(io.BytesIO(final_bytes), allow_pickle=False)
        require(final.dtype == np.float32 and np.array_equal(final, previous["mu"]), "terminal genome/checkpoint identity")
        same(summary["changed_parameters"], int(np.count_nonzero(final != parent)), "final changed parameters")
        close(summary["l2_distance_from_parent"], float(np.linalg.norm(final - parent)), "final genome displacement")
        return dict(transitions=records, parent_sha256=hashlib.sha256(parent.tobytes()).hexdigest(),
                    final_file_sha256=hashlib.sha256(final_bytes).hexdigest(), last_training_elapsed=elapsed)

    def run(self, folder, index, development=False):
        config, parent = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count, generations = config["expected_families"], config["expected_generations"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=generations, expected_generations=generations, completed_families=count, expected_families=count,
            source_label=f"A{index}", training_seed=config["seed"], evaluation_seed=config["evaluation_seed"],
            artifact_validity="passed", scientific_result=not development), folder + " summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "process time allowance")
        training = self.evolution(folder, config, summary, parent)
        lines = self.read(folder + "/evaluation.jsonl").decode("utf-8").splitlines()
        require(len(lines) == len(summary["raw_files"]) == count, "evaluation manifest length")
        rows, elapsed = [], training["last_training_elapsed"]
        for index_in_run, line in enumerate(lines):
            record = json.loads(line)
            relative = f"families/family_{index_in_run:04d}.npz"
            raw = self.manifest(folder, relative, summary["raw_files"][index_in_run])
            rows.append(pair_events(raw, index_in_run))
            self.register_keys(raw["keys"], relative)
            subset(record, dict(family=index_in_run, rule=index_in_run % 2, keys=raw["keys"].tolist(),
                                raw_file=summary["raw_files"][index_in_run]), "evaluation log")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "evaluation elapsed order")
            elapsed = record["elapsed_seconds"]
        estimated = measurements(rows)
        same(summary["measurements"], estimated, "reconstructed evaluation")
        report = dict(run=folder, source_commit=config["git"]["commit"], training=training, measurements=estimated,
            competing_outcomes={arm: dict(born_inside=count * 4 - estimated["outside_born_agents"],
                outside_died_before_arrival=sum(r[arm]["died_before_arrival"] for r in rows)) for arm in ARMS})
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "development has no scientific gate")
        else:
            gate = base.scientific_gate(estimated["latency_improvement"], estimated["survival_difference"])
            same(summary["gate"], gate, "scientific gate")
            same(summary["gate_status"], "passed" if gate["passed"] else "failed", "gate status")
            report["gate"] = gate
        return report


def self_check():
    base.self_check()
    previous = dict(mu=np.zeros(155, np.float32), m=np.zeros(155, np.float32),
                    v=np.zeros(155, np.float32), t=np.array(0), key=np.array([0, 1], np.uint32))
    noise = np.tile(np.arange(1, 9, dtype=np.float32)[:, None], (1, 155)) / 8
    eps = np.concatenate((noise, -noise))
    samples = dict(eps=eps, candidate_genomes=.05 * eps,
                   fitness=np.repeat(np.arange(16, dtype=np.float32)[:, None], 4, axis=1))
    update, scores = adam_update(previous, samples)
    require(np.all(update["mu"] < 0) and np.all(update["v"] >= 0), "selection follows higher-ranked negative perturbations")
    close(scores, np.arange(16, dtype=np.float32), "candidate scores")
    for field in ("eps", "candidate_genomes", "fitness"):
        broken = {k: v.copy() for k, v in samples.items()}
        broken[field][0, 0] += 500
        try:
            adam_update(previous, broken)
        except base.AuditError:
            continue
        raise base.AuditError("failed to reject corrupted " + field)
    cfg = dict(G=4, M=2, T=4)
    raw = dict(keys=np.array([[11, 12], [13, 14]], np.uint32), rule=np.array(0),
               steps=np.array(4), agents=np.array(2), initial_position=np.array([[0, 0], [2, 2]]))
    for arm in ARMS:
        prefix = arm + "_"
        after = np.ones((4, 2), bool)
        after[-1, 1] = False
        sheltered = np.zeros((4, 2), bool)
        sheltered[2 if arm == "parent" else 3, 1] = True
        shelter = np.zeros((4, 4), bool)
        shelter[0, 0] = True
        values = dict(food=np.zeros((4, 2), bool), poison=np.zeros((4, 2), bool),
                      alive_before=np.ones((4, 2), bool), alive_after=after, sheltered=sheltered,
                      shelter=shelter, initial_sheltered=np.array([True, False]),
                      first_entry=np.array([0, 3 if arm == "parent" else 5]), fitness=np.array(0.),
                      energy_lost=np.zeros((4, 2)), birth_connections=np.array([1, 2]),
                      birth_sha256=np.zeros((2, 32), np.uint8))
        for name in ("active", "remembered", "tactic_attempted"):
            values["persistence_" + name] = np.zeros((4, 2), bool)
        raw.update({prefix + name: value for name, value in values.items()})
    row = pair_events(raw, 0, cfg)
    estimate = measurements([row, row], agents=2)
    same(estimate["latency_improvement"], dict(mean=-2., se=0.), "fatal arrival is nonarrival")
    same(estimate["outside_born_agents"], 2, "inside birth excluded")
    for name in ("parent_first_entry", "descendant_alive_after"):
        broken = {k: v.copy() for k, v in raw.items()}
        broken[name].flat[-1] = 1
        try:
            pair_events(broken, 0, cfg)
        except base.AuditError:
            continue
        raise base.AuditError("failed to reject corrupted " + name)
    audit = Auditor.__new__(Auditor)
    audit.used_keys = set()
    audit.register_keys(raw["keys"], "synthetic")
    try:
        audit.register_keys(raw["keys"], "synthetic repeat")
    except base.AuditError:
        return
    raise base.AuditError("failed to reject reused keys")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print(json.dumps(dict(status="self-check-passed", scientific_inputs_opened=False)))
        return 0
    if args.source_commit is None:
        parser.error("--source-commit is required")
    auditor = None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            benchmark = auditor.completed_bundle(True)[0]
            report["development_audit"] = auditor.run(benchmark, 0, True)
        report["runs"] = [auditor.run(folder, i, args.development) for i, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit,
            registered_code_trees=auditor.expected_trees, final_evaluation_audited=not args.development,
            limitations=["Physics and random-key derivation are not replayed; saved inputs, outputs and code provenance are checked.",
                         "Adam is reconstructed with float32 roundoff tolerance of 2e-5 relative and 2e-6 absolute.",
                         "Three fixed parent lineages and short continuations bound the performance conclusion.",
                         "Birth-to-shelter arrival can precede cold, and survival after arrival is separately reported."])
        if not args.development:
            report["scientific_gate_passed"] = all(row["gate"]["passed"] for row in report["runs"])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
