"""Read-only E119 audit of continued ES, inherited archives and paired outcomes.

NumPy arithmetic reconstructs saved training updates and outcome estimates.
Final mode opens outcomes only after all three canonical panels are complete.
Exit zero certifies the evidence independently of the scientific gate.
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
import audit_shelter_evolution as previous


base = previous.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "inherited-evolution-v1"
SOURCE_COMMIT = "cd08b59c02994c048533a539ae65d580369bb29c"
ARMS = ("founder_empty", "parent_previous", "parent_latest", "child_previous", "child_latest", "child_empty")
CONTRASTS = {
    "child_vs_parent": ("parent_previous", "child_latest"),
    "child_vs_founder": ("founder_empty", "child_latest"),
    "inherited_vs_empty": ("child_empty", "child_latest"),
    "genotype_same_memory": ("parent_latest", "child_latest"),
    "latest_lessons": ("child_previous", "child_latest"),
    "parent_lessons": ("parent_previous", "parent_latest"),
}
SOURCES = (
    ("runs/20261121-182049_E118-benchmark", 1,
     "0406de7317b3bf89a580f6803fa207eede85f091176fd423e443961e373867ad",
     "9c8fa76b3b2b3da22707199cf00988621d873f5901dcd45c7115354b87286cde",
     "fe073e1d9c840cc5e460f657338b705ae584182177a04184bceada035e02f998",
     "e6432ea822ea4d59682f729e741c73e1100ccc8aa1bca281e52298ee46728954"),
    ("runs/20261121-182237_E118-A0", 10,
     "c56d774ddb2130bfb9fc07a5f143ab29ae701ffbecf4d7130128881a5470f3e4",
     "2e1bea556060085e34c2dc7cf027f639d3f36b4ee592c64b1135e8c09907a4d2",
     "7d7645e791440e6603d9f35cf043abf17701e597ad6f2592b7b76430de6b597f",
     "110910637b4b0476bc8ef6bf65153e515f033dd1e602dff911ffb1b025a86fe1"),
    ("runs/20261121-182511_E118-A1", 10,
     "4d2f766a5435b809165629b47fd972b3a1160a9ee2fa97c13b214a6f0712c807",
     "1984cacb45e87a3fdc29bb03bb83a424873f333619008086ca98b0c768a2e78e",
     "038e3ca3308f213476a1c09bf565978f76e72e9c170d29b9bc507eea5e2e9bcb",
     "d5f57b44e785789098004a43ab6b6c1ce7e92bb32bc199e21a1fcd046bf1b254"),
    ("runs/20261121-182806_E118-A2", 10,
     "88592fec7002c64bb51df24b31ba2cc99a3e611397eef3ebfd78e807d5ad203e",
     "d0c8ce66eba91dba04151b70347ba7eae084ee9e35616ed2dd43d0a81883a740",
     "8b4f2496ec8d8f1a240fc6a76572d42655be4496646e19b30cb4329940d69ce1",
     "8b267b339284fbd33cb52eeb36921bee7a7b1a54b7c6d0c376ee7091f67d3f06"),
)


def block_shape(cfg):
    return 6, 5 * (2 * cfg["R"] + 1) ** 2 + 4


def overlay(fresh, mask, archive, bound=3.):
    """Reconstruct the registered float32, masked, clipped inherited weights."""
    require(fresh.dtype == archive.dtype == np.float32 and mask.shape == fresh.shape,
            "inheritance block shape/dtype")
    require(np.isfinite(fresh).all() and np.isfinite(archive).all()
            and np.all(np.abs(fresh) <= bound) and np.all(np.abs(archive) <= bound)
            and np.all(fresh[~mask] == 0), "invalid inherited baseline")
    delta = archive * mask
    initialized = fresh.copy()
    np.copyto(initialized, np.clip(initialized + delta, -bound, bound), where=delta != 0)
    return initialized


def projection(raw, arm):
    result = {name: raw[name] for name in ("keys", "steps", "agents", "rule", "initial_position")}
    for name, value in raw.items():
        if name.startswith(arm + "_"):
            suffix = name[len(arm) + 1:]
            result["parent_" + suffix] = value
            result["descendant_" + suffix] = value
    return result


def initialized_block(raw, arm, archive, cfg):
    shape = (cfg["M"], *block_shape(cfg))
    fresh = typed(raw, arm + "_fresh_sensor_weights", shape, "f")
    initial = typed(raw, arm + "_initial_sensor_weights", shape, "f")
    mask = typed(raw, arm + "_sensor_mask", shape, "biu", binary=True).astype(bool)
    expected = overlay(fresh, mask, archive, cfg["wmax"])
    require(np.array_equal(initial, expected), arm + " archive was not applied to actual initialized weights")
    for field in ("fresh_sha256", "initial_sha256"):
        require(typed(raw, arm + "_" + field, (cfg["M"], 32), "u").dtype == np.uint8,
                arm + " weight fingerprint dtype")
    if not archive.any():
        require(np.array_equal(raw[arm + "_fresh_sha256"], raw[arm + "_initial_sha256"]),
                arm + " empty archive changes actual weights")
    return int(np.count_nonzero(np.any(initial != fresh, axis=(-2, -1))))


def donor_archive(raw, genome, old_archive, round_index, cfg=base.CFG):
    worlds, agents, steps = 4, cfg["M"], cfg["T"]
    shape = block_shape(cfg)
    for name, value in (("round_index", round_index), ("worlds", worlds), ("agents", agents), ("steps", steps)):
        require(int(typed(raw, name, (), "iu")) == value, "donor " + name)
    require(float(typed(raw, "weight_bound", (), "f")) == cfg["wmax"], "donor weight bound")
    require(np.array_equal(typed(raw, "genome", (155,), "f"), genome), "donors used a different parent genome")
    before = typed(raw, "archive_before", shape, "f")
    after = typed(raw, "archive_after", shape, "f")
    require(np.array_equal(before, old_archive), "donor archive continuity")
    keys = typed(raw, "keys", (worlds, 2, 2), "u")
    require(keys.dtype == np.uint32, "donor key dtype")
    require(np.array_equal(typed(raw, "rules", (worlds,), "iu"), np.arange(worlds) % 2), "donor food rules")
    mask = typed(raw, "donor_mask", (worlds, agents, *shape), "biu", binary=True).astype(bool)
    delta = typed(raw, "donor_delta", mask.shape, "f")
    counts = typed(raw, "donor_counts", shape, "iuf")
    require(np.array_equal(counts, mask.sum(axis=(0, 1))) and np.all(delta[~mask] == 0),
            "archive averaging must include all present donor edges")
    initial = typed(raw, "donor_initial_sensor_weights", mask.shape, "f")
    terminal = typed(raw, "donor_terminal_sensor_weights", mask.shape, "f")
    require(np.array_equal(delta, terminal - initial), "archive delta is not terminal minus actual initial weights")
    require(np.all(np.abs(terminal) <= cfg["wmax"]), "donor terminal weights exceed bound")
    expected = np.clip(before + delta.sum(axis=(0, 1)) / np.maximum(counts, 1).astype(np.float32),
                       -cfg["wmax"], cfg["wmax"])
    require(np.array_equal(after, expected), "accumulated archive differs from all-donor average")
    changed, deaths, exposures = 0, 0, 0
    for index in range(worlds):
        single = {name: np.asarray(value)[index] for name, value in raw.items() if name.startswith("donor_")
                  and name not in ("donor_counts",)}
        single.update(keys=keys[index], steps=np.array(steps), agents=np.array(agents),
                      rule=np.array(index % 2), initial_position=raw["initial_positions"][index])
        event = previous.pair_events(projection(single, "donor"), index, cfg)["parent"]
        changed += initialized_block(single, "donor", before, cfg)
        require(np.array_equal(single["donor_sensor_mask"], mask[index]), "donor mask disagreement")
        deaths += event["deaths"]
        exposures += int((single["donor_alive_before"][steps // 2:].astype(bool)
                          & ~single["donor_sheltered"][steps // 2:].astype(bool)).sum())
    return dict(donors=worlds * agents, donor_deaths=deaths, donor_cold_exposure_steps=exposures,
                inherited_donors=changed, changed_archive_entries=int(np.count_nonzero(after != before)))


def family_events(raw, index, inputs, cfg=base.CFG):
    rows = {}
    for arm in ARMS:
        genome, archive = inputs[arm]
        require(np.array_equal(typed(raw, arm + "_genome", (155,), "f"), genome), arm + " genotype")
        require(np.array_equal(typed(raw, arm + "_archive", block_shape(cfg), "f"), archive), arm + " archive")
        rows[arm] = previous.pair_events(projection(raw, arm), index, cfg)["parent"]
        initialized_block(raw, arm, archive, cfg)
        for field in ("shelter", "initial_sheltered"):
            require(np.array_equal(raw[arm + "_" + field], raw["founder_empty_" + field]), "six-arm paired " + field)
    for group in (("parent_previous", "parent_latest"), ("child_previous", "child_latest", "child_empty")):
        for arm in group[1:]:
            for field in ("birth_sha256", "fresh_sha256", "fresh_sensor_weights", "sensor_mask"):
                require(np.array_equal(raw[group[0] + "_" + field], raw[arm + "_" + field]), "same-genome pairing " + field)
    return rows


def measurements(rows, agents=4):
    first_pair = [dict(parent=row["founder_empty"], descendant=row["child_latest"]) for row in rows]
    info = previous.measurements(first_pair, agents)
    result = {name: info[name] for name in ("families", "agents_per_family", "assigned_agents",
                                           "outside_born_agents", "rule_counts")}
    for arm in ARMS:
        paired = [dict(parent=row[arm], descendant=row[arm]) for row in rows]
        result[arm] = previous.measurements(paired, agents)["parent"]
    result["contrasts"] = {}
    for name, (parent, child) in CONTRASTS.items():
        stats = previous.measurements([dict(parent=row[parent], descendant=row[child]) for row in rows], agents)
        latency, survival = stats["latency_improvement"], stats["survival_difference"]
        result["contrasts"][name] = dict(latency_improvement=latency, survival_difference=survival,
                                         gate=base.scientific_gate(latency, survival))
    return result


class Auditor(previous.Auditor):
    def completed_bundle(self, development=False):
        folders, pending = [], []
        count, generations = (4, 1) if development else (128, 10)
        names = ["E119-benchmark"] if development else [f"E119-A{i}" for i in range(3)]
        for name in names:
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, name + " duplicate canonical attempt")
            if not candidates or not (candidates[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            expected = dict(status="finished", generation=generations, expected_generations=generations,
                            completed_families=count, expected_families=count)
            files = {p.name for p in self.path(folder + "/families").glob("*.npz")}
            states = {p.name for p in self.path(folder + "/generations").glob("*.npz")}
            donors = {p.name for p in self.path(folder + "/donors").glob("*.npz")}
            if (any(status.get(k) != v for k, v in expected.items())
                    or files != {f"family_{i:04d}.npz" for i in range(count)}
                    or states != {f"generation_{i:04d}.npz" for i in range(generations + 1)}
                    | {f"archive_{i:04d}.npz" for i in range(generations + 1)}
                    | {f"samples_{i:04d}.npz" for i in range(1, generations + 1)}
                    or donors != {f"donor_{i:04d}.npz" for i in range(1, generations + 1)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
                continue
            folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; outcomes remain unopened")
        return folders

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, generations, donor, evaluation = (4, 1, 45, 46) if development else (128, 10, 2026112401 + index, 2026112501 + index)
        name = "E119-benchmark" if development else f"E119-A{index}"
        split = "development" if development else "final"
        run, initial_generation, genome_hash, checkpoint_hash, config_hash, status_hash = SOURCES[0 if development else index + 1]
        genome_path = run + "/genome_final.npy"
        checkpoint_path = run + f"/generations/generation_{initial_generation:04d}.npz"
        subset(config, dict(protocol=PROTOCOL, cfg=base.CFG, cfg_overrides={}, ecology=base.ECOLOGY,
            persistence=previous.PERSISTENCE, plastic=True, controller="rules-v0", genome_size=155,
            condition_split=split, seed=donor, donor_seed=donor, evaluation_seed=evaluation, eval_worlds=count,
            expected_families=count, expected_generations=generations, initial_generation=initial_generation,
            l2=.005, selection="fixed final mean after registered additional generations",
            inheritance="cumulative masked sensor-to-output donor deltas before every ES update",
            evaluation_arms=list(ARMS), max_seconds=900 if development else 3600,
            scientific_result=not development), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=donor, generations=generations, eval_worlds=count, pop=16, worlds=4, sigma=.05, lr=.02,
            start="file:" + genome_path, checkpoint_every=10, no_plastic=False, allow_dirty=False,
            quick=False, resume=None, cfg=None, objective=None, memory_condition=None,
            max_train_seconds=None), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), name + " mechanism flags")
        subset(config["source"], dict(label=f"A{index}", run=run, initial_generation=initial_generation,
            path=genome_path, sha256=genome_hash, checkpoint=checkpoint_path,
            checkpoint_sha256=checkpoint_hash, config_sha256=config_hash, status_sha256=status_hash,
            commit=SOURCE_COMMIT), name + " source")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40,
                name + " clean source")
        require(self.code_trees(provenance["commit"]) == self.expected_trees, name + " source trees")
        for path, expected in ((genome_path, genome_hash), (checkpoint_path, checkpoint_hash),
                               (run + "/config.json", config_hash), (run + "/status.json", status_hash)):
            require(hashlib.sha256(self.read(path)).hexdigest() == expected, path + " source fingerprint")
        source_config = self.json(run + "/config.json")
        require(source_config["git"]["commit"] == SOURCE_COMMIT and source_config["git"]["dirty"] is False,
                "source E118 provenance")
        require(self.read(folder + "/checkpoint_start.npz") == self.read(checkpoint_path), "copied source checkpoint")
        require(self.read(folder + "/genome_start.npy") == self.read(genome_path), "copied source genome")
        state = self.state(self.arrays(checkpoint_path), initial_generation)
        founder = np.load(io.BytesIO(self.read(genome_path)), allow_pickle=False)
        require(np.array_equal(state["mu"], founder), "source mean differs from final genome")
        # Prior worlds must remain separate even when the optimizer RNG continues.
        for step in range(1, initial_generation + 1):
            samples = self.arrays(run + f"/generations/samples_{step:04d}.npz")
            self.register_keys(samples["world_keys"], "source training worlds")
            self.register_keys(samples["development_keys"][:8].reshape(-1, 2), "source training births")
        for line in self.read(run + "/evaluation.jsonl").decode("utf-8").splitlines():
            self.register_keys(np.asarray(json.loads(line)["keys"], np.uint32), "source held-out keys")
        return config, state

    def archive(self, folder, index, artifact):
        arrays = self.manifest(folder, f"generations/archive_{index:04d}.npz", artifact)
        require(set(arrays) == {"archive"}, "archive checkpoint fields")
        archive = typed(arrays, "archive", block_shape(base.CFG), "f")
        require(archive.dtype == np.float32 and np.all(np.abs(archive) <= 3), "archive bounds/dtype")
        return archive

    def evolution(self, folder, config, summary, original):
        generations, start = config["expected_generations"], config["initial_generation"]
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == len(summary["donors"]) == generations
                and len(summary["generations"]) == len(summary["archives"]) == generations + 1,
                "generation/archival accounting")
        trajectory = np.load(io.BytesIO(self.read(folder + "/genome_trajectory.npy")), allow_pickle=False)
        memories = np.load(io.BytesIO(self.read(folder + "/archive_trajectory.npy")), allow_pickle=False)
        require(trajectory.shape == (generations + 1, 155) and trajectory.dtype == np.float32, "genome trajectory")
        require(memories.shape == (generations + 1, *block_shape(base.CFG)) and memories.dtype == np.float32,
                "archive trajectory")
        state = self.state(self.manifest(folder, "generations/generation_0000.npz", summary["generations"][0]), start)
        require(all(np.array_equal(state[name], original[name]) for name in original),
                "continuation changed an E118 optimizer value")
        archive = self.archive(folder, 0, summary["archives"][0])
        require(not archive.any() and np.array_equal(memories[0], archive), "starting archive must be empty")
        require(np.array_equal(trajectory[0], original["mu"]), "founder trajectory")
        elapsed, records, state_keys = 0., [], {state["key"].tobytes()}
        for generation, line in enumerate(lines, 1):
            row = json.loads(line)
            same(row["gen"], start + generation - 1, "continued optimizer generation")
            same(row["generation"], generation, "additional generation")
            same(row["checkpoint"], summary["generations"][generation], "state manifest link")
            same(row["archive"], summary["archives"][generation], "archive manifest link")
            same(row["donor"], summary["donors"][generation - 1], "donor manifest link")
            donor = self.manifest(folder, f"donors/donor_{generation:04d}.npz", row["donor"])
            inherit = donor_archive(donor, state["mu"], archive, generation - 1)
            self.register_keys(donor["keys"].reshape(-1, 2), "donor worlds and births")
            newest = self.archive(folder, generation, row["archive"])
            require(np.array_equal(newest, donor["archive_after"]) and np.array_equal(memories[generation], newest),
                    "archive checkpoint/trajectory differs from donor result")
            samples = self.manifest(folder, f"generations/samples_{generation:04d}.npz", row["samples"])
            require(np.array_equal(typed(samples, "archive", block_shape(base.CFG), "f"), newest),
                    "candidates used a different archive")
            shape = (16, 4, 4, *block_shape(base.CFG))
            birth = typed(samples, "candidate_birth", shape, "f")
            initial = typed(samples, "candidate_initial", shape, "f")
            mask = typed(samples, "candidate_mask", shape, "biu", binary=True).astype(bool)
            require(np.array_equal(initial, overlay(birth, mask, newest)), "candidate archive transmission")
            inherit["candidate_births_changed"] = int(np.count_nonzero(np.any(initial != birth, axis=(-2, -1))))
            world_keys = typed(samples, "world_keys", (4, 2), "u")
            development_keys = typed(samples, "development_keys", (16, 4, 2), "u")
            require(world_keys.dtype == development_keys.dtype == np.uint32
                    and np.array_equal(development_keys[:8], development_keys[8:]), "antithetic birth pairing")
            self.register_keys(np.concatenate((world_keys, development_keys[:8].reshape(-1, 2))), "training sampling")
            current = self.state(self.manifest(folder, f"generations/generation_{generation:04d}.npz", row["checkpoint"]),
                                 start + generation)
            require(np.array_equal(trajectory[generation], current["mu"]), "genome trajectory/checkpoint identity")
            require(current["key"].tobytes() not in state_keys, "optimizer key reuse")
            state_keys.add(current["key"].tobytes())
            expected, scores = previous.adam_update(state, samples)
            for name, value in expected.items():
                previous.close(current[name], value, "optimizer " + name)
            for name, value in (("fit_mean", scores.mean()), ("fit_best", scores.max()), ("fit_std", scores.std())):
                previous.close(row[name], value, "training " + name)
            for name in ("acc_first", "acc_second", "survived"):
                require(np.isfinite(row[name]) and 0 <= row[name] <= 1, name + " bounds")
            changed = int(np.count_nonzero(current["mu"] != state["mu"]))
            require(changed > 0 and row["changed_parameters"] == changed, "recorded genome change")
            same(row["changed_archive_entries"], inherit["changed_archive_entries"], "recorded archive change")
            previous.close(row["l2_distance_from_founder"], float(np.linalg.norm(current["mu"] - original["mu"])),
                           "genome distance from founder")
            require(elapsed <= row["elapsed_seconds"] <= summary["elapsed_seconds"], "training time order")
            elapsed = row["elapsed_seconds"]
            records.append(dict(generation=generation, optimizer_generation=start + generation,
                changed_parameters=changed, **inherit,
                maximum_adam_roundoff=max(float(np.max(np.abs(current[k] - v))) for k, v in expected.items())))
            parent, old_archive = state["mu"].copy(), archive.copy()
            state, archive = current, newest
        final = np.load(io.BytesIO(self.read(folder + "/genome_final.npy")), allow_pickle=False)
        require(final.dtype == np.float32 and np.array_equal(final, state["mu"]), "final genome identity")
        same(summary["changed_parameters"], int(np.count_nonzero(final != original["mu"])), "final changed parameters")
        previous.close(summary["l2_distance_from_founder"], float(np.linalg.norm(final - original["mu"])), "final displacement")
        zero = np.zeros_like(archive)
        inputs = dict(founder_empty=(original["mu"], zero), parent_previous=(parent, old_archive),
                      parent_latest=(parent, archive), child_previous=(final, old_archive),
                      child_latest=(final, archive), child_empty=(final, zero))
        return dict(transitions=records, last_training_elapsed=elapsed), inputs

    def run(self, folder, index, development=False):
        config, original = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count, generations = config["expected_families"], config["expected_generations"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=generations, optimizer_generation=config["initial_generation"] + generations,
            expected_generations=generations, completed_families=count, expected_families=count,
            source_label=f"A{index}", donor_seed=config["seed"], evaluation_seed=config["evaluation_seed"],
            artifact_validity="passed", scientific_result=not development), folder + " summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "process time allowance")
        training, inputs = self.evolution(folder, config, summary, original)
        lines = self.read(folder + "/evaluation.jsonl").decode("utf-8").splitlines()
        require(len(lines) == len(summary["raw_files"]) == count, "evaluation manifest length")
        rows, elapsed = [], training["last_training_elapsed"]
        for number, line in enumerate(lines):
            record = json.loads(line)
            relative = f"families/family_{number:04d}.npz"
            raw = self.manifest(folder, relative, summary["raw_files"][number])
            rows.append(family_events(raw, number, inputs))
            self.register_keys(raw["keys"], "held-out family")
            subset(record, dict(family=number, rule=number % 2, keys=raw["keys"].tolist(),
                                raw_file=summary["raw_files"][number]), "evaluation log")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "evaluation time order")
            elapsed = record["elapsed_seconds"]
        estimated = measurements(rows)
        same(summary["measurements"], estimated, "reconstructed evaluation")
        report = dict(run=folder, source_commit=config["git"]["commit"], source=config["source"],
                      training=training, measurements=estimated)
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary,
                    "development must not decide the scientific gate")
        else:
            gate = estimated["contrasts"]["child_vs_parent"]["gate"]
            same(summary["gate"], gate, "primary scientific gate")
            same(summary["gate_status"], "passed" if gate["passed"] else "failed", "scientific gate status")
            report["gate"] = gate
        return report


def self_check():
    previous.self_check()
    fresh = np.array([[[2.9, -0., 0.]]], np.float32)
    mask = np.array([[[True, True, False]]])
    archive = np.array([[.5, 0., .5]], np.float32)
    initialized = overlay(fresh, mask, archive)
    require(np.array_equal(initialized, np.array([[[3., -0., 0.]]], np.float32))
            and np.signbit(initialized[0, 0, 1]), "clipping, absent edges and signed-zero preservation")
    broken = fresh.copy()
    broken[0, 0, 2] = 1
    try:
        overlay(broken, mask, archive)
    except base.AuditError:
        return
    raise base.AuditError("accepted a weight on an absent edge")


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
            limitations=["Physics and random-key derivation are not replayed. Saved inputs and events are reconstructed.",
                         "Donor terminal arrays use the engine's last-live state. Every recorded donor contributes by edge presence.",
                         "Adam reconstruction permits float32 roundoff of 2e-5 relative and 2e-6 absolute.",
                         "The immediate-parent primary changes both genotype and archive; the other five contrasts locate contributions.",
                         "Three lineages and ten additional updates bound the performance conclusion.",
                         "Birth-to-shelter arrival can precede cold. Survival after arrival remains in the final outcome."])
        if not args.development:
            report["scientific_gate_passed"] = all(run["gate"]["passed"] for run in report["runs"])
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
