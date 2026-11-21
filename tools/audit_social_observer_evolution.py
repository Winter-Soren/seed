"""Read-only E122 observer selection audit; NumPy only, no simulator replay."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_learning as social
from audit_shelter_evolution import close

base = social.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
PROTOCOL = "social-observer-train-v1"
ARMS = (*social.ARMS, "starting_observed")
COMPARATORS = (*social.CONTROLS, "starting_observed")
CFG = social.CFG


def average_ranks(scores):
    """Equal values occupy the average of their sorted rank positions."""
    scores = np.asarray(scores)
    ranks = np.empty(len(scores), np.float32)
    order = np.argsort(scores, kind="stable")
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = np.float32((cursor + end - 1) / 2)
        cursor = end
    return ranks / np.float32(len(scores) - 1) - np.float32(.5)


def optimizer_transition(previous, samples, steps=300):
    eps = typed(samples, "eps", (16, 155), "f")
    candidates = typed(samples, "candidate_genomes", (16, 155), "f")
    fitness = typed(samples, "fitness", (16, 4), "f")
    require(np.array_equal(eps[8:], -eps[:8]), "antithetic perturbations")
    close(candidates, previous["mu"] + np.float32(.05) * eps, "sampled observer genomes")
    events = {name: typed(samples, "candidate_" + name, (16, 4, steps, 1), "biuf", binary=True)
              for name in ("food", "poison", "alive_before", "alive_after")}
    first = np.empty((16, 4), np.float32)
    for candidate in range(16):
        for family in range(4):
            trace = {name: value[candidate, family] for name, value in events.items()}
            first[candidate, family] = social.events(trace, "", steps)["first_food"]
    require(np.array_equal(fitness, first), "training fitness must equal all-assigned first-food events")
    scores = first.mean(axis=1, dtype=np.float32)
    weights = average_ranks(scores)
    close(typed(samples, "rank_weights", (16,), "f"), weights, "tied average ranks")
    informative = bool(np.any(scores != scores[0]))
    update = {name: previous[name].copy() for name in ("mu", "m", "v", "t")}
    if informative:
        gradient = (weights[:, None] * eps).sum(0) / np.float32(.8) - np.float32(.005) * previous["mu"]
        step = int(previous["t"]) + 1
        update["m"] = np.float32(.9) * previous["m"] + np.float32(.1) * gradient
        update["v"] = np.float32(.999) * previous["v"] + np.float32(.001) * gradient ** 2
        update["mu"] = previous["mu"] + np.float32(.02) * (update["m"] / np.float32(1 - .9 ** step)) / (
            np.sqrt(update["v"] / np.float32(1 - .999 ** step)) + np.float32(1e-8))
        update["t"] = np.asarray(step, previous["t"].dtype)
    return update, scores, informative


def family_events(raw, index, original, endpoint, cfg=CFG):
    result = social.family_events(raw, index, cfg)
    for field, expected in (("teacher_genome", original), ("starting_genome", original), ("evolved_genome", endpoint)):
        value = typed(raw, field, (155,), "f")
        require(value.dtype == np.float32 and np.array_equal(value, expected), field + " identity")
    prefix, steps = "starting_observed_", cfg["T"]
    require(np.array_equal(raw[prefix + "frames"], raw["observed_frames"]), "original observer receives same movie")
    for field in ("watch_motor", "watch_denergy", "solo_initial_energy", "solo_initial_alive", "solo_initial_denergy"):
        require(np.array_equal(raw[prefix + field], raw["observed_" + field]), "original observer own body " + field)
    birth = typed(raw, prefix + "birth_sha256", (32,), "u")
    require(birth.dtype == np.uint8 and np.array_equal(birth, raw[prefix + "watch_initial_W_sha256"]),
            "original observer uses its own birth weights")
    for state in ("W", "x", "Wp"):
        for moment in ("watch_initial", "watch_final", "solo_initial"):
            require(typed(raw, prefix + moment + "_" + state + "_sha256", (32,), "u").dtype == np.uint8,
                    "original observer state fingerprint dtype")
        require(np.array_equal(raw[prefix + "watch_final_" + state + "_sha256"],
                               raw[prefix + "solo_initial_" + state + "_sha256"]), "original observer neural continuity")
        if state != "W":
            require(np.array_equal(raw[prefix + "watch_initial_" + state + "_sha256"],
                                   raw["observed_watch_initial_" + state + "_sha256"]), "original observer neural reset")
    outcome = social.events(raw, prefix, steps)
    for field in ("first_food", "first_poison", "no_meal", "first_meal_step", "fitness"):
        value = typed(raw, prefix + field, (), "biuf")
        require(value == outcome[field], "original observer " + field)
    changed = int(typed(raw, prefix + "weight_change_count", (), "iu"))
    neurons = 4 * (2 * cfg["R"] + 1) ** 2 + 3 + cfg["H"] + 6
    require(0 <= changed <= neurons ** 2, "original observer changed weights")
    outcome["weight_change_count"] = changed
    result["starting_observed"] = outcome
    return result


def measurements(rows):
    result = social.measurements(rows)
    original = [row["starting_observed"] for row in rows]
    result["arms"]["starting_observed"] = dict(
        first_food=base.world_stats([row["first_food"] for row in original]),
        survival=base.world_stats([row["survival"] for row in original]),
        fitness=base.world_stats([row["fitness"] for row in original]),
        changed_weights=base.world_stats([row["weight_change_count"] for row in original]),
        **{name + "_count": sum(row[name] for row in original) for name in ("first_food", "first_poison", "no_meal")},
        survivors=sum(row["survival"] for row in original))
    food = base.world_stats([int(row["observed"]["first_food"]) - int(row["starting_observed"]["first_food"]) for row in rows])
    survival = base.world_stats([int(row["observed"]["survival"]) - int(row["starting_observed"]["survival"]) for row in rows])
    result["contrasts"]["starting_observed"] = dict(first_food_difference=food, survival_difference=survival,
        gate=social.scientific_gate(food, survival))
    return result


def combined_gate(estimates):
    return {field: all(estimates["contrasts"][arm]["gate"][field] for arm in COMPARATORS)
            for field in ("passed", "first_food_passed", "survival_passed")}


class Auditor(social.Auditor):
    def completed_bundle(self, development=False):
        names = ["E122-benchmark"] if development else [f"E122-A{i}" for i in range(3)]
        count, attempts = (4, 1) if development else (128, 10)
        folders, pending = [], []
        for name in names:
            paths = sorted(self.path("runs").glob("*_" + name))
            require(len(paths) <= 1, "duplicate canonical " + name)
            if not paths or not (paths[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = paths[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            expected = dict(status="finished", attempted_rounds=attempts, expected_attempts=attempts,
                            completed_families=count, expected_families=count)
            files = {"families": {f"family_{i:04d}.npz" for i in range(count)},
                     "teachers": {f"teacher_{i:04d}.npz" for i in range(1, attempts + 1)},
                     "generations": {f"attempt_{i:04d}.npz" for i in range(attempts + 1)} |
                                    {f"samples_{i:04d}.npz" for i in range(1, attempts + 1)}}
            if (any(status.get(k) != v for k, v in expected.items())
                    or any({p.name for p in self.path(folder + "/" + kind).glob("*.npz")} != names for kind, names in files.items())
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; outcomes remain unopened")
        return folders

    def manifest(self, folder, relative, saved, index=None):
        arrays = self.arrays(folder + "/" + relative)
        expected = dict(path=relative, sha256=self.hashes[folder + "/" + relative])
        if index is not None:
            expected["index"] = index
        same(saved, expected, relative + " manifest")
        return arrays

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, attempts, training, evaluation = (4, 1, 49, 50) if development else (128, 10, 2026112801 + index, 2026112901 + index)
        split, name = ("development", "E122-benchmark") if development else ("final", f"E122-A{index}")
        path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, branch="O", cfg=CFG, cfg_overrides={}, condition_split=split,
            ecology=None, persistence="off", plastic=True, controller="rules-v0", genome_size=155,
            seed=training, evaluation_seed=evaluation, eval_worlds=count, expected_families=count,
            expected_attempts=attempts, l2=.005, arms=list(ARMS), controls=list(COMPARATORS),
            practice_steps=150, watch_steps=150, solo_steps=300,
            training_key_order=social.KEY_ORDER[:3], evaluation_key_order=social.KEY_ORDER,
            max_seconds=900 if development else 3600), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=training, generations=attempts, pop=16, worlds=4, sigma=.05, lr=.02, eval_worlds=count,
            start="file:" + path, no_plastic=False, allow_dirty=False, quick=False, resume=None,
            cfg=None, objective=None, memory_condition=None, max_train_seconds=None, checkpoint_every=10), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), "unexpected mechanism flags")
        subset(config["source"], dict(label=f"A{index}", path=path, sha256=digest, training_seed=index), "original source")
        same(config["teacher_source"], config["source"], "fixed teacher source")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40
                and self.code_trees(provenance["commit"]) == self.expected_trees, "registered clean source")
        content = self.read(path)
        require(hashlib.sha256(content).hexdigest() == digest and self.read(folder + "/genome_start.npy") == content
                and self.read(folder + "/teacher_genome.npy") == content, "original observer and fixed teacher bytes")
        original = np.load(io.BytesIO(content), allow_pickle=False)
        require(original.shape == (155,) and original.dtype == np.float32 and np.isfinite(original).all(), "original genome")
        return config, original

    def state(self, state):
        require(set(state) == {"mu", "m", "v", "t", "key"}, "optimizer state fields")
        for name in ("mu", "m", "v"):
            require(typed(state, name, (155,), "f").dtype == np.float32, "optimizer float32 " + name)
        require(np.all(state["v"] >= 0) and int(typed(state, "t", (), "iu")) >= 0, "optimizer counter/moment bounds")
        self.register_keys(typed(state, "key", (2,), "u")[None], "optimizer state")
        return state

    def training_batch(self, batch, samples, original):
        require(np.array_equal(typed(batch, "teacher_genome", (155,), "f"), original), "fixed training teacher")
        worlds = typed(samples, "world_keys", (4, 2), "u")
        roles = typed(batch, "role_keys", (4, 3, 2), "u")
        births = typed(samples, "development_keys", (16, 4, 2), "u")
        require(np.array_equal(births[:8], births[8:]), "antithetic pupil pairing")
        require(np.array_equal(typed(batch, "world_keys", (4, 2), "u"), worlds)
                and np.array_equal(typed(batch, "solo_world_keys", (4, 2), "u"), roles[:, 2]), "teacher batch keys")
        for keys in (worlds, roles.reshape(-1, 2), births[:8].reshape(-1, 2)):
            self.register_keys(keys, "training role")
        require(np.array_equal(typed(batch, "rules", (4,), "iu"), [0, 1, 0, 1]), "balanced training rules")
        movies = typed(batch, "movies", (4, 150, 103), "f")
        cameras = typed(batch, "camera_positions", (4, 150, 2), "iu")
        for family in range(4):
            raw = {name: value[family] for name, value in batch.items() if name.startswith("teacher_") and name != "teacher_genome"}
            raw["rule"] = np.array(family % 2)
            movie, camera, _, _ = social.reconstruct_movie(raw)
            require(np.array_equal(movies[family], movie) and np.array_equal(cameras[family], camera), "training public movie")

    def evolution(self, folder, config, summary, original):
        attempts = config["expected_attempts"]
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == attempts and len(summary["attempts"]) == attempts + 1, "attempt accounting")
        trajectory = np.load(io.BytesIO(self.read(folder + "/genome_trajectory.npy")), allow_pickle=False)
        require(trajectory.shape == (attempts + 1, 155) and trajectory.dtype == np.float32, "genome trajectory")
        previous = self.state(self.manifest(folder, "generations/attempt_0000.npz", summary["attempts"][0]))
        require(np.array_equal(previous["mu"], original) and not previous["m"].any() and not previous["v"].any()
                and int(previous["t"]) == 0 and np.array_equal(trajectory[0], original), "fresh original optimizer")
        records, elapsed = [], 0.
        for attempt, line in enumerate(lines, 1):
            row = json.loads(line)
            same(row["checkpoint"], summary["attempts"][attempt], "attempt manifest link")
            current = self.state(self.manifest(folder, f"generations/attempt_{attempt:04d}.npz", row["checkpoint"]))
            require(np.array_equal(trajectory[attempt], current["mu"]), "trajectory checkpoint")
            samples = self.manifest(folder, f"generations/samples_{attempt:04d}.npz", row["samples"])
            batch = self.manifest(folder, f"teachers/teacher_{attempt:04d}.npz", row["teacher"])
            self.training_batch(batch, samples, original)
            expected, scores, informative = optimizer_transition(previous, samples)
            for field, value in expected.items():
                if informative and field != "t":
                    close(current[field], value, "optimizer " + field)
                else:
                    require(np.array_equal(current[field], value), "tied skip/update count " + field)
            changed = int(np.count_nonzero(current["mu"] != previous["mu"]))
            subset(row, dict(attempt=attempt, informative=informative, all_tied=not informative,
                optimizer_updates=int(current["t"]), t=int(current["t"]), changed_parameters=changed,
                distinct_scores=len(np.unique(scores))), "attempt metadata")
            for name, value in (("fit_mean", scores.mean()), ("fit_best", scores.max()), ("fit_std", scores.std())):
                close(row[name], value, "training " + name)
            require(elapsed <= row["elapsed_seconds"] <= summary["elapsed_seconds"], "training elapsed order")
            elapsed = row["elapsed_seconds"]
            records.append(dict(attempt=attempt, informative=informative, optimizer_updates=int(current["t"]), changed_parameters=changed,
                maximum_adam_roundoff=max(float(np.max(np.abs(current[k] - v))) for k, v in expected.items())))
            previous = current
        endpoint = np.load(io.BytesIO(self.read(folder + "/genome_final.npy")), allow_pickle=False)
        require(endpoint.dtype == np.float32 and np.array_equal(endpoint, previous["mu"]), "frozen endpoint")
        same(summary["optimizer_updates"], int(previous["t"]), "final informative updates")
        same(summary["changed_parameters"], int(np.count_nonzero(endpoint != original)), "final genome changes")
        return endpoint, dict(attempts=records, last_training_elapsed=elapsed)

    def run(self, folder, index, development=False):
        config, original = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count, attempts = config["expected_families"], config["expected_attempts"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            attempted_rounds=attempts, expected_attempts=attempts, completed_families=count, expected_families=count,
            source_label=f"A{index}", training_seed=config["seed"], evaluation_seed=config["evaluation_seed"],
            artifact_validity="passed", scientific_result=not development), "completed summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "process time allowance")
        endpoint, training = self.evolution(folder, config, summary, original)
        lines = self.read(folder + "/evaluation.jsonl").decode("utf-8").splitlines()
        require(len(lines) == len(summary["raw_files"]) == count, "evaluation accounting")
        rows, elapsed = [], training["last_training_elapsed"]
        for family, line in enumerate(lines):
            record = json.loads(line)
            raw = self.manifest(folder, f"families/family_{family:04d}.npz", summary["raw_files"][family], family)
            rows.append(family_events(raw, family, original, endpoint))
            self.register_keys(raw["keys"], "fresh evaluation family")
            subset(record, dict(family=family, rule=family % 2, yoked_swap=family // 2 % 2, keys=raw["keys"].tolist(),
                                raw_file=summary["raw_files"][family]), "evaluation assignment")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "evaluation elapsed order")
            elapsed = record["elapsed_seconds"]
        estimates = measurements(rows)
        same(summary["measurements"], estimates, "independent five-arm measurements")
        report = dict(run=folder, source_commit=config["git"]["commit"], training=training, measurements=estimates)
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "development gate")
        else:
            report["gate"] = combined_gate(estimates)
            same(summary["gate"], report["gate"], "all-four-comparator gate")
            same(summary["gate_status"], "passed" if report["gate"]["passed"] else "failed", "gate status")
        return report


def self_check():
    require(np.array_equal(average_ranks(np.array([0., 1., 1., 2.])), [-.5, 0., 0., .5]), "average tied ranks")
    require(not average_ranks(np.ones(16)).any(), "all ties have zero ranks")
    previous = dict(mu=np.full(155, .1, np.float32), m=np.full(155, .2, np.float32),
                    v=np.full(155, .3, np.float32), t=np.array(2, np.int32))
    noise = np.tile(np.arange(1, 9, dtype=np.float32)[:, None], (1, 155)) / 8
    eps = np.concatenate((noise, -noise))
    samples = dict(eps=eps, candidate_genomes=previous["mu"] + np.float32(.05) * eps,
                   fitness=np.zeros((16, 4), np.float32), rank_weights=np.zeros(16, np.float32))
    for field in ("food", "poison", "alive_before", "alive_after"):
        samples["candidate_" + field] = np.full((16, 4, 4, 1), field.startswith("alive"), bool)
    update, scores, informative = optimizer_transition(previous, samples, 4)
    require(not informative and all(np.array_equal(update[k], previous[k]) for k in update), "tied round skips all optimizer state")
    samples["candidate_food"][8:, :, 0, 0] = True
    samples["fitness"][8:] = 1
    samples["rank_weights"] = average_ranks(samples["fitness"].mean(1))
    update, scores, informative = optimizer_transition(previous, samples, 4)
    require(informative and int(update["t"]) == 3 and not np.array_equal(update["mu"], previous["mu"]), "informative optimizer transition")
    for field in ("fitness", "rank_weights", "eps", "candidate_genomes", "candidate_alive_before"):
        broken = {k: v.copy() for k, v in samples.items()}
        broken[field].flat[0] = 10 if broken[field].dtype != bool else False
        try:
            optimizer_transition(previous, broken, 4)
        except base.AuditError:
            continue
        raise base.AuditError("failed to reject corrupt " + field)
    sys.path.insert(0, str(ROOT))
    from tests.test_social_learning_audit import synthetic_family, CFG as small_cfg
    raw = synthetic_family()
    raw.update({"starting_" + name: value.copy() for name, value in list(raw.items()) if name.startswith("observed_")})
    original = np.zeros(155, np.float32)
    raw.update(teacher_genome=original.copy(), starting_genome=original.copy(), evolved_genome=original.copy())
    row = family_events(raw, 0, original, original, small_cfg)
    require(not combined_gate(measurements([row, row]))["passed"], "no improvement cannot pass")
    row["observed"]["first_food"] = True
    require(combined_gate(measurements([row, row]))["passed"], "every comparator must lose")
    row["starting_observed"]["first_food"] = True
    require(not combined_gate(measurements([row, row]))["passed"], "original-observer comparison is mandatory")
    for field in ("teacher_genome", "starting_observed_frames", "starting_observed_watch_denergy"):
        broken = {k: v.copy() for k, v in raw.items()}
        broken[field].flat[0] += 1
        try:
            family_events(broken, 0, original, original, small_cfg)
        except base.AuditError:
            continue
        raise base.AuditError("failed to reject corrupt " + field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print(json.dumps(dict(status="self-check passed", simulator_imported=False)))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required for an artifact audit")
    auditor, report = None, dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            report["development_audit"] = auditor.run(auditor.completed_bundle(True)[0], 0, True)
        report["runs"] = [auditor.run(folder, index, args.development) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            final_evaluation_audited=not args.development, limitations=[
                "Neural dynamics, random-key derivation and noise generation are not replayed.",
                "Training teacher films and candidate events are checked; per-candidate watch states are not saved.",
                "Saved state hashes verify solo carry and body resets; they do not establish a successful mechanism.",
                "This conditional branch is authorized only after the E120 failure and E121 pass decisions."])
        if not args.development:
            report["scientific_gate_passed"] = all(run["gate"]["passed"] for run in report["runs"])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
