"""Read-only E121 reconstruction of fixed decoders from public movie pixels."""
import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "seed" / "seed.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_learning as film


base = film.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
PROTOCOL = "social-channel-v1"
ARMS = ("observed", "yoked", "no_demo", "scene_only")
METHODS = ("track", "scene")
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def channels(frames, radius):
    width = 2 * radius + 1
    require(frames.ndim == 2 and frames.shape[1] == 4 * width ** 2 + 3
            and np.isfinite(frames).all(), "public movie shape/values")
    images = frames[:, :4 * width ** 2].reshape(len(frames), 4, width, width)
    require(np.isin(images[:, :3], (0, 1)).all()
            and not np.any(images[:, 0].astype(bool) & images[:, 1].astype(bool)), "public item/occupancy pixels")
    return images


def color_at(images, frame, destination):
    width = images.shape[-1]
    if not all(0 <= int(value) < width for value in destination):
        return -1
    row, column = (int(value) for value in destination)
    if images[frame, 0, row, column]:
        return 0
    if images[frame, 1, row, column]:
        return 1
    return -1


def decision(colors, identified):
    votes = np.array([np.count_nonzero(colors == label) for label in (0, 1)], np.int32)
    prediction = -1 if votes[0] == votes[1] else int(np.argmax(votes))
    return dict(step_color=colors, step_identified=identified, votes=votes,
                prediction=np.asarray(prediction, np.int32))


def track_reader(frames, radius=2):
    """Use consecutive visible sprites and the registered camera geometry only."""
    images = channels(np.asarray(frames), radius)
    colors = np.full(len(images) - 1, -1, np.int8)
    identified = np.zeros(len(colors), bool)
    sightings = [np.argwhere(frame[2] > 0) for frame in images]
    for index in range(len(colors)):
        if len(sightings[index]) != 1 or len(sightings[index + 1]) != 1:
            continue
        displacement = sightings[index + 1][0] - radius + np.array([1, 0])
        if tuple(displacement) not in MOVES:
            continue
        destination = sightings[index][0] + displacement
        if np.any(destination < 0) or np.any(destination >= 2 * radius + 1):
            continue
        identified[index] = True
        colors[index] = color_at(images, index, destination)
    return decision(colors, identified)


def scene_reader(frames, radius=2):
    """Infer uniquely matched camera shifts by exact overlap-error fractions."""
    images = channels(np.asarray(frames), radius)
    width = 2 * radius + 1
    shifts = np.zeros((len(images) - 1, 2), np.int32)
    accepted = np.zeros(len(shifts), bool)
    rows, columns = np.indices((width, width))
    for index in range(len(shifts)):
        errors = []
        for dr, dc in MOVES:
            overlap = (rows + dr >= 0) & (rows + dr < width) & (columns + dc >= 0) & (columns + dc < width)
            target_row, target_col = rows[overlap], columns[overlap]
            earlier = images[index, :2, target_row + dr, target_col + dc]
            later = images[index + 1, :2, target_row, target_col]
            errors.append(Fraction(int(np.count_nonzero(earlier != later)), earlier.size))
        best = min(errors)
        winners = [candidate for candidate, error in enumerate(errors) if error == best]
        if len(winners) == 1:
            shifts[index] = MOVES[winners[0]]
            accepted[index] = True
    colors = np.full(len(images) - 2, -1, np.int8)
    identified = np.zeros(len(colors), bool)
    for index in range(len(colors)):
        if not accepted[index:index + 2].all():
            continue
        destination = radius + shifts[index] + shifts[index + 1] - np.array([1, 0])
        if np.any(destination < 0) or np.any(destination >= width):
            continue
        identified[index] = True
        colors[index] = color_at(images, index, destination)
    return dict(**decision(colors, identified), shifts=shifts, shift_identified=accepted)


def score(prediction, rule):
    return .5 if prediction == -1 else float(prediction == rule)


def family(raw, index, cfg=film.CFG):
    for name, expected in (("family_index", index), ("rule", index % 2), ("yoked_swap", index // 2 % 2),
                           ("steps", cfg["T"]), ("watch_start", cfg["T"] // 2), ("radius", cfg["R"]),
                           ("grid_size", cfg["G"])):
        require(int(typed(raw, name, (), "biu" if name == "yoked_swap" else "iu")) == expected, "movie " + name)
    require(float(typed(raw, "initial_energy", (), "f")) == cfg["e0"], "observer energy constant")
    require(typed(raw, "keys", (2, 2), "u").dtype == np.uint32, "teacher key dtype")
    movie, cameras, visible, teacher = film.reconstruct_movie(raw, cfg)
    require(np.array_equal(raw["camera_positions"], cameras), "camera positions")
    conditions = film.control_frames(movie, bool(index // 2 % 2), cfg["R"])
    result = dict(visible_frames=int(visible.sum()), methods={}, teacher=teacher)
    for method, decoder in (("track", track_reader), ("scene", scene_reader)):
        result["methods"][method] = {}
        for arm, frames in conditions.items():
            require(np.array_equal(raw[arm + "_frames"], frames), arm + " public frames")
            decoded = decoder(frames, cfg["R"])
            for name, value in decoded.items():
                actual = np.asarray(raw[method + "_" + arm + "_" + name])
                require(actual.dtype == value.dtype and np.array_equal(actual, value), method + " " + arm + " " + name)
            predicted = int(decoded["prediction"])
            accuracy = score(predicted, index % 2)
            require(float(raw[method + "_" + arm + "_score"]) == accuracy, "post-prediction score")
            votes = decoded["votes"]
            result["methods"][method][arm] = dict(score=accuracy, votes=votes,
                coverage=bool(votes.sum() > 0), no_vote=bool(votes.sum() == 0),
                tie=bool(votes[0] == votes[1]), identified=int(decoded["step_identified"].sum()))
    for method in METHODS:
        require(result["methods"][method]["no_demo"]["no_vote"], "blank movie produced a vote")
    require(result["methods"]["track"]["scene_only"]["no_vote"], "sprite decoder read absent sprite")
    return result


def channel_gate(accuracy, coverage, paired):
    coverage_passed = bool(coverage["mean"] >= .5)
    accuracy_passed = bool(accuracy["mean"] >= .6)
    chance_passed = bool(accuracy["mean"] - .5 > 2 * accuracy["se"])
    yoked_passed = bool(paired["mean"] > 2 * paired["se"])
    return dict(passed=coverage_passed and accuracy_passed and chance_passed and yoked_passed,
                coverage_passed=coverage_passed, accuracy_passed=accuracy_passed,
                chance_passed=chance_passed, yoked_passed=yoked_passed)


def measurements(rows):
    result = dict(families=len(rows), rule_counts={str(i): sum(j % 2 == i for j in range(len(rows))) for i in (0, 1)})
    for method in METHODS:
        arms, scores = {}, {}
        for arm in ARMS:
            values = [row["methods"][method][arm] for row in rows]
            scores[arm] = np.asarray([item["score"] for item in values])
            arms[arm] = dict(accuracy=base.world_stats(scores[arm]),
                coverage=base.world_stats([item["coverage"] for item in values]),
                no_vote_fraction=base.world_stats([item["no_vote"] for item in values]),
                tie_fraction=base.world_stats([item["tie"] for item in values]),
                votes=base.world_stats([int(item["votes"].sum()) for item in values]),
                identifiable_steps=base.world_stats([item["identified"] for item in values]))
        result[method] = dict(arms=arms, chance_difference=base.world_stats(scores["observed"] - .5),
            comparisons={arm: base.world_stats(scores["observed"] - scores[arm]) for arm in ARMS[1:]})
    return result


class Auditor(film.Auditor):
    def completed_bundle(self, development=False):
        names = ["E121-benchmark"] if development else [f"E121-A{i}" for i in range(3)]
        count = 4 if development else 128
        folders, pending = [], []
        for name in names:
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, name + " duplicate canonical attempts")
            if not candidates or not (candidates[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            state = self.json(folder + "/status.json")
            files = {path.name for path in self.path(folder + "/families").glob("*.npz")}
            if (state.get("status") != "finished" or state.get("completed_families") != count
                    or state.get("expected_families") != count
                    or files != {f"family_{i:04d}.npz" for i in range(count)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
                continue
            folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; final scores remain unopened")
        return folders

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master = (4, 48) if development else (128, 2026112701 + index)
        name = "E121-benchmark" if development else f"E121-A{index}"
        split = "development" if development else "final"
        source_path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, cfg=film.CFG, cfg_overrides={}, ecology=None,
            persistence="off", plastic=True, controller="rules-v0", genome_size=155,
            condition_split=split, generation=0, seed=master, eval_seed=master, eval_worlds=count,
            expected_families=count, arms=list(ARMS), practice_steps=150, watch_steps=150,
            methods=list(METHODS), key_order=["teacher_world", "teacher_birth"], raw_schema=PROTOCOL,
            gate=dict(minimum_coverage=.5, minimum_accuracy=.6, standard_errors=2., all_seeds=3),
            max_seconds=900 if development else 3600, scientific_result=not development), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=master, generations=0, eval_worlds=count, start="file:" + source_path,
            no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None,
            objective=None, memory_condition=None, max_train_seconds=None), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), name + " mechanism flags")
        subset(config["source"], dict(label=f"A{index}", path=source_path, sha256=digest), name + " source")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40
                and self.code_trees(provenance["commit"]) == self.expected_trees, name + " clean source")
        original = self.read(source_path)
        require(hashlib.sha256(original).hexdigest() == digest
                and self.read(folder + "/genome_start.npy") == original, name + " genome bytes")
        genome = np.load(io.BytesIO(original), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), name + " genotype")
        return config

    def prior_keys(self):
        """Read only key records from E120, preserving separation from its outcomes."""
        folders = sorted(self.path("runs").glob("*_E120-*"))
        for folder in folders:
            path = folder.relative_to(self.root).as_posix() + "/metrics.jsonl"
            if not self.path(path).is_file():
                continue
            for line in self.read(path).decode("utf-8").splitlines():
                self.register_keys(np.asarray(json.loads(line)["keys"], np.uint32), "prior E120 movie/solo keys")

    def run(self, folder, index, development=False):
        config = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count = config["expected_families"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=0, completed_families=count, expected_families=count, source_label=f"A{index}",
            eval_seed=config["seed"], eval_worlds=count, artifact_validity="passed",
            scientific_result=not development), folder + " summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "wall limit")
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == count and len(summary["raw_files"]) == count, "family accounting")
        rows, elapsed = [], 0.
        for number, line in enumerate(lines):
            relative = f"families/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + relative)
            artifact = dict(index=number, path=relative, sha256=self.hashes[folder + "/" + relative])
            same(summary["raw_files"][number], artifact, "raw manifest")
            reconstructed = family(raw, number)
            self.register_keys(raw["keys"], "E121 teacher keys")
            record = json.loads(line)
            subset(record, dict(family=number, rule=number % 2, yoked_swap=number // 2 % 2,
                keys=raw["keys"].tolist(), raw_file=artifact), "movie record")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "movie elapsed order")
            elapsed = record["elapsed_seconds"]
            rows.append(reconstructed)
        estimates = measurements(rows)
        same(summary["measurements"], estimates, "independent decoder estimates")
        report = dict(run=folder, source_commit=config["git"]["commit"], measurements=estimates,
            demonstration=dict(visible_frames=sum(row["visible_frames"] for row in rows),
                teacher_alive_at_demo_start=sum(bool(row["teacher"]["alive_before"][150, 0]) for row in rows)))
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "benchmark scientific gate")
        else:
            track = estimates["track"]
            gate = channel_gate(track["arms"]["observed"]["accuracy"], track["arms"]["observed"]["coverage"],
                                track["comparisons"]["yoked"])
            same(summary["gate"], gate, "four-criterion scientific gate")
            same(summary["gate_status"], "passed" if gate["passed"] else "failed", "scientific gate status")
            report["gate"] = gate
        return report


def self_check():
    width, pixels = 5, 25
    frames = np.zeros((3, 4 * pixels + 3), np.float32)
    frames[:, -1] = 1
    views = frames[:, :4 * pixels].reshape(3, 4, width, width)
    views[0, 2, 1, 2] = 1
    views[1, 2, 1, 3] = 1
    views[2, 2, 1, 2] = 1
    views[0, 0, 1, 3] = views[1, 0, 1, 3] = 1
    decoded = track_reader(frames)
    require(decoded["votes"].tolist() == [2, 0] and int(decoded["prediction"]) == 0,
            "known rightward contact followed by stay")
    swapped = film.control_frames(frames, True, 2)["yoked"]
    require(track_reader(swapped)["votes"].tolist() == [0, 2], "color permutation changes predicted color")
    tied = frames.copy()
    tied[1, 1 * width + 3] = 0
    tied[1, pixels + 1 * width + 3] = 1
    require(track_reader(tied)["votes"].tolist() == [1, 1]
            and score(int(track_reader(tied)["prediction"]), 0) == .5, "equal positive votes receive half credit")
    absent = film.control_frames(frames, False, 2)["no_demo"]
    for reader in (track_reader, scene_reader):
        empty = reader(absent)
        require(empty["votes"].sum() == 0 and score(int(empty["prediction"]), 1) == .5,
                "blank views abstain with half credit")
    scene_only = film.control_frames(frames, False, 2)["scene_only"]
    require(track_reader(scene_only)["votes"].sum() == 0, "absent teacher prevents visible tracking")

    grid = np.random.default_rng(19).integers(0, 3, (10, 10))
    grid[4, 5] = 1
    cameras = np.array([[4, 4], [5, 4], [5, 5], [4, 5]])
    movie = np.zeros((len(cameras), 4 * pixels + 3), np.float32)
    for index, (row, column) in enumerate(cameras):
        patch = grid[row - 2:row + 3, column - 2:column + 3]
        movie[index, :pixels] = (patch == 1).ravel()
        movie[index, pixels:2 * pixels] = (patch == 2).ravel()
        movie[index, -1] = 1
    inferred = scene_reader(movie)
    require(np.array_equal(inferred["shifts"], np.diff(cameras, axis=0))
            and inferred["shift_identified"].all() and int(inferred["step_color"][0]) == 0,
            "known camera motion identifies the public destination color")
    permuted = scene_reader(film.control_frames(movie, True, 2)["yoked"])
    require(np.array_equal(permuted["shifts"], inferred["shifts"])
            and np.array_equal(permuted["votes"], inferred["votes"][::-1]), "image shift color invariance")

    gate = channel_gate(dict(mean=.6, se=.049), dict(mean=.5, se=.1), dict(mean=.2, se=.09))
    require(gate["passed"], "inclusive coverage and accuracy thresholds")
    require(not channel_gate(dict(mean=.625, se=.0625), dict(mean=1., se=0.),
                             dict(mean=.25, se=.01))["chance_passed"], "strict chance threshold")
    require(not channel_gate(dict(mean=1., se=0.), dict(mean=1., se=0.),
                             dict(mean=.25, se=.125))["yoked_passed"], "strict paired-control threshold")


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
        auditor.prior_keys()
        if not args.development:
            benchmark = auditor.completed_bundle(True)[0]
            report["development_audit"] = auditor.run(benchmark, 0, True)
        report["runs"] = [auditor.run(folder, index, args.development) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit,
            registered_code_trees=auditor.expected_trees, final_evaluation_audited=not args.development,
            limitations=["The audit reconstructs public frames from saved teacher traces and decodes only those frames.",
                         "Teacher neural dynamics and random-key generation are not replayed.",
                         "Half-credit abstentions stay in the denominator for every score and control comparison.",
                         "The gate concerns these fixed readers of the stimulus. Agents' use of observation is measured separately.",
                         "Failure of these two readers cannot rule out every possible decoder."])
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
