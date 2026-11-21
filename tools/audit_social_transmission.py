"""Independent E124 transmission reconstruction; NumPy only and read-only.

Only complete three-lineage panels may be opened in final mode. A valid artifact
audit is separate from retained efficacy and cumulative improvement.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_belief as prior

film, channel, base = prior.film, prior.channel, prior.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
CFG = prior.CFG
PROTOCOL = "social-transmission-v1"
ARMS = ("predecessor", "repeated_founder", "yoked", "belief_reset")
CONTROLS = ARMS[1:]
MAX_ARCHIVE_BYTES, MAX_EXPANDED_BYTES, MAX_ARRAYS = 16 << 20, 64 << 20, 1024


def bounded_arrays(content):
    require(len(content) <= MAX_ARCHIVE_BYTES, "chain archive exceeds compressed bound")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= MAX_ARRAYS and sum(entry.file_size for entry in entries) <= MAX_EXPANDED_BYTES,
                "chain archive exceeds expanded bound")
        require(len({entry.filename for entry in entries}) == len(entries)
                and all(entry.filename.endswith(".npy") and "/" not in entry.filename and "\\" not in entry.filename
                        for entry in entries), "chain archive entry names")
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    require(all(value.dtype.kind in "biuf" and np.isfinite(value).all() for value in values.values()), "nonfinite/non-numeric chain")
    return values


def life_fields(raw, prefix):
    return {name[len(prefix):]: value for name, value in raw.items()
            if name.startswith(prefix) and name[len(prefix):] not in ("movie", "camera_positions")}


def public_movie(life, rule):
    """A teacher movie comes from its own solo trace, never its observation input."""
    raw = dict(rule=np.asarray(rule), teacher_initial_grid=life["solo_initial_grid"],
               teacher_initial_position=life["solo_initial_position"])
    raw.update({"teacher_" + name: life["solo_" + name] for name in ("grid", "position", "signal", "energy")})
    raw.update({"teacher_" + name: life[name] for name in ("food", "poison", "alive_before", "alive_after")})
    return film.reconstruct_movie(raw)


def check_life(life, rule, reliability, frames=None, *, reset=False):
    founder = frames is None
    require(float(typed(life, "q_birth", (), "f")) == .5, "fresh neutral belief")
    outcome = prior.neural_arm(life, "", birth=founder)
    own = dict(life, rule=np.asarray(rule))
    _, _, trajectory = prior.own_life(own, "", "solo_")
    signal = typed(life, "solo_signal", (CFG["T"], 1), "f")
    require(np.all(np.abs(signal) <= 1) and not signal[~outcome["alive_before"]].any(), "own public signal bounds/death")
    enabled = bool(typed(life, "readout_enabled", (), "biu", binary=True))
    if founder:
        require(not enabled and not any(name.startswith("belief_") for name in life), "founder receives no demonstration readout")
        require(float(life["q"]) == .5 and float(life["z"]) == 0
                and np.array_equal(life["track_votes"], [0, 0]) and int(life["track_prediction"]) == -1
                and life["track_step_color"].shape == life["track_step_identified"].shape == (0,), "founder neutral decoder")
        votes, q = np.zeros(2, np.int32), .5
    else:
        require(enabled and np.array_equal(life["frames"], frames), "assigned public movie")
        decoded, _, expected_q = prior.decoder(life, "", frames, reliability)
        q = float(typed(life, "q", (), "f"))
        prior.close(q, np.float32(.5 if reset else expected_q), "publicly derived calibrated belief", 1e-8)
        prior.readout(life, "", trajectory, q)
        votes = decoded["votes"]
    return outcome | dict(q=q, votes=votes.copy())


def chain(raw, index, original, reliability):
    for name, expected in (("family_index", index), ("chain_index", index), ("rule", index % 2),
                           ("steps", 300), ("watch_start", 150), ("agents", 1), ("radius", 2),
                           ("grid_size", 16), ("hidden_units", 48), ("cohorts", 3)):
        require(int(typed(raw, name, (), "iu")) == expected, "chain metadata " + name)
    require(bool(typed(raw, "yoked_swap", (), "biu", binary=True)) == bool(index // 2 % 2), "balanced color mapping")
    require(float(typed(raw, "initial_energy", (), "f")) == CFG["e0"]
            and float(typed(raw, "lambda_value", (), "f")) == reliability, "frozen body/calibration")
    genome = typed(raw, "original_genome", (155,), "f")
    require(genome.dtype == np.float32 and np.array_equal(genome, original), "fixed original genotype")
    keys = typed(raw, "keys", (4, 2, 2), "u")
    require(keys.dtype == np.uint32 and len({key.tobytes() for key in keys.reshape(-1, 2)}) == 8, "fresh chain roles")
    rule = index % 2
    founder = life_fields(raw, "founder_")
    result = dict(founder=check_life(founder, rule, reliability), cohorts={}, rule=rule, selected_lambda=reliability)
    initial_movie, initial_camera, _, _ = public_movie(founder, rule)
    require(np.array_equal(raw["founder_movie"], initial_movie)
            and np.array_equal(raw["founder_camera_positions"], initial_camera), "founder own solo movie")
    source = founder
    for cohort in range(1, 4):
        movie, camera, _, _ = public_movie(source, rule)
        require(int(typed(raw, f"c{cohort}_movie_source_stage", (), "iu")) == cohort - 1
                and np.array_equal(raw[f"c{cohort}_predecessor_movie"], movie)
                and np.array_equal(raw[f"c{cohort}_camera_positions"], camera), "immediate predecessor own solo movie")
        conditions = film.control_frames(movie, bool(index // 2 % 2), CFG["R"])
        expected = dict(predecessor=movie, repeated_founder=initial_movie,
                        yoked=conditions["yoked"], belief_reset=movie)
        lives = {arm: life_fields(raw, f"c{cohort}_{arm}_") for arm in ARMS}
        first = lives["predecessor"]
        for arm, life in lives.items():
            for field in ("solo_initial_grid", "solo_initial_position", "birth_sha256"):
                require(np.array_equal(life[field], first[field]), "paired cohort newborn " + field)
            for state in ("W", "x", "Wp"):
                require(np.array_equal(life["watch_initial_" + state + "_sha256"], first["watch_initial_" + state + "_sha256"]),
                        "matched genetic initial state")
            result["cohorts"].setdefault(str(cohort), {})[arm] = check_life(life, rule, reliability, expected[arm], reset=arm == "belief_reset")
        reset = lives["belief_reset"]
        preserved = ("frames", "watch_motor", "watch_denergy", "weight_change_count") + tuple(
            stage + "_" + state + "_sha256" for stage in ("watch_initial", "watch_final", "solo_initial") for state in ("W", "x", "Wp"))
        require(all(np.array_equal(reset[field], first[field]) for field in preserved), "reset changes only acquired belief")
        if cohort == 1:
            repeated = lives["repeated_founder"]
            require(set(repeated) == set(first) and all(np.array_equal(first[field], repeated[field]) for field in first),
                    "cohort-one predecessor and founder bit identity")
        source = first
    return result


def interval(values):
    result = base.world_stats(values)
    return result | dict(lower_2se=None if result["se"] is None else result["mean"] - 2 * result["se"],
                         upper_2se=None if result["se"] is None else result["mean"] + 2 * result["se"])


def contrast(left, right):
    first = interval([int(a["first_food"]) - int(b["first_food"]) for a, b in zip(left, right)])
    survival = interval([int(a["survival"]) - int(b["survival"]) for a, b in zip(left, right)])
    defined = first["se"] is not None
    efficacy = bool(first["lower_2se"] > 0) if defined else None
    survives = bool(survival["lower_2se"] > -.10) if defined else None
    return dict(first_food_difference=first, survival_difference=survival,
                gate=dict(passed=efficacy and survives if defined else None,
                          first_food_passed=efficacy, survival_passed=survives))


def arm_stats(values):
    mapping = dict(first_food="first_food", first_poison="first_poison", no_meal="no_meal", survival="survival",
                   fitness="fitness", first_meal_step="first_meal_step", changed_weights="weight_change_count", belief_probability="q")
    result = {target: interval([row[source] for row in values]) for target, source in mapping.items()}
    votes = np.stack([row["votes"] for row in values])
    result.update(decoder_coverage=interval(votes.sum(1) > 0), decoder_strict_coverage=interval(votes[:, 0] != votes[:, 1]),
        decoder_votes=interval(votes.sum(1)), survivors=sum(row["survival"] for row in values),
        **{name + "_count": sum(row[name] for row in values) for name in ("first_food", "first_poison", "no_meal")})
    return result


def measurements(rows):
    require(len(rows) > 0 and len({row["selected_lambda"] for row in rows}) == 1, "one frozen calibration per panel")
    cohorts = {}
    for cohort in ("1", "2", "3"):
        arms = {arm: [row["cohorts"][cohort][arm] for row in rows] for arm in ARMS}
        cohorts[cohort] = dict(arms={arm: arm_stats(values) for arm, values in arms.items()},
            contrasts={arm: contrast(arms["predecessor"], arms[arm]) for arm in CONTROLS})
    cross = contrast([row["cohorts"]["3"]["predecessor"] for row in rows],
                     [row["cohorts"]["1"]["predecessor"] for row in rows])
    cumulative = dict(c3_vs_c1=cross, c3_vs_repeated_founder=cohorts["3"]["contrasts"]["repeated_founder"])
    defined = len(rows) > 1
    efficacy = all(cohorts[c]["contrasts"][arm]["gate"]["first_food_passed"]
                   for c in cohorts for arm in ("yoked", "belief_reset")) if defined else None
    safety = (all(cohorts[c]["contrasts"][arm]["gate"]["survival_passed"] for c in cohorts for arm in CONTROLS)
              and cross["gate"]["survival_passed"]) if defined else None
    improved = efficacy and all(item["gate"]["first_food_passed"] for item in cumulative.values()) if defined else None
    return dict(protocol=PROTOCOL, chains=len(rows), cohorts=cohorts, selected_lambda=rows[0]["selected_lambda"],
        agents_per_chain=13, assigned_agents=len(rows) * 13,
        rule_counts={str(rule): sum(row["rule"] == rule for row in rows) for rule in (0, 1)},
        founder=arm_stats([row["founder"] for row in rows]), cross_cohort=dict(c3_vs_c1=cross),
        cumulative_contrasts=cumulative, survival_floor=-.10,
        retained_efficacy=dict(passed=efficacy and safety if defined else None, first_food_passed=efficacy, survival_passed=safety),
        cumulative_improvement=dict(passed=improved and safety if defined else None, first_food_passed=improved, survival_passed=safety))


class Auditor(prior.Auditor):
    def arrays(self, relative):
        require(self.path(relative).stat().st_size <= MAX_ARCHIVE_BYTES, "compressed artifact exceeds bound")
        return bounded_arrays(self.read(relative))

    def completed_bundle(self, development=False):
        names = ["E124-benchmark"] if development else [f"E124-A{i}" for i in range(3)]
        count, folders, pending = (4 if development else 128), [], []
        for name in names:
            paths = sorted(self.path("runs").glob("*_" + name))
            require(len(paths) <= 1, "duplicate canonical " + name)
            if not paths or not (paths[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = paths[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            files = {path.name for path in self.path(folder + "/chains").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_chains") != count
                    or status.get("expected_chains") != count or files != {f"chain_{i:04d}.npz" for i in range(count)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; chain outcomes remain unopened")
        return folders

    def prior_keys(self):
        super().prior_keys()
        # E122 held-out assignments and all E123 assignments are separate from its ES samples.
        for experiment in (122, 123):
            for directory in sorted(self.path("runs").glob(f"*_E{experiment}-*")):
                folder = directory.relative_to(self.root).as_posix()
                if self.path(folder + "/config.json").is_file():
                    config = self.json(folder + "/config.json")
                    self.used_masters.update(config[name] for name in ("seed", "eval_seed", "evaluation_seed", "calibration_seed") if name in config)
                for filename in ("metrics.jsonl", "evaluation.jsonl"):
                    if self.path(folder + "/" + filename).is_file():
                        for line in self.read(folder + "/" + filename).decode("utf-8").splitlines():
                            record = json.loads(line)
                            if "keys" in record:
                                keys = np.asarray(record["keys"], np.uint32)
                                require(keys.shape[-1] == 2, "earlier assignment key shape")
                                self.used_keys.update(key.tobytes() for key in keys.reshape(-1, 2))

    def pinned(self, artifact):
        require(self.path(artifact["path"]).stat().st_size <= MAX_ARCHIVE_BYTES, "source artifact exceeds bound")
        content = self.read(artifact["path"])
        same(hashlib.sha256(content).hexdigest(), artifact["sha256"], "source artifact hash")
        return content

    def calibration(self, source, lineage, development, current_source):
        name = "E123-benchmark" if development else f"E123-A{lineage}"
        count, master = (4, 51) if development else (32, 2026113001 + lineage)
        folder = source["folder"]
        require(self.path(folder).parent == self.path("runs") and self.path(folder).name.endswith("_" + name), "source calibration folder")
        subset(source, dict(name=name, calibration_seed=master, expected_calibration_families=count), "source calibration identity")
        same(source["config"]["path"], folder + "/config.json", "source config path")
        same(source["selection"]["path"], folder + "/calibration/selection.json", "source selection path")
        config = json.loads(self.pinned(source["config"]))
        subset(config, dict(protocol=prior.PROTOCOL, condition_split="development" if development else "final",
            cfg=CFG, generation=0, calibration_seed=master, calibration_worlds=count,
            expected_calibration_families=count, source=current_source), "source original calibration")
        require(config["git"].get("dirty") is False and config["git"].get("commit") == source["commit"]
                and len(source["commit"]) == 40, "source calibration clean commit")
        selection_bytes = self.pinned(source["selection"])
        selection = json.loads(selection_bytes)
        require(len(selection["raw_files"]) == len(source["raw_files"]) == count, "all calibration families")
        rows, keys_seen = [], set()
        for index, (artifact, registered) in enumerate(zip(selection["raw_files"], source["raw_files"])):
            relative = f"calibration/family_{index:04d}.npz"
            same(artifact["index"], index, "calibration ordering")
            same(artifact["path"], relative, "calibration relative path")
            same(registered, dict(index=index, path=folder + "/" + relative, sha256=artifact["sha256"]), "calibration source manifest")
            raw = bounded_arrays(self.pinned(registered))
            keys = typed(raw, "keys", (4, 2), "u")
            tokens = [key.tobytes() for key in keys]
            require(keys.dtype == np.uint32 and len(set(tokens)) == 4 and not keys_seen.intersection(tokens), "calibration key uniqueness")
            keys_seen.update(tokens)
            rows.append(prior.calibration_family(raw, index))
        self.used_keys.update(keys_seen)
        fitted = prior.select_reliability(rows)
        subset(selection, dict(**fitted, status="frozen", calibration_seed=master,
            completed_families=count, expected_families=count), "independent frozen calibration")
        return selection_bytes, fitted["selected_lambda"]

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master = (4, 2026120200) if development else (128, 2026120201 + index)
        split, name = ("development", "E124-benchmark") if development else ("final", f"E124-A{index}")
        path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={},
            ecology=None, persistence="off", plastic=True, controller="rules-v0", genome_size=155,
            generation=0, seed=master, eval_seed=master, eval_worlds=count, expected_chains=count,
            cohorts=3, stages=["founder", "c1", "c2", "c3"], key_order=["solo_world", "pupil_birth"],
            arms=list(ARMS), controls=list(CONTROLS), founder_steps=300, watch_steps=150, solo_steps=300,
            max_seconds=900 if development else 3600, raw_schema=PROTOCOL), "registered transmission config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split, seed=master,
            generations=0, eval_worlds=count, start="file:" + path, no_plastic=False, allow_dirty=False,
            quick=False, resume=None, cfg=None, objective=None, memory_condition=None, max_train_seconds=None,
            pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10), "registered arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), "unexpected mechanism flags")
        require(master not in self.used_masters, "prior/development/final master reuse")
        self.used_masters.add(master)
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40
                and self.code_trees(provenance["commit"]) == self.expected_trees, "clean registered source")
        subset(config["source"], dict(label=f"A{index}", path=path, sha256=digest), "original genome source")
        content = self.read(path)
        require(hashlib.sha256(content).hexdigest() == digest and self.read(folder + "/genome_start.npy") == content, "fixed original genome bytes")
        original = np.load(io.BytesIO(content), allow_pickle=False)
        require(original.shape == (155,) and original.dtype == np.float32 and np.isfinite(original).all(), "original genotype")
        selection, reliability = self.calibration(config["calibration_source"], index, development, config["source"])
        frozen = dict(path="calibration/selection.json", sha256=hashlib.sha256(selection).hexdigest())
        same(config["frozen_calibration"], frozen, "frozen calibration manifest")
        require(self.read(folder + "/" + frozen["path"]) == selection and config["selected_lambda"] == reliability,
                "copied frozen calibration is unchanged")
        return config, original, reliability

    def run(self, folder, index, development=False):
        config, original, reliability = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count = config["expected_chains"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=0, source_label=f"A{index}", completed_chains=count, expected_chains=count,
            eval_seed=config["seed"], eval_worlds=count, scientific_result=not development, artifact_validity="passed",
            selected_lambda=reliability, calibration_source=config["calibration_source"], frozen_calibration=config["frozen_calibration"]), "completed summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "invocation time allowance")
        records = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(records) == len(summary["raw_files"]) == count, "complete chain accounting")
        rows, elapsed = [], 0.
        for number, line in enumerate(records):
            path = f"chains/chain_{number:04d}.npz"
            raw = self.arrays(folder + "/" + path)
            artifact = dict(index=number, path=path, sha256=self.hashes[folder + "/" + path])
            same(summary["raw_files"][number], artifact, "chain raw manifest")
            digest = config["frozen_calibration"]["sha256"]
            require(typed(raw, "calibration_sha256", (32,), "u").dtype == np.uint8
                    and raw["calibration_sha256"].tobytes().hex() == digest, "chain calibration fingerprint")
            row = chain(raw, number, original, reliability)
            self.register_keys(raw["keys"].reshape(8, 2), "fresh paired chain roles")
            record = json.loads(line)
            subset(record, dict(chain=number, rule=number % 2, yoked_swap=number // 2 % 2,
                keys=raw["keys"].tolist(), raw_file=artifact, calibration_sha256=digest), "chain assignment")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "chain time order")
            elapsed = record["elapsed_seconds"]
            rows.append(row)
        estimates = measurements(rows)
        gates = {name: estimates.pop(name) for name in ("retained_efficacy", "cumulative_improvement")}
        same(summary["measurements"], estimates, "independent chain statistics and intervals")
        subset(summary["safeguards"], dict(world_keys_disjoint=True, calibration_source_unchanged=True,
            original_genome_unchanged=True, three_cohort_four_arm_evidence_validated=True), "verified safeguards")
        report = dict(run=folder, source_commit=config["git"]["commit"], measurements=estimates)
        for name, gate in gates.items():
            if development:
                require(summary[name + "_status"] == "not_applicable" and name not in summary, "benchmark gate")
            else:
                same(summary[name], gate, name + " reconstructed decision")
                same(summary[name + "_status"], "passed" if gate["passed"] else "failed", name + " status")
                report[name] = gate
        return report


def synthetic_chain():
    """Artificial stationary lives: exercise evidence plumbing without any engine."""
    original = np.zeros(155, np.float32)
    raw = dict(family_index=np.asarray(2), chain_index=np.asarray(2), rule=np.asarray(0), yoked_swap=np.asarray(True),
        steps=np.asarray(300), watch_start=np.asarray(150), agents=np.asarray(1), radius=np.asarray(2),
        grid_size=np.asarray(16), hidden_units=np.asarray(48), cohorts=np.asarray(3), initial_energy=np.asarray(1., np.float32),
        lambda_value=np.asarray(2.), original_genome=original, keys=np.arange(1, 17, dtype=np.uint32).reshape(4, 2, 2))
    energies, energy = [], np.float32(1.)
    for _ in range(300):
        energy = np.float32(energy - np.float32(.003))
        energies.append(energy)

    def life(stage, frames, *, birth=False, reset=False):
        grid = np.zeros((16, 16), np.uint8)
        grid.flat[:24], grid.flat[24:48] = 1, 2
        grid = np.roll(grid, stage, axis=0)
        frames = np.empty((0, 103), np.float32) if birth else frames.copy()
        zero, alive = np.zeros((300, 1), bool), np.ones((300, 1), bool)
        row = dict(frames=frames, watch_motor=np.tile(np.array([1, 0, 0, 0, 0], np.float32), (len(frames), 1)),
            watch_denergy=np.zeros(len(frames), np.float32), birth_sha256=np.full(32, stage + 1, np.uint8),
            solo_initial_energy=np.array([1.], np.float32), solo_initial_alive=np.array([True]), solo_initial_denergy=np.array([0.], np.float32),
            food=zero.copy(), poison=zero.copy(), alive_before=alive.copy(), alive_after=alive.copy(),
            first_food=np.asarray(False), first_poison=np.asarray(False), no_meal=np.asarray(True), first_meal_step=np.asarray(301),
            fitness=np.asarray(0.), weight_change_count=np.asarray(0), q_birth=np.asarray(.5), readout_enabled=np.asarray(not birth),
            solo_initial_grid=grid, solo_initial_position=np.array([[8, 8]], np.int32),
            solo_grid=np.repeat(grid[None], 300, axis=0), solo_position=np.repeat(np.array([[[8, 8]]], np.int32), 300, axis=0),
            solo_energy=np.asarray(energies, np.float32)[:, None], solo_signal=np.full((300, 1), .1 * (stage + 1), np.float32))
        initial = dict(W=row["birth_sha256"], x=film.fingerprint(np.zeros(157, np.float32)),
                       Wp=film.fingerprint(np.zeros((103, 48), np.float32)))
        for state, digest in initial.items():
            for moment in ("watch_initial", "watch_final", "solo_initial"):
                row[moment + "_" + state + "_sha256"] = digest.copy()
        decoded = dict(step_color=np.empty(0, np.int8), step_identified=np.empty(0, bool), votes=np.zeros(2, np.int32),
                       prediction=np.asarray(-1, np.int32)) if birth else channel.track_reader(frames)
        z, q = prior.belief(decoded["votes"], 2.)
        row.update({"track_" + name: value for name, value in decoded.items()})
        row.update(z=np.asarray(z), q=np.asarray(.5 if reset else q, np.float32))
        if not birth:
            _, _, trajectory = prior.own_life(dict(row, rule=np.asarray(0)), "", "solo_")
            row.update(belief_q=np.full((300, 1), row["q"], np.float32), belief_bias=np.zeros((300, 1, 5), np.float32),
                belief_action=np.zeros((300, 1), np.int32), belief_efference=np.tile(np.array([[[1, 0, 0, 0, 0]]], np.float32), (300, 1, 1)),
                belief_denergy=trajectory["signed_energy"], belief_unbiased_logits=np.zeros((300, 1, 5), np.float32),
                belief_biased_logits=np.zeros((300, 1, 5), np.float32))
        return row

    founder = life(0, None, birth=True)
    raw.update({"founder_" + name: value for name, value in founder.items()})
    movie, camera, _, _ = public_movie(founder, 0)
    raw.update(founder_movie=movie, founder_camera_positions=camera)
    source = founder
    for cohort in range(1, 4):
        movie, camera, _, _ = public_movie(source, 0)
        raw.update({f"c{cohort}_predecessor_movie": movie, f"c{cohort}_camera_positions": camera,
                    f"c{cohort}_movie_source_stage": np.asarray(cohort - 1)})
        control = film.control_frames(movie, True, 2)
        for arm in ARMS:
            frames = raw["founder_movie"] if arm == "repeated_founder" else control["yoked"] if arm == "yoked" else movie
            current = life(cohort, frames, reset=arm == "belief_reset")
            raw.update({f"c{cohort}_{arm}_" + name: value for name, value in current.items()})
            if arm == "predecessor":
                source = current
    return raw, original


def self_check():
    import copy
    checks = []

    def rejected(action, name):
        try:
            action()
        except base.AuditError:
            return
        raise base.AuditError("accepted corrupted " + name)

    raw, original = synthetic_chain()
    observed = chain(raw, 2, original, 2.)
    checks.append("fresh founder and twelve assigned cohort lives")
    for field in ("c2_predecessor_movie", "c2_predecessor_q_birth", "c1_repeated_founder_solo_signal", "original_genome"):
        changed = dict(raw)
        changed[field] = raw[field].copy()
        changed[field].flat[0] += 1
        rejected(lambda: chain(changed, 2, original, 2.), field)
    changed = dict(raw, keys=raw["keys"].copy())
    changed["keys"][1, 0] = changed["keys"][0, 0]
    rejected(lambda: chain(changed, 2, original, 2.), "reused cohort key")
    changed = dict(raw, c2_predecessor_movie=raw["founder_movie"])
    rejected(lambda: chain(changed, 2, original, 2.), "substituted earlier founder movie")
    checks.append("movie provenance, matched founder, frozen genotype and neutral births reject corruption")
    rows = [copy.deepcopy(observed) for _ in range(16)]
    for row in rows:
        for cohort in ("1", "2", "3"):
            row["cohorts"][cohort]["predecessor"]["first_food"] = True
    outcome = measurements(rows)
    require(outcome["retained_efficacy"]["passed"] and not outcome["cumulative_improvement"]["passed"],
            "retained efficacy does not imply rising performance")
    for row in rows[:8]:
        row["cohorts"]["1"]["predecessor"]["first_food"] = False
    require(measurements(rows)["cumulative_improvement"]["passed"], "independent cumulative gate")
    for row in rows[:4]:
        row["cohorts"]["3"]["predecessor"]["survival"] = False
    require(not measurements(rows)["retained_efficacy"]["passed"], "strict survival safeguard")
    require(not contrast([dict(first_food=True, survival=False)], [dict(first_food=False, survival=True)])["gate"]["passed"],
            "one chain cannot decide a gate")
    require(interval([0., 1.]) == dict(mean=.5, se=.5, lower_2se=-.5, upper_2se=1.5), "paired chain uncertainty")
    checks.append("retention, cumulative improvement, survival loss and chain-level intervals")
    stream = io.BytesIO()
    np.savez_compressed(stream, **raw)
    restored = bounded_arrays(stream.getvalue())
    require(set(restored) == set(raw) and all(np.array_equal(restored[name], raw[name]) for name in raw), "bounded archive roundtrip")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("outside/array.npy", b"not an array")
    rejected(lambda: bounded_arrays(stream.getvalue()), "nested archive path")
    checks.append("bounded numeric archive loading")
    return dict(status="passed", tests=len(checks), checks=checks, simulator_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2, allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required for an artifact audit")
    auditor, report = None, dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        auditor.prior_keys()
        if not args.development:
            report["development_audit"] = auditor.run(auditor.completed_bundle(True)[0], 0, True)
        report["runs"] = [auditor.run(folder, index, args.development) for index, folder in enumerate(folders)]
        report.update(status="passed", final_evaluation_audited=not args.development, registered_source=args.source_commit,
            registered_code_trees=auditor.expected_trees, limitations=[
                "The controller and calibration are frozen: transmission is not genetic evolution.",
                "Public films, votes, states and events are checked; neural dynamics and random-key generation are not replayed.",
                "Recorded birth hashes and neutral states support fresh births; they cannot independently replay genetic development.",
                "The parent workflow must establish the E123 efficacy, reset and survival prerequisites before activation.",
                "Passing the survival confidence screen permits limited loss; it does not establish equality or no harm."])
        if not args.development:
            for name in ("retained_efficacy", "cumulative_improvement"):
                report[name + "_passed"] = all(run[name]["passed"] for run in report["runs"])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
