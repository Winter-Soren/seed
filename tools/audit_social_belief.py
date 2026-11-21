"""Independent, read-only NumPy reconstruction for E123 calibrated social belief.

Final mode opens outcomes only after all three canonical panels are complete.
Exit zero means artifact integrity passed; the scientific gate is separate.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "HANDOFF.md").is_file() and (path / "seed" / "seed.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_channel as channel

film, base = channel.film, channel.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
CFG = film.CFG
PROTOCOL = "social-belief-readout-v1"
ARMS = ("observed", "yoked", "scene_only", "no_demo", "belief_reset", "birth_parent")
CONTROLS = ARMS[1:]
GRID = (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)
MOVES = np.array(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)), np.int32)


def close(actual, expected, label, tolerance=4e-7):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, atol=tolerance, rtol=tolerance), label)


def belief(votes, reliability):
    votes = np.asarray(votes)
    require(votes.shape == (2,) and votes.dtype.kind in "iu" and np.all(votes >= 0), "decoder vote counts")
    z = float(int(votes[0]) - int(votes[1])) / max(1, int(votes.sum()))
    return z, 1. / (1. + math.exp(-float(reliability) * z))


def choose_loss(losses):
    return min(losses, key=lambda row: (row["mean_log_loss"], abs(row["lambda"]), -row["lambda"]))["lambda"]


def select_reliability(rows):
    """Fit only public evidence plus a pupil's own first contact and outcome."""
    require(len(rows) > 0, "no assigned calibration families")
    usable = [row for row in rows if row["contact_color"] != -1]
    losses = []
    for value in GRID:
        terms = []
        for row in usable:
            require(row["contact_color"] in (0, 1) and row["contact_outcome"] in (-1, 1), "own calibration label")
            signed = value * row["z"] * (1 if row["contact_color"] == 0 else -1)
            target = int(row["contact_outcome"] > 0)
            terms.append(float(np.logaddexp(0., signed) - target * signed))
        stats = base.world_stats(terms) if terms else dict(mean=None, se=None)
        losses.append({"lambda": value, "mean_log_loss": stats["mean"], "se_log_loss": stats["se"]})
    selected = choose_loss(losses) if usable else 0.
    payload = [{**row, "votes": np.asarray(row["votes"]).tolist()} for row in rows]
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(selected_lambda=selected, families=len(rows), contacts=len(usable),
                contact_coverage=base.world_stats([row["contact_color"] != -1 for row in rows]),
                vote_coverage=base.world_stats([bool(np.sum(row["votes"])) for row in rows]),
                vote_tie_fraction=base.world_stats([row["votes"][0] == row["votes"][1] for row in rows]),
                losses=losses, calibration_inputs_sha256=fingerprint)


def movie(raw, index, cfg=CFG):
    require(all(np.isfinite(value).all() for value in raw.values()), "nonfinite family evidence")
    for name, expected in (("family_index", index), ("rule", index % 2), ("steps", cfg["T"]),
                           ("watch_start", cfg["T"] // 2), ("radius", cfg["R"]), ("grid_size", cfg["G"]),
                           ("agents", 1), ("hidden_units", cfg["H"])):
        require(int(typed(raw, name, (), "iu")) == expected, "registered " + name)
    require(bool(typed(raw, "yoked_swap", (), "biu", binary=True)) == bool(index // 2 % 2), "balanced movie permutation")
    require(float(typed(raw, "initial_energy", (), "f")) == cfg["e0"], "newborn energy")
    require(typed(raw, "keys", (4, 2), "u").dtype == np.uint32, "family role keys")
    frames, cameras, visible, teacher = film.reconstruct_movie(raw, cfg)
    require(np.array_equal(raw["camera_positions"], cameras), "public camera reconstruction")
    return frames, dict(visible_frames=int(visible.sum()), teacher=teacher)


def decoder(raw, prefix, frames, reliability):
    decoded = channel.track_reader(frames, CFG["R"])
    for name, expected in decoded.items():
        actual = raw[prefix + "track_" + name]
        require(actual.dtype == expected.dtype and np.array_equal(actual, expected), prefix + "public decoder " + name)
    z, q = belief(decoded["votes"], reliability)
    close(raw[prefix + "z"], z, prefix + "movie evidence", 1e-12)
    return decoded, z, q


def own_life(raw, event_prefix, geometry_prefix, cfg=CFG):
    """Reconstruct contact colors and actual own energy changes from saved worlds."""
    steps, size = cfg["T"], cfg["G"]
    outcome = film.events(raw, event_prefix, steps)
    initial_grid = typed(raw, "solo_initial_grid", (size, size), "iu")
    initial_position = typed(raw, "solo_initial_position", (1, 2), "iu")
    grids = typed(raw, geometry_prefix + "grid", (steps, size, size), "iu")
    positions = typed(raw, geometry_prefix + "position", (steps, 1, 2), "iu")
    energy = typed(raw, geometry_prefix + "energy", (steps, 1), "f")
    require(np.isin(initial_grid, (0, 1, 2)).all() and np.isin(grids, (0, 1, 2)).all(), "own item labels")
    require(np.sum(initial_grid == 1) == cfg["n_items"] and np.sum(initial_grid == 2) == cfg["n_items"], "own initial item counts")
    require(np.all((initial_position >= 0) & (initial_position < size))
            and np.all((positions >= 0) & (positions < size)), "own position bounds")
    pre_grid = np.concatenate((initial_grid[None], grids[:-1]))
    pre_position = np.concatenate((initial_position[None], positions[:-1]))
    pre_energy = np.concatenate((np.full((1, 1), cfg["e0"], np.float32), energy[:-1]))
    active = outcome["alive_before"]
    item = pre_grid[np.arange(steps), positions[:, 0, 0], positions[:, 0, 1]]
    contacts = (item > 0) & active[:, 0]
    require(np.array_equal(contacts, (outcome["food"] | outcome["poison"])[:, 0]), "own contacts from arrival grid")
    rule = int(raw["rule"])
    require(np.array_equal(outcome["food"][:, 0], contacts & (item == 1 + rule))
            and np.array_equal(outcome["poison"][:, 0], contacts & (item == 2 - rule)), "own event labels")
    expected_energy = np.clip(pre_energy - cfg["decay"] * active + cfg["gain"] * outcome["food"]
                              - cfg["gain"] * outcome["poison"], 0, cfg["e_max"])
    close(energy, expected_energy, "own energy accounting")
    require(np.array_equal(outcome["alive_after"], energy > 0), "own survival accounting")
    moves = (positions - pre_position + size // 2) % size - size // 2
    require(all(tuple(value) in map(tuple, MOVES) for value in moves[:, 0]), "own one-step movement")
    require(not moves[~active].any(), "dead pupil movement")
    signed_energy = (energy - pre_energy) * active
    first = int(np.flatnonzero(contacts)[0]) if contacts.any() else None
    label = dict(contact_step=steps + 1 if first is None else first + 1,
                 contact_color=-1 if first is None else int(item[first]) - 1,
                 contact_outcome=0 if first is None else int(np.sign(signed_energy[first, 0])),
                 contact_denergy=0. if first is None else float(signed_energy[first, 0]))
    require(first is None or label["contact_outcome"] in (-1, 1), "first contact has an observed energy outcome")
    return outcome, label, dict(pre_grid=pre_grid, pre_position=pre_position, positions=positions,
                                signed_energy=signed_energy, active=active)


def calibration_family(raw, index):
    frames, _ = movie(raw, index)
    require(np.array_equal(raw["observed_frames"], frames), "calibration public film")
    decoded, z, _ = decoder(raw, "", frames, 0.)
    require(float(typed(raw, "q", (), "f")) == .5, "calibration decoder is neutral")
    neural_arm(raw, "pupil_", birth=True)
    require(not any(name.startswith("pupil_belief_") for name in raw), "calibration pupil received a readout")
    _, label, _ = own_life(raw, "pupil_", "pupil_")
    for name, expected in label.items():
        close(raw[name], expected, "own calibration " + name)
    return dict(z=z, votes=decoded["votes"], contact_color=label["contact_color"],
                contact_outcome=label["contact_outcome"])


def gate(first, survival):
    return film.scientific_gate(first, survival)


def all_control_gate(contrasts):
    require(set(contrasts) == set(CONTROLS), "all five controls required")
    decided = all(row["gate"]["passed"] is not None for row in contrasts.values())
    return {name: all(row["gate"][name] for row in contrasts.values()) if decided else None
            for name in ("passed", "first_food_passed", "survival_passed")}


def neural_arm(raw, prefix, *, birth=False):
    sensory = 4 * (2 * CFG["R"] + 1) ** 2 + 3
    neurons, steps = sensory + CFG["H"] + 6, 0 if birth else CFG["T"] // 2
    initial_zero = dict(x=film.fingerprint(np.zeros(neurons, np.float32)),
                        Wp=film.fingerprint(np.zeros((sensory, CFG["H"]), np.float32)))
    for name in ("W", "x", "Wp"):
        states = [typed(raw, prefix + stage + "_" + name + "_sha256", (32,), "u")
                  for stage in ("watch_initial", "watch_final", "solo_initial")]
        require(all(value.dtype == np.uint8 for value in states)
                and np.array_equal(states[1], states[2]), prefix + "acquired state continuity")
        if birth:
            require(np.array_equal(states[0], states[1]), prefix + "unobserved birth state")
        if name in initial_zero:
            require(np.array_equal(states[0], initial_zero[name]), prefix + "newborn neural reset")
    require(np.array_equal(typed(raw, prefix + "birth_sha256", (32,), "u"),
                           raw[prefix + "watch_initial_W_sha256"]), prefix + "own birth weights")
    require(typed(raw, prefix + "frames", (steps, sensory), "f").dtype == np.float32, prefix + "frame dtype")
    require(np.array_equal(typed(raw, prefix + "watch_motor", (steps, 5), "f"),
                           np.tile(np.array([1, 0, 0, 0, 0], np.float32), (steps, 1)))
            and not typed(raw, prefix + "watch_denergy", (steps,), "f").any(), prefix + "passive observation")
    for name, value in (("energy", CFG["e0"]), ("alive", 1), ("denergy", 0)):
        require(np.array_equal(typed(raw, prefix + "solo_initial_" + name, (1,), "fbiu"), [value]), prefix + "newborn body " + name)
    result = film.events(raw, prefix, CFG["T"])
    for name in ("first_food", "first_poison", "no_meal", "first_meal_step", "fitness"):
        same(np.asarray(raw[prefix + name]).item(), result[name], prefix + "saved " + name)
    count = int(typed(raw, prefix + "weight_change_count", (), "iu"))
    require(0 <= count <= neurons ** 2 and (not birth or count == 0), prefix + "changed watch weights")
    return result | dict(weight_change_count=count)


def readout(raw, prefix, trajectory, q):
    steps = len(trajectory["pre_grid"])
    positions = trajectory["pre_position"][:, 0]
    destinations = (positions[:, None] + MOVES[None]) % CFG["G"]
    items = trajectory["pre_grid"][np.arange(steps)[:, None], destinations[..., 0], destinations[..., 1]]
    sign = (items == 1).astype(np.float32) - (items == 2).astype(np.float32)
    expected = np.float32(3) * (np.float32(2) * np.float32(q) - np.float32(1)) * sign
    bias = typed(raw, prefix + "belief_bias", (steps, 1, 5), "f")
    close(bias[:, 0], expected, prefix + "local visible action bias")
    require(np.max(np.abs(bias), initial=0) <= 3., prefix + "bounded action bias")
    actions = typed(raw, prefix + "belief_action", (steps, 1), "iu")[:, 0]
    require(np.isin(actions, np.arange(5)).all(), prefix + "action indices")
    require(np.array_equal(typed(raw, prefix + "belief_efference", (steps, 1, 5), "f"),
                           np.eye(5, dtype=np.float32)[actions, None]), prefix + "actual action efference copy")
    expected_positions = (positions + MOVES[actions] * trajectory["active"]) % CFG["G"]
    require(np.array_equal(trajectory["positions"][:, 0], expected_positions), prefix + "sampled action displacement")
    require(np.array_equal(typed(raw, prefix + "belief_q", (steps, 1), "f"),
                           np.full((steps, 1), q, np.float32)), prefix + "fixed lifetime belief")
    require(np.array_equal(typed(raw, prefix + "belief_denergy", (steps, 1), "f"),
                           trajectory["signed_energy"]), prefix + "actual own reward")
    logits = typed(raw, prefix + "belief_unbiased_logits", (steps, 1, 5), "f")
    close(typed(raw, prefix + "belief_biased_logits", (steps, 1, 5), "f"), logits + bias,
          prefix + "biased logits")


def evaluation_family(raw, index, reliability):
    require(float(typed(raw, "lambda_value", (), "f")) == reliability, "frozen lambda in held-out family")
    result = film.family_events(raw, index)
    movie_frames = raw["observed_frames"]
    for arm in ARMS:
        prefix = arm + "_"
        outcome = neural_arm(raw, prefix, birth=arm == "birth_parent")
        _, _, trajectory = own_life(raw, prefix, prefix + "solo_")
        require(np.array_equal(raw[prefix + "birth_sha256"], raw["observed_birth_sha256"]), "paired newborn identity")
        enabled = bool(typed(raw, prefix + "readout_enabled", (), "biu", binary=True))
        if arm == "birth_parent":
            require(not enabled and not any(name.startswith(prefix + "belief_") for name in raw), "unassisted parent")
            require(float(raw[prefix + "q"]) == .5 and float(raw[prefix + "z"]) == 0
                    and np.array_equal(raw[prefix + "track_votes"], [0, 0])
                    and int(raw[prefix + "track_prediction"]) == -1
                    and raw[prefix + "track_step_color"].shape == (0,)
                    and raw[prefix + "track_step_identified"].shape == (0,), "empty parent evidence")
            q = .5
        else:
            require(enabled, arm + " assigned readout")
            frames = typed(raw, prefix + "frames", movie_frames.shape, "f")
            decoded, _, expected_q = decoder(raw, prefix, frames, reliability)
            if arm == "belief_reset":
                for name in ("frames", "watch_motor", "watch_denergy", "weight_change_count"):
                    require(np.array_equal(raw[prefix + name], raw["observed_" + name]), "reset preserves " + name)
                for state in ("W", "x", "Wp"):
                    for stage in ("watch_initial", "watch_final", "solo_initial"):
                        suffix = stage + "_" + state + "_sha256"
                        require(np.array_equal(raw[prefix + suffix], raw["observed_" + suffix]), "reset preserves acquired " + suffix)
                expected_q = .5
            if arm in ("no_demo", "scene_only"):
                require(not decoded["votes"].any() and expected_q == .5, arm + " public tracking abstains")
            q = float(typed(raw, prefix + "q", (), "f"))
            close(q, np.float32(expected_q), arm + " calibrated probability", 1e-8)
            readout(raw, prefix, trajectory, q)
        result[arm] = outcome | dict(q=q)
    return result | dict(selected_lambda=reliability)


def measurements(rows):
    result = film.measurements(rows)
    arms, contrasts = {}, {}
    for arm in ARMS:
        values = [row[arm] for row in rows]
        arms[arm] = dict(first_food=base.world_stats([row["first_food"] for row in values]),
            survival=base.world_stats([row["survival"] for row in values]),
            fitness=base.world_stats([row["fitness"] for row in values]),
            first_food_count=sum(row["first_food"] for row in values),
            survivors=sum(row["survival"] for row in values),
            first_poison_count=sum(row["first_poison"] for row in values),
            no_meal_count=sum(row["no_meal"] for row in values),
            no_meal=base.world_stats([row["no_meal"] for row in values]),
            first_meal_step=base.world_stats([row["first_meal_step"] for row in values]),
            changed_weights=base.world_stats([row["weight_change_count"] for row in values]),
            belief_probability=base.world_stats([row["q"] for row in values]))
    for arm in CONTROLS:
        first = base.world_stats([int(row["observed"]["first_food"]) - int(row[arm]["first_food"]) for row in rows])
        survival = base.world_stats([int(row["observed"]["survival"]) - int(row[arm]["survival"]) for row in rows])
        contrasts[arm] = dict(first_food_difference=first, survival_difference=survival, gate=gate(first, survival))
    return result | dict(arms=arms, contrasts=contrasts, selected_lambda=rows[0]["selected_lambda"])


class Auditor(film.Auditor):
    def completed_bundle(self, development=False):
        names = ["E123-benchmark"] if development else [f"E123-A{i}" for i in range(3)]
        count, calibrations = (4, 4) if development else (128, 32)
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
            calibration_files = {path.name for path in self.path(folder + "/calibration").glob("*.npz")}
            if (state.get("status") != "finished" or state.get("completed_families") != count
                    or state.get("expected_families") != count
                    or state.get("completed_calibration_families") != calibrations
                    or state.get("expected_calibration_families") != calibrations
                    or files != {f"family_{i:04d}.npz" for i in range(count)}
                    or calibration_files != {f"family_{i:04d}.npz" for i in range(calibrations)}
                    or not self.path(folder + "/calibration/selection.json").is_file()
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
                continue
            folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; held-out outcomes remain unopened")
        return folders

    def prior_keys(self):
        """Read old assignment records; prior outcome arrays are not unpacked."""
        self.used_masters = set()
        for experiment in (120, 121, 122):
            for directory in sorted(self.path("runs").glob(f"*_E{experiment}-*")):
                folder = directory.relative_to(self.root).as_posix()
                config_path, metrics_path = folder + "/config.json", folder + "/metrics.jsonl"
                if self.path(config_path).is_file():
                    config = self.json(config_path)
                    self.used_masters.update(config[name] for name in ("seed", "eval_seed") if name in config)
                if not self.path(metrics_path).is_file():
                    continue
                for line in self.read(metrics_path).decode("utf-8").splitlines():
                    record = json.loads(line)
                    key_arrays = []
                    if "keys" in record:
                        key_arrays.append(np.asarray(record["keys"], np.uint32))
                    if experiment == 122:
                        for field, names in (("samples", ("world_keys", "development_keys")),
                                             ("teacher", ("role_keys",))):
                            if field in record:
                                artifact = record[field]
                                content = self.read(folder + "/" + artifact["path"])
                                require(hashlib.sha256(content).hexdigest() == artifact["sha256"], "old key artifact hash")
                                with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                                    key_arrays.extend(np.asarray(arrays[name]) for name in names)
                    for keys in key_arrays:
                        require(keys.dtype == np.uint32 and keys.shape[-1] == 2, "prior random keys")
                        self.used_keys.update(key.tobytes() for key in keys.reshape(-1, 2))

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, calibration_count = (4, 4) if development else (128, 32)
        calibration_seed, eval_seed = (51, 52) if development else (2026113001 + index, 2026120101 + index)
        name = "E123-benchmark" if development else f"E123-A{index}"
        split = "development" if development else "final"
        source_path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, cfg=CFG, cfg_overrides={}, ecology=None,
            persistence="off", plastic=True, controller="rules-v0", genome_size=155,
            condition_split=split, generation=0, seed=calibration_seed, calibration_seed=calibration_seed,
            eval_seed=eval_seed, calibration_worlds=calibration_count, eval_worlds=count,
            expected_calibration_families=calibration_count, expected_families=count,
            practice_steps=150, watch_steps=150, solo_steps=300, key_order=film.KEY_ORDER,
            arms=list(ARMS), controls=list(CONTROLS), lambda_grid=list(GRID),
            calibration_input_fields=["z", "votes", "contact_color", "contact_outcome"],
            raw_schema=PROTOCOL, calibration_max_seconds=900,
            max_seconds=900 if development else 3600), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=calibration_seed, generations=0, eval_worlds=count, start="file:" + source_path,
            no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None,
            objective=None, memory_condition=None, max_train_seconds=None), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), name + " mechanism flags")
        require(calibration_seed != eval_seed and not {calibration_seed, eval_seed}.intersection(self.used_masters),
                "calibration/test/prior master seed reuse")
        self.used_masters.update((calibration_seed, eval_seed))
        subset(config["source"], dict(label=f"A{index}", path=source_path, sha256=digest), name + " source")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40
                and self.code_trees(provenance["commit"]) == self.expected_trees, name + " clean source")
        original = self.read(source_path)
        require(hashlib.sha256(original).hexdigest() == digest
                and self.read(folder + "/genome_start.npy") == original, name + " original genome bytes")
        genome = np.load(io.BytesIO(original), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), name + " genotype")
        return config

    def run(self, folder, index, development=False):
        config = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count, calibration_count = config["expected_families"], config["expected_calibration_families"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=0, completed_families=count, expected_families=count, source_label=f"A{index}",
            completed_calibration_families=calibration_count, expected_calibration_families=calibration_count,
            calibration_seed=config["calibration_seed"], eval_seed=config["eval_seed"], eval_worlds=count,
            artifact_validity="passed", scientific_result=not development), folder + " summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"]
                and 0 < summary["calibration_seconds"] <= config["calibration_max_seconds"]
                and 0 < summary["evaluation_seconds"] <= summary["elapsed_seconds"], "registered wall limits")
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()]
        require(len(records) == calibration_count + 1 + count
                and len(summary["calibration_raw_files"]) == calibration_count
                and len(summary["raw_files"]) == count, "calibration/evaluation accounting")
        elapsed, calibration_rows = 0., []
        for number in range(calibration_count):
            record = records[number]
            path = f"calibration/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + path)
            artifact = dict(index=number, path=path, sha256=self.hashes[folder + "/" + path])
            same(summary["calibration_raw_files"][number], artifact, "calibration raw manifest")
            subset(record, dict(phase="calibration", family=number, rule=number % 2,
                keys=raw["keys"].tolist(), raw_file=artifact), "calibration assignment")
            require(elapsed <= record["elapsed_seconds"] <= summary["calibration_seconds"], "calibration event order")
            elapsed = record["elapsed_seconds"]
            self.register_keys(raw["keys"], "calibration family")
            calibration_rows.append(calibration_family(raw, number))
        path = "calibration/selection.json"
        selection = self.json(folder + "/" + path)
        fitted = select_reliability(calibration_rows)
        subset(selection, dict(**fitted, status="frozen", calibration_seed=config["calibration_seed"],
            completed_families=calibration_count, expected_families=calibration_count,
            raw_files=summary["calibration_raw_files"]), "independent calibration fit")
        same(summary["calibration"], selection, "preserved calibration selection")
        frozen = dict(path=path, sha256=self.hashes[folder + "/" + path])
        same(summary["frozen_calibration"], frozen, "calibration freeze hash")
        witness = records[calibration_count]
        subset(witness, dict(phase="calibration_frozen", **frozen, selected_lambda=fitted["selected_lambda"]), "freeze witness")
        require(elapsed <= selection["elapsed_seconds"] <= witness["elapsed_seconds"]
                and witness["elapsed_seconds"] == summary["calibration_seconds"], "freeze precedes evaluation")
        elapsed, rows = witness["elapsed_seconds"], []
        for number in range(count):
            record = records[calibration_count + 1 + number]
            path = f"families/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + path)
            artifact = dict(index=number, path=path, sha256=self.hashes[folder + "/" + path])
            same(summary["raw_files"][number], artifact, "held-out raw manifest")
            subset(record, dict(phase="evaluation", family=number, rule=number % 2,
                keys=raw["keys"].tolist(), raw_file=artifact, calibration_sha256=frozen["sha256"]), "held-out assignment")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "held-out event order")
            elapsed = record["elapsed_seconds"]
            require(typed(raw, "calibration_sha256", (32,), "u").dtype == np.uint8
                    and raw["calibration_sha256"].tobytes().hex() == frozen["sha256"], "held-out frozen calibration")
            self.register_keys(raw["keys"], "held-out family")
            rows.append(evaluation_family(raw, number, fitted["selected_lambda"]))
        estimates = measurements(rows)
        same(summary["measurements"], estimates, "independent six-arm measurements")
        subset(summary["safeguards"], dict(world_keys_disjoint=True, calibration_frozen_before_evaluation=True,
            original_genome_unchanged=True, six_arm_evidence_validated=True), "verified safeguards")
        report = dict(run=folder, source_commit=config["git"]["commit"], calibration=fitted,
                      frozen_calibration=frozen, measurements=estimates)
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "benchmark scientific gate")
        else:
            verdict = all_control_gate(estimates["contrasts"])
            same(summary["gate"], verdict, "five-control intersection gate")
            same(summary["gate_status"], "passed" if verdict["passed"] else "failed", "complete scientific verdict")
            report["gate"] = verdict
        return report


def self_check():
    """Tiny artificial arrays only: no simulator, run folders or source Git reads."""
    checks = []

    def rejected(action, label):
        try:
            action()
        except base.AuditError:
            return
        raise base.AuditError(label + " was accepted")

    positive = [dict(z=1., votes=np.array([3, 0], np.int32), contact_color=color,
                     contact_outcome=1 if color == 0 else -1) for color in (0, 1, 0, 1)]
    fitted = select_reliability(positive)
    require(fitted["selected_lambda"] == 4. and fitted["contacts"] == 4
            and len(fitted["losses"]) == 9, "known positive reliability")
    negative = [{**row, "contact_outcome": -row["contact_outcome"]} for row in positive]
    require(select_reliability(negative)["selected_lambda"] == -4., "negative reliability retained")
    checks.append("positive and negative reliability")

    missing = [dict(z=0., votes=np.zeros(2, np.int32), contact_color=-1, contact_outcome=0)] * 4
    no_fit = select_reliability(missing)
    require(no_fit["selected_lambda"] == 0. and no_fit["contacts"] == 0
            and all(row["mean_log_loss"] is None and row["se_log_loss"] is None for row in no_fit["losses"]),
            "missing labels cannot be invented")
    tied = [{**row, "z": 0., "votes": np.array([1, 1], np.int32)} for row in positive]
    require(select_reliability(tied)["selected_lambda"] == 0., "zero evidence exact tie")
    require(choose_loss([{"lambda": -1., "mean_log_loss": 2.}, {"lambda": 1., "mean_log_loss": 2.},
                         {"lambda": -2., "mean_log_loss": 2.}]) == 1., "exact positive tie break")
    checks.append("missing contacts and exact tie ordering")

    frames = np.zeros((2, 103), np.float32)
    images = frames[:, :100].reshape(2, 4, 5, 5)
    images[0, 2, 2, 2], images[1, 2, 1, 2], images[0, 0, 2, 2] = 1, 1, 1
    vote = channel.track_reader(frames)["votes"]
    require(np.array_equal(vote, [1, 0]), "known public contact")
    _, q = belief(vote, 2.)
    images[:, [0, 1]] = images[:, [1, 0]]
    reverse = channel.track_reader(frames)["votes"]
    require(np.array_equal(reverse, [0, 1]), "public color swap")
    close(belief(reverse, 2.)[1], 1. - q, "belief swap symmetry", 1e-14)
    require(belief(np.zeros(2, np.int32), 4.) == (0., .5), "absence stays neutral")
    checks.append("public contact, permutation and abstention")

    cfg = dict(CFG, T=2, n_items=1)
    initial = np.zeros((16, 16), np.uint8)
    initial[3, 3], initial[10, 10] = 1, 2
    grid = np.repeat(initial[None], 2, axis=0)
    grid[:, 3, 3] = 0
    raw = dict(rule=np.asarray(0, np.int32), solo_initial_grid=initial,
        solo_initial_position=np.array([[3, 2]], np.int32),
        pupil_grid=grid, pupil_position=np.array([[[3, 3]], [[3, 4]]], np.int32),
        pupil_energy=np.array([[1.247], [1.244]], np.float32),
        pupil_food=np.array([[1], [0]], bool), pupil_poison=np.zeros((2, 1), bool),
        pupil_alive_before=np.ones((2, 1), bool), pupil_alive_after=np.ones((2, 1), bool))
    _, label, trajectory = own_life(raw, "pupil_", "pupil_", cfg)
    require(label["contact_color"] == 0 and label["contact_outcome"] == 1 and label["contact_step"] == 1,
            "own visible first contact and energy target")
    broken = dict(raw, pupil_food=np.zeros((2, 1), bool))
    rejected(lambda: own_life(broken, "pupil_", "pupil_", cfg), "forged contact event")
    poisoned = dict(raw, rule=np.asarray(1, np.int32), pupil_food=raw["pupil_poison"],
        pupil_poison=raw["pupil_food"], pupil_energy=np.array([[.747], [.744]], np.float32))
    require(own_life(poisoned, "pupil_", "pupil_", cfg)[1]["contact_outcome"] == -1, "own negative reward")
    checks.append("own arrival and energy evidence")

    bias = np.zeros((2, 1, 5), np.float32)
    bias[0, 0, 4] = 1.5
    hook = dict(belief_bias=bias, belief_action=np.full((2, 1), 4, np.int32),
                belief_efference=np.eye(5, dtype=np.float32)[[4, 4], None],
                belief_q=np.full((2, 1), .75, np.float32), belief_denergy=trajectory["signed_energy"],
                belief_unbiased_logits=np.zeros((2, 1, 5), np.float32), belief_biased_logits=bias.copy())
    readout(hook, "", trajectory, .75)
    broken_hook = dict(hook, belief_efference=np.zeros((2, 1, 5), np.float32))
    rejected(lambda: readout(broken_hook, "", trajectory, .75), "incorrect action efference")
    neutral = dict(hook, belief_q=np.full((2, 1), .5, np.float32),
                   belief_bias=np.zeros_like(bias), belief_biased_logits=np.zeros_like(bias))
    readout(neutral, "", trajectory, .5)
    checks.append("bounded local readout, neutral reset and action feedback")

    born = dict(frames=np.empty((0, 103), np.float32), watch_motor=np.empty((0, 5), np.float32),
        watch_denergy=np.empty(0, np.float32), birth_sha256=np.arange(32, dtype=np.uint8),
        solo_initial_energy=np.array([1.], np.float32), solo_initial_alive=np.array([1.], np.float32),
        solo_initial_denergy=np.array([0.], np.float32), food=np.zeros((300, 1), bool),
        poison=np.zeros((300, 1), bool), alive_before=np.ones((300, 1), bool), alive_after=np.ones((300, 1), bool),
        first_food=np.asarray(False), first_poison=np.asarray(False), no_meal=np.asarray(True),
        first_meal_step=np.asarray(301, np.int32), fitness=np.asarray(0., np.float32),
        weight_change_count=np.asarray(0, np.int32))
    states = dict(W=born["birth_sha256"], x=film.fingerprint(np.zeros(157, np.float32)),
                  Wp=film.fingerprint(np.zeros((103, 48), np.float32)))
    for state, digest in states.items():
        for stage in ("watch_initial", "watch_final", "solo_initial"):
            born[stage + "_" + state + "_sha256"] = digest.copy()
    neural_arm(born, "", birth=True)
    tampered = dict(born, solo_initial_Wp_sha256=np.zeros(32, np.uint8))
    rejected(lambda: neural_arm(tampered, "", birth=True), "changed parent neural state")
    checks.append("unassisted parent state and hash continuity")

    first, survival = dict(mean=.5, se=.1), dict(mean=0., se=.1)
    contrasts = {name: dict(gate=gate(first, survival)) for name in CONTROLS}
    require(all_control_gate(contrasts)["passed"], "all-five successful contrasts")
    contrasts["birth_parent"] = dict(gate=gate(dict(mean=.2, se=.1), survival))
    require(not all_control_gate(contrasts)["passed"], "strict two-SE first-food boundary")
    contrasts["birth_parent"] = dict(gate=gate(first, dict(mean=-.3, se=.1)))
    require(not all_control_gate(contrasts)["passed"], "survival veto")
    rejected(lambda: all_control_gate({key: value for key, value in contrasts.items() if key != "birth_parent"}),
             "missing real-parent comparison")
    checks.append("all-five gate, strict boundary and survival veto")
    return dict(status="passed", tests=len(checks), checks=checks)


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
        parser.error("--source-commit is required for artifact verification")
    auditor, report = None, dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        auditor.prior_keys()
        if not args.development:
            report["development_audit"] = auditor.run(auditor.completed_bundle(True)[0], 0, True)
        report["runs"] = [auditor.run(folder, index, args.development) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit,
            registered_code_trees=auditor.expected_trees, final_evaluation_audited=not args.development,
            limitations=["Neural recurrent/plasticity updates and random-key derivation/sampling are not replayed.",
                "Public films, own contacts, energy, action bias and efference are reconstructed from saved evidence.",
                "Matched birth/acquired neural states are verified by recorded SHA256 fingerprints.",
                "Calibration freeze order is witnessed by the immutable selection, recorded hashes and event chronology.",
                "The external belief/readout is engineered; these original genomes do not evolve in this experiment."])
        if not args.development:
            report["scientific_gate_passed"] = all(run["gate"]["passed"] for run in report["runs"])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
