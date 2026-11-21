"""Read-only NumPy audit of E125's randomized hidden-rule reversal.

The integrity verdict is separate from the all-three-lineage transfer gate.
This saved-controller diagnostic does not decide the formal task-ladder rung.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_transmission as chain

prior, film, channel, base = chain.prior, chain.film, chain.channel, chain.base
require, same, subset, typed = base.require, base.same, base.subset, base.typed
CFG, MOVES = prior.CFG, prior.MOVES
PROTOCOL = "reversal-random-v1"
ARMS = ("observed", "belief_reset", "birth_parent", "direct_baseline")
CONDITIONS = ("stationary", "reversal")
FLIPS = (100, 125, 150, 175)
WINDOWS = dict(pre=(-50, 0), early=(0, 50), late=(50, 100))
DIRECT = (
    ("runs/20261119-033803_E113-direct-s0/genome_final.npy", "0a47a3dbbb7482a42ef069cf6b54b44e511ffaf2d169d92914c10fdf75a4bb61"),
    ("runs/20261119-040602_E113-direct-s1/genome_final.npy", "0e77c3083a88c8216e57bff0667b5f594542447c14f6ae7338ec0e090b9d7d3b"),
    ("runs/20261119-041609_E113-direct-s2/genome_final.npy", "473124dfe1c2a0d77a9a4b7cdde5cb36c41f61b0664e17ede86ae88502971bbd"))
CALIBRATION = (
    ("runs/20261121-223900_E123-benchmark", "c4c0093600edbdb35c8b330d72a2b61d938b4fb756a2947e78316c116547a15c", "a79e663b8889067defe53c16c2e649ae398027426f61ebf3b2cfe0cb35200bdc"),
    ("runs/20261121-224042_E123-A0", "029a7f5ce0354da59d747358ba8000128673f25bad7dd97b32a552dc73cecccb", "b1d49b1fcc42e6a2bba6ea190106cadc216319b1ebab29750c10d5f365742c02"),
    ("runs/20261121-224439_E123-A1", "8a03cf06c3615d0d8d08f57d2fcc4a9f4bfa1ad00ff9697f37b387e3f4f2d88b", "3fe92f620be735d9a3843faf3cd651e7dfa5498e588b89fb87dbe51ae39027d9"),
    ("runs/20261121-224833_E123-A2", "7a3aa382b8e7af57f4507e768905c6e8b49abc836c632097653d13c5279bef8a", "63feeacae1fd6f3766e049914e64a975827f76d175801050a465b27328e00425"))


def identical(left, right, label):
    require(left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes() == right.tobytes(), label)


def direct_weights(flat):
    require(flat.shape == (8424,) and flat.dtype == np.float32 and np.isfinite(flat).all(), "fixed recurrent genome")
    weights = np.zeros((157, 157), np.float32)
    legal = (np.arange(157)[:, None] >= 103) & ~np.eye(157, dtype=bool)
    weights[legal] = flat
    return weights


def window_events(outcome, boundary):
    """One endpoint per assigned family; no-contact and death are not removed."""
    food, poison = outcome["food"][:, 0], outcome["poison"][:, 0]
    result = {}
    for name, (start, stop) in WINDOWS.items():
        f, p = food[boundary + start:boundary + stop], poison[boundary + start:boundary + stop]
        contact = f | p
        first = int(np.flatnonzero(contact)[0]) if contact.any() else None
        result[name] = dict(first_food=bool(first is not None and f[first]),
            first_poison=bool(first is not None and p[first]), no_meal=first is None,
            food_count=int(f.sum()), poison_count=int(p.sum()), fitness=int(f.sum()) - int(p.sum()),
            first_meal_step=51 if first is None else first + 1)
    return dict(windows=result, survival=dict(at_flip=bool(outcome["alive_before"][boundary, 0]),
        after_early=bool(outcome["alive_after"][boundary + 49, 0]),
        after_late=bool(outcome["alive_after"][boundary + 99, 0]), end=bool(outcome["alive_after"][-1, 0])),
        fitness=int(food.sum()) - int(poison.sum()))


def own_world(life, initial_rule, boundary, reversal):
    """Derive rewards from arrival pixels and the scoring-only rule schedule."""
    steps, size = CFG["T"], CFG["G"]
    outcome = film.events(life, "", steps)
    initial_grid = typed(life, "solo_initial_grid", (size, size), "iu")
    initial_position = typed(life, "solo_initial_position", (1, 2), "iu")
    grids = typed(life, "solo_grid", (steps, size, size), "iu")
    positions = typed(life, "solo_position", (steps, 1, 2), "iu")
    energy = typed(life, "solo_energy", (steps, 1), "f")
    require(np.all((energy >= 0) & (energy <= CFG["e_max"])), "own energy bounds")
    require(np.isin(initial_grid, (0, 1, 2)).all() and np.isin(grids, (0, 1, 2)).all()
            and np.sum(initial_grid == 1) == CFG["n_items"] and np.sum(initial_grid == 2) == CFG["n_items"], "own item maps")
    require(np.all((initial_position >= 0) & (initial_position < size))
            and np.all((positions >= 0) & (positions < size)), "own position bounds")
    pre_grid = np.concatenate((initial_grid[None], grids[:-1]))
    pre_position = np.concatenate((initial_position[None], positions[:-1]))
    pre_energy = np.concatenate((np.full((1, 1), CFG["e0"], np.float32), energy[:-1]))
    active = outcome["alive_before"]
    item = pre_grid[np.arange(steps), positions[:, 0, 0], positions[:, 0, 1]]
    schedule = np.full(steps, initial_rule, np.int32)
    if reversal:
        schedule[boundary:] = 1 - initial_rule
    food = (item == 1 + schedule) & active[:, 0]
    poison = (item == 2 - schedule) & active[:, 0]
    require(np.array_equal(outcome["food"][:, 0], food) and np.array_equal(outcome["poison"][:, 0], poison),
            "current-rule contacts from pre-action visible items")
    expected_energy = np.clip(pre_energy - CFG["decay"] * active
        + CFG["gain"] * food[:, None] - CFG["gain"] * poison[:, None], 0, CFG["e_max"])
    prior.close(energy, expected_energy, "own outcome energy accounting")
    require(np.array_equal(outcome["alive_after"], energy > 0), "own survival accounting")
    own_delta = (energy - pre_energy) * active
    previous_delta = np.concatenate((np.zeros((1, 1), np.float32), own_delta[:-1]))
    offsets = np.arange(-2, 3)
    rr = (pre_position[:, 0, 0, None, None] + offsets[None, :, None]) % size
    cc = (pre_position[:, 0, 1, None, None] + offsets[None, None, :]) % size
    patch = pre_grid[np.arange(steps)[:, None, None], rr, cc]
    observation = np.concatenate(((patch == 1).reshape(steps, 25), (patch == 2).reshape(steps, 25),
        np.zeros((steps, 50), np.float32), pre_energy - np.float32(1),
        np.float32(4) * previous_delta, np.ones((steps, 1), np.float32)), axis=1).astype(np.float32)[:, None]
    return outcome, dict(pre_grid=pre_grid, pre_position=pre_position, positions=positions,
                        signed_energy=own_delta, active=active, observation=observation)


def stats(values):
    values = list(values)
    return base.world_stats(values) if values else dict(mean=None, se=None)


def transfer_gate(late, recovery, late_difference, survival_difference):
    if any(value["se"] is None for value in (late, recovery, late_difference, survival_difference)):
        return dict(passed=None, late_level_passed=bool(late["mean"] >= .60), late_chance_passed=None,
                    recovery_passed=None, accuracy_noninferiority_passed=None, survival_noninferiority_passed=None)
    criteria = dict(late_level_passed=late["mean"] >= .60,
        late_chance_passed=late["mean"] - .50 > 2 * late["se"],
        recovery_passed=recovery["mean"] > 2 * recovery["se"],
        accuracy_noninferiority_passed=late_difference["mean"] - 2 * late_difference["se"] > -.10,
        survival_noninferiority_passed=survival_difference["mean"] - 2 * survival_difference["se"] > -.10)
    return dict(passed=all(criteria.values()), **criteria)


def paired_contrasts(rows, arm):
    stationary = [row[arm]["stationary"] for row in rows]
    reversal = [row[arm]["reversal"] for row in rows]
    first = lambda row, when: int(row["windows"][when]["first_food"])
    recovery = stats([(first(r, "late") - first(r, "early")) - (first(s, "late") - first(s, "early"))
                         for s, r in zip(stationary, reversal)])
    differences = {window + "_first_food_difference": stats([first(r, window) - first(s, window)
                    for s, r in zip(stationary, reversal)]) for window in WINDOWS}
    survival = stats([int(r["survival"]["end"]) - int(s["survival"]["end"]) for s, r in zip(stationary, reversal)])
    return differences | dict(recovery_difference=recovery, end_survival_difference=survival)


def condition_summary(values):
    windows = {}
    for window in WINDOWS:
        observations = [row["windows"][window] for row in values]
        food = np.array([row["food_count"] for row in observations])
        poison = np.array([row["poison_count"] for row in observations])
        fields = ("first_food", "first_poison", "no_meal", "food_count", "poison_count", "fitness", "first_meal_step")
        windows[window] = {field: stats(row[field] for row in observations) for field in fields}
        windows[window].update(accuracy=base.ratio_stats(food, food + poison) if values else dict(mean=None, se=None),
            food_encounters=int(food.sum()), poison_encounters=int(poison.sum()), encounters=int((food + poison).sum()),
            assigned_families=len(values))
    return dict(families=len(values), windows=windows, fitness=stats(row["fitness"] for row in values),
                survival={point: stats(row["survival"][point] for row in values)
                          for point in ("at_flip", "after_early", "after_late", "end")})


def measurements(rows):
    arms = {}
    for arm in ARMS:
        conditions = {}
        for condition in CONDITIONS:
            values = [row[arm][condition] for row in rows]
            conditions[condition] = condition_summary(values)
            conditions[condition]["end_survivor_only"] = condition_summary([value for value in values if value["survival"]["end"]])
            conditions[condition]["alive_at_flip_only"] = condition_summary([value for value in values if value["survival"]["at_flip"]])
        contrasts = paired_contrasts(rows, arm)
        verdict = transfer_gate(conditions["reversal"]["windows"]["late"]["first_food"],
            contrasts["recovery_difference"], contrasts["late_first_food_difference"], contrasts["end_survival_difference"])
        arms[arm] = dict(conditions=conditions, reversal_minus_stationary=contrasts, gate=verdict)
    comparisons = {}
    for name, left, right in (("within_social", "observed", "belief_reset"), ("within_birth", "birth_parent", "direct_baseline")):
        comparisons[name] = {}
        for condition in CONDITIONS:
            comparisons[name][condition] = dict(windows={window: {field + "_difference": stats(
                float(row[left][condition]["windows"][window][field]) - float(row[right][condition]["windows"][window][field])
                for row in rows) for field in ("first_food", "fitness")} for window in WINDOWS},
                end_survival_difference=stats(int(row[left][condition]["survival"]["end"])
                    - int(row[right][condition]["survival"]["end"]) for row in rows))
    return dict(families=len(rows), primary="observed", selected_lambda=rows[0]["lambda_value"], arms=arms,
        gate=arms["observed"]["gate"], **comparisons,
        assignment_counts={f"{rule}:{flip}": sum(row["rule"] == rule and row["flip_step"] == flip for row in rows)
                           for rule in (0, 1) for flip in FLIPS}, neutral_selected_birth_duplicates="birth_parent",
        preflip_identity=dict(all_recorded_fields_identical=True,
            limitation="Per-frame W and Wp were not recorded; their intermediate identity is not independently replayed."))


def family(raw, index, rule, boundary, original, direct_flat, weights, reliability):
    for name, expected in (("family_index", index), ("rule", rule), ("flip_step", boundary),
        ("steps", 300), ("agents", 1), ("grid_size", 16), ("radius", 2), ("hidden_units", 48)):
        require(int(typed(raw, name, (), "iu")) == expected, "registered family " + name)
    require(float(typed(raw, "initial_energy", (), "f")) == 1.
            and float(typed(raw, "lambda_value", (), "f")) == reliability, "fixed body/calibration")
    identical(typed(raw, "original_genome", (155,), "f"), original, "fixed original genotype")
    identical(typed(raw, "direct_genome", (8424,), "f"), direct_flat, "fixed direct genotype")
    identical(typed(raw, "direct_weights", (157, 157), "f"), weights, "unclipped direct decoding")
    require(typed(raw, "keys", (4, 2), "u").dtype == np.uint32, "family random-key dtype")
    movie, cameras, _, _ = film.reconstruct_movie(raw)
    identical(raw["observed_frames"], movie, "natural initial-rule demonstration")
    identical(raw["camera_positions"], cameras, "public camera reconstruction")
    _, _, q = prior.decoder(raw, "", movie, reliability)
    prior.close(raw["q"], np.float32(q), "calibrated initial-rule belief", 1e-8)
    zero_hash = dict(x=film.fingerprint(np.zeros(157, np.float32)),
                     Wp=film.fingerprint(np.zeros((103, 48), np.float32)))
    for arm in ARMS[:-1]:
        prefix = arm + "_"
        length = 0 if arm == "birth_parent" else 150
        expected = movie if length else np.empty((0, 103), np.float32)
        identical(typed(raw, prefix + "frames", (length, 103), "f"), expected, arm + " assigned preparation")
        identical(typed(raw, prefix + "watch_motor", (length, 5), "f"),
                  np.tile(np.array([1, 0, 0, 0, 0], np.float32), (length, 1)), arm + " passive action clamp")
        require(not typed(raw, prefix + "watch_denergy", (length,), "f").any(), arm + " no teacher reward")
        require(float(typed(raw, prefix + "q", (), "f")) == (float(raw["q"]) if arm == "observed" else .5), arm + " assigned belief")
        for state in ("W", "x", "Wp"):
            initial, final = (typed(raw, prefix + "watch_" + stage + "_" + state + "_sha256", (32,), "u")
                              for stage in ("initial", "final"))
            require(initial.dtype == final.dtype == np.uint8, "preparation hash dtype")
            identical(initial, raw["observed_watch_initial_" + state + "_sha256"], "matched own birth")
            if state in zero_hash:
                identical(initial, zero_hash[state], "initial neural reset")
            if arm == "birth_parent":
                identical(initial, final, "no observation updates for original parent")
            elif arm == "belief_reset":
                identical(final, raw["observed_watch_final_" + state + "_sha256"], "reset preserves acquired neural state")
    result = dict(rule=rule, flip_step=boundary, lambda_value=reliability)
    traces = ("grid", "position", "energy", "signal", "alive", "surprise", "food", "poison",
              "alive_before", "alive_after", "observation", "action", "action_key", "efference", "activity", "denergy")
    shapes = dict(observation=(300, 1, 103), action=(300, 1), action_key=(300, 2),
                  efference=(300, 1, 5), activity=(300, 1, 157), denergy=(300, 1),
                  signal=(300, 1), alive=(300, 1), surprise=(300,))
    for arm in ARMS:
        result[arm] = {}
        for condition in CONDITIONS:
            prefix = arm + "_" + condition + "_"
            life = chain.life_fields(raw, prefix)
            life.update(solo_initial_grid=raw["solo_initial_grid"], solo_initial_position=raw["solo_initial_position"])
            life.update({"solo_" + field: life[field] for field in ("grid", "position", "energy")})
            outcome, trajectory = own_world(life, rule, boundary, condition == "reversal")
            for name, shape in shapes.items():
                typed(life, name, shape, "iu" if name in ("action", "action_key") else "fbiu")
            require(life["action_key"].dtype == np.uint32, "actual action random-key dtype")
            identical(life["action_key"], raw["observed_stationary_action_key"], "matched action random streams")
            identical(life["observation"], trajectory["observation"], "own observations contain no hidden boundary cue")
            identical(life["activity"][:, :, :103], life["observation"], "actual sensor activity")
            identical(life["denergy"], trajectory["signed_energy"], "actual own energy delta")
            require(np.array_equal(life["alive"], outcome["alive_after"]), "recorded body survival")
            actions = life["action"][:, 0]
            require(np.isin(actions, np.arange(5)).all(), "sampled action bounds")
            position = (trajectory["pre_position"][:, 0] + MOVES[actions] * trajectory["active"]) % 16
            require(np.array_equal(life["position"][:, 0], position), "sampled action movement")
            identical(life["efference"], np.eye(5, dtype=np.float32)[actions, None], "sampled-action feedback")
            identical(life["activity"][:, :, -6:-1], life["efference"], "actual neural motor feedback")
            prior.close(life["signal"], life["activity"][:, :, -1] * trajectory["active"], "actual public signal")
            require(np.all(np.abs(life["signal"]) <= 1) and np.all(life["surprise"] >= 0), "signal/surprise bounds")
            for name, expected in (("energy", 1.), ("alive", 1.), ("denergy", 0.)):
                require(np.array_equal(typed(life, "initial_" + name, (1,), "fbiu"), [expected]), "fresh solo body")
            for state in ("W", "x", "Wp"):
                initial, final = (typed(life, stage + "_" + state + "_sha256", (32,), "u")
                                  for stage in ("initial", "final"))
                require(initial.dtype == final.dtype == np.uint8, "actual neural-state hashes")
                if arm == "direct_baseline":
                    expected = film.fingerprint(weights) if state == "W" else zero_hash[state]
                    identical(initial, expected, "direct birth state")
                    if state != "x":
                        identical(final, expected, "fixed direct weights/predictor")
                else:
                    identical(initial, raw[f"{arm}_watch_final_{state}_sha256"], "transmitted own prepared state")
            identical(life["final_x_sha256"], film.fingerprint(life["activity"][-1, 0]), "terminal activity fingerprint")
            if arm in ("observed", "belief_reset"):
                prior.readout(life, "", trajectory, float(raw[arm + "_q"]))
                for name in ("action", "efference", "denergy"):
                    identical(life["belief_" + name], life[name], "belief trace uses actual " + name)
            else:
                require(not any(name.startswith("belief_") for name in life), "unassisted birth panel")
            if arm == "direct_baseline":
                prior.close(life["surprise"], np.mean(life["observation"] ** 2, axis=(1, 2)), "fixed predictor surprise")
            endpoints = window_events(outcome, boundary)
            for field in endpoints["windows"]["pre"]:
                expected = np.asarray([endpoints["windows"][window][field] for window in WINDOWS],
                                      bool if field in ("first_food", "first_poison", "no_meal") else np.int32)
                identical(life["window_" + field], expected, "assigned-family window " + field)
            identical(life["boundary_survival"], np.asarray(list(endpoints["survival"].values()), bool), "boundary survival")
            require(float(life["fitness"]) == endpoints["fitness"], "lifetime encounters")
            result[arm][condition] = endpoints
        left, right = arm + "_stationary_", arm + "_reversal_"
        for name in traces:
            identical(raw[left + name][:boundary], raw[right + name][:boundary], arm + " pre-boundary " + name)
        for name in ("observation", "action", "action_key", "efference", "activity", "alive_before", "position", "grid"):
            identical(raw[left + name][boundary], raw[right + name][boundary], arm + " boundary pre-reward " + name)
        for state in ("W", "x", "Wp"):
            identical(raw[left + "initial_" + state + "_sha256"], raw[right + "initial_" + state + "_sha256"], "paired initial neural state")
        if arm in ("observed", "belief_reset"):
            for field in ("q", "action", "bias", "unbiased_logits", "biased_logits", "efference", "denergy"):
                identical(raw[left + "belief_" + field][:boundary], raw[right + "belief_" + field][:boundary], "paired pre-boundary belief path")
        require(bool(typed(raw, arm + "_preflip_identical", (), "biu", binary=True)), "saved pre-boundary verdict")
    return result


def assignment_table(raw, count):
    require(set(raw) == {"key", "permutation", "table"}, "assignment artifact fields")
    require(typed(raw, "key", (2,), "u").dtype == np.uint32, "dedicated assignment key")
    permutation = typed(raw, "permutation", (count,), "iu")
    table = typed(raw, "table", (count, 2), "iu")
    require(permutation.dtype == table.dtype == np.int32
            and np.array_equal(np.sort(permutation), np.arange(count)), "assignment permutation")
    require(count in (4, 128), "registered family count")
    cells = np.asarray([(0, 100), (1, 125), (0, 150), (1, 175)] if count == 4 else
        [(rule, flip) for rule in (0, 1) for flip in FLIPS for _ in range(16)], np.int32)
    identical(table, cells[permutation], "randomized balanced rule/time cells")
    return table


class Auditor(chain.Auditor):
    def completed_bundle(self, development=False):
        names = ["E125-benchmark"] if development else [f"E125-A{i}" for i in range(3)]
        count, folders, pending = 4 if development else 128, [], []
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
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no held-out outcomes opened")
        return folders

    def prior_keys(self):
        super().prior_keys()
        for directory in sorted(self.path("runs").glob("*_E124-*")):
            folder = directory.relative_to(self.root).as_posix()
            if self.path(folder + "/config.json").is_file():
                config = self.json(folder + "/config.json")
                self.used_masters.update(config[field] for field in ("seed", "eval_seed") if field in config)
            if self.path(folder + "/metrics.jsonl").is_file():
                for line in self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines():
                    keys = np.asarray(json.loads(line)["keys"], np.uint32)
                    require(keys.shape == (4, 2, 2), "earlier chain assignment keys")
                    self.used_keys.update(key.tobytes() for key in keys.reshape(-1, 2))

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master = (4, 2026120300) if development else (128, 2026120301 + index)
        split, name = ("development", "E125-benchmark") if development else ("final", f"E125-A{index}")
        source_path, source_sha = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={},
            ecology=None, persistence="off", generation=0, seed=master, eval_seed=master, eval_worlds=count,
            expected_families=count, arms=list(ARMS), conditions=list(CONDITIONS), primary="observed",
            key_order=film.KEY_ORDER, flip_steps=list(FLIPS), practice_steps=150, watch_steps=150, solo_steps=300,
            windows={key: list(value) for key, value in WINDOWS.items()}, max_seconds=900 if development else 3600,
            raw_schema=PROTOCOL, assignment_columns=["rule", "flip_step"], assignment_stream_tag=0xE125,
            fair_birth_panel=["birth_parent", "direct_baseline"], selected_neutral_birth_reuses="birth_parent",
            formal_ladder_status="not_decided"), "registered world transfer")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split, seed=master,
            generations=0, eval_worlds=count, start="file:" + source_path, no_plastic=False, allow_dirty=False,
            quick=False, resume=None, cfg=None, objective=None, memory_condition=None, max_train_seconds=None,
            pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10), "registered arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), "no additional mechanism")
        require(master not in self.used_masters, "old/development/final master overlap")
        self.used_masters.add(master)
        git = config["git"]
        require(git.get("dirty") is False and len(git.get("commit", "")) == 40
                and self.code_trees(git["commit"]) == self.expected_trees, "clean frozen scientific source")
        subset(config["source"], dict(label=f"A{index}", training_seed=index,
            path=source_path, sha256=source_sha), "original genome source")
        original_bytes = self.pinned(dict(path=source_path, sha256=source_sha))
        require(self.read(folder + "/genome_start.npy") == original_bytes, "unchanged original genome copy")
        original = np.load(io.BytesIO(original_bytes), allow_pickle=False)
        require(original.shape == (155,) and original.dtype == np.float32 and np.isfinite(original).all(), "original genotype")
        direct_path, direct_sha = DIRECT[index]
        same(config["direct_source"], dict(label=f"direct{index}", training_seed=index,
            path=direct_path, sha256=direct_sha), "fair direct genotype binding")
        direct_bytes = self.pinned(dict(path=direct_path, sha256=direct_sha))
        require(self.read(folder + "/direct_genome.npy") == direct_bytes, "unchanged direct genome copy")
        direct_flat = np.load(io.BytesIO(direct_bytes), allow_pickle=False)
        weights = direct_weights(direct_flat)
        cal_folder, config_sha, selection_sha = CALIBRATION[0 if development else index + 1]
        calibration = config["calibration_source"]
        subset(calibration, dict(folder=cal_folder, commit="b7a6dfc91b08b8cc44dd66510752e8f6b8b5e758",
            config=dict(path=cal_folder + "/config.json", sha256=config_sha),
            selection=dict(path=cal_folder + "/calibration/selection.json", sha256=selection_sha)), "immutable E123 binding")
        selection, reliability = self.calibration(calibration, index, development, config["source"])
        frozen = dict(path="calibration/selection.json", sha256=selection_sha)
        same(config["frozen_calibration"], frozen, "frozen calibration manifest")
        require(self.read(folder + "/" + frozen["path"]) == selection
                and config["selected_lambda"] == reliability == 4., "unchanged learned reliability")
        same(config["assignments"]["path"], "assignments.npz", "assignment artifact location")
        assignments = self.arrays(folder + "/assignments.npz")
        same(config["assignments"]["sha256"], self.hashes[folder + "/assignments.npz"], "frozen randomized schedule")
        table = assignment_table(assignments, count)
        self.register_keys(assignments["key"][None], "dedicated assignment stream")
        return config, original, direct_flat, weights, reliability, table

    def run(self, folder, index, development=False):
        config, original, direct_flat, weights, reliability, table = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count = config["expected_families"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=0, source_label=f"A{index}", completed_families=count, expected_families=count,
            eval_seed=config["seed"], eval_worlds=count, scientific_result=not development, artifact_validity="passed",
            primary="observed", formal_ladder_status="not_decided", selected_lambda=reliability,
            calibration_source=config["calibration_source"], frozen_calibration=config["frozen_calibration"],
            assignments=config["assignments"]), "completed transfer summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "registered time allowance")
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()]
        require(len(records) == len(summary["raw_files"]) == count, "every assigned family")
        rows, elapsed = [], 0.
        for number, record in enumerate(records):
            path = f"families/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + path)
            artifact = dict(index=number, path=path, sha256=self.hashes[folder + "/" + path])
            same(summary["raw_files"][number], artifact, "raw family manifest")
            for field, digest in (("calibration", config["frozen_calibration"]["sha256"]),
                                  ("assignments", config["assignments"]["sha256"])):
                value = typed(raw, field + "_sha256", (32,), "u")
                require(value.dtype == np.uint8 and value.tobytes().hex() == digest, "family frozen " + field)
            rule, flip = (int(value) for value in table[number])
            subset(record, dict(family=number, rule=rule, flip_step=flip, keys=raw["keys"].tolist(),
                raw_file=artifact, calibration_sha256=config["frozen_calibration"]["sha256"],
                assignments_sha256=config["assignments"]["sha256"]), "randomized family assignment")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "family chronology")
            elapsed = record["elapsed_seconds"]
            self.register_keys(typed(raw, "keys", (4, 2), "u"), "new teacher/solo world and birth keys")
            rows.append(family(raw, number, rule, flip, original, direct_flat, weights, reliability))
            self.register_keys(raw["observed_stationary_action_key"], "fresh recorded action draws")
        estimates = measurements(rows)
        verdict = estimates.pop("gate")
        same(summary["measurements"], estimates, "independent all-assigned paired estimates")
        subset(summary["safeguards"], dict(world_keys_disjoint=True, calibration_source_unchanged=True,
            sources_unchanged=True, balanced_schedule_preserved=True, paired_traces_validated=True), "verified safeguards")
        report = dict(run=folder, source_commit=config["git"]["commit"], measurements=estimates)
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "benchmark scientific gate")
        else:
            same(summary["gate"], verdict, "complete transfer intersection")
            same(summary["gate_status"], "passed" if verdict["passed"] else "failed", "transfer verdict")
            report["gate"] = verdict
        return report


def self_check():
    """Artificial arrays only; no simulator, RNG generation, Git or run reads."""
    checks = []

    def rejected(action, description):
        try:
            action()
        except base.AuditError:
            return
        raise base.AuditError("accepted " + description)

    cells = np.asarray([(rule, flip) for rule in (0, 1) for flip in FLIPS for _ in range(16)], np.int32)
    permutation = np.arange(127, -1, -1, dtype=np.int32)
    manifest = dict(key=np.array([1, 2], np.uint32), permutation=permutation, table=cells[permutation])
    assignment_table(manifest, 128)
    corrupt = dict(manifest, permutation=np.zeros(128, np.int32))
    rejected(lambda: assignment_table(corrupt, 128), "reused assignment position")
    corrupt = dict(manifest, table=manifest["table"].copy())
    corrupt["table"][0] = [0, 100]
    rejected(lambda: assignment_table(corrupt, 128), "altered rule/time cell")
    checks.append("balanced immutable assignment permutation")

    f = 100
    food, poison = np.zeros((300, 1), bool), np.zeros((300, 1), bool)
    food[[f - 1, f + 49, f + 99, f + 100], 0] = True
    poison[[f, f + 50], 0] = True
    outcome = dict(food=food, poison=poison, alive_before=np.ones((300, 1), bool), alive_after=np.ones((300, 1), bool))
    windows = window_events(outcome, f)
    require(windows["windows"]["pre"]["first_food"] and windows["windows"]["pre"]["first_meal_step"] == 50,
            "last pre-boundary item")
    require(all(windows["windows"][name]["first_poison"] and windows["windows"][name]["food_count"] == 1
                and windows["windows"][name]["poison_count"] == 1 for name in ("early", "late")), "half-open boundary windows")
    empty = {name: np.zeros((300, 1), bool) for name in ("food", "poison", "alive_before", "alive_after")}
    absent = window_events(empty, f)
    require(all(row["no_meal"] and not row["first_food"] and row["first_meal_step"] == 51
                for row in absent["windows"].values()), "death and missing contacts remain failures")
    estimate = condition_summary([windows, absent])
    require(estimate["families"] == 2 and estimate["windows"]["pre"]["first_food"]["mean"] == .5,
            "all assigned families retained")
    require(condition_summary([absent])["windows"]["late"]["accuracy"]["mean"] is None,
            "no contact-conditioned invented zero")
    checks.append("exact boundary windows, deaths and missing denominators")

    grid = np.zeros((16, 16), np.uint8)
    grid.flat[:23], grid.flat[24:48], grid[8, 8] = 1, 2, 1
    energy, alive = np.float32(1), True
    energies, before, after, food, poison = [], [], [], [], []
    for turn in range(300):
        hit, bad = alive and turn < f, alive and turn >= f
        before.append([alive])
        energy = np.float32(np.clip(energy - np.float32(.003) * alive
            + np.float32(.25) * hit - np.float32(.25) * bad, 0, 2))
        alive = bool(energy > 0)
        food.append([hit])
        poison.append([bad])
        energies.append([energy])
        after.append([alive])
    life = dict(solo_initial_grid=grid, solo_initial_position=np.array([[8, 8]], np.int32),
        solo_grid=np.repeat(grid[None], 300, axis=0), solo_position=np.tile(np.array([[[8, 8]]], np.int32), (300, 1, 1)),
        solo_energy=np.asarray(energies, np.float32), food=np.asarray(food, bool), poison=np.asarray(poison, bool),
        alive_before=np.asarray(before, bool), alive_after=np.asarray(after, bool))
    events, trace = own_world(life, 0, f, True)
    require(events["food"][f - 1, 0] and events["poison"][f, 0], "first flipped reward index")
    require(trace["observation"].shape == (300, 1, 103)
            and not trace["observation"][:, :, 50:100].any(), "no hidden time/change sensory channels")
    corrupt = dict(life, food=life["food"].copy(), poison=life["poison"].copy())
    corrupt["food"][f], corrupt["poison"][f] = True, False
    rejected(lambda: own_world(corrupt, 0, f, True), "old label at first flipped contact")
    checks.append("visible-contact rule schedule and actual energy accounting")

    import copy
    rows = [{"observed": {condition: copy.deepcopy(absent) for condition in CONDITIONS}} for _ in range(2)]
    rows[1]["observed"]["reversal"]["windows"]["late"]["first_food"] = True
    rows[1]["observed"]["stationary"]["windows"]["early"]["first_food"] = True
    recovery = paired_contrasts(rows, "observed")["recovery_difference"]
    same(recovery, dict(mean=1., se=1.), "paired difference-in-differences uncertainty")
    checks.append("paired family recovery error rather than independent-arm errors")

    late, recovery, margin = dict(mean=.8, se=.02), dict(mean=.3, se=.05), dict(mean=0., se=.02)
    require(transfer_gate(late, recovery, margin, margin)["passed"], "known gate pass")
    require(not transfer_gate(late, dict(mean=.1, se=.05), margin, margin)["passed"], "strict recovery boundary")
    require(not transfer_gate(late, recovery, dict(mean=-.1, se=0.), margin)["passed"], "strict accuracy margin")
    require(not transfer_gate(late, recovery, margin, dict(mean=-.1, se=0.))["passed"], "strict survival margin")
    require(not transfer_gate(dict(mean=.59, se=.01), recovery, margin, margin)["passed"], "absolute late threshold")
    checks.append("unchanged prospective intersection and ten-point margins")

    flat = np.arange(8424, dtype=np.float32)
    weights = direct_weights(flat)
    require(not weights[:103].any() and not np.diag(weights).any() and weights.max() == 8423.,
            "direct genome decode without clipping or sensor/self weights")
    rejected(lambda: identical(np.array([0.], np.float32), np.array([-0.], np.float32), "changed bits"),
             "signed-zero bit mismatch")
    checks.append("direct genotype layout and exact recorded-trace identity")
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
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            final_evaluation_audited=not args.development, formal_ladder_status="not_decided",
            limitations=["Neural recurrent/plasticity updates, random-key derivation, permutation generation and sampling are not replayed.",
                "All recorded pre-boundary world, observation, action and neural-activity traces match byte for byte.",
                "Per-frame W/Wp are absent; fingerprints establish initial/terminal states, not intermediate weight trajectories.",
                "Own sensory inputs, event labels, energy changes, sampled-action feedback and the fixed belief readout are reconstructed.",
                "Social preparation/reset and fair birth-state comparisons are reported separately; no prepared-versus-direct superiority claim.",
                "The genomes and E123 calibration stay fixed; this diagnostic cannot decide the formal evolution-and-lesion ladder gate."])
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
