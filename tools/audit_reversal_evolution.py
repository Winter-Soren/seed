"""Independent E126 arithmetic/evidence audit; PRNG replay, no simulator.

Completed training prefixes can be audited without finishing the study. The
scientific gate remains undecided until all six registered endpoints and all
three complete paired evaluation panels are available.
"""
import argparse
import hashlib
import io
import json
import re
import importlib.metadata
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_reversal as reversal
from audit_social_observer_evolution import average_ranks
from audit_shelter_evolution import close

base, film, prior = reversal.base, reversal.film, reversal.prior
require, same, subset, typed = base.require, base.same, base.subset, base.typed
CFG = reversal.CFG
PROTOCOL = "reversal-evolution-v1"
CONTROLLERS = ("plastic", "direct")
DIMENSIONS = dict(plastic=155, direct=8424)
FLIPS = reversal.FLIPS
EVENTS = ("food", "poison", "alive_before", "alive_after")
ARMS = ("plastic_stationary", "plastic_reversal", "lesion_stationary", "lesion_reversal", "direct_stationary", "direct_reversal")
PREFIX_FIELDS = ("grid", "position", "energy", "signal", "alive", "surprise", *EVENTS,
                 "observation", "action", "action_key", "efference", "activity", "denergy", "hebbian_enabled", "predictor_enabled")


def digest(value):
    return np.frombuffer(hashlib.sha256(np.asarray(value).tobytes()).digest(), np.uint8)


def random_runtime():
    # Replay only the pinned random primitives, never import the Seed simulator.
    import jax
    import jax.numpy as jnp
    require(jax.default_backend() == "cpu", "registered CPU PRNG runtime required")
    return jax, jnp


def regenerate_half(samples, pop, dimension):
    key = typed(samples, "noise_key", (2,), "u")
    require(key.dtype == np.uint32, "archived noise key")
    jax, jnp = random_runtime()
    half = np.asarray(jax.random.normal(jnp.asarray(key), (pop // 2, dimension), dtype=jnp.float32))
    expected = typed(samples, "half_eps_sha256", (32,), "u")
    require(expected.dtype == np.uint8 and np.array_equal(digest(half), expected), "same-runtime half-noise regeneration")
    return half


def unpack_events(samples, pop=128, worlds=4, steps=300):
    shape = (pop, worlds, 2, steps, 1)
    require(np.array_equal(typed(samples, "event_shape", (5,), "iu"), shape), "packed event shape")
    bits = int(np.prod(shape))
    result = {}
    for name in EVENTS:
        packed = typed(samples, name + "_packed", ((bits + 7) // 8,), "u")
        require(packed.dtype == np.uint8, "packed event dtype")
        unpacked = np.unpackbits(packed, bitorder="little")
        require(not unpacked[bits:].any(), "nonzero packed padding")
        result[name] = unpacked[:bits].reshape(shape).astype(bool)
    food, poison, before, after = (result[name] for name in EVENTS)
    require(before[..., 0, :].all() and np.array_equal(before[..., 1:, :], after[..., :-1, :])
            and not np.any(after & ~before) and not np.any(food & poison)
            and not np.any((food | poison) & ~before), "assigned training event/death timing")
    return result


def late_scores(events, flips):
    food, poison = events["food"], events["poison"]
    counts = np.empty((*food.shape[:3], 2), np.int32)
    for world, boundary in enumerate(flips):
        counts[:, world, :, 0] = food[:, world, :, int(boundary) + 50:int(boundary) + 100, :].sum(axis=(-2, -1))
        counts[:, world, :, 1] = poison[:, world, :, int(boundary) + 50:int(boundary) + 100, :].sum(axis=(-2, -1))
    returns = counts[..., 0] - counts[..., 1]
    return returns.min(axis=2).astype(np.float32), returns, counts


def optimizer_transition(previous, samples, controller, pop=128, worlds=4, steps=300):
    dimension = DIMENSIONS[controller]
    half = regenerate_half(samples, pop, dimension)
    eps = np.concatenate((half, -half))
    candidates = np.asarray(previous["mu"] + np.float32(.1) * eps, np.float32)
    hashes = typed(samples, "candidate_sha256", (pop, 32), "u")
    require(hashes.dtype == np.uint8 and np.array_equal(hashes, np.stack([digest(row) for row in candidates])),
            "actual candidates differ from mean plus antithetic perturbations")
    flips = typed(samples, "flip_steps", (worlds,), "iu")
    permutation = typed(samples, "permutation", (worlds,), "iu")
    require(worlds == 4 and np.array_equal(np.sort(permutation), np.arange(4))
            and np.array_equal(flips, np.asarray(FLIPS)[permutation]), "balanced randomized training boundaries")
    events = unpack_events(samples, pop, worlds, steps)
    fitness, returns, counts = late_scores(events, flips)
    require(np.array_equal(typed(samples, "fitness", (pop, worlds), "f"), fitness), "minimum before averaging base worlds")
    require(np.array_equal(typed(samples, "rule_late_return", (pop, worlds, 2), "iu"), returns)
            and np.array_equal(typed(samples, "rule_window_counts", (pop, worlds, 2, 3, 2), "iu")[..., 2, :], counts),
            "late window event arithmetic")
    require(np.array_equal(typed(samples, "survived", (pop, worlds), "f"), events["alive_after"][..., -1, 0].mean(2)),
            "all-assigned training survival")
    scores = fitness.mean(axis=1, dtype=np.float32)
    ranks = average_ranks(scores)
    close(typed(samples, "rank_weights", (pop,), "f"), ranks, "average tied centered ranks")
    informative = bool(np.any(scores != scores[0]))
    expected = {name: previous[name].copy() for name in ("mu", "m", "v", "t")}
    if informative:
        gradient = (ranks[:, None] * eps).sum(0) / np.float32(pop * .1) - np.float32(.005) * previous["mu"]
        step = int(previous["t"]) + 1
        expected["m"] = np.float32(.9) * previous["m"] + np.float32(.1) * gradient
        expected["v"] = np.float32(.999) * previous["v"] + np.float32(.001) * gradient ** 2
        expected["mu"] = previous["mu"] + np.float32(.1) * (expected["m"] / np.float32(1 - .9 ** step)) / (
            np.sqrt(expected["v"] / np.float32(1 - .999 ** step)) + np.float32(1e-8))
        expected["t"] = np.asarray(step, previous["t"].dtype)
    return expected, scores, informative


def initial_state(state, controller, master, development_tag=None):
    require(not state["m"].any() and not state["v"].any() and int(state["t"]) == 0, "zero fresh Adam state")
    jax, jnp = random_runtime()
    key, initial = jax.random.split(jax.random.PRNGKey(master))
    if controller == "direct":
        flat = np.asarray(jax.random.normal(initial, (8424,)) / jnp.sqrt(float(156)))
    else:
        hebb = np.asarray(np.float32(.1) * jax.random.normal(initial, (5, 5, 4))).ravel()
        flat = np.concatenate((np.full(25, -1, np.float32), hebb, np.array([-1.5, -3., 1., 0., 0.], np.float32),
                               np.zeros(25, np.float32)))
    if development_tag is not None:
        key = jax.random.fold_in(key, development_tag)
    require(np.array_equal(state["mu"], flat) and np.array_equal(state["key"], np.asarray(key)),
            "fresh same-runtime blank/direct initializer and aligned ES stream")


def summarize_life(events, boundary):
    return reversal.window_events(events, boundary)


def prefix_hash(life, stop):
    fingerprint = hashlib.sha256()
    for name in PREFIX_FIELDS:
        value = np.ascontiguousarray(life[name][:stop])
        fingerprint.update(name.encode() + b"\0" + value.dtype.str.encode() + b"\0")
        fingerprint.update(np.asarray(value.shape, np.int64).tobytes())
        fingerprint.update(value.tobytes())
    return np.frombuffer(fingerprint.digest(), np.uint8)


def evaluation_world(raw, index, boundary, plastic, direct):
    for name, expected in (("base_index", index), ("flip_step", boundary), ("steps", 300),
                           ("grid_size", 16), ("radius", 2), ("hidden_units", 48)):
        require(int(typed(raw, name, (), "iu")) == expected, "world metadata " + name)
    require(float(typed(raw, "initial_energy", (), "f")) == 1., "birth energy")
    weights = reversal.direct_weights(direct)
    for name, value in (("plastic_genome_sha256", plastic), ("direct_genome_sha256", direct), ("direct_weights_sha256", weights)):
        reversal.identical(typed(raw, name, (32,), "u"), digest(value), "frozen endpoint " + name)
    keys = typed(raw, "keys", (2, 2), "u")
    require(keys.dtype == np.uint32 and len({key.tobytes() for key in keys}) == 2, "base world/birth keys")
    zeros = dict(x=digest(np.zeros(157, np.float32)), Wp=digest(np.zeros((103, 48), np.float32)))
    result = dict(flip_step=boundary, keys=keys.copy(), plastic_genome_sha256=digest(plastic),
                  direct_genome_sha256=digest(direct))
    traces, action_keys, birth_w = {}, None, None
    for arm in ARMS:
        result[arm] = []
        for rule in (0, 1):
            prefix = arm + "_"
            life = {name[len(prefix):]: value[rule] for name, value in raw.items() if name.startswith(prefix)}
            traces[arm, rule] = life
            life.update(solo_initial_grid=raw["initial_grid"], solo_initial_position=raw["initial_position"],
                        **{"solo_" + field: life[field] for field in ("grid", "position", "energy")})
            outcome, trajectory = reversal.own_world(life, rule, boundary, arm.endswith("reversal"))
            shapes = dict(observation=(300, 1, 103), activity=(300, 1, 157), action=(300, 1),
                action_key=(300, 2), efference=(300, 1, 5), denergy=(300, 1), signal=(300, 1), alive=(300, 1), surprise=(300,))
            for name, shape in shapes.items():
                typed(life, name, shape, "iu" if name in ("action", "action_key") else "fbiu")
            require(life["action_key"].dtype == np.uint32, "action random-key dtype")
            action_keys = life["action_key"] if action_keys is None else action_keys
            reversal.identical(life["action_key"], action_keys, "common action streams across replays/controllers/interventions")
            reversal.identical(life["observation"], trajectory["observation"], "intact own observations contain no boundary cue")
            reversal.identical(life["activity"][:, :, :103], life["observation"], "actual sensor activity")
            reversal.identical(life["denergy"], trajectory["signed_energy"], "actual personal reward")
            require(np.array_equal(life["alive"], outcome["alive_after"]), "body survival")
            actions = life["action"][:, 0]
            require(np.isin(actions, np.arange(5)).all(), "action bounds")
            expected_position = (trajectory["pre_position"][:, 0] + prior.MOVES[actions] * trajectory["active"]) % 16
            require(np.array_equal(life["position"][:, 0], expected_position), "sampled action movement")
            reversal.identical(life["efference"], np.eye(5, dtype=np.float32)[actions, None], "actual action feedback")
            reversal.identical(life["activity"][:, :, -6:-1], life["efference"], "motor activity feedback")
            prior.close(life["signal"], life["activity"][:, :, -1] * trajectory["active"], "actual signal")
            require(np.all(np.abs(life["signal"]) <= 1) and np.all(life["surprise"] >= 0), "bounded signal/surprise")
            direct_arm, lesion = arm.startswith("direct"), arm.startswith("lesion")
            hebbian = np.zeros(300, bool) if direct_arm else np.arange(300) < (boundary + 1 if lesion else 300)
            require(np.array_equal(typed(life, "hebbian_enabled", (300,), "biu", binary=True), hebbian)
                    and np.array_equal(typed(life, "predictor_enabled", (300,), "biu", binary=True), np.full(300, not direct_arm)),
                    "only Hebbian updates stop on the first postflip-outcome tick")
            for name, expected in (("energy", 1.), ("alive", 1.), ("denergy", 0.)):
                require(np.array_equal(typed(life, "initial_" + name, (1,), "fbiu"), [expected]), "fresh body")
            for state in ("W", "x", "Wp"):
                for moment in ("initial", "probe", "final"):
                    require(typed(life, moment + "_" + state + "_sha256", (32,), "u").dtype == np.uint8, "actual state fingerprint")
                if state in zeros:
                    reversal.identical(life["initial_" + state + "_sha256"], zeros[state], "no inherited neural activity/predictor")
            if direct_arm:
                for moment in ("initial", "probe", "final"):
                    reversal.identical(life[moment + "_W_sha256"], digest(weights), "unchanged unbounded direct weights")
                    reversal.identical(life[moment + "_Wp_sha256"], zeros["Wp"], "direct prediction disabled")
            else:
                birth_w = life["initial_W_sha256"] if birth_w is None else birth_w
                reversal.identical(life["initial_W_sha256"], birth_w, "same plastic birth under both rules and conditions")
            if lesion:
                reversal.identical(life["probe_W_sha256"], life["final_W_sha256"], "lesion endpoint weights stay fixed")
            reversal.identical(life["final_x_sha256"], digest(life["activity"][-1, 0]), "terminal activity hash")
            require(not any(name.startswith("belief_") for name in life), "no social readout")
            for field, stop in (("preflip_sha256", boundary), ("prelesion_sha256", boundary + 1)):
                reversal.identical(life[field], prefix_hash(life, stop), "actual prefix evidence")
            endpoints = summarize_life(outcome, boundary)
            for field in endpoints["windows"]["pre"]:
                expected = np.asarray([endpoints["windows"][window][field] for window in reversal.WINDOWS],
                    bool if field in ("first_food", "first_poison", "no_meal") else np.int32)
                reversal.identical(life["window_" + field], expected, "assigned window outcome")
            reversal.identical(life["boundary_survival"], np.asarray(list(endpoints["survival"].values()), bool), "survival window")
            require(float(life["fitness"]) == endpoints["fitness"], "whole-life return")
            result[arm].append(endpoints)
    for rule in (0, 1):
        for controller in ("plastic", "lesion", "direct"):
            stationary, changed = (traces[controller + "_" + condition, rule] for condition in ("stationary", "reversal"))
            reversal.identical(stationary["preflip_sha256"], changed["preflip_sha256"], "same history before hidden reversal")
            for field in ("observation", "action", "action_key", "efference", "activity", "alive_before", "position", "grid"):
                reversal.identical(stationary[field][boundary], changed[field][boundary], "boundary action precedes reward change")
        for condition in ("stationary", "reversal"):
            intact, lesion = (traces[controller + "_" + condition, rule] for controller in ("plastic", "lesion"))
            for name in ("prelesion_sha256", "probe_W_sha256", "probe_x_sha256", "probe_Wp_sha256"):
                reversal.identical(intact[name], lesion[name], "lesion retains first postflip action/outcome and boundary state")
    return result


def pair_vector(values, window, field, operation="mean"):
    data = np.asarray([[life["windows"][window][field] for life in pair] for pair in values], np.float64)
    require(data.ndim == 2 and data.shape[1] == 2, "two assigned rule replays per independent base world")
    return data.min(axis=1) if operation == "min" else data.mean(axis=1)


def criteria(plastic_j, direct_j, late_first, lesion_difference, plastic_survival, survival_difference):
    defined = all(value["se"] is not None for value in (plastic_j, direct_j, late_first, lesion_difference,
                                                       plastic_survival, survival_difference))
    combined = float(np.hypot(plastic_j["se"], direct_j["se"])) if defined else None
    performance = dict(direct_comparison_passed=plastic_j["mean"] - direct_j["mean"] > 2 * combined if defined else None,
        absolute_return_passed=plastic_j["mean"] > 2 * plastic_j["se"] if defined else None,
        first_food_level_passed=late_first["mean"] >= .60,
        first_food_chance_passed=late_first["mean"] - .50 > 2 * late_first["se"] if defined else None,
        lesion_passed=lesion_difference["mean"] > 2 * lesion_difference["se"] if defined else None)
    survival = dict(absolute_passed=plastic_survival["mean"] >= .50,
        relative_passed=survival_difference["mean"] - 2 * survival_difference["se"] > -.10 if defined else None)
    return dict(combined_se=combined, performance=performance, survival=survival,
                performance_passed=all(performance.values()) if defined else None,
                survival_passed=all(survival.values()) if defined else None)


def replicated_gate(panels):
    require(len(panels) == 3, "all three matched lineage decisions are required")
    if any(panel["performance_passed"] is None or panel["survival_passed"] is None for panel in panels):
        return dict(passed=None, qualifying_seeds=None, survival_passed=None)
    wins = sum(panel["performance_passed"] for panel in panels)
    survival = all(panel["survival_passed"] for panel in panels)
    return dict(passed=wins >= 2 and survival, qualifying_seeds=wins, survival_passed=survival)


def measurements(rows):
    require(len(rows) > 0, "nonempty assigned base-world panel")
    stats = reversal.chain.interval
    arms, vectors = {}, {}
    for arm in ARMS:
        pairs = [row[arm] for row in rows]
        joint = np.asarray([all(life["survival"]["end"] for life in pair) for pair in pairs], float)
        j = pair_vector(pairs, "late", "fitness", "min")
        first = pair_vector(pairs, "late", "first_food")
        vectors[arm] = dict(J=j, first=first, joint=joint)
        windows = {}
        for window in reversal.WINDOWS:
            food = pair_vector(pairs, window, "food_count")
            poison = pair_vector(pairs, window, "poison_count")
            windows[window] = dict(**{field: stats(pair_vector(pairs, window, field)) for field in ("first_food", "first_poison", "no_meal")},
                food_count=stats(food), poison_count=stats(poison), fitness=stats(food - poison),
                contact_accuracy=base.ratio_stats(food, food + poison))
        survival = np.asarray([[life["survival"]["end"] for life in pair] for pair in pairs], float)
        arms[arm] = dict(late_min_return=stats(j), late_first_food=stats(first), joint_survival=stats(joint),
            mean_survival=stats(survival.mean(1)), per_rule_survival={str(rule): stats(survival[:, rule]) for rule in (0, 1)},
            survival_points={point: stats([np.mean([life["survival"][point] for life in pair]) for pair in pairs])
                             for point in ("at_flip", "after_early", "after_late", "end")},
            whole_return=stats([np.mean([life["fitness"] for life in pair]) for pair in pairs]), windows=windows)
    p, d, lesion = (vectors[name] for name in ("plastic_reversal", "direct_reversal", "lesion_reversal"))
    pj, dj = arms["plastic_reversal"]["late_min_return"], arms["direct_reversal"]["late_min_return"]
    mechanism = stats(p["J"] - lesion["J"])
    survival_difference = stats(p["joint"] - d["joint"])
    gates = criteria(pj, dj, arms["plastic_reversal"]["late_first_food"], mechanism,
                     arms["plastic_reversal"]["joint_survival"], survival_difference)
    defined = len(rows) > 1
    return dict(protocol=PROTOCOL, base_worlds=len(rows), rule_replays_per_base=2, lives_per_arm=2 * len(rows),
        world_keys_sha256=digest(np.stack([row["keys"] for row in rows])).tobytes().hex(),
        plastic_genome_sha256=rows[0]["plastic_genome_sha256"].tobytes().hex(),
        direct_genome_sha256=rows[0]["direct_genome_sha256"].tobytes().hex(),
        flip_counts={str(flip): sum(row["flip_step"] == flip for row in rows) for flip in FLIPS}, arms=arms,
        plastic_vs_direct=dict(difference=pj["mean"] - dj["mean"], combined_se=gates["combined_se"],
            paired_difference=stats(p["J"] - d["J"]), passed=gates["performance"]["direct_comparison_passed"]),
        plastic_vs_lesion=dict(difference=mechanism, passed=gates["performance"]["lesion_passed"]),
        survival=dict(difference=survival_difference, absolute_passed=gates["survival"]["absolute_passed"] if defined else None,
            relative_passed=gates["survival"]["relative_passed"], passed=gates["survival_passed"]),
        positive_return_passed=gates["performance"]["absolute_return_passed"],
        late_accuracy_passed=gates["performance"]["first_food_level_passed"] and gates["performance"]["first_food_chance_passed"] if defined else None,
        stationary_lesion_difference=stats(vectors["plastic_stationary"]["J"] - vectors["lesion_stationary"]["J"]),
        gate=dict(performance_passed=gates["performance_passed"], survival_passed=gates["survival_passed"],
                  passed=gates["performance_passed"] and gates["survival_passed"] if defined else None))


class Auditor(reversal.Auditor):
    def __init__(self, root, source):
        require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source), "source must be the full immutable registration commit")
        super().__init__(root, source)
        self.common_settings, self.exposures, self.key_owners = None, {}, {}

    def prior_keys(self):
        super().prior_keys()
        for directory in sorted(self.path("runs").glob("*_E125-*")):
            folder = directory.relative_to(self.root).as_posix()
            if self.path(folder + "/config.json").is_file():
                config = self.json(folder + "/config.json")
                self.used_masters.update(config[name] for name in ("seed", "eval_seed") if name in config)
            if self.path(folder + "/metrics.jsonl").is_file():
                for line in self.read(folder + "/metrics.jsonl").decode().splitlines():
                    record = json.loads(line)
                    if "keys" in record:
                        self.used_keys.update(key.tobytes() for key in np.asarray(record["keys"], np.uint32).reshape(-1, 2))

    def settings(self, config):
        settings = config["settings"]
        subset(settings, dict(protocol=PROTOCOL, cfg=CFG, pop=128, base_world_pairs=4, actual_lifetimes=8,
            sigma=.1, lr=.1, l2=.005, total_attempts=200, flip_steps=list(FLIPS), late_window=[50, 100],
            objective="mean of individual paired-rule minimum late food-minus-poison",
            tie_rule="average centered ranks; all ties consume attempt without optimizer update"), "registered training settings")
        git = config["git"]
        require(git.get("dirty") is False and len(git.get("commit", "")) == 40
                and self.code_trees(git["commit"]) == self.expected_trees, "clean registered source")
        if self.common_settings is None:
            jax, _ = random_runtime()
            subset(settings["runtime"], dict(numpy=np.__version__, jax=jax.__version__,
                jaxlib=importlib.metadata.version("jaxlib")), "same numerical runtime")
            same(settings["prng"], dict(implementation=str(jax.config.jax_default_prng_impl),
                threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
                enable_x64=bool(jax.config.jax_enable_x64), backend=jax.default_backend()), "same PRNG configuration")
            require({"seed/seed.py", "seed/direct.py", *["scripts/" + name + ".py" for name in
                ("reversal_evolution", "run_reversal_evolution", "social_observer_evolution", "social_reversal",
                 "social_learning", "run_cold_donor", "run_shelter_evolution")]}.issubset(settings["code"]),
                "scientific code manifest must include transitive training and trace helpers")
            for path, expected in settings["code"].items():
                require(path.endswith(".py") and (path.startswith("seed/") or path.startswith("scripts/"))
                        and ".." not in Path(path).parts, "scientific source path")
                content = base.subprocess.run(["git", "show", self.source + ":" + path], cwd=self.root,
                    capture_output=True, check=True, timeout=15).stdout
                same(hashlib.sha256(content).hexdigest(), expected, "pinned code " + path)
            self.common_settings = settings
        else:
            same(settings, self.common_settings, "all segments/controllers retain identical science and runtime")

    def artifact(self, folder, artifact):
        target = self.path(folder + "/" + artifact["path"])
        require(target.is_relative_to(self.path(folder)) and target != self.path(folder), "artifact leaves segment")
        arrays = self.arrays(target.relative_to(self.root).as_posix())
        same(self.hashes[target.relative_to(self.root).as_posix()], artifact["sha256"], "immutable artifact hash")
        return arrays

    def state(self, arrays, attempt, controller, prefix=""):
        state = {name: typed(arrays, prefix + name, (DIMENSIONS[controller],), "f") for name in ("mu", "m", "v")}
        require(all(value.dtype == np.float32 for value in state.values()) and np.all(state["v"] >= 0), "optimizer vectors and variance")
        state.update(t=typed(arrays, prefix + "t", (), "iu"), key=typed(arrays, prefix + "key", (2,), "u"))
        require(state["key"].dtype == np.uint32 and 0 <= int(state["t"]) <= attempt
                and int(typed(arrays, "attempted_round", (), "iu")) == attempt, "separate attempt and update counters")
        return state

    def sampling(self, previous, samples, controller, master, attempt, development):
        jax, jnp = random_runtime()
        next_key, noise_key, world_key, birth_key = jax.random.split(jnp.asarray(previous["key"]), 4)
        worlds = np.asarray(jax.random.split(world_key, 4))
        births = np.asarray(jax.random.split(birth_key, 64 * 4)).reshape(64, 4, 2)
        assignment = np.asarray(jax.random.fold_in(jnp.asarray(worlds[0]), 0xE126))
        permutation = np.asarray(jax.random.permutation(jnp.asarray(assignment), 4), np.int32)
        expected = dict(next_key=np.asarray(next_key), noise_key=np.asarray(noise_key), world_keys=worlds,
            development_keys=np.concatenate((births, births)), assignment_key=assignment, permutation=permutation,
            flip_steps=np.asarray(FLIPS, np.int32)[permutation])
        for field, value in expected.items():
            reversal.identical(samples[field], value, "same-runtime sampled " + field)
        exposure = ("development", controller, attempt) if development else ("training", master, attempt)
        shared = hashlib.sha256(b"".join(expected[field].tobytes() for field in
            ("next_key", "world_keys", "development_keys", "assignment_key", "permutation", "flip_steps"))).hexdigest()
        if exposure in self.exposures:
            old = self.exposures[exposure]
            require(old["sha256"] == shared and controller not in old["controllers"], "matched attempted exposure or duplicate architecture")
            old["controllers"].add(controller)
        else:
            self.exposures[exposure] = dict(sha256=shared, controllers={controller})
        keys = np.concatenate((worlds, births.reshape(-1, 2), assignment[None], np.asarray(noise_key)[None], np.asarray(next_key)[None]))
        tokens = [key.tobytes() for key in keys]
        require(len(set(tokens)) == len(tokens), "distinct sampled key roles")
        for token in tokens:
            require(token not in self.used_keys and (token not in self.key_owners or self.key_owners[token] == exposure),
                    "prior, development or different-attempt key reuse")
            self.key_owners[token] = exposure

    def segments(self, target):
        require(re.fullmatch(r"E126-(?:plastic|direct)-s[012]|E126-benchmark-(?:plastic|direct)", target), "registered target name")
        development = "benchmark" in target
        candidates = sorted(self.path("runs").glob("*_" + target + ("" if development else "-part*")))
        if not candidates:
            raise base.Incomplete(target + " has no segment; no scientific decision")
        parts = {}
        for path in candidates:
            suffix = path.name.split("_", 1)[-1]
            match = re.fullmatch(re.escape(target) + r"-part(\d{3})", suffix) if not development else None
            require(development or match is not None, "malformed segment name")
            part = 1 if development else int(match[1])
            require(part not in parts, "branched/duplicate segment")
            parts[part] = path.relative_to(self.root).as_posix()
        require(sorted(parts) == list(range(1, max(parts) + 1)), "consecutive immutable segment chain")
        return [parts[part] for part in sorted(parts)]

    def training_target(self, target):
        development = "benchmark" in target
        controller = "plastic" if "plastic" in target else "direct"
        master = 2026120400 if development else 2026120401 + int(target[-1])
        require(master not in self.used_masters, "old evaluation master used for training")
        expected_attempts = 3 if development else 200
        previous, parent, founder, attempt, reports, sealed = None, None, None, 0, [], None
        for part, folder in enumerate(self.segments(target), 1):
            require(sealed is None, "a sealed target cannot be extended")
            config = self.json(folder + "/config.json")
            self.settings(config)
            subset(config, dict(protocol=PROTOCOL, stage="training", condition_split="development" if development else "final",
                target=target, architecture=controller, controller="rules-v0" if controller == "plastic" else "direct-rnn-v1",
                seed=master, part=part, start_attempt=attempt, expected_attempts=expected_attempts,
                genome_size=DIMENSIONS[controller], max_segment_attempts=3 if development else 50,
                soft_seconds=780 if development else 3000, max_seconds=900 if development else 3600,
                sealing_recovery=bool(parent is not None and attempt == 200),
                scientific_result=False), "registered target segment")
            subset(config["args"], dict(start="blank", seed=master, generations=expected_attempts, pop=128, worlds=4,
                sigma=.1, lr=.1, checkpoint_every=10, eval_worlds=128, allow_dirty=False, no_plastic=False,
                cfg=None, objective=None, memory_condition=None), "fixed segment command")
            same(config["parent"], parent, "exact preceding journal/state reference")
            current = self.state(self.artifact(folder, config["start_checkpoint"]), attempt, controller)
            if previous is None:
                tag = (0xE12600 if controller == "plastic" else 0xE12601) if development else None
                initial_state(current, controller, master, tag)
                founder = config["founder"]
                same(founder, dict(folder=folder, genome=dict(path="genome_start.npy", sha256=hashlib.sha256(self.read(folder + "/genome_start.npy")).hexdigest()),
                    checkpoint=config["start_checkpoint"], master_seed=master,
                    initialization="blank rules" if controller == "plastic" else "normal/sqrt(156)", development_stream_tag=tag), "fresh founder provenance")
            else:
                require(all(np.array_equal(current[name], previous[name]) for name in previous), "resume preserves all optimizer state exactly")
                same(config["founder"], founder, "same initializer across segments")
            genome = np.load(io.BytesIO(self.read(folder + "/genome_start.npy")), allow_pickle=False)
            reversal.identical(genome, current["mu"], "segment entry genome")
            journal = self.read(folder + "/metrics.jsonl") if self.path(folder + "/metrics.jsonl").is_file() else b""
            lines = journal.splitlines(keepends=True)
            complete = lines if not lines or lines[-1].endswith(b"\n") else lines[:-1]
            last_artifact, records, elapsed = config["start_checkpoint"], [], 0.
            for line in complete:
                row = json.loads(line)
                same(row["attempt"], attempt + 1, "one fresh attempted round")
                same(row["samples"]["path"], f"samples/attempt_{attempt + 1:04d}.npz", "attempt artifact ordering")
                samples = self.artifact(folder, row["samples"])
                self.sampling(current, samples, controller, master, attempt + 1, development)
                expected, scores, informative = optimizer_transition(current, samples, controller)
                following = self.state(samples, attempt + 1, controller, "next_")
                for field, value in expected.items():
                    if informative and field != "t":
                        close(following[field], value, "independent Adam " + field)
                    else:
                        require(np.array_equal(following[field], value), "exact tie skip/update count " + field)
                subset(row, dict(informative=informative, optimizer_updates=int(following["t"]),
                    changed_parameters=int(np.count_nonzero(following["mu"] != current["mu"]))), "attempt outcome accounting")
                for name, value in (("fit_mean", scores.mean()), ("fit_best", scores.max()), ("fit_std", scores.std())):
                    close(row[name], value, "selection " + name)
                require(row["iteration_seconds"] > 0 and elapsed <= row["elapsed_seconds"] <= config["max_seconds"], "attempt timing")
                elapsed = row["elapsed_seconds"]
                current, attempt, last_artifact = following, attempt + 1, row["samples"]
                checkpoint = f"checkpoints/attempt_{attempt:04d}.npz"
                if self.path(folder + "/" + checkpoint).is_file():
                    saved = self.state(self.arrays(folder + "/" + checkpoint), attempt, controller)
                    require(all(np.array_equal(saved[name], current[name]) for name in current), "immutable checkpoint matches journal transition")
                records.append(row)
            require(attempt - config["start_attempt"] <= config["max_segment_attempts"]
                    and attempt <= expected_attempts, "registered exposure cap")
            if config["sealing_recovery"]:
                require(not records and all(np.array_equal(current[name], previous[name]) for name in current),
                        "sealing recovery cannot perform another optimizer attempt")
            ignored = len(journal) - sum(len(line) for line in complete)
            if self.path(folder + "/summary.json").is_file():
                summary = self.json(folder + "/summary.json")
                subset(summary, dict(attempted_rounds=attempt, optimizer_updates=int(current["t"]), target=target,
                    start_attempt=config["start_attempt"], expected_attempts=expected_attempts,
                    protocol=PROTOCOL, stage="training", condition_split=config["condition_split"],
                    controller=config["controller"], training_seed=master,
                    scientific_result=False, gate_status="undecided", attempts=records, parent=parent, founder=founder), "segment summary")
                terminal = self.state(self.artifact(folder, summary["terminal_checkpoint"]), attempt, controller)
                require(all(np.array_equal(terminal[name], current[name]) for name in current), "terminal checkpoint")
                terminal_genome = self.read(folder + "/" + summary["terminal_genome"]["path"])
                same(hashlib.sha256(terminal_genome).hexdigest(), summary["terminal_genome"]["sha256"], "terminal genome artifact")
                reversal.identical(np.load(io.BytesIO(terminal_genome), allow_pickle=False), current["mu"], "fixed latest genome")
                if (not development and attempt == 200 and summary["status"] == "target_finished"
                        and summary["artifact_validity"] == "passed" and ignored == 0):
                    sealed = dict(summary=dict(path="summary.json", sha256=self.hashes[folder + "/summary.json"]),
                                  terminal_checkpoint=summary["terminal_checkpoint"], terminal_genome=summary["terminal_genome"])
            parent = dict(folder=folder, config_sha256=self.hashes[folder + "/config.json"], journal_sha256=hashlib.sha256(journal).hexdigest(),
                completed_attempt=attempt, terminal_state=last_artifact,
                terminal_state_kind="checkpoint" if last_artifact == config["start_checkpoint"] else "samples", ignored_unflushed_journal_bytes=ignored)
            reports.append(dict(folder=folder, completed_segment_attempts=len(records), completed_attempt=attempt,
                optimizer_updates=int(current["t"]), ignored_unflushed_journal_bytes=ignored))
            previous = current
        return dict(state=previous, source=parent, sealed=sealed, report=dict(target=target, completed_attempts=attempt,
            target_attempts=expected_attempts, optimizer_updates=int(previous["t"]), segments=reports,
            sealed_endpoint=sealed is not None, scientific_gate="undecided"))

    def complete_evaluation_bundle(self):
        folders, pending = [], []
        for seed in range(3):
            name = f"E126-eval-s{seed}"
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, "duplicate final panel")
            if not candidates or not (candidates[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            files = {path.name for path in self.path(folder + "/worlds").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_worlds") != 128
                    or status.get("expected_worlds") != 128 or files != {f"world_{i:04d}.npz" for i in range(128)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final world outcomes opened")
        return folders

    def final_panel(self, folder, index, endpoints):
        config = self.json(folder + "/config.json")
        self.settings(config)
        master = 2026120501 + index
        subset(config, dict(protocol=PROTOCOL, stage="evaluation", condition_split="final", seed_index=index,
            eval_seed=master, expected_worlds=128, max_seconds=3600, scientific_result=False), "final panel config")
        subset(config["args"], dict(name=f"E126-eval-s{index}", seed=master, generations=0, start="blank", eval_worlds=128,
            pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10, resume=None, controller=None,
            max_train_seconds=None, allow_dirty=False, no_plastic=False, cfg=None, objective=None,
            memory_condition=None), "fixed final invocation")
        require(master not in self.used_masters, "prior/development evaluation master reuse")
        self.used_masters.add(master)
        require(all(value["sealed"] is not None for value in endpoints.values()), "all six endpoints sealed before final outcomes")
        sources = {name: dict(**value["source"], sealed=value["sealed"]) for name, value in endpoints.items()}
        same(config["sources"], sources, "all six frozen attempt-200 sources")
        plastic, direct = (endpoints[f"E126-{controller}-s{index}"]["state"]["mu"] for controller in CONTROLLERS)
        for name, expected in (("plastic_genome.npy", plastic), ("direct_genome.npy", direct)):
            reversal.identical(np.load(io.BytesIO(self.read(folder + "/" + name)), allow_pickle=False), expected, "frozen final genome")
        jax, jnp = random_runtime()
        schedule = self.artifact(folder, config["assignments"])
        key = jax.random.fold_in(jax.random.PRNGKey(master), 0xE126)
        permutation = np.asarray(jax.random.permutation(key, 128), np.int32)
        flips = np.repeat(np.asarray(FLIPS, np.int32), 32)[permutation]
        for name, value in (("key", np.asarray(key)), ("permutation", permutation), ("flip_steps", flips)):
            reversal.identical(schedule[name], value, "independent fixed final schedule")
        require(np.asarray(key).tobytes() not in self.key_owners, "final schedule must not reuse a training key")
        self.register_keys(np.asarray(key)[None], "final assignment stream")
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split="final", seed_index=index,
            evaluation_seed=master, completed_worlds=128, expected_worlds=128, sources=sources,
            scientific_result=False, gate_status="undecided", artifact_validity="passed"), "complete paired panel")
        require(0 < summary["elapsed_seconds"] <= 3600, "final invocation time cap")
        records = self.read(folder + "/evaluation.jsonl").decode().splitlines()
        require(len(records) == len(summary["raw_files"]) == 128, "all 128 independent bases retained")
        rows, elapsed = [], 0.
        for world, line in enumerate(records):
            raw = self.artifact(folder, summary["raw_files"][world])
            same(summary["raw_files"][world]["index"], world, "final world ordering")
            same(summary["raw_files"][world]["path"], f"worlds/world_{world:04d}.npz", "final world path")
            keys = np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(master), world), 2))
            reversal.identical(raw["keys"], keys, "fresh final base keys")
            require(all(key.tobytes() not in self.key_owners for key in keys), "training/evaluation key overlap")
            self.register_keys(keys, "held-out base world")
            rows.append(evaluation_world(raw, world, int(flips[world]), plastic, direct))
            record = json.loads(line)
            subset(record, dict(world=world, flip_step=int(flips[world]), keys=keys.tolist(), raw_file=summary["raw_files"][world]), "final assignment journal")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"] and record["world_seconds"] > 0, "final world time order")
            elapsed = record["elapsed_seconds"]
        estimates = measurements(rows)
        same(summary["measurements"], estimates, "independent clustered outcomes and gate")
        return dict(run=folder, seed_index=index, measurements=estimates, gate=estimates["gate"])


def self_check():
    """Small PRNG/arithmetic fixtures only; schedule this after the E125 decision."""
    jax, jnp = random_runtime()
    checks = []
    pop = 4

    def sample(previous, controller, informative):
        dimension = DIMENSIONS[controller]
        key = jax.random.PRNGKey(126)
        half = jax.random.normal(key, (pop // 2, dimension))
        actual = jnp.asarray(previous["mu"]) + .1 * jnp.concatenate((half, -half))
        events = {name: np.full((pop, 4, 2, 300, 1), name.startswith("alive"), bool) for name in EVENTS}
        if informative:
            for world, boundary in enumerate(FLIPS):
                events["food"][2:, world, :, boundary + 50, 0] = True
        fitness, returns, counts = late_scores(events, FLIPS)
        all_counts = np.zeros((pop, 4, 2, 3, 2), np.int32)
        all_counts[..., 2, :] = counts
        result = dict(noise_key=np.asarray(key), half_eps_sha256=digest(np.asarray(half)),
            candidate_sha256=np.stack([digest(value) for value in np.asarray(actual)]),
            flip_steps=np.asarray(FLIPS, np.int32), permutation=np.arange(4, dtype=np.int32),
            fitness=fitness, rule_late_return=returns, rule_window_counts=all_counts,
            survived=np.ones((pop, 4), np.float32), rank_weights=average_ranks(fitness.mean(1)),
            event_shape=np.asarray(events["food"].shape))
        result.update({name + "_packed": np.packbits(value.reshape(-1), bitorder="little") for name, value in events.items()})
        return result

    for controller in CONTROLLERS:
        dimension = DIMENSIONS[controller]
        previous = dict(mu=np.linspace(-.1, .1, dimension, dtype=np.float32), m=np.zeros(dimension, np.float32),
                        v=np.zeros(dimension, np.float32), t=np.asarray(0, np.int32))
        samples = sample(previous, controller, True)
        following, _, informative = optimizer_transition(previous, samples, controller, pop=pop)
        require(informative and int(following["t"]) == 1 and not np.array_equal(following["mu"], previous["mu"]), "informative Adam fixture")
        tied = sample(following, controller, False)
        skipped, _, informative = optimizer_transition(following, tied, controller, pop=pop)
        require(not informative and all(np.array_equal(skipped[field], following[field]) for field in following), "ties preserve nonzero moments and genome")
        changed = dict(samples, candidate_sha256=samples["candidate_sha256"].copy())
        changed["candidate_sha256"][0, 0] ^= 1
        try:
            optimizer_transition(previous, changed, controller, pop=pop)
        except base.AuditError:
            pass
        else:
            raise base.AuditError("changed candidate hash was accepted")
    checks.append("both genome dimensions: same-runtime candidate arithmetic, informative Adam, exact all-tie skip")
    events = {name: np.full((1, 4, 2, 300, 1), name.startswith("alive"), bool) for name in EVENTS}
    for world, boundary in enumerate(FLIPS):
        events["food"][0, world, 0, [boundary + 49, boundary + 50, boundary + 99, boundary + 100], 0] = True
        events["poison"][0, world, 1, boundary + 50, 0] = True
    fitness, returns, _ = late_scores(events, FLIPS)
    require(np.array_equal(returns[0], [[2, -1]] * 4) and np.array_equal(fitness, [[-1] * 4]), "exact late window and minimum before averaging rules")
    checks.append("paired specialist cannot hide negative replay; exact f+50/f+100 boundaries")
    good = criteria(dict(mean=2., se=.1), dict(mean=0., se=.1), dict(mean=.75, se=.05),
                    dict(mean=1., se=.1), dict(mean=.75, se=.05), dict(mean=0., se=0.))
    require(good["performance_passed"] and good["survival_passed"], "complete qualifying seed")
    uncertain = criteria(dict(mean=1., se=1.), dict(mean=0., se=1.), dict(mean=.75, se=.05),
                        dict(mean=1., se=.1), dict(mean=.75, se=.05), dict(mean=0., se=0.))
    require(not uncertain["performance"]["direct_comparison_passed"], "combined SE cannot be replaced by zero paired SE")
    require(replicated_gate([good, good, dict(performance_passed=False, survival_passed=True)])["passed"], "two same qualifying seeds suffice")
    require(not replicated_gate([good, good, dict(performance_passed=False, survival_passed=False)])["passed"], "third-seed survival veto")
    require(not criteria(dict(mean=0., se=0.), dict(mean=-3., se=0.), dict(mean=.75, se=.05),
        dict(mean=1., se=0.), dict(mean=1., se=0.), dict(mean=0., se=0.))["performance_passed"], "avoidance is not useful positive return")
    checks.append("combined uncertainty, absolute utility, two qualifying seeds and all-three survival")
    life = {name: np.arange(5, dtype=np.float32) for name in PREFIX_FIELDS}
    changed = {name: value.copy() for name, value in life.items()}
    changed["food"][2] += 1
    require(np.array_equal(prefix_hash(life, 2), prefix_hash(changed, 2))
            and not np.array_equal(prefix_hash(life, 3), prefix_hash(changed, 3)), "first postflip outcome remains in lesion prefix")
    require(reversal.direct_weights(np.full(8424, 4., np.float32)).max() == 4., "direct weights remain unbounded")
    checks.append("prefix endpoint timing and unclipped direct encoding")
    require("seed.seed" not in sys.modules, "self-check must not import the simulator")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", tests=len(checks), checks=checks,
                simulator_imported=False, prng_replayed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--target", action="append", help="audit an available immutable training prefix without reading final outcomes")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2, allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required for artifact verification")
    training_only = args.training_only or args.development or bool(args.target)
    report, auditor = dict(protocol=PROTOCOL, mode="training-prefix" if training_only else "final"), None
    try:
        auditor = Auditor(args.root, args.source_commit)
        if not training_only:
            folders = auditor.complete_evaluation_bundle()
        auditor.prior_keys()
        candidates = [f"E126-benchmark-{controller}" for controller in CONTROLLERS] if args.development else [
            f"E126-{controller}-s{seed}" for seed in range(3) for controller in CONTROLLERS]
        if training_only:
            targets = args.target or [name for name in candidates if any(auditor.path("runs").glob("*_" + name + ("" if "benchmark" in name else "-part*")))]
            if not targets:
                raise base.Incomplete("no available training prefixes; scientific gate remains undecided")
            report["training"] = [auditor.training_target(target)["report"] for target in targets]
            report.update(status="passed", scientific_gate_status="undecided", final_evaluation_audited=False,
                          note="Completed prefixes are valid evidence. No further compute is implied by this audit.")
        else:
            for controller in CONTROLLERS:
                benchmark = auditor.training_target("E126-benchmark-" + controller)
                require(benchmark["report"]["completed_attempts"] == 3, "completed registered benchmark")
            endpoints = {name: auditor.training_target(name) for name in candidates}
            if any(value["report"]["completed_attempts"] != 200 or value["sealed"] is None for value in endpoints.values()):
                raise base.Incomplete("six exact attempt-200 endpoints are not sealed; no final outcomes opened")
            for seed in range(3):
                for attempt in range(1, 201):
                    require(auditor.exposures["training", 2026120401 + seed, attempt]["controllers"] == set(CONTROLLERS),
                            "matched exposure at every attempted round")
            panels = [auditor.final_panel(folder, index, endpoints) for index, folder in enumerate(folders)]
            report.update(status="passed", final_evaluation_audited=True, training=[value["report"] for value in endpoints.values()],
                          panels=panels, scientific_gate=replicated_gate([panel["gate"] for panel in panels]))
        report.update(registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            limitations=["Exact random initialization, perturbations and candidate digests require the archived JAX CPU runtime/configuration.",
                "World events and optimizer arithmetic are reconstructed; candidate fitness is not regenerated by rerunning the simulator.",
                "Lesion masks, full recorded prefixes and boundary hashes are checked; intermediate neural updates are not independently replayed.",
                "Reported episode outcomes are clustered by independent base world; the controller parameterizations remain different.",
                "Operational daily/study budget ledgers are managed by the supervisor, not inferred from scientific success."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), scientific_gate_status="undecided", final_evaluation_audited=False)
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
