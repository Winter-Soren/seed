"""Independent E127 event/probability audit. No simulator or held-out tuning."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_social_reversal as prior
import audit_reversal_evolution as source_bytes

base, film = prior.base, prior.film
require, same, subset, typed = base.require, base.same, base.subset, base.typed
PROTOCOL = "encounter-diagnostic-v1"
DEVELOPMENT_NAME = "E127-benchmark"
RETRY_NAME = "E127-benchmark-retry1"
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_frozen", "direct_stationary", "direct_reversal")
CONTACTS = (0, 2, 8, 16)
STATE_FIELDS = ("W", "x", "Wp", "mask")
BODY = ("grid", "pos", "energy", "alive", "sig", "denergy", "key", "t", "counts")
MOVES, CFG = prior.MOVES, prior.CFG


def exact(left, right, label):
    prior.identical(np.asarray(left), np.asarray(right), label)


def fingerprint(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8)


def softmax(logits, probabilities):
    require(logits.shape == probabilities.shape and logits.shape[-1] == 5
            and np.isfinite(logits).all() and np.isfinite(probabilities).all(), "five finite unmasked action probabilities")
    exponent = np.exp(logits.astype(np.float64) - logits.max(axis=-1, keepdims=True))
    require(np.all((probabilities >= 0) & (probabilities <= 1)), "action probability bounds")
    prior.prior.close(probabilities, exponent / exponent.sum(axis=-1, keepdims=True), "original five-action softmax")
    prior.prior.close(probabilities.sum(axis=-1), np.ones(probabilities.shape[:-1]), "all action mass retained")


def observation(grid, position, energy, denergy):
    offsets = np.arange(-2, 3)
    patch = grid[(position[0] + offsets[:, None]) % 16, (position[1] + offsets[None, :]) % 16]
    return np.concatenate(((patch == 1).ravel(), (patch == 2).ravel(), np.zeros(50),
        [np.float32(energy) - np.float32(1), np.float32(4) * np.float32(denergy), 1])).astype(np.float32)


def local_vectors(probabilities, placements):
    probabilities = probabilities.astype(np.float64)
    require(probabilities.shape == (2, 4, 12, 5), "two rules, four checkpoints, twelve orientations and five actions")
    require(placements.shape == (12, 2) and set(map(tuple, placements.tolist()))
            == {(a, b) for a in range(1, 5) for b in range(1, 5) if a != b}, "all ordered cardinal placements")
    food, poison = [], []
    for initial_rule in (0, 1):
        # Initial rule0 means A was food, so B is the common reversed target.
        food.append(probabilities[initial_rule, :, np.arange(12), placements[:, 1 - initial_rule]].T)
        poison.append(probabilities[initial_rule, :, np.arange(12), placements[:, initial_rule]].T)
    good, bad = np.stack(food).mean(axis=2), np.stack(poison).mean(axis=2)
    return dict(p_food=good, p_poison=bad, approach=good + bad, other=1 - good - bad, S=good - bad)


def stat(values):
    values = np.asarray(values, np.float64)
    return {**base.world_stats(values), "n": len(values)}


def prefixed(values, prefix):
    return {name[len(prefix):]: value for name, value in values.items() if name.startswith(prefix)}


def pair_grid(placement):
    grid = np.zeros((16, 16), np.int32)
    cells = (np.array([8, 8]) + MOVES[placement]) % 16
    grid[cells[:, 0], cells[:, 1]] = (1, 2)
    return grid


def grid_outcomes(food, poison, alive):
    food, poison = np.asarray(food, bool), np.asarray(poison, bool)
    contact = food | poison
    first = int(np.flatnonzero(contact)[0]) if contact.any() else None
    good, bad = float(first is not None and food[first]), float(first is not None and poison[first])
    return dict(first_food=good, first_poison=bad, no_contact=float(first is None), food_minus_poison=good - bad,
        latency=len(food) + 1 if first is None else first + 1, net_return=int(food.sum()) - int(poison.sum()), survival=float(alive[-1]))


def trace_events(trace, initial, rules, *, forced=None, script=None, hebbian=None, predictor=False):
    """Reconstruct observations, actual contacts and feedback from physical states."""
    steps = len(rules)
    shapes = dict(grid_before=(steps, 16, 16), grid=(steps, 16, 16), position_before=(steps, 1, 2),
        position=(steps, 1, 2), observation=(steps, 1, 103), incoming_activity=(steps, 1, 157),
        activity=(steps, 1, 157), decision_motor=(steps, 1, 5), logits=(steps, 1, 5),
        probabilities=(steps, 1, 5), action=(steps, 1), sampled_action=(steps, 1), efference=(steps, 1, 5),
        action_key=(steps, 2), regrowth_key=(steps, 2), key_before=(steps, 2), key_after=(steps, 2),
        regrowth_cell=(steps, 1, 2), t=(steps,), rule=(steps,), forced_action=(steps,))
    for name, shape in shapes.items():
        integer = name in ("grid_before", "grid", "position_before", "position", "action", "sampled_action",
            "action_key", "regrowth_key", "key_before", "key_after", "regrowth_cell", "t", "rule", "forced_action")
        typed(trace, name, shape, "iu" if integer else "f")
    for name in ("energy_before", "energy", "unclipped_energy", "signal_before", "signal", "incoming_denergy", "denergy",
                 "alive_before", "alive_after", "food", "poison", "item"):
        typed(trace, name, (steps, 1), "biuf")
    for name in ("alive_before", "alive_after", "food", "poison"):
        require(np.isin(trace[name], (0, 1)).all(), "binary actual events " + name)
    require(np.all(np.abs(trace["decision_motor"]) <= 1), "bounded neural motor output")
    require(np.array_equal(trace["rule"], rules) and np.array_equal(trace["t"], np.arange(steps)), "actual hidden schedule and tick clock")
    require(np.array_equal(trace["hebbian_enabled"], hebbian)
            and np.array_equal(trace["predictor_enabled"], np.full(steps, predictor)), "actual update masks and intervention timing")
    softmax(trace["logits"], trace["probabilities"])
    exact(trace["logits"], np.float32(3) * trace["decision_motor"], "original motor logits")
    exact(trace["efference"], trace["activity"][:, :, -6:-1], "actual motor activity")
    exact(trace["incoming_activity"][1:], trace["activity"][:-1], "previous actual activity reaches the next brain tick")
    exact(fingerprint(trace["incoming_activity"][0]), initial["x_sha256"], "actual initial recurrent state")
    exact(trace["key_before"][0], initial["key"], "initial random stream")
    exact(trace["key_before"][1:], trace["key_after"][:-1], "random stream continuity")
    require(np.isin(trace["action"], np.arange(5)).all() and np.isin(trace["sampled_action"], np.arange(5)).all(), "five-way actual and sampled actions")
    if forced is None:
        require(np.all(trace["forced_action"] == -1), "free grid must not impose actions")
        exact(trace["action"], trace["sampled_action"], "unassisted sampled movement")
    else:
        require(np.array_equal(trace["action"][:, 0], forced) and np.array_equal(trace["forced_action"], forced), "actual imposed action")
    grid, pos = initial["grid"].copy(), initial["pos"].copy()
    energy, alive, delta, signal = (initial[name].copy() for name in ("energy", "alive", "denergy", "sig"))
    for tick in range(steps):
        if script is not None and tick % 2 == 0:
            grid = pair_grid(script[tick // 2, :2])
        exact(trace["grid_before"][tick], grid, "actual pre-action resources")
        exact(trace["position_before"][tick], pos, "body position continuity")
        for field, expected in (("energy_before", energy), ("alive_before", alive), ("incoming_denergy", delta), ("signal_before", signal)):
            prior.prior.close(trace[field][tick], expected, "physical feedback continuity " + field)
        expected_obs = observation(grid, pos[0], energy[0], delta[0])[None]
        exact(trace["observation"][tick], expected_obs, "only visible scene and actual body feedback")
        exact(trace["activity"][tick, :, :103], expected_obs, "sensory clamp")
        action = int(trace["action"][tick, 0])
        next_pos = (pos + MOVES[action] * alive[:, None].astype(np.int32)) % 16
        require(np.array_equal(trace["position"][tick], next_pos), "real action moves the body")
        item = grid[next_pos[0, 0], next_pos[0, 1]]
        food, poison = bool(alive[0] and item == 1 + rules[tick]), bool(alive[0] and item == 2 - rules[tick])
        require(int(trace["item"][tick, 0]) == item and bool(trace["food"][tick, 0]) == food
                and bool(trace["poison"][tick, 0]) == poison, "actual current-rule food/poison contact")
        raw_energy = energy - np.float32(.003) * alive + np.float32(.25) * food - np.float32(.25) * poison
        prior.prior.close(trace["unclipped_energy"][tick], raw_energy, "ordinary energy equation")
        prior.prior.close(trace["energy"][tick], np.clip(raw_energy, 0, 2), "normal energy clipping")
        prior.prior.close(trace["denergy"][tick], (trace["energy"][tick] - energy) * alive, "actual outcome feedback")
        require(np.array_equal(trace["alive_after"][tick], alive * (trace["energy"][tick] > 0)), "death cannot be reset or omitted")
        exact(trace["efference"][tick], np.eye(5, dtype=np.float32)[[action]], "imposed or sampled action feeds back")
        prior.prior.close(trace["signal"][tick], trace["activity"][tick, :, -1] * alive, "actual public signal")
        if food or poison:
            grid[next_pos[0, 0], next_pos[0, 1]] = 0
            if script is None:
                cell = trace["regrowth_cell"][tick, 0]
                require(np.all((cell >= 0) & (cell < 16)), "regrowth cell bounds")
                grid[cell[0], cell[1]] = item
        exact(trace["grid"][tick], grid, "consumption and declared regrowth")
        pos, energy, alive, delta, signal = (trace[name][tick] for name in ("position", "energy", "alive_after", "denergy", "signal"))
    return grid_outcomes(trace["food"][:, 0], trace["poison"][:, 0], trace["alive_after"][:, 0])


def audit_family(raw, index, pre, plastic, direct):
    require(int(raw["base_index"]) == index and int(raw["pre_contacts"]) == pre and pre in (8, 16), "fixed family assignment")
    exact(raw["initial_rules"], np.array([0, 1], np.int32), "both initial rules retained")
    exact(raw["readout_contacts"], np.asarray(CONTACTS, np.int32), "fixed checkpoint times")
    require(int(raw["grid_steps"]) == 50, "fixed transfer horizon")
    for name, genome in (("plastic", plastic), ("direct", direct)):
        exact(raw[name + "_genome_sha256"], fingerprint(genome), "frozen original genome")
    keys = typed(raw, "keys", (6, 2), "u")
    script = typed(raw, "script", (pre + 16, 5), "iu")
    require(np.isin(script[:, :2], np.arange(1, 5)).all() and np.all(script[:, 0] != script[:, 1])
            and np.isin(script[:, 2], (1, 2)).all(), "visible A/B contact placements")
    require(np.array_equal(script[:, 3], script[np.arange(len(script)), script[:, 2] - 1])
            and np.array_equal(MOVES[script[:, 4]], -MOVES[script[:, 3]]), "contact the declared color then return")
    require(np.all(np.sort(script[:, 2].reshape(-1, 2), axis=1) == [1, 2]), "one real contact of each color per pair")
    for block in script.reshape(-1, 8, 5):
        require(set(map(tuple, block[:, [2, 3]].tolist())) == {(color, action) for color in (1, 2) for action in range(1, 5)},
                "every color/direction combination exactly once per block")
    placements = typed(raw, "probe_placements", (12, 2), "iu")
    ticks, cutoff = 2 * len(script), 2 * pre + 1
    forced = script[:, [3, 4]].reshape(-1)
    decoded = prior.direct_weights(direct)[None]
    legal = ((np.arange(157)[:, None] >= 103) & ~np.eye(157, dtype=bool)).astype(np.float32)[None]
    fixed_direct = dict(W=fingerprint(decoded), Wp=fingerprint(np.zeros((1, 103, 48), np.float32)), mask=fingerprint(legal))
    result, lives, paired_grid = dict(local={}, grid={}), {}, None
    for arm in ARMS:
        data = prefixed(raw, arm + "_")
        result["local"][arm] = local_vectors(data["probe_probabilities"], placements)
        grid_results = []
        for rule in (0, 1):
            life = {name: value[rule] for name, value in data.items()}
            lives[arm, rule] = life
            birth, snapshots = prefixed(life, "birth_"), prefixed(life, "snapshot_")
            for state in STATE_FIELDS:
                require(typed(birth, state + "_sha256", (32,), "u").dtype == np.uint8, "actual birth-state hashes")
            require(np.array_equal(birth["grid"], np.zeros((16, 16))) and np.array_equal(birth["pos"], [[8, 8]]), "fresh empty hub")
            exact(birth["key"], keys[2], "separate chamber stream")
            exact(birth["x_sha256"], fingerprint(np.zeros((1, 157), np.float32)), "fresh recurrent activity")
            exact(birth["Wp_sha256"], fingerprint(np.zeros((1, 103, 48), np.float32)), "fresh predictor")
            for name, expected in (("energy", 1), ("alive", 1), ("sig", 0), ("denergy", 0), ("t", 0), ("counts", 0)):
                require(np.all(birth[name] == expected), "fresh chamber body " + name)
            rules = np.full(ticks, rule, np.int32)
            if not arm.endswith("stationary"):
                rules[2 * pre:] = 1 - rule
            plastic_arm = arm.startswith("plastic")
            hebbian = np.full(ticks, plastic_arm, bool)
            if arm.endswith("frozen"):
                hebbian[cutoff:] = False
            guided = prefixed(life, "guided_")
            trace_events(guided, birth, rules, forced=forced, script=script, hebbian=hebbian, predictor=plastic_arm)
            require(np.all(guided["alive_after"] == 1) and np.all((guided["unclipped_energy"] > 0)
                    & (guided["unclipped_energy"] < 2)), "declared chamber survival/no-clipping apparatus")
            for point, contacts in enumerate(CONTACTS):
                stop = 2 * (pre + contacts)
                expected_counts = np.zeros((1, 2, 2), np.float32)
                expected_counts[0, 0] = [guided[name][:stop].sum() for name in ("food", "poison")]
                for name, expected in (("grid", guided["grid"][stop - 1]), ("pos", guided["position"][stop - 1]),
                    ("energy", guided["energy"][stop - 1]), ("alive", guided["alive_after"][stop - 1]),
                    ("sig", guided["signal"][stop - 1]), ("denergy", guided["denergy"][stop - 1]),
                    ("key", guided["key_after"][stop - 1]), ("counts", expected_counts)):
                    exact(snapshots[name][point], expected, "snapshot after reward-processing return tick " + name)
                require(int(snapshots["t"][point]) == stop, "snapshot clock follows the return tick")
                exact(snapshots["x_sha256"][point], fingerprint(guided["activity"][stop - 1]), "actual acquired recurrent state")
            for name in STATE_FIELDS:
                hashes = typed(snapshots, name + "_sha256", (4, 32), "u")
                require(hashes.dtype == np.uint8, "actual neural state fingerprints")
                exact(life["probe_initial_" + name + "_sha256"], np.broadcast_to(hashes[:, None], (4, 12, 32)), "every probe gets the entire clone")
                exact(life["grid_initial_" + name + "_sha256"], hashes[-1], "grid gets unprobed terminal brain")
            for name in ("W", "Wp", "mask"):
                exact(life["probe_initial_" + name + "_sha256"], life["probe_final_" + name + "_sha256"], "readout freezes both weight updates")
            for name in ("hebbian_enabled", "predictor_enabled", "feedback_applied"):
                require(not typed(life, "probe_" + name, (4, 12), "biu", binary=True).any(), "assay never learns or returns feedback")
            for name in BODY:
                exact(life["probe_initial_" + name], life["probe_final_" + name], "discarded assay produces no move/reward/body feedback")
            for stem in ("probe_initial_", "grid_initial_"):
                for field, expected in (("energy", 1), ("alive", 1), ("sig", 0), ("denergy", 0), ("t", 0), ("counts", 0)):
                    require(np.all(life[stem + field] == expected), "standardized fresh readout body " + field)
            require(np.all(life["probe_initial_pos"] == 8), "probe uses neutral hub")
            softmax(life["probe_logits"], life["probe_probabilities"])
            exact(life["probe_logits"], np.float32(3) * life["probe_decision_motor"], "unmasked probe logits")
            for orient, placement in enumerate(placements):
                scene = pair_grid(placement)
                for point in range(4):
                    exact(life["probe_initial_grid"][point, orient], scene, "each declared two-item probe")
                    exact(life["probe_observation"][point, orient], observation(scene, np.array([8, 8]), 1, 0)[None], "neutral body and no hidden rule in probe")
            initial, final = prefixed(life, "grid_initial_"), prefixed(life, "grid_final_")
            exact(initial["key"], keys[4], "fresh independent autonomous action stream")
            require(np.sum(initial["grid"] == 1) == np.sum(initial["grid"] == 2) == 24, "fresh ordinary target resource counts")
            if paired_grid is None:
                paired_grid = (initial["grid"], initial["pos"])
            exact(initial["grid"], paired_grid[0], "same target layout for every treatment and rule")
            exact(initial["pos"], paired_grid[1], "same target birth location")
            trace = {name: value for name, value in prefixed(life, "grid_").items() if not name.startswith(("initial_", "final_"))}
            grid_results.append(trace_events(trace, initial, np.full(50, 1 - rule, np.int32),
                hebbian=np.full(50, plastic_arm and not arm.endswith("frozen")), predictor=plastic_arm))
            expected_counts = np.zeros((1, 2, 2), np.float32)
            expected_counts[0, 0] = [trace[name].sum() for name in ("food", "poison")]
            for name, expected in (("grid", trace["grid"][-1]), ("pos", trace["position"][-1]), ("energy", trace["energy"][-1]),
                ("alive", trace["alive_after"][-1]), ("sig", trace["signal"][-1]), ("denergy", trace["denergy"][-1]),
                ("key", trace["key_after"][-1]), ("counts", expected_counts)):
                exact(final[name], expected, "actual transfer terminal state " + name)
            require(int(final["t"]) == 50, "fifty assigned transfer turns")
            exact(final["x_sha256"], fingerprint(trace["activity"][-1]), "actual final recurrent activity")
            for state in (birth, *[{name: value[point] for name, value in snapshots.items()} for point in range(4)], initial, final):
                exact(state["mask_sha256"], birth["mask_sha256"], "unchanged inherited structural mask")
                if not plastic_arm:
                    for name, expected in fixed_direct.items():
                        exact(state[name + "_sha256"], expected, "unbounded direct weights and frozen predictor")
            if arm.endswith("frozen"):
                exact(snapshots["W_sha256"][1:], np.broadcast_to(snapshots["W_sha256"][1], (3, 32)), "Hebbian cutoff preserves later guided weights")
                exact(final["W_sha256"], initial["W_sha256"], "Hebbian intervention persists in transfer")
        result["grid"][arm] = {name: np.asarray([value[name] for value in grid_results], np.float64) for name in grid_results[0]}
    for rule in (0, 1):
        for controller in ("plastic", "direct"):
            reference = lives[controller + "_stationary", rule]
            for arm in (name for name in ARMS if name.startswith(controller)):
                other = lives[arm, rule]
                for name, value in reference.items():
                    if name.startswith("birth_"):
                        exact(other[name], value, "identical fresh controller birth")
                    elif name.startswith("guided_"):
                        exact(other[name][:2 * pre], value[:2 * pre], "identical physical pre-boundary history")
                    elif name.startswith(("snapshot_", "probe_")):
                        exact(other[name][0], value[0], "identical complete boundary clone and zero-contact readout")
        intact, frozen = lives["plastic_reversal", rule], lives["plastic_frozen", rule]
        for name, value in intact.items():
            if name.startswith("guided_"):
                exact(frozen[name][:cutoff], value[:cutoff], "lesion begins only when changed feedback reaches the brain")
    reference = lives[ARMS[0], 0]
    for (arm, rule), life in lives.items():
        for name, value in life.items():
            if name.startswith("birth_"):
                exact(value, lives[arm, 0][name], "birth independent of initial hidden rule")
        for stem in ("guided", "grid"):
            for field in ("key_before", "key_after", "action_key", "regrowth_key", "regrowth_cell"):
                exact(life[stem + "_" + field], reference[stem + "_" + field], "matched random streams across controllers, rules and interventions")
        exact(life["probe_action_key"], reference["probe_action_key"], "matched discarded probe randomness")
    return result


def beyond(value, threshold=0.):
    return value["se"] is not None and value["mean"] >= threshold and value["mean"] > 2 * value["se"]


def lineage_gates(old, current, local_difference, grid_difference, grid_preference, hebbian_difference=None):
    result = dict(local_pass=bool(beyond(old) and beyond(current, .10) and beyond(local_difference, .10)),
                  transfer_pass=bool(beyond(grid_difference, .10) and beyond(grid_preference)))
    if hebbian_difference is not None:
        result["hebbian_local_pass"] = bool(beyond(hebbian_difference, .05))
    return result


def replicated_gates(panels):
    require(len(panels) == 3, "three complete lineage panels required")
    result = {}
    for architecture in ("plastic", "direct"):
        local = [index for index, panel in enumerate(panels) if panel["architectures"][architecture]["local_pass"]]
        transfer = [index for index in local if panels[index]["architectures"][architecture]["transfer_pass"]]
        selected = dict(local=local, transfer=transfer)
        if architecture == "plastic":
            selected["hebbian_local"] = [index for index in local if panels[index]["hebbian"]["hebbian_local_pass"]]
        result[architecture] = {name: dict(count=len(indices), seeds=indices, passed=len(indices) >= 2)
                                for name, indices in selected.items()}
    return result


def empty_development_evidence(config, status, summary, journal, family_entries):
    """Only an unsuccessful zero-record development attempt can precede retry1."""
    subset(config, dict(protocol=PROTOCOL, condition_split="development", generation=0,
        eval_seed=2026120600, expected_families=4, scientific_result=False), "original development identity")
    subset(config["args"], dict(name=DEVELOPMENT_NAME, seed=2026120600, generations=0, eval_worlds=4,
        allow_dirty=False, resume=None), "original benchmark command")
    require(config.get("retry_of") is None, "the predecessor must be the original attempt")
    for record in (status, summary):
        subset(record, dict(status="failed", completed_families=0, expected_families=4, family_seconds=[]), "empty failed predecessor")
    subset(summary, dict(protocol=PROTOCOL, condition_split="development", evaluation_seed=2026120600,
        seed_index=0, source_label="A0", scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=[]),
        "original failure evidence")
    require(isinstance(summary.get("error"), str) and bool(summary["error"])
            and "measurements" not in summary and journal == b"" and not family_entries,
            "retry1 cannot replace any recorded family, outcome or partial journal")


def summarize(rows):
    require(len(rows) >= 2, "at least two independent families for SE")
    local = {arm: {key: np.stack([row["local"][arm][key] for row in rows])
                   for key in rows[0]["local"][arm]} for arm in ARMS}
    grid = {arm: {key: np.stack([row["grid"][arm][key] for row in rows]).astype(np.float64)
                  for key in rows[0]["grid"][arm]} for arm in ARMS}
    result = dict(families=len(rows), initial_rule_replays=2 * len(rows), arms={}, architectures={}, hebbian={},
                  by_initial_rule={"0": {}, "1": {}})
    for arm in ARMS:
        result["arms"][arm] = dict(
            local=[dict(contacts=contact, **{key: stat(value[:, :, point].mean(1)) for key, value in local[arm].items()})
                   for point, contact in enumerate(CONTACTS)],
            grid={key: stat(value.mean(1)) for key, value in grid[arm].items()})
    for architecture in ("plastic", "direct"):
        changed, unchanged = architecture + "_reversal", architecture + "_stationary"
        s, old_s = local[changed]["S"], local[unchanged]["S"]
        good, old_good = grid[changed]["first_food"], grid[unchanged]["first_food"]
        local_estimates = dict(pre_old=stat(-old_s[:, :, 0].mean(1)), post_target=stat(s[:, :, -1].mean(1)),
            reversal_minus_stationary=stat((s[:, :, -1] - old_s[:, :, -1]).mean(1)))
        transfer = dict(reversal_minus_stationary=stat((good - old_good).mean(1)),
                        food_minus_poison=stat(grid[changed]["food_minus_poison"].mean(1)))
        gates = lineage_gates(local_estimates["pre_old"], local_estimates["post_target"], local_estimates["reversal_minus_stationary"],
                              transfer["reversal_minus_stationary"], transfer["food_minus_poison"])
        result["architectures"][architecture] = dict(local=local_estimates, transfer=transfer, **gates)
        for rule in (0, 1):
            result["by_initial_rule"][str(rule)][architecture] = dict(S=stat(s[:, rule, -1]),
                delta_S=stat(s[:, rule, -1] - old_s[:, rule, -1]), G=stat(good[:, rule]),
                delta_G=stat(good[:, rule] - old_good[:, rule]))
    difference = stat((local["plastic_reversal"]["S"][:, :, -1] - local["plastic_frozen"]["S"][:, :, -1]).mean(1))
    result["hebbian"] = dict(local_difference=difference,
        transfer_difference=stat((grid["plastic_reversal"]["first_food"] - grid["plastic_frozen"]["first_food"]).mean(1)),
        hebbian_local_pass=bool(beyond(difference, .05)))
    return result


class Auditor(prior.chain.Auditor):
    def __init__(self, root, source):
        require(re.fullmatch(r"[0-9a-f]{40}", source), "full immutable source commit required")
        super().__init__(root, source)
        self.code = None
        self.development_recovery = None

    def empty_benchmark(self):
        found = list(self.path("runs").glob("*_" + DEVELOPMENT_NAME))
        require(len(found) == 1, "retry1 requires the unique preserved original benchmark")
        predecessor = found[0]
        folder = predecessor.relative_to(self.root).as_posix()
        config, status, summary = (self.json(folder + "/" + name) for name in ("config.json", "status.json", "summary.json"))
        require((predecessor / "families").is_dir(), "original empty family directory must remain")
        require(not any(predecessor.rglob("family_*.npz")), "original attempt must contain no saved family anywhere")
        empty_development_evidence(config, status, summary, self.read(folder + "/metrics.jsonl"),
                                   list((predecessor / "families").iterdir()))
        old_commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", old_commit), "original clean source identity")
        self.code_trees(old_commit)
        require(set(config["source_code"]) == {"seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py",
                "scripts/run_encounter_diagnostic.py"}, "complete original source manifest")
        context = type("SourceContext", (), {"root": self.root, "source": old_commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
        for path, digest in config["source_code"].items():
            require(path in ("seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py", "scripts/run_encounter_diagnostic.py"),
                    "original source manifest path")
            blob = base.subprocess.run(["git", "show", old_commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        for field, pinned, copied in (("source", base.ORIGINALS[0], "plastic_genome.npy"),
                                     ("direct_source", prior.DIRECT[0], "direct_genome.npy")):
            path, digest = pinned
            subset(config[field], dict(path=path, sha256=digest, training_seed=0), "original benchmark controller")
            content = self.read(path)
            same(hashlib.sha256(content).hexdigest(), digest, "original benchmark source hash")
            require(self.read(folder + "/" + copied) == content, "original controller copy must remain unchanged")
        same(config["assignments"]["path"], "assignments.npz", "original assignment artifact path")
        schedule = self.artifact(folder, config["assignments"])
        files = {}
        for path in sorted(predecessor.rglob("*")):
            if path.is_file():
                require(path.resolve().is_relative_to(predecessor.resolve()), "predecessor artifact leaves its folder")
                relative = path.relative_to(predecessor).as_posix()
                files[relative] = hashlib.sha256(self.read(folder + "/" + relative)).hexdigest()
        return dict(run=folder, files=files), config, schedule

    def completed_bundle(self, development=False):
        count = 4 if development else 128
        folders, pending = [], []
        name = DEVELOPMENT_NAME
        if development:
            retries = list(self.path("runs").glob("*_" + DEVELOPMENT_NAME + "-retry*"))
            require(len(retries) <= 1 and all(path.name.endswith("_" + RETRY_NAME) for path in retries), "only the single registered retry1 is allowed")
            if retries:
                self.development_recovery, _, _ = self.empty_benchmark()
                name = RETRY_NAME
        for name in ([name] if development else [f"E127-A{i}" for i in range(3)]):
            found = list(self.path("runs").glob("*_" + name))
            require(len(found) <= 1, "duplicate canonical panel " + name)
            if not found or not (found[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = found[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            files = {p.name for p in (found[0] / "families").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_families") != count
                    or status.get("expected_families") != count or not (found[0] / "summary.json").is_file()
                    or files != {f"family_{i:04d}.npz" for i in range(count)}):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return folders

    def artifact(self, folder, manifest):
        path = self.path(folder + "/" + manifest["path"])
        require(path.is_relative_to(self.path(folder)), "artifact leaves its run")
        relative = path.relative_to(self.root).as_posix()
        result = self.arrays(relative)
        same(self.hashes[relative], manifest["sha256"], "exact archived artifact hash")
        return result

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master = (4, 2026120600) if development else (128, 2026120601 + index)
        retry = development and self.path(folder).name.endswith("_" + RETRY_NAME)
        name = RETRY_NAME if retry else DEVELOPMENT_NAME if development else f"E127-A{index}"
        old_schedule = None
        if retry:
            proof, old_config, old_schedule = self.empty_benchmark()
            same(config.get("retry_of"), proof, "exact preserved empty predecessor receipt")
            self.development_recovery = proof
            for field in ("protocol", "source", "direct_source", "cfg", "cfg_overrides", "condition_split", "generation",
                "eval_seed", "expected_families", "statistical_unit", "assignment_stream_tag", "arms", "key_roles",
                "initial_rules", "pre_contact_choices", "post_contacts", "readout_contacts", "grid_steps", "probe_placements",
                "raw_schema", "raw_pattern", "runtime", "max_seconds", "scientific_result"):
                same(config[field], old_config[field], "retry preserves registered setting " + field)
            unchanged = lambda args: {key: value for key, value in args.items() if key not in ("name", "hypothesis")}
            same(unchanged(config["args"]), unchanged(old_config["args"]), "retry changes only its identity/comment")
        else:
            require(config.get("retry_of") is None, "only registered development retry1 can carry a predecessor")
        subset(config, dict(protocol=PROTOCOL, condition_split="development" if development else "final",
            cfg=CFG, cfg_overrides={}, generation=0, eval_seed=master, expected_families=count,
            statistical_unit="independent paired base family", arms=list(ARMS),
            key_roles=["script", "birth", "chamber", "grid", "action", "probe"], initial_rules=[0, 1],
            pre_contact_choices=[8, 16], post_contacts=16, readout_contacts=list(CONTACTS), grid_steps=50,
            probe_placements=12, raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
            assignment_stream_tag=0xE127, max_seconds=180 if development else 3600, scientific_result=False), "registered panel")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, allow_dirty=False,
            resume=None, cfg=None, no_plastic=False, objective=None, memory_condition=None), "registered invocation")
        require(config["git"]["dirty"] is False and self.code_trees(config["git"]["commit"]) == self.expected_trees,
                "clean registered scientific source")
        expected_paths = {"seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py", "scripts/run_encounter_diagnostic.py"}
        require(set(config["source_code"]) == expected_paths, "complete scientific source manifest")
        if self.code is None:
            attributes = source_bytes.Auditor.source_attributes(self, list(config["source_code"]))
            for path, digest in config["source_code"].items():
                blob = base.subprocess.run(["git", "show", self.source + ":" + path], cwd=self.root,
                    capture_output=True, check=True, timeout=15).stdout
                source_bytes.source_hash_representation(blob, digest, attributes[path])
            self.code = config["source_code"]
        same(config["source_code"], self.code, "same scientific source across panels")
        originals = []
        for field, pinned, copied, shape in (("source", base.ORIGINALS[index], "plastic_genome.npy", (155,)),
                                             ("direct_source", prior.DIRECT[index], "direct_genome.npy", (8424,))):
            path, digest = pinned
            subset(config[field], dict(path=path, sha256=digest, training_seed=index), "original controller source")
            content = self.read(path)
            same(hashlib.sha256(content).hexdigest(), digest, "original controller digest")
            require(self.read(folder + "/" + copied) == content, "copied original bytes")
            value = np.load(io.BytesIO(content), allow_pickle=False)
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), "original controller format")
            originals.append(value)
        schedule = self.artifact(folder, config["assignments"])
        if old_schedule is not None:
            require(set(schedule) == set(old_schedule), "retry preserves assignment fields")
            for field in schedule:
                exact(schedule[field], old_schedule[field], "retry keeps the original development assignment")
        permutation = typed(schedule, "permutation", (count,), "iu")
        require(np.array_equal(np.sort(permutation), np.arange(count)), "complete assignment permutation")
        exact(schedule["pre_contacts"], np.repeat(np.array([8, 16], np.int32), count // 2)[permutation], "balanced hidden boundary")
        # Only random-key primitives are replayed; the simulator is never imported.
        jax, _ = source_bytes.random_runtime()
        same(config["runtime"]["jax"], jax.__version__, "archived PRNG runtime")
        key = jax.random.fold_in(jax.random.PRNGKey(master), 0xE127)
        exact(schedule["key"], np.asarray(key), "independent assignment stream")
        exact(permutation, np.asarray(jax.random.permutation(key, count), np.int32), "independent assigned boundaries")
        self.register_keys(typed(schedule, "key", (2,), "u")[None], "panel assignment")
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "all assigned families retained")
        for family, record in enumerate(records):
            subset(record, dict(family=family, pre_contacts=int(schedule["pre_contacts"][family])), "assignment journal")
            keys = np.asarray(record["keys"], np.uint32)
            require(keys.shape == (6, 2), "separate script, birth, chamber, grid, action and probe roles")
            expected_keys = np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(master), family), 6))
            exact(keys, expected_keys, "fixed assigned family key roles")
            self.register_keys(keys, "independent family")
        return config, schedule, records, originals

    def panel(self, folder, index, development):
        config, schedule, records, genomes = self.metadata(folder, index, development)
        count = config["expected_families"]
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, seed_index=index, evaluation_seed=config["eval_seed"],
            artifact_validity="passed", scientific_result=False, gate_status="not_applicable" if development else "undecided"), "complete panel")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"], "complete bounded run")
        rows = []
        for family, record in enumerate(records):
            manifest = summary["raw_files"][family]
            subset(manifest, dict(index=family, path=f"families/family_{family:04d}.npz"), "ordered immutable family")
            same(record["raw_file"], manifest, "journal artifact")
            raw = self.artifact(folder, manifest)
            exact(raw["keys"], np.asarray(record["keys"], np.uint32), "actual assigned keys")
            rows.append(audit_family(raw, family, int(schedule["pre_contacts"][family]), *genomes))
        measured = summarize(rows)
        same(summary["measurements"], measured, "independent probabilities, assigned outcomes, SE and gates")
        if development and self.development_recovery is not None:
            same(self.empty_benchmark()[0], self.development_recovery, "preserved predecessor remained unchanged during audit")
        return dict(index=index, run=folder, measurements=measured)


def self_check():
    placements = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
    uniform = np.full((2, 4, 12, 5), .2, np.float32)
    softmax(np.zeros_like(uniform), uniform)
    vectors = local_vectors(uniform, placements)
    for field, expected in (("p_food", .2), ("p_poison", .2), ("approach", .4), ("other", .6), ("S", 0)):
        prior.prior.close(vectors[field], np.full((2, 4), expected), "uniform five-way floor")
    for obsolete in (False, True):
        policy = np.zeros_like(uniform)
        for rule in (0, 1):
            actions = placements[:, rule if obsolete else 1 - rule]
            policy[rule, :, np.arange(12), actions] = 1
        require(np.all(local_vectors(policy, placements)["S"] == (-1 if obsolete else 1)), "current versus obsolete rule fixture")
    masked = np.zeros_like(uniform)
    masked[..., 1:3] = .5
    missing = uniform.copy()
    missing[..., 0] = 0
    for corrupted in (masked, np.zeros_like(uniform), missing):
        try:
            softmax(np.zeros_like(corrupted), corrupted)
        except base.AuditError:
            continue
        raise base.AuditError("renormalized, zeroed or missing probability mass accepted")
    outcomes, rows = [], []
    for pair in ((1, -1), (1, 1), (-1, -1), (0, 0)):
        per_rule = []
        for outcome in pair:
            food, poison = np.zeros(50, bool), np.zeros(50, bool)
            food[-1], poison[0] = outcome == 1, outcome == -1
            per_rule.append(grid_outcomes(food, poison, np.full(50, outcome != 0)))
        grid = {name: np.asarray([value[name] for value in per_rule]) for name in per_rule[0]}
        rows.append(dict(local={arm: vectors for arm in ARMS}, grid={arm: grid for arm in ARMS}))
        outcomes.append(grid["food_minus_poison"].mean())
    measured = summarize(rows)
    expected = dict(mean=0., se=float(np.sqrt(1 / 6)), n=4)
    same(measured["architectures"]["plastic"]["transfer"]["food_minus_poison"], expected, "signed Y has its own family SE")
    require(rows[-1]["grid"][ARMS[0]]["first_food"].sum() == 0
            and np.all(rows[-1]["grid"][ARMS[0]]["latency"] == 51), "dead and missing lives remain assigned failures")
    certainty = lambda mean, se=0.: dict(mean=mean, se=se, n=4)
    require(lineage_gates(certainty(.2), certainty(.1), certainty(.1), certainty(.1), certainty(.1))["local_pass"], "inclusive effect floor")
    require(not lineage_gates(certainty(.2), certainty(.1), certainty(.1, .05), certainty(.1), certainty(.1))["local_pass"], "strict two-SE threshold")
    require(not lineage_gates(certainty(0), certainty(.2), certainty(.2), certainty(.2), certainty(.2))["local_pass"], "must first establish old preference")
    panels = [dict(architectures={architecture: dict(local_pass=i < 2, transfer_pass=i > 0)
                    for architecture in ("plastic", "direct")}, hebbian=dict(hebbian_local_pass=i != 1)) for i in range(3)]
    require(not replicated_gates(panels)["plastic"]["transfer"]["passed"]
            and not replicated_gates(panels)["plastic"]["hebbian_local"]["passed"], "same qualifying lineages must replicate each effect")
    weights = prior.direct_weights(np.full(8424, 4., np.float32))
    require(weights.max() == 4 and not np.array_equal(fingerprint(weights), fingerprint(np.clip(weights, -3, 3))), "unbounded direct-weight fixture")
    original = dict(protocol=PROTOCOL, condition_split="development", generation=0, eval_seed=2026120600,
        expected_families=4, scientific_result=False, args=dict(name=DEVELOPMENT_NAME, seed=2026120600,
            generations=0, eval_worlds=4, allow_dirty=False, resume=None))
    stopped = dict(status="failed", completed_families=0, expected_families=4, family_seconds=[])
    failure = dict(stopped, protocol=PROTOCOL, condition_split="development", evaluation_seed=2026120600,
        seed_index=0, source_label="A0", scientific_result=False, artifact_validity="incomplete", gate_status="undecided", raw_files=[], error="fixture failure")
    empty_development_evidence(original, stopped, failure, b"", [])
    cases = [(stopped, dict(failure, measurements={}), b"", []),
             (stopped, dict(failure, raw_files=["family_0000.npz"]), b"", []),
             (dict(stopped, completed_families=1), failure, b"", []),
             (stopped, failure, b"\n", []), (stopped, failure, b"", ["family_0000.npz"]),
             (dict(stopped, status="interrupted"), failure, b"", [])]
    for state, result, journal, files in cases:
        try:
            empty_development_evidence(original, state, result, journal, files)
        except base.AuditError:
            continue
        raise base.AuditError("nonempty or unregistered retry predecessor accepted")
    require("seed.seed" not in sys.modules and "jax" not in sys.modules, "self-check must use no simulator or JAX")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=6,
        coverage=["five-action floors and current/obsolete rule readers", "no-contact/death, horizon and paired-family signed SE",
                  "effect sizes and strict uncertainty gates", "same-lineage replication intersections", "unclipped direct weights",
                  "retry1 accepts only an empty failed original; partial evidence and nonfailed attempts are rejected"],
        simulator_imported=False)


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
        parser.error("--source-commit is required")
    auditor = None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            # Development assignment metadata only: no scores or raw events.
            auditor.metadata(auditor.completed_bundle(True)[0], 0, True)
        panels = [auditor.panel(folder, index, args.development) for index, folder in enumerate(folders)]
        if auditor.development_recovery is not None:
            same(auditor.empty_benchmark()[0], auditor.development_recovery, "original failure remains preserved at audit completion")
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, development_recovery=auditor.development_recovery,
            replicated_gates=None if args.development else replicated_gates([p["measurements"] for p in panels]),
            limitations=["Events, body feedback, cloned-state fingerprints and probabilities are reconstructed; neural weight updates are not independently replayed.",
                "Key assignment uses the archived JAX random primitives; action sampling and neural matrix operations are not rerun.",
                "Guided exposure and fresh-body readouts are explicit interventions; weak grid transfer does not isolate navigation.",
                "Only original E103/E113 controllers are assessed. This does not decide E126 or demonstrate broader adaptation."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), replicated_gates=None)
        code = 2
    except (base.AuditError, OSError, KeyError, ValueError, TypeError, IndexError, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", replicated_gates=None)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
