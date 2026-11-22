"""Independent E133 history, contact-timing, physical solo and decision audit.

The verifier imports no controller or simulator. JAX is loaded only when the
registered random assignments are reconstructed; synthetic checks use NumPy.
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_retention_diagnostic as shared

encounter, base, source_bytes = shared.encounter, shared.base, shared.source_bytes
require, same, subset, typed = shared.require, shared.same, shared.subset, shared.typed
exact, fingerprint, stat = shared.exact, shared.fingerprint, shared.stat
CFG, MOVES, NEURAL = shared.CFG, shared.MOVES, shared.STATE_FIELDS
PROTOCOL = "contact-gate-v1"
POLICIES = ("native", "contact", "sparse", "reopened")
HISTORIES = ("stationary", "reversal")
ARMS = tuple(policy + "_" + history for policy in POLICIES for history in HISTORIES)
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "sparse")
SOURCE_PATHS = {"seed/seed.py", "seed/exploration.py", "seed/direct.py", "seed/encounter_diagnostic.py",
    "seed/contact_gate.py", "scripts/run_contact_gate.py", "scripts/run_cold_donor.py",
    "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py", "scripts/run_shelter_evolution.py"}
APPARATUS = dict(pre_contacts=16, post_contacts=8, block_steps=4, delay_ticks=33, grid_steps=50, local_placements=12)
PLACEMENTS = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
FIELDS = ("local_S", "solo_Y", "first_food", "first_poison", "no_contact", "exposure_survival", "solo_survival")
TRACE_FIELDS = ("observation", "prior_observation", "prior_efference", "action", "item", "energy_before", "energy",
    "denergy", "incoming_denergy", "alive_before", "alive_after", "hebbian", "predictor", "weight_same")
PROBE_FIELDS = ("probe_logits", "probe_probabilities", "probe_alive", "checkpoint_hashes", "grid_initial_hashes")
GRID_FIELDS = ("grid_food", "grid_poison", "grid_alive_before", "grid_alive_after", "grid_action", "grid_item",
    "grid_energy_before", "grid_energy", "grid_denergy", "grid_regrowth_cell")
COMPONENT_FLOORS = dict(pre_old=0., gate_immediate=.10, native_immediate=.10, gate_delayed=.10,
    gate_solo_delayed=0., retention_advantage=.025, gate_local_vs_sparse=.05, delay_ablation=.025,
    gate_solo_vs_native=.05, gate_solo_vs_sparse=.05)
SAFEGUARD_NAMES = ("acquisition_vs_native", "gate_own_retention", "stationary_local_native", "stationary_local_sparse",
    "stationary_solo_native", "stationary_solo_sparse", "solo_survival_stationary_native", "solo_survival_stationary_sparse",
    "solo_survival_reversal_native", "solo_survival_reversal_sparse")


def numeric(values, name, shape, dtype):
    value = typed(values, name, shape, np.dtype(dtype).kind)
    require(value.dtype == np.dtype(dtype) and np.isfinite(value).all(), "exact native finite dtype " + name)
    return value


def physical_equal(actual, expected, label):
    """Allow only local float32 expression roundoff, never a missed clip/meal."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
        and np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                   <= 8 * np.finfo(np.float32).eps * np.maximum(1., np.abs(expected))), label)


@lru_cache(maxsize=1)
def random_helpers():
    """Only random assignments are replayed, independently of simulation code."""
    jax, jnp = source_bytes.random_runtime()

    @jax.jit
    def derive(keys):
        kd, kc, ko = jax.random.split(keys[0], 3)
        directions = jax.vmap(lambda k: jax.random.permutation(k, jnp.arange(1, 5)))(jax.random.split(kd, 4))
        first = jax.random.randint(kc, (4, 4), 1, 3)
        colors = jnp.stack((first, 3-first), axis=-1).reshape(-1)
        outward = jnp.repeat(directions, 2, axis=1).reshape(-1)
        offsets = jax.random.randint(ko, (32,), 1, 4)
        other = (outward-1+offsets) % 4+1
        a, b = jnp.where(colors == 1, outward, other), jnp.where(colors == 2, outward, other)
        script = jnp.stack((a, b, colors, outward, jnp.asarray([0, 2, 1, 4, 3])[outward]), axis=1)[:24]
        k1, k2 = jax.random.split(keys[3])
        cells = jax.random.choice(k1, 256, (48,), replace=False)
        grid = jnp.zeros(256, jnp.int32).at[cells[:24]].set(1).at[cells[24:]].set(2).reshape(16, 16)

        def step(previous, _):
            following, _, regrowth = jax.random.split(previous, 3)
            return following, jax.random.randint(regrowth, (1, 2), 0, 16)[0]

        return dict(script=script.astype(jnp.int32), solo_grid=grid,
            solo_position=jax.random.randint(k2, (1, 2), 0, 16)[0],
            solo_regrowth_cell=jax.lax.scan(step, keys[4], None, length=50)[1],
            sparse_offsets=jax.random.randint(keys[6], (8,), 0, 4, dtype=jnp.int32))
    return jax, derive


def contact_mask(prior_observation, prior_efference, alive):
    """The destination's previous visible item, with no current/future outcome."""
    obs, eff, alive = np.asarray(prior_observation), np.asarray(prior_efference), np.asarray(alive)
    require(obs.shape[:-1] == eff.shape[:-1] == alive.shape and obs.shape[-1] == 103 and eff.shape[-1] == 5,
            "contact gate uses aligned prior observation and actual efference")
    require(np.isfinite(obs).all() and np.isfinite(eff).all() and np.isin(eff, (0, 1)).all()
            and np.isin(eff.sum(-1), (0, 1)).all(), "actual action is one-hot or absent at birth")
    indices = (MOVES[:, 0] + 2) * 5 + MOVES[:, 1] + 2
    visible = obs[..., indices] + obs[..., indices + 25]
    return (np.sum(visible * eff, axis=-1) > 0) & alive.astype(bool)


def next_energy(energy, alive, item, rule):
    food, poison = bool(alive and item == 1+rule), bool(alive and item == 2-rule)
    energy = np.float32(energy)
    updated = np.float32(np.clip(energy - np.float32(.003) * np.float32(alive)
        + np.float32(.25) * np.float32(food) - np.float32(.25) * np.float32(poison), 0., 2.))
    return updated, np.float32((updated-energy) * np.float32(alive)), bool(alive and updated > 0), food, poison


def pre_history(data, script, rule):
    """Rebuild the common 16 physical outward/return encounters."""
    pos, energy, delta, grid = np.asarray([8, 8], np.int32), np.float32(1), np.float32(0), None
    for tick in range(32):
        if tick % 2 == 0:
            grid = encounter.pair_grid(script[tick // 2, :2])
        observation = encounter.observation(grid, pos, energy, delta)
        exact(data["observation"][tick], observation, "common prehistory sees only physical scene and realized feedback")
        action = int(script[tick // 2, 3 + tick % 2])
        require(data["action"][tick] == action, "common prehistory actual scripted action")
        pos = (pos + MOVES[action]) % 16
        item = int(grid[tuple(pos)])
        require(data["item"][tick] == item, "common prehistory actual item")
        expected, de, alive, _, _ = next_energy(energy, True, item, rule)
        physical_equal(data["energy"][tick], expected, "prehistory clips its actual energy")
        physical_equal(data["denergy"][tick], de, "prehistory reports realized, clipped feedback")
        require(bool(data["alive_after"][tick]) == alive and alive, "registered common exposure must remain alive")
        grid[tuple(pos)] = 0
        energy, delta = data["energy"][tick], data["denergy"][tick]
    exact(pos, np.asarray([8, 8], np.int32), "prehistory ends at the original hub")
    return dict(energy=energy, denergy=delta, observation=data["observation"][-1],
        efference=np.eye(5, dtype=np.float32)[int(data["action"][-1])])


def treatment_history(data, script, sparse_offsets, rule, policy, initial):
    """Rebuild all 32 matched treatment and 33 empty-delay ticks."""
    grid, pos = np.zeros((16, 16), np.int32), np.asarray([8, 8], np.int32)
    energy, delta, previous, efference = (initial[name] for name in ("energy", "denergy", "observation", "efference"))
    for tick in range(65):
        offset = tick % 4
        if tick >= 32 or offset == 0:
            grid[:] = 0
        if tick < 32 and offset == 2:
            grid = encounter.pair_grid(script[16 + tick // 4, :2])
        exact(data["prior_observation"][tick], previous, "gate receives previous actual observation")
        exact(data["prior_efference"][tick], efference, "gate receives previous executed action")
        physical_equal(data["energy_before"][tick], energy, "treatment energy carries without resets")
        physical_equal(data["incoming_denergy"][tick], delta, "contact outcome reaches precisely its next neural tick")
        expected_obs = encounter.observation(grid, pos, energy, delta)
        exact(data["observation"][tick], expected_obs, "matched visible physical input with actual feedback")
        require(bool(data["alive_before"][tick]) and bool(data["predictor"][tick]), "living exposure keeps native predictor active")
        if policy == "native" or policy == "reopened" and tick >= 32:
            enabled = True
        elif policy == "sparse":
            enabled = tick < 32 and offset == int(sparse_offsets[tick // 4])
        else:
            enabled = bool(contact_mask(previous, efference, np.asarray(True)))
        require(bool(data["hebbian"][tick]) == enabled, "locally observed or independently assigned update timing")
        require(enabled or bool(data["weight_same"][tick]), "disabled update must preserve every W bit")
        action = int(script[16 + tick // 4, offset + 1]) if tick < 32 and offset >= 2 else 0
        require(data["action"][tick] == action, "identical predetermined history action")
        pos = (pos + MOVES[action]) % 16
        item = int(grid[tuple(pos)])
        require(data["item"][tick] == item, "actual contact follows visible grid and executed move")
        expected, de, alive, _, _ = next_energy(energy, True, item, rule)
        physical_equal(data["energy"][tick], expected, "treatment clips actual physical energy")
        physical_equal(data["denergy"][tick], de, "treatment uses realized clipped feedback")
        require(bool(data["alive_after"][tick]) == alive and alive, "registered paired exposure must remain alive")
        grid[tuple(pos)] = 0
        previous, efference = data["observation"][tick], np.eye(5, dtype=np.float32)[action]
        energy, delta = data["energy"][tick], data["denergy"][tick]
    count = int(data["hebbian"].sum())
    require(count == dict(native=65, contact=8, sparse=8, reopened=41)[policy], "exact matched eight-update sparse dose")
    exact(pos, np.asarray([8, 8], np.int32), "delay has no body teleport")


def local_score(logits, probabilities, alive, rule):
    require(logits.shape == probabilities.shape == (12, 5), "all twelve placements and five actions retained")
    if not alive:
        require(not np.any(logits) and not np.any(probabilities), "dead local clone is not revived or given a policy")
        return 0.
    require(np.all(np.abs(logits) <= 3), "unmasked bounded native local logits")
    encounter.softmax(logits, probabilities)
    values = probabilities.astype(np.float64)
    return float(np.mean(values[np.arange(12), PLACEMENTS[:, rule]] - values[np.arange(12), PLACEMENTS[:, 1-rule]]))


def solo_trace(data, grid, position, regrowth, rule, alive):
    """Rebuild the 50-turn fresh physical world from independent random cells."""
    if not alive:
        require(all(not np.any(value) for value in data.values()), "dead source receives no solo body, actions or events")
        return dict(first_food=0., first_poison=0., no_contact=1., solo_Y=0., solo_survival=0.)
    grid, pos, energy, live = grid.copy(), position.copy(), np.float32(1), True
    exact(data["grid_regrowth_cell"], regrowth, "same independently regenerated solo resource stream")
    for tick in range(50):
        require(bool(data["grid_alive_before"][tick]) == live, "solo survival is absorbing without revival")
        physical_equal(data["grid_energy_before"][tick], energy, "solo body energy carries without reset")
        action = int(data["grid_action"][tick])
        require(0 <= action <= 4, "solo action keeps all five ordinary choices")
        pos = (pos + MOVES[action] * int(live)) % 16
        item = int(grid[tuple(pos)])
        require(data["grid_item"][tick] == item, "solo item follows independently rebuilt movement and resource grid")
        updated, delta, after, food, poison = next_energy(energy, live, item, rule)
        require(bool(data["grid_food"][tick]) == food and bool(data["grid_poison"][tick]) == poison,
                "solo food and poison labels follow actual item and current rule")
        physical_equal(data["grid_energy"][tick], updated, "solo clips actual resource energy")
        physical_equal(data["grid_denergy"][tick], delta, "solo feedback is actual clipped energy change")
        require(bool(data["grid_alive_after"][tick]) == after, "solo death follows actual energy")
        if food or poison:
            grid[tuple(pos)] = 0
            grid[tuple(regrowth[tick])] = item
        energy, live = data["grid_energy"][tick], after
    outcome = encounter.grid_outcomes(data["grid_food"], data["grid_poison"], data["grid_alive_after"])
    return {name: outcome[source] for name, source in (("first_food", "first_food"), ("first_poison", "first_poison"),
        ("no_contact", "no_contact"), ("solo_Y", "food_minus_poison"), ("solo_survival", "survival"))}


def audit_family(raw, index, genome, keys, randomness):
    top = {"base_index", "keys", "initial_rules", "genome_sha256", "script", "sparse_offsets", "solo_grid", "solo_position"}
    pre_shapes = dict(observation=((2, 32, 103), np.float32), action=((2, 32), np.int32), energy=((2, 32), np.float32),
        denergy=((2, 32), np.float32), item=((2, 32), np.int32), alive_after=((2, 32), bool),
        logits=((2, 12, 5), np.float32), probabilities=((2, 12, 5), np.float32), neural_hashes=((2, 4, 32), np.uint8), alive=((2,), bool))
    arm_shapes = {name: ((2, 65), np.float32) for name in ("energy_before", "energy", "denergy", "incoming_denergy")}
    arm_shapes.update({name: ((2, 65), bool) for name in ("alive_before", "alive_after", "hebbian", "predictor", "weight_same")})
    arm_shapes.update({name: ((2, 65), np.int32) for name in ("action", "item")})
    arm_shapes.update(observation=((2, 65, 103), np.float32), prior_observation=((2, 65, 103), np.float32),
        prior_efference=((2, 65, 5), np.float32), probe_logits=((2, 2, 12, 5), np.float32),
        probe_probabilities=((2, 2, 12, 5), np.float32), probe_alive=((2, 2), bool),
        checkpoint_hashes=((2, 2, 4, 32), np.uint8), grid_initial_hashes=((2, 2, 4, 32), np.uint8),
        grid_regrowth_cell=((2, 2, 50, 2), np.int32))
    arm_shapes.update({name: ((2, 2, 50), bool) for name in ("grid_food", "grid_poison", "grid_alive_before", "grid_alive_after")})
    arm_shapes.update({name: ((2, 2, 50), np.int32) for name in ("grid_action", "grid_item")})
    arm_shapes.update({name: ((2, 2, 50), np.float32) for name in ("grid_energy_before", "grid_energy", "grid_denergy")})
    require(set(raw) == top | {"pre_" + name for name in pre_shapes} | {arm + "_" + name for arm in ARMS for name in arm_shapes},
            "complete exact registered contact-gate evidence schema")
    for name, shape, dtype in (("base_index", (), np.int32), ("keys", (7, 2), np.uint32), ("initial_rules", (2,), np.int32),
        ("genome_sha256", (32,), np.uint8), ("script", (24, 5), np.int32), ("sparse_offsets", (8,), np.int32),
        ("solo_grid", (16, 16), np.int32), ("solo_position", (2,), np.int32)):
        numeric(raw, name, shape, dtype)
    exact(raw["base_index"], np.asarray(index, np.int32), "every original assigned family in order")
    exact(raw["keys"], keys, "seven independently assigned role streams")
    exact(raw["initial_rules"], np.asarray([0, 1], np.int32), "both initial rules remain assigned")
    exact(raw["genome_sha256"], fingerprint(genome), "original fixed descendant genome")
    for name in ("script", "sparse_offsets", "solo_grid", "solo_position"):
        exact(raw[name], randomness[name], "independently reconstructed " + name)
    require(np.isin(raw["sparse_offsets"], np.arange(4)).all(), "sparse timing is chosen before each replay")
    script = raw["script"]
    require(np.isin(script[:, :2], np.arange(1, 5)).all() and np.all(script[:, 0] != script[:, 1])
        and np.isin(script[:, 2], (1, 2)).all(), "valid ordered visible colors and assigned contacts")
    exact(script[:, 3], script[np.arange(24), script[:, 2]-1], "outward move matches scheduled color")
    exact(script[:, 4], np.asarray([0, 2, 1, 4, 3], np.int32)[script[:, 3]], "return exactly reverses outward movement")
    require(all(set(script[start:start+2, 2]) == {1, 2} for start in range(0, 24, 2)), "every adjacent contact pair samples both colors")
    pre = {name: numeric(raw, "pre_" + name, *spec) for name, spec in pre_shapes.items()}
    require(np.all(pre["alive"]) and np.all(pre["alive_after"]), "pre-exposure acquisition cannot silently lose assigned lives")
    exact(pre["neural_hashes"][0, 3], pre["neural_hashes"][1, 3], "one birth mask independent of hidden initial rule")
    initials = [pre_history({name: value[rule] for name, value in pre.items()}, script, rule) for rule in (0, 1)]
    result = dict(pre_old=np.asarray([local_score(pre["logits"][rule], pre["probabilities"][rule], pre["alive"][rule], rule)
        for rule in (0, 1)]), sparse_feedback_overlap=float(np.mean(raw["sparse_offsets"] == 3)), arms={})
    arms = {}
    for arm in ARMS:
        policy, history = arm.split("_")
        data = {name: numeric(raw, arm + "_" + name, *spec) for name, spec in arm_shapes.items()}
        arms[arm] = data
        exact(data["grid_initial_hashes"], data["checkpoint_hashes"], "every solo clone uses the complete unprobed checkpoint")
        exact(data["checkpoint_hashes"][:, :, 3], np.repeat(pre["neural_hashes"][:, None, 3], 2, axis=1), "structural mask never changes")
        exact(data["probe_alive"], data["alive_after"][:, [31, 64]], "readout eligibility comes from actual exposure liveness")
        if policy in ("contact", "sparse"):
            exact(data["checkpoint_hashes"][:, 0, 0], data["checkpoint_hashes"][:, 1, 0], "empty disabled delay preserves acquired W bits")
        endpoints = {field: np.zeros((2, 2), np.float64) for field in FIELDS}
        for initial_rule in (0, 1):
            rule = initial_rule if history == "stationary" else 1-initial_rule
            treatment_history({name: data[name][initial_rule] for name in TRACE_FIELDS}, script, raw["sparse_offsets"], rule, policy, initials[initial_rule])
            for checkpoint in (0, 1):
                alive = bool(data["probe_alive"][initial_rule, checkpoint])
                endpoints["local_S"][initial_rule, checkpoint] = local_score(data["probe_logits"][initial_rule, checkpoint],
                    data["probe_probabilities"][initial_rule, checkpoint], alive, rule)
                endpoints["exposure_survival"][initial_rule, checkpoint] = float(alive)
                outcome = solo_trace({name: data[name][initial_rule, checkpoint] for name in GRID_FIELDS}, raw["solo_grid"],
                    raw["solo_position"], randomness["solo_regrowth_cell"], rule, alive)
                for field, value in outcome.items():
                    endpoints[field][initial_rule, checkpoint] = value
        result["arms"][arm] = endpoints
    shared_fields = ("observation", "prior_observation", "prior_efference", "action", "item", "energy_before", "energy",
                     "denergy", "incoming_denergy", "alive_before", "alive_after", "predictor")
    for history in HISTORIES:
        for policy in POLICIES[1:]:
            for field in shared_fields:
                exact(arms[policy + "_" + history][field], arms["native_" + history][field], "identical history, actions and received feedback across update policies")
        contact, reopened = arms["contact_" + history], arms["reopened_" + history]
        for field in TRACE_FIELDS:
            exact(contact[field][:, :32], reopened[field][:, :32], "reopened ablation shares acquired contact history")
        for field in PROBE_FIELDS + GRID_FIELDS:
            exact(contact[field][:, 0], reopened[field][:, 0], "reopened ablation starts from the same complete acquired brain and paired readout")
    return result


def decisions(components, safeguards):
    criteria = {}
    for name, floor in COMPONENT_FLOORS.items():
        value = components[name]
        criteria[name] = bool(value["se"] is not None and value["mean"] >= floor and value["mean"] > 2 * value["se"])
    for name in SAFEGUARD_NAMES:
        value = safeguards[name]
        criteria[name] = bool(value["lower_2se"] is not None and value["lower_2se"] > -.05)
    return dict(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, value in criteria.items() if not value])


def summarize(rows):
    n = len(rows)
    require(n >= 2, "sample uncertainty requires at least two independent families")
    pre = np.asarray([row["pre_old"] for row in rows], np.float64)
    require(pre.shape == (n, 2) and np.isfinite(pre).all() and np.all(np.abs(pre) <= 1), "both pre-acquisition rule replays retained")
    overlap = np.asarray([row["sparse_feedback_overlap"] for row in rows], np.float64)
    require(overlap.shape == (n,) and np.isfinite(overlap).all() and np.all((0 <= overlap) & (overlap <= 1))
        and np.all(overlap*8 == np.floor(overlap*8)), "realized sparse feedback overlap counts all eight blocks")
    values = {arm: {field: np.asarray([row["arms"][arm][field] for row in rows], np.float64) for field in FIELDS} for arm in ARMS}
    for arm, fields in values.items():
        require(all(value.shape == (n, 2, 2) and np.isfinite(value).all() for value in fields.values()), "all assigned family/rule/checkpoint outcomes")
        for field in ("first_food", "first_poison", "no_contact", "exposure_survival", "solo_survival"):
            require(np.isin(fields[field], (0, 1)).all(), "binary all-assigned endpoint " + field)
        require(np.all(np.abs(fields["local_S"]) <= 1) and np.array_equal(fields["solo_Y"], fields["first_food"] - fields["first_poison"])
            and np.all(fields["first_food"] + fields["first_poison"] + fields["no_contact"] == 1), "signed first-item score retains no-contact denominator")
        require(np.all(fields["exposure_survival"][:, :, 1] <= fields["exposure_survival"][:, :, 0])
            and np.all(fields["solo_survival"] <= fields["exposure_survival"]), "exposure and readout death cannot revive")
        dead = fields["exposure_survival"] == 0
        require(all(np.all(fields[name][dead] == 0) for name in ("local_S", "solo_Y", "first_food", "first_poison", "solo_survival"))
                and np.all(fields["no_contact"][dead] == 1), "dead assigned sources remain zero-scored")
    family = {arm: {field: value.mean(axis=1) for field, value in fields.items()} for arm, fields in values.items()}
    native, contact, sparse, reopened = (family[policy + "_reversal"] for policy in POLICIES)
    local_change = lambda arm: arm["local_S"][:, 1] - arm["local_S"][:, 0]
    solo_gain = lambda control: contact["solo_Y"][:, 1] - family[control + "_reversal"]["solo_Y"][:, 1]
    stationary_gain = lambda control, field: family["contact_stationary"][field][:, 1] - family[control + "_stationary"][field][:, 1]
    components = dict(pre_old=stat(pre.mean(axis=1)), gate_immediate=stat(contact["local_S"][:, 0]),
        native_immediate=stat(native["local_S"][:, 0]), gate_delayed=stat(contact["local_S"][:, 1]),
        gate_solo_delayed=stat(contact["solo_Y"][:, 1]), retention_advantage=stat(local_change(contact) - local_change(native)),
        gate_local_vs_sparse=stat(contact["local_S"][:, 1] - sparse["local_S"][:, 1]),
        delay_ablation=stat(contact["local_S"][:, 1] - reopened["local_S"][:, 1]),
        gate_solo_vs_native=stat(solo_gain("native")), gate_solo_vs_sparse=stat(solo_gain("sparse")),
        adjusted_solo_vs_native=stat(solo_gain("native") - stationary_gain("native", "solo_Y")),
        adjusted_solo_vs_sparse=stat(solo_gain("sparse") - stationary_gain("sparse", "solo_Y")),
        sparse_feedback_overlap=stat(overlap))
    safeguards = dict(acquisition_vs_native=shared.change(contact["local_S"][:, 0] - native["local_S"][:, 0]),
        gate_own_retention=shared.change(local_change(contact)),
        stationary_local_native=shared.change(stationary_gain("native", "local_S")),
        stationary_local_sparse=shared.change(stationary_gain("sparse", "local_S")),
        stationary_solo_native=shared.change(stationary_gain("native", "solo_Y")),
        stationary_solo_sparse=shared.change(stationary_gain("sparse", "solo_Y")))
    for history in HISTORIES:
        for control in ("native", "sparse"):
            safeguards["solo_survival_" + history + "_" + control] = shared.change(
                family["contact_" + history]["solo_survival"][:, 1] - family[control + "_" + history]["solo_survival"][:, 1])
    return dict(families=n, arms={arm: {field: [stat(value[:, point]) for point in (0, 1)] for field, value in fields.items()}
        for arm, fields in family.items()}, components=components, safeguards=safeguards, gate=decisions(components, safeguards))


def replicated_gates(panels):
    require(len(panels) == 3, "all three original lineages are required for replication")
    passed = []
    for index, panel in enumerate(panels):
        gate = decisions(panel["components"], panel["safeguards"])
        same(panel["gate"], gate, "every registered criterion within the same lineage")
        if gate["passed"]:
            passed.append(index)
    return dict(contact_gate=dict(count=len(passed), seeds=passed, passed=len(passed) == 3))


class Auditor(shared.Auditor):
    """Bounded immutable archives, source provenance and sealed final bundles."""
    RUN_PREFIX, MASTER, PROTOCOL = "E133", 2026121200, PROTOCOL
    MAX_SECONDS = {"development": 220, "final": 600}

    def completed_bundle(self, development=False):
        count, names = (4, ["E133-benchmark"]) if development else (64, [f"E133-A{i}" for i in range(3)])
        complete, pending = [], []
        for name in names:
            found = list(self.path("runs").glob("*_" + name))
            require(len(found) <= 1, "duplicate registered panel " + name)
            if not found or not (found[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = found[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            files = {path.name for path in (found[0] / "families").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_families") != count or status.get("expected_families") != count
                    or not (found[0] / "summary.json").is_file() or files != {f"family_{i:04d}.npz" for i in range(count)}):
                pending.append(name)
            else:
                complete.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return complete

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master, name = (4, self.MASTER, "E133-benchmark") if development else (64, self.MASTER + 1 + index, f"E133-A{index}")
        split = "development" if development else "final"
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={}, generation=0,
            seed_index=index, eval_seed=master, expected_families=count, source_generations=[200],
            statistical_unit="independent family averaging both initially assigned rule replays", arms=list(ARMS),
            key_roles=list(KEY_ROLES), initial_rules=[0, 1], apparatus=APPARATUS, assignment_tag=0xE133,
            raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", max_seconds=self.MAX_SECONDS[split],
            scientific_result=False), "registered single-creature matched-history apparatus")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=PROTOCOL,
            condition_split=split, allow_dirty=False, resume=None, cfg=None, no_plastic=False, quick=False,
            objective=None, memory_condition=None, max_train_seconds=None, pop=128, worlds=8, sigma=.1, lr=.1,
            checkpoint_every=10), "fresh fixed-source invocation")
        require(config["args"].get("controller") in (None, "rules-v0")
            and config["args"].get("structural_plasticity") in (None, "off") and not config["args"].get("persistence", False)
            and config.get("retry_of") is None and config.get("recovery_of") is None, "no unregistered mechanisms or replacement attempts")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
            and self.code_trees(commit) == self.expected_trees, "clean scientific source matches registration")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete scientific source manifest")
        context = type("SourceContext", (), {"root": self.root, "source": commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(SOURCE_PATHS))
        for path, digest in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        if self.code is None:
            self.code = config["source_code"]
        same(config["source_code"], self.code, "development and all final panels use the same scientific bytes")
        path, digest = base.ORIGINALS[index]
        subset(config["source_genome"], dict(path=path, sha256=digest, training_seed=index), "fixed original E103 descendant")
        content = self.read(path)
        same(hashlib.sha256(content).hexdigest(), digest, "pinned original genome")
        require(self.read(folder + "/genome_final.npy") == content, "copied genome remains identical")
        genome = np.load(io.BytesIO(content), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite encoding")
        same(config["args"]["start"], "file:" + path, "correct original lineage")
        same(config["assignments"]["path"], "assignments.npz", "complete assignment table")
        assignment = self.artifact(folder, config["assignments"])
        require(set(assignment) == {"family_keys", "initial_rules"}, "exact assigned family schema")
        exact(assignment["initial_rules"], np.asarray([0, 1], np.int32), "both original rule copies")
        jax, derive = random_helpers()
        same(config["runtime"]["jax"], jax.__version__, "same pinned random runtime")
        root = jax.random.fold_in(jax.random.PRNGKey(master), 0xE133)
        keys = np.stack([np.asarray(jax.random.split(jax.random.fold_in(root, i), 7)) for i in range(count)])
        exact(assignment["family_keys"], keys, "independently derived distinct role streams")
        self.register_keys(keys.reshape(-1, 2), "new family scripts, births, sparse schedules and held-out readouts")
        randomness = [{name: np.asarray(value) for name, value in derive(key).items()} for key in keys]
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "every assigned family is journaled once")
        for i, record in enumerate(records):
            subset(record, dict(family=i, keys=keys[i].tolist()), "ordered complete family journal")
            require(np.isfinite(record["family_seconds"]) and record["family_seconds"] >= 0
                and np.isfinite(record["elapsed_seconds"]) and 0 < record["elapsed_seconds"] <= self.MAX_SECONDS[split],
                "finite bounded timing records")
        return config, records, genome, keys, randomness

    def panel(self, folder, index, development, prepared):
        config, records, genome, keys, randomness = prepared
        count = config["expected_families"]
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, seed_index=index, evaluation_seed=config["eval_seed"],
            scientific_result=False, artifact_validity="passed", gate_status="not_applicable" if development else "undecided"),
            "complete immutable panel")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"], "complete panel within registered cap")
        for name in ("family_seconds", "family_wall_seconds"):
            timings = np.asarray(summary[name], np.float64)
            require(timings.shape == (count,) and np.isfinite(timings).all() and np.all(timings >= 0), "all timing observations retained")
        same(summary["family_seconds"], [record["family_seconds"] for record in records], "journal duration linkage")
        rows = []
        for i, record in enumerate(records):
            started = time.monotonic()
            manifest = summary["raw_files"][i]
            subset(manifest, dict(index=i, path=f"families/family_{i:04d}.npz"), "ordered original raw family")
            same(record["raw_file"], manifest, "journal and raw-file linkage")
            rows.append(audit_family(self.artifact(folder, manifest), i, genome, keys[i], randomness[i]))
            self.family_seconds.append(time.monotonic() - started)
        measured = summarize(rows)
        same(summary["measurements"], measured, "independent endpoints, paired family SE and all registered criteria")
        return dict(index=index, run=folder, measurements=measured)


def synthetic_rows():
    """Small abstract endpoints, deliberately separate from physical fixtures."""
    rows = []
    for index in range(4):
        arms = {}
        for arm in ARMS:
            policy, history = arm.split("_")
            immediate = .40 + .02 * index
            delayed = immediate if policy in ("contact", "sparse") else immediate-.10
            if policy == "sparse":
                delayed -= .10
            first_food = float(policy in ("contact", "reopened") or history == "stationary")
            values = dict(local_S=[immediate, delayed], solo_Y=[1., first_food], first_food=[1., first_food],
                first_poison=[0., 0.], no_contact=[0., 1-first_food], exposure_survival=[1., 1.], solo_survival=[1., 1.])
            arms[arm] = {name: np.broadcast_to(value, (2, 2)).copy() for name, value in values.items()}
        rows.append(dict(pre_old=np.asarray([.3 + .02 * index] * 2), sparse_feedback_overlap=.25, arms=arms))
    return rows


def self_check():
    observation, action = np.zeros(103, np.float32), np.eye(5, dtype=np.float32)[1]
    destination = int((MOVES[1, 0]+2)*5+MOVES[1, 1]+2)
    observation[destination] = 1
    require(bool(contact_mask(observation, action, np.asarray(True)))
        and not bool(contact_mask(observation, action, np.asarray(False)))
        and not bool(contact_mask(observation, np.zeros(5, np.float32), np.asarray(True))), "visible previous destination and actual living move")
    energy, delta, alive, food, poison = next_energy(np.float32(2), True, 1, 0)
    require(energy == 2 and delta == 0 and alive and food and not poison, "upper clipping can hide an actually observed food contact")
    energy, delta, alive, food, poison = next_energy(np.float32(.05), True, 2, 0)
    require(energy == 0 and delta == -np.float32(.05) and not alive and poison and not food,
            "fatal contact counts with clipped realized loss, then death is absorbing")
    require(next_energy(energy, alive, 1, 0) == (np.float32(0), np.float32(0), False, False, False),
            "a later apparent food does not revive a dead body")
    measured = summarize(synthetic_rows())
    require(measured["gate"]["passed"] and replicated_gates([measured] * 3)["contact_gate"]["passed"], "every registered gate in every lineage")
    components, safeguards = copy.deepcopy(measured["components"]), copy.deepcopy(measured["safeguards"])
    components["retention_advantage"] = dict(mean=.025, se=.0125, n=4)
    require(not decisions(components, safeguards)["criteria"]["retention_advantage"], "strict paired evidence at inclusive minimum")
    components["retention_advantage"]["se"] = 0.
    safeguards["gate_own_retention"]["lower_2se"] = -.05
    require(not decisions(components, safeguards)["criteria"]["gate_own_retention"], "strict retention safeguard")
    dead = {name: np.zeros((50, 2) if name == "grid_regrowth_cell" else (50,), np.int32
        if name in ("grid_action", "grid_item", "grid_regrowth_cell") else bool if name in
        ("grid_food", "grid_poison", "grid_alive_before", "grid_alive_after") else np.float32) for name in GRID_FIELDS}
    result = solo_trace(dead, np.zeros((16, 16), np.int32), np.asarray([8, 8], np.int32), np.zeros((50, 2), np.int32), 0, False)
    require(result["no_contact"] == 1 and result["solo_Y"] == result["solo_survival"] == 0, "dead assignments remain in the denominator")
    require("jax" not in sys.modules and not any(name == "seed" or name.startswith("seed.") for name in sys.modules), "synthetic self-check imports no simulation or PRNG")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=7, simulator_imported=False,
        coverage=["previous visible contact and executed action", "clipped-zero food feedback", "fatal clipped outcome and no revival", "all-three paired gates",
            "strict evidence and retention bounds", "dead all-assigned denominator", "independent NumPy-only checks"])


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
    started, auditor = time.monotonic(), None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            development = auditor.completed_bundle(True)[0]
            prepared_dev = auditor.metadata(development, 0, True)
            auditor.panel(development, 0, True, prepared_dev)
        prepared = [auditor.metadata(folder, index, args.development) for index, folder in enumerate(folders)]
        panels = [auditor.panel(folder, index, args.development, prepared[index]) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, replicated_gates=None if args.development else replicated_gates([panel["measurements"] for panel in panels]),
            limitations=["Physical forced histories and solo movement, resources, clipping, death, first-item outcomes and paired-family decisions are independently reconstructed. Dense neural dynamics are not replayed.",
                "Disabled-W identity is an in-kernel full-array bit comparison checked against saved endpoint hashes; without dense per-tick W arrays the auditor cannot independently recompute that predicate or exclude an unrecorded transient.",
                "Role assignments, script, sparse timing, solo world and resource-regrowth cells are independently regenerated with the pinned JAX PRNG; categorical action draws and developmental wiring are not rerun.",
                "The common solo readout uses ordinary updates. This tests acquired-state transfer, not live autonomous deployment of the contact gate. A retained preference is credited only after the registered acquisition and delayed-comparison checks."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), replicated_gates=None)
        code = 2
    except (base.AuditError, OSError, EOFError, KeyError, ValueError, TypeError, IndexError, AttributeError,
            zipfile.BadZipFile, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", replicated_gates=None)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        elapsed, reconstructed = time.monotonic() - started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed-reconstructed,
            reconstruction_seconds=reconstructed, family_seconds=auditor.family_seconds, completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
