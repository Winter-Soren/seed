"""Independent E132 geometry, voluntary feedback and all-birth paired audit.

Uses only recorded physical evidence and small established provenance helpers;
it does not replay neural dynamics or import the simulator.
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
CFG, MOVES, BODY, NEURAL = shared.CFG, shared.MOVES, shared.BODY, shared.STATE_FIELDS
PROTOCOL = "voluntary-offers-v1"
ARMS = ("near_stationary", "near_reversal", "sham_stationary", "sham_reversal")
KEY_ROLES = ("birth", "geometry", "exposure_action", "solo_world", "solo_action", "probe")
SOURCE_PATHS = {"seed/seed.py", "seed/exploration.py", "seed/direct.py", "seed/encounter_diagnostic.py",
    "seed/voluntary_offers.py", "scripts/run_voluntary_offers.py", "scripts/run_cold_donor.py",
    "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py", "scripts/run_shelter_evolution.py"}
APPARATUS = dict(acquisition_blocks=8, continuation_blocks=8, block_steps=8, split_tick=64,
    exposure_ticks=128, feedback_ticks=1, solo_steps=50, local_placements=12)
FIELDS = ("local_S", "both_colors", "solo_Y", "first_food", "first_poison", "no_contact", "exposure_survival",
          "readout_survival", "solo_survival", "intake_A", "intake_B", "realized_energy")
TRACE_FIELDS = ("grid_before", "grid", "position_before", "position", "energy_before", "energy", "unclipped_energy",
    "signal_before", "signal", "incoming_denergy", "denergy", "alive_before", "alive_after", "item", "food", "poison",
    "observation", "decision_motor", "logits", "probabilities", "sampled_action", "action", "action_key", "regrowth_key",
    "regrowth_cell", "key_before", "key_after", "efference", "t", "rule", "hebbian_enabled", "predictor_enabled",
    "skipped", "external_tick", "incoming_efference", "carried_denergy")
PROBE_FIELDS = ("observation", "decision_motor", "logits", "probabilities", "sampled_action", "action_key",
    "hebbian_enabled", "predictor_enabled", "feedback_applied", "initial_neural_matches", "final_frozen_matches", "performed")
STATE_FIELDS = tuple(name + "_sha256" for name in NEURAL) + BODY


def offered_grid(position, offsets):
    offsets = np.asarray(offsets)
    require(offsets.shape == (2, 2) and offsets.dtype == np.int32 and np.all((offsets >= 0) & (offsets < 16))
            and not np.any(np.all(offsets == 0, axis=1)) and not np.array_equal(offsets[0], offsets[1]),
            "two ordered distinct noncentral offered cells")
    cells = (position[0] + offsets) % 16
    result = np.zeros((16, 16), np.int32)
    result[cells[:, 0], cells[:, 1]] = (1, 2)
    return result


@lru_cache(maxsize=1)
def random_helpers():
    """Only PRNG operations, never an environment/controller import."""
    jax, jnp = source_bytes.random_runtime()

    @jax.jit
    def derive(keys):
        block_keys = jax.random.split(keys[1], 16)

        def geometry(key):
            near_key, sham_key = jax.random.split(key)
            near = jnp.asarray(MOVES[1:])[jax.random.permutation(near_key, 4)[:2]] % 16
            cells = jax.random.permutation(sham_key, 255)[:2] + 1
            return near, jnp.stack((cells // 16, cells % 16), axis=1)

        near, sham = jax.vmap(geometry)(block_keys)

        def stream(key, length):
            def tick(previous, _):
                next_key, action, regrowth = jax.random.split(previous, 3)
                cell = jax.random.randint(regrowth, (1, 2), 0, 16)
                return next_key, dict(before=previous, after=next_key, action=action, regrowth=regrowth, cell=cell)
            return jax.lax.scan(tick, key, None, length=length)[1]

        k1, k2 = jax.random.split(keys[3])
        cells = jax.random.choice(k1, 256, (48,), replace=False)
        grid = jnp.zeros(256, jnp.int32).at[cells[:24]].set(1).at[cells[24:]].set(2).reshape(16, 16)
        exposure, solo = stream(keys[2], 129), stream(keys[4], 50)
        probe = jax.random.split(keys[5], 12)
        return dict(geometry_block_keys=block_keys, geometry_near_offsets=near, geometry_sham_offsets=sham,
            probe_keys=probe, probe_action_keys=jax.vmap(lambda key: jax.random.split(key, 3)[1])(probe),
            solo_grid=grid, solo_position=jax.random.randint(k2, (1, 2), 0, 16),
            **{"exposure_" + name: value for name, value in exposure.items()},
            **{"solo_" + name: value for name, value in solo.items()})
    return jax, derive


def select(values, prefix, names):
    return {name: values[prefix + name] for name in names}


def same_state(actual, expected, label, *, neural=True):
    for name in STATE_FIELDS if neural else BODY:
        exact(actual[name], expected[name], label + " " + name)


def state_valid(state):
    shared.state_hashes(state)
    shapes = dict(grid=(16, 16), pos=(1, 2), energy=(1,), alive=(1,), sig=(1,), denergy=(1,), key=(2,), t=(), counts=(1, 2, 2))
    for name, shape in shapes.items():
        typed(state, name, shape, "iu" if name in ("grid", "pos", "key", "t") else "f")
    require(np.isin(state["grid"], (0, 1, 2)).all() and np.all((state["pos"] >= 0) & (state["pos"] < 16))
            and np.isin(state["alive"], (0, 1)).all() and np.all((state["energy"] >= 0) & (state["energy"] <= 2)), "valid physical saved carry")


def trace(trace, initial, final, rule, external_start, randomness, *, offsets=None, clear=False, regrowth=False):
    """Rebuild the actual sparse/full world; no dense neural state is guessed."""
    state_valid(initial)
    state_valid(final)
    steps = 64 if offsets is not None else (50 if regrowth else 1)
    shapes = dict(grid_before=(steps, 16, 16), grid=(steps, 16, 16), position_before=(steps, 1, 2), position=(steps, 1, 2),
        observation=(steps, 1, 103), decision_motor=(steps, 1, 5), logits=(steps, 1, 5), probabilities=(steps, 1, 5),
        sampled_action=(steps, 1), action=(steps, 1), action_key=(steps, 2), regrowth_key=(steps, 2),
        regrowth_cell=(steps, 1, 2), key_before=(steps, 2), key_after=(steps, 2), efference=(steps, 1, 5),
        incoming_efference=(steps, 1, 5), t=(steps,), rule=(steps,), skipped=(steps,), external_tick=(steps,),
        hebbian_enabled=(steps,), predictor_enabled=(steps,))
    integers = {"grid_before", "grid", "position_before", "position", "sampled_action", "action", "action_key", "regrowth_key",
                "regrowth_cell", "key_before", "key_after", "t", "rule", "external_tick"}
    for name, shape in shapes.items():
        typed(trace, name, shape, "iu" if name in integers else ("b" if name in ("skipped", "hebbian_enabled", "predictor_enabled") else "f"))
    for name in ("energy_before", "energy", "unclipped_energy", "signal_before", "signal", "incoming_denergy", "denergy",
                 "carried_denergy", "alive_before", "alive_after", "item", "food", "poison"):
        value = typed(trace, name, (steps, 1), "b" if name in ("food", "poison") else ("i" if name == "item" else "f"))
        require(value.dtype == (np.bool_ if name in ("food", "poison") else (np.int32 if name == "item" else np.float32)), "native event/feedback dtype")
    require(np.isin(trace["alive_before"], (0, 1)).all() and np.isin(trace["alive_after"], (0, 1)).all()
            and np.all(np.abs(trace["signal"]) <= 1), "binary liveness and bounded public signal")
    exact(trace["external_tick"], np.arange(external_start, external_start + steps, dtype=np.int32), "all scheduled slots retained")
    require(np.all(trace["rule"] == rule), "current actual rule has not leaked or shifted")
    exact(trace["action"], trace["sampled_action"], "the apparatus never replaces the chosen action")
    exact(trace["incoming_efference"][1:], trace["efference"][:-1], "next neural tick receives the actual action")
    alive = trace["alive_before"][:, 0] > 0
    exact(trace["skipped"], ~alive, "only actual dead assignments are skipped")
    for name in ("hebbian_enabled", "predictor_enabled"):
        exact(trace[name], alive, "ordinary learning is active precisely on living turns")
    require(np.all(np.abs(trace["decision_motor"][alive]) <= 1), "ordinary bounded motor activity")
    encounter.softmax(trace["logits"][alive], trace["probabilities"][alive])
    exact(trace["logits"][alive], np.float32(3) * trace["decision_motor"][alive], "unmasked native motor logits")
    body = {name: np.asarray(initial[name]).copy() for name in BODY}
    if clear and body["alive"][0] > 0:
        body["grid"][:] = 0
    for slot in range(steps):
        live = bool(body["alive"][0])
        if offsets is not None and slot % 8 == 0 and live:
            body["grid"] = offered_grid(body["pos"], offsets[slot // 8])
        for field, name in (("grid_before", "grid"), ("position_before", "pos"), ("energy_before", "energy"),
                            ("signal_before", "sig"), ("incoming_denergy", "denergy"), ("alive_before", "alive"),
                            ("key_before", "key"), ("t", "t")):
            exact(trace[field][slot], body[name], "actual incoming body/key " + field)
        action = int(trace["action"][slot, 0])
        if live:
            require(0 <= action < 5, "every living action remains voluntary and five-way")
            clock = int(body["t"])
            for field, name in (("key_before", "before"), ("key_after", "after"), ("action_key", "action"),
                                ("regrowth_key", "regrowth"), ("regrowth_cell", "cell")):
                exact(trace[field][slot], randomness[name][clock], "independent native action/regrowth stream")
            exact(trace["observation"][slot], encounter.observation(body["grid"], body["pos"][0], body["energy"][0], body["denergy"][0])[None],
                  "only visible offered resources and realized feedback")
            position = (body["pos"] + MOVES[action]) % 16
            item = int(body["grid"][tuple(position[0])])
            food, poison = item == 1 + rule, item == 2 - rule
            require(int(trace["item"][slot, 0]) == item and bool(trace["food"][slot, 0]) == food
                    and bool(trace["poison"][slot, 0]) == poison, "actual item identity and current-rule consumption")
            raw = body["energy"].astype(np.float64) - .003 + .25 * food - .25 * poison
            require(np.all(np.abs(trace["unclipped_energy"][slot] - raw) <= 8 * np.finfo(np.float32).eps), "native energy equation before clipping")
            energy = np.clip(trace["unclipped_energy"][slot], 0, 2)
            exact(trace["energy"][slot], energy, "real clipped energy")
            exact(trace["denergy"][slot], energy - body["energy"], "only the actual realized change is returned")
            exact(trace["carried_denergy"][slot], trace["denergy"][slot], "current feedback becomes the next pending outcome")
            exact(trace["efference"][slot], np.eye(5, dtype=np.float32)[[action]], "the brain receives the executed action")
            if item:
                body["grid"][tuple(position[0])] = 0
                if regrowth:
                    body["grid"][tuple(trace["regrowth_cell"][slot, 0])] = item
            body.update(pos=position, energy=energy, alive=(energy > 0).astype(np.float32), sig=trace["signal"][slot],
                        denergy=trace["denergy"][slot], key=trace["key_after"][slot], t=np.asarray(clock + 1, np.int32))
            body["counts"][0, int(clock >= 150)] += np.asarray([food, poison], np.float32)
        else:
            require(action == -1 and not trace["food"][slot].any() and not trace["poison"][slot].any()
                    and not trace["denergy"][slot].any() and not trace["item"][slot].any(), "dead slots cannot act, earn feedback or consume")
            for name in ("observation", "decision_motor", "logits", "probabilities", "action_key", "regrowth_key", "regrowth_cell"):
                require(not trace[name][slot].any(), "skipped dead slot has no fabricated neural decision or draw")
            exact(trace["efference"][slot], trace["incoming_efference"][slot], "frozen dead motor activity")
            exact(trace["unclipped_energy"][slot], body["energy"], "dead body does not decay further")
            exact(trace["carried_denergy"][slot], body["denergy"], "dead body retains its last pending feedback without replay")
        for field, name in (("grid", "grid"), ("position", "pos"), ("energy", "energy"), ("alive_after", "alive"),
                            ("signal", "sig"), ("key_after", "key")):
            exact(trace[field][slot], body[name], "actual outgoing state " + field)
    same_state(final, body, "physical endpoint and full encounter counts", neural=False)
    exact(final["mask_sha256"], initial["mask_sha256"], "apparatus never changes topology")
    if not initial["alive"][0]:
        same_state(final, initial, "a dead carry remains completely frozen")
    return body


def local_probe(probe, source, placements, randomness, rule):
    performed = bool(typed(probe, "performed", (), "b"))
    require(performed == bool(source["alive"][0]), "readout eligibility follows the true post-feedback state")
    for name in NEURAL:
        exact(probe["source_" + name + "_sha256"], source[name + "_sha256"], "local probe uses the unprobed source brain")
    require(typed(probe, "initial_neural_matches", (12, 4), "b").all()
            and typed(probe, "final_frozen_matches", (12, 3), "b").all(), "actual clone and frozen-neural identity predicates")
    for name in ("hebbian_enabled", "predictor_enabled", "feedback_applied"):
        require(not typed(probe, name, (12,), "b").any(), "local readout is unreinforced and discarded")
    probabilities = typed(probe, "probabilities", (12, 5), "f")
    logits = typed(probe, "logits", (12, 5), "f")
    typed(probe, "decision_motor", (12, 5), "f")
    typed(probe, "observation", (12, 1, 103), "f")
    typed(probe, "sampled_action", (12,), "i")
    typed(probe, "action_key", (12, 2), "u")
    for name in BODY:
        typed(probe, "initial_" + name, (12,) + np.shape(source[name]), "iu" if name in ("grid", "pos", "key", "t") else "f")
    if not performed:
        for name in BODY:
            exact(probe["initial_" + name], np.broadcast_to(source[name], (12,) + np.shape(source[name])), "dead readout does not reset or revive its body")
        for name in ("probabilities", "logits", "decision_motor", "observation", "action_key"):
            require(not probe[name].any(), "unexecuted dead probe cannot supply a choice")
        require(np.all(probe["sampled_action"] == -1), "dead readout has no sampled action")
        return 0.
    encounter.softmax(logits, probabilities)
    motor = typed(probe, "decision_motor", (12, 5), "f")
    require(np.all(np.abs(motor) <= 1), "bounded ordinary readout motors")
    exact(logits, np.float32(3) * motor, "unchanged five-way decision logits")
    require(np.isin(typed(probe, "sampled_action", (12,), "iu"), np.arange(5)).all(), "discarded choice sample remains five-way")
    exact(probe["action_key"], randomness["probe_action_keys"], "independent paired local action stream")
    exact(probe["initial_key"], randomness["probe_keys"], "all twelve assigned local copies")
    for name, value in (("energy", 1), ("alive", 1), ("sig", 0), ("denergy", 0), ("t", 0), ("counts", 0)):
        require(np.all(probe["initial_" + name] == value), "standardized living probe body " + name)
    exact(probe["initial_pos"], np.full((12, 1, 2), 8, np.int32), "local readout at the neutral hub")
    for orientation, placement in enumerate(placements):
        scene = encounter.pair_grid(placement)
        exact(probe["initial_grid"][orientation], scene, "both visible alternatives in each ordered placement")
        exact(probe["observation"][orientation], encounter.observation(scene, np.asarray([8, 8]), 1, 0)[None],
              "local readout has no hidden-rule observation")
    values = probabilities.astype(np.float64)
    return float((values[np.arange(12), placements[:, rule]] - values[np.arange(12), placements[:, 1-rule]]).mean())


def audit_family(raw, index, genome, keys, randomness):
    probe_names = PROBE_FIELDS + tuple("source_" + name + "_sha256" for name in NEURAL) + tuple("initial_" + name for name in BODY)
    common_fields = {prefix + name for prefix in ("birth_", "split_", "feedback_") for name in STATE_FIELDS}
    common_fields |= {prefix + name for prefix in ("exposure_", "flush_") for name in TRACE_FIELDS}
    common_fields |= {"probe_" + name for name in probe_names}
    arm_fields = {prefix + name for prefix in ("split_", "exposure_end_", "feedback_", "solo_initial_", "solo_final_") for name in STATE_FIELDS}
    arm_fields |= {prefix + name for prefix in ("exposure_", "flush_", "solo_") for name in TRACE_FIELDS}
    arm_fields |= {"probe_" + name for name in probe_names} | {"rule", "solo_performed"}
    top = {"base_index", "keys", "initial_rules", "genome_sha256", "probe_placements", "block_steps", "pre_blocks", "post_blocks", "grid_steps",
           "geometry_block_keys", "geometry_near_offsets", "geometry_sham_offsets"}
    require(set(raw) == top | {"common_" + name for name in common_fields} | {arm + "_" + name for arm in ARMS for name in arm_fields},
            "complete exact sparse-offer evidence schema")
    exact(raw["base_index"], np.asarray(index, np.int32), "every assigned family in order")
    exact(raw["keys"], keys, "independently assigned role streams")
    exact(raw["initial_rules"], np.asarray([0, 1], np.int32), "both original rule copies retained")
    exact(raw["genome_sha256"], fingerprint(genome), "fixed original descendant genome")
    for name, value in (("block_steps", 8), ("pre_blocks", 8), ("post_blocks", 8), ("grid_steps", 50)):
        exact(raw[name], np.asarray(value, np.int32), "registered apparatus duration")
    placements = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
    exact(raw["probe_placements"], placements, "all twelve ordered local placements")
    for name in ("geometry_block_keys", "geometry_near_offsets", "geometry_sham_offsets"):
        exact(raw[name], randomness[name], "rule-blind random ordered A/B geometry")
    for kind in ("near", "sham"):
        offsets = raw["geometry_" + kind + "_offsets"]
        require(offsets.shape == (16, 2, 2), "all assigned acquisition and continuation blocks")
        for pair in offsets:
            offered_grid(np.asarray([[8, 8]], np.int32), pair)
        if kind == "near":
            require(all(tuple(offset) in set(map(tuple, MOVES[1:] % 16)) for offset in offsets.reshape(-1, 2)), "near offers are cardinal neighbors")
    stream = {name: randomness["exposure_" + name] for name in ("before", "after", "action", "regrowth", "cell")}
    solo_stream = {name: randomness["solo_" + name] for name in stream}
    result = dict(pre_old=[], arms={arm: {field: [] for field in FIELDS} for arm in ARMS})
    births = []
    for rule in (0, 1):
        common = {name: raw["common_" + name][rule] for name in common_fields}
        birth, split, feedback = (select(common, prefix + "_", STATE_FIELDS) for prefix in ("birth", "split", "feedback"))
        state_valid(birth)
        exact(birth["grid"], np.zeros((16, 16), np.int32), "fresh empty acquisition apparatus")
        exact(birth["pos"], np.full((1, 2), 8, np.int32), "fresh central newborn")
        exact(birth["key"], keys[2], "assigned native exposure action stream")
        exact(birth["x_sha256"], fingerprint(np.zeros((1, 157), np.float32)), "unconditioned newborn activity")
        exact(birth["Wp_sha256"], fingerprint(np.zeros((1, 103, 48), np.float32)), "untrained newborn predictor")
        for name, value in (("energy", 1), ("alive", 1), ("sig", 0), ("denergy", 0), ("t", 0), ("counts", 0)):
            require(np.all(birth[name] == value), "native unassisted newborn body")
        births.append(birth)
        pre, flush = select(common, "exposure_", TRACE_FIELDS), select(common, "flush_", TRACE_FIELDS)
        exact(pre["incoming_efference"][0], np.zeros((1, 5), np.float32), "newborn has no previous action")
        trace(pre, birth, split, rule, 0, stream, offsets=raw["geometry_near_offsets"][:8])
        exact(flush["incoming_efference"][0], pre["efference"][-1], "discarded pre-readout processes the final real action")
        trace(flush, split, feedback, rule, 64, stream, clear=True)
        result["pre_old"].append(local_probe(select(common, "probe_", probe_names), feedback, placements, randomness, rule))
        for arm in ARMS:
            data = {name: raw[arm + "_" + name][rule] for name in arm_fields}
            final_rule = rule if arm.endswith("stationary") else 1-rule
            exact(data["rule"], np.asarray(final_rule, np.int32), "actual continuation and solo food rule")
            start, end, post, solo_initial, solo_final = (select(data, prefix + "_", STATE_FIELDS)
                for prefix in ("split", "exposure_end", "feedback", "solo_initial", "solo_final"))
            same_state(start, split, "continuation clones unflushed state, never the discarded pre-readout")
            exposure, flush, solo = (select(data, prefix + "_", TRACE_FIELDS) for prefix in ("exposure", "flush", "solo"))
            exact(exposure["incoming_efference"][0], pre["efference"][-1], "first continuation processes the actual last acquisition move")
            offsets = raw["geometry_" + arm.split("_")[0] + "_offsets"][8:]
            trace(exposure, start, end, final_rule, 64, stream, offsets=offsets)
            exact(flush["incoming_efference"][0], exposure["efference"][-1], "terminal flush consumes the true final exposure feedback")
            trace(flush, end, post, final_rule, 128, stream, clear=True)
            local = local_probe(select(data, "probe_", probe_names), post, placements, randomness, final_rule)
            performed = bool(typed(data, "solo_performed", (), "b"))
            require(performed == bool(post["alive"][0]), "solo readout cannot revive an exposure or feedback death")
            for name in NEURAL:
                exact(solo_initial[name + "_sha256"], post[name + "_sha256"], "solo and discarded local probe start from the same unprobed brain")
            if performed:
                exact(solo_initial["grid"], randomness["solo_grid"], "fresh paired full-density solo layout")
                exact(solo_initial["pos"], randomness["solo_position"], "fresh paired solo placement")
                exact(solo_initial["key"], keys[4], "independent paired solo action stream")
                for name, value in (("energy", 1), ("alive", 1), ("sig", 0), ("denergy", 0), ("t", 0), ("counts", 0)):
                    require(np.all(solo_initial[name] == value), "only a living readout receives the standardized body")
            else:
                same_state(solo_initial, post, "dead solo placeholders retain the full frozen carry")
            exact(solo["incoming_efference"][0], flush["efference"][-1], "unprobed acquired activity survives readout branching")
            trace(solo, solo_initial, solo_final, final_rule, 0, solo_stream, regrowth=True)
            outcomes = encounter.grid_outcomes(solo["food"][:, 0], solo["poison"][:, 0], solo["alive_after"][:, 0])
            contacted = exposure["food"] | exposure["poison"]
            a, b = (int((contacted & (exposure["item"] == color)).sum()) for color in (1, 2))
            values = dict(local_S=local, both_colors=float(a > 0 and b > 0), solo_Y=outcomes["food_minus_poison"],
                first_food=outcomes["first_food"], first_poison=outcomes["first_poison"], no_contact=outcomes["no_contact"],
                exposure_survival=float(end["alive"][0]), readout_survival=float(post["alive"][0]), solo_survival=float(solo_final["alive"][0]),
                intake_A=float(a), intake_B=float(b), realized_energy=float(exposure["denergy"].astype(np.float64).sum()))
            for field, value in values.items():
                result["arms"][arm][field].append(value)
    same_state(births[0], births[1], "birth is independent of the hidden initial food rule")
    return dict(pre_old=np.asarray(result["pre_old"], np.float64),
        arms={arm: {name: np.asarray(value, np.float64) for name, value in fields.items()} for arm, fields in result["arms"].items()})


def decisions(components, safeguards):
    def evidence(name, floor=0.):
        value = components[name]
        return value["se"] is not None and value["mean"] >= floor and value["mean"] > 2 * value["se"]
    def retained(name):
        value = safeguards[name]
        return value["lower_2se"] is not None and value["lower_2se"] > -.05
    criteria = dict(pre_old_positive=evidence("pre_old"), near_reversal_local_positive=evidence("near_reversal_local"),
        both_colors_improve=evidence("both_colors_gain", .10), near_reversal_solo_positive=evidence("near_reversal_solo"),
        solo_gain=evidence("solo_gain", .05), adjusted_solo_gain=evidence("adjusted_solo_gain"),
        stationary_solo_noninferiority=retained("stationary_solo"),
        reversal_survival_noninferiority=retained("reversal_exposure_survival"),
        stationary_survival_noninferiority=retained("stationary_exposure_survival"))
    return dict(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, passed in criteria.items() if not passed])


def measure(values, old, rule=None):
    def average(value):
        return value.mean(axis=1) if rule is None else value[:, rule]
    means = {arm: {field: average(value) for field, value in fields.items()} for arm, fields in values.items()}
    near, sham = means["near_reversal"], means["sham_reversal"]
    stationary = means["near_stationary"]["solo_Y"] - means["sham_stationary"]["solo_Y"]
    gain = near["solo_Y"] - sham["solo_Y"]
    components = dict(pre_old=stat(average(old)), near_reversal_local=stat(near["local_S"]),
        both_colors_gain=stat(near["both_colors"] - sham["both_colors"]), near_reversal_solo=stat(near["solo_Y"]),
        solo_gain=stat(gain), adjusted_solo_gain=stat(gain - stationary))
    safeguards = {name: shared.change(value) for name, value in dict(stationary_solo=stationary,
        reversal_exposure_survival=near["exposure_survival"] - sham["exposure_survival"],
        stationary_exposure_survival=means["near_stationary"]["exposure_survival"] - means["sham_stationary"]["exposure_survival"]).items()}
    return dict(pre_old_local=stat(average(old)), arms={arm: {field: stat(value) for field, value in fields.items()} for arm, fields in means.items()},
                components=components, safeguards=safeguards)


def summarize(rows):
    require(len(rows) >= 2, "family standard error requires at least two independent assigned families")
    old = np.stack([row["pre_old"] for row in rows]).astype(np.float64)
    require(old.shape == (len(rows), 2) and np.isfinite(old).all() and np.all(np.abs(old) <= 1), "all pre-split assigned rule copies")
    values = {arm: {field: np.stack([row["arms"][arm][field] for row in rows]).astype(np.float64) for field in FIELDS} for arm in ARMS}
    for arm, fields in values.items():
        require(all(value.shape == old.shape and np.isfinite(value).all() for value in fields.values()), "complete all-birth paired endpoints")
        for field in ("both_colors", "first_food", "first_poison", "no_contact", "exposure_survival", "readout_survival", "solo_survival"):
            require(np.isin(fields[field], (0, 1)).all(), "binary all-assigned endpoint " + field)
        require(np.all(fields["first_food"] + fields["first_poison"] + fields["no_contact"] == 1)
                and np.array_equal(fields["solo_Y"], fields["first_food"] - fields["first_poison"]), "first-contact signed score and full denominator")
        require(np.all(fields["readout_survival"] <= fields["exposure_survival"])
                and np.all(fields["solo_survival"] <= fields["readout_survival"]), "no readout resurrects an exposure death")
        dead = fields["readout_survival"] == 0
        require(np.all(fields["local_S"][dead] == 0) and np.all(fields["solo_Y"][dead] == 0)
                and np.all(fields["no_contact"][dead] == 1), "dead assignments remain zero-scored, not omitted")
        require(np.all(np.abs(fields["local_S"]) <= 1), "bounded signed local preference")
        for name in ("intake_A", "intake_B"):
            require(np.all((fields[name] >= 0) & (fields[name] <= 8)) and np.all(fields[name] == np.floor(fields[name])),
                    "one offered item per color in each of eight blocks")
        require(np.array_equal(fields["both_colors"], (fields["intake_A"] > 0) & (fields["intake_B"] > 0)), "actual intake defines the access manipulation")
    pooled = measure(values, old)
    return dict(families=len(rows), initial_rule_replays=2 * len(rows), continuation_lives=8 * len(rows),
        agents_per_rule=1, apparatus=APPARATUS, **pooled,
        gate=decisions(pooled["components"], pooled["safeguards"]),
        by_initial_rule={str(rule): measure(values, old, rule) for rule in (0, 1)})


def replicated_gates(panels):
    require(len(panels) == 3, "the access diagnosis requires all three original lineages")
    passed = []
    for index, panel in enumerate(panels):
        gate = decisions(panel["components"], panel["safeguards"])
        same(panel["gate"], gate, "all conditions within the same lineage")
        if gate["passed"]:
            passed.append(index)
    return dict(voluntary_access=dict(count=len(passed), seeds=passed, passed=len(passed) == 3))


class Auditor(shared.Auditor):
    """Reuse bounded archives, path containment and exact evidence hashing."""
    RUN_PREFIX, MASTER, PROTOCOL = "E132", 2026121100, PROTOCOL
    MAX_SECONDS = {"development": 180, "final": 500}

    def completed_bundle(self, development=False):
        count, names = (4, ["E132-benchmark"]) if development else (64, [f"E132-A{i}" for i in range(3)])
        complete, pending = [], []
        for name in names:
            paths = list(self.path("runs").glob("*_" + name))
            require(len(paths) <= 1, "duplicate registered panel " + name)
            if not paths or not (paths[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = paths[0].relative_to(self.root).as_posix()
            state = self.json(folder + "/status.json")
            files = {path.name for path in (paths[0] / "families").glob("*.npz")}
            if (state.get("status") != "finished" or state.get("completed_families") != count or state.get("expected_families") != count
                    or not (paths[0] / "summary.json").is_file() or files != {f"family_{i:04d}.npz" for i in range(count)}):
                pending.append(name)
            else:
                complete.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return complete

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master, name = (4, self.MASTER, "E132-benchmark") if development else (64, self.MASTER + 1 + index, f"E132-A{index}")
        split = "development" if development else "final"
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={}, generation=0,
            seed_index=index, eval_seed=master, expected_families=count, source_generations=[200],
            statistical_unit="independent family averaging both initially assigned rule replays", arms=list(ARMS),
            key_roles=list(KEY_ROLES), initial_rules=[0, 1], apparatus=APPARATUS, assignment_tag=0xE132,
            raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz", max_seconds=self.MAX_SECONDS[split],
            scientific_result=False), "registered one-creature apparatus and paired assignments")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=PROTOCOL,
            condition_split=split, allow_dirty=False, resume=None, cfg=None, no_plastic=False, quick=False,
            objective=None, memory_condition=None, max_train_seconds=None, pop=128, worlds=8, sigma=.1, lr=.1,
            checkpoint_every=10), "fresh fixed-source invocation")
        require(config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off") and not config["args"].get("persistence", False)
                and config.get("retry_of") is None and config.get("recovery_of") is None, "no unregistered mechanisms or restarts")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
                and self.code_trees(commit) == self.expected_trees, "actual clean scientific source matches registration")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete scientific dependency manifest")
        context = type("SourceContext", (), {"root": self.root, "source": commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(SOURCE_PATHS))
        for path, digest in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        if self.code is None:
            self.code = config["source_code"]
        same(config["source_code"], self.code, "all panels and development share identical scientific bytes")
        path, digest = base.ORIGINALS[index]
        subset(config["source_genome"], dict(path=path, sha256=digest, training_seed=index), "original fixed E103 descendant")
        content = self.read(path)
        same(hashlib.sha256(content).hexdigest(), digest, "pinned original genome")
        require(self.read(folder + "/genome_final.npy") == content, "source copy remains identical")
        genome = np.load(io.BytesIO(content), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite genome format")
        same(config["args"]["start"], "file:" + path, "correct original lineage")
        same(config["assignments"]["path"], "assignments.npz", "complete assignment manifest")
        assignment = self.artifact(folder, config["assignments"])
        require(set(assignment) == {"family_keys", "initial_rules"}, "exact assigned family table")
        exact(assignment["initial_rules"], np.asarray([0, 1], np.int32), "both rule copies")
        jax, derive = random_helpers()
        same(config["runtime"]["jax"], jax.__version__, "same pinned random runtime")
        root = jax.random.fold_in(jax.random.PRNGKey(master), 0xE132)
        keys = np.stack([np.asarray(jax.random.split(jax.random.fold_in(root, i), 6)) for i in range(count)])
        exact(assignment["family_keys"], keys, "new independently derived role streams")
        self.register_keys(keys.reshape(-1, 2), "new family worlds, birth, geometry, actions and readouts")
        randomness = [{name: np.asarray(value) for name, value in derive(key).items()} for key in keys]
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "every assigned family appears once")
        for i, record in enumerate(records):
            subset(record, dict(family=i, keys=keys[i].tolist()), "ordered family journal")
            require(np.isfinite(record["family_seconds"]) and record["family_seconds"] >= 0
                    and np.isfinite(record["elapsed_seconds"]) and 0 < record["elapsed_seconds"] <= self.MAX_SECONDS[split],
                    "bounded finite timing records")
        return config, records, genome, keys, randomness

    def panel(self, folder, index, development, prepared):
        config, records, genome, keys, randomness = prepared
        count = config["expected_families"]
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, seed_index=index, evaluation_seed=config["eval_seed"],
            scientific_result=False, artifact_validity="passed", gate_status="not_applicable" if development else "undecided"), "complete immutable panel")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"], "complete panel within cap")
        for field in ("family_seconds", "family_wall_seconds"):
            values = np.asarray(summary[field], np.float64)
            require(values.shape == (count,) and np.isfinite(values).all() and np.all(values >= 0), "all timing observations retained")
        same(summary["family_seconds"], [record["family_seconds"] for record in records], "journal timing linkage")
        rows = []
        for i, record in enumerate(records):
            started = time.monotonic()
            manifest = summary["raw_files"][i]
            subset(manifest, dict(index=i, path=f"families/family_{i:04d}.npz"), "ordered raw family")
            same(record["raw_file"], manifest, "exact journal and saved artifact linkage")
            rows.append(audit_family(self.artifact(folder, manifest), i, genome, keys[i], randomness[i]))
            self.family_seconds.append(time.monotonic() - started)
        measured = summarize(rows)
        same(summary["measurements"], measured, "independent all-birth endpoints, family errors and registered gates")
        return dict(index=index, run=folder, measurements=measured)


def synthetic_trace(*, energy=1., rule=0):
    """Hand-written eight-tick walk visits each offered color once per block."""
    state = dict(grid=np.zeros((16, 16), np.int32), pos=np.asarray([[8, 8]], np.int32),
        energy=np.asarray([energy], np.float32), alive=np.asarray([1.], np.float32), sig=np.zeros(1, np.float32),
        denergy=np.zeros(1, np.float32), key=np.asarray([1, 2], np.uint32), t=np.asarray(0, np.int32), counts=np.zeros((1, 2, 2), np.float32))
    state.update({name + "_sha256": np.zeros(32, np.uint8) for name in NEURAL})
    initial = copy.deepcopy(state)
    offsets = np.tile(np.asarray([[[15, 0], [0, 15]]], np.int32), (8, 1, 1))
    sequence = np.arange(1, 131, dtype=np.uint32).reshape(65, 2)
    randomness = dict(before=sequence[:-1], after=sequence[1:], action=sequence[:-1] + 200,
                      regrowth=sequence[:-1] + 400, cell=np.zeros((64, 1, 2), np.int32))
    records, motor = [], np.zeros((1, 5), np.float32)
    for slot in range(64):
        live = bool(state["alive"][0])
        if live and slot % 8 == 0:
            state["grid"] = offered_grid(state["pos"], offsets[slot // 8])
        before = copy.deepcopy(state)
        incoming = motor.copy()
        clock = int(state["t"])
        if live:
            action = (1, 0, 2, 3, 0, 4, 0, 0)[slot % 8]
            obs = encounter.observation(state["grid"], state["pos"][0], state["energy"][0], state["denergy"][0])[None]
            state["pos"] = (state["pos"] + MOVES[action]) % 16
            item = int(state["grid"][tuple(state["pos"][0])])
            food, poison = item == 1+rule, item == 2-rule
            unclipped = state["energy"] - np.float32(.003) + np.float32(.25) * food - np.float32(.25) * poison
            state["energy"] = np.clip(unclipped, 0, 2)
            delta = state["energy"] - before["energy"]
            state.update(denergy=delta, alive=(state["energy"] > 0).astype(np.float32), key=randomness["after"][clock], t=np.asarray(clock+1, np.int32))
            if item:
                state["grid"][tuple(state["pos"][0])] = 0
            state["counts"][0, 0] += np.asarray([food, poison], np.float32)
            motor = np.eye(5, dtype=np.float32)[[action]]
            ka, kr, cell = (randomness[name][clock] for name in ("action", "regrowth", "cell"))
            probabilities = np.full((1, 5), .2, np.float32)
        else:
            action, item, food, poison = -1, 0, False, False
            unclipped, delta, obs = state["energy"], np.zeros(1, np.float32), np.zeros((1, 103), np.float32)
            ka, kr, cell = np.zeros(2, np.uint32), np.zeros(2, np.uint32), np.zeros((1, 2), np.int32)
            probabilities = np.zeros((1, 5), np.float32)
        records.append(dict(grid_before=before["grid"], grid=state["grid"].copy(), position_before=before["pos"], position=state["pos"].copy(),
            energy_before=before["energy"], energy=state["energy"].copy(), unclipped_energy=unclipped.copy(), signal_before=before["sig"],
            signal=state["sig"].copy(), incoming_denergy=before["denergy"], denergy=delta.copy(), carried_denergy=state["denergy"].copy(),
            alive_before=before["alive"], alive_after=state["alive"].copy(), food=np.asarray([food]), poison=np.asarray([poison]),
            item=np.asarray([item], np.int32), action=np.asarray([action], np.int32), sampled_action=np.asarray([action], np.int32),
            observation=obs, decision_motor=np.zeros((1, 5), np.float32), logits=np.zeros((1, 5), np.float32), probabilities=probabilities,
            action_key=ka.copy(), regrowth_key=kr.copy(), regrowth_cell=cell.copy(), key_before=before["key"], key_after=state["key"].copy(),
            efference=motor.copy(), incoming_efference=incoming, t=np.asarray(clock, np.int32), rule=np.asarray(rule, np.int32),
            external_tick=np.asarray(slot, np.int32), skipped=np.asarray(not live), hebbian_enabled=np.asarray(live), predictor_enabled=np.asarray(live)))
    return {name: np.stack([record[name] for record in records]) for name in TRACE_FIELDS}, initial, state, offsets, randomness


def synthetic_rows():
    rows = []
    for preference in (.2, .4, .6, .8):
        arms = {}
        for arm in ARMS:
            safe = arm != "sham_reversal"
            a, b = 1., float(arm.startswith("near_"))
            values = dict(local_S=preference, both_colors=b, solo_Y=float(safe), first_food=float(safe),
                first_poison=0., no_contact=float(not safe), exposure_survival=1., readout_survival=1., solo_survival=1.,
                intake_A=a, intake_B=b, realized_energy=-.192)
            arms[arm] = {name: np.full(2, value, np.float64) for name, value in values.items()}
        rows.append(dict(pre_old=np.full(2, preference), arms=arms))
    return rows


def self_check():
    values, initial, final, offsets, randomness = synthetic_trace()
    trace(values, initial, final, 0, 0, randomness, offsets=offsets)
    require(int(values["food"].sum()) == int(values["poison"].sum()) == 8, "one consumption per offered color per block")
    altered = copy.deepcopy(values)
    altered["grid_before"][1, 7, 8] = 1
    try:
        trace(altered, initial, final, 0, 0, randomness, offsets=offsets)
    except base.AuditError:
        pass
    else:
        raise base.AuditError("a consumed item silently regrew inside its block")
    values, initial, final, offsets, randomness = synthetic_trace(energy=.05, rule=1)
    trace(values, initial, final, 1, 0, randomness, offsets=offsets)
    require(values["skipped"][1:].all() and int(final["t"]) == 1 and float(final["denergy"][0]) < 0,
            "fatal intake counts and later slots preserve the frozen carry")
    measured = summarize(synthetic_rows())
    require(measured["gate"]["passed"] and replicated_gates([measured] * 3)["voluntary_access"]["passed"], "all registered conditions across three lineages")
    comparisons = copy.deepcopy(measured["components"])
    comparisons["solo_gain"] = dict(mean=.05, se=.025, n=4)
    require(not decisions(comparisons, measured["safeguards"])["passed"], "strict evidence at the inclusive effect-size floor")
    require("jax" not in sys.modules and not any(name.startswith("seed.") for name in sys.modules), "synthetic checks import no simulator or random runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=5, simulator_imported=False,
        coverage=["fixed color conservation and voluntary motion", "within-block regrowth corruption", "fatal intake and fully frozen death",
                  "paired all-birth gates and all-three replication", "strict evidence boundary"])


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
            auditor.metadata(auditor.completed_bundle(True)[0], 0, True)
        prepared = [auditor.metadata(folder, i, args.development) for i, folder in enumerate(folders)]
        panels = [auditor.panel(folder, i, args.development, prepared[i]) for i, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, replicated_gates=None if args.development else replicated_gates([panel["measurements"] for panel in panels]),
            limitations=["Geometry, resources, movement, energy, death, first-contact scores and paired-family decisions are independently reconstructed. Dense neural dynamics are not replayed.",
                "Saved source fingerprints and in-kernel identity predicates check clone/readout consistency; without dense matrices this audit cannot independently recompute those identity predicates or exclude an unrecorded neural transient.",
                "Actual and sampled actions must match, their logits/probabilities are checked, and all assigned random keys are regenerated. The categorical action draw itself is not replayed.",
                "Near access changes visual experience, intake and feedback together; this diagnostic cannot separately identify their contributions."])
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
        elapsed, reconstruction = time.monotonic() - started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed - reconstruction,
            reconstruction_seconds=reconstruction, family_seconds=auditor.family_seconds, completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
