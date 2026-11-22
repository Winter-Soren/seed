"""Independent E135 causal first-view, frozen-policy and trajectory audit.

No simulator or evaluator is imported. Saved first-decision neural arrays are
read with NumPy; JAX is used only to reconstruct assigned randomness/actions.
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
import audit_replay_selection as shared
import audit_scene_transfer_physics as physics

encounter, base, source_bytes = shared.encounter, shared.base, shared.source_bytes
require, same, subset, typed = base.require, base.same, base.subset, base.typed
exact, fingerprint, prefixed = shared.exact, shared.fingerprint, shared.prefixed
CFG, MOVES, NEURAL, PLACEMENTS = shared.CFG, shared.MOVES, shared.NEURAL, shared.PLACEMENTS
PROTOCOL = "scene-transfer-v1"
ARMS, HISTORIES, VIEWS = ("parent", "descendant"), ("stationary", "reversal"), ("masked", "full")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe_action", "continuation", "solo_layout", "solo_action")
CELLS = ((8, 16), (16, 32), (8, 32), (16, 16))
FIELDS = ("Y", "first_food", "first_poison", "no_contact", "survival", "latency")
SOURCE_PATHS = {"seed/seed.py", "seed/encounter_diagnostic.py", "seed/replay_selection.py",
    "scripts/social_observer_evolution.py", "scripts/run_replay_selection.py", "scripts/run_experiment.py",
    "scripts/run_cold_donor.py", "scripts/run_shelter_evolution.py", "seed/scene_transfer.py", "scripts/run_scene_transfer.py"}
ENDPOINT_SOURCE = "e514d3fea404f2e765d0b065dd26ac407eab2695"
ENDPOINTS = (
    dict(seed_index=0, run="runs/20261122-214433_E134-A0-train",
        config_sha256="818cd175a86e6c814656e5aa0d933d996f568dea8803f2f0dad29bdfa30c8fb7",
        summary_sha256="b0494b82e94c0c9a7a872cc82886491873b0decb95a642712f65ab3657938a4c",
        checkpoint_sha256="a3a333f16010c2531c837d62228f5f0fa90c099a69a3c15188c0f06e9da39dcf",
        endpoint_sha256="936176dfea318a4ec698393e5b693877b8a1bd3dd4406db6d9c76778dfe76378"),
    dict(seed_index=1, run="runs/20261122-214515_E134-A1-train",
        config_sha256="101811f422b2a2772f60b763f951bcba84f94a00f0da98b1ce9ee3263f6fc0e6",
        summary_sha256="3abf0e5aae5e0fd4faa247948662700a5c764711ece9b481611a4fae2743f95a",
        checkpoint_sha256="db450a9064c87ac50581d9bd9fc224cf96446aae08672e93e365217393c57d81",
        endpoint_sha256="ed054c08356b30ea7ccd80f4346f09a4f07cd371729c3d86e7b4a14aa837f37a"),
    dict(seed_index=2, run="runs/20261122-214556_E134-A2-train",
        config_sha256="e30dea71dd52f7c1639b2676751ce6eed477e339152f39bfbbde7df5044f2a8f",
        summary_sha256="0e4faf98c274313598a2ab9d101f0d46ee1846f70873cf9127eab931a2050882",
        checkpoint_sha256="03553b4d9755ebcd694620f7de73186803d7d275a7ab67130cf2d4b3e7990f79",
        endpoint_sha256="bcf5ad20fd1e38c322bb459c3d694f64a7fe8ca018720a47c7942a49d2191e14"))
NEURAL_SHAPES = dict(W=(1, 157, 157), x=(1, 157), Wp=(1, 103, 48), mask=(1, 157, 157))
TRACE_SHAPES = dict(grid_before=(16, 16), position_before=(1, 2), energy_before=(1,), signal_before=(1,),
    incoming_denergy=(1,), observation=(1, 103), decision_motor=(1, 5), logits=(1, 5), probabilities=(1, 5),
    sampled_action=(1,), forced_action=(), action=(1,), action_key=(2,), regrowth_key=(2,), regrowth_cell=(1, 2),
    key_before=(2,), key_after=(2,), grid=(16, 16), position=(1, 2), energy=(1,), unclipped_energy=(1,),
    signal=(1,), alive_before=(1,), alive_after=(1,), denergy=(1,), efference=(1, 5), food=(1,), poison=(1,),
    item=(1,), surprise=(), rule=(), t=(), hebbian_enabled=(), predictor_enabled=())
INTEGER_FIELDS = {"grid_before", "position_before", "sampled_action", "forced_action", "action", "regrowth_cell",
                  "grid", "position", "item", "rule", "t"}
KEY_FIELDS = {"action_key", "regrowth_key", "key_before", "key_after"}
BOOL_FIELDS = {"food", "poison", "hebbian_enabled", "predictor_enabled"}


def numeric(values, name, shape, dtype):
    value = typed(values, name, shape, np.dtype(dtype).kind)
    require(value.dtype == np.dtype(dtype) and np.isfinite(value).all(), "exact finite native evidence type " + name)
    return value


def trace_arrays(raw, prefix, leading):
    return {name: numeric(raw, prefix + name, leading + shape, np.int32 if name in INTEGER_FIELDS else
        np.uint32 if name in KEY_FIELDS else bool if name in BOOL_FIELDS else np.float32) for name, shape in TRACE_SHAPES.items()}


@lru_cache(maxsize=1)
def random_helpers():
    jax, jnp, script, world, tape = shared.random_helpers()
    hub = np.asarray([8, 8])
    excluded = ((hub + MOVES) * np.asarray([16, 1])).sum(axis=-1)
    eligible = jnp.asarray(np.setdiff1d(np.arange(256, dtype=np.int32), excluded))

    @jax.jit
    def scene_assignments(keys):
        cells = jax.random.choice(keys[3], eligible, (46,), replace=False)
        background = jnp.zeros(256, jnp.int32).at[cells[:23]].set(1).at[cells[23:]].set(2).reshape(16, 16)
        return dict(background=background, continuation=jax.random.randint(keys[5], (), 0, 12),
                    probe_keys=jax.random.split(keys[4], 12))

    @jax.jit
    def first_tape(keys, logits):
        def one(key, logit):
            following, action_key, regrowth_key = jax.random.split(key, 3)
            return key, following, action_key, regrowth_key, jax.random.categorical(action_key, logit), jax.random.randint(regrowth_key, (1, 2), 0, 16)
        return jax.vmap(one)(keys, logits)

    def derive(keys):
        values = {name: np.asarray(value) for name, value in scene_assignments(keys).items()}
        values["script"] = np.asarray(script(keys[0]))
        values["reference_grid"], values["reference_pos"] = (np.asarray(value) for value in world(keys[6]))
        return values
    return jax, derive, tape, first_tape


def neutral_body(grid, position, key):
    return dict(grid=np.asarray(grid).copy(), pos=np.asarray(position).copy(), energy=np.ones(1, np.float32),
        alive=np.ones(1, np.float32), denergy=np.zeros(1, np.float32), sig=np.zeros(1, np.float32), key=np.asarray(key).copy())


def frozen_policy(snapshot, observations):
    """Independent float64 tanh recurrence with a float32 roundoff envelope.

    Each matrix product uses the standard gamma_n absolute dot-product bound.
    Propagate that interval through monotone tanh, allowing eight float32 unit
    roundoffs for its implementation. Sensors are clamped after both microticks.
    This checks the real saved acquired brain without using a simulator operator.
    """
    weight = snapshot["W"][0].astype(np.float64)
    obs = np.asarray(observations, np.float64)
    require(obs.ndim == 2 and obs.shape[1] == 103, "frozen first-decision input dimensions")
    x = np.broadcast_to(snapshot["x"][0], (len(obs), 157)).astype(np.float64).copy()
    error = np.zeros_like(x)
    x[:, :103] = obs
    unit = np.finfo(np.float32).eps / 2
    gamma = 157*unit / (1-157*unit)
    absolute_weight = np.abs(weight).T
    for _ in range(2):
        product = x @ weight.T
        bound = error @ absolute_weight + gamma*((np.abs(x)+error) @ absolute_weight)
        result = np.tanh(product)
        error = np.maximum(np.tanh(product+bound)-result, result-np.tanh(product-bound)) + 8*unit
        x = result
        x[:, :103], error[:, :103] = obs, 0
    logits = 3*x[:, -6:-1]
    return logits, 3*error[:, -6:-1] + unit*np.abs(logits) + 32*np.finfo(np.float64).eps


def check_frozen_policy(snapshot, trace):
    obs = trace["observation"].reshape(-1, 103)
    expected, error = frozen_policy(snapshot, obs)
    actual = trace["logits"].reshape(-1, 5).astype(np.float64)
    require(np.all(np.abs(actual-expected) <= error), "frozen first logits follow the saved acquired W and x")


def trace_physics(data, initial, rules, *, learning=True, script=None, gap=0, masked_first=None):
    physics.verify(data, initial, rules, random_helpers()[2], learning=learning,
                   script=script, gap=gap, masked_first=masked_first)
    return outcomes(data)


def outcomes(trace):
    food, poison, alive = (trace[name][:, 0] for name in ("food", "poison", "alive_after"))
    contact = food | poison
    first = int(np.flatnonzero(contact)[0]) if np.any(contact) else None
    good, bad = float(first is not None and food[first]), float(first is not None and poison[first])
    return dict(Y=good-bad, first_food=good, first_poison=bad, no_contact=float(first is None),
                survival=float(alive[-1]), latency=float(50 if first is None else first))


def first_physics(data, background, probe_keys, target, *, random=None):
    """Vectorized checks of all 24 first decisions, each with a fresh body."""
    flat = {name: value.reshape((24,)+TRACE_SHAPES[name]) for name, value in data.items()}
    keys = np.broadcast_to(probe_keys, (2, 12, 2)).reshape(24, 2)
    if random is None:
        random = random_helpers()[3](keys, flat["logits"])
    for name, expected in zip(("key_before", "key_after", "action_key", "regrowth_key", "sampled_action", "regrowth_cell"), random):
        exact(flat[name], np.asarray(expected), "paired native first-decision randomness " + name)
    require(np.all(flat["t"] == 0) and np.all(flat["rule"] == target) and np.all(flat["forced_action"] == -1),
            "all first decisions begin independently at tick zero under their actual rule")
    require(not np.any(flat["hebbian_enabled"]) and not np.any(flat["predictor_enabled"]), "first decision freezes both update processes")
    encounter.softmax(flat["logits"], flat["probabilities"])
    exact(flat["logits"], np.float32(3)*flat["decision_motor"], "native first motor logits")
    require(np.all(np.abs(flat["decision_motor"]) <= 1) and np.all(np.abs(flat["signal"]) <= 1), "bounded first motor/signal")
    exact(flat["action"], flat["sampled_action"], "native first categorical action is actually executed")
    actions = flat["action"][:, 0]
    require(np.isin(actions, np.arange(5)).all(), "all five immediate action alternatives retained")
    focal = np.stack([encounter.pair_grid(pair) for pair in PLACEMENTS])
    full = np.broadcast_to(background+focal, (2, 12, 16, 16)).reshape(24, 16, 16).copy()
    exact(flat["grid_before"], full, "masked and full views share identical complete physical grids")
    position = np.full((24, 1, 2), 8, np.int32)
    exact(flat["position_before"], position, "neutral first body coordinate")
    for name in ("energy_before", "alive_before", "alive_after"):
        exact(flat[name], np.ones((24, 1), np.float32), "standardized living first body " + name)
    for name in ("signal_before", "incoming_denergy"):
        exact(flat[name], np.zeros((24, 1), np.float32), "neutral first physical feedback " + name)
    views = np.concatenate((focal, background+focal))
    obs = np.stack([encounter.observation(grid, np.asarray([8, 8]), 1, 0)[None] for grid in views])
    exact(flat["observation"], obs, "masked first observation removes only nonfocal A/B cues")
    moved = (position+MOVES[actions][:, None]) % 16
    exact(flat["position"], moved, "the first free action moves on the real grid")
    ordinal = np.arange(24)
    item = full[ordinal, moved[:, 0, 0], moved[:, 0, 1]][:, None]
    food, poison = item == 1+target, item == 2-target
    exact(flat["item"], item, "first resource follows the unchanged immediate alternatives")
    exact(flat["food"], food, "first food label follows physical target")
    exact(flat["poison"], poison, "first poison label follows physical target")
    energy = np.ones((24, 1), np.float32)-np.float32(.003)+np.float32(.25)*food-np.float32(.25)*poison
    shared.physical_close(flat["unclipped_energy"], energy, "ordinary first physical energy")
    shared.physical_close(flat["energy"], np.clip(energy, 0, 2), "ordinary first clipping")
    shared.physical_close(flat["denergy"], flat["energy"]-np.float32(1), "realized first energy feedback")
    exact(flat["efference"], np.eye(5, dtype=np.float32)[actions, None], "first action efference copy")
    eaten = (food | poison)[:, 0]
    full[ordinal[eaten], moved[eaten, 0, 0], moved[eaten, 0, 1]] = 0
    cells = flat["regrowth_cell"][:, 0]
    require(np.all((cells >= 0) & (cells < 16)), "first regrowth cells are valid")
    full[ordinal[eaten], cells[eaten, 0], cells[eaten, 1]] = item[eaten, 0]
    exact(flat["grid"], full, "first physical consumption and regrowth never follow the masked view")


def first_decomposition(probabilities, target):
    """Within-placement signed score = approach times conditional preference."""
    values = probabilities[:, :, 0, :].astype(np.float64)
    good = np.take_along_axis(values, PLACEMENTS[None, :, target, None], axis=-1)[..., 0]
    bad = np.take_along_axis(values, PLACEMENTS[None, :, 1-target, None], axis=-1)[..., 0]
    approach = good+bad
    conditional = np.divide(good-bad, approach, out=np.zeros_like(approach), where=approach != 0)
    other = np.all(np.arange(5)[None, :, None] != PLACEMENTS[:, None, :], axis=-1)
    parts = dict(p_food=good.mean(-1), p_poison=bad.mean(-1), p_other=np.where(other, values, 0).sum(-1).mean(-1),
                 approach=approach.mean(-1), conditional=conditional.mean(-1))
    decomposition = dict(approach=float(np.mean((approach[0]-approach[1])*(conditional[0]+conditional[1])/2)),
        conditional=float(np.mean((conditional[0]-conditional[1])*(approach[0]+approach[1])/2)))
    require(np.isclose(sum(decomposition.values()), (good[0]-bad[0]-good[1]+bad[1]).mean(), atol=1e-12),
            "approach/conditional parts sum to the signed masked-minus-full score")
    return parts, decomposition


def audit_family(raw, index, parent, descendant, keys, pre, gap, randomness=None):
    require((pre, gap) == CELLS[index % 4], "fixed balanced schedule depends only on assigned family index")
    steps = 2*(pre+16)+gap
    groups = {"exposure_": (2, 2, 2, steps), "first_": (2, 2, 2, 2, 12),
              "scene_": (2, 2, 2, 2, 50), "reference_": (2, 2, 2, 50)}
    top = {"base_index", "keys", "pre_contacts", "gap", "continuation", "initial_rules", "genome_sha256", "script",
           "background", "reference_pos"}
    neural = {prefix+name for prefix in ("acquired_", "scene_initial_", "reference_initial_") for name in NEURAL}
    expected_fields = top | neural | {"acquired_"+name+"_sha256" for name in NEURAL}
    expected_fields |= {prefix+name for prefix in groups for name in TRACE_SHAPES}
    require(set(raw) == expected_fields, "exact compact first-view and acquired-neural evidence schema")
    for name, expected in (("base_index", index), ("pre_contacts", pre), ("gap", gap)):
        exact(numeric(raw, name, (), np.int32), np.asarray(expected, np.int32), "registered family identity " + name)
    numeric(raw, "continuation", (), np.int32)
    exact(numeric(raw, "keys", (8, 2), np.uint32), keys, "eight independent role keys")
    exact(numeric(raw, "initial_rules", (2,), np.int32), np.asarray([0, 1], np.int32), "both initial hidden rules")
    exact(numeric(raw, "genome_sha256", (2, 32), np.uint8), np.stack((fingerprint(parent), fingerprint(descendant))), "pinned parent and attempt-ten descendant")
    if randomness is None:
        randomness = random_helpers()[1](keys)
    exact(numeric(raw, "script", (pre+16, 5), np.int32), randomness["script"][:pre+16], "independently derived balanced contact script")
    for name, shape in (("background", (16, 16)), ("reference_pos", (1, 2))):
        exact(numeric(raw, name, shape, np.int32), randomness[name], "independent assigned " + name)
    exact(raw["continuation"], randomness["continuation"], "continuation was assigned before observing any choice")
    continuation = int(raw["continuation"])
    require(0 <= continuation < 12, "one preassigned ordered placement continues")
    cross = (np.asarray([8, 8])+MOVES) % 16
    require(np.all(raw["background"][cross[:, 0], cross[:, 1]] == 0)
        and np.count_nonzero(raw["background"] == 1) == np.count_nonzero(raw["background"] == 2) == 23,
        "balanced background preserves the hub and all immediate alternatives")
    traces = {prefix[:-1]: trace_arrays(raw, prefix, shape) for prefix, shape in groups.items()}
    acquired = {name: numeric(raw, "acquired_"+name, (2, 2, 2)+shape, np.float32) for name, shape in NEURAL_SHAPES.items()}
    for name, shape in NEURAL_SHAPES.items():
        numeric(raw, "acquired_"+name+"_sha256", (2, 2, 2, 32), np.uint8)
        reference = numeric(raw, "reference_initial_"+name, (2, 2, 2)+shape, np.float32)
        scene = numeric(raw, "scene_initial_"+name, (2, 2, 2, 2)+shape, np.float32)
        exact(reference, acquired[name], "ordinary reference uses the unprobed acquired " + name)
        exact(scene, np.repeat(acquired[name][:, :, :, None], 2, axis=3), "both scenes copy the same complete acquired " + name)
    require(np.isin(acquired["mask"], (0, 1)).all() and np.all(np.abs(acquired["W"]) <= 3), "native acquired mask and clipped weights")
    for name, value in traces["exposure"].items():
        exact(value[:, :, 0, :2*pre], value[:, :, 1, :2*pre], "stationary/reversal clones share the entire old-rule history " + name)
    local, sampled = np.empty((2, 2, 2, 2)), np.empty((2, 2, 2, 2))
    scene_out = {name: np.empty((2, 2, 2, 2)) for name in FIELDS}
    reference_out = {name: np.empty((2, 2, 2)) for name in FIELDS}
    choice = {name: np.empty((2, 2, 2, 2)) for name in ("p_food", "p_poison", "p_other", "approach", "conditional")}
    decomposition = {name: np.empty((2, 2, 2)) for name in ("approach", "conditional")}
    n_distractors = int(np.count_nonzero(raw["background"][6:11, 6:11]))
    for genotype in (0, 1):
        for initial in (0, 1):
            for history in (0, 1):
                address, target = (genotype, initial, history), initial ^ history
                exposure = {name: value[address] for name, value in traces["exposure"].items()}
                initial_body = neutral_body(np.zeros((16, 16), np.int32), np.asarray([[8, 8]], np.int32), keys[2])
                rules = np.full(steps, target, np.int32)
                rules[:2*pre] = initial
                trace_physics(exposure, initial_body, rules, script=raw["script"], gap=gap)
                snapshot = {name: value[address] for name, value in acquired.items()}
                for name, array in snapshot.items():
                    exact(fingerprint(array), raw["acquired_"+name+"_sha256"][address], "actual acquired neural bytes " + name)
                exact(snapshot["x"][:, :103], exposure["observation"][-1], "acquired sensor state is the actual last exposure input")
                exact(snapshot["x"][:, -6:-1], exposure["efference"][-1], "acquired motor carry is the last executed action")
                exact(snapshot["x"][:, -1], exposure["signal"][-1], "acquired living signal carry")
                first = {name: value[address] for name, value in traces["first"].items()}
                check_frozen_policy(snapshot, first)
                first_physics(first, raw["background"], randomness["probe_keys"], target)
                parts, contributions = first_decomposition(first["probabilities"], target)
                for name, values in parts.items():
                    choice[name][address] = values
                for name, value in contributions.items():
                    decomposition[name][address] = value
                for view in (0, 1):
                    probability = first["probabilities"][view, :, 0].astype(np.float64)
                    local[address+(view,)] = np.mean(probability[np.arange(12), PLACEMENTS[:, target]]
                                                   - probability[np.arange(12), PLACEMENTS[:, 1-target]])
                    actions = first["action"][view, :, 0]
                    sampled[address+(view,)] = np.mean((actions == PLACEMENTS[:, target]).astype(float)
                        - (actions == PLACEMENTS[:, 1-target]).astype(float))
                    trace = {name: value[address+(view,)] for name, value in traces["scene"].items()}
                    for name in TRACE_SHAPES:
                        exact(trace[name][0], first[name][view, continuation], "continuation begins with exactly the preassigned first decision " + name)
                    focal = encounter.pair_grid(PLACEMENTS[continuation])
                    body = neutral_body(raw["background"]+focal, np.asarray([[8, 8]], np.int32), randomness["probe_keys"][continuation])
                    flags = np.ones(50, bool)
                    flags[0] = False
                    behavior = trace_physics(trace, body, np.full(50, target, np.int32), learning=flags,
                                             masked_first=focal if view == 0 else None)
                    for name, value in behavior.items():
                        scene_out[name][address+(view,)] = value
                if n_distractors == 0:
                    for name in TRACE_SHAPES:
                        exact(first[name][0], first[name][1], "empty visible background makes views identical " + name)
                        exact(traces["scene"][name][address+(0,)], traces["scene"][name][address+(1,)], "identical first input continues identically " + name)
                reference = {name: value[address] for name, value in traces["reference"].items()}
                # The archived reference_grid name holds post-transition grids.
                # The independently derived initial layout must match every clone's grid_before[0].
                reference_body = neutral_body(randomness["reference_grid"], raw["reference_pos"], keys[7])
                behavior = trace_physics(reference, reference_body, np.full(50, target, np.int32))
                for name, value in behavior.items():
                    reference_out[name][address] = value
    return dict(base_index=index, n_distractors=n_distractors, local=local, sampled=sampled,
                scene=scene_out, reference=reference_out, first_choice=choice, decomposition=decomposition)


def estimate(values):
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "independent finite family vector")
    n, mean = len(values), float(values.mean())
    se = float(np.sqrt(np.square(values-mean).sum() / (n*(n-1))))
    return dict(n=n, mean=mean, se=se, lower_2se=mean-2*se, upper_2se=mean+2*se)


def behavior_summary(values):
    report = {name: estimate(array) for name, array in values.items()}
    good = values["first_food"]
    contact = good+values["first_poison"]
    n, total = len(good), int(2*contact.sum())
    food = int(2*good.sum())
    ratio = None if total == 0 else food/total
    error = None if total == 0 else estimate((good-ratio*contact)/contact.mean())["se"]
    report["contact_accuracy"] = dict(n=n, mean=ratio, se=error, food_count=food, contact_count=total,
                                      assigned_lives=2*n, descriptive=True)
    return report


def summarize_rows(rows):
    n = len(rows)
    require(n >= 2 and [row["base_index"] for row in rows] == list(range(n)), "complete consecutive independent family denominator")
    counts = np.asarray([row["n_distractors"] for row in rows], np.float64)
    require(np.isfinite(counts).all() and np.all((counts >= 0) & (counts <= 20)) and np.all(counts == np.floor(counts)),
            "actual visible nonfocal distractor count")
    values = {name: np.stack([row[name] for row in rows]).astype(np.float64) for name in ("local", "sampled")}
    for name, value in values.items():
        require(value.shape == (n, 2, 2, 2, 2) and np.isfinite(value).all() and np.all(np.abs(value) <= 1), "all assigned first choices " + name)
    behavior = {kind: {field: np.stack([row[kind][field] for row in rows]).astype(np.float64) for field in FIELDS}
                for kind in ("scene", "reference")}
    for kind, fields in behavior.items():
        shape = (n, 2, 2, 2, 2) if kind == "scene" else (n, 2, 2, 2)
        require(all(value.shape == shape and np.isfinite(value).all() for value in fields.values()), "all assigned trajectory endpoints " + kind)
        for name in ("first_food", "first_poison", "no_contact", "survival"):
            require(np.isin(fields[name], (0, 1)).all(), "binary all-assigned endpoint " + name)
        exact(fields["Y"], fields["first_food"]-fields["first_poison"], "first signed resource score")
        require(np.all(fields["first_food"]+fields["first_poison"]+fields["no_contact"] == 1), "no-contact is never omitted")
        require(np.all((fields["latency"] >= 0) & (fields["latency"] <= 50))
            and np.all(fields["latency"] == np.floor(fields["latency"]))
            and np.array_equal(fields["latency"] == 50, fields["no_contact"] == 1), "zero-based first contact and 50 for no contact")
    choice = {field: np.stack([row["first_choice"][field] for row in rows]).astype(np.float64)
              for field in ("p_food", "p_poison", "p_other", "approach", "conditional")}
    decomposition = {field: np.stack([row["decomposition"][field] for row in rows]).astype(np.float64)
                     for field in ("approach", "conditional")}
    require(all(value.shape == (n, 2, 2, 2, 2) and np.isfinite(value).all() for value in choice.values()), "full first-choice descriptive axes")
    require(all(value.shape == (n, 2, 2, 2) and np.isfinite(value).all() for value in decomposition.values()), "within-placement decomposition axes")
    require(np.allclose(values["local"], choice["p_food"]-choice["p_poison"], rtol=0, atol=1e-12)
        and np.allclose(choice["approach"], choice["p_food"]+choice["p_poison"], rtol=0, atol=1e-12), "signed first-choice decomposition preserves actual probabilities")
    require(np.allclose(sum(decomposition.values()), values["local"][..., 0]-values["local"][..., 1], rtol=0, atol=1e-12),
            "approach and conditional components add to each view contrast before averaging")
    values = {name: value.mean(axis=2) for name, value in values.items()}
    behavior = {kind: {field: value.mean(axis=2) for field, value in fields.items()} for kind, fields in behavior.items()}
    choice = {field: value.mean(axis=2) for field, value in choice.items()}
    decomposition = {field: value.mean(axis=2) for field, value in decomposition.items()}
    histories = {}
    for history, label in enumerate(HISTORIES):
        masked, full = (values["local"][:, 1, history]-values["local"][:, 0, history]).T
        scene_masked, scene_full = (behavior["scene"]["Y"][:, 1, history]-behavior["scene"]["Y"][:, 0, history]).T
        report = dict(masked_gain=estimate(masked), full_gain=estimate(full), attenuation=estimate(masked-full),
            reference_gain=estimate(behavior["reference"]["Y"][:, 1, history]-behavior["reference"]["Y"][:, 0, history]),
            scene_gain_masked=estimate(scene_masked), scene_gain_full=estimate(scene_full), scene_attenuation=estimate(scene_masked-scene_full))
        for name, array in values.items():
            report[name] = {arm: {view: estimate(array[:, g, history, v]) for v, view in enumerate(VIEWS)} for g, arm in enumerate(ARMS)}
        report["first_choice"] = {arm: {view: {field: estimate(array[:, g, history, v]) for field, array in choice.items()}
                                        for v, view in enumerate(VIEWS)} for g, arm in enumerate(ARMS)}
        report["decomposition"] = {field: estimate(array[:, 1, history]-array[:, 0, history]) for field, array in decomposition.items()}
        report["scene"] = {arm: {view: behavior_summary({field: array[:, g, history, v] for field, array in behavior["scene"].items()})
                                 for v, view in enumerate(VIEWS)} for g, arm in enumerate(ARMS)}
        report["reference"] = {arm: behavior_summary({field: array[:, g, history] for field, array in behavior["reference"].items()})
                               for g, arm in enumerate(ARMS)}
        histories[label] = report
    return dict(families=n, distractors=estimate(counts), zero_distractors=estimate(counts == 0), histories=histories)


GATE_FIELDS = ("isolated_replicates", "attenuation_confirmed", "passed", "practical_loss_excluded",
               "advantage_erased", "partial_attenuation", "reference_deficit")


def gate(summary):
    values = summary["histories"]["stationary"]
    isolated, attenuation, full = (values[name] for name in ("masked_gain", "attenuation", "full_gain"))
    replicates = bool(isolated["mean"] >= .05 and isolated["lower_2se"] > 0)
    attenuated = bool(attenuation["mean"] >= .05 and attenuation["lower_2se"] > 0)
    passed = replicates and attenuated
    return dict(isolated_replicates=replicates, attenuation_confirmed=attenuated, passed=passed,
        practical_loss_excluded=bool(replicates and attenuation["upper_2se"] < .05),
        advantage_erased=bool(passed and full["upper_2se"] <= 0),
        partial_attenuation=bool(passed and full["lower_2se"] > 0),
        reference_deficit=bool(values["reference_gain"]["upper_2se"] < 0))


def replicated_gates(panels):
    require([panel["index"] for panel in panels] == [0, 1, 2], "all three original lineages separately")
    for panel in panels:
        same(panel["gate"], gate(panel["measurements"]), "independent stationary decision classifications")
    return {name: dict(count=sum(panel["gate"][name] for panel in panels),
        seeds=[panel["index"] for panel in panels if panel["gate"][name]], passed=all(panel["gate"][name] for panel in panels)) for name in GATE_FIELDS}


class Auditor(shared.Auditor):
    MAX_SECONDS = {"development": 150, "final": 370}

    def __init__(self, root, source):
        super().__init__(root, source)
        self.admission_commit = None
        self.endpoint_arrays = None

    def artifact(self, folder, manifest):
        require(set(manifest) in ({"path", "sha256", "bytes"}, {"index", "path", "sha256", "bytes"}), "complete bounded artifact reference")
        return super().artifact(folder, {name: manifest[name] for name in ("path", "sha256", "bytes")})

    def completed_bundle(self, development=False):
        count, names = (4, ["E135-dev"]) if development else (128, [f"E135-A{i}" for i in range(3)])
        folders, pending = [], []
        for name in names:
            paths = list(self.path("runs").glob("*_"+name))
            require(len(paths) <= 1, "one immutable registered attempt " + name)
            if not paths or not (paths[0]/"status.json").is_file():
                pending.append(name)
                continue
            folder = paths[0].relative_to(self.root).as_posix()
            status = self.json(folder+"/status.json")
            if (status.get("status") != "finished" or status.get("completed_families") != count
                    or status.get("expected_families") != count or not (paths[0]/"summary.json").is_file()
                    or {file.name for file in (paths[0]/"families").glob("*.npz")} != {f"family_{i:04d}.npz" for i in range(count)}):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending)+" incomplete; no final measurements opened")
        return folders

    def endpoints(self):
        if self.endpoint_arrays is not None:
            return self.endpoint_arrays
        arrays = []
        for index, receipt in enumerate(ENDPOINTS):
            content = {}
            for name, field in (("config.json", "config_sha256"), ("summary.json", "summary_sha256"),
                ("generations/attempt_0010.npz", "checkpoint_sha256"), ("genome_final.npy", "endpoint_sha256")):
                value = self.read(receipt["run"]+"/"+name)
                same(hashlib.sha256(value).hexdigest(), receipt[field], "pre-existing attempt-ten receipt " + name)
                content[name] = value
            config, report = (json.loads(content[name]) for name in ("config.json", "summary.json"))
            subset(config, dict(protocol="replay-selection-v1", mode="training", seed_index=index, expected_attempts=10), "original completed selection source")
            subset(config["git"], dict(commit=ENDPOINT_SOURCE, dirty=False), "frozen source of attempt-ten endpoint")
            subset(report, dict(status="finished", completed_attempts=10, completed_families=0,
                                endpoint_sha256=receipt["endpoint_sha256"]), "fixed terminal endpoint, no checkpoint choice")
            genome = np.load(io.BytesIO(content["genome_final.npy"]), allow_pickle=False)
            require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite descendant encoding")
            with np.load(io.BytesIO(content["generations/attempt_0010.npz"]), allow_pickle=False) as checkpoint:
                exact(checkpoint["mu"], genome, "endpoint is the final registered optimizer mean")
                require(int(checkpoint["t"]) == report["optimizer_updates"], "registered terminal optimizer count")
            path, digest = base.ORIGINALS[index]
            subset(config["source"], dict(path=path, sha256=digest, training_seed=index), "correct original training parent")
            same(hashlib.sha256(self.read(path)).hexdigest(), digest, "original parent pin before new assignments")
            arrays.append(genome)
        self.endpoint_arrays = arrays
        return arrays

    def metadata(self, folder, index, development):
        config = self.json(folder+"/config.json")
        mode = "development" if development else "final"
        count, master, name = (4, 2026121500, "E135-dev") if development else (128, 2026121501+index, f"E135-A{index}")
        subset(config, dict(protocol=PROTOCOL, mode=mode, seed_index=index, evaluation_seed=master, expected_families=count,
            cfg=CFG, key_roles=list(KEY_ROLES), arms=list(ARMS), histories=list(HISTORIES), views=list(VIEWS),
            endpoint_source=ENDPOINT_SOURCE, train_receipts=list(ENDPOINTS), raw_schema=PROTOCOL,
            raw_pattern="families/family_{index:04d}.npz", cold_families=[0, 1], warm_families=list(range(2, count)),
            max_seconds=self.MAX_SECONDS[mode], scientific_result=False), "registered fixed subjects and matched scene design")
        path, digest = base.ORIGINALS[index]
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=PROTOCOL,
            condition_split=mode, start="file:"+path, pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10,
            no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None, objective=None,
            memory_condition=None, max_train_seconds=None), "fresh registered invocation")
        require(config["args"].get("controller") in (None, "rules-v0") and config["args"].get("structural_plasticity") in (None, "off")
            and not config["args"].get("persistence", False) and config.get("retry_of") is None and config.get("recovery_of") is None,
            "no unregistered controller changes or replacement attempts")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit), "clean immutable run source")
        if development:
            same(commit, self.source, "development uses the scientific anchor exactly")
        else:
            self.git("merge-base", "--is-ancestor", self.source, commit)
            changed_code = self.git("diff", "--name-only", self.source, commit, "--", "seed", "scripts", "tools", "tests", "notebooks").splitlines()
            require(set(changed_code) <= {"tools/audit_scene_transfer.py"},
                    "admission descendant changes only tracking material and the registered read-only layout correction")
            if self.admission_commit is None:
                self.admission_commit = commit
            same(commit, self.admission_commit, "all three final panels use the same admitted clean commit")
        require(self.code_trees(commit) == self.expected_trees and set(config["source_code"]) == SOURCE_PATHS,
                "registered scientific code trees and complete transitive manifest")
        context = type("SourceContext", (), {"root": self.root, "source": self.source})()
        attributes = source_bytes.Auditor.source_attributes(context, list(SOURCE_PATHS))
        for source, value in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", self.source+":"+source], cwd=self.root, capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, value, attributes[source])
        if self.source_manifest is None:
            self.source_manifest = config["source_code"]
        same(config["source_code"], self.source_manifest, "all panels retain exact scientific source bytes")
        subset(config["source"], dict(path=path, sha256=digest, training_seed=index), "original E103 parent")
        endpoint = ENDPOINTS[index]
        same(config["endpoint"], dict(path=endpoint["run"]+"/genome_final.npy", sha256=endpoint["endpoint_sha256"]), "fixed attempt-ten descendant")
        descendants = self.endpoints()
        parent_bytes = self.read(path)
        require(hashlib.sha256(parent_bytes).hexdigest() == digest and self.read(folder+"/genome_start.npy") == parent_bytes,
                "exact copied original parent bytes")
        require(self.read(folder+"/genome_final.npy") == self.read(config["endpoint"]["path"]), "exact copied pinned endpoint bytes")
        parent = np.load(io.BytesIO(parent_bytes), allow_pickle=False)
        require(parent.shape == (155,) and parent.dtype == np.float32 and np.isfinite(parent).all(), "finite native parent encoding")
        same(config["assignments"]["path"], "assignments.npz", "complete assignment manifest")
        assignments = self.artifact(folder, config["assignments"])
        require(set(assignments) == {"family_keys", "pre_contacts", "gap", "initial_rules"}, "exact assigned family table")
        jax, derive, _, _ = random_helpers()
        same(config["runtime"]["jax"], jax.__version__, "pinned random-runtime version")
        root = jax.random.fold_in(jax.random.PRNGKey(master), 0xE135)
        keys = np.stack([np.asarray(jax.random.split(jax.random.fold_in(root, i), 8)) for i in range(count)])
        exact(assignments["family_keys"], keys, "independent registered role derivation")
        exact(assignments["initial_rules"], np.asarray([0, 1], np.int32), "both initial rules assigned")
        schedule = np.asarray([CELLS[i % 4] for i in range(count)], np.int32)
        exact(assignments["pre_contacts"], schedule[:, 0], "balanced old-contact schedule")
        exact(assignments["gap"], schedule[:, 1], "balanced gap schedule")
        self.register_keys(keys, "separate development/final family role streams")
        random = [derive(key) for key in keys]
        records = [json.loads(line) for line in self.read(folder+"/evaluation.jsonl").decode().splitlines()]
        require(len(records) == count, "every assigned family appears once")
        for i, record in enumerate(records):
            subset(record, dict(index=i, keys=keys[i].tolist(), pre_contacts=int(schedule[i, 0]), gap=int(schedule[i, 1])), "ordered assigned family journal")
            require(np.isfinite(record["family_seconds"]) and record["family_seconds"] >= 0
                and np.isfinite(record["elapsed_seconds"]) and 0 < record["elapsed_seconds"] <= self.MAX_SECONDS[mode], "finite bounded recorded timing")
        return config, parent, descendants[index], keys, schedule, random, records

    def panel(self, folder, index, development, prepared):
        config, parent, descendant, keys, schedule, random, records = prepared
        count, report = config["expected_families"], self.json(folder+"/summary.json")
        subset(report, dict(status="finished", protocol=PROTOCOL, mode=config["mode"], seed_index=index,
            completed_families=count, expected_families=count, scientific_result=False, artifact_validity="passed",
            gate_status="not_applicable" if development else "undecided"), "complete immutable scene-transfer panel")
        require(len(report["raw_files"]) == count and 0 < report["elapsed_seconds"] <= config["max_seconds"], "complete panel within its fixed allowance")
        for field in ("family_seconds", "family_wall_seconds"):
            value = np.asarray(report[field], np.float64)
            require(value.shape == (count,) and np.isfinite(value).all() and np.all(value >= 0), "all measured family durations")
        same(report["family_seconds"], [record["family_seconds"] for record in records], "journal timing linkage")
        require(np.all(np.asarray(report["family_wall_seconds"]) >= report["family_seconds"]), "whole-family wall times include all recorded work")
        rows = []
        for i, record in enumerate(records):
            started = time.monotonic()
            artifact = report["raw_files"][i]
            subset(artifact, dict(index=i, path=f"families/family_{i:04d}.npz"), "ordered original family evidence")
            same(record["raw_file"], artifact, "journal and saved evidence reference")
            rows.append(audit_family(self.artifact(folder, artifact), i, parent, descendant, keys[i],
                                     int(schedule[i, 0]), int(schedule[i, 1]), random[i]))
            self.family_seconds.append(time.monotonic()-started)
        measurements = summarize_rows(rows)
        decision = gate(measurements)
        same(report["measurements"], measurements, "independent all-family statistics and descriptives")
        same(report["gate"], decision, "independent preregistered decision")
        return dict(index=index, run=folder, measurements=measurements, gate=decision)


def synthetic_first():
    """Literal all-stay, zero-weight fixture without simulator or random runtime."""
    background = np.zeros((16, 16), np.int32)
    background[6, 6], background[14, 14] = 2, 1
    keys = np.arange(24, dtype=np.uint32).reshape(12, 2)
    data = {name: np.zeros((2, 12)+shape, np.int32 if name in INTEGER_FIELDS else np.uint32 if name in KEY_FIELDS
        else bool if name in BOOL_FIELDS else np.float32) for name, shape in TRACE_SHAPES.items()}
    for view in (0, 1):
        for rotation, placement in enumerate(PLACEMENTS):
            focal = encounter.pair_grid(placement)
            grid = background+focal
            data["grid_before"][view, rotation] = grid
            data["grid"][view, rotation] = grid
            data["position_before"][view, rotation] = data["position"][view, rotation] = 8
            data["energy_before"][view, rotation] = data["alive_before"][view, rotation] = data["alive_after"][view, rotation] = 1
            data["energy"][view, rotation] = data["unclipped_energy"][view, rotation] = np.float32(1)-np.float32(.003)
            data["denergy"][view, rotation] = data["energy"][view, rotation]-np.float32(1)
            data["observation"][view, rotation] = encounter.observation(focal if view == 0 else grid, np.asarray([8, 8]), 1, 0)[None]
            data["forced_action"][view, rotation] = -1
            data["probabilities"][view, rotation] = .2
            data["efference"][view, rotation, 0, 0] = 1
            for offset, field in enumerate(("key_before", "key_after", "action_key", "regrowth_key")):
                data[field][view, rotation] = keys[rotation]+np.uint32(100*offset)
    random = tuple(data[name].reshape((24,)+TRACE_SHAPES[name]) for name in
        ("key_before", "key_after", "action_key", "regrowth_key", "sampled_action", "regrowth_cell"))
    snapshot = {name: np.zeros(shape, np.float32) for name, shape in NEURAL_SHAPES.items()}
    return data, snapshot, background, keys, random


def synthetic_rows():
    rows = []
    for index in range(4):
        local = np.full((2, 2, 2, 2), .1+.02*index, np.float64)
        local[1, :, :, 0] += .2
        choice = dict(p_food=(.8+local)/2, p_poison=(.8-local)/2, p_other=np.full_like(local, .2),
                      approach=np.full_like(local, .8), conditional=local/.8)
        scene, reference = {}, {}
        for name in FIELDS:
            scene[name] = np.full_like(local, 1. if name in ("Y", "first_food", "survival") else 0.)
            reference[name] = np.full((2, 2, 2), 1. if name in ("Y", "first_food", "survival") else 0.)
        reference["Y"][1] = -1
        reference["first_food"][1] = 0
        reference["first_poison"][1] = 1
        rows.append(dict(base_index=index, n_distractors=index, local=local, sampled=np.zeros_like(local), first_choice=choice,
            decomposition=dict(approach=np.zeros((2, 2, 2)), conditional=local[..., 0]-local[..., 1]), scene=scene, reference=reference))
    return rows


def self_check():
    data, snapshot, background, keys, random = synthetic_first()
    first_physics(data, background, keys, 0, random=random)
    check_frozen_policy(snapshot, data)
    for field, address, value in (("grid_before", (0, 0, 6, 6), 0), ("observation", (1, 0, 0, 25), 0.),
        ("hebbian_enabled", (0, 0), True), ("energy", (0, 0, 0), 1.)):
        altered = copy.deepcopy(data)
        altered[field][address] = value
        try:
            first_physics(altered, background, keys, 0, random=random)
        except base.AuditError:
            pass
        else:
            raise base.AuditError("first-view physical/input corruption was accepted: " + field)
    snapshot["W"][0, 151, 7] = .5
    expected, _ = frozen_policy(snapshot, data["observation"].reshape(-1, 103))
    neural = {name: value.copy() for name, value in data.items()}
    neural["logits"] = expected.astype(np.float32).reshape(2, 12, 1, 5)
    check_frozen_policy(snapshot, neural)
    neural["logits"][0, 0, 0, 0] += .1
    try:
        check_frozen_policy(snapshot, neural)
    except base.AuditError:
        pass
    else:
        raise base.AuditError("incorrect first frozen neural calculation accepted")
    probability = np.zeros((2, 12, 1, 5), np.float32)
    for view, good, bad in ((0, .6, .1), (1, .3, .2)):
        for rotation, placement in enumerate(PLACEMENTS):
            probability[view, rotation, 0] = (1-good-bad)/3
            probability[view, rotation, 0, placement[0]] = good
            probability[view, rotation, 0, placement[1]] = bad
    parts, decomposition = first_decomposition(probability, 0)
    require(np.isclose(sum(decomposition.values()), .4, atol=1e-7) and parts["p_other"][1] > parts["p_other"][0],
            "descriptive split preserves signed attenuation and actual other-action probability")
    measured = summarize_rows(synthetic_rows())
    require(gate(measured)["passed"] and gate(measured)["advantage_erased"] and gate(measured)["reference_deficit"],
            "paired first-view and separate reference decision")
    require(measured["histories"]["stationary"]["attenuation"]["se"] < 1e-15
        and measured["histories"]["stationary"]["local"]["parent"]["masked"]["se"] > 0, "paired family attenuation SE")
    no_contact = {name: np.full(4, 1. if name in ("no_contact", "survival") else 50. if name == "latency" else 0.) for name in FIELDS}
    conditional = behavior_summary(no_contact)["contact_accuracy"]
    require(conditional["mean"] is None and conditional["se"] is None and conditional["contact_count"] == 0
        and conditional["assigned_lives"] == 8, "zero-contact ratio remains null with full assigned denominator")
    boundary = copy.deepcopy(measured)
    stationary = boundary["histories"]["stationary"]
    stationary["masked_gain"] = dict(n=4, mean=.05, se=0., lower_2se=.05, upper_2se=.05)
    stationary["attenuation"] = dict(n=4, mean=.05, se=.025, lower_2se=0., upper_2se=.10)
    require(gate(boundary)["isolated_replicates"] and not gate(boundary)["attenuation_confirmed"], "inclusive minimum with strict evidence")
    stationary["attenuation"] = dict(n=4, mean=0., se=.025, lower_2se=-.05, upper_2se=.05)
    require(not gate(boundary)["practical_loss_excluded"], "practical exclusion upper bound is strict")
    panels = [dict(index=i, measurements=measured, gate=gate(measured)) for i in range(3)]
    require(replicated_gates(panels)["passed"]["passed"], "all three lineages qualify separately")
    failed = copy.deepcopy(measured)
    failed["histories"]["stationary"]["masked_gain"] = dict(n=4, mean=0., se=0., lower_2se=0., upper_2se=0.)
    panels[2] = dict(index=2, measurements=failed, gate=gate(failed))
    require(replicated_gates(panels)["passed"]["count"] == 2 and not replicated_gates(panels)["passed"]["passed"], "one lineage miss prevents replication")
    require("jax" not in sys.modules and not any(name == "seed" or name.startswith("seed.") for name in sys.modules),
            "pure independent fixtures import no simulator or random runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=7, simulator_imported=False,
        coverage=["matched physical first decisions", "mask, body and learning corruption rejection", "saved-state frozen tanh reconstruction",
                  "signed approach/conditional decomposition", "paired family errors and empty-contact denominator",
                  "inclusive floors and strict evidence/exclusion boundaries", "all-three-lineage replication"])


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
            auditor.panel(development, 0, True, auditor.metadata(development, 0, True))
        prepared = [auditor.metadata(folder, index, args.development) for index, folder in enumerate(folders)]
        panels = [auditor.panel(folder, index, args.development, prepared[index]) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            registered_source_code=auditor.source_manifest, admission_commit=auditor.admission_commit,
            panels=panels, replicated_gates=None if args.development else replicated_gates(panels),
            limitations=["All assigned geometry, first-observation masks, native sampled actions, physical trajectories, paired-family estimates and preregistered decisions are independently reconstructed.",
                "First frozen logits are independently recomputed from literal acquired W/x using float64 tanh recurrence and a declared float32 roundoff envelope. Acquired plastic neural dynamics and developmental wiring are not rerun.",
                "Literal acquired arrays, their hashes and exact scene/reference clone equality verify recorded initial states; no dense later neural trajectory is archived to exclude an unrecorded later neural transient.",
                "The first tick alone masks background cues and freezes both update processes in both views. Subsequent scene movement uses full observations and native updates; the separate ordinary reference also uses native updates from its first tick.",
                "Contact-conditioned accuracy and first-choice decompositions are descriptive. Zero-contact assignments remain in primary scores, and only the stationary first-choice contrast determines the registered architecture-selection diagnosis."])
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
        elapsed, reconstruct = time.monotonic()-started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed-reconstruct,
            reconstruction_seconds=reconstruct, family_seconds=auditor.family_seconds, family_wall_seconds=auditor.family_seconds,
            completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
