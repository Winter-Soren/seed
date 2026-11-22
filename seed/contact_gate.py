"""Matched one-creature replay for the E133 Hebbian timing comparison."""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import encounter_diagnostic as encounter, seed


PROTOCOL = "contact-gate-v1"
ASSIGNMENT_TAG = 0xE133
CFG = encounter.CFG
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "sparse")
APPARATUS = dict(pre_contacts=16, post_contacts=8, block_steps=4, delay_ticks=33,
                 grid_steps=50, local_placements=12)
ARMS = tuple(policy + "_" + history for policy in ("native", "contact", "sparse", "reopened")
             for history in ("stationary", "reversal"))
FIELDS = ("local_S", "solo_Y", "first_food", "first_poison", "no_contact",
          "exposure_survival", "solo_survival")
NATIVE, CONTACT, SPARSE = range(3)
GRID_FIELDS = ("food", "poison", "alive_before", "alive_after", "action", "item",
               "energy_before", "energy", "denergy", "regrowth_cell")
GRID_TYPES = (np.bool_, np.bool_, np.bool_, np.bool_, np.int32, np.int32,
              np.float32, np.float32, np.float32, np.int32)


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves the assigned E133 stream")
    tagged = jax.random.fold_in(encounter._master(master), ASSIGNMENT_TAG)
    return np.asarray(jax.random.split(jax.random.fold_in(tagged, int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if count not in (4, 64):
        raise ValueError("E133 assigns four rehearsal or 64 final families")
    return dict(family_keys=np.stack([evaluation_keys(master, i) for i in range(count)]),
                initial_rules=np.asarray([0, 1], np.int32))


def contact_due(previous_observation, previous_efference, alive):
    """The past view and the move taken identify the contacted item in M1."""
    side = 2 * CFG.R + 1
    area = side**2
    dy, dx = seed.MOVES[jnp.argmax(previous_efference)]
    cell = area // 2 + dy * side + dx
    item_seen = (previous_observation[cell] + previous_observation[area + cell]) > 0
    return alive & jnp.any(previous_efference == 1) & item_seen


def replay_tick(state, genome, rule, action, mode, sparse_due):
    inputs = seed.dims(CFG)[1]
    previous = state["x"][0, :inputs]
    efference = state["x"][0, -6:-1]
    alive = state["alive"][0] > 0
    enabled = alive & ((mode == NATIVE) | ((mode == CONTACT) & contact_due(previous, efference, alive))
                       | ((mode == SPARSE) & sparse_due))
    after, trace = encounter._step(state, genome, rule, action, "plastic", CFG,
                                   hebbian=enabled, regrowth=False)
    before_bits = jax.lax.bitcast_convert_type(state["W"], jnp.uint32)
    after_bits = jax.lax.bitcast_convert_type(after["W"], jnp.uint32)
    result = {name: trace[name][0] for name in ("observation", "action", "item", "energy_before",
                                               "energy", "denergy", "incoming_denergy")}
    result.update(prior_observation=previous, prior_efference=efference,
                  alive_before=alive, alive_after=trace["alive_after"][0] > 0,
                  hebbian=trace["hebbian_enabled"], predictor=trace["predictor_enabled"],
                  weight_same=jnp.all(before_bits == after_bits))
    return after, result


@jax.jit
def _replay(flat, state, steps, rule, mode):
    genome = encounter._unravel(CFG)(flat)

    def step(carry, row):
        # Empty stays clear the apparatus; the return retains the unconsumed item.
        scene = jax.lax.switch(row[0], (
            lambda: jnp.zeros_like(carry["grid"]),
            lambda: encounter._pair_grid(carry["pos"], row[1:3], CFG),
            lambda: carry["grid"],
        ))
        return replay_tick(dict(carry, grid=scene), genome, rule, row[3], mode, row[4] > 0)

    return jax.lax.scan(step, state, steps)


def _steps(script, sparse_offsets):
    blocks = np.zeros((APPARATUS["post_contacts"], APPARATUS["block_steps"], 5), np.int32)
    blocks[:, 2, 0] = 1
    blocks[:, 2, 1:3] = script[:, :2]
    blocks[:, 2, 3] = script[:, 3]
    blocks[:, 3, 0] = 2
    blocks[:, 3, 3] = script[:, 4]
    blocks[np.arange(len(blocks)), sparse_offsets, 4] = 1
    return blocks.reshape((-1, 5))


def _hashes(state):
    return np.stack([encounter.array_hash(state[name]) for name in encounter.NEURAL])


def probe(flat, state, grid, position, action_key, probe_keys, rule):
    alive = bool(np.asarray(state["alive"])[0])
    checkpoint_hashes = _hashes(state)
    evidence = dict(probe_alive=np.asarray(alive), checkpoint_hashes=checkpoint_hashes)
    if not alive:
        return dict(evidence, probe_probabilities=np.zeros((12, 5), np.float32),
                    probe_logits=np.zeros((12, 5), np.float32), grid_initial_hashes=np.zeros_like(checkpoint_hashes),
                    **{"grid_" + name: np.zeros((50, 2) if name == "regrowth_cell" else (50,), dtype)
                       for name, dtype in zip(GRID_FIELDS, GRID_TYPES)})

    local = encounter._readouts("plastic", CFG)(flat, state, probe_keys)
    fresh = encounter._fresh_body(state, grid, position, action_key, CFG)
    _, trace = encounter._free_grid("plastic", CFG)(flat, fresh, rule)
    return dict(evidence, probe_probabilities=np.asarray(local["probabilities"][:, 0]),
                probe_logits=np.asarray(local["logits"][:, 0]), grid_initial_hashes=_hashes(fresh),
                **{"grid_" + name: np.asarray(trace[name][:, 0], dtype)
                   for name, dtype in zip(GRID_FIELDS, GRID_TYPES)})


def _trace(first, second):
    return {name: np.concatenate((np.asarray(first[name]), np.asarray(second[name]))) for name in first}


def evaluate_world(flat, keys, index, c=CFG):
    encounter._config(c)
    keys = np.asarray(keys)
    if (keys.shape != (len(KEY_ROLES), 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != len(KEY_ROLES)
            or isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG):
        raise ValueError("E133 requires one complete assigned family with disjoint role keys")
    script_key, birth_key, chamber_key, grid_key, action_key, probe_key, sparse_key = map(jnp.asarray, keys)
    flat = encounter._genome(flat, "plastic", c)
    script = encounter.contact_script(script_key, 16)[:24]
    offsets = np.asarray(jax.random.randint(sparse_key, (8,), 0, 4, dtype=jnp.int32))
    steps = jnp.asarray(_steps(script[16:], offsets))
    delay = jnp.zeros((APPARATUS["delay_ticks"], 5), jnp.int32)
    grid, position = seed.init_world(grid_key, c)
    probe_keys = jax.random.split(probe_key, len(encounter.PLACEMENTS))
    row = dict(base_index=np.asarray(index, np.int32), keys=keys, initial_rules=np.asarray([0, 1], np.int32),
               genome_sha256=encounter.array_hash(flat), script=script, sparse_offsets=offsets,
               solo_grid=np.asarray(grid), solo_position=np.asarray(position[0]))
    old_records = []
    arms = {arm: [] for arm in ARMS}
    for initial_rule in (0, 1):
        birth = encounter._birth(flat, "plastic", birth_key, chamber_key, c)
        old, prior = encounter._chamber("plastic", c)(flat, birth, jnp.asarray(script[:16]), initial_rule)
        if not bool(np.asarray(old["alive"])[0]):
            raise ValueError("the registered common replay must keep every exposure animal alive")
        local = encounter._readouts("plastic", c)(flat, old, probe_keys)
        old_records.append(dict(**{name: np.asarray(prior[name][:, 0]) for name in
                                  ("observation", "action", "energy", "denergy", "item")},
                                alive_after=np.asarray(prior["alive_after"][:, 0] > 0),
                                probabilities=np.asarray(local["probabilities"][:, 0]),
                                logits=np.asarray(local["logits"][:, 0]),
                                neural_hashes=_hashes(old), alive=np.asarray(old["alive"][0] > 0)))
        for history, rule in (("stationary", initial_rule), ("reversal", 1-initial_rule)):
            for name, mode in (("native", NATIVE), ("contact", CONTACT), ("sparse", SPARSE)):
                acquired, acquisition = _replay(flat, old, steps, rule, mode)
                now = probe(flat, acquired, grid, position, action_key, probe_keys, rule)
                schedules = ((name, mode), ("reopened", NATIVE)) if name == "contact" else ((name, mode),)
                for arm, delay_mode in schedules:
                    delayed, continuation = _replay(flat, acquired, delay, rule, delay_mode)
                    later = probe(flat, delayed, grid, position, action_key, probe_keys, rule)
                    arms[arm + "_" + history].append(dict(
                        **_trace(acquisition, continuation), **encounter._stack([now, later])))
    row.update({"pre_" + name: value for name, value in encounter._stack(old_records).items()})
    for arm, records in arms.items():
        row.update({arm + "_" + name: value for name, value in encounter._stack(records).items()})
    return row


def _signed_local(probabilities, rule):
    placements = encounter.PLACEMENTS
    selection = np.arange(len(placements))
    return float(np.mean(probabilities[selection, placements[:, rule]].astype(np.float64) -
                         probabilities[selection, placements[:, 1-rule]].astype(np.float64)))


def summary_input(row):
    records = {}
    old = np.asarray([_signed_local(row["pre_probabilities"][rule], rule) for rule in (0, 1)])
    for arm in ARMS:
        value = encounter._arm(row, arm)
        target = [0, 1] if arm.endswith("_stationary") else [1, 0]
        local = np.asarray([[_signed_local(prob, rule) for prob in value["probe_probabilities"][initial]]
                            for initial, rule in enumerate(target)])
        food, poison = value["grid_food"], value["grid_poison"]
        contact = food | poison
        first = np.argmax(contact, axis=-1)[..., None]
        first_food = np.take_along_axis(food, first, axis=-1)[..., 0]
        first_poison = np.take_along_axis(poison, first, axis=-1)[..., 0]
        records[arm] = dict(local_S=local, solo_Y=first_food.astype(int)-first_poison.astype(int),
                            first_food=first_food, first_poison=first_poison, no_contact=~contact.any(axis=-1),
                            exposure_survival=value["probe_alive"], solo_survival=value["grid_alive_after"][..., -1])
    return dict(base_index=int(row["base_index"]), pre_old=old, arms=records,
                sparse_feedback_overlap=float(np.mean(row["sparse_offsets"] == 3)))
