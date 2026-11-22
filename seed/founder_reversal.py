"""E130: original founders/descendants in unassisted four-animal lifetimes.

Native ``seed.rollout`` supplies all events without a new world or controller.
Every genotype/condition replays both initial rules with the same world/action
and development keys. A reversal changes food labels before zero-indexed eating
step 150. No agent is replaced, selected for survival, or omitted from a bin.

Each arm's raw events have axes [initial_rule, step, original_agent]. Compact
outcomes have axes [initial_rule, fifty_step_bin, original_agent]. Q is the
first item's value (+1 food, -1 poison, 0 no contact including death). Counts
are all encounters, not first-item flags. Statistics belong to the runner.
"""
from functools import lru_cache
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed


PROTOCOL = "founder-reversal-v1"
CFG = seed.Cfg()
ARMS = ("gen0_stationary", "gen0_reversal", "gen200_stationary", "gen200_reversal")
KEY_ROLES = ("world_action", "development")
ASSIGNMENT_TAG = 0xE130
FLIP_STEP = 150
BIN_EDGES = (0, 50, 100, 150, 200, 250, 300)
EVENTS = ("food", "poison", "alive_before", "alive_after")


def _hash(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8).copy()


def _same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return actual.shape == expected.shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def _master(master):
    if isinstance(master, bool) or not isinstance(master, Integral) or not 0 <= master < 2 ** 32:
        raise ValueError("master seed must be a uint32 integer")
    return jax.random.PRNGKey(int(master))


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves the reserved founder-reversal stream")
    stream = jax.random.fold_in(_master(master), ASSIGNMENT_TAG)
    return np.asarray(jax.random.split(jax.random.fold_in(stream, int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if isinstance(count, bool) or count not in (4, 64):
        raise ValueError("E130 assigns four development or sixty-four final families")
    return dict(family_keys=np.stack([evaluation_keys(master, index) for index in range(count)]),
                initial_rules=np.asarray([0, 1], np.int32))


def _inputs(keys, index, c):
    keys = np.asarray(keys)
    if c != CFG:
        raise ValueError("E130 requires unchanged native Cfg: four animals and 300 turns")
    if (keys.shape != (2, 2) or keys.dtype != np.uint32 or _same(keys[0], keys[1])
            or isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG):
        raise ValueError("invalid independent world/action and development keys")
    return keys


@lru_cache(None)
def _unravel(c):
    return ravel_pytree(seed.init_genome(c, jax.random.PRNGKey(0)))[1]


def _genome(flat, c):
    flat = np.asarray(flat)
    if flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("original controller must keep its finite float32 genome encoding")
    return _unravel(c)(jnp.asarray(flat))


def evaluate_world(start_flat, final_flat, keys, index, c=CFG):
    """Run eight ordinary lives, paired in world, birth, actions and initial rule."""
    keys = _inputs(keys, index, c)
    genomes = {"gen0": _genome(start_flat, c), "gen200": _genome(final_flat, c)}
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(),
        birth_keys=np.asarray(jax.random.split(jnp.asarray(keys[1]), c.M)),
        initial_rules=np.asarray([0, 1], np.int32), bin_edges=np.asarray(BIN_EDGES, np.int32),
        flip_step=np.asarray(FLIP_STEP, np.int32),
        gen0_genome_sha256=_hash(start_flat), gen200_genome_sha256=_hash(final_flat))
    for stage, genome in genomes.items():
        for condition in ("stationary", "reversal"):
            lives = []
            for rule in (0, 1):
                result = seed.rollout(genome, jnp.asarray(keys[0]), jnp.asarray(keys[1]), jnp.asarray(rule, jnp.int32),
                    c, plastic=True, record=False, predictor_plastic=True, structural=False,
                    flip_step=FLIP_STEP if condition == "reversal" else None, diagnostics=True)
                result = jax.tree.map(np.asarray, result)
                for event in EVENTS:
                    values = result["events"][event]
                    if values.shape != (c.T, c.M) or not np.isin(values, (0, 1)).all():
                        raise ValueError("native rollout returned invalid binary events")
                lives.append({**{event: result["events"][event].astype(bool) for event in EVENTS},
                    "native_counts": result["counts"], "native_fitness": result["fitness"],
                    "native_survived": result["survived"]})
            row.update({stage + "_" + condition + "_" + name: np.stack([life[name] for life in lives]) for name in lives[0]})
    validate_world(row, c)
    return row


def _arm(row, arm):
    prefix = arm + "_"
    return {name[len(prefix):]: np.asarray(value) for name, value in row.items() if name.startswith(prefix)}


def _validate_events(events):
    arrays = {name: np.asarray(events[name]) for name in EVENTS}
    if any(value.shape != (2, CFG.T, CFG.M) or value.dtype != np.bool_ for value in arrays.values()):
        raise ValueError("events must retain both rules, all 300 turns and all four original animals")
    food, poison, before, after = (arrays[name] for name in EVENTS)
    if (np.any(food & poison) or np.any((food | poison) & ~before) or not np.all(before[:, 0])
            or np.any(after & ~before) or not np.array_equal(before[:, 1:], after[:, :-1])):
        raise ValueError("event stream contains conflicting meals, post-death events or replacement")
    return food, poison, before, after


def bin_events(events):
    """First-item values for every assigned animal; no conditional denominators."""
    food, poison, before, after = _validate_events(events)
    shape = (2, len(BIN_EDGES) - 1, 50, CFG.M)
    food, poison, before, after = (value.reshape(shape) for value in (food, poison, before, after))
    contact = food | poison
    any_contact = contact.any(axis=2)
    first = contact.argmax(axis=2)[:, :, None, :]
    first_food = np.take_along_axis(food, first, axis=2)[:, :, 0] & any_contact
    first_poison = np.take_along_axis(poison, first, axis=2)[:, :, 0] & any_contact
    return dict(Q_by_rule=first_food.astype(np.int8) - first_poison.astype(np.int8),
        food_count_by_rule=food.sum(axis=2, dtype=np.int16), poison_count_by_rule=poison.sum(axis=2, dtype=np.int16),
        no_contact_by_rule=~any_contact, alive_start_by_rule=before[:, :, 0].copy(),
        alive_end_by_rule=after[:, :, -1].copy(), final_survival_by_rule=after[:, -1, -1].copy())


def validate_world(row, c=CFG):
    keys = _inputs(row["keys"], int(row["base_index"]), c)
    for name, expected in (("initial_rules", np.asarray([0, 1], np.int32)),
                           ("bin_edges", np.asarray(BIN_EDGES, np.int32)), ("flip_step", np.asarray(FLIP_STEP, np.int32))):
        if not _same(row[name], expected):
            raise ValueError("changed the assigned rules, bins or switch time")
    if not _same(row["birth_keys"], np.asarray(jax.random.split(jnp.asarray(keys[1]), c.M))):
        raise ValueError("original animal slots do not have the paired development keys")
    for stage in ("gen0", "gen200"):
        digest = np.asarray(row[stage + "_genome_sha256"])
        if digest.shape != (32,) or digest.dtype != np.uint8:
            raise ValueError("missing original source-genome byte identity")
    for arm in ARMS:
        values = _arm(row, arm)
        food, poison, before, after = _validate_events(values)
        counts = np.stack([np.stack([event[:, start:stop].sum(axis=(1, 2), dtype=np.int32) for event in (food, poison)], axis=-1)
                           for start, stop in ((0, FLIP_STEP), (FLIP_STEP, c.T))], axis=1).astype(np.float32)
        fitness = ((food.astype(np.int32) - poison.astype(np.int32)).sum(axis=(1, 2)) / c.M).astype(np.float32)
        survival = after[:, -1].mean(axis=1).astype(np.float32)
        for name, expected in (("native_counts", counts), ("native_fitness", fitness), ("native_survived", survival)):
            if not _same(values[name], expected):
                raise ValueError("native aggregate differs from all-assigned individual events")
    for stage in ("gen0", "gen200"):
        stationary, reversal = (_arm(row, stage + "_" + condition) for condition in ("stationary", "reversal"))
        for event in EVENTS:
            if not _same(stationary[event][:, :FLIP_STEP], reversal[event][:, :FLIP_STEP]):
                raise ValueError("paired histories differ before the hidden switch")
        if (not _same(stationary["alive_before"][:, FLIP_STEP], reversal["alive_before"][:, FLIP_STEP])
                or not _same(stationary["food"][:, FLIP_STEP], reversal["poison"][:, FLIP_STEP])
                or not _same(stationary["poison"][:, FLIP_STEP], reversal["food"][:, FLIP_STEP])):
            raise ValueError("switch did not first relabel the same eating action at turn150")


def summary_input(row):
    validate_world(row)
    compact = {name: np.asarray(row[name]).copy() for name in
        ("base_index", "keys", "birth_keys", "initial_rules", "bin_edges", "flip_step", "gen0_genome_sha256", "gen200_genome_sha256")}
    for arm in ARMS:
        compact.update({arm + "_" + name: value for name, value in bin_events(_arm(row, arm)).items()})
    return compact
