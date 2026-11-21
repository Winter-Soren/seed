"""E118 shelter selection and paired parent–descendant evaluation."""
from functools import lru_cache
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import cold_donor
from seed import seed
from seed.ecology import EcologyV1
from seed.persistence import PersistenceV1


PROTOCOL = "shelter-evolution-v1"
ECOLOGY = EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02)
PERSISTENCE = PersistenceV1()
ARMS = ("parent", "descendant")
EVENTS = ("food", "poison", "alive_before", "alive_after", "sheltered")


@lru_cache
def _unravel(c):
    return ravel_pytree(seed.init_genome(c, jax.random.PRNGKey(0)))[1]


def shelter_score(initial_sheltered, sheltered, alive_after):
    """Reward earlier living arrival among outside births, with zero for nonarrival."""
    steps = sheltered.shape[0]
    turns = jnp.arange(1, steps + 1, dtype=jnp.int32)[:, None]
    arrival = jnp.where(sheltered & alive_after, turns, steps + 1).min(0)
    outside = ~initial_sheltered
    return jnp.where(outside, steps + 1 - arrival, 0).sum() / jnp.maximum(outside.sum(), 1)


def make_evaluator(c=seed.Cfg(), worlds=4):
    """Use the existing ES interface with shelter arrival as its fitness."""
    if isinstance(worlds, bool) or not isinstance(worlds, Integral) or worlds < 2 or worlds % 2:
        raise ValueError("training requires a positive even number of worlds")
    unravel = _unravel(c)
    rules = jnp.arange(worlds) % 2

    def life(g, world_key, development_key, rule):
        result = seed.rollout(g, world_key, development_key, rule, c, plastic=True,
                              diagnostics=True, ecology=ECOLOGY, persistence=PERSISTENCE)
        ecology = result["ecology"]
        reward = shelter_score(ecology["initial_sheltered"], ecology["sheltered"],
                               result["events"]["alive_after"] > 0)
        return dict(fitness=reward, counts=result["counts"], survived=result["survived"])

    def candidate(flat, world_keys, development_keys):
        return jax.vmap(life, in_axes=(None, 0, 0, 0))(
            unravel(flat), world_keys, development_keys, rules)

    @jax.jit
    def evaluate(population, world_keys, development_keys):
        return jax.vmap(candidate, in_axes=(0, None, 0))(population, world_keys, development_keys)

    return evaluate, unravel


def evaluation_keys(master_seed, index):
    """One world and one developmental stream per independent paired family."""
    for value in (master_seed, index):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < 2 ** 32:
            raise ValueError("evaluation seed and index must be uint32 integers")
    return np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(int(master_seed)), int(index)), 2))


def evaluate_pair(parent_flat, descendant_flat, master_seed, index, c=seed.Cfg()):
    """Record both genotypes on matched worlds and developmental random streams."""
    keys = evaluation_keys(master_seed, index)
    rule = int(index) % 2
    _, positions = seed.init_world(keys[0], c)
    row = dict(keys=keys, rule=np.asarray(rule, np.int32), steps=np.asarray(c.T, np.int32),
               agents=np.asarray(c.M, np.int32), initial_position=np.asarray(positions))
    for arm, flat in zip(ARMS, (parent_flat, descendant_flat)):
        flat = np.asarray(flat)
        if flat.shape != (seed.genome_size(c),) or flat.dtype.kind != "f" or not np.isfinite(flat).all():
            raise ValueError("parent and descendant must be finite flat rules genomes")
        result = seed.rollout(_unravel(c)(flat), keys[0], keys[1], rule, c, plastic=True,
                              diagnostics=True, ecology=ECOLOGY, persistence=PERSISTENCE,
                              _return_weights=True)
        result = jax.tree.map(np.asarray, result)
        ecology = result["ecology"]
        for name in EVENTS:
            source = ecology if name == "sheltered" else result["events"]
            if not np.isin(source[name], (0, 1)).all():
                raise ValueError("lifetime returned nonbinary events")
            row[arm + "_" + name] = np.asarray(source[name], bool)
        for name in ("shelter", "initial_sheltered", "energy_lost"):
            row[arm + "_" + name] = np.asarray(ecology[name])
        row[arm + "_first_entry"] = cold_donor.first_entry(
            row[arm + "_initial_sheltered"], row[arm + "_sheltered"], row[arm + "_alive_after"])
        row[arm + "_fitness"] = result["fitness"]
        weights = result["_weight_state"]
        if not np.isfinite(weights["birth_W"]).all() or not np.isfinite(weights["terminal_W"]).all():
            raise ValueError("lifetime returned nonfinite weights")
        row[arm + "_birth_connections"] = np.count_nonzero(weights["mask"], axis=(1, 2))
        row[arm + "_birth_sha256"] = np.stack([
            np.frombuffer(hashlib.sha256(agent.tobytes()).digest(), np.uint8)
            for agent in weights["birth_W"]])
        for name in ("active", "remembered", "tactic_attempted"):
            if not np.isin(result["persistence"][name], (0, 1)).all():
                raise ValueError("lifetime returned nonbinary persistence events")
            row[arm + "_persistence_" + name] = np.asarray(result["persistence"][name], bool)
    validate_pair(row, c)
    return row


def validate_pair(row, c=None):
    """Reject invalid events, pairing and denominators before accepting evidence."""
    steps, agents = (cold_donor._scalar(row, name, integer=True) for name in ("steps", "agents"))
    rule = cold_donor._scalar(row, "rule", integer=True)
    if steps < 1 or agents < 1 or rule not in (0, 1):
        raise ValueError("invalid dimensions or rule")
    if c is not None and (steps, agents) != (c.T, c.M):
        raise ValueError("evaluation dimensions differ from configuration")
    keys = cold_donor._array(row, "keys", (2, 2), integer=True)
    if keys.dtype != np.uint32 or np.array_equal(keys[0], keys[1]):
        raise ValueError("world and development keys must be distinct uint32 pairs")
    position = cold_donor._array(row, "initial_position", (agents, 2), integer=True)
    for arm in ARMS:
        prefix = arm + "_"
        food, poison, before, after, sheltered = [
            cold_donor._array(row, prefix + name, (steps, agents), binary=True) for name in EVENTS]
        initial = cold_donor._array(row, prefix + "initial_sheltered", (agents,), binary=True)
        if (not before[0].all() or np.any(food & poison) or np.any((food | poison | sheltered) & ~before)
                or np.any(after & ~before) or not np.array_equal(before[1:], after[:-1])):
            raise ValueError("events contain impossible death or encounter timing")
        arrival = cold_donor._array(row, prefix + "first_entry", (agents,), integer=True)
        if not np.array_equal(arrival, cold_donor.first_entry(initial, sheltered, after)):
            raise ValueError("first entry differs from living arrival events")
        shelter = np.asarray(row[prefix + "shelter"])
        if (shelter.ndim != 2 or shelter.shape[0] != shelter.shape[1]
                or not np.isin(shelter, (0, 1)).all()
                or np.any(position < 0) or np.any(position >= shelter.shape[0])):
            raise ValueError("invalid shelter map or birth positions")
        if c is not None and shelter.shape != (c.G, c.G):
            raise ValueError("shelter map differs from configuration")
        if not np.array_equal(initial, shelter[position[:, 0], position[:, 1]]):
            raise ValueError("inside births differ from shelter and positions")
        fitness = cold_donor._scalar(row, prefix + "fitness")
        expected = (food.sum(0, dtype=np.int64) - poison.sum(0, dtype=np.int64)).mean()
        if fitness != expected:
            raise ValueError("food fitness differs from saved encounters")
        loss = cold_donor._array(row, prefix + "energy_lost", (steps, agents))
        if np.any(loss < 0) or np.any(loss[~before] != 0):
            raise ValueError("invalid additional energy loss")
        connections = cold_donor._array(row, prefix + "birth_connections", (agents,), integer=True)
        if np.any(connections < 0):
            raise ValueError("invalid birth connection count")
        hashes = cold_donor._array(row, prefix + "birth_sha256", (agents, 32), integer=True)
        if hashes.dtype != np.uint8:
            raise ValueError("birth hashes must be SHA256 bytes")
        for name in ("active", "remembered", "tactic_attempted"):
            flags = cold_donor._array(row, prefix + "persistence_" + name, (steps, agents), binary=True)
            if np.any(flags & ~before):
                raise ValueError("persistence activity after death")
    for name in ("shelter", "initial_sheltered"):
        if not np.array_equal(row["parent_" + name], row["descendant_" + name]):
            raise ValueError("parent and descendant starting worlds differ")


def summarize(rows):
    """Pooled outside-born latency with paired family-cluster uncertainty."""
    rows = list(rows)
    if not rows:
        raise ValueError("at least one completed family is required")
    for row in rows:
        validate_pair(row)
    steps, agents = int(rows[0]["steps"]), int(rows[0]["agents"])
    if any((int(row["steps"]), int(row["agents"])) != (steps, agents) for row in rows):
        raise ValueError("family dimensions must match")
    keys = [key.tobytes() for row in rows for key in row["keys"]]
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation families must use distinct keys")
    outside = ~np.stack([row["parent_initial_sheltered"] for row in rows]).astype(bool)
    counts = outside.sum(1)
    by_arm = {}
    latencies, survivals = {}, {}
    for arm in ARMS:
        arrival = np.stack([row[arm + "_first_entry"] for row in rows])
        alive = np.stack([row[arm + "_alive_after"] for row in rows]).astype(bool)
        sheltered = np.stack([row[arm + "_sheltered"] for row in rows]).astype(bool)
        latencies[arm] = (arrival * outside).sum(1)
        survivals[arm] = alive[:, -1].mean(1)
        by_arm[arm] = dict(
            latency=cold_donor._ratio_se(latencies[arm], counts),
            survival=cold_donor._mean_se(survivals[arm]),
            fitness=cold_donor._mean_se([row[arm + "_fitness"] for row in rows]),
            living_arrivals=int(((arrival <= steps) & outside).sum()),
            before_cold_arrivals=int(((arrival <= steps // 2) & outside).sum()),
            cold_sheltered_fraction=cold_donor._mean_se(
                (sheltered[:, steps // 2:] & alive[:, steps // 2:]).mean((1, 2))),
            deaths=int((~alive[:, -1]).sum()),
            birth_connections=cold_donor._mean_se([
                np.asarray(row[arm + "_birth_connections"]).mean() for row in rows]))
    latency = cold_donor._ratio_se(latencies["parent"] - latencies["descendant"], counts)
    survival = cold_donor._mean_se(survivals["descendant"] - survivals["parent"])
    decided = latency["se"] is not None and survival["se"] is not None
    latency_passed = bool(latency["mean"] > 2 * latency["se"]) if decided else None
    survival_passed = bool(survival["mean"] >= -2 * survival["se"]) if decided else None
    return dict(families=len(rows), agents_per_family=agents, assigned_agents=len(rows) * agents,
                outside_born_agents=int(outside.sum()),
                rule_counts={str(rule): sum(int(row["rule"]) == rule for row in rows) for rule in (0, 1)},
                **by_arm, latency_improvement=latency, survival_difference=survival,
                gate=dict(passed=latency_passed and survival_passed if decided else None,
                          latency_passed=latency_passed, survival_passed=survival_passed))
