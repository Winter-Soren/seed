"""E119 selection with a training-only archive passed between generations."""
from functools import partial
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, shelter_evolution as shelter
from seed import seed


PROTOCOL = "inherited-evolution-v1"
ECOLOGY, PERSISTENCE = shelter.ECOLOGY, shelter.PERSISTENCE
ARMS = ("founder_empty", "parent_previous", "parent_latest", "child_previous", "child_latest", "child_empty")
CONTRASTS = {
    "child_vs_parent": ("parent_previous", "child_latest"),
    "child_vs_founder": ("founder_empty", "child_latest"),
    "inherited_vs_empty": ("child_empty", "child_latest"),
    "genotype_same_memory": ("parent_latest", "child_latest"),
    "latest_lessons": ("child_previous", "child_latest"),
    "parent_lessons": ("parent_previous", "parent_latest"),
}
evaluation_keys = shelter.evaluation_keys


def empty_archive(c=seed.Cfg()):
    _, sensors, outputs, _, _ = seed.dims(c, ecology=ECOLOGY)
    return np.zeros((outputs, sensors), np.float32)


def _inputs(flat, archive, c):
    flat, archive = np.asarray(flat), np.asarray(archive)
    if flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("a finite float32 rules genome is required")
    if (archive.shape != empty_archive(c).shape or archive.dtype != np.float32
            or not np.isfinite(archive).all() or np.any(np.abs(archive) > c.wmax)):
        raise ValueError("archive must be finite bounded float32 sensory-to-output weights")
    return flat, archive


def _world_count(worlds):
    if isinstance(worlds, bool) or not isinstance(worlds, Integral) or worlds < 2 or worlds % 2:
        raise ValueError("donor and training world counts must be positive and even")


def donor_keys(master_seed, round_index, worlds=4):
    """A domain-separated donor stream, independent of the ES state key."""
    evaluation_keys(master_seed, round_index)
    _world_count(worlds)
    key = jax.random.fold_in(jax.random.PRNGKey(int(master_seed)), 0xE119)
    key = jax.random.fold_in(key, int(round_index))
    return np.asarray(jax.random.split(key, worlds * 2)).reshape(worlds, 2, 2)


@partial(jax.jit, static_argnames=("c",))
def _life(flat, archive, world_key, development_key, rule, c):
    genome = shelter._unravel(c)(flat)
    _, sensors, outputs, neurons, _ = seed.dims(c, ecology=ECOLOGY)
    birth, mask = jax.vmap(lambda key: seed.develop(genome, key, c, ecology=ECOLOGY))(
        jax.random.split(development_key, c.M))
    fresh = jnp.clip(birth, -c.wmax, c.wmax)
    block = fresh[:, neurons - outputs:, :sensors]
    overlay = archive * mask[:, neurons - outputs:, :sensors]
    # Preserve signed zeros when the archive contributes nothing.
    initial_block = jnp.where(overlay != 0, jnp.clip(block + overlay, -c.wmax, c.wmax), block)
    initial = fresh.at[:, neurons - outputs:, :sensors].set(initial_block)
    result = seed.rollout(genome, world_key, development_key, rule, c, plastic=True,
                          diagnostics=True, ecology=ECOLOGY, persistence=PERSISTENCE,
                          _initial_weights=initial, _return_weights=True)
    result["_fresh_W"] = fresh
    return result


def make_evaluator(c=seed.Cfg(), worlds=4):
    """Return the existing ES interface with one extra dynamic archive argument."""
    _world_count(worlds)
    _, sensors, outputs, neurons, _ = seed.dims(c, ecology=ECOLOGY)
    rules = jnp.arange(worlds) % 2

    def life(flat, archive, wk, dk, rule):
        result = _life(flat, archive, wk, dk, rule, c)
        ecology, state = result["ecology"], result["_weight_state"]
        reward = shelter.shelter_score(ecology["initial_sheltered"], ecology["sheltered"],
                                       result["events"]["alive_after"] > 0)
        return dict(fitness=reward, counts=result["counts"], survived=result["survived"],
                    candidate_birth=result["_fresh_W"][:, neurons - outputs:, :sensors],
                    candidate_initial=state["initial_W"][:, neurons - outputs:, :sensors],
                    candidate_mask=state["mask"][:, neurons - outputs:, :sensors].astype(bool))

    def candidate(flat, wkeys, dkeys, archive):
        return jax.vmap(life, in_axes=(None, None, 0, 0, 0))(flat, archive, wkeys, dkeys, rules)

    @jax.jit
    def evaluate(population, world_keys, development_keys, archive):
        return jax.vmap(candidate, in_axes=(0, None, 0, None))(
            population, world_keys, development_keys, archive)

    return evaluate, shelter._unravel(c)


def _hashes(weights):
    return np.stack([np.frombuffer(hashlib.sha256(np.ascontiguousarray(w).tobytes()).digest(), np.uint8)
                     for w in weights])


def _record(result, c):
    result = jax.tree.map(np.asarray, result)
    ecology, state = result["ecology"], result["_weight_state"]
    _, sensors, outputs, neurons, _ = seed.dims(c, ecology=ECOLOGY)
    row = {name: np.asarray((ecology if name == "sheltered" else result["events"])[name], bool)
           for name in shelter.EVENTS}
    row.update({name: np.asarray(ecology[name]) for name in ("shelter", "initial_sheltered", "energy_lost")})
    row.update(fitness=np.asarray(result["fitness"]),
               first_entry=cold_donor.first_entry(row["initial_sheltered"], row["sheltered"], row["alive_after"]),
               birth_connections=np.count_nonzero(state["mask"], axis=(1, 2)),
               birth_sha256=_hashes(state["birth_W"]), initial_sha256=_hashes(state["initial_W"]),
               fresh_sha256=_hashes(result["_fresh_W"]), terminal_sha256=_hashes(state["terminal_W"]),
               fresh_sensor_weights=result["_fresh_W"][:, neurons - outputs:, :sensors],
               initial_sensor_weights=state["initial_W"][:, neurons - outputs:, :sensors],
               terminal_sensor_weights=state["terminal_W"][:, neurons - outputs:, :sensors],
               sensor_mask=state["mask"][:, neurons - outputs:, :sensors].astype(bool))
    for name in ("active", "remembered", "tactic_attempted"):
        row["persistence_" + name] = np.asarray(result["persistence"][name], bool)
    return row


def overlay_archive(fresh, mask, archive, bound):
    """NumPy counterpart used for the saved-evidence checks."""
    initial = np.asarray(fresh).copy()
    delta = archive * mask
    np.copyto(initial, np.clip(initial + delta, -bound, bound), where=delta != 0)
    return initial


def update_archive(archive, delta, mask, bound):
    """Average all assigned present edges, including dead donors and zero changes."""
    counts = np.asarray(mask, bool).sum(axis=(0, 1), dtype=np.int32)
    update = np.asarray(delta).sum(axis=(0, 1)) / np.maximum(counts, 1).astype(np.float32)
    return np.clip(archive + update, -bound, bound), counts


def donor_update(flat, old_archive, master_seed, round_index, c=seed.Cfg(), worlds=4):
    flat, old_archive = _inputs(flat, old_archive, c)
    keys = donor_keys(master_seed, round_index, worlds)
    rows = [_record(_life(flat, old_archive, wk, dk, index % 2, c), c)
            for index, (wk, dk) in enumerate(keys)]
    raw = dict(genome=flat.copy(), archive_before=old_archive.copy(), keys=keys,
               rules=np.arange(worlds, dtype=np.int32) % 2, round_index=np.asarray(round_index, np.int32),
               worlds=np.asarray(worlds, np.int32), agents=np.asarray(c.M, np.int32),
               steps=np.asarray(c.T, np.int32), weight_bound=np.asarray(c.wmax, np.float32),
               initial_positions=np.stack([np.asarray(seed.init_world(wk, c)[1]) for wk, _ in keys]))
    raw.update({"donor_" + name: np.stack([row[name] for row in rows]) for name in rows[0]})
    raw["donor_delta"] = raw["donor_terminal_sensor_weights"] - raw["donor_initial_sensor_weights"]
    raw["donor_mask"] = raw["donor_sensor_mask"].copy()
    raw["archive_after"], raw["donor_counts"] = update_archive(
        old_archive, raw["donor_delta"], raw["donor_mask"], c.wmax)
    validate_donors(raw, c)
    return raw


def _pair_row(row, parent, child):
    pair = {name: row[name] for name in ("keys", "rule", "steps", "agents", "initial_position")}
    for prefix, arm in (("parent_", parent), ("descendant_", child)):
        pair.update({prefix + name[len(arm) + 1:]: value for name, value in row.items()
                     if name.startswith(arm + "_")})
    return pair


def _validate_overlay(row, prefix, archive, agents, bound):
    archive = np.asarray(archive)
    if archive.ndim != 2 or not all(archive.shape) or archive.dtype != np.float32:
        raise ValueError("invalid archive layout")
    if not np.isfinite(archive).all() or np.any(np.abs(archive) > bound):
        raise ValueError("archive is nonfinite or unbounded")
    shape = (agents,) + archive.shape
    fresh, initial, terminal = [cold_donor._array(row, prefix + name + "_sensor_weights", shape)
                                for name in ("fresh", "initial", "terminal")]
    mask = cold_donor._array(row, prefix + "sensor_mask", shape, binary=True)
    if (any(value.dtype != np.float32 for value in (fresh, initial, terminal))
            or any(np.any(np.abs(value) > bound) or np.any(value[~mask] != 0)
                   for value in (fresh, initial, terminal))
            or not np.array_equal(initial, overlay_archive(fresh, mask, archive, bound))):
        raise ValueError("initial weights differ from the bounded masked archive overlay")
    for field in ("initial_sha256", "fresh_sha256", "terminal_sha256"):
        hashes = cold_donor._array(row, prefix + field, (agents, 32), integer=True)
        if hashes.dtype != np.uint8:
            raise ValueError("weight fingerprints must contain SHA256 bytes")


def validate_donors(raw, c=seed.Cfg()):
    _inputs(raw["genome"], raw["archive_before"], c)
    worlds = cold_donor._scalar(raw, "worlds", integer=True)
    _world_count(worlds)
    keys = cold_donor._array(raw, "keys", (worlds, 2, 2), integer=True)
    if keys.dtype != np.uint32 or len({key.tobytes() for key in keys.reshape(-1, 2)}) != worlds * 2:
        raise ValueError("donor worlds require unique world and birth keys")
    if (int(raw["steps"]) != c.T or int(raw["agents"]) != c.M or float(raw["weight_bound"]) != c.wmax
            or not np.array_equal(raw["rules"], np.arange(worlds) % 2)):
        raise ValueError("donor dimensions or food rules differ from the protocol")
    for index in range(worlds):
        row = dict(keys=keys[index], rule=np.asarray(index % 2), steps=raw["steps"], agents=raw["agents"],
                   initial_position=raw["initial_positions"][index])
        row.update({name: np.asarray(value)[index] for name, value in raw.items()
                    if name.startswith("donor_") and name not in ("donor_delta", "donor_mask", "donor_counts")})
        shelter.validate_pair(_pair_row(row, "donor", "donor"), c)
        _validate_overlay(row, "donor_", raw["archive_before"], c.M, c.wmax)
    if not np.array_equal(raw["donor_mask"], raw["donor_sensor_mask"]):
        raise ValueError("donor accounting mask differs from actual wiring")
    delta = raw["donor_terminal_sensor_weights"] - raw["donor_initial_sensor_weights"]
    if not np.array_equal(raw["donor_delta"], delta):
        raise ValueError("donor delta differs from last living minus actual initial weights")
    expected, counts = update_archive(raw["archive_before"], delta, raw["donor_mask"], c.wmax)
    if not np.array_equal(raw["archive_after"], expected) or not np.array_equal(raw["donor_counts"], counts):
        raise ValueError("archive update differs from all present donor edges")


def evaluate_world(arms, master_seed, index, c=seed.Cfg()):
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("provide exactly the six registered genotype and archive arms")
    keys = evaluation_keys(master_seed, index)
    row = dict(keys=keys, rule=np.asarray(index % 2, np.int32), steps=np.asarray(c.T, np.int32),
               agents=np.asarray(c.M, np.int32), weight_bound=np.asarray(c.wmax, np.float32),
               initial_position=np.asarray(seed.init_world(keys[0], c)[1]))
    for arm in ARMS:
        flat, archive = _inputs(*arms[arm], c)
        row.update({arm + "_" + name: value for name, value in
                    _record(_life(flat, archive, keys[0], keys[1], index % 2, c), c).items()})
        row[arm + "_genome"], row[arm + "_archive"] = flat.copy(), archive.copy()
    validate_world(row, c)
    return row


def validate_world(row, c=None):
    agents = cold_donor._scalar(row, "agents", integer=True)
    bound = cold_donor._scalar(row, "weight_bound")
    if bound <= 0:
        raise ValueError("weight bound must be positive")
    for arm in ARMS:
        shelter.validate_pair(_pair_row(row, "founder_empty", arm), c)
        _validate_overlay(row, arm + "_", row[arm + "_archive"], agents, bound)
        if c is not None:
            _inputs(row[arm + "_genome"], row[arm + "_archive"], c)
    for first, other in (("parent_previous", "parent_latest"), ("child_previous", "child_latest"),
                         ("child_previous", "child_empty")):
        for field in ("genome", "fresh_sensor_weights", "sensor_mask", "birth_sha256", "fresh_sha256"):
            if not np.array_equal(row[first + "_" + field], row[other + "_" + field]):
                raise ValueError("same-genome memory controls differ in their newborn wiring")
    for first, other in (("parent_previous", "child_previous"), ("parent_latest", "child_latest")):
        if not np.array_equal(row[first + "_archive"], row[other + "_archive"]):
            raise ValueError("same-archive genotype controls have different memories")
    if np.any(row["founder_empty_archive"]) or np.any(row["child_empty_archive"]):
        raise ValueError("fresh-start control archive must be empty")


def summarize(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("at least one complete family is required")
    for row in rows:
        validate_world(row)
    comparisons = {}
    by_arm = {}
    for name, (parent, child) in CONTRASTS.items():
        result = shelter.summarize([_pair_row(row, parent, child) for row in rows])
        by_arm[parent], by_arm[child] = result["parent"], result["descendant"]
        comparisons[name] = {key: result[key] for key in ("latency_improvement", "survival_difference", "gate")}
    metadata = {key: result[key] for key in
                ("families", "agents_per_family", "assigned_agents", "outside_born_agents", "rule_counts")}
    return dict(**metadata, **by_arm, contrasts=comparisons, gate=comparisons["child_vs_parent"]["gate"])
