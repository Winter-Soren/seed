"""Paired lifetime transfer with a fixed genome and explicit newborn cohorts.

Archives contain only sensor-to-motor weight changes. Their neuron identities
match across siblings, while hidden neurons have no learned alignment. Callers
must keep development/training archives separate from held-out evaluations.
Donors export their last update while alive. The engine's delayed learning rule
does not credit the action that causes death after that agent has died.
"""
import hashlib
from dataclasses import asdict
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import seed


def _key(value):
    try:
        data = np.asarray(jax.random.key_data(value))
    except (TypeError, ValueError) as error:
        raise ValueError("keys must be JAX threefry keys") from error
    if data.shape != (2,) or data.dtype != np.uint32:
        raise ValueError("keys must contain two uint32 words")
    return jnp.asarray(data)


def _inputs(g, wkeys, rules, c, ecologies, on_life):
    if on_life is not None and not callable(on_life):
        raise ValueError("on_life must be callable")
    world_keys = tuple(_key(value) for value in wkeys)
    if not world_keys or len(rules) != len(world_keys):
        raise ValueError("provide one rule for each lifetime")
    if len({np.asarray(key).tobytes() for key in world_keys}) != len(world_keys):
        raise ValueError("each lifetime requires a fresh world key")
    if any(not isinstance(rule, Integral) or isinstance(rule, (bool, np.bool_))
           or rule not in (0, 1) for rule in rules):
        raise ValueError("rules must be integers zero or one")
    ecologies = (None,) * len(world_keys) if ecologies is None else tuple(ecologies)
    if len(ecologies) != len(world_keys):
        raise ValueError("provide one ecology setting for each lifetime")
    dimensions = [seed.dims(c, ecology=ecology) for ecology in ecologies]
    if any(shape != dimensions[0] for shape in dimensions):
        raise ValueError("transfer requires one sensor layout across all lifetimes")
    if not np.isfinite(c.wmax) or c.wmax <= 0:
        raise ValueError("wmax must be finite and positive")
    K = dimensions[0][4]
    shapes = dict(conn=(K, K), wscale=(K, K), hebb=(K, K, 4),
                  log_eta=(), mod=(3,), log_plr=())
    if not isinstance(g, dict) or set(g) != set(shapes):
        raise ValueError("transfer requires a rules genome")
    for name, shape in shapes.items():
        value = np.asarray(g[name])
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"invalid genome array {name}")
    genome = {name: jnp.asarray(value) for name, value in g.items()}
    return genome, world_keys, tuple(int(rule) for rule in rules), ecologies, dimensions[0]


def _birth(g, dkey, c, ecology):
    weights, mask = jax.vmap(lambda key: seed.develop(g, key, c, ecology=ecology))(
        jax.random.split(dkey, c.M))
    weights, mask = np.asarray(weights), np.asarray(mask)
    if not np.isfinite(weights).all():
        raise ValueError("developed weights must be finite")
    return weights, mask


def _life(g, wkey, dkey, rule, c, ecology, plastic, *, weights=None, alive=None):
    if weights is not None:
        birth, mask = _birth(g, dkey, c, ecology)
        weights = np.asarray(weights)
        if (weights.shape != birth.shape or weights.dtype.kind != "f"
                or not np.isfinite(weights).all() or np.any(np.abs(weights) > c.wmax)
                or np.any(weights[mask == 0] != 0)):
            raise ValueError("transferred weights must be finite, bounded and match the birth mask")
    if alive is not None:
        alive = np.asarray(alive)
        if alive.shape != (c.M,) or not np.isin(alive, (0, 1)).all():
            raise ValueError("eligibility must be a binary value per original agent")
    result = seed.rollout(g, wkey, dkey, rule, c, plastic=plastic, diagnostics=True,
                          ecology=ecology, _initial_weights=weights,
                          _initial_alive=alive, _return_weights=True)
    result = jax.tree.map(np.asarray, result)
    state = result.pop("_weight_state")
    if not np.isfinite(state["terminal_W"]).all():
        raise ValueError("terminal donor weights must be finite")
    result["weights"] = {
        name + "_sha256": [hashlib.sha256(agent.tobytes()).hexdigest() for agent in state[name]]
        for name in ("birth_W", "initial_W", "terminal_W")}
    change = state["terminal_W"] - state["initial_W"]
    result["weights"].update(
        changed_connections=np.count_nonzero(change, axis=(1, 2)),
        mean_abs_change=np.mean(np.abs(change), axis=(1, 2)),
        final_alive=state["alive"].copy())
    return result, state


def _record(index, wkey, dkey, rule, ecology, c):
    return dict(life=index, agent_ids=np.column_stack((np.full(c.M, index), np.arange(c.M))),
                world_key=np.asarray(wkey), development_key=np.asarray(dkey), rule=rule,
                ecology=None if ecology is None else dict(version=ecology.version, **asdict(ecology)))


def _emit(row, lives, on_life):
    if on_life is None:
        lives.append(row)
    else:
        on_life(row)


def _archive_start(g, c, ecology, dimensions, wkeys, dkeys, saved):
    _, D, O, _, _ = dimensions
    digest = hashlib.sha256()
    for name in sorted(g):
        value = np.asarray(g[name])
        digest.update(name.encode() + value.dtype.str.encode() + str(value.shape).encode()
                      + value.tobytes())
    metadata = dict(protocol="lineage-v1", state_version=1,
                    method="sensor-motor-mean-delta-v1", genome_sha256=digest.hexdigest(),
                    layout=dict(sensors=D, motors=O, hidden=c.H, hidden_types=c.n_htypes,
                                view_radius=c.R, weight_bound=float(c.wmax),
                                ecology_version=None if ecology is None else ecology.version))
    empty_keys = np.empty((0, 2), np.uint32)
    if saved is None:
        return metadata, np.zeros((O, D), np.float32), 0, empty_keys, empty_keys.copy()
    required = set(metadata) | {"archive", "next_life", "used_world_keys", "used_development_keys"}
    if not isinstance(saved, dict) or set(saved) != required:
        raise ValueError("invalid collective archive checkpoint fields")
    if any(saved[name] != value for name, value in metadata.items()):
        raise ValueError("archive protocol, genome, method and sensor layout must match")
    next_life = saved["next_life"]
    if not isinstance(next_life, Integral) or isinstance(next_life, (bool, np.bool_)) or next_life < 1:
        raise ValueError("archive next_life must count the completed cohorts")
    archive = np.asarray(saved["archive"])
    if (archive.shape != (O, D) or archive.dtype.kind != "f"
            or not np.isfinite(archive).all() or np.any(np.abs(archive) > c.wmax)):
        raise ValueError("archive weights must be finite, bounded and match the sensor layout")
    history = []
    for name, incoming in (("used_world_keys", wkeys), ("used_development_keys", dkeys)):
        keys = np.asarray(saved[name])
        if keys.shape != (next_life, 2) or keys.dtype != np.uint32:
            raise ValueError("archive key history must match its completed cohort count")
        seen = {key.tobytes() for key in keys}
        if len(seen) != next_life or any(np.asarray(key).tobytes() in seen for key in incoming):
            raise ValueError("continued cohorts require unused world and development keys")
        history.append(keys.copy())
    return metadata, archive.copy(), int(next_life), *history


def _archive_checkpoint(metadata, archive, next_life, wkeys, dkeys):
    return dict(metadata, layout=metadata["layout"].copy(), archive=archive.copy(),
                next_life=next_life, used_world_keys=wkeys.copy(),
                used_development_keys=dkeys.copy())


def same_individual(g, wkeys, dkey, rules, c, *, ecologies=None, plastic=True, on_life=None):
    """Give first-life survivors matched inherited-weight and fresh-weight lives.

    All original slots remain in outcomes. Agents dead after the first life begin
    both second arms with zero energy and eligibility; no survivor renormalization.
    Only W transfers. Energy, position, activations and prediction weights reset.
    With on_life, completed records are streamed and the returned lives is empty.
    """
    g, wkeys, rules, ecologies, _ = _inputs(g, wkeys, rules, c, ecologies, on_life)
    if len(wkeys) != 2:
        raise ValueError("same-individual comparison requires exactly two lifetimes")
    dkey = _key(dkey)
    first, state = _life(g, wkeys[0], dkey, rules[0], c, ecologies[0], plastic)
    eligibility = state["alive"].copy()
    lives = []
    first_record = _record(0, wkeys[0], dkey, rules[0], ecologies[0], c)
    first_record["first"] = first
    _emit(first_record, lives, on_life)
    inherited, _ = _life(g, wkeys[1], dkey, rules[1], c, ecologies[1], plastic,
                         weights=state["terminal_W"], alive=eligibility)
    fresh, _ = _life(g, wkeys[1], dkey, rules[1], c, ecologies[1], plastic,
                     alive=eligibility)
    second_record = _record(1, wkeys[1], dkey, rules[1], ecologies[1], c)
    second_record.update(inherited=inherited, fresh=fresh, eligibility=eligibility,
                         agent_ids=np.column_stack((np.zeros(c.M, int), np.arange(c.M))))
    _emit(second_record, lives, on_life)
    return dict(protocol="lineage-v1", mode="same-individual", config=c._asdict(),
                genome=jax.tree.map(np.asarray, g), lives=lives, completed_lives=2)


def collective_lives(g, wkeys, dkeys, rules, c, *, ecologies=None, plastic=True,
                     on_life=None, archive_state=None):
    """Run successive newborn cohorts with a shared sensory-to-action archive.

    Every life has a paired fresh-weight control with identical world and birth
    keys. Each archive update averages terminal-minus-initial changes over donors
    possessing that edge, including dead donors and zero changes. Missing edges
    are excluded. Subtracting actual initial weights avoids recounting inheritance.
    With on_life, records are streamed and lives is empty in the returned result.
    archive_state accepts a prior result's compact checkpoint and continues its
    cohort identities. Each completed record also includes a checkpoint. Changes
    to world pressures are allowed while the genome, learning method and sensor
    layout remain fixed. Previously consumed world and development keys are rejected.
    """
    g, wkeys, rules, ecologies, dimensions = _inputs(g, wkeys, rules, c, ecologies, on_life)
    _, D, O, n, _ = dimensions
    dkeys = tuple(_key(value) for value in dkeys)
    if len(dkeys) != len(wkeys):
        raise ValueError("provide one development key for each cohort")
    if len({np.asarray(key).tobytes() for key in dkeys}) != len(dkeys):
        raise ValueError("newborn cohorts require fresh development keys")
    metadata, archive, start_life, old_wkeys, old_dkeys = _archive_start(
        g, c, ecologies[0], dimensions, wkeys, dkeys, archive_state)
    used_wkeys = np.concatenate((old_wkeys, np.asarray(wkeys)))
    used_dkeys = np.concatenate((old_dkeys, np.asarray(dkeys)))
    lives = []
    for index, (wkey, dkey, rule, ecology) in enumerate(zip(wkeys, dkeys, rules, ecologies), start_life):
        birth, mask = _birth(g, dkey, c, ecology)
        fresh_weights = np.clip(birth, -c.wmax, c.wmax)
        inherited_weights = fresh_weights.copy()
        sensor_motor = inherited_weights[:, n - O:, :D]
        delta = archive * mask[:, n - O:, :D]
        # An empty archive must preserve even the signed zeros of the fresh brain.
        np.copyto(sensor_motor, np.clip(sensor_motor + delta, -c.wmax, c.wmax), where=delta != 0)
        inherited, state = _life(g, wkey, dkey, rule, c, ecology, plastic,
                                 weights=inherited_weights)
        fresh, _ = _life(g, wkey, dkey, rule, c, ecology, plastic, weights=fresh_weights)
        donor_delta = (state["terminal_W"] - state["initial_W"])[:, n - O:, :D]
        donor_mask = state["mask"][:, n - O:, :D]
        donor_counts = donor_mask.sum(axis=0)
        update = donor_delta.sum(axis=0) / np.maximum(donor_counts, 1)
        row = _record(index, wkey, dkey, rule, ecology, c)
        row.update(inherited=inherited, fresh=fresh, archive_before=archive.copy(),
                   donor_delta=donor_delta.copy(), donor_mask=donor_mask.copy(),
                   donor_counts=donor_counts)
        archive = np.clip(archive + update, -c.wmax, c.wmax)
        row["archive_after"] = archive.copy()
        row["archive_state"] = _archive_checkpoint(
            metadata, archive, index + 1, used_wkeys[:index + 1], used_dkeys[:index + 1])
        _emit(row, lives, on_life)
    return dict(protocol="lineage-v1", mode="collective", config=c._asdict(),
                genome=jax.tree.map(np.asarray, g), lives=lives,
                completed_lives=len(wkeys), final_archive=archive.copy(),
                archive_state=_archive_checkpoint(metadata, archive, start_life + len(wkeys),
                                                  used_wkeys, used_dkeys))
