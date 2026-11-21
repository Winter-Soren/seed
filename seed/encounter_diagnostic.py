"""E127: guided physical encounters and discarded, paired policy readouts.

The chamber is a declared presentation apparatus; only the fresh-grid readout
regrows consumed items. Neural arithmetic and ordinary transition expressions
are retained from seed.rollout. No training or shared-engine mutation occurs.
"""
from functools import lru_cache
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct, seed


PROTOCOL = "encounter-diagnostic-v1"
CFG = seed.Cfg(M=1)
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_frozen", "direct_stationary", "direct_reversal")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe")
CONTACTS = (0, 2, 8, 16)
GRID_STEPS = 50
ASSIGNMENT_TAG = 0xE127
NEURAL = ("W", "x", "Wp", "mask")
BODY = ("grid", "pos", "energy", "alive", "sig", "denergy", "key", "t", "counts")
LOCAL_FIELDS = ("p_food", "p_poison", "approach", "other", "S")
GRID_FIELDS = ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")
PLACEMENTS = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
OPPOSITE = np.asarray([0, 2, 1, 4, 3], np.int32)
# Five positive float32 numerators: four additions have relative error gamma4;
# rounding each quotient contributes u. Hence the exact stored mass differs from
# one by at most (gamma4 + u)/(1 - gamma4). Check it in float64, with a small
# binary64 accumulation margin; a second float32 sum adds avoidable roundoff.
_F32_U = float(np.finfo(np.float32).eps) / 2
_GAMMA4 = 4 * _F32_U / (1 - 4 * _F32_U)
PROBABILITY_MASS_BOUND = (_GAMMA4 + _F32_U) / (1 - _GAMMA4) + 8 * np.finfo(np.float64).eps
# Conservative per-class agreement allowance for this bounded [-3, 3] softmax,
# including float32 subtraction/exp/normalization. This independent reference
# check rejects redistributed or masked mass; it is not a universal XLA proof.
SOFTMAX_RELATIVE_BOUND = 32 * _F32_U


def _config(c):
    if c != CFG:
        raise ValueError("E127 fixes the original world/controller dimensions with one agent")


def array_hash(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8).copy()


def _identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _master(master):
    if isinstance(master, bool) or not isinstance(master, Integral) or not 0 <= master < 2 ** 32:
        raise ValueError("master seed must be a uint32 integer")
    return jax.random.PRNGKey(int(master))


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves its reserved stream")
    return np.asarray(jax.random.split(jax.random.fold_in(_master(master), int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if count not in (4, 128):
        raise ValueError("E127 assigns four development or 128 final families")
    key = jax.random.fold_in(_master(master), ASSIGNMENT_TAG)
    permutation = np.asarray(jax.random.permutation(key, count), np.int32)
    return dict(key=np.asarray(key), permutation=permutation,
                pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2)[permutation])


def contact_script(key, pre_contacts):
    """Each adjacent pair contacts both colors; every block covers color x direction."""
    if pre_contacts not in (8, 16):
        raise ValueError("hidden boundary must follow eight or sixteen contacts")
    kd, kc, ko = jax.random.split(jnp.asarray(key), 3)
    directions = np.asarray(jax.vmap(lambda k: jax.random.permutation(k, jnp.arange(1, 5)))(
        jax.random.split(kd, 4)), np.int32)
    first = np.asarray(jax.random.randint(kc, (4, 4), 1, 3), np.int32)
    chosen = np.stack((first, 3 - first), axis=-1).reshape(-1)
    outward = np.repeat(directions, 2, axis=1).reshape(-1)
    offsets = np.asarray(jax.random.randint(ko, (32,), 1, 4), np.int32)
    other = (outward - 1 + offsets) % 4 + 1
    a, b = np.where(chosen == 1, outward, other), np.where(chosen == 2, outward, other)
    return np.stack((a, b, chosen, outward, OPPOSITE[outward]), axis=1)[:pre_contacts + 16].astype(np.int32)


@lru_cache(None)
def _unravel(c):
    return ravel_pytree(seed.init_genome(c, jax.random.PRNGKey(0)))[1]


def _genome(flat, controller, c):
    flat = np.asarray(flat)
    size = seed.genome_size(c) if controller == "plastic" else direct.genome_size(c)
    if flat.shape != (size,) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("diagnostic requires its unchanged finite float32 source encoding")
    return jnp.asarray(flat)


def _birth(flat, controller, birth_key, chamber_key, c):
    _, inputs, _, neurons, _ = seed.dims(c)
    if controller == "plastic":
        g = _unravel(c)(flat)
        W, mask = jax.vmap(lambda k: seed.develop(g, k, c))(jax.random.split(birth_key, c.M))
    else:
        W = direct.decode(flat, c)[None]
        legal = (np.arange(neurons)[:, None] >= inputs) & ~np.eye(neurons, dtype=bool)
        mask = jnp.asarray(legal, jnp.float32)[None]
    return dict(grid=jnp.zeros((c.G, c.G), jnp.int32), pos=jnp.full((1, 2), c.G // 2, jnp.int32),
        W=W, mask=mask, x=jnp.zeros((1, neurons)), Wp=jnp.zeros((1, inputs, c.H)),
        energy=jnp.full((1,), c.e0), alive=jnp.ones(1), sig=jnp.zeros(1), denergy=jnp.zeros(1),
        counts=jnp.zeros((1, 2, 2)), key=chamber_key, t=jnp.asarray(0, jnp.int32))


def _think(s, g, obs, controller, c, hebbian, predictor):
    _, inputs, outputs, neurons, _ = seed.dims(c)
    if controller == "plastic":
        types = seed.neuron_types(c)
        hebb = g["hebb"][types[:, None], types[None, :]]

        def neural(update):
            return jax.vmap(lambda w, x, wp, o, de, mk: seed.brain_step(
                w, x, wp, o, de, mk, hebb, jnp.exp(g["log_eta"]), jnp.exp(g["log_plr"]),
                g["mod"], update, c, predictor_plastic=float(predictor)))(
                    s["W"], s["x"], s["Wp"], obs, s["denergy"], s["mask"])

        result = neural(float(hebbian)) if isinstance(hebbian, bool) else jax.lax.cond(
            hebbian, lambda _: neural(1.), lambda _: neural(0.), operand=None)
        W, x, Wp, surprise = result
    else:
        # Preserve the original direct path: in particular never clip its weights.
        W, Wp = s["W"], s["Wp"]
        x = s["x"].at[:, :inputs].set(obs)
        for _ in range(c.micro):
            x = jax.vmap(lambda w, state: jnp.tanh(w @ state))(W, x).at[:, :inputs].set(obs)
        surprise = jnp.mean(obs ** 2, axis=1)
    logits = 3.0 * x[:, neurons - outputs:neurons - 1]
    return W, x, Wp, surprise, logits


def _step(s, g, rule, forced, controller, c, *, hebbian=True, predictor=True, regrowth=True):
    """Ordinary one-agent transition, with an explicit imposed-action/apparatus switch."""
    _, _, outputs, neurons, _ = seed.dims(c)
    key, ka, kr = jax.random.split(s["key"], 3)
    obs = seed.observe(s["grid"], s["pos"], s["alive"], s["sig"], s["energy"], s["denergy"], c)
    W, x, Wp, surprise, logits = _think(s, g, obs, controller, c, hebbian, predictor)
    motor = x[:, neurons - outputs:neurons - 1]
    sampled = jax.random.categorical(ka, logits)
    act = jnp.where(forced >= 0, jnp.full((1,), forced, sampled.dtype), sampled)
    x = x.at[:, neurons - outputs:neurons - 1].set(jax.nn.one_hot(act, outputs - 1))
    sig = x[:, neurons - 1] * s["alive"]
    alive = s["alive"]
    pos = (s["pos"] + seed.MOVES[act] * alive[:, None].astype(jnp.int32)) % c.G
    item = s["grid"][pos[:, 0], pos[:, 1]]
    eat = (alive > 0) & (item > 0)
    food, poison = eat & (item == 1 + rule), eat & (item == 2 - rule)
    unclipped = s["energy"] - c.decay * alive + c.gain * food - c.gain * poison
    energy = jnp.clip(unclipped, 0.0, c.e_max)
    denergy = (energy - s["energy"]) * alive
    alive_new = alive * (energy > 0)
    grid = s["grid"].at[pos[:, 0], pos[:, 1]].set(jnp.where(eat, 0, item))
    newc = jax.random.randint(kr, (1, 2), 0, c.G)
    if regrowth:
        grid = grid.at[newc[:, 0], newc[:, 1]].set(jnp.where(eat, item, grid[newc[:, 0], newc[:, 1]]))
    half = jnp.asarray(s["t"] >= c.T // 2, dtype=jnp.int32)
    counts = s["counts"].at[:, half, 0].add(food).at[:, half, 1].add(poison)
    ns = dict(grid=grid, pos=pos, W=W, x=x, Wp=Wp, mask=s["mask"], energy=energy, alive=alive_new,
              sig=sig, denergy=denergy, counts=counts, key=key, t=s["t"] + 1)
    trace = dict(grid_before=s["grid"], position_before=s["pos"], energy_before=s["energy"],
        signal_before=s["sig"], incoming_denergy=s["denergy"], incoming_activity=s["x"],
        observation=obs, decision_motor=motor, logits=logits, probabilities=jax.nn.softmax(logits),
        sampled_action=sampled, forced_action=jnp.asarray(forced, jnp.int32), action=act,
        action_key=ka, regrowth_key=kr, regrowth_cell=newc, key_before=s["key"], key_after=key,
        grid=grid, position=pos, energy=energy, unclipped_energy=unclipped, signal=sig,
        alive_before=alive, alive_after=alive_new, denergy=denergy, activity=x,
        efference=x[:, neurons - outputs:neurons - 1], food=food, poison=poison, item=item,
        surprise=jnp.mean(surprise), rule=jnp.asarray(rule, jnp.int32), t=s["t"],
        hebbian_enabled=jnp.asarray(hebbian) & (controller == "plastic"),
        predictor_enabled=jnp.asarray(predictor and controller == "plastic"))
    return ns, trace


def _pair_grid(position, placement, c):
    cells = (position[0] + seed.MOVES[placement]) % c.G
    return jnp.zeros((c.G, c.G), jnp.int32).at[cells[:, 0], cells[:, 1]].set(jnp.asarray([1, 2], jnp.int32))


@lru_cache(None)
def _chamber(controller, c, hebbian_cutoff=None):
    @jax.jit
    def run(flat, state, script, rule):
        g = _unravel(c)(flat) if controller == "plastic" else None

        def trial(s, presentation):
            # Explicit apparatus reset happens only after the previous hub return.
            s = dict(s, grid=_pair_grid(s["pos"], presentation[:2], c))
            enabled = True if hebbian_cutoff is None else s["t"] < hebbian_cutoff
            s, outward = _step(s, g, rule, presentation[3], controller, c,
                hebbian=enabled, predictor=controller == "plastic", regrowth=False)
            enabled = True if hebbian_cutoff is None else s["t"] < hebbian_cutoff
            s, returning = _step(s, g, rule, presentation[4], controller, c,
                hebbian=enabled, predictor=controller == "plastic", regrowth=False)
            return s, jax.tree.map(lambda a, b: jnp.stack((a, b)), outward, returning)

        state, trace = jax.lax.scan(trial, state, script)
        return state, jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), trace)
    return run


def _fresh_body(snapshot, grid, position, key, c):
    return dict(snapshot, grid=grid, pos=position, energy=jnp.full((1,), c.e0), alive=jnp.ones(1),
                sig=jnp.zeros(1), denergy=jnp.zeros(1), counts=jnp.zeros((1, 2, 2)), key=key,
                t=jnp.asarray(0, jnp.int32))


@lru_cache(None)
def _readouts(controller, c):
    @jax.jit
    def run(flat, snapshot, keys):
        g = _unravel(c)(flat) if controller == "plastic" else None

        def one(placement, key):
            pos = jnp.full((1, 2), c.G // 2, jnp.int32)
            initial = _fresh_body(snapshot, _pair_grid(pos, placement, c), pos, key, c)
            obs = seed.observe(initial["grid"], pos, initial["alive"], initial["sig"], initial["energy"], initial["denergy"], c)
            W, x, Wp, surprise, logits = _think(initial, g, obs, controller, c, False, False)
            _, action_key, _ = jax.random.split(key, 3)
            return dict(initial=initial, final=dict(initial, W=W, x=x, Wp=Wp), observation=obs,
                        decision_motor=x[:, -6:-1], logits=logits, probabilities=jax.nn.softmax(logits),
                        sampled_action=jax.random.categorical(action_key, logits), action_key=action_key,
                        surprise=jnp.mean(surprise), hebbian_enabled=jnp.asarray(False),
                        predictor_enabled=jnp.asarray(False), feedback_applied=jnp.asarray(False))
        return jax.vmap(one)(jnp.asarray(PLACEMENTS), keys)
    return run


@lru_cache(None)
def _free_grid(controller, c, hebbian=True, steps=GRID_STEPS):
    @jax.jit
    def run(flat, initial, rule):
        g = _unravel(c)(flat) if controller == "plastic" else None
        return jax.lax.scan(lambda s, _: _step(s, g, rule, -1, controller, c,
            hebbian=hebbian, predictor=controller == "plastic", regrowth=True), initial, None, length=steps)
    return run


def _state_evidence(state):
    return {**{name + "_sha256": array_hash(state[name]) for name in NEURAL},
            **{name: np.asarray(state[name]).copy() for name in BODY}}


def _stack(rows):
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def _probe_evidence(result):
    result = jax.tree.map(np.asarray, result)
    output = {name: value[:, 0] if name in ("logits", "probabilities", "decision_motor", "sampled_action") else value
              for name, value in result.items() if name not in ("initial", "final")}
    for phase in ("initial", "final"):
        state = result[phase]
        for name in NEURAL:
            output[phase + "_" + name + "_sha256"] = np.stack([array_hash(v) for v in state[name]])
        for name in BODY:
            output[phase + "_" + name] = state[name]
    return output


def evaluate_world(plastic_flat, direct_flat, keys, index, pre_contacts, c=CFG):
    _config(c)
    keys = np.asarray(keys)
    if (keys.shape != (6, 2) or keys.dtype != np.uint32 or len({k.tobytes() for k in keys}) != 6
            or isinstance(index, bool) or not isinstance(index, Integral) or index < 0 or pre_contacts not in (8, 16)):
        raise ValueError("invalid independent family keys/index or registered hidden boundary")
    flats = {"plastic": _genome(plastic_flat, "plastic", c), "direct": _genome(direct_flat, "direct", c)}
    script = contact_script(keys[0], int(pre_contacts))
    grid, position = seed.init_world(jnp.asarray(keys[3]), c)
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(), pre_contacts=np.asarray(pre_contacts, np.int32),
        initial_rules=np.asarray([0, 1], np.int32), script=script, probe_placements=PLACEMENTS.copy(),
        readout_contacts=np.asarray(CONTACTS, np.int32), grid_steps=np.asarray(GRID_STEPS, np.int32),
        plastic_genome_sha256=array_hash(plastic_flat), direct_genome_sha256=array_hash(direct_flat))
    arms = {arm: [] for arm in ARMS}
    for initial_rule in (0, 1):
        for controller, flat in flats.items():
            birth = _birth(flat, controller, jnp.asarray(keys[1]), jnp.asarray(keys[2]), c)
            boundary, pre_trace = _chamber(controller, c)(flat, birth, jnp.asarray(script[:pre_contacts]), initial_rule)
            for arm in (name for name in ARMS if name.startswith(controller)):
                rule = initial_rule if arm.endswith("stationary") else 1 - initial_rule
                cutoff = 2 * int(pre_contacts) + 1 if arm.endswith("frozen") else None
                state, snapshots, traces = boundary, [boundary], [pre_trace]
                for start, stop in zip(CONTACTS[:-1], CONTACTS[1:]):
                    state, trace = _chamber(controller, c, cutoff)(flat, state,
                        jnp.asarray(script[pre_contacts + start:pre_contacts + stop]), rule)
                    snapshots.append(state)
                    traces.append(trace)
                guided = jax.tree.map(lambda *values: np.concatenate([np.asarray(v) for v in values]), *traces)
                record = {"guided_" + name: value for name, value in guided.items()}
                record.update({"birth_" + name: value for name, value in _state_evidence(birth).items()})
                record.update({"snapshot_" + name: value for name, value in _stack([_state_evidence(s) for s in snapshots]).items()})
                probes = []
                for ordinal, snapshot in enumerate(snapshots):
                    probe_keys = jax.random.split(jax.random.fold_in(jnp.asarray(keys[5]), ordinal), len(PLACEMENTS))
                    probes.append(_probe_evidence(_readouts(controller, c)(flat, snapshot, probe_keys)))
                record.update({"probe_" + name: value for name, value in _stack(probes).items()})
                initial = _fresh_body(state, grid, position, jnp.asarray(keys[4]), c)
                final, trace = _free_grid(controller, c, not arm.endswith("frozen"))(flat, initial, 1 - initial_rule)
                record.update({"grid_initial_" + name: value for name, value in _state_evidence(initial).items()})
                record.update({"grid_final_" + name: value for name, value in _state_evidence(final).items()})
                record.update({"grid_" + name: np.asarray(value) for name, value in trace.items()})
                arms[arm].append(record)
    for arm, records in arms.items():
        row.update({arm + "_" + name: value for name, value in _stack(records).items()})
    validate_world(row, c)
    return row


def _arm(row, arm):
    prefix = arm + "_"
    return {name[len(prefix):]: np.asarray(value) for name, value in row.items() if name.startswith(prefix)}


def validate_probabilities(probabilities, logits):
    """Validate recorded float32 outputs without renormalizing or modifying them."""
    probabilities, logits = np.asarray(probabilities), np.asarray(logits)
    if (probabilities.shape != logits.shape or probabilities.ndim < 1 or probabilities.shape[-1] != 5
            or probabilities.dtype != np.float32 or logits.dtype != np.float32
            or not np.isfinite(probabilities).all() or not np.isfinite(logits).all()
            or np.any(probabilities <= 0) or np.any(probabilities >= 1) or np.any(np.abs(logits) > 3)):
        raise ValueError("readout requires all five finite positive float32 probabilities and bounded logits")
    p64, logits64 = probabilities.astype(np.float64), logits.astype(np.float64)
    if np.any(np.abs(p64.sum(axis=-1) - 1.) > PROBABILITY_MASS_BOUND):
        raise ValueError("five-action probability mass exceeds its float32 rounding bound")
    exponent = np.exp(logits64 - logits64.max(axis=-1, keepdims=True))
    reference = exponent / exponent.sum(axis=-1, keepdims=True)
    if np.any(np.abs(p64 - reference) > SOFTMAX_RELATIVE_BOUND * reference):
        raise ValueError("recorded probabilities disagree with the original five-action logits")


def validate_world(row, c=CFG):
    _config(c)
    pre = int(row["pre_contacts"])
    if pre not in (8, 16) or not _identical(row["initial_rules"], np.asarray([0, 1], np.int32)):
        raise ValueError("family must retain both initial rules and its assigned boundary")
    script = np.asarray(row["script"])
    if script.shape != (pre + 16, 5) or not np.array_equal(script[:, 4], OPPOSITE[script[:, 3]]):
        raise ValueError("invalid apparatus script")
    if not np.all(np.sort(script[:, 2].reshape(-1, 2), axis=1) == [1, 2]):
        raise ValueError("every contact pair must contain both colors")
    for block in script.reshape(-1, 8, 5):
        if sorted(map(tuple, block[:, [2, 3]].tolist())) != [(color, direction) for color in (1, 2) for direction in range(1, 5)]:
            raise ValueError("contact block does not counterbalance color/direction")
    ticks = 2 * (pre + 16)
    for arm in ARMS:
        value = _arm(row, arm)
        if value["guided_action"].shape != (2, ticks, 1) or value["probe_probabilities"].shape != (2, 4, 12, 5):
            raise ValueError("lost guided turns or probe alternatives")
        forced = script[:, [3, 4]].reshape(-1)
        if not np.array_equal(value["guided_action"][:, :, 0], np.broadcast_to(forced, (2, ticks))):
            raise ValueError("actual action differs from imposed movement")
        if (not np.all(value["guided_alive_after"] == 1) or np.any(value["guided_unclipped_energy"] <= 0)
                or np.any(value["guided_unclipped_energy"] >= c.e_max)):
            raise ValueError("apparatus violated its alive/no-clipping guarantee")
        for stem in ("guided", "grid"):
            action = value[stem + "_action"]
            if not np.array_equal(value[stem + "_efference"], np.eye(5, dtype=np.float32)[action]):
                raise ValueError("motor efference differs from the actual action")
            if not np.array_equal(value[stem + "_observation"][..., -2], 4 * value[stem + "_incoming_denergy"]):
                raise ValueError("real prior feedback did not enter the next sensory tick")
            if not _identical(value[stem + "_incoming_activity"][:, 1:], value[stem + "_activity"][:, :-1]):
                raise ValueError("recurrent activity/efference was not carried across ticks")
        expected_hebb = np.full(ticks, arm.startswith("plastic"), bool)
        if arm.endswith("frozen"):
            expected_hebb[2 * pre + 1:] = False
        if not np.array_equal(value["guided_hebbian_enabled"], np.broadcast_to(expected_hebb, (2, ticks))):
            raise ValueError("Hebbian intervention started on the wrong feedback tick")
        for name in NEURAL:
            snapshots = value["snapshot_" + name + "_sha256"]
            if not np.array_equal(value["probe_initial_" + name + "_sha256"], np.broadcast_to(snapshots[:, :, None], (2, 4, 12, 32))):
                raise ValueError("probe did not receive the complete unprobed neural snapshot")
            if not np.array_equal(value["grid_initial_" + name + "_sha256"], snapshots[:, -1]):
                raise ValueError("grid did not receive the complete unprobed neural snapshot")
        for name in ("W", "Wp", "mask"):
            if not _identical(value["probe_initial_" + name + "_sha256"], value["probe_final_" + name + "_sha256"]):
                raise ValueError("discarded readout changed a frozen weight or mask")
        if any(np.any(value["probe_" + field]) for field in ("hebbian_enabled", "predictor_enabled", "feedback_applied")):
            raise ValueError("discarded readout must have no learning or outcome feedback")
        for field, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.), ("sig", 0.), ("t", 0), ("counts", 0.)):
            for stem in ("probe_initial_", "grid_initial_"):
                if not np.all(value[stem + field] == expected):
                    raise ValueError("readout did not begin with its declared fresh neutral body")
        if not _identical(value["probe_logits"], np.float32(3) * value["probe_decision_motor"]):
            raise ValueError("recorded probe logits differ from the actual five motor activations")
        validate_probabilities(value["probe_probabilities"], value["probe_logits"])
    for controller in ("plastic", "direct"):
        left, right = (_arm(row, controller + "_" + condition) for condition in ("stationary", "reversal"))
        if not _identical(left["probe_probabilities"][:, 0], right["probe_probabilities"][:, 0]):
            raise ValueError("zero-contact cloned readouts differ")


def summary_input(row):
    validate_world(row)
    compact = {name: np.asarray(row[name]).copy() for name in
               ("base_index", "keys", "pre_contacts", "plastic_genome_sha256", "direct_genome_sha256")}
    for arm in ARMS:
        value = _arm(row, arm)
        by_rule = local_outcomes(value["probe_probabilities"])
        compact[arm + "_local_by_rule"] = by_rule
        compact[arm + "_local"] = by_rule.mean(0)
        f, p = value["grid_food"][:, :, 0], value["grid_poison"][:, :, 0]
        any_contact = (f | p).any(1)
        first = (f | p).argmax(1)
        first_food = f[np.arange(2), first] & any_contact
        first_poison = p[np.arange(2), first] & any_contact
        by_rule = np.stack((first_food, first_poison, ~any_contact,
            first_food.astype(int) - first_poison.astype(int), np.where(any_contact, first + 1, 51),
            (f.astype(int) - p.astype(int)).sum(1), value["grid_alive_after"][:, -1, 0]), axis=1).astype(np.float64)
        compact[arm + "_grid_by_rule"] = by_rule
        compact[arm + "_grid"] = by_rule.mean(0)
    return compact


def local_outcomes(probabilities):
    """Five-way readouts, averaged over orientations only; retain both initial rules."""
    probs = np.asarray(probabilities, np.float64)
    if probs.shape != (2, 4, 12, 5) or not np.isfinite(probs).all():
        raise ValueError("local outcome requires both rules, four snapshots and all twelve five-way scenes")
    food = np.stack([probs[rule, :, np.arange(12), PLACEMENTS[:, 1 - rule]] for rule in (0, 1)])
    poison = np.stack([probs[rule, :, np.arange(12), PLACEMENTS[:, rule]] for rule in (0, 1)])
    # Advanced indexing yields orientation x time. Orientations are repeated measures.
    return np.stack((food, poison, food + poison, 1 - food - poison, food - poison), axis=-1).mean(1)


def _stats(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary requires finite assigned-family scalar outcomes")
    return dict(mean=float(values.mean()), se=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                n=len(values))


def _positive(stat, minimum=None):
    if stat["se"] is None:
        return None
    return bool(stat["mean"] > 2 * stat["se"] and (minimum is None or stat["mean"] >= minimum))


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty assigned panel")
    rows = [summary_input(row) if "script" in row else row for row in rows]
    tokens = [np.asarray(row["keys"]).tobytes() for row in rows]
    if len(set(tokens)) != len(rows) or len({int(row["base_index"]) for row in rows}) != len(rows):
        raise ValueError("family duplicates cannot be counted as replication")
    vectors, arms = {}, {}
    for arm in ARMS:
        local = np.stack([row[arm + "_local"] for row in rows])
        grid = np.stack([row[arm + "_grid"] for row in rows])
        if local.shape != (len(rows), 4, 5) or grid.shape != (len(rows), 7):
            raise ValueError("compact family outcomes have incorrect dimensions")
        vectors[arm] = dict(local=local, grid=grid)
        arms[arm] = dict(local=[dict(contacts=contact, **{name: _stats(local[:, time, field])
            for field, name in enumerate(LOCAL_FIELDS)}) for time, contact in enumerate(CONTACTS)],
            grid={name: _stats(grid[:, field]) for field, name in enumerate(GRID_FIELDS)})
    architectures = {}
    for controller in ("plastic", "direct"):
        r, s = vectors[controller + "_reversal"], vectors[controller + "_stationary"]
        local = dict(pre_old=_stats(-s["local"][:, 0, 4]), post_target=_stats(r["local"][:, -1, 4]),
            reversal_minus_stationary=_stats(r["local"][:, -1, 4] - s["local"][:, -1, 4]))
        transfer = dict(reversal_minus_stationary=_stats(r["grid"][:, 0] - s["grid"][:, 0]),
                        food_minus_poison=_stats(r["grid"][:, 3]))
        local_checks = (_positive(local["pre_old"]), _positive(local["post_target"], .1),
                        _positive(local["reversal_minus_stationary"], .1))
        grid_checks = (_positive(transfer["reversal_minus_stationary"], .1), _positive(transfer["food_minus_poison"]))
        architectures[controller] = dict(local=local, transfer=transfer,
            local_pass=None if None in local_checks else all(local_checks),
            transfer_pass=None if None in grid_checks else all(grid_checks))
    r, frozen = vectors["plastic_reversal"], vectors["plastic_frozen"]
    hebbian = dict(local_difference=_stats(r["local"][:, -1, 4] - frozen["local"][:, -1, 4]),
                   transfer_difference=_stats(r["grid"][:, 0] - frozen["grid"][:, 0]))
    hebbian["hebbian_local_pass"] = _positive(hebbian["local_difference"], .05)
    by_rule = {}
    for rule in (0, 1):
        by_rule[str(rule)] = {}
        for controller in ("plastic", "direct"):
            local_r, local_s, grid_r, grid_s = [np.stack([row[controller + "_" + condition + "_" + field + "_by_rule"]
                for row in rows])[:, rule] for field, condition in
                (("local", "reversal"), ("local", "stationary"), ("grid", "reversal"), ("grid", "stationary"))]
            by_rule[str(rule)][controller] = dict(S=_stats(local_r[:, -1, 4]),
                delta_S=_stats(local_r[:, -1, 4] - local_s[:, -1, 4]),
                G=_stats(grid_r[:, 0]), delta_G=_stats(grid_r[:, 0] - grid_s[:, 0]))
    return dict(families=len(rows), initial_rule_replays=2 * len(rows), arms=arms,
                architectures=architectures, hebbian=hebbian, by_initial_rule=by_rule)
