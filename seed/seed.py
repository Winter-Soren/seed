"""Seed v0: evolving brains that learn on their own.

Three nested loops:
  evolution   (generations)  - a population of tiny genomes improves by evolution strategies
  development (birth)        - each genome GROWS a brain from wiring rules (twins differ)
  learning    (lifetime)     - no labels, no backprop: local Hebbian plasticity gated by a
                               neuromodulator, plus a local prediction loop (surprise signal)

The world: a small torus grid with two kinds of blobs (A and B). In every new world a hidden
rule decides which one is food and which is poison, so no fixed instinct can win -- agents
must LEARN within their lifetime. Agents live in families (same genome), see each other,
and can emit a signal that others can sense.
"""
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from seed.structural import INTERVAL, STREAM_TAG, birth_caps, rewire, validate_structure
from seed.ecology import EcologyV1, apply_costs, make_shelter
from seed.persistence import PersistenceV1, bias_logits, initial_memory, validate_persistence


class Cfg(NamedTuple):
    G: int = 16          # world is G x G (wraps around)
    M: int = 4           # agents per world (one family)
    T: int = 300         # lifetime in steps
    R: int = 2           # view radius -> (2R+1)^2 patch
    n_items: int = 24    # blobs of each kind
    H: int = 48          # hidden neurons
    n_htypes: int = 3    # hidden neuron types (genome sets rules per type pair)
    e0: float = 1.0      # starting energy
    decay: float = 0.003 # energy cost per step of being alive
    gain: float = 0.25   # energy from food (lost from poison)
    e_max: float = 2.0
    micro: int = 2       # neural updates per world step
    wmax: float = 3.0    # synapse weight bound


def dims(c, *, ecology=None):
    V = 2 * c.R + 1
    D = 4 * V * V + 3          # 4 visual channels + energy + energy change + bias
    if ecology is not None:
        if not isinstance(ecology, EcologyV1):
            raise ValueError("ecology must be an EcologyV1 instance or None")
        D += V * V + 1         # visible shelter patch and current cold-season bit
    O = 6                      # 5 movement logits + 1 signal
    n = D + c.H + O
    K = 2 + c.n_htypes         # sensor, hidden types..., motor
    return V, D, O, n, K


def neuron_types(c, *, ecology=None):
    V, D, O, n, K = dims(c, ecology=ecology)
    assert c.H % c.n_htypes == 0
    ht = np.repeat(np.arange(1, 1 + c.n_htypes), c.H // c.n_htypes)
    return jnp.array(np.concatenate([np.zeros(D, int), ht, np.full(O, K - 1)]))


# ----------------------------------------------------------------------------- genome
def init_genome(c, key):
    """The genome holds RULES, not weights: ~155 numbers in total."""
    K = dims(c)[4]
    return dict(
        conn=jnp.full((K, K), -1.0),                        # wiring prob. logit (post type, pre type)
        wscale=jnp.zeros((K, K)),                           # initial synapse strength (log)
        hebb=0.1 * jax.random.normal(key, (K, K, 4)),       # plasticity rule A,B,C,D per type pair
        log_eta=jnp.array(-1.5),                            # plasticity rate (log)
        mod=jnp.array([1.0, 0.0, 0.0]),                     # neuromodulator: energy change, surprise, bias
        log_plr=jnp.array(-3.0),                            # prediction-loop learning rate
    )


def biology_prior(c, key=None):
    """A genome seeded with a rule nature already found: dopamine-gated Hebbian learning.
    Synapses from senses to movement strengthen when a recent action raised energy
    ("that was good") and weaken when it lowered it. Everything else starts neutral,
    and evolution is free to change all of it."""
    K = dims(c)[4]
    g = init_genome(c, jax.random.PRNGKey(0) if key is None else key)
    hebb = jnp.zeros((K, K, 4)).at[K - 1, 0, 0].set(1.0)   # motor <- sensor: modulator x pre x post
    return dict(g, hebb=hebb, log_eta=jnp.array(-1.0), mod=jnp.array([3.0, 0.0, 0.0]),
                conn=g["conn"].at[K - 1, 0].set(0.0))


def genome_size(c):
    return ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))[0].size


# ----------------------------------------------------------------------------- development
def develop(g, key, c, *, ecology=None):
    """Grow one brain from the genome's wiring rules. Different keys -> different twins."""
    V, D, O, n, K = dims(c, ecology=ecology)
    types = neuron_types(c, ecology=ecology)
    k1, k2 = jax.random.split(key)
    tp, tq = types[:, None], types[None, :]
    P = jax.nn.sigmoid(g["conn"])[tp, tq]
    P = P * (tp != 0) * (1.0 - jnp.eye(n))                  # sensors are clamped to the world
    mask = jax.random.bernoulli(k1, P).astype(jnp.float32)
    fanin = jnp.maximum(P.sum(1, keepdims=True), 1.0)
    W = mask * jax.random.normal(k2, (n, n)) * jnp.exp(g["wscale"])[tp, tq] / jnp.sqrt(fanin)
    return W, mask


# ----------------------------------------------------------------------------- the world
def observe(grid, pos, alive, sig, energy, denergy, c, *, ecology=None, shelter=None, step=0):
    G, R, M = c.G, c.R, c.M
    offs = jnp.arange(-R, R + 1)
    rows = (pos[:, 0, None] + offs[None, :]) % G
    cols = (pos[:, 1, None] + offs[None, :]) % G
    amap = (grid == 1).astype(jnp.float32)
    bmap = (grid == 2).astype(jnp.float32)
    occ = jnp.zeros((G, G)).at[pos[:, 0], pos[:, 1]].add(alive)
    smap = jnp.zeros((G, G)).at[pos[:, 0], pos[:, 1]].add(alive * sig)
    patch = lambda mp: mp[rows[:, :, None], cols[:, None, :]]
    pa, pb, po, ps = patch(amap), patch(bmap), patch(occ), patch(smap)
    po = po.at[:, R, R].add(-alive)                          # don't see yourself
    ps = ps.at[:, R, R].add(-alive * sig)
    flat = lambda a: a.reshape(M, -1)
    obs = jnp.concatenate(
        [flat(pa), flat(pb), flat(po), flat(ps),
         energy[:, None] - 1.0, 4.0 * denergy[:, None], jnp.ones((M, 1))], axis=1)
    if ecology is not None:
        dims(c, ecology=ecology)
        if shelter is None or shelter.shape != (c.G, c.G):
            raise ValueError("ecology observations require the current world's shelter map")
        obs = jnp.concatenate([obs, flat(patch(shelter)), jnp.full((M, 1), step >= c.T // 2)], axis=1)
    return obs


MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


def init_world(key, c):
    k1, k2 = jax.random.split(key)
    cells = jax.random.choice(k1, c.G * c.G, (2 * c.n_items,), replace=False)
    vals = jnp.concatenate([jnp.ones(c.n_items, jnp.int32), 2 * jnp.ones(c.n_items, jnp.int32)])
    grid = jnp.zeros(c.G * c.G, jnp.int32).at[cells].set(vals).reshape(c.G, c.G)
    pos = jax.random.randint(k2, (c.M, 2), 0, c.G)
    return grid, pos


# ----------------------------------------------------------------------------- one brain tick
def brain_step(W, x, Wp, obs, denergy, mask, hebb, eta, plr, modw, plastic, c, *, predictor_plastic=None, ecology=None):
    V, D, O, n, K = dims(c, ecology=ecology)
    h = x[D:D + c.H]
    # prediction loop: the brain predicted this observation last step; learn from the miss
    err = obs - Wp @ h
    surprise = jnp.mean(err ** 2)
    pp = plastic if predictor_plastic is None else predictor_plastic
    Wp = jnp.clip(Wp + pp * plr * jnp.outer(err, h), -c.wmax, c.wmax)
    # neuromodulator ("that mattered"): driven by felt energy change and surprise
    m = jnp.tanh(modw[0] * 10.0 * denergy + modw[1] * 10.0 * surprise + modw[2])
    # three-factor Hebbian update on the activity that produced the last action
    post, pre = x[:, None], x[None, :]
    dW = hebb[..., 0] * post * pre + hebb[..., 1] * pre + hebb[..., 2] * post + hebb[..., 3]
    W = jnp.clip(W + plastic * eta * m * dW * mask, -c.wmax, c.wmax)
    # think
    x = x.at[:D].set(obs)
    for _ in range(c.micro):
        x = jnp.tanh(W @ x).at[:D].set(obs)
    return W, x, Wp, surprise


# ----------------------------------------------------------------------------- one lifetime
@partial(jax.jit, static_argnames=("c", "plastic", "record", "predictor_plastic", "structural", "flip_step", "diagnostics", "ecology", "persistence", "_return_weights"))
def rollout(g, wkey, dkey, rule, c, plastic=True, record=False, *, predictor_plastic=None,
            structural=False, flip_step=None, diagnostics=False, direct_weights=None, ecology=None,
            persistence=None, _initial_weights=None, _initial_alive=None, _return_weights=False):
    """One family lives one life in one world. rule=0: A is food; rule=1: B is food."""
    if flip_step is not None and (type(flip_step) is not int or not 0 < flip_step < c.T):
        raise ValueError("flip_step must be an integer strictly inside the lifetime")
    validate_persistence(persistence, ecology)
    V, D, O, n, K = dims(c, ecology=ecology)
    pl = 1.0 if plastic else 0.0
    pp = pl if predictor_plastic is None else float(predictor_plastic)
    if direct_weights is None:
        types = neuron_types(c, ecology=ecology)
        hebb = g["hebb"][types[:, None], types[None, :]]
        eta, plr = jnp.exp(g["log_eta"]), jnp.exp(g["log_plr"])
        W, mask = jax.vmap(lambda k: develop(g, k, c, ecology=ecology))(jax.random.split(dkey, c.M))
    else:
        if plastic or pp or structural:
            raise ValueError("direct weights require frozen weights/predictor and no structural updates")
        if direct_weights.shape != (n, n) or not jnp.issubdtype(direct_weights.dtype, jnp.floating):
            raise ValueError("direct weights must be a floating-point (neurons, neurons) matrix")
        W = jnp.broadcast_to(direct_weights, (c.M, n, n))
    if _initial_weights is not None or _initial_alive is not None or _return_weights:
        if structural or direct_weights is not None:
            raise ValueError("lineage hooks require the unchanged rules controller and fixed topology")
        birth_W = W
        if _initial_weights is not None:
            if _initial_weights.shape != W.shape or not jnp.issubdtype(_initial_weights.dtype, jnp.floating):
                raise ValueError("lineage weights must match the family's floating-point weight shape")
            W = _initial_weights
        if _initial_alive is not None and _initial_alive.shape != (c.M,):
            raise ValueError("lineage eligibility must match the original family slots")
    grid, pos = init_world(wkey, c)
    shelter = make_shelter(wkey, c) if ecology is not None else None
    food_t, poison_t = 1 + rule, 2 - rule
    M = c.M
    carry = dict(
        grid=grid, pos=pos, W=W, x=jnp.zeros((M, n)), Wp=jnp.zeros((M, D, c.H)),
        energy=jnp.full((M,), c.e0), alive=jnp.ones(M), sig=jnp.zeros(M),
        denergy=jnp.zeros(M), counts=jnp.zeros((M, 2, 2)), key=wkey, t=0)
    if _initial_alive is not None:
        carry.update(alive=_initial_alive, energy=carry["energy"] * _initial_alive)
    if _return_weights:
        carry["last_alive_W"] = W
    if persistence is not None:
        carry["persistence"] = initial_memory(M)
    if structural:
        legal = (types[:, None] != 0) & ~jnp.eye(n, dtype=bool)
        caps = birth_caps(mask, legal)
        P = jax.nn.sigmoid(g['conn'])[types[:, None], types[None, :]] * legal
        scale = jnp.exp(g['wscale'])[types[:, None], types[None, :]] / jnp.sqrt(jnp.maximum(P.sum(1, keepdims=True), 1.))
        structural_keys = jax.vmap(lambda k: jax.random.fold_in(k, STREAM_TAG))(jax.random.split(dkey, M))
        edges = (mask > 0).sum((1, 2))
        zeros = jnp.zeros(M, dtype=jnp.int32)
        carry.update(mask=mask, structure=dict(
            added=zeros, removed=zeros, initial_edges=edges, final_edges=edges,
            min_edges=edges, max_edges=edges, edge_cap=caps.sum(1),
            cap_violations=zeros, illegal_edges=zeros, invalid_weights=zeros,
            invalid_proposals=zeros + (~jnp.isfinite(scale)).sum(),
            initial_max_abs_weight=jnp.abs(W).max((1, 2)), max_abs_weight=jnp.zeros(M)))

    def step(s, _):
        key, ka, kr = jax.random.split(s["key"], 3)
        if ecology is None:
            obs = observe(s["grid"], s["pos"], s["alive"], s["sig"], s["energy"], s["denergy"], c)
        else:
            obs = observe(s["grid"], s["pos"], s["alive"], s["sig"], s["energy"], s["denergy"], c,
                          ecology=ecology, shelter=shelter, step=s["t"])
        if direct_weights is None:
            W, x, Wp, surprise = jax.vmap(
                lambda W, x, Wp, o, de, mk: brain_step(
                    W, x, Wp, o, de, mk, hebb, eta, plr, g["mod"], pl, c, predictor_plastic=pp, ecology=ecology)
            )(s["W"], s["x"], s["Wp"], obs, s["denergy"], s['mask'] if structural else mask)
        else:
            # Fixed direct weights bypass even the clipping performed by a frozen plastic brain.
            W, Wp = s["W"], s["Wp"]
            x = s["x"].at[:, :D].set(obs)
            for _ in range(c.micro):
                x = jax.vmap(lambda w, state: jnp.tanh(w @ state))(W, x).at[:, :D].set(obs)
            surprise = jnp.mean(obs ** 2, axis=1)
        if structural:
            # The current neural tick/action is already computed; rewiring affects the next tick.
            def event(_):
                draws = jax.vmap(lambda k: jax.random.normal(jax.random.fold_in(k, s['t']), (n,)))(structural_keys)
                return jax.vmap(lambda w, mk, x0, cap, draw, alive: rewire(
                    w, mk, x0, legal, cap, scale, draw, wmax=c.wmax, alive=alive > 0, plastic=plastic)
                )(W, s['mask'], s['x'], caps, draws, s['alive'])

            new_mask = s['mask']
            delta = dict(added=jnp.zeros(M, jnp.int32), removed=jnp.zeros(M, jnp.int32),
                         invalid_proposals=jnp.zeros(M, jnp.int32))
            if plastic:
                W, new_mask, delta = jax.lax.cond((s['t'] > 0) & (s['t'] % INTERVAL == 0), event,
                                                 lambda _: (W, new_mask, delta), operand=None)
            edges = (new_mask > 0).sum((1, 2))
            stats = dict(s['structure'])
            for k in delta:
                stats[k] = stats[k] + delta[k]
            stats.update(final_edges=edges, min_edges=jnp.minimum(stats['min_edges'], edges),
                         max_edges=jnp.maximum(stats['max_edges'], edges),
                         max_abs_weight=jnp.maximum(stats['max_abs_weight'], jnp.abs(W).max((1, 2))))
            stats['cap_violations'] += ((new_mask > 0).sum(2) > caps).sum(1)
            stats['illegal_edges'] += ((new_mask > 0) & ~legal).sum((1, 2))
            stats['invalid_weights'] += ((~jnp.isfinite(W)) | (jnp.abs(W) > c.wmax) |
                                         ((new_mask == 0) & (W != 0))).sum((1, 2))
        logits = 3.0 * x[:, n - O:n - 1]
        if persistence is not None:
            local_shelter = obs[:, D - V * V - 1:D - 1].reshape(M, V, V)
            logits, pursuit, pursuit_events = bias_logits(
                logits, local_shelter, obs[:, D - 1] > 0, s["alive"],
                s["persistence"], persistence, MOVES)
        act = jax.random.categorical(ka, logits)
        # efference copy: motor neurons "know" which action was actually taken, so the
        # next Hebbian update links what was seen to what was done
        x = x.at[:, n - O:n - 1].set(jax.nn.one_hot(act, O - 1))
        sig = x[:, n - 1] * s["alive"]
        alive = s["alive"]
        pos = (s["pos"] + MOVES[act] * alive[:, None].astype(jnp.int32)) % c.G
        item = s["grid"][pos[:, 0], pos[:, 1]]
        eat = (alive > 0) & (item > 0)
        if flip_step is None:
            current_food, current_poison = food_t, poison_t
        else:
            current_rule = jnp.where(s["t"] >= flip_step, 1 - rule, rule)
            current_food, current_poison = 1 + current_rule, 2 - current_rule
        is_food = eat & (item == current_food)
        is_pois = eat & (item == current_poison)
        e_new = s["energy"] - c.decay * alive + c.gain * is_food - c.gain * is_pois
        before_ecology = e_new
        exposed = ~shelter[pos[:, 0], pos[:, 1]] if ecology is not None else True
        e_new = apply_costs(e_new, act, alive, s["t"], c.T // 2, ecology, exposed=exposed)
        e_new = jnp.clip(e_new, 0.0, c.e_max)
        denergy = (e_new - s["energy"]) * alive
        alive_new = alive * (e_new > 0)
        grid = s["grid"].at[pos[:, 0], pos[:, 1]].set(jnp.where(eat, 0, item))
        newc = jax.random.randint(kr, (M, 2), 0, c.G)       # eaten blobs regrow elsewhere
        grid = grid.at[newc[:, 0], newc[:, 1]].set(jnp.where(eat, item, grid[newc[:, 0], newc[:, 1]]))
        half = jnp.asarray(s["t"] >= c.T // 2, dtype=jnp.int32)
        counts = s["counts"].at[:, half, 0].add(is_food).at[:, half, 1].add(is_pois)
        ns = dict(grid=grid, pos=pos, W=W, x=x, Wp=Wp, energy=e_new, alive=alive_new, sig=sig,
                  denergy=denergy, counts=counts, key=key, t=s["t"] + 1)
        out = dict(surprise=jnp.mean(surprise))
        if persistence is not None:
            pursuit["last_action"] = act
            ns["persistence"] = pursuit
            if record or diagnostics:
                previous_action = s["persistence"]["last_action"]
                out["persistence"] = dict(pursuit_events, previous_action=previous_action,
                                          selected_action=act,
                                          action_switched=pursuit_events["tactic_attempted"] & (act != previous_action))
        if _return_weights:
            # No post-death credit: the fatal transition's delayed reward is not learned after death.
            ns["last_alive_W"] = jnp.where(alive[:, None, None] > 0, W, s["last_alive_W"])
        if ecology is not None and (record or diagnostics):
            out.update(sheltered=(~exposed) & (alive > 0),
                       ecology_energy_lost=jnp.clip(before_ecology, 0.0, c.e_max) - e_new)
        if structural:
            ns.update(mask=new_mask, structure=stats)
            if record:
                out.update(structural_added=delta['added'].sum(), structural_removed=delta['removed'].sum())
        if record:
            out.update(grid=grid, pos=pos, alive=alive_new, sig=sig, energy=e_new)
        if diagnostics:
            out.update(food=is_food, poison=is_pois, alive_before=alive, alive_after=alive_new)
        return ns, out

    s, traj = jax.lax.scan(step, carry, None, length=c.T)
    counts = s["counts"]                                     # (M, half, food/poison)
    fitness = jnp.mean(counts[:, :, 0].sum(1) - counts[:, :, 1].sum(1))
    res = dict(fitness=fitness, counts=counts.sum(0), survived=s["alive"].mean(),
               surprise_first=traj["surprise"][: c.T // 2].mean(),
               surprise_last=traj["surprise"][c.T // 2:].mean())
    if structural:
        res['structure'] = jax.tree.map(jnp.mean, s['structure'])
        if record:
            res['structural_state'] = dict(mask=s['mask'], W=s['W'], x=s['x'], row_caps=caps)
    if record:
        res["traj"] = traj
        if direct_weights is not None:
            res["direct_state"] = dict(W=s["W"], x=s["x"], Wp=s["Wp"])
    if diagnostics:
        res["events"] = {key: traj[key] for key in ("food", "poison", "alive_before", "alive_after")}
    if _return_weights:
        res["_weight_state"] = dict(birth_W=birth_W, mask=mask, initial_W=W,
                                    terminal_W=s["last_alive_W"], alive=s["alive"])
    if ecology is not None and (record or diagnostics):
        res["ecology"] = dict(shelter=shelter,
                              initial_sheltered=shelter[pos[:, 0], pos[:, 1]] & (carry["alive"] > 0),
                              sheltered=traj["sheltered"],
                              energy_lost=traj["ecology_energy_lost"])
    if persistence is not None and (record or diagnostics):
        res["persistence"] = traj["persistence"]
    return res


# ----------------------------------------------------------------------------- evolution
def make_evaluator(c, E, plastic=True, *, predictor_plastic=None, structural=False, ecology=None, persistence=None):
    """Evaluate a population of flat genomes in E worlds each (half with each rule)."""
    validate_persistence(persistence, ecology)
    options = {} if persistence is None else dict(persistence=persistence)
    _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
    rules = jnp.arange(E) % 2

    def one(flat, wkeys, dkeys):
        g = unravel(flat)
        r = jax.vmap(lambda wk, dk, ru: rollout(
            g, wk, dk, ru, c, plastic, predictor_plastic=predictor_plastic,
            structural=structural, ecology=ecology, **options))(wkeys, dkeys, rules)
        return r

    @jax.jit
    def evaluate(pop, wkeys, dkeys):
        return jax.vmap(one, in_axes=(0, None, 0))(pop, wkeys, dkeys)

    return evaluate, unravel


def centered_ranks(f):
    r = jnp.argsort(jnp.argsort(f))
    return r / (f.size - 1) - 0.5


def es_init(c, mean=None, seed=0):
    """Evolution state: the mean genome, Adam moments, step count and RNG key.
    Kept as plain arrays so it can be checkpointed and resumed exactly."""
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    flat0, _ = ravel_pytree(init_genome(c, k0))
    mu = flat0 if mean is None else jnp.asarray(mean)
    return dict(mu=mu, m=jnp.zeros_like(mu), v=jnp.zeros_like(mu), t=0, key=key)


def es_step(state, evaluate, pop, E, sigma, lr, l2=0.005):
    """One generation of OpenAI-style ES (antithetic sampling, rank shaping, Adam)."""
    assert pop % 2 == 0
    mu, m_, v_, key = state["mu"], state["m"], state["v"], state["key"]
    key, ke, kw, kd = jax.random.split(key, 4)
    eps = jax.random.normal(ke, (pop // 2, mu.size))
    eps = jnp.concatenate([eps, -eps])
    popu = mu + sigma * eps
    wkeys = jax.random.split(kw, E)                           # same worlds for everyone (fair test)
    dk = jax.random.split(kd, pop // 2 * E).reshape(pop // 2, E, 2)
    dkeys = jnp.concatenate([dk, dk])                         # antithetic twins share development noise
    res = evaluate(popu, wkeys, dkeys)
    if 'structure' in res:
        validate_structure(res['structure'])
    fit = res["fitness"].mean(1)
    w = centered_ranks(fit)
    grad = (w[:, None] * eps).sum(0) / (pop * sigma) - l2 * mu
    b1, b2 = 0.9, 0.999
    t = state["t"] + 1
    m_ = b1 * m_ + (1 - b1) * grad
    v_ = b2 * v_ + (1 - b2) * grad ** 2
    mu = mu + lr * (m_ / (1 - b1 ** t)) / (jnp.sqrt(v_ / (1 - b2 ** t)) + 1e-8)
    cnt = res["counts"].sum((0, 1))                           # (half, food/poison)
    acc = cnt[:, 0] / jnp.maximum(cnt.sum(1), 1)
    row = dict(gen=t - 1, fit_mean=float(fit.mean()), fit_best=float(fit.max()),
               fit_std=float(fit.std()), acc_first=float(acc[0]), acc_second=float(acc[1]),
               survived=float(res["survived"].mean()))
    if 'structure' in res:
        row['structure'] = {k: float(v.mean()) for k, v in res['structure'].items()}
    return dict(mu=mu, m=m_, v=v_, t=t, key=key), row


def format_row(row):
    return (f"gen {row['gen']:4d} | fitness mean {row['fit_mean']:6.2f} best {row['fit_best']:6.2f} | "
            f"food-choice accuracy 1st half {row['acc_first']:.2f} -> 2nd half {row['acc_second']:.2f} | "
            f"survived {row['survived']:.2f}")


def evolve(c, generations=200, pop=128, E=4, sigma=0.1, lr=0.1, seed=0, plastic=True,
           log_every=10, mean=None, *, structural=False, ecology=None, persistence=None):
    """Run ES for a number of generations. Returns (mean genome, unravel, history)."""
    _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
    options = {} if persistence is None else dict(persistence=persistence)
    evaluate, _ = make_evaluator(c, E, plastic, structural=structural, ecology=ecology, **options)
    state = es_init(c, mean, seed)
    hist = []
    for gen in range(generations):
        state, row = es_step(state, evaluate, pop, E, sigma, lr)
        hist.append(row)
        if log_every and (gen % log_every == 0 or gen == generations - 1):
            print(format_row(row))
    return state["mu"], unravel, hist


def test_genome(mu, c, n_worlds=64, plastic=True, seed=123, *, predictor_plastic=None, structural=False):
    """Fresh, unseen worlds. Returns averages over n_worlds (half with each rule)."""
    evaluate, _ = make_evaluator(c, n_worlds, plastic, predictor_plastic=predictor_plastic, structural=structural)
    key = jax.random.PRNGKey(seed)
    kw, kd = jax.random.split(key)
    res = evaluate(mu[None], jax.random.split(kw, n_worlds),
                   jax.random.split(kd, n_worlds)[None])
    cnt = res["counts"].sum((0, 1))
    acc = cnt[:, 0] / jnp.maximum(cnt.sum(1), 1)
    result = dict(fitness=float(res["fitness"].mean()), acc_first=float(acc[0]),
                acc_second=float(acc[1]), survived=float(res["survived"].mean()),
                eaten=int(cnt.sum()))
    if structural:
        validate_structure(res['structure'])
        result['structure'] = {k: float(v.mean()) for k, v in res['structure'].items()}
    return result
