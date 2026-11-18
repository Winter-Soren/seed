"""EVAL v3: per-agent learning, survival-conditioned bins and paired reference policies."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed.seed import Cfg, MOVES, init_genome, init_world, observe, rollout
from seed import direct

PROTOCOL = "v3"
EVAL_SEED = 2026111901
N_WORLDS = 256
MEALS = (1, 2, 3, 5, 10, 20)
BIN_WIDTH = 50
BATCH_WORLDS = 16
EVENT_KEYS = ("food", "poison", "alive_before", "alive_after")


def validate_request(a, c):
    if (a.generations != 0 or not a.start.startswith("file:") or a.resume or a.quick
            or a.no_plastic or getattr(a, "structural_plasticity", "off") != "off"
            or getattr(a, "controller", None) not in (None, "rules-v0")
            or c != Cfg() or a.eval_worlds != N_WORLDS
            or getattr(a, "condition_split", None) is not None or getattr(a, "cfg", None)):
        raise ValueError("EVAL v3 requires saved file:, zero generations, default Cfg, 256 worlds, "
                         "and no resume/quick/no-plastic/structural changes, condition split or Cfg overrides")


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected a finite nonempty per-world vector")
    return dict(mean=float(values.mean()),
                se=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def ratio_se(numerator, denominator):
    """Ratio of sums, with a paired delta-method SE clustered by world."""
    numerator, denominator = (np.asarray(x, dtype=np.float64) for x in (numerator, denominator))
    if (numerator.ndim != 1 or not len(numerator) or numerator.shape != denominator.shape
            or not np.isfinite(numerator).all() or not np.isfinite(denominator).all()):
        raise ValueError("ratio inputs must be finite matching per-world vectors")
    denom = denominator.mean()
    if denom == 0:
        return dict(mean=None, se=None)
    value = float(numerator.mean() / denom)
    residual = (numerator - value * denominator) / denom
    return dict(mean=value, se=mean_se(residual)["se"])


def _accuracy(food, poison):
    worlds = food.shape[0]
    f = food.reshape(worlds, -1).sum(1, dtype=np.int64)
    p = poison.reshape(worlds, -1).sum(1, dtype=np.int64)
    estimate = ratio_se(f, f + p)
    return dict(accuracy=estimate["mean"], accuracy_se=estimate["se"], food=int(f.sum()),
                poison=int(p.sum()), encounters=int((f + p).sum()),
                contributing_worlds=int(np.count_nonzero(f + p)))


def summarize_events(events, *, bin_width=BIN_WIDTH):
    arrays = {key: np.asarray(events[key]) for key in EVENT_KEYS}
    shape = arrays["food"].shape
    if (len(shape) != 3 or min(shape) < 1 or type(bin_width) is not int or bin_width < 1
            or any(x.shape != shape or not np.isin(x, (0, 1)).all() for x in arrays.values())):
        raise ValueError("events must be matching binary (world,time,agent) arrays")
    food, poison, before, after = (arrays[key].astype(bool) for key in EVENT_KEYS)
    if ((food & poison).any() or ((food | poison) & ~before).any() or (after & ~before).any()
            or not np.array_equal(before[:, 1:], after[:, :-1])):
        raise ValueError("inconsistent encounter/survival events")
    worlds, steps, agents = shape
    eaten = food | poison
    ordinal = np.cumsum(eaten, axis=1)
    fitness = (food.sum((1, 2), dtype=np.int64) - poison.sum((1, 2), dtype=np.int64)) / agents
    fit = mean_se(fitness)
    survival = mean_se(after[:, -1].mean(1))
    meals = []
    for meal in MEALS:
        select = eaten & (ordinal == meal)
        counts = _accuracy(food & select, poison & select)
        meals.append(dict(meal=meal, eligible_agents=int(select.sum()),
                          total_agents=worlds * agents, **counts))
    first = (food & eaten & (ordinal == 1)).any(1)
    twentieth = (food & eaten & (ordinal == 20)).any(1)
    eligible = eaten.sum(1) >= 20
    difference = ratio_se(((twentieth.astype(int) - first) * eligible).sum(1), eligible.sum(1))
    bins, survivor_bins = [], []
    survivors = after[:, -1, None, :]
    for start in range(0, steps, bin_width):
        stop = min(start + bin_width, steps)
        for result, keep in ((bins, np.ones_like(survivors)), (survivor_bins, survivors)):
            f, p = food[:, start:stop] & keep, poison[:, start:stop] & keep
            result.append(dict(start=start, stop=stop, complete=stop-start == bin_width,
                               contributing_agents=int((f | p).any(1).sum()), **_accuracy(f, p)))
    return dict(fitness=fit["mean"], fitness_se=fit["se"], n_worlds=worlds,
                total_agents=worlds * agents, survived=survival["mean"], survived_se=survival["se"],
                food=int(food.sum()), poison=int(poison.sum()), eaten=int(eaten.sum()),
                meal_accuracy=meals, bins=bins, survivor_bins=survivor_bins,
                meal20_minus_meal1_matched=dict(difference=difference["mean"], se=difference["se"],
                                               eligible_agents=int(eligible.sum()),
                                               contributing_worlds=int(np.count_nonzero(eligible.sum(1)))))


def greedy_logits(obs, rule, c):
    """Local rule-knower: approach visible food, avoiding an immediately poisonous cell."""
    width = 2 * c.R + 1
    patches = obs[:, :2 * width * width].reshape(c.M, 2, width, width)
    food = patches[:, rule]
    poison = patches[:, 1 - rule]
    offsets = jnp.stack(jnp.meshgrid(jnp.arange(-c.R, c.R + 1),
                                   jnp.arange(-c.R, c.R + 1), indexing="ij"), -1).reshape(-1, 2)
    distance = jnp.abs(offsets[None, :, :] - MOVES[:, None, :]).sum(-1)
    nearest = jnp.where(food.reshape(c.M, 1, -1) > 0, distance[None], jnp.inf).min(-1)
    nearest = jnp.where(jnp.isfinite(nearest), nearest, 0.)
    destinations = MOVES + c.R
    visible = ((destinations >= 0) & (destinations < width)).all(1)
    clipped = jnp.clip(destinations, 0, width - 1)
    dangerous = (poison[:, clipped[:, 0], clipped[:, 1]] > 0) & visible
    safe = ~dangerous
    safe = jnp.where(safe.any(1, keepdims=True), safe, True)
    cost = jnp.where(safe, nearest, jnp.inf)
    return jnp.where(cost == cost.min(1, keepdims=True), 0., -jnp.inf)


@partial(jax.jit, static_argnames=("c", "policy", "flip_step"))
def reference_rollout(wkey, rule, c, *, policy, flip_step=None):
    if policy not in ("random_walk", "greedy"):
        raise ValueError("unknown reference policy")
    if flip_step is not None and (type(flip_step) is not int or not 0 < flip_step < c.T):
        raise ValueError("flip_step must be an interior integer")
    grid, pos = init_world(wkey, c)
    carry = dict(grid=grid, pos=pos, energy=jnp.full((c.M,), c.e0), alive=jnp.ones(c.M),
                 denergy=jnp.zeros(c.M), key=wkey, t=0)

    def step(s, _):
        key, ka, kr = jax.random.split(s["key"], 3)
        current_rule = rule if flip_step is None else jnp.where(s["t"] >= flip_step, 1-rule, rule)
        if policy == "greedy":
            obs = observe(s["grid"], s["pos"], s["alive"], jnp.zeros(c.M),
                          s["energy"], s["denergy"], c)
            logits = greedy_logits(obs, current_rule, c)
        else:
            logits = jnp.zeros((c.M, 5))
        act = jax.random.categorical(ka, logits)
        alive = s["alive"]
        pos = (s["pos"] + MOVES[act] * alive[:, None].astype(jnp.int32)) % c.G
        item = s["grid"][pos[:, 0], pos[:, 1]]
        eat = (alive > 0) & (item > 0)
        food, poison = eat & (item == 1 + current_rule), eat & (item == 2 - current_rule)
        energy = jnp.clip(s["energy"] - c.decay * alive + c.gain * food - c.gain * poison, 0., c.e_max)
        denergy = (energy - s["energy"]) * alive
        alive_after = alive * (energy > 0)
        grid = s["grid"].at[pos[:, 0], pos[:, 1]].set(jnp.where(eat, 0, item))
        newc = jax.random.randint(kr, (c.M, 2), 0, c.G)
        grid = grid.at[newc[:, 0], newc[:, 1]].set(jnp.where(eat, item, grid[newc[:, 0], newc[:, 1]]))
        state = dict(grid=grid, pos=pos, energy=energy, alive=alive_after,
                     denergy=denergy, key=key, t=s["t"] + 1)
        return state, dict(food=food, poison=poison, alive_before=alive, alive_after=alive_after)

    _, events = jax.lax.scan(step, carry, None, length=c.T)
    fitness = (events["food"].sum() - events["poison"].sum()) / c.M
    return dict(events=events, fitness=fitness)


def evaluate_conditions(mu, c, n_worlds=N_WORLDS, eval_seed=EVAL_SEED, *, controller="rules-v0"):
    if n_worlds < 2 or not np.isfinite(np.asarray(mu)).all():
        raise ValueError("EVAL v3 requires at least two worlds and a finite genome")
    if controller not in ("rules-v0", direct.FORMAT):
        raise ValueError("unknown controller format")
    is_direct = controller == direct.FORMAT
    if eval_seed == EVAL_SEED and (c != Cfg() or n_worlds != N_WORLDS):
        raise ValueError("held-out EVAL v3 seed is reserved for the complete registered panel")
    kw, kd = jax.random.split(jax.random.PRNGKey(eval_seed))
    wkeys, dkeys = jax.random.split(kw, n_worlds), jax.random.split(kd, n_worlds)
    rules = jnp.arange(n_worlds) % 2
    if is_direct:
        genome = None
        weights = direct.decode(jnp.asarray(mu), c)
    else:
        _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
        genome = unravel(jnp.asarray(mu))
    results, arrays = {}, dict(wkeys=np.asarray(wkeys), dkeys=np.asarray(dkeys), rules=np.asarray(rules))
    for condition, flip in (("stationary", None), ("reversal", c.T // 2)):
        panels = {}
        for controller in ("genome", "random_walk", "greedy"):
            if controller == "genome" and is_direct:
                evaluate = jax.jit(jax.vmap(lambda w, d, r: rollout(
                    None, w, d, r, c, plastic=False, predictor_plastic=False,
                    direct_weights=weights, flip_step=flip, diagnostics=True)))
            elif controller == "genome":
                evaluate = jax.jit(jax.vmap(lambda w, d, r: rollout(
                    genome, w, d, r, c, plastic=True, predictor_plastic=True,
                    structural=False, flip_step=flip, diagnostics=True)))
            else:
                evaluate = jax.jit(jax.vmap(lambda w, d, r: reference_rollout(
                    w, r, c, policy=controller, flip_step=flip)))
            chunks = []
            for start in range(0, n_worlds, BATCH_WORLDS):
                stop = min(start + BATCH_WORLDS, n_worlds)
                raw = evaluate(wkeys[start:stop], dkeys[start:stop], rules[start:stop])
                chunks.append(jax.tree.map(np.asarray, raw))
            events = {key: np.concatenate([chunk["events"][key] for chunk in chunks]) for key in EVENT_KEYS}
            fitness = np.concatenate([chunk["fitness"] for chunk in chunks])
            reconstructed = (events["food"].sum((1, 2), dtype=np.int64)
                             - events["poison"].sum((1, 2), dtype=np.int64)) / c.M
            if not np.array_equal(fitness, reconstructed):
                raise ValueError("recorded events disagree with engine fitness")
            metrics = summarize_events(events)
            metrics.update(flip_step=flip, controller=controller)
            panels[controller] = metrics
            prefix = f"{condition}_{controller}_"
            arrays[prefix + "fitness"] = fitness
            arrays.update({prefix + key: value.astype(bool) for key, value in events.items()})
        greedy = arrays[f"{condition}_greedy_fitness"]
        for controller, metrics in panels.items():
            fitness = arrays[f"{condition}_{controller}_fitness"]
            fraction = ratio_se(fitness, greedy)
            difference = mean_se(np.asarray(fitness, dtype=np.float64) - greedy)
            metrics.update(greedy_fraction=fraction["mean"], greedy_fraction_se=fraction["se"],
                           fitness_minus_greedy=difference["mean"], fitness_minus_greedy_se=difference["se"])
        results[condition] = panels
    return results, arrays
