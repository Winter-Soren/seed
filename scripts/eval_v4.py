"""EVAL v4 paired worlds: individual outcomes across opposite food rules."""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct
from seed.seed import Cfg, genome_size, init_genome, init_world, rollout
from scripts import eval_v3

PROTOCOL = "v4"
EVAL_SEED = 2026111902
N_WORLDS = 256
SPLIT_SEEDS = {"development": 41, "final": EVAL_SEED}
REFERENCES = ("fixed_A", "fixed_B", "mixture", "random_walk", "greedy")
CONTROLLERS = ("rules-v0", direct.FORMAT, *REFERENCES)
CONDITIONS = ("stationary_r0", "stationary_r1", "reversal_r0", "reversal_r1")
CONTRAST_METRICS = ("stationary_fitness", "individual_min_return", "joint_survival",
                    "post_flip_return", "reversal_final_survival")
PAIRING_KEYS = ("wkeys", "dkeys", "initial_grid", "initial_pos", "agent_ids")


def _distribution(values):
    values, counts = np.unique(values, return_counts=True)
    return dict(values=values.tolist(), counts=counts.tolist())


def summarize_condition(events, boundary):
    summary = eval_v3.summarize_events(events)
    food, poison, before, after = (np.asarray(events[key], dtype=bool) for key in eval_v3.EVENT_KEYS)
    if type(boundary) is not int or not 0 < boundary < food.shape[1]:
        raise ValueError("paired boundary must be an interior integer")
    if not before[:, 0].all():
        raise ValueError("all starting individuals must be alive")
    eaten = food | poison
    deaths = before & ~after
    returns = food.sum(1, dtype=np.int64) - poison.sum(1, dtype=np.int64)
    at_flip = before[:, boundary]
    final_alive = after[:, -1]
    survival = eval_v3.mean_se(at_flip.mean(1))
    summary.update(survival_at_flip=survival["mean"], survival_at_flip_se=survival["se"],
                   deaths=int(deaths.sum()), zero_encounter_agents=int((~eaten.any(1)).sum()),
                   agent_return_distribution=_distribution(returns))
    for bin_ in summary["bins"]:
        start, stop = bin_["start"], bin_["stop"]
        no_meal = ~eaten[:, start:stop].any(1)
        at_start = before[:, start]
        bin_.update(deaths=int(deaths[:, start:stop].sum()),
                    zero_encounter_agents=int(no_meal.sum()),
                    agents_alive_at_start=int(at_start.sum()),
                    alive_at_start_without_encounters=int((no_meal & at_start).sum()))
    per_agent = dict(agent_return=returns, survival_at_flip=at_flip, final_alive=final_alive,
                     death_step=np.where(deaths.any(1), deaths.argmax(1), -1).astype(np.int32),
                     no_encounters=~eaten.any(1))
    return summary, per_agent


def summarize_pairs(raw):
    """Pair individual identities first, then average within each independent base world."""
    r0, r1 = (np.asarray(raw[f"stationary_r{i}_agent_return"], dtype=np.float64) for i in (0, 1))
    q0, q1 = (np.asarray(raw[f"reversal_r{i}_agent_return"], dtype=np.float64) for i in (0, 1))
    if (r0.ndim != 2 or min(r0.shape) < 1 or r0.shape[0] < 2
            or any(x.shape != r0.shape or not np.isfinite(x).all() for x in (r0, r1, q0, q1))):
        raise ValueError("paired returns must be matching finite (world,agent) arrays")
    for label in CONDITIONS:
        alive = np.asarray(raw[label + "_final_alive"])
        if alive.shape != r0.shape or not np.isin(alive, (0, 1)).all():
            raise ValueError("paired survival must be binary and match individual returns")
    minimum = np.minimum(r0, r1)
    joint = np.asarray(raw["stationary_r0_final_alive"]) & np.asarray(raw["stationary_r1_final_alive"])
    final_reversal = (np.asarray(raw["reversal_r0_final_alive"], dtype=float)
                      + np.asarray(raw["reversal_r1_final_alive"], dtype=float)) / 2
    post = []
    for rule in (0, 1):
        prefix = f"reversal_r{rule}_"
        food, poison = (np.asarray(raw[prefix + key]) for key in ("food", "poison"))
        boundary = food.shape[1] // 2
        post.append(food[:, boundary:].sum(1, dtype=np.int64) - poison[:, boundary:].sum(1, dtype=np.int64))
    vectors = dict(stationary_fitness=((r0 + r1)/2).mean(1),
                   individual_min_return=minimum.mean(1), joint_survival=joint.mean(1),
                   post_flip_return=((post[0] + post[1])/2).mean(1),
                   reversal_final_survival=final_reversal.mean(1),
                   reversal_fitness=((q0 + q1)/2).mean(1))
    summaries = {name: eval_v3.mean_se(vector) for name, vector in vectors.items()}
    summaries["individual_min_return"]["distribution"] = _distribution(minimum)
    arrays = {"paired_" + name: value for name, value in vectors.items()}
    arrays.update(paired_individual_minimum=minimum, paired_individual_joint_survival=joint)
    return summaries, arrays


def evaluate_controller(mu_or_none, c, n_worlds=N_WORLDS, eval_seed=EVAL_SEED, *,
                        controller="rules-v0", on_condition=None):
    if controller not in CONTROLLERS or type(n_worlds) is not int or n_worlds < 2:
        raise ValueError("EVAL v4 needs a known controller and at least two base worlds")
    if eval_seed not in SPLIT_SEEDS.values():
        raise ValueError("use the registered held-out seed or development seed 41")
    if eval_seed == EVAL_SEED:
        if c != Cfg() or n_worlds != N_WORLDS:
            raise ValueError("held-out EVAL v4 fixes default Cfg and 256 base worlds")
    elif (c.T > 30 or c.M > 4 or c.H > 6 or c.R > 1 or c.G > 8 or n_worlds > 4):
        raise ValueError("development uses only tiny fixtures")
    if (not all(np.isfinite(value) for value in c)
            or c.T < 4 or c.T % 2 or c.G < 1 or c.M < 1 or c.R < 0 or c.H < 1
            or c.n_htypes < 1 or c.H % c.n_htypes or c.n_items < 0 or 2*c.n_items > c.G*c.G):
        raise ValueError("invalid paired world dimensions or item count")
    if controller == "mixture" and c.M % 2:
        raise ValueError("specialist mixture requires an even family size")
    if controller in REFERENCES:
        if mu_or_none is not None:
            raise ValueError("reference policies do not accept a genome")
        genome, weights = None, None
    else:
        mu = np.asarray(mu_or_none)
        expected = direct.genome_size(c) if controller == direct.FORMAT else genome_size(c)
        if mu.shape != (expected,) or not np.issubdtype(mu.dtype, np.floating) or not np.isfinite(mu).all():
            raise ValueError("controller requires a finite floating-point genome of its explicit format")
        if controller == direct.FORMAT:
            genome, weights = None, direct.decode(jnp.asarray(mu), c)
        else:
            _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
            genome, weights = unravel(jnp.asarray(mu)), None
    kw, kd = jax.random.split(jax.random.PRNGKey(eval_seed))
    wkeys, dkeys = jax.random.split(kw, n_worlds), jax.random.split(kd, n_worlds)
    grids, positions = jax.vmap(lambda key: init_world(key, c))(wkeys)
    common = dict(wkeys=np.asarray(wkeys), dkeys=np.asarray(dkeys),
                  initial_grid=np.asarray(grids), initial_pos=np.asarray(positions),
                  agent_ids=np.broadcast_to(np.arange(c.M), (n_worlds, c.M)))
    if controller in ("fixed_A", "fixed_B", "mixture"):
        targets = (np.zeros(c.M, int) if controller == "fixed_A" else np.ones(c.M, int)
                   if controller == "fixed_B" else (np.arange(c.M) >= c.M//2).astype(int))
        common["specialist_targets"] = targets
    arrays, conditions = dict(common), {}
    boundary = c.T // 2
    for label in CONDITIONS:
        rule = int(label[-1])
        flip = boundary if label.startswith("reversal") else None
        if controller in REFERENCES:
            evaluate = jax.jit(jax.vmap(lambda w, d: eval_v3.reference_rollout(
                w, rule, c, policy=controller, flip_step=flip)))
        elif controller == direct.FORMAT:
            evaluate = jax.jit(jax.vmap(lambda w, d: rollout(
                None, w, d, rule, c, plastic=False, predictor_plastic=False,
                direct_weights=weights, flip_step=flip, diagnostics=True)))
        else:
            evaluate = jax.jit(jax.vmap(lambda w, d: rollout(
                genome, w, d, rule, c, plastic=True, predictor_plastic=True,
                structural=False, flip_step=flip, diagnostics=True)))
        chunks = []
        for start in range(0, n_worlds, eval_v3.BATCH_WORLDS):
            stop = min(start + eval_v3.BATCH_WORLDS, n_worlds)
            chunks.append(jax.tree.map(np.asarray, evaluate(wkeys[start:stop], dkeys[start:stop])))
        events = {key: np.concatenate([chunk["events"][key] for chunk in chunks])
                  for key in eval_v3.EVENT_KEYS}
        fitness = np.concatenate([chunk["fitness"] for chunk in chunks])
        metrics, per_agent = summarize_condition(events, boundary)
        if not np.array_equal(fitness, per_agent["agent_return"].mean(1)):
            raise ValueError("paired events disagree with engine fitness")
        metrics.update(initial_rule=rule, flip_step=flip, controller=controller)
        condition_raw = {label + "_" + key: value.astype(bool) for key, value in events.items()}
        condition_raw.update({label + "_" + key: value for key, value in per_agent.items()})
        condition_raw[label + "_fitness"] = fitness
        condition_raw[label + "_rule"] = np.full(n_worlds, rule, np.int32)
        arrays.update(condition_raw)
        conditions[label] = metrics
        if on_condition is not None:
            on_condition(label, metrics, {**common, **condition_raw})
    for rule in (0, 1):
        for key in eval_v3.EVENT_KEYS:
            if not np.array_equal(arrays[f"stationary_r{rule}_{key}"][:, :boundary],
                                  arrays[f"reversal_r{rule}_{key}"][:, :boundary]):
                raise ValueError("stationary and reversal pre-flip events are not identical")
    paired, paired_arrays = summarize_pairs(arrays)
    arrays.update(paired_arrays)
    return dict(conditions=conditions, paired=paired), arrays


def compare_panels(raw_by_panel):
    rows = []
    for seed in range(3):
        a, d = raw_by_panel[f"A{seed}"], raw_by_panel[f"direct{seed}"]
        for key in PAIRING_KEYS:
            if not np.array_equal(a[key], d[key]):
                raise ValueError(f"paired controller {key} mismatch")
        for metric in CONTRAST_METRICS:
            av, dv = (np.asarray(raw["paired_" + metric], dtype=float) for raw in (a, d))
            if (av.ndim != 1 or av.shape != dv.shape or len(av) < 2
                    or len(av) != len(a["wkeys"])):
                raise ValueError("contrast vectors must match the base worlds")
            delta = av - dv
            rows.append(dict(seed=seed, metric=metric, **eval_v3.mean_se(delta),
                             n_worlds=len(delta), per_world_difference=delta.tolist()))
    return rows
