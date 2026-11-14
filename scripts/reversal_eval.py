"""REVERSAL v1 prototype: matched hidden-rule reversal and compact recovery accounting."""
from collections import Counter
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed.seed import Cfg, init_genome, rollout

PROTOCOL = "reversal-v1"
SPLIT_SEEDS = {"development": 8210, "final": 8211}
CONDITIONS = (("noflip_learning", False, True), ("noflip_frozen", False, False),
              ("flip_learning", True, True), ("flip_frozen", True, False))
WINDOW, MIN_ENCOUNTERS, ACCURACY = 25, 5, 0.7


def validate_request(a, c):
    if getattr(a, "structural_plasticity", "off") not in (None, "off"):
        raise ValueError("reversal-v1's registered first panel is weight-only; structural mode must be off")
    if any(not math.isfinite(value) for value in c):
        raise ValueError("reversal-v1 configuration values must be finite")
    if a.generations != 0 or not a.start.startswith("file:") or a.resume or a.quick or a.no_plastic:
        raise ValueError("reversal-v1 requires saved file:, zero generations and no resume/quick/no-plastic")
    if c.T < 4 or c.T % 2 or c.G < 1 or c.M < 1 or c.H < 1 or c.n_htypes < 1 or c.H % c.n_htypes or c.R < 0:
        raise ValueError("reversal-v1 requires even T>=4 and valid world/brain dimensions")
    if c.n_items < 0 or 2*c.n_items > c.G*c.G or a.eval_worlds < 1:
        raise ValueError("reversal-v1 needs a valid item count and at least one world")
    if a.condition_split == "final":
        if c != Cfg() or a.eval_worlds != 128:
            raise ValueError("reversal-v1 final fixes default Cfg, T300/flip150, 128 worlds")
    elif c.T > 30 or c.M > 2 or c.H > 6 or c.R > 1 or c.G > 8 or a.eval_worlds > 4:
        raise ValueError("development uses tiny fixtures: T<=30, M<=2, H<=6, R<=1, G<=8, worlds<=4")


def summarize_events(events, boundary):
    """A window hit is a recovery proxy; censoring never becomes fake zero accuracy."""
    keys = ("food", "poison", "alive_before", "alive_after")
    arrays = {key: np.asarray(events[key]) for key in keys}
    shape = arrays["food"].shape
    if (len(shape) != 3 or min(shape) < 1 or type(boundary) is not int or not 0 < boundary < shape[1]
            or any(a.shape != shape or not np.isin(a, (0, 1)).all() for a in arrays.values())):
        raise ValueError("events must be matching binary (world,time,agent) arrays and an interior boundary")
    food, poison, before, after = (arrays[key].astype(bool) for key in keys)
    if ((food & poison).any() or ((food | poison) & ~before).any()
            or (after & ~before).any() or not np.array_equal(before[:, 1:], after[:, :-1])):
        raise ValueError("inconsistent encounter/survival events")
    worlds, steps, family = shape

    def counts(f, p):
        f, p = int(f.sum()), int(p.sum())
        return dict(food=f, poison=p, accuracy=f/(f+p) if f+p else None)

    windows = []
    for start in range(boundary, steps, WINDOW):
        stop = min(start+WINDOW, steps)
        windows.append(dict(start=start, stop=stop, complete=stop-start == WINDOW,
                            **counts(food[:, start:stop], poison[:, start:stop]),
                            survived=int(after[:, stop-1].sum())))
    agents = []
    for world in range(worlds):
        for agent in range(family):
            f, p = food[world, :, agent], poison[world, :, agent]
            at_risk = bool(before[world, boundary, agent])
            post = counts(f[boundary:], p[boundary:])
            latency, eligible = None, False
            if at_risk:
                for window in windows:
                    if not window["complete"]:
                        continue
                    sample = counts(f[window["start"]:window["stop"]], p[window["start"]:window["stop"]])
                    if sample["food"] + sample["poison"] >= MIN_ENCOUNTERS:
                        eligible = True
                        if sample["accuracy"] >= ACCURACY and after[world, window["stop"]-1, agent]:
                            latency = window["stop"] - boundary
                            break
            reason = (None if latency is not None else "dead_at_boundary" if not at_risk else
                      "died_before_window_hit" if not after[world, -1, agent] else
                      "no_post_encounters" if post["food"]+post["poison"] == 0 else
                      "no_complete_window" if not any(w["complete"] for w in windows) else
                      "criterion_not_met" if eligible else "insufficient_window_encounters")
            agents.append(dict(world=world, agent=agent, initial_rule=world % 2,
                               pre=counts(f[:boundary], p[:boundary]), post=post,
                               alive_at_boundary=at_risk, survived=bool(after[world, -1, agent]),
                               window_hit_latency=latency, censor_reason=reason))
    latencies = [a["window_hit_latency"] for a in agents if a["window_hit_latency"] is not None]
    risk = sum(a["alive_at_boundary"] for a in agents)
    total = counts(food, poison)
    return dict(fitness=(total["food"]-total["poison"])/len(agents), eaten=total["food"]+total["poison"],
                pre=counts(food[:, :boundary], poison[:, :boundary]),
                post=counts(food[:, boundary:], poison[:, boundary:]),
                post_fitness=(int(food[:, boundary:].sum())-int(poison[:, boundary:].sum()))/len(agents),
                alive_at_boundary=risk/len(agents), survived=float(after[:, -1].mean()),
                recovered_agents=len(latencies), agents_at_risk=risk, total_agents=len(agents),
                window_hit_fraction_all=len(latencies)/len(agents),
                window_hit_fraction_at_risk=len(latencies)/risk if risk else None,
                median_window_hit_latency=float(np.median(latencies)) if latencies else None,
                censored=dict(Counter(a["censor_reason"] for a in agents if a["censor_reason"])),
                post_windows=windows, agents=agents)


def evaluate_conditions(mu, c, n_worlds, split):
    seed, boundary = SPLIT_SEEDS[split], c.T//2
    kw, kd = jax.random.split(jax.random.PRNGKey(seed))
    wkeys, dkeys = jax.random.split(kw, n_worlds), jax.random.split(kd, n_worlds)
    _, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))
    results = {}
    for label, flip, plastic in CONDITIONS:
        @jax.jit
        def evaluate(flat, wk, dk):
            g = unravel(flat)
            return jax.vmap(lambda w, d, rule: rollout(
                g, w, d, rule, c, plastic=plastic, predictor_plastic=plastic,
                flip_step=boundary if flip else None, diagnostics=True))(wk, dk, jnp.arange(n_worlds) % 2)
        raw = evaluate(jnp.asarray(mu), wkeys, dkeys)
        metrics = summarize_events(raw["events"], boundary)
        food, poison = (np.asarray(raw["events"][key]) for key in ("food", "poison"))
        event_counts = np.stack([np.stack([f[:, start:stop].sum((1, 2)) for f in (food, poison)], 1)
                                 for start, stop in ((0, boundary), (boundary, c.T))], 1)
        net = event_counts[..., 0].sum(1)-event_counts[..., 1].sum(1)
        if not np.array_equal(event_counts, raw["counts"]) or not np.array_equal(net, np.asarray(raw["fitness"])*c.M):
            raise ValueError("event counts/fitness differ from rollout")
        results[label] = dict(flip_step=boundary if flip else None, hebbian=plastic,
                              predictor=plastic, structural=False, metrics=metrics)
    return results
