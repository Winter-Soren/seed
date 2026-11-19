"""Paired stationary-rule objectives for the unchanged plastic controller and ES."""
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed.seed import Cfg, init_genome, rollout

OBJECTIVES = ("paired-mean", "paired-min")


def make_evaluator(c, E, objective):
    """Return an ES evaluator over E base worlds, each replayed under both rules.

    Each replay starts a fresh lifetime from the same world and development keys.
    The minimum objective pairs starting individuals before averaging the family.
    Counts pool both replays; survival averages them. E counts pairs, not lifetimes.
    """
    if objective not in OBJECTIVES:
        raise ValueError("objective must be paired-mean or paired-min")
    if isinstance(E, bool) or not isinstance(E, Integral) or E < 1:
        raise ValueError("E must be a positive integer number of base worlds")
    if not isinstance(c, Cfg):
        raise ValueError("paired objectives require a valid Cfg")
    dimensions = (c.G, c.M, c.T, c.R, c.n_items, c.H, c.n_htypes, c.micro)
    if (any(isinstance(value, bool) or not isinstance(value, Integral) for value in dimensions)
            or min(c.G, c.M, c.H, c.n_htypes, c.micro) < 1 or c.T < 2
            or c.R < 0 or c.n_items < 0 or 2*c.n_items > c.G*c.G
            or c.H % c.n_htypes):
        raise ValueError("invalid paired world dimensions or item count")
    if (not all(np.isfinite(value) for value in c)
            or min(c.e0, c.e_max, c.wmax) <= 0 or c.e0 > c.e_max
            or min(c.decay, c.gain) < 0):
        raise ValueError("invalid paired world energy or weight bounds")
    flat0, unravel = ravel_pytree(init_genome(c, jax.random.PRNGKey(0)))

    def one(flat, wkeys, dkeys):
        genome = unravel(flat)

        def pair(wkey, dkey):
            replays = [rollout(
                genome, wkey, dkey, rule, c, plastic=True, predictor_plastic=True,
                structural=False, flip_step=None, diagnostics=True) for rule in (0, 1)]
            returns = [result["events"]["food"].sum(0) - result["events"]["poison"].sum(0)
                       for result in replays]
            individual = (jnp.minimum(*returns) if objective == "paired-min"
                          else (returns[0] + returns[1]) / 2)
            return dict(fitness=individual.mean(),
                        counts=replays[0]["counts"] + replays[1]["counts"],
                        survived=(replays[0]["survived"] + replays[1]["survived"]) / 2)

        return jax.vmap(pair)(wkeys, dkeys)

    @jax.jit
    def evaluate(population, wkeys, dkeys):
        if (population.ndim != 2 or population.shape[0] < 1 or population.shape[1] != flat0.size
                or not jnp.issubdtype(population.dtype, jnp.floating)):
            raise ValueError("population requires flat floating-point rule genomes")
        if wkeys.shape[:1] != (E,) or dkeys.shape[:2] != (population.shape[0], E):
            raise ValueError("key batches must match the population and E base worlds")
        return jax.vmap(one, in_axes=(0, None, 0))(population, wkeys, dkeys)

    return evaluate, unravel
