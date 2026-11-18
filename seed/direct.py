"""Versioned directly encoded recurrent controller with fixed lifetime weights."""
import jax
import jax.numpy as jnp
import numpy as np

from seed.seed import dims, rollout

FORMAT = "direct-rnn-v1"


def genome_size(c):
    _, sensors, _, neurons, _ = dims(c)
    return (neurons - sensors) * (neurons - 1)


def init_genome(c, key):
    neurons = dims(c)[3]
    return jax.random.normal(key, (genome_size(c),)) / jnp.sqrt(float(neurons - 1))


def decode(flat, c):
    flat = jnp.asarray(flat)
    if flat.shape != (genome_size(c),) or not np.issubdtype(flat.dtype, np.floating):
        raise ValueError("direct-rnn-v1 requires a flat floating-point vector of the registered size")
    _, sensors, _, neurons, _ = dims(c)
    legal = (np.arange(neurons)[:, None] >= sensors) & ~np.eye(neurons, dtype=bool)
    rows, columns = np.nonzero(legal)
    return jnp.zeros((neurons, neurons), dtype=flat.dtype).at[rows, columns].set(flat)


def make_evaluator(c, E):
    rules = jnp.arange(E) % 2

    def one(flat, wkeys, dkeys):
        weights = decode(flat, c)
        return jax.vmap(lambda wk, dk, rule: rollout(
            None, wk, dk, rule, c, plastic=False, predictor_plastic=False,
            direct_weights=weights))(wkeys, dkeys, rules)

    @jax.jit
    def evaluate(population, wkeys, dkeys):
        return jax.vmap(one, in_axes=(0, None, 0))(population, wkeys, dkeys)

    return evaluate, lambda flat: decode(flat, c)
