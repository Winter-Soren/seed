"""Bounded activity-driven growth/pruning among existing neurons (default off).

Matrices use (post, pre). Birth caps never change within a lifetime. This is a
fixed engineering rule, not an evolved rule or a biologically calibrated model.
"""
import jax.numpy as jnp
import numpy as np

MODE = 'activity-grow-prune-v1'
INTERVAL = 10
PRUNE_FRACTION = 0.01
STREAM_TAG = 107


class StructuralValidationError(ValueError):
    pass


def birth_caps(mask, legal):
    return jnp.minimum(jnp.sum(mask > 0, axis=-1) + 1, legal.sum(-1))


def rewire(W, mask, activity, legal, caps, scale, draws, *, wmax,
           alive=True, plastic=True, enabled=True, event=True):
    """One pure event; draws supplies one signed normal draw per target row.

    Validation errors are explicit outputs so jit/vmap can retain them. Public
    evaluation/training boundaries must call validate_structure before use.
    """
    present = mask > 0
    active = jnp.asarray(alive) & plastic & enabled & event
    score = jnp.abs(activity[:, None] * activity[None, :])
    weak = legal & present & (score == 0) & (jnp.abs(W) <= PRUNE_FRACTION * wmax)
    prune_index = jnp.argmin(jnp.where(weak, jnp.abs(W), jnp.inf), axis=1)
    rows = jnp.arange(W.shape[0])
    prune = active & jnp.any(weak, axis=1)
    removed = jnp.zeros_like(present).at[rows, prune_index].set(prune)
    after_prune = present & ~removed
    degree = after_prune.sum(1)
    # An isolated target needs a presynaptic bootstrap; coactivity alone is zero.
    score = jnp.where((degree == 0)[:, None], jnp.abs(activity)[None, :], score)
    candidates = legal & ~present & (score > 0)
    grow_index = jnp.argmax(jnp.where(candidates, score, -jnp.inf), axis=1)
    grow = active & (degree < caps) & jnp.any(candidates, axis=1)
    proposal = draws * scale[rows, grow_index]
    invalid = grow & ~jnp.isfinite(proposal)
    proposal = jnp.clip(proposal, -wmax, wmax)
    grow = grow & ~invalid & (proposal != 0)
    added = jnp.zeros_like(present).at[rows, grow_index].set(grow)
    new_mask = (after_prune | added).astype(mask.dtype)
    new_W = jnp.where(removed, 0., W)
    new_W = new_W.at[rows, grow_index].set(jnp.where(grow, proposal, new_W[rows, grow_index]))
    return new_W, new_mask, dict(
        added=jnp.sum((new_mask > 0) & ~present),
        removed=jnp.sum(present & (new_mask == 0)), invalid_proposals=invalid.sum())


def validate_structure(diagnostics):
    """Reject failed structural validation before accepting fitness or ES updates."""
    for key in ('invalid_proposals', 'cap_violations', 'illegal_edges', 'invalid_weights'):
        if key in diagnostics and np.any(np.asarray(diagnostics[key]) != 0):
            raise StructuralValidationError(f'structural validation failed: {key}={np.asarray(diagnostics[key]).sum()}')
