"""Optional bounded shelter pursuit from local cues; no map or hidden-rule access."""
from dataclasses import dataclass
from numbers import Integral
from typing import ClassVar

import jax
import jax.numpy as jnp

from seed.ecology import EcologyV1


@dataclass(frozen=True)
class PersistenceV1:
    memory_steps: int = 4
    stall_steps: int = 2
    version: ClassVar[str] = "persistence-v1"

    def __post_init__(self):
        for name in ("memory_steps", "stall_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.memory_steps <= 16:
            raise ValueError("memory_steps must be between 1 and 16")
        if not 1 <= self.stall_steps <= self.memory_steps:
            raise ValueError("stall_steps must be between 1 and memory_steps")


def validate_persistence(persistence, ecology):
    if persistence is not None:
        if not isinstance(persistence, PersistenceV1) or not isinstance(ecology, EcologyV1):
            raise ValueError("persistence requires a PersistenceV1 setting and ecology-v1 observations")


def initial_memory(agents):
    zeros = jnp.zeros(agents, dtype=jnp.int32)
    return dict(offset=jnp.zeros((agents, 2), dtype=jnp.int32), remaining=zeros,
                last_distance=zeros, last_visible=zeros > 0, last_active=zeros > 0,
                last_action=zeros, stalls=zeros, attempts=zeros)


def bias_logits(logits, shelter_patch, cold, alive, memory, trait, moves):
    """Reweight existing actions by at most one logit, with two tactic attempts per goal.

    Inputs contain only the egocentric shelter cue, season, life status and own
    past actions. Remembered offsets use action odometry, not true coordinates.
    Observed progress means a smaller nearest-visible-shelter distance; it does
    not establish that a particular shelter pixel has been tracked.
    """
    radius = shelter_patch.shape[1] // 2
    axis = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    offsets = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    distance = jnp.abs(offsets).sum(1)
    cues = shelter_patch.reshape(logits.shape[0], -1) > 0
    visible = cues.any(1)
    nearest = offsets[jnp.argmin(jnp.where(cues, distance, 2 * radius + 1), axis=1)]
    predicted = memory["offset"] - moves[memory["last_action"]]
    target = jnp.where(visible[:, None], nearest, predicted)
    target_distance = jnp.abs(target).sum(1)
    sheltered = shelter_patch[:, radius, radius] > 0
    active = ((alive > 0) & cold & ~sheltered & (target_distance > 0)
              & (visible | (memory["remaining"] > 0)))
    comparable = active & memory["last_active"] & visible & memory["last_visible"]
    progress = comparable & (target_distance < memory["last_distance"])
    stalled = comparable & ~progress
    stalls = jnp.where(stalled, jnp.minimum(memory["stalls"] + 1, trait.stall_steps), 0)
    attempted = active & (stalls >= trait.stall_steps) & (memory["attempts"] < 2)

    row_action = jnp.where(target[:, 0] < 0, 1, 2)
    column_action = jnp.where(target[:, 1] < 0, 3, 4)
    row_first = jnp.abs(target[:, 0]) >= jnp.abs(target[:, 1])
    toward = jnp.where(row_first, row_action, column_action)
    other_progress = jnp.where(row_first, target[:, 1] != 0, target[:, 0] != 0)
    perpendicular = jnp.where(row_first, 3, 1) + memory["attempts"] % 2
    alternative = jnp.where(other_progress, jnp.where(row_first, column_action, row_action), perpendicular)
    alternative = jnp.where(toward != memory["last_action"], toward, alternative)
    preferred = jnp.where(attempted, alternative, toward)
    adjustment = (jax.nn.one_hot(preferred, 5, dtype=logits.dtype)
                  - attempted[:, None] * jax.nn.one_hot(memory["last_action"], 5, dtype=logits.dtype))
    adjusted = jnp.where(active[:, None], logits + adjustment, logits)

    remaining = jnp.where(visible, trait.memory_steps, jnp.maximum(memory["remaining"] - 1, 0))
    updated = dict(offset=jnp.where(active[:, None], target, 0), remaining=jnp.where(active, remaining, 0),
                   last_distance=jnp.where(active, target_distance, 0), last_visible=active & visible,
                   last_active=active, last_action=jnp.where(active, memory["last_action"], 0),
                   stalls=jnp.where(active & ~attempted, stalls, 0),
                   attempts=jnp.where(active, memory["attempts"] + attempted.astype(jnp.int32), 0))
    events = dict(active=active, remembered=active & ~visible, progress_observed=progress,
                  stalled=stalled, tactic_attempted=attempted)
    return adjusted, updated, events
