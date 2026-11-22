"""Default-off E131 action quota, using only age and observed realized energy.

No item labels, world rule, reversal time, future outcome or remaining horizon
is an input to the decision. The native sampler remains outside this module.
"""
import jax
import jax.numpy as jnp
import numpy as np

BLOCK = 25
THRESHOLD = 12
MODES = ("adaptive", "random_quota")
SCHEDULE_ROLES = ("phase", "direction", "offset")


def make_schedule(key, agents, steps):
    blocks = (steps + BLOCK - 1) // BLOCK
    streams = jax.random.split(jnp.asarray(key), 3)
    phase_keys = jax.random.split(streams[0], agents)
    direction_keys = jax.random.split(streams[1], blocks * agents).reshape(blocks, agents, 2)
    offset_keys = jax.random.split(streams[2], blocks * agents).reshape(blocks, agents, 2)

    def draws(keys, low, high):
        return np.asarray(jax.vmap(lambda k: jax.random.randint(k, (), low, high, dtype=jnp.int32))(
            keys.reshape(-1, 2))).reshape(keys.shape[:-1])

    return dict(stream_keys=np.asarray(streams), phase_keys=np.asarray(phase_keys),
        direction_keys=np.asarray(direction_keys), offset_keys=np.asarray(offset_keys),
        phase_delay=draws(phase_keys, 0, BLOCK), directions=draws(direction_keys, 1, 5),
        quota_offsets=draws(offset_keys, 0, BLOCK))


def validate_schedule(schedule, agents, steps):
    blocks = (steps + BLOCK - 1) // BLOCK
    if schedule is None or set(schedule) != {"phase_delay", "directions", "quota_offsets"}:
        raise ValueError("exploration requires the independent preassigned schedule")
    for name, shape in (("phase_delay", (agents,)), ("directions", (blocks, agents)), ("quota_offsets", (blocks, agents))):
        if schedule[name].shape != shape or schedule[name].dtype != jnp.int32:
            raise ValueError("exploration schedule changed its fixed shape or integer encoding")


def initial_state(agents):
    return dict(counter=jnp.zeros(agents, jnp.int32), spent=jnp.zeros(agents, bool))


def inactive_evidence(ordinary_action):
    zeros = jnp.zeros_like(ordinary_action)
    false = jnp.zeros_like(ordinary_action, dtype=bool)
    return dict(counter_before=zeros, counter=zeros, spent_before=false, spent=false,
        active=false, block_index=zeros - 1, block_offset=zeros - 1, direction=zeros,
        quota_offset=zeros - 1, due=false, override=false)


def choose_action(mode, state, schedule, age, observed_denergy, alive, ordinary_action):
    """Spend at most once per phased block; positive feedback is seen next turn."""
    if mode not in MODES:
        raise ValueError("unknown bounded exploration policy")
    next_counter = jnp.where(observed_denergy > 0, 0, jnp.minimum(THRESHOLD, state["counter"] + 1))
    counter = jnp.where((alive > 0) & (age > 0), next_counter, state["counter"])
    elapsed = age - schedule["phase_delay"]
    active = elapsed >= 0
    block = jnp.where(active, elapsed // BLOCK, -1)
    offset = jnp.where(active, elapsed % BLOCK, -1)
    # Before the public phase starts, the first schedule entry is read but
    # cannot act. There is no final-block or remaining-horizon adjustment.
    index = jnp.maximum(block, 0)
    animal = jnp.arange(ordinary_action.shape[0])
    direction = schedule["directions"][index, animal]
    quota_offset = schedule["quota_offsets"][index, animal]
    spent_before = jnp.where(active & (offset == 0), False, state["spent"])
    due = ((counter >= THRESHOLD) | (offset == BLOCK - 1)) if mode == "adaptive" else offset == quota_offset
    override = active & (alive > 0) & ~spent_before & due
    spent = spent_before | override
    executed = jnp.where(override, direction, ordinary_action)
    evidence = dict(counter_before=state["counter"], counter=counter, spent_before=spent_before, spent=spent,
        active=active, block_index=block, block_offset=offset, direction=direction,
        quota_offset=quota_offset, due=due, override=override)
    return executed, dict(counter=counter, spent=spent), evidence
