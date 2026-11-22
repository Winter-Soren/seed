"""E129: disable only Hebbian updates during one 33-turn empty interval.

Acquisition, active retention and fresh-body readouts are E128 helpers. The
observed frozen scan calls E127's same step and read-only frozen-weight path.
Neither the predictor nor autonomous action/feedback dynamics are disabled.
Readouts at 0 and 33 are standalone clones with ordinary learning restored.

Raw schema is E128's with six ARMS and two behavioral endpoints. The actual
one-turn continuation is recorded as `continuation_*`, without a behavioral
probe. Frozen traces additionally contain two bool[2,33] arrays named
`retention_W_input_matches_acquired` and `retention_W_output_matches_acquired`.
They compare actual float32 words to the acquired W, including signed zero.
No per-turn dense matrix is saved; independent audit checks the observation
path/source and endpoint hashes rather than replaying every neural operation.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import encounter_diagnostic as encounter, retention_diagnostic as retention, seed


PROTOCOL = "retention-intervention-v1"
CFG = retention.CFG
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_hebbian_off_stationary",
        "plastic_hebbian_off_reversal", "direct_stationary", "direct_reversal")
HEBBIAN_OFF_ARMS = ("plastic_hebbian_off_stationary", "plastic_hebbian_off_reversal")
KEY_ROLES = retention.KEY_ROLES
ASSIGNMENT_TAG = 0xE129
RETENTION_TICKS = (0, 33)
CONTINUATION_TICK = 1
GRID_STEPS = retention.GRID_STEPS
LOCAL_FIELDS = retention.LOCAL_FIELDS
GRID_FIELDS = retention.GRID_FIELDS


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves its reserved intervention stream")
    return np.asarray(jax.random.split(jax.random.fold_in(encounter._master(master), int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if isinstance(count, bool) or count not in (4, 128):
        raise ValueError("E129 assigns four development or 128 final families")
    key = jax.random.fold_in(encounter._master(master), ASSIGNMENT_TAG)
    permutation = np.asarray(jax.random.permutation(key, count), np.int32)
    return dict(key=np.asarray(key), permutation=permutation,
                pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2)[permutation])


def frozen_retention_states(flat, acquired, key, rule, c=CFG):
    """Exactly E128's 1+32 continuation; the only intervention is Hebbian=False."""
    initial = dict(acquired, grid=jnp.zeros_like(acquired["grid"]), key=jnp.asarray(key))
    one, first = encounter._free_grid("plastic", c, hebbian=False, steps=1,
        verify_weight_identity=True)(flat, initial, rule, acquired["W"])
    last, remainder = encounter._free_grid("plastic", c, hebbian=False, steps=32,
        verify_weight_identity=True)(flat, one, rule, acquired["W"])
    return (initial, one, last), retention._trace_join(first, remainder)


def _retention_evidence(states, trace):
    return {**{"delay_" + name: value for name, value in encounter._stack([
                encounter._state_evidence(states[0]), encounter._state_evidence(states[2])]).items()},
            **{"continuation_" + name: value for name, value in encounter._state_evidence(states[1]).items()},
            **{"retention_" + name: np.asarray(value) for name, value in trace.items()}}


def evaluate_world(plastic_flat, direct_flat, keys, index, pre_contacts, c=CFG):
    keys = retention._inputs(keys, index, pre_contacts, c, assignment_tag=ASSIGNMENT_TAG)
    flats = {"plastic": encounter._genome(plastic_flat, "plastic", c),
             "direct": encounter._genome(direct_flat, "direct", c)}
    role = {name: jnp.asarray(key) for name, key in zip(KEY_ROLES, keys)}
    script = encounter.contact_script(keys[0], int(pre_contacts))
    grid, position = seed.init_world(role["grid"], c)
    probe_keys = jax.random.split(role["probe"], len(encounter.PLACEMENTS))
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(), pre_contacts=np.asarray(pre_contacts, np.int32),
        initial_rules=np.asarray([0, 1], np.int32), target_rules=np.asarray([1, 0], np.int32), script=script,
        probe_placements=encounter.PLACEMENTS.copy(), probe_keys=np.asarray(probe_keys),
        retention_ticks=np.asarray(RETENTION_TICKS, np.int32), continuation_tick=np.asarray(CONTINUATION_TICK, np.int32),
        grid_steps=np.asarray(GRID_STEPS, np.int32), plastic_genome_sha256=encounter.array_hash(plastic_flat),
        direct_genome_sha256=encounter.array_hash(direct_flat))
    arms = {arm: [] for arm in ARMS}
    for initial_rule, controller, history, flat, rule, acquired, acquisition in retention._acquired_histories(
            flats, role, script, pre_contacts, probe_keys, c):
        states, trace = retention.retention_states(flat, acquired, role["retention"], rule, controller, c)
        normal_readouts = retention._readout_evidence(flat, (states[0], states[2]), grid, position,
            role["action"], probe_keys, 1 - initial_rule, controller, c)
        arms[controller + "_" + history].append({**acquisition, **_retention_evidence(states, trace), **normal_readouts})
        if controller == "plastic":
            frozen, frozen_trace = frozen_retention_states(flat, acquired, role["retention"], rule, c)
            delayed_readout = retention._readout_evidence(flat, (frozen[2],), grid, position,
                role["action"], probe_keys, 1 - initial_rule, controller, c)
            # Both treatments have the same immediate copy. Reuse its real
            # evidence; do not simulate a duplicate or let a probe feed back.
            readouts = {name: np.concatenate((normal_readouts[name][:1], value)) for name, value in delayed_readout.items()}
            arms["plastic_hebbian_off_" + history].append({**acquisition, **_retention_evidence(frozen, frozen_trace), **readouts})
    for arm, records in arms.items():
        row.update({arm + "_" + name: value for name, value in encounter._stack(records).items()})
    validate_world(row, c)
    return row


def validate_world(row, c=CFG):
    retention.validate_world(row, c, arms=ARMS, retention_ticks=RETENTION_TICKS,
                             hebbian_off=HEBBIAN_OFF_ARMS, assignment_tag=ASSIGNMENT_TAG)
    if not encounter._identical(row["continuation_tick"], np.asarray(CONTINUATION_TICK, np.int32)):
        raise ValueError("changed the registered continuation boundary")
    for arm in ARMS:
        value = encounter._arm(row, arm)
        for state, trace in dict(pos="position", energy="energy", alive="alive_after", sig="signal",
                                denergy="denergy", key="key_after", grid="grid").items():
            retention._require_same(value["continuation_" + state], value["retention_" + trace][:, 0],
                                    "the 32-turn continuation did not retain the first real turn")
        retention._require_same(value["continuation_t"], value["acquired_t"] + np.int32(1), "continuation clock reset")
        retention._require_same(value["continuation_counts"], value["acquired_counts"], "empty turn changed counts")
        for rule in (0, 1):
            retention._require_same(value["continuation_x_sha256"][rule], encounter.array_hash(value["retention_activity"][rule, 0]),
                                    "continuation lost the first neural state")
        if arm in HEBBIAN_OFF_ARMS:
            for field in ("W_input_matches_acquired", "W_output_matches_acquired"):
                proof = value["retention_" + field]
                if proof.shape != (2, 33) or proof.dtype != np.bool_ or not np.all(proof):
                    raise ValueError("frozen retention changed a carried weight representation")
            for stem in ("continuation_", "delay_"):
                actual, acquired = value[stem + "W_sha256"], value["acquired_W_sha256"]
                expected = np.broadcast_to(acquired[:, None], actual.shape) if actual.ndim == 3 else acquired
                retention._require_same(actual, expected, "frozen checkpoint differs from the acquired weights")
            normal = encounter._arm(row, "plastic_" + arm.rsplit("_", 1)[-1])
            for name, item in value.items():
                if name.startswith(("birth_", "pre_", "baseline_probe_", "guided_", "acquired_")):
                    retention._require_same(item, normal[name], "intervention contaminated the shared acquisition")
                elif name.startswith(("delay_", "probe_", "grid_")):
                    retention._require_same(item[:, 0], normal[name][:, 0], "immediate treatment clones differ")
            # Same random inputs and physical cost; actions may diverge because
            # of the intervention, so their equality is deliberately not required.
            for field in ("key_before", "key_after", "action_key", "regrowth_key", "energy", "denergy", "alive_after"):
                retention._require_same(value["retention_" + field], normal["retention_" + field],
                                        "retention treatment changed a paired random input or physical cost")
        elif arm.startswith("direct"):
            for name in ("W", "Wp", "mask"):
                retention._require_same(value["continuation_" + name + "_sha256"], value["birth_" + name + "_sha256"],
                                        "fixed direct state changed at the continuation boundary")


def summary_input(row):
    validate_world(row)
    return retention._compact_outcomes(row, arms=ARMS, retention_ticks=RETENTION_TICKS)
