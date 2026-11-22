"""E128: active empty-world retention, using the unchanged E127 operators.

One numeric row contains two initial-rule replays. Each ARMS prefix has birth,
pre-boundary and acquired state evidence; a full guided trace; a baseline_probe;
delay state evidence at RETENTION_TICKS; one retention trace; and probe/grid
readouts at each delay. The rule axis is first, followed (where applicable) by
delay, orientation or turn. State evidence uses E127's neural byte hashes and
full physical carry. No dense weight history is written.

The 1- and 33-turn states belong to ONE continuous ordinary action stream. Its
empty scene still supplies hunger, previous energy change, bias, motor feedback
and recurrent activity. Plastic learning and prediction remain active. Only
discarded local probes freeze updates; both local and grid readouts get E127's
same fresh physical body, while keeping the entire delayed neural state.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import encounter_diagnostic as encounter, seed


PROTOCOL = "retention-diagnostic-v1"
CFG = encounter.CFG
ARMS = ("plastic_stationary", "plastic_reversal", "direct_stationary", "direct_reversal")
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "retention")
ASSIGNMENT_TAG = 0xE128
RETENTION_TICKS = (0, 1, 33)
GRID_STEPS = encounter.GRID_STEPS
LOCAL_FIELDS = encounter.LOCAL_FIELDS
GRID_FIELDS = encounter.GRID_FIELDS


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves its reserved retention stream")
    return np.asarray(jax.random.split(jax.random.fold_in(encounter._master(master), int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if isinstance(count, bool) or count not in (4, 128):
        raise ValueError("E128 assigns four development or 128 final families")
    key = jax.random.fold_in(encounter._master(master), ASSIGNMENT_TAG)
    permutation = np.asarray(jax.random.permutation(key, count), np.int32)
    return dict(key=np.asarray(key), permutation=permutation,
                pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2)[permutation])


def _inputs(keys, index, pre_contacts, c):
    encounter._config(c)
    keys = np.asarray(keys)
    if (keys.shape != (len(KEY_ROLES), 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != len(KEY_ROLES)
            or isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG
            or isinstance(pre_contacts, bool) or not isinstance(pre_contacts, Integral) or pre_contacts not in (8, 16)):
        raise ValueError("invalid independent retention family keys/index or hidden boundary")
    return keys


def _trace_join(*traces):
    return jax.tree.map(lambda *values: np.concatenate([np.asarray(value) for value in values]), *traces)


def retention_states(flat, acquired, key, rule, controller, c=CFG):
    """Clone all carry, clear only the scene, and continue one action-key stream."""
    if controller not in ("plastic", "direct"):
        raise ValueError("retention requires an original plastic or direct controller")
    initial = dict(acquired, grid=jnp.zeros_like(acquired["grid"]), key=jnp.asarray(key))
    one, first = encounter._free_grid(controller, c, steps=1)(flat, initial, rule)
    last, remainder = encounter._free_grid(controller, c, steps=32)(flat, one, rule)
    return (initial, one, last), _trace_join(first, remainder)


def evaluate_world(plastic_flat, direct_flat, keys, index, pre_contacts, c=CFG):
    """Replay acquisition once/history, then clone only at registered readouts."""
    keys = _inputs(keys, index, pre_contacts, c)
    flats = {"plastic": encounter._genome(plastic_flat, "plastic", c),
             "direct": encounter._genome(direct_flat, "direct", c)}
    role = {name: jnp.asarray(key) for name, key in zip(KEY_ROLES, keys)}
    script = encounter.contact_script(keys[0], int(pre_contacts))
    grid, position = seed.init_world(role["grid"], c)
    probe_keys = jax.random.split(role["probe"], len(encounter.PLACEMENTS))
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(), pre_contacts=np.asarray(pre_contacts, np.int32),
        initial_rules=np.asarray([0, 1], np.int32), target_rules=np.asarray([1, 0], np.int32), script=script,
        probe_placements=encounter.PLACEMENTS.copy(), probe_keys=np.asarray(probe_keys),
        retention_ticks=np.asarray(RETENTION_TICKS, np.int32), grid_steps=np.asarray(GRID_STEPS, np.int32),
        plastic_genome_sha256=encounter.array_hash(plastic_flat), direct_genome_sha256=encounter.array_hash(direct_flat))
    arms = {arm: [] for arm in ARMS}
    for initial_rule in (0, 1):
        for controller, flat in flats.items():
            birth = encounter._birth(flat, controller, role["birth"], role["chamber"], c)
            pre, pre_trace = encounter._chamber(controller, c)(flat, birth,
                jnp.asarray(script[:pre_contacts]), initial_rule)
            baseline = encounter._probe_evidence(encounter._readouts(controller, c)(flat, pre, probe_keys))
            shared = {"birth_" + name: value for name, value in encounter._state_evidence(birth).items()}
            shared.update({"pre_" + name: value for name, value in encounter._state_evidence(pre).items()})
            shared.update({"baseline_probe_" + name: value for name, value in baseline.items()})
            for history in ("stationary", "reversal"):
                rule = initial_rule if history == "stationary" else 1 - initial_rule
                acquired, guided = pre, [pre_trace]
                # Preserve E127's acquisition scan boundaries and arithmetic.
                for start, stop in zip(encounter.CONTACTS[:-1], encounter.CONTACTS[1:]):
                    acquired, trace = encounter._chamber(controller, c)(flat, acquired,
                        jnp.asarray(script[pre_contacts + start:pre_contacts + stop]), rule)
                    guided.append(trace)
                states, retention = retention_states(flat, acquired, role["retention"], rule, controller, c)
                record = dict(shared)
                record.update({"guided_" + name: value for name, value in _trace_join(*guided).items()})
                record.update({"acquired_" + name: value for name, value in encounter._state_evidence(acquired).items()})
                record.update({"delay_" + name: value for name, value in
                    encounter._stack([encounter._state_evidence(state) for state in states]).items()})
                record.update({"retention_" + name: value for name, value in retention.items()})
                probes, grids = [], []
                for state in states:
                    probes.append(encounter._probe_evidence(encounter._readouts(controller, c)(flat, state, probe_keys)))
                    initial = encounter._fresh_body(state, grid, position, role["action"], c)
                    final, trace = encounter._free_grid(controller, c)(flat, initial, 1 - initial_rule)
                    evidence = {"initial_" + name: value for name, value in encounter._state_evidence(initial).items()}
                    evidence.update({"final_" + name: value for name, value in encounter._state_evidence(final).items()})
                    evidence.update({name: np.asarray(value) for name, value in trace.items()})
                    grids.append(evidence)
                record.update({"probe_" + name: value for name, value in encounter._stack(probes).items()})
                record.update({"grid_" + name: value for name, value in encounter._stack(grids).items()})
                arms[controller + "_" + history].append(record)
    for arm, records in arms.items():
        row.update({arm + "_" + name: value for name, value in encounter._stack(records).items()})
    validate_world(row, c)
    return row


def _require_same(actual, expected, message):
    if not encounter._identical(actual, expected):
        raise ValueError(message)


def _validate_trace(value, stem, time_axis, plastic):
    def field(name):
        return np.moveaxis(value[stem + "_" + name], time_axis, 0)

    action = field("action")
    _require_same(field("efference"), np.eye(5, dtype=np.float32)[action], "motor efference differs from actual action")
    _require_same(field("observation")[..., -2], np.float32(4) * field("incoming_denergy"), "actual energy feedback was not observed")
    _require_same(field("incoming_denergy")[1:], field("denergy")[:-1], "energy feedback broke between turns")
    _require_same(field("incoming_activity")[1:], field("activity")[:-1], "neural carry broke between turns")
    _require_same(field("key_before")[1:], field("key_after")[:-1], "action-key stream restarted")
    _require_same(field("logits"), np.float32(3) * field("decision_motor"), "logits differ from actual motors")
    encounter.validate_probabilities(field("probabilities"), field("logits"))
    if np.any(field("hebbian_enabled") != plastic) or np.any(field("predictor_enabled") != plastic):
        raise ValueError("retention experiment changed the existing learning switches")
    if stem != "guided":
        if np.any(field("forced_action") != -1) or not np.array_equal(action, field("sampled_action")):
            raise ValueError("autonomous life used an imposed action")


def _validate_probe(value, stem, snapshot, with_delay, c):
    p = value[stem + "probabilities"]
    shape = (2, 3, 12, 5) if with_delay else (2, 12, 5)
    if p.shape != shape:
        raise ValueError("missing rule, delay, placement or original action alternative")
    for name in encounter.NEURAL:
        expected = np.broadcast_to(value[snapshot + name + "_sha256"][..., None, :], shape[:-1] + (32,))
        _require_same(value[stem + "initial_" + name + "_sha256"], expected, "probe lost its complete neural snapshot")
    for name in ("W", "Wp", "mask"):
        _require_same(value[stem + "initial_" + name + "_sha256"], value[stem + "final_" + name + "_sha256"],
                      "discarded readout changed frozen state")
    if any(np.any(value[stem + field]) for field in ("hebbian_enabled", "predictor_enabled", "feedback_applied")):
        raise ValueError("discarded readout learned or supplied outcome feedback")
    _require_same(value[stem + "logits"], np.float32(3) * value[stem + "decision_motor"], "probe logits differ from actual motors")
    encounter.validate_probabilities(p, value[stem + "logits"])
    for field, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.), ("sig", 0.), ("t", 0), ("counts", 0.)):
        if np.any(value[stem + "initial_" + field] != expected):
            raise ValueError("local readout did not start with the matched fresh body")


def validate_world(row, c=CFG):
    """Check the new clone/continuation contract; independent audit checks physics."""
    pre, index = int(row["pre_contacts"]), int(row["base_index"])
    keys = _inputs(row["keys"], index, pre, c)
    for name, expected in (("initial_rules", [0, 1]), ("target_rules", [1, 0]), ("retention_ticks", RETENTION_TICKS)):
        _require_same(row[name], np.asarray(expected, np.int32), "changed registered rule or retention schedule")
    _require_same(row["probe_placements"], encounter.PLACEMENTS, "changed local probe placements")
    if int(row["grid_steps"]) != GRID_STEPS:
        raise ValueError("changed independent-grid horizon")
    for name, value in row.items():
        value = np.asarray(value)
        if value.dtype.kind not in "biuf" or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError("raw evidence must contain finite numeric arrays: " + name)
    script = np.asarray(row["script"])
    if script.shape != (pre + 16, 5) or not np.array_equal(script[:, 4], encounter.OPPOSITE[script[:, 3]]):
        raise ValueError("invalid physical contact script")
    if not np.all(np.sort(script[:, 2].reshape(-1, 2), axis=1) == [1, 2]):
        raise ValueError("a contact pair lost its color balance")
    for block in script.reshape(-1, 8, 5):
        if sorted(map(tuple, block[:, [2, 3]].tolist())) != [(color, direction) for color in (1, 2) for direction in range(1, 5)]:
            raise ValueError("contact block lost its color/direction balance")
    ticks = 2 * (pre + 16)
    reference_grid = encounter._arm(row, ARMS[0])
    for arm in ARMS:
        value, plastic = encounter._arm(row, arm), arm.startswith("plastic")
        if (value["guided_action"].shape != (2, ticks, 1) or value["retention_action"].shape != (2, 33, 1)
                or value["grid_action"].shape != (2, 3, GRID_STEPS, 1)):
            raise ValueError("lost an assigned acquisition, retention or grid turn")
        if not np.array_equal(value["guided_action"][:, :, 0], np.broadcast_to(script[:, [3, 4]].reshape(-1), (2, ticks))):
            raise ValueError("acquisition departed from its physical contact script")
        if (np.any(value["guided_alive_after"] != 1) or np.any(value["guided_unclipped_energy"] <= 0)
                or np.any(value["guided_unclipped_energy"] >= c.e_max)):
            raise ValueError("acquisition violated its original alive/no-clipping guarantee")
        for stem, axis in (("guided", 1), ("retention", 1), ("grid", 2)):
            _validate_trace(value, stem, axis, plastic)
        for field in ("grid_before", "grid", "food", "poison", "item"):
            if np.any(value["retention_" + field]):
                raise ValueError("retention interval contained an item or eating outcome")
        if np.any(value["retention_observation"][..., :-3]):
            raise ValueError("empty single-agent retention supplied a visual cue")
        for name in encounter.NEURAL:
            _require_same(value["delay_" + name + "_sha256"][:, 0], value["acquired_" + name + "_sha256"],
                          "empty-world entry changed neural carry")
            _require_same(value["grid_initial_" + name + "_sha256"], value["delay_" + name + "_sha256"],
                          "independent grid lost its full delayed neural carry")
        for name in encounter.BODY:
            if name not in ("grid", "key"):
                _require_same(value["delay_" + name][:, 0], value["acquired_" + name], "empty-world entry changed physical carry")
        _require_same(value["delay_key"][:, 0], np.broadcast_to(keys[6], (2, 2)), "wrong retention action stream")
        if np.any(value["delay_grid"]):
            raise ValueError("retention checkpoint is not an empty world")
        mapping = dict(pos="position", energy="energy", alive="alive_after", sig="signal", denergy="denergy", key="key_after")
        for ordinal, turn in enumerate(RETENTION_TICKS[1:], 1):
            for state, trace in mapping.items():
                _require_same(value["delay_" + state][:, ordinal], value["retention_" + trace][:, turn - 1],
                              "saved delay state is not the actual trajectory prefix")
            for rule in (0, 1):
                _require_same(value["delay_x_sha256"][rule, ordinal], encounter.array_hash(value["retention_activity"][rule, turn - 1]),
                              "delayed neural activity is not the actual prefix state")
        _require_same(value["delay_t"], value["acquired_t"][:, None] + np.asarray(RETENTION_TICKS, np.int32), "retention clock was reset")
        _require_same(value["delay_counts"], np.broadcast_to(value["acquired_counts"][:, None], value["delay_counts"].shape), "empty retention changed meal counts")
        _require_same(value["retention_incoming_denergy"][:, 0], value["acquired_denergy"], "first blank tick erased pending real feedback")
        for rule in (0, 1):
            _require_same(encounter.array_hash(value["retention_incoming_activity"][rule, 0]), value["acquired_x_sha256"][rule],
                          "first blank tick erased acquired activity")
        _validate_probe(value, "baseline_probe_", "pre_", False, c)
        _validate_probe(value, "probe_", "delay_", True, c)
        for stem in ("baseline_probe_", "probe_"):
            _require_same(value[stem + "initial_key"], np.broadcast_to(row["probe_keys"], value[stem + "initial_key"].shape),
                          "local readout keys differ across treatments")
        for field, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.), ("sig", 0.), ("t", 0), ("counts", 0.)):
            if np.any(value["grid_initial_" + field] != expected):
                raise ValueError("grid readout did not start with the matched fresh body")
        for field in ("grid", "pos", "key"):
            expected = np.broadcast_to(reference_grid["grid_initial_" + field][0, 0], value["grid_initial_" + field].shape)
            _require_same(value["grid_initial_" + field], expected, "fresh-grid conditions differ across histories or delays")
        _require_same(value["grid_initial_key"], np.broadcast_to(keys[4], value["grid_initial_key"].shape), "wrong independent-grid action key")
        for name in (("W", "Wp", "mask") if not plastic else ("mask",)):
            born = value["birth_" + name + "_sha256"]
            for stem in ("pre_", "acquired_", "delay_", "grid_final_"):
                actual = value[stem + name + "_sha256"]
                expected = np.broadcast_to(born[:, None], actual.shape) if actual.ndim == 3 else born
                _require_same(actual, expected, "fixed weights or structural mask changed")
    for controller in ("plastic", "direct"):
        left, right = [encounter._arm(row, controller + "_" + history) for history in ("stationary", "reversal")]
        for name in left:
            if name.startswith(("birth_", "pre_", "baseline_probe_")):
                _require_same(left[name], right[name], "matched histories did not clone the same acquired pre-boundary state")


def summary_input(row):
    """Small float64 per-rule outcomes; the runner owns all SEs and gates.

    `<arm>_pre_local_by_rule`: [2,5], toward the eventual reversed target.
    `<arm>_pre_old_by_rule`: [2], its signed preference for the original target.
    `<arm>_local_by_rule`: [2,3,5], in RETENTION_TICKS/LOCAL_FIELDS order.
    `<arm>_grid_by_rule`: [2,3,7], in RETENTION_TICKS/GRID_FIELDS order.
    No orientation or initial-rule replay is treated as an independent family.
    """
    validate_world(row)
    compact = {name: np.asarray(row[name]).copy() for name in
        ("base_index", "keys", "pre_contacts", "initial_rules", "retention_ticks", "plastic_genome_sha256", "direct_genome_sha256")}
    for arm in ARMS:
        value = encounter._arm(row, arm)
        probabilities = np.concatenate((value["baseline_probe_probabilities"][:, None], value["probe_probabilities"]), axis=1)
        local = encounter.local_outcomes(probabilities)
        compact[arm + "_pre_local_by_rule"] = local[:, 0]
        compact[arm + "_pre_old_by_rule"] = -local[:, 0, 4]
        compact[arm + "_local_by_rule"] = local[:, 1:]
        food, poison = (value["grid_" + name][..., 0].reshape(6, GRID_STEPS) for name in ("food", "poison"))
        contacted = (food | poison).any(1)
        first = (food | poison).argmax(1)
        first_food, first_poison = food[np.arange(6), first] & contacted, poison[np.arange(6), first] & contacted
        compact[arm + "_grid_by_rule"] = np.stack((first_food, first_poison, ~contacted,
            first_food.astype(int) - first_poison.astype(int), np.where(contacted, first + 1, GRID_STEPS + 1),
            (food.astype(int) - poison.astype(int)).sum(1), value["grid_alive_after"][..., -1, 0].reshape(6)), axis=-1).astype(np.float64).reshape(2, 3, len(GRID_FIELDS))
    return compact
