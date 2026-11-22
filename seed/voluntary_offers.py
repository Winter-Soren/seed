"""E132: voluntary sparse offers, followed by discarded local and solo probes.

The apparatus changes item placement, never the sampled action or neural rule.
Dead carries are absorbing. External schedule slots continue, but their frozen
internal clock/key and last observed feedback are not advanced or replayed.
"""
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import seed, encounter_diagnostic as encounter


PROTOCOL = "voluntary-offers-v1"
CFG = encounter.CFG
ARMS = ("near_stationary", "near_reversal", "sham_stationary", "sham_reversal")
KEY_ROLES = ("birth", "geometry", "exposure_action", "solo_world", "solo_action", "probe")
ASSIGNMENT_TAG = 0xE132
BLOCK_STEPS, PRE_BLOCKS, POST_BLOCKS, GRID_STEPS = 8, 8, 8, 50
TRACE_FIELDS = (
    "grid_before", "grid", "position_before", "position", "energy_before", "energy",
    "unclipped_energy", "signal_before", "signal", "incoming_denergy", "denergy",
    "alive_before", "alive_after", "item", "food", "poison", "observation",
    "decision_motor", "logits", "probabilities", "sampled_action", "action", "action_key",
    "regrowth_key", "regrowth_cell", "key_before", "key_after", "efference", "t", "rule",
    "hebbian_enabled", "predictor_enabled")
PROBE_FIELDS = ("observation", "decision_motor", "logits", "probabilities", "sampled_action",
                "action_key", "hebbian_enabled", "predictor_enabled", "feedback_applied")


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("family index leaves the E132 reserved stream")
    root = jax.random.fold_in(encounter._master(master), ASSIGNMENT_TAG)
    return np.asarray(jax.random.split(jax.random.fold_in(root, int(index)), len(KEY_ROLES)))


def assignments_manifest(master, count):
    if count not in (4, 64):
        raise ValueError("E132 assigns four development or 64 final families")
    return dict(family_keys=np.stack([evaluation_keys(master, i) for i in range(count)]),
                initial_rules=np.asarray([0, 1], np.int32))


def geometry(key, c=CFG):
    """Ordered A/B offsets use independent near/sham draws, shared by all copies."""
    block_keys = jax.random.split(jnp.asarray(key), PRE_BLOCKS + POST_BLOCKS)

    def one(block_key):
        near_key, sham_key = jax.random.split(block_key)
        near = seed.MOVES[1:][jax.random.permutation(near_key, 4)[:2]] % c.G
        cells = jax.random.permutation(sham_key, c.G * c.G - 1)[:2] + 1
        sham = jnp.stack((cells // c.G, cells % c.G), axis=1)
        return near, sham

    near, sham = jax.vmap(one)(block_keys)
    return dict(geometry_block_keys=np.asarray(block_keys), geometry_near_offsets=np.asarray(near, np.int32),
                geometry_sham_offsets=np.asarray(sham, np.int32))


def _inputs(keys, index, c):
    keys = np.asarray(keys)
    if (c != CFG or keys.shape != (len(KEY_ROLES), 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != len(KEY_ROLES)
            or isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG):
        raise ValueError("E132 requires its fixed one-creature dimensions and independent family roles")


def _same_bits(left, right):
    return jnp.all(jax.lax.bitcast_convert_type(left, jnp.uint32)
                   == jax.lax.bitcast_convert_type(right, jnp.uint32))


def _tick(state, genome, rule, external_tick, c, regrowth):
    """Use the unchanged voluntary E127 transition only for an alive creature."""
    def live(s):
        final, raw = encounter._step(s, genome, rule, -1, "plastic", c, regrowth=regrowth)
        return final, {name: raw[name] for name in TRACE_FIELDS}

    def dead(s):
        zeros = jnp.zeros((1,), jnp.float32)
        motors = jnp.zeros((1, 5), jnp.float32)
        # No observation, decision or physical transition is executed in this slot.
        raw = dict(grid_before=s["grid"], grid=s["grid"], position_before=s["pos"], position=s["pos"],
            energy_before=s["energy"], energy=s["energy"], unclipped_energy=s["energy"],
            signal_before=s["sig"], signal=s["sig"], incoming_denergy=s["denergy"], denergy=zeros,
            alive_before=s["alive"], alive_after=s["alive"], item=jnp.zeros((1,), jnp.int32),
            food=jnp.zeros((1,), bool), poison=jnp.zeros((1,), bool),
            observation=jnp.zeros((1, seed.dims(c)[1]), jnp.float32), decision_motor=motors,
            logits=motors, probabilities=motors, sampled_action=jnp.full((1,), -1, jnp.int32),
            action=jnp.full((1,), -1, jnp.int32), action_key=jnp.zeros((2,), jnp.uint32),
            regrowth_key=jnp.zeros((2,), jnp.uint32), regrowth_cell=jnp.zeros((1, 2), jnp.int32),
            key_before=s["key"], key_after=s["key"], efference=s["x"][:, -6:-1], t=s["t"],
            rule=jnp.asarray(rule, jnp.int32), hebbian_enabled=jnp.asarray(False),
            predictor_enabled=jnp.asarray(False))
        return s, raw

    final, trace = jax.lax.cond(state["alive"][0] > 0, live, dead, state)
    return final, dict(trace, skipped=state["alive"][0] <= 0,
        external_tick=jnp.asarray(external_tick, jnp.int32), incoming_efference=state["x"][:, -6:-1],
        carried_denergy=final["denergy"])


@lru_cache(None)
def _offers(c):
    @jax.jit
    def run(flat, initial, offsets, rule, start):
        genome = encounter._unravel(c)(flat)

        def block(state, schedule):
            ordinal, pair = schedule
            cells = (state["pos"][0] + pair) % c.G
            grid = jnp.zeros((c.G, c.G), jnp.int32).at[cells[:, 0], cells[:, 1]].set(jnp.asarray([1, 2], jnp.int32))
            state = dict(state, grid=jnp.where(state["alive"][0] > 0, grid, state["grid"]))
            return jax.lax.scan(lambda s, tick: _tick(s, genome, rule, tick, c, False),
                state, start + ordinal * BLOCK_STEPS + jnp.arange(BLOCK_STEPS, dtype=jnp.int32))

        final, trace = jax.lax.scan(block, initial, (jnp.arange(offsets.shape[0], dtype=jnp.int32), offsets))
        return final, jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trace)
    return run


@lru_cache(None)
def _ticks(c, steps, regrowth):
    @jax.jit
    def run(flat, initial, rule, start):
        genome = encounter._unravel(c)(flat)
        return jax.lax.scan(lambda s, tick: _tick(s, genome, rule, tick, c, regrowth),
            initial, start + jnp.arange(steps, dtype=jnp.int32))
    return run


def _flush(flat, state, rule, external_tick, c):
    # A discarded/live flush clears only the apparatus; a dead state stays exact.
    initial = dict(state, grid=jnp.where(state["alive"][0] > 0, jnp.zeros_like(state["grid"]), state["grid"]))
    return _ticks(c, 1, False)(flat, initial, rule, external_tick)


@lru_cache(None)
def _compact_readouts(c):
    @jax.jit
    def run(flat, snapshot, keys):
        raw = encounter._readouts("plastic", c)(flat, snapshot, keys)
        out = {name: raw[name][:, 0] if name in ("decision_motor", "logits", "probabilities", "sampled_action")
               else raw[name] for name in PROBE_FIELDS}
        out.update({"initial_" + name: raw["initial"][name] for name in encounter.BODY})
        out["initial_neural_matches"] = jnp.stack([jax.vmap(lambda x: _same_bits(x, snapshot[name]))(
            raw["initial"][name]) for name in encounter.NEURAL], axis=1)
        out["final_frozen_matches"] = jnp.stack([jax.vmap(_same_bits)(raw["initial"][name], raw["final"][name])
            for name in ("W", "Wp", "mask")], axis=1)
        return out
    return run


def _local_probe(flat, snapshot, keys, c):
    performed = bool(np.asarray(snapshot["alive"])[0] > 0)
    if performed:
        out = jax.tree.map(np.asarray, _compact_readouts(c)(flat, snapshot, keys))
    else:
        # These are explicitly unexecuted slots, not a revived fresh-body probe.
        out = dict(observation=np.zeros((12, 1, seed.dims(c)[1]), np.float32),
            decision_motor=np.zeros((12, 5), np.float32), logits=np.zeros((12, 5), np.float32),
            probabilities=np.zeros((12, 5), np.float32), sampled_action=np.full(12, -1, np.int32),
            action_key=np.zeros((12, 2), np.uint32), hebbian_enabled=np.zeros(12, bool),
            predictor_enabled=np.zeros(12, bool), feedback_applied=np.zeros(12, bool),
            initial_neural_matches=np.ones((12, 4), bool), final_frozen_matches=np.ones((12, 3), bool))
        out.update({"initial_" + name: np.broadcast_to(np.asarray(snapshot[name]), (12,) + np.shape(snapshot[name])).copy()
                    for name in encounter.BODY})
    out.update({"source_" + name + "_sha256": encounter.array_hash(snapshot[name]) for name in encounter.NEURAL})
    out["performed"] = np.asarray(performed)
    return out


def _solo(flat, snapshot, grid, position, key, rule, c):
    performed = bool(np.asarray(snapshot["alive"])[0] > 0)
    initial = encounter._fresh_body(snapshot, grid, position, key, c) if performed else snapshot
    final, trace = _ticks(c, GRID_STEPS, True)(flat, initial, rule, 0)
    return initial, final, trace, np.asarray(performed)


def _record(record, prefix, values):
    record.update({prefix + name: np.asarray(value) for name, value in values.items()})


def evaluate_world(flat, keys, index, c=CFG):
    _inputs(keys, index, c)
    flat = encounter._genome(flat, "plastic", c)
    keys = np.asarray(keys)
    schedule = geometry(keys[1], c)
    grid, position = seed.init_world(jnp.asarray(keys[3]), c)
    probe_keys = jax.random.split(jnp.asarray(keys[5]), len(encounter.PLACEMENTS))
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(), initial_rules=np.asarray([0, 1], np.int32),
        genome_sha256=encounter.array_hash(flat), probe_placements=encounter.PLACEMENTS.copy(),
        block_steps=np.asarray(BLOCK_STEPS, np.int32), pre_blocks=np.asarray(PRE_BLOCKS, np.int32),
        post_blocks=np.asarray(POST_BLOCKS, np.int32), grid_steps=np.asarray(GRID_STEPS, np.int32), **schedule)
    commons, arms = [], {arm: [] for arm in ARMS}
    for initial_rule in (0, 1):
        birth = encounter._birth(flat, "plastic", jnp.asarray(keys[0]), jnp.asarray(keys[2]), c)
        split, pre_trace = _offers(c)(flat, birth, jnp.asarray(schedule["geometry_near_offsets"][:PRE_BLOCKS]),
            initial_rule, 0)
        common = {}
        _record(common, "birth_", encounter._state_evidence(birth))
        _record(common, "exposure_", pre_trace)
        _record(common, "split_", encounter._state_evidence(split))
        feedback, flush = _flush(flat, split, initial_rule, PRE_BLOCKS * BLOCK_STEPS, c)
        _record(common, "flush_", flush)
        _record(common, "feedback_", encounter._state_evidence(feedback))
        _record(common, "probe_", _local_probe(flat, feedback, probe_keys, c))
        commons.append(common)
        for arm in ARMS:
            rule = initial_rule if arm.endswith("stationary") else 1 - initial_rule
            offsets = schedule["geometry_" + arm.split("_")[0] + "_offsets"][PRE_BLOCKS:]
            end, trace = _offers(c)(flat, split, jnp.asarray(offsets), rule, PRE_BLOCKS * BLOCK_STEPS)
            record = dict(rule=np.asarray(rule, np.int32))
            _record(record, "split_", encounter._state_evidence(split))
            _record(record, "exposure_", trace)
            _record(record, "exposure_end_", encounter._state_evidence(end))
            feedback, flush = _flush(flat, end, rule, (PRE_BLOCKS + POST_BLOCKS) * BLOCK_STEPS, c)
            _record(record, "flush_", flush)
            _record(record, "feedback_", encounter._state_evidence(feedback))
            _record(record, "probe_", _local_probe(flat, feedback, probe_keys, c))
            initial, final, trace, performed = _solo(flat, feedback, grid, position, jnp.asarray(keys[4]), rule, c)
            record["solo_performed"] = performed
            _record(record, "solo_initial_", encounter._state_evidence(initial))
            _record(record, "solo_", trace)
            _record(record, "solo_final_", encounter._state_evidence(final))
            arms[arm].append(record)
    _record(row, "common_", encounter._stack(commons))
    for arm, values in arms.items():
        _record(row, arm + "_", encounter._stack(values))
    validate_world(row, c)
    return row


def _validate_trace(value, prefix, steps, c):
    data = {name: value[prefix + name] for name in TRACE_FIELDS +
            ("skipped", "external_tick", "incoming_efference", "carried_denergy")}
    if data["action"].shape != (2, steps, 1) or data["skipped"].shape != (2, steps):
        raise ValueError("lost a scheduled voluntary or skipped slot")
    live = ~data["skipped"]
    if (not np.array_equal(live, data["alive_before"][..., 0] > 0)
            or not np.array_equal(data["action"], data["sampled_action"])
            or np.any(data["action"][live] < 0) or np.any(data["action"][live] > 4)
            or np.any(data["action"][~live] != -1)
            or np.any(data["food"] & data["poison"])
            or np.any(data["alive_after"] > data["alive_before"])):
        raise ValueError("invalid voluntary decision or nonabsorbing death")
    for field in ("hebbian_enabled", "predictor_enabled"):
        if not np.array_equal(data[field], live):
            raise ValueError("live plasticity must stay active and dead slots must not update")
    if (not np.array_equal(data["efference"][live], np.eye(5, dtype=np.float32)[data["action"][live]])
            or not np.array_equal(data["observation"][live, :, -2], np.float32(4) * data["incoming_denergy"][live])
            or not encounter._identical(data["incoming_efference"][:, 1:], data["efference"][:, :-1])
            or not encounter._identical(data["incoming_denergy"][:, 1:], data["carried_denergy"][:, :-1])):
        raise ValueError("actual motor/energy feedback was not carried")
    if (np.any(data["food"][~live]) or np.any(data["poison"][~live]) or np.any(data["denergy"][~live])
            or not encounter._identical(data["key_before"][~live], data["key_after"][~live])
            or not encounter._identical(data["energy_before"][~live], data["energy"][~live])):
        raise ValueError("a skipped dead slot changed physical state or generated an outcome")
    encounter.validate_probabilities(data["probabilities"][live], data["logits"][live])
    if np.any(data["energy"] < 0) or np.any(data["energy"] > c.e_max):
        raise ValueError("body energy left its native bounds")


def _validate_probe(value, source, c):
    performed = value["probe_performed"]
    probabilities, logits = value["probe_probabilities"], value["probe_logits"]
    if (performed.shape != (2,) or probabilities.shape != (2, 12, 5)
            or not np.array_equal(performed, value[source + "alive"][:, 0] > 0)
            or np.any(probabilities[~performed]) or np.any(logits[~performed])):
        raise ValueError("readout eligibility must follow the actual post-feedback living state")
    encounter.validate_probabilities(probabilities[performed], logits[performed])
    if not encounter._identical(logits, np.float32(3) * value["probe_decision_motor"]):
        raise ValueError("local logits differ from the actual five decision motors")
    if (not np.all(value["probe_initial_neural_matches"]) or not np.all(value["probe_final_frozen_matches"])
            or any(np.any(value["probe_" + field]) for field in ("hebbian_enabled", "predictor_enabled", "feedback_applied"))):
        raise ValueError("local probe changed frozen weights or received the wrong neural carry")
    for name in encounter.NEURAL:
        if not encounter._identical(value["probe_source_" + name + "_sha256"], value[source + name + "_sha256"]):
            raise ValueError("local source fingerprint differs from the post-feedback state")
    for field, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.), ("sig", 0.), ("t", 0), ("counts", 0.)):
        if not np.all(value["probe_initial_" + field][performed] == expected):
            raise ValueError("living local probe lacks its fresh neutral body")
    for name in encounter.BODY:
        expected = value[source + name][~performed]
        expected = np.repeat(expected[:, None], 12, axis=1)
        if not encounter._identical(value["probe_initial_" + name][~performed], expected):
            raise ValueError("dead local placeholder revived or changed the frozen body")


def validate_world(row, c=CFG):
    _inputs(row["keys"], int(row["base_index"]), c)
    if np.shape(row["genome_sha256"]) != (32,) or np.asarray(row["genome_sha256"]).dtype != np.uint8:
        raise ValueError("missing source genome byte fingerprint")
    if not encounter._identical(row["initial_rules"], np.asarray([0, 1], np.int32)):
        raise ValueError("each independent family must retain both initial rules")
    if not encounter._identical(row["probe_placements"], encounter.PLACEMENTS):
        raise ValueError("local probe must include all twelve ordered alternatives")
    for name, expected in (("block_steps", BLOCK_STEPS), ("pre_blocks", PRE_BLOCKS),
                           ("post_blocks", POST_BLOCKS), ("grid_steps", GRID_STEPS)):
        if int(row[name]) != expected:
            raise ValueError("apparatus duration differs from its frozen protocol")
    for key, value in row.items():
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("nonfinite raw evidence: " + key)
    for kind in ("near", "sham"):
        offsets = row["geometry_" + kind + "_offsets"]
        if (offsets.shape != (16, 2, 2) or np.any(offsets < 0) or np.any(offsets >= c.G)
                or np.any(np.all(offsets == 0, axis=-1)) or np.any(np.all(offsets[:, 0] == offsets[:, 1], axis=-1))):
            raise ValueError("offer geometry must contain two distinct noncentral offsets")
        if kind == "near" and not np.all(np.any(np.all(offsets[:, :, None] == np.asarray(seed.MOVES[1:] % c.G), axis=-1), axis=-1)):
            raise ValueError("near offers must occupy cardinal neighbors")
    common = encounter._arm(row, "common")
    _validate_trace(common, "exposure_", PRE_BLOCKS * BLOCK_STEPS, c)
    _validate_trace(common, "flush_", 1, c)
    _validate_probe(common, "feedback_", c)
    for arm in ARMS:
        value = encounter._arm(row, arm)
        expected_rules = row["initial_rules"] if arm.endswith("stationary") else 1 - row["initial_rules"]
        if not np.array_equal(value["rule"], expected_rules):
            raise ValueError("continuation used the wrong final food rule")
        for name in encounter.NEURAL:
            suffix = name + "_sha256"
            if not encounter._identical(value["split_" + suffix], common["split_" + suffix]):
                raise ValueError("continuation did not clone the common unflushed neural state")
            if not encounter._identical(value["solo_initial_" + suffix], value["feedback_" + suffix]):
                raise ValueError("solo probe did not clone the unprobed post-feedback neural state")
        for name in encounter.BODY:
            if not encounter._identical(value["split_" + name], common["split_" + name]):
                raise ValueError("continuation did not clone the common unflushed body/key")
        _validate_trace(value, "exposure_", POST_BLOCKS * BLOCK_STEPS, c)
        _validate_trace(value, "flush_", 1, c)
        _validate_trace(value, "solo_", GRID_STEPS, c)
        _validate_probe(value, "feedback_", c)
        performed = value["solo_performed"]
        if not np.array_equal(performed, value["probe_performed"]):
            raise ValueError("local and solo eligibility differ")
        for name in encounter.BODY:
            if not encounter._identical(value["solo_initial_" + name][~performed], value["feedback_" + name][~performed]):
                raise ValueError("dead solo assignment was reset or revived")
        for field, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.), ("sig", 0.), ("t", 0), ("counts", 0.)):
            if not np.all(value["solo_initial_" + field][performed] == expected):
                raise ValueError("living solo probe lacks its fresh neutral body")


def _preference(probabilities, rules):
    p = np.asarray(probabilities, np.float64)
    return np.asarray([(p[i, np.arange(12), encounter.PLACEMENTS[:, int(rule)]]
        - p[i, np.arange(12), encounter.PLACEMENTS[:, 1 - int(rule)]]).mean()
        for i, rule in enumerate(rules)], np.float64)


def summary_input(row):
    """Only within-family per-rule endpoints remain resident in the panel writer."""
    validate_world(row)
    compact = {name: np.asarray(row[name]).copy() for name in ("base_index", "keys", "initial_rules", "genome_sha256")}
    compact["pre_old_S_by_rule"] = _preference(row["common_probe_probabilities"], row["initial_rules"])
    for arm in ARMS:
        value = encounter._arm(row, arm)
        f, p = value["solo_food"][:, :, 0], value["solo_poison"][:, :, 0]
        contact = f | p
        any_contact, first = contact.any(1), contact.argmax(1)
        first_food = f[np.arange(2), first] & any_contact
        first_poison = p[np.arange(2), first] & any_contact
        contacted = value["exposure_food"] | value["exposure_poison"]
        intake_a = (contacted & (value["exposure_item"] == 1)).sum((1, 2))
        intake_b = (contacted & (value["exposure_item"] == 2)).sum((1, 2))
        fields = dict(local_S=_preference(value["probe_probabilities"], value["rule"]),
            both_colors=(intake_a > 0) & (intake_b > 0), solo_Y=first_food.astype(int) - first_poison.astype(int),
            first_food=first_food, first_poison=first_poison, no_contact=~any_contact,
            exposure_survival=value["exposure_end_alive"][:, 0], readout_survival=value["feedback_alive"][:, 0],
            solo_survival=value["solo_final_alive"][:, 0], intake_A=intake_a, intake_B=intake_b,
            realized_energy=value["exposure_denergy"].astype(np.float64).sum((1, 2)))
        compact.update({arm + "_" + name + "_by_rule": np.asarray(values, np.float64) for name, values in fields.items()})
    return compact
