"""E134 native-genome selection from controlled experience and free choices."""
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import encounter_diagnostic as encounter, seed


PROTOCOL = "replay-selection-v1"
CFG = encounter.CFG
ASSIGNMENT_TAG = 0xE134
KEY_ROLES = encounter.KEY_ROLES
GENOTYPES = ("parent", "descendant")
HISTORIES = ("stationary", "reversal")
FEEDBACK = ("informative", "yoked")
TRAIN_CELLS = np.asarray([(8, 0), (8, 8), (16, 0), (16, 8)], np.int32)
GRID_STEPS = 50
TRACE_OMIT = ("incoming_activity", "activity")


def _integer(value, allowed, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in allowed:
        raise ValueError(f"invalid {label}")
    return int(value)


def _keys(keys, count):
    keys = np.asarray(keys)
    if (keys.shape != (count, 2) or keys.dtype != np.uint32
            or len({key.tobytes() for key in keys}) != count):
        raise ValueError("expected distinct uint32 keys")
    return keys


def training_batch(world_keys):
    """Materialize rule-blind scripts once per ES attempt outside the JIT."""
    world_keys = _keys(world_keys, 4)
    role_keys = np.stack([np.asarray(jax.random.split(key, 3)) for key in world_keys])
    schedule_key = np.asarray(jax.random.fold_in(world_keys[0], ASSIGNMENT_TAG))
    order = np.asarray(jax.random.permutation(schedule_key, 4))
    cells = TRAIN_CELLS[order]
    scripts = np.zeros((4, 32, 5), np.int32)
    for i, (pre, _) in enumerate(cells):
        script = encounter.contact_script(role_keys[i, 0], int(pre))
        scripts[i, :len(script)] = script
    return dict(world_keys=world_keys.copy(), role_keys=role_keys, schedule_key=schedule_key,
                scripts=scripts, pre_contacts=cells[:, 0], gap=cells[:, 1])


@lru_cache(None)
def _empty_gap(ticks):
    @jax.jit
    def run(flat, state, rule):
        genome = encounter._unravel(CFG)(flat)
        state = dict(state, grid=jnp.zeros_like(state["grid"]))
        return jax.lax.scan(lambda carry, _: encounter._step(
            carry, genome, rule, 0, "plastic", CFG, regrowth=False), state, None, length=ticks)
    return run


def _training_history(flat, boundary, script, rule, gap):
    acquired, trace = encounter._chamber("plastic", CFG)(flat, boundary, script, rule)
    delayed, empty = _empty_gap(gap)(flat, acquired, rule)
    energy = jnp.concatenate((trace["unclipped_energy"].reshape(-1), empty["unclipped_energy"].reshape(-1)))
    counts = jnp.stack((trace["food"].sum(), trace["poison"].sum())).astype(jnp.float32)
    return delayed, counts, jnp.stack((energy.min(), energy.max()))


def _score_family(flat, script, birth_key, chamber_key, probe_key, pre_contacts, gap):
    probe_keys = jax.random.split(jax.random.fold_in(probe_key, 1), len(encounter.PLACEMENTS))

    def initial_rule(rule):
        birth = encounter._birth(flat, "plastic", birth_key, chamber_key, CFG)
        boundary, pre = encounter._chamber("plastic", CFG)(flat, birth, script[:pre_contacts], rule)
        pre_counts = jnp.stack((pre["food"].sum(), pre["poison"].sum())).astype(jnp.float32)
        pre_bounds = jnp.stack((pre["unclipped_energy"].min(), pre["unclipped_energy"].max()))

        def continuation(history):
            target = jnp.bitwise_xor(rule, history)
            delayed, post_counts, bounds = _training_history(
                flat, boundary, script[pre_contacts:pre_contacts + 16], target, gap)
            probe = encounter._readouts("plastic", CFG)(flat, delayed, probe_keys)
            return dict(probabilities=probe["probabilities"][:, 0], logits=probe["logits"][:, 0],
                        counts=jnp.stack((pre_counts, post_counts)), survived=delayed["alive"][0],
                        exposure_energy_bounds=jnp.stack((jnp.minimum(pre_bounds[0], bounds[0]),
                                                         jnp.maximum(pre_bounds[1], bounds[1]))))
        return jax.lax.map(continuation, jnp.arange(2, dtype=jnp.int32))

    result = jax.lax.map(initial_rule, jnp.arange(2, dtype=jnp.int32))
    probabilities = result["probabilities"]
    placements = jnp.asarray(encounter.PLACEMENTS)
    targets = jnp.bitwise_xor(jnp.arange(2)[:, None], jnp.arange(2)[None, :])
    good = placements[:, targets].transpose(1, 2, 0)
    bad = placements[:, 1 - targets].transpose(1, 2, 0)
    food = jnp.take_along_axis(probabilities, good[..., None], axis=-1)[..., 0]
    poison = jnp.take_along_axis(probabilities, bad[..., None], axis=-1)[..., 0]
    return dict(fitness=(food - poison).mean(), probabilities=probabilities, logits=result["logits"],
                counts=result["counts"].mean((0, 1)), survived=result["survived"].min(),
                exposure_energy_bounds=jnp.stack((result["exposure_energy_bounds"][..., 0].min(),
                                                 result["exposure_energy_bounds"][..., 1].max())))


@lru_cache(None)
def _population_cell(pre_contacts, gap):
    @jax.jit
    def run(candidates, script, birth_keys, chamber_key, probe_key):
        return jax.lax.map(lambda pair: _score_family(
            pair[0], script, pair[1], chamber_key, probe_key, pre_contacts, gap), (candidates, birth_keys))
    return run


def make_evaluator(c=CFG):
    encounter._config(c)

    def evaluate(candidates, batch, birth_keys):
        candidates = np.asarray(candidates)
        if (candidates.ndim != 2 or candidates.shape[1:] != (155,) or candidates.dtype != np.float32
                or not np.isfinite(candidates).all() or np.shape(birth_keys) != (len(candidates), 4, 2)
                or np.asarray(birth_keys).dtype != np.uint32):
            raise ValueError("candidate and antithetic birth arrays have invalid dimensions")
        if sorted(zip(np.asarray(batch["pre_contacts"]).tolist(), np.asarray(batch["gap"]).tolist())) != list(map(tuple, TRAIN_CELLS)):
            raise ValueError("training must contain each registered schedule cell once")
        records = []
        for i in range(4):
            pre, gap = int(batch["pre_contacts"][i]), int(batch["gap"][i])
            records.append(_population_cell(pre, gap)(
                jnp.asarray(candidates), jnp.asarray(batch["scripts"][i]), jnp.asarray(birth_keys[:, i]),
                jnp.asarray(batch["role_keys"][i, 1]), jnp.asarray(batch["role_keys"][i, 2])))
        result = jax.tree.map(lambda *values: jnp.stack(values, axis=1), *records)
        bounds = np.asarray(result["exposure_energy_bounds"])
        if (not np.isfinite(np.asarray(result["fitness"])).all() or not np.isfinite(bounds).all()
                or bounds[..., 0].min() <= 0 or bounds[..., 1].max() >= c.e_max
                or not np.all(np.asarray(result["survived"]) == 1)):
            raise ValueError("candidate replay violated finite score or guaranteed survival")
        return result
    return evaluate


def _compact_trace(trace):
    out = {name: np.asarray(value) for name, value in trace.items() if name not in TRACE_OMIT}
    out["explicit_learning_denergy"] = np.asarray(trace["incoming_denergy"])
    return out


def _join_trace(*traces):
    return {name: np.concatenate([np.asarray(trace[name]) for trace in traces], axis=0) for name in traces[0]}


def _prefix(row, prefix, value):
    row.update({prefix + name: item for name, item in value.items()})


def evaluate_family(parent, endpoint, keys, index, pre_contacts, gap, sham_rule, c=CFG):
    encounter._config(c)
    keys = _keys(keys, len(KEY_ROLES))
    pre_contacts = _integer(pre_contacts, (8, 16), "old-contact count")
    gap = _integer(gap, (16, 32), "held-out gap")
    sham_rule = _integer(sham_rule, (0, 1), "sham mapping")
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("invalid family index")
    flats = [encounter._genome(flat, "plastic", c) for flat in (parent, endpoint)]
    script = encounter.contact_script(keys[0], pre_contacts)
    birth_key, chamber_key, action_key = map(jnp.asarray, (keys[1], keys[2], keys[4]))
    grid, position = seed.init_world(jnp.asarray(keys[3]), c)
    old_keys = jax.random.split(jax.random.fold_in(jnp.asarray(keys[5]), 0), 12)
    probe_keys = jax.random.split(jax.random.fold_in(jnp.asarray(keys[5]), 1), 12)
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(),
               pre_contacts=np.asarray(pre_contacts, np.int32), gap=np.asarray(gap, np.int32),
               sham_rule=np.asarray(sham_rule, np.int32), script=script,
               initial_rules=np.asarray([0, 1], np.int32),
               genome_sha256=np.stack([encounter.array_hash(flat) for flat in flats]),
               target_grid=np.asarray(grid), target_position=np.asarray(position))
    genotypes = []
    for flat in flats:
        rules = []
        for initial in (0, 1):
            birth = encounter._birth(flat, "plastic", birth_key, chamber_key, c)
            boundary, prior = encounter._chamber("plastic", c)(flat, birth, jnp.asarray(script[:pre_contacts]), initial)
            record = {}
            _prefix(record, "birth_", encounter._state_evidence(birth))
            _prefix(record, "boundary_", encounter._state_evidence(boundary))
            _prefix(record, "old_", encounter._probe_evidence(encounter._readouts("plastic", c)(flat, boundary, old_keys)))
            histories = []
            solo_records = []
            for history in (0, 1):
                target = initial ^ history
                feedback_records = []
                for feedback in (0, 1):
                    actual_rule = target if feedback == 0 else sham_rule
                    acquired, post = encounter._chamber("plastic", c)(flat, boundary, jnp.asarray(script[pre_contacts:]), actual_rule)
                    delayed, empty = _empty_gap(gap)(flat, acquired, actual_rule)
                    exposure = _join_trace(prior, post, empty)
                    item = {}
                    _prefix(item, "exposure_", _compact_trace(exposure))
                    _prefix(item, "acquired_", encounter._state_evidence(delayed))
                    _prefix(item, "local_", encounter._probe_evidence(encounter._readouts("plastic", c)(flat, delayed, probe_keys)))
                    feedback_records.append(item)
                    if feedback == 0:
                        fresh = encounter._fresh_body(delayed, grid, position, action_key, c)
                        final, trace = encounter._free_grid("plastic", c)(flat, fresh, target)
                        solo = {}
                        _prefix(solo, "solo_initial_", encounter._state_evidence(fresh))
                        _prefix(solo, "solo_final_", encounter._state_evidence(final))
                        _prefix(solo, "solo_", _compact_trace(trace))
                        solo_records.append(solo)
                histories.append(encounter._stack(feedback_records))
            record.update(encounter._stack(histories))
            record.update(encounter._stack(solo_records))
            rules.append(record)
        genotypes.append(encounter._stack(rules))
    row.update(encounter._stack(genotypes))
    validate_family(row)
    return row


def validate_family(row):
    """Local admission checks; the independent auditor reconstructs decisions."""
    probabilities, logits = np.asarray(row["local_probabilities"]), np.asarray(row["local_logits"])
    if probabilities.shape != (2, 2, 2, 2, 12, 5) or logits.shape != probabilities.shape:
        raise ValueError("local readout axes differ from genotype/rule/history/feedback/placement/action")
    for prob, logit in zip(probabilities.reshape(-1, 5), logits.reshape(-1, 5)):
        encounter.validate_probabilities(prob, logit)
    for prob, logit in zip(row["old_probabilities"].reshape(-1, 5), row["old_logits"].reshape(-1, 5)):
        encounter.validate_probabilities(prob, logit)
    for name, value in row.items():
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("nonfinite family field " + name)
        if name.startswith(("exposure_", "acquired_", "local_")):
            if not encounter._identical(array[:, :, 0, 1], array[:, :, 1, 1]):
                raise ValueError("rule-uninformative replays differ before scoring in " + name)
            for initial in (0, 1):
                matching_history = initial ^ int(row["sham_rule"])
                if not encounter._identical(array[:, initial, 0, 1], array[:, initial, matching_history, 0]):
                    raise ValueError("sham mapping differs from its identical informative replay in " + name)
    energy, unclipped = row["exposure_energy"], row["exposure_unclipped_energy"]
    if (not encounter._identical(energy, unclipped) or energy.min() < .462 - 1e-5
            or energy.max() > 1.25 + 1e-5 or not np.all(row["exposure_alive_after"] == 1)):
        raise ValueError("scripted exposure violated the deterministic energy bound")
    if (not np.all(row["solo_forced_action"] == -1)
            or not encounter._identical(row["solo_action"], row["solo_sampled_action"])):
        raise ValueError("the solo action was imposed")
    obs = row["local_observation"]
    if not encounter._identical(obs, np.broadcast_to(obs[0, 0, 0, 0], obs.shape).copy()):
        raise ValueError("local probe has immediate geometry or physiology cues")
    if np.any(row["local_feedback_applied"]) or np.any(row["local_hebbian_enabled"]) or np.any(row["local_predictor_enabled"]):
        raise ValueError("discarded local readout changed its experience")
    for name in encounter.NEURAL:
        acquired = row["acquired_" + name + "_sha256"]
        probe = row["local_initial_" + name + "_sha256"]
        if not encounter._identical(probe, np.broadcast_to(acquired[..., None, :], probe.shape).copy()):
            raise ValueError("local probe did not preserve acquired " + name)
        if not encounter._identical(row["solo_initial_" + name + "_sha256"], acquired[:, :, :, 0]):
            raise ValueError("solo probe did not preserve acquired " + name)


def signed_local(probabilities, rule):
    values = np.asarray(probabilities, np.float64)
    if values.shape[-2:] != (12, 5) or rule not in (0, 1):
        raise ValueError("signed local score needs twelve native five-way choices")
    choice = np.arange(12)
    return (values[..., choice, encounter.PLACEMENTS[:, rule]]
            - values[..., choice, encounter.PLACEMENTS[:, 1 - rule]]).mean(-1)


def summary_input(row):
    local = np.empty((2, 2, 2, 2), np.float64)
    old = np.empty((2, 2), np.float64)
    for initial in (0, 1):
        old[:, initial] = signed_local(row["old_probabilities"][:, initial], initial)
        for history in (0, 1):
            local[:, initial, history] = signed_local(row["local_probabilities"][:, initial, history], initial ^ history)
    food = np.asarray(row["solo_food"])[..., 0].astype(bool)
    poison = np.asarray(row["solo_poison"])[..., 0].astype(bool)
    contact = food | poison
    first = np.argmax(contact, axis=-1)[..., None]
    first_food = np.take_along_axis(food, first, axis=-1)[..., 0]
    first_poison = np.take_along_axis(poison, first, axis=-1)[..., 0]
    return dict(base_index=int(row["base_index"]), old=old, local=local,
                solo_Y=first_food.astype(float) - first_poison.astype(float),
                first_food=first_food, first_poison=first_poison, no_contact=~contact.any(-1),
                contact_latency=np.where(contact.any(-1), first[..., 0], GRID_STEPS),
                solo_survival=np.asarray(row["solo_alive_after"])[..., -1, 0])
