"""E131: paired native lives with one bounded, independently phased action quota."""
from numbers import Integral
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from seed import exploration, founder_reversal as founder, seed

PROTOCOL = "timed-exploration-v1"
CFG = seed.Cfg()
POLICIES = ("unchanged", "adaptive", "random_quota")
CONDITIONS = ("stationary", "reversal")
ARMS = tuple(policy + "_" + condition for policy in POLICIES for condition in CONDITIONS)
KEY_ROLES = ("world_action", "development", "exploration")
ASSIGNMENT_TAG = 0xE131
FLIP_STEP = founder.FLIP_STEP
BIN_EDGES = founder.BIN_EDGES
EVENTS = founder.EVENTS
POLICY_PARAMETERS = dict(block_length=25, quota_per_block=1, counter_cap=12, trigger=12,
    phase_delay_min=0, phase_delay_max=24, quota_offset_min=0, quota_offset_max=24,
    cardinal_min=1, cardinal_max=4, feedback="positive_observed_realized_denergy")


def evaluation_keys(master, index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG:
        raise ValueError("invalid exploration family index")
    stream = jax.random.fold_in(founder._master(master), ASSIGNMENT_TAG)
    return np.asarray(jax.random.split(jax.random.fold_in(stream, int(index)), 3))


def assignments_manifest(master, count):
    if isinstance(count, bool) or count not in (4, 64):
        raise ValueError("E131 assigns four development or sixty-four final families")
    return dict(family_keys=np.stack([evaluation_keys(master, index) for index in range(count)]),
                initial_rules=np.asarray([0, 1], np.int32))


def _inputs(keys, index, c):
    keys = np.asarray(keys)
    if (c != CFG or keys.shape != (3, 2) or keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 3
            or isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < ASSIGNMENT_TAG):
        raise ValueError("E131 requires native four-animal geometry and three separate role keys")
    return keys


def evaluate_world(flat, keys, index, c=CFG):
    keys = _inputs(keys, index, c)
    genome = founder._genome(flat, c)
    proof = exploration.make_schedule(keys[2], c.M, c.T)
    schedule = {name: jnp.asarray(proof[name]) for name in ("phase_delay", "directions", "quota_offsets")}
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(),
        birth_keys=np.asarray(jax.random.split(jnp.asarray(keys[1]), c.M)), initial_rules=np.asarray([0, 1], np.int32),
        bin_edges=np.asarray(BIN_EDGES, np.int32), flip_step=np.asarray(FLIP_STEP, np.int32), genome_sha256=founder._hash(flat),
        **{("schedule_stream_keys" if name == "stream_keys" else name): value for name, value in proof.items()})
    seconds = []
    for policy in POLICIES:
        started = perf_counter()
        for condition in CONDITIONS:
            lives = []
            for rule in (0, 1):
                enabled = policy != "unchanged"
                raw = seed.rollout(genome, jnp.asarray(keys[0]), jnp.asarray(keys[1]), jnp.asarray(rule, jnp.int32), c,
                    plastic=True, predictor_plastic=True, record=False, diagnostics=True,
                    flip_step=FLIP_STEP if condition == "reversal" else None,
                    exploration=policy if enabled else None, _exploration_schedule=schedule if enabled else None,
                    _exploration_record=True)
                # Timing includes actual completion and host materialization of
                # every trace, not asynchronous dispatch of fitness alone.
                jax.block_until_ready(raw)
                raw = jax.tree.map(np.asarray, raw)
                for name in EVENTS:
                    if raw["events"][name].shape != (c.T, c.M) or not np.isin(raw["events"][name], (0, 1)).all():
                        raise ValueError("native rollout returned invalid event evidence")
                lives.append({**{name: raw["events"][name].astype(bool) for name in EVENTS},
                    **raw["exploration"], "native_counts": raw["counts"], "native_fitness": raw["fitness"],
                    "native_survived": raw["survived"]})
            row.update({policy + "_" + condition + "_" + name: np.stack([life[name] for life in lives]) for name in lives[0]})
        seconds.append(perf_counter() - started)
    row["policy_seconds"] = np.asarray(seconds, np.float64)
    validate_world(row, c)
    return row


def _require_same(actual, expected, message):
    if not founder._same(actual, expected):
        raise ValueError(message)


def validate_world(row, c=CFG):
    keys = _inputs(row["keys"], int(row["base_index"]), c)
    for name, expected in (("initial_rules", [0, 1]), ("bin_edges", BIN_EDGES)):
        _require_same(row[name], np.asarray(expected, np.int32), "changed rule/window assignment")
    _require_same(row["flip_step"], np.asarray(FLIP_STEP, np.int32), "changed hidden switch")
    _require_same(row["birth_keys"], np.asarray(jax.random.split(jnp.asarray(keys[1]), c.M)), "changed native birth streams")
    expected_schedule = exploration.make_schedule(keys[2], c.M, c.T)
    for name, expected in expected_schedule.items():
        _require_same(row["schedule_stream_keys" if name == "stream_keys" else name], expected, "changed independent exploration schedule")
    if np.asarray(row["genome_sha256"]).shape != (32,) or np.asarray(row["genome_sha256"]).dtype != np.uint8:
        raise ValueError("missing fixed genome byte identity")
    timing = np.asarray(row["policy_seconds"])
    if timing.shape != (3,) or timing.dtype != np.float64 or not np.isfinite(timing).all() or np.any(timing <= 0):
        raise ValueError("missing synchronized per-policy timings")
    for arm in ARMS:
        value = founder._arm(row, arm)
        food, poison, before, after = founder._validate_events(value)
        counts = np.stack([np.stack([event[:, start:stop].sum((1, 2)) for event in (food, poison)], axis=-1)
                           for start, stop in ((0, FLIP_STEP), (FLIP_STEP, c.T))], axis=1).astype(np.float32)
        for field, expected in (("native_counts", counts),
                ("native_fitness", (counts[:, :, 0] - counts[:, :, 1]).sum(1) / np.float32(c.M)),
                ("native_survived", after[:, -1].mean(1).astype(np.float32))):
            _require_same(value[field], expected, "native summary differs from assigned events")
        for field in ("observed_denergy", "energy_before", "energy_after", "unclipped_energy", "denergy"):
            if value[field].shape != (2, c.T, c.M) or value[field].dtype != np.float32 or not np.isfinite(value[field]).all():
                raise ValueError("missing actual finite physical feedback")
        _require_same(value["observed_denergy"][:, 1:], value["denergy"][:, :-1], "feedback moved to the wrong neural turn")
        _require_same(value["energy_before"][:, 1:], value["energy_after"][:, :-1], "physical energy was reset")
        _require_same(value["denergy"], (value["energy_after"] - value["energy_before"]) * before.astype(np.float32), "feedback is not realized post-clipping energy")
        if np.any(value["observed_denergy"][:, 0]) or not np.all(value["energy_before"][:, 0] == c.e0):
            raise ValueError("lives did not start with native body feedback")
        for field in ("ordinary_action", "action"):
            if value[field].shape != (2, c.T, c.M) or value[field].dtype != np.int32 or not np.isin(value[field], range(5)).all():
                raise ValueError("invalid original or executed action")
        override = value["override"]
        if override.shape != before.shape or override.dtype != np.bool_ or np.any(override & ~before):
            raise ValueError("invalid override or override after death")
        _require_same(value["action"], np.where(override, value["direction"], value["ordinary_action"]), "executed action is not the declared bounded override")
        _require_same(value["efference"], np.eye(5, dtype=np.float32)[value["action"]], "brain received an unexecuted action")
        _require_same(value["incoming_efference"][:, 1:], value["efference"][:, :-1], "actual action was not carried to the next neural turn")
        if np.any(value["incoming_efference"][:, 0]):
            raise ValueError("newborn carried a previous action")
        _require_same(value["override_changed"], override & (value["action"] != value["ordinary_action"]), "override/action-change distinction lost")
        if arm.startswith("unchanged") and override.any():
            raise ValueError("ordinary controller received an override")
        if not arm.startswith("unchanged"):
            for rule in (0, 1):
                for animal, phase in enumerate(row["phase_delay"]):
                    if override[rule, :phase, animal].any():
                        raise ValueError("exploration fired before its independent phase")
                    for start in range(int(phase), c.T, exploration.BLOCK):
                        stop = min(start + exploration.BLOCK, c.T)
                        spent = int(override[rule, start:stop, animal].sum())
                        if spent > 1 or (stop - start == exploration.BLOCK and before[rule, start:stop, animal].all() and spent != 1):
                            raise ValueError("bounded override quota violated")
    for policy in POLICIES:
        stationary, reversal = (founder._arm(row, policy + "_" + condition) for condition in CONDITIONS)
        for name in (*EVENTS, "observed_denergy", "energy_before", "energy_after", "action", "ordinary_action", "override", "efference"):
            _require_same(stationary[name][:, :FLIP_STEP], reversal[name][:, :FLIP_STEP], "paired policy histories differ before switch")
        _require_same(stationary["action"][:, FLIP_STEP], reversal["action"][:, FLIP_STEP], "policy used the upcoming hidden switch")
        _require_same(stationary["food"][:, FLIP_STEP], reversal["poison"][:, FLIP_STEP], "switch did not relabel the same first action")
        _require_same(stationary["poison"][:, FLIP_STEP], reversal["food"][:, FLIP_STEP], "switch did not reverse both labels")


def summary_input(row):
    validate_world(row)
    compact = {name: np.asarray(row[name]).copy() for name in
        ("base_index", "keys", "birth_keys", "initial_rules", "bin_edges", "flip_step", "genome_sha256")}
    for arm in ARMS:
        value = founder._arm(row, arm)
        compact.update({arm + "_" + name: result for name, result in founder.bin_events(value).items()})
        for label, events in (("first_food", value["food"]), ("first_any", value["food"] | value["poison"])):
            window = events[:, 150:200]
            compact[arm + "_latency_" + label + "_by_rule"] = np.where(window.any(1), window.argmax(1), 50).astype(np.int32)
        compact[arm + "_override_count_by_rule"] = value["override"].sum(1, dtype=np.int32)
        compact[arm + "_changed_action_count_by_rule"] = value["override_changed"].sum(1, dtype=np.int32)
        compact[arm + "_alive_turns_by_rule"] = value["alive_before"].sum(1, dtype=np.int32)
    return compact
