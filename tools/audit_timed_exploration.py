"""Independent E131 exploration scheduling, event and family-paired audit."""
import argparse
import copy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_founder_reversal as prior

base, source_bytes = prior.base, prior.source_bytes
require, same, subset, typed = prior.require, prior.same, prior.subset, prior.typed
exact, fingerprint, stat = prior.exact, prior.fingerprint, prior.stat
PROTOCOL = "timed-exploration-v1"
POLICIES = ("unchanged", "adaptive", "random_quota")
ARMS = tuple(policy + "_" + history for policy in POLICIES for history in ("stationary", "reversal"))
CONTROLS = {"ordinary": "unchanged", "quota": "random_quota"}
EDGES, CFG = prior.EDGES, prior.CFG
SOURCE_PATHS = prior.SOURCE_PATHS | {"seed/exploration.py", "seed/timed_exploration.py", "scripts/run_timed_exploration.py"}
POLICY_PARAMETERS = dict(block_length=25, quota_per_block=1, counter_cap=12, trigger=12,
    phase_delay_min=0, phase_delay_max=24, quota_offset_min=0, quota_offset_max=24,
    cardinal_min=1, cardinal_max=4, feedback="positive_observed_realized_denergy")
STATE_INTS = ("counter_before", "counter", "block_index", "block_offset", "direction", "quota_offset")
STATE_BOOLS = ("spent_before", "spent", "active", "due", "override")


def reconstruct_policy(mode, observed, living, ordinary, schedule):
    """Recompute the small scheduling state from public, preceding feedback only."""
    require(mode in POLICIES and observed.shape == living.shape == ordinary.shape == (300, 4), "complete per-animal policy inputs")
    require(living.dtype == np.bool_ and ordinary.dtype.kind in "iu" and np.all((ordinary >= 0) & (ordinary < 5)),
            "actual ordinary actions and liveness")
    phase, directions, offsets = (schedule[name] for name in ("phase_delay", "directions", "quota_offsets"))
    require(phase.shape == (4,) and directions.shape == offsets.shape == (12, 4)
            and all(value.dtype == np.int32 for value in (phase, directions, offsets))
            and np.all((phase >= 0) & (phase < 25)) and np.all((offsets >= 0) & (offsets < 25))
            and np.all((directions >= 1) & (directions <= 4)), "paired independent phase, direction and quota schedule")
    out = {field: np.zeros((300, 4), np.int32) for field in STATE_INTS}
    out.update({field: np.zeros((300, 4), bool) for field in STATE_BOOLS})
    if mode == "unchanged":
        for name in ("block_index", "block_offset", "quota_offset"):
            out[name][:] = -1
        out["action"] = ordinary.copy()
        return out
    # Scalar transitions keep this oracle separate from the vectorized policy.
    for animal in range(4):
        counter, spent = 0, False
        for tick in range(300):
            old_counter = counter
            if living[tick, animal] and tick:
                counter = 0 if observed[tick, animal] > 0 else min(counter + 1, 12)
            elapsed = tick - int(phase[animal])
            active = elapsed >= 0
            block, offset = divmod(elapsed, 25) if active else (-1, -1)
            if active and offset == 0:
                spent = False
            prior_spent = spent
            direction = int(directions[max(block, 0), animal])
            quota = int(offsets[max(block, 0), animal])
            due = (counter == 12 or offset == 24) if mode == "adaptive" else offset == quota
            override = active and bool(living[tick, animal]) and not spent and due
            spent = spent or override
            for field, value in dict(counter_before=old_counter, counter=counter, spent_before=prior_spent, spent=spent,
                active=active, block_index=block, block_offset=offset, direction=direction, quota_offset=quota,
                due=due, override=override).items():
                out[field][tick, animal] = value
    out["action"] = np.where(out["override"], out["direction"], ordinary).astype(ordinary.dtype)
    return out


def audit_life(life, mode, schedule, action_keys):
    shape = (300, 4)
    arrays = {name: typed(life, name, shape, "b") for name in ("food", "poison", "alive_before", "alive_after")}
    before, after = arrays["alive_before"], arrays["alive_after"]
    energy = {name: typed(life, name, shape, "f") for name in
              ("observed_denergy", "energy_before", "unclipped_energy", "energy_after", "denergy")}
    require(all(value.dtype == np.float32 for value in energy.values()), "native float32 energy evidence")
    exact(energy["energy_before"][0], np.ones(4, np.float32), "original energy at birth")
    exact(energy["observed_denergy"][0], np.zeros(4, np.float32), "birth has no earlier outcome")
    exact(energy["energy_before"][1:], energy["energy_after"][:-1], "actual energy continuity")
    exact(energy["observed_denergy"][1:], energy["denergy"][:-1], "feedback reaches scheduling exactly one turn later")
    raw_energy = energy["energy_before"].astype(np.float64) - .003 * before + .25 * arrays["food"] - .25 * arrays["poison"]
    # The native JIT may fuse the three float32 operations. This bound covers
    # their rounding; clipping and the recorded realized feedback stay exact.
    require(np.all(np.abs(energy["unclipped_energy"].astype(np.float64) - raw_energy) <= 8 * np.finfo(np.float32).eps),
            "ordinary metabolism and actual consumed resource rewards")
    exact(energy["energy_after"], np.clip(energy["unclipped_energy"], np.float32(0), np.float32(2)), "post-reward clipping")
    exact(energy["denergy"], (energy["energy_after"] - energy["energy_before"]) * before, "actual clipped energy change")
    exact(after, before & (energy["energy_after"] > 0), "survival follows realized energy")
    ordinary = typed(life, "ordinary_action", shape, "iu")
    expected = reconstruct_policy(mode, energy["observed_denergy"], before, ordinary, schedule)
    for field in STATE_INTS:
        exact(typed(life, field, shape, "i"), expected[field], "independently reconstructed " + field)
    for field in STATE_BOOLS:
        exact(typed(life, field, shape, "b"), expected[field], "independently reconstructed " + field)
    exact(typed(life, "action", shape, "iu"), expected["action"], "ordinary draw versus actually executed move")
    exact(typed(life, "efference", shape + (5,), "f"), np.eye(5, dtype=np.float32)[expected["action"]],
          "learning receives the action actually taken")
    incoming = typed(life, "incoming_efference", shape + (5,), "f")
    exact(incoming[0], np.zeros((4, 5), np.float32), "newborn motor activity starts unconditioned")
    exact(incoming[1:], life["efference"][:-1], "the next neural tick receives the actual previous action")
    exact(typed(life, "override_changed", shape, "b"), expected["override"] & (expected["action"] != ordinary),
          "an override counts even when its direction matches the ordinary draw")
    exact(typed(life, "action_key", (300, 2), "u"), action_keys, "ordinary stochastic stream is paired and unchanged")
    if mode != "unchanged":
        for animal in range(4):
            for start in range(int(schedule["phase_delay"][animal]), 300, 25):
                stop = min(start + 25, 300)
                used = int(expected["override"][start:stop, animal].sum())
                require(used <= 1, "never more than one pulse in a phased block")
                if stop - start == 25 and before[start:stop, animal].all():
                    require(used == 1, "each fully lived complete block spends exactly one pulse")
    return arrays, expected["override"]


@lru_cache(maxsize=1)
def random_helpers():
    """Replay only the pinned PRNG; this imports no environment or controller."""
    jax, jnp = source_bytes.random_runtime()

    @jax.jit
    def derive(keys):
        streams = jax.random.split(keys[2], 3)
        phase_keys = jax.random.split(streams[0], 4)
        direction_keys = jax.random.split(streams[1], 48).reshape(12, 4, 2)
        offset_keys = jax.random.split(streams[2], 48).reshape(12, 4, 2)
        phase = jax.vmap(lambda key: jax.random.randint(key, (), 0, 25, dtype=jnp.int32))(phase_keys)
        direction = jax.vmap(lambda key: jax.random.randint(key, (), 1, 5, dtype=jnp.int32))(direction_keys.reshape(-1, 2)).reshape(12, 4)
        offset = jax.vmap(lambda key: jax.random.randint(key, (), 0, 25, dtype=jnp.int32))(offset_keys.reshape(-1, 2)).reshape(12, 4)

        def step(key, _):
            next_key, action_key, _ = jax.random.split(key, 3)
            return next_key, action_key

        _, action_keys = jax.lax.scan(step, keys[0], None, length=300)
        return dict(birth_keys=jax.random.split(keys[1], 4), schedule_stream_keys=streams,
            phase_keys=phase_keys, direction_keys=direction_keys, offset_keys=offset_keys,
            phase_delay=phase, directions=direction, quota_offsets=offset, action_keys=action_keys)
    return jax, derive


def audit_family(raw, index, genome, keys, random):
    fields = {"base_index", "keys", "birth_keys", "initial_rules", "bin_edges", "flip_step", "genome_sha256", "policy_seconds",
              "schedule_stream_keys", "phase_keys", "direction_keys", "offset_keys", "phase_delay", "directions", "quota_offsets"}
    life_fields = {*STATE_INTS, *STATE_BOOLS, "food", "poison", "alive_before", "alive_after",
        "observed_denergy", "energy_before", "energy_after", "unclipped_energy", "denergy", "ordinary_action", "action",
        "action_key", "efference", "incoming_efference", "override_changed", "native_counts", "native_fitness", "native_survived"}
    require(set(raw) == fields | {arm + "_" + field for arm in ARMS for field in life_fields}, "exact complete exploration evidence schema")
    exact(raw["base_index"], np.asarray(index, np.int32), "ordered original family")
    exact(raw["keys"], keys, "three assigned independent roles")
    exact(raw["initial_rules"], np.asarray([0, 1], np.int32), "both rule assignments retained")
    exact(raw["bin_edges"], np.asarray(EDGES, np.int32), "unchanged fifty-turn bins")
    exact(raw["flip_step"], np.asarray(150, np.int32), "hidden switch before consumption on turn150")
    exact(raw["genome_sha256"], fingerprint(genome), "fixed original E103 descendant")
    for name, value in random.items():
        if name != "action_keys":
            exact(raw[name], value, "independently drawn " + name)
    seconds = typed(raw, "policy_seconds", (3,), "f")
    require(seconds.dtype == np.float64 and np.all(seconds > 0), "all synchronized policy timings are retained")
    schedule = {name: raw[name] for name in ("phase_delay", "directions", "quota_offsets")}
    rows, lives = {}, {}
    for arm in ARMS:
        prefix = arm + "_"
        life = {field: raw[prefix + field] for field in life_fields}
        overrides = []
        mode = arm.rsplit("_", 1)[0]
        for rule in (0, 1):
            _, override = audit_life({name: value[rule] for name, value in life.items()}, mode, schedule, random["action_keys"])
            overrides.append(override)
        food, poison, before, after = (life[name] for name in ("food", "poison", "alive_before", "alive_after"))
        rows[arm] = event_outcomes(food, poison, before, after, np.stack(overrides))
        rows[arm]["changed_action_count"] = life["override_changed"].sum(1, dtype=np.int32)
        counts = np.stack([np.stack([value[:, start:stop].sum(axis=(1, 2)) for value in (food, poison)], axis=-1)
                           for start, stop in ((0, 150), (150, 300))], axis=1).astype(np.float32)
        exact(life["native_counts"], counts, "native contact counts across all original animals")
        exact(life["native_fitness"], (counts[:, :, 0] - counts[:, :, 1]).sum(1) / np.float32(4), "native food-minus-poison score")
        exact(life["native_survived"], after[:, -1].mean(1).astype(np.float32), "native final survival")
        lives[arm] = life
    for mode in POLICIES:
        steady, changed = lives[mode + "_stationary"], lives[mode + "_reversal"]
        for field, value in steady.items():
            if value.ndim >= 2 and value.shape[1] == 300:
                exact(changed[field][:, :150], value[:, :150], "identical same-policy pre-switch histories")
        for field in (*STATE_INTS, *STATE_BOOLS, "alive_before", "observed_denergy", "energy_before", "ordinary_action", "action", "efference", "incoming_efference", "action_key"):
            exact(changed[field][:, 150], steady[field][:, 150], "the decision cannot see the upcoming reversal outcome")
        exact(changed["food"][:, 150], steady["poison"][:, 150], "turn150 changes labels on the same contact")
        exact(changed["poison"][:, 150], steady["food"][:, 150], "turn150 reverses both item labels")
    return rows


def event_outcomes(food, poison, before, after, override):
    result = prior.event_outcomes(food, poison, before, after)
    require(override.shape == food.shape and override.dtype == np.bool_ and not (override & ~before).any(),
            "only living original animals receive overrides")
    for name, contact in (("first_food_latency", food), ("first_any_latency", food | poison)):
        post = contact[:, 150:200]
        result[name] = np.where(post.any(1), post.argmax(1), 50).astype(np.int32)
    result["override_count"] = override.sum(1, dtype=np.int32)
    result["live_turns"] = before.sum(1, dtype=np.int32)
    return result


def decisions(components, comparisons):
    def evidence(value, floor=0.):
        return value["se"] is not None and value["mean"] >= floor and value["mean"] > 2 * value["se"]
    criteria = dict(adaptive_late_positive=evidence(components["adaptive_reversal_late"]))
    for control in ("ordinary", "quota"):
        value = comparisons[control]
        criteria[control + "_latency_improves"] = evidence(value["latency_saved"])
        criteria[control + "_late_Q_improves"] = evidence(value["late_Q_difference"])
        criteria[control + "_adjusted_Q_improves"] = evidence(value["adjusted_late_Q"], .05)
        for name, criterion in (("stationary_late_Q", "stationary_late_noninferiority"),
                                ("reversal_end_survival", "reversal_survival_noninferiority"),
                                ("stationary_end_survival", "stationary_survival_noninferiority")):
            lower = value["safeguards"][name]["lower_2se"]
            criteria[control + "_" + criterion] = lower is not None and lower > -.05
    return dict(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, passed in criteria.items() if not passed])


def population_summary(values, rule=None):
    def family(array):
        selected = array if rule is None else array[:, rule:rule+1]
        return selected.astype(np.float64).mean(axis=(1, selected.ndim - 1))
    reduced = {arm: {name: family(array) for name, array in fields.items()} for arm, fields in values.items()}
    arms = {}
    for arm in ARMS:
        selected = values[arm] if rule is None else {name: value[:, rule:rule+1] for name, value in values[arm].items()}
        rate = selected["override_count"].sum(axis=(1, 2)) / selected["live_turns"].sum(axis=(1, 2))
        arms[arm] = dict(windows=[dict(index=index, start=start, end=stop,
            **{name: stat(reduced[arm][name][:, index]) for name in ("Q", "first_food", "first_poison", "no_contact", "survival")})
            for index, (start, stop) in enumerate(zip(EDGES[:-1], EDGES[1:]))],
            **{name: stat(reduced[arm][name]) for name in ("end_survival", "first_food_latency", "first_any_latency", "override_count", "changed_action_count")},
            override_live_fraction=stat(rate),
            changed_action_live_fraction=stat(selected["changed_action_count"].sum(axis=(1, 2)) / selected["live_turns"].sum(axis=(1, 2))))
    adaptive, stationary = reduced["adaptive_reversal"], reduced["adaptive_stationary"]
    components = dict(adaptive_reversal_late=stat(adaptive["Q"][:, 5]))
    comparisons = {}
    for control, policy in CONTROLS.items():
        changed, steady = reduced[policy + "_reversal"], reduced[policy + "_stationary"]
        late_gain = adaptive["Q"][:, 5] - changed["Q"][:, 5]
        steady_gain = stationary["Q"][:, 5] - steady["Q"][:, 5]
        comparisons[control] = dict(latency_saved=stat(changed["first_food_latency"] - adaptive["first_food_latency"]),
            late_Q_difference=stat(late_gain), adjusted_late_Q=stat(late_gain - steady_gain), safeguards=dict(
                stationary_late_Q=prior.bounded_difference(steady_gain),
                reversal_end_survival=prior.bounded_difference(adaptive["end_survival"] - changed["end_survival"]),
                stationary_end_survival=prior.bounded_difference(stationary["end_survival"] - steady["end_survival"])))
    return dict(arms=arms, components=components, comparisons=comparisons)


def summarize(rows):
    require(len(rows) >= 2, "sample uncertainty needs independent assigned families")
    fields = {"Q", "first_food", "first_poison", "no_contact", "survival", "end_survival",
              "first_food_latency", "first_any_latency", "override_count", "changed_action_count", "live_turns"}
    require(all(set(row) == set(ARMS) and all(set(row[arm]) == fields for arm in ARMS) for row in rows),
            "complete outcomes for six assigned policies and schedules")
    values = {arm: {field: np.stack([row[arm][field] for row in rows]) for field in fields} for arm in ARMS}
    window_fields = {"Q", "first_food", "first_poison", "no_contact", "survival"}
    for arm in ARMS:
        v = values[arm]
        for field, array in v.items():
            expected = (len(rows), 2, 6, 4) if field in window_fields else (len(rows), 2, 4)
            require(array.shape == expected and np.isfinite(array).all(), "all original rules/animals/windows retained")
        require(np.isin(v["Q"], (-1, 0, 1)).all(), "signed first-item values")
        for field, expected in (("first_food", v["Q"] == 1), ("first_poison", v["Q"] == -1), ("no_contact", v["Q"] == 0)):
            require(np.array_equal(v[field], expected), "all-assigned first-item partition")
        require(np.isin(v["survival"], (0, 1)).all() and np.array_equal(v["end_survival"], v["survival"][:, :, -1]),
                "all-assigned survival partition")
        require(np.all((v["first_any_latency"] >= 0) & (v["first_any_latency"] <= v["first_food_latency"]))
                and np.all(v["first_food_latency"] <= 50), "food latency and all-assigned missing-food sentinel")
        require(np.all((v["live_turns"] >= 1) & (v["live_turns"] <= 300))
                and np.all((v["override_count"] >= 0) & (v["override_count"] <= v["live_turns"]))
                and np.all((v["changed_action_count"] >= 0) & (v["changed_action_count"] <= v["override_count"])), "realized live exposure")
    pooled = population_summary(values)
    return dict(families=len(rows), initial_rule_replays=2 * len(rows), policy_history_rollouts=12 * len(rows),
        agents_per_rule=4, bin_edges=list(EDGES), flip_step=150, **pooled,
        gate=decisions(pooled["components"], pooled["comparisons"]),
        by_initial_rule={str(rule): population_summary(values, rule) for rule in (0, 1)})


def replicated_gates(panels):
    require(len(panels) == 3, "all three separately registered original lineages required")
    passed = []
    for index, panel in enumerate(panels):
        expected = decisions(panel["components"], panel["comparisons"])
        same(panel["gate"], expected, "all behavior criteria within the same lineage")
        if expected["passed"]:
            passed.append(index)
    return dict(timed_exploration=dict(count=len(passed), seeds=passed, passed=len(passed) == 3))


def operational_cost(times):
    value = np.asarray(times, np.float64)
    require(value.shape == (4, 3) and np.isfinite(value).all() and np.all(value > 0), "four synchronized timing records for all policies")
    ordinary, adaptive, quota = map(float, np.median(value[1:], axis=0))
    return dict(warm_family_indices=[1, 2, 3], ordinary_median_seconds=ordinary,
        adaptive_median_seconds=adaptive, quota_median_seconds=quota, ratio=adaptive / ordinary,
        quota_ratio=quota / ordinary, max_ratio=1.5, passed=adaptive / ordinary <= 1.5)


class Auditor(prior.Auditor):
    RUN_PREFIX = "E131"
    MASTER = 2026121000
    PROTOCOL = PROTOCOL
    SOURCE_PATHS = SOURCE_PATHS
    MAX_SECONDS = {"development": 200, "final": 600}

    def completed_bundle(self, development=False):
        count, names = (4, ["E131-benchmark"]) if development else (64, [f"E131-A{index}" for index in range(3)])
        folders, pending = [], []
        for name in names:
            found = list(self.path("runs").glob("*_" + name))
            require(len(found) <= 1, "duplicate registered exploration panel " + name)
            if not found or not (found[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = found[0].relative_to(self.root).as_posix()
            state = self.json(folder + "/status.json")
            filenames = {file.name for file in (found[0] / "families").glob("*.npz")}
            if (state.get("status") != "finished" or state.get("completed_families") != count
                    or state.get("expected_families") != count or not (found[0] / "summary.json").is_file()
                    or filenames != {f"family_{family:04d}.npz" for family in range(count)}):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return folders

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master, name = (4, self.MASTER, "E131-benchmark") if development else (64, self.MASTER + 1 + index, f"E131-A{index}")
        split = "development" if development else "final"
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={}, generation=0,
            seed_index=index, eval_seed=master, expected_families=count, source_generations=[200],
            statistical_unit="independent family averaging four original animals and both rules",
            arms=list(ARMS), key_roles=["world_action", "development", "exploration"], initial_rules=[0, 1], bin_edges=list(EDGES),
            flip_step=150, raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
            max_seconds=self.MAX_SECONDS[split], scientific_result=False, policy_order=list(POLICIES),
            assignment_tag=0xE131, policy_parameters=POLICY_PARAMETERS), "registered phased exploration comparison")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=PROTOCOL,
            condition_split=split, allow_dirty=False, resume=None, cfg=None, no_plastic=False,
            quick=False, objective=None, memory_condition=None, max_train_seconds=None,
            pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10), "fresh immutable invocation")
        require(config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off")
                and not config["args"].get("persistence", False)
                and config.get("retry_of") is None and config.get("recovery_of") is None, "no unregistered intervention or recovery")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
                and self.code_trees(commit) == self.expected_trees, "clean scientific source matches registration")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete exploration and native transitive source manifest")
        context = type("SourceContext", (), {"root": self.root, "source": commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
        for path, digest in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        if self.code is None:
            self.code = config["source_code"]
        same(config["source_code"], self.code, "development and all lineages use identical scientific source bytes")
        path, digest = base.ORIGINALS[index]
        subset(config["source_genome"], dict(path=path, sha256=digest, training_seed=index), "fixed original E103 descendant")
        content = self.read(path)
        same(hashlib.sha256(content).hexdigest(), digest, "pinned original genome file")
        require(self.read(folder + "/genome_final.npy") == content, "unchanged copied original genome")
        genome = np.load(io.BytesIO(content), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite rules-v0 genome")
        same(config["args"]["start"], "file:" + path, "correct original source lineage")
        same(config["assignments"]["path"], "assignments.npz", "complete assignment archive")
        schedule = self.artifact(folder, config["assignments"])
        require(set(schedule) == {"family_keys", "initial_rules"}, "only the registered fixed family assignment")
        exact(schedule["initial_rules"], np.asarray([0, 1], np.int32), "both hidden-rule assignments")
        jax, derive = random_helpers()
        same(config["runtime"]["jax"], jax.__version__, "same pinned random runtime")
        root = jax.random.fold_in(jax.random.PRNGKey(master), 0xE131)
        expected = np.stack([np.asarray(jax.random.split(jax.random.fold_in(root, family), 3)) for family in range(count)])
        exact(schedule["family_keys"], expected, "new tagged family world/birth/exploration roles")
        self.register_keys(expected.reshape(-1, 2), "fresh native and policy roots")
        random = [{name: np.asarray(value) for name, value in derive(keys).items()} for keys in expected]
        for evidence in random:
            for field in ("birth_keys", "schedule_stream_keys", "phase_keys", "direction_keys", "offset_keys"):
                self.register_keys(evidence[field].reshape(-1, 2), "independent descendant " + field)
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "every assigned family appears once in the journal")
        for family, record in enumerate(records):
            subset(record, dict(family=family, keys=expected[family].tolist()), "ordered complete assignment journal")
            seconds = np.asarray(record["policy_seconds"], np.float64)
            require(seconds.shape == (3,) and np.isfinite(seconds).all() and (seconds > 0).all()
                    and np.isfinite(record["family_seconds"]) and seconds.sum() <= record["family_seconds"]
                    and np.isfinite(record["elapsed_seconds"]) and 0 < record["elapsed_seconds"] <= self.MAX_SECONDS[split],
                    "complete synchronized timings inside the family and invocation budgets")
        return config, schedule, records, genome, random

    def panel(self, folder, index, development, prepared):
        config, schedule, records, genome, random = prepared
        count = config["expected_families"]
        state, summary = self.json(folder + "/status.json"), self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, evaluation_seed=config["eval_seed"], seed_index=index,
            artifact_validity="passed", scientific_result=False, gate_status="not_applicable" if development else "undecided"),
            "complete immutable exploration panel")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"],
                "all assigned families within the registered cap")
        for document in (state, summary):
            subset(document, dict(status="finished", completed_families=count, expected_families=count, policy_order=list(POLICIES)),
                   "all policy timing observations retained")
            for field in ("family_seconds", "family_wall_seconds"):
                values = np.asarray(document[field], np.float64)
                require(values.shape == (count,) and np.isfinite(values).all() and (values >= 0).all(), "complete nonnegative family timings")
            exact(np.asarray(document["family_policy_seconds"], np.float64),
                  np.asarray([row["policy_seconds"] for row in records], np.float64), "raw policy timings linked to journal and state")
            same(document["family_seconds"], [row["family_seconds"] for row in records], "journal duration linkage")
        rows = []
        for family, record in enumerate(records):
            started = time.monotonic()
            manifest = summary["raw_files"][family]
            subset(manifest, dict(index=family, path=f"families/family_{family:04d}.npz"), "every raw family retained in order")
            same(record["raw_file"], manifest, "journal and exact raw artifact linkage")
            raw = self.artifact(folder, manifest)
            exact(raw["policy_seconds"], np.asarray(record["policy_seconds"], np.float64), "actual synchronized timings archived unchanged")
            rows.append(audit_family(raw, family, genome, schedule["family_keys"][family], random[family]))
            self.family_seconds.append(time.monotonic() - started)
        measured = summarize(rows)
        same(summary["measurements"], measured, "independent raw outcomes, family uncertainty and all behavior gates")
        result = dict(index=index, run=folder, measurements=measured)
        if development:
            result["operational_cost"] = operational_cost(state["family_policy_seconds"])
        return result


def synthetic_life(mode="adaptive", *, food=None, poison=None, schedule=None):
    """Hand-assigned outcomes for bookkeeping checks; no world or neural engine."""
    schedule = schedule or dict(phase_delay=np.asarray([0, 5, 13, 24], np.int32),
        directions=np.tile(np.arange(1, 5, dtype=np.int32), (12, 1)),
        quota_offsets=np.tile(np.asarray([0, 7, 18, 24], np.int32), (12, 1)))
    food = np.zeros((300, 4), bool) if food is None else food.copy()
    poison = np.zeros((300, 4), bool) if poison is None else poison.copy()
    life = {name: np.zeros((300, 4), np.float32) for name in
            ("energy_before", "unclipped_energy", "energy_after", "observed_denergy", "denergy")}
    before, after = np.zeros((300, 4), bool), np.zeros((300, 4), bool)
    energy, living = np.ones(4, np.float32), np.ones(4, bool)
    for tick in range(300):
        before[tick] = living
        food[tick] &= living
        poison[tick] &= living
        unclipped = energy - np.float32(.003) * living + np.float32(.25) * food[tick] - np.float32(.25) * poison[tick]
        next_energy = np.clip(unclipped, np.float32(0), np.float32(2))
        life["energy_before"][tick], life["unclipped_energy"][tick], life["energy_after"][tick] = energy, unclipped, next_energy
        life["denergy"][tick] = (next_energy - energy) * living
        if tick:
            life["observed_denergy"][tick] = life["denergy"][tick - 1]
        living = living & (next_energy > 0)
        after[tick], energy = living, next_energy
    ordinary = np.tile(np.asarray([1, 0, 3, 0], np.int32), (300, 1))
    state = reconstruct_policy(mode, life["observed_denergy"], before, ordinary, schedule)
    efference = np.eye(5, dtype=np.float32)[state["action"]]
    incoming = np.zeros_like(efference)
    incoming[1:] = efference[:-1]
    keys = np.arange(600, dtype=np.uint32).reshape(300, 2)
    life.update(food=food, poison=poison, alive_before=before, alive_after=after, ordinary_action=ordinary,
        **state, efference=efference, incoming_efference=incoming, action_key=keys,
        override_changed=state["override"] & (state["action"] != ordinary))
    return life, schedule, keys


def synthetic_rows():
    rows = []
    for family in range(4):
        row = {}
        for arm in ARMS:
            q = np.zeros((2, 6, 4), np.int8)
            q[:, :, :2] = 1
            q[:, 3] = 1
            if arm.endswith("reversal"):
                q[:, 5] = 0
                q[:, 5, :family + int(arm.startswith("adaptive_"))] = 1
            overrides = np.full((2, 4), 0 if arm.startswith("unchanged_") else 12, np.int32)
            latency = np.full((2, 4), 10 + family if arm.startswith("adaptive_") else 20 + family, np.int32)
            row[arm] = dict(Q=q, first_food=q == 1, first_poison=q == -1, no_contact=q == 0,
                survival=np.ones_like(q, bool), end_survival=np.ones((2, 4), bool),
                first_food_latency=latency, first_any_latency=latency.copy(), override_count=overrides,
                changed_action_count=overrides // 2, live_turns=np.full((2, 4), 300, np.int32))
        rows.append(row)
    return rows


def self_check():
    life, schedule, keys = synthetic_life()
    audit_life(life, "adaptive", schedule, keys)
    require(np.flatnonzero(life["override"][:, 0]).tolist() == [12] + list(range(25, 300, 25)),
            "birth, no-positive feedback and cross-block counter timing")
    require(not life["override"][:24, 3].any() and life["override"][24, 3], "independent phase prevents early override")
    require(life["override"][:, 0].any() and not life["override_changed"][:, 0].any(), "pulse and changed action are distinct")
    for field, index in (("counter", (12, 0)), ("spent_before", (25, 0)), ("observed_denergy", (12, 0)),
                         ("efference", (12, 0, 0)), ("incoming_efference", (13, 0, 0))):
        corrupt = copy.deepcopy(life)
        corrupt[field][index] = not bool(corrupt[field][index]) if corrupt[field].dtype == np.bool_ else corrupt[field][index] + 1
        try:
            audit_life(corrupt, "adaptive", schedule, keys)
        except base.AuditError:
            continue
        raise base.AuditError("corrupted policy evidence accepted: " + field)
    empty = np.zeros((2, 300, 4), bool)
    events = event_outcomes(empty, empty, np.ones_like(empty), np.ones_like(empty), empty)
    require((events["first_food_latency"] == 50).all(), "every no-food animal remains assigned latency50")
    result = summarize(synthetic_rows())
    require(result["gate"]["passed"] and result["comparisons"]["ordinary"]["late_Q_difference"]["se"] == 0,
            "paired family contrasts have their own uncertainty")
    components, comparisons = copy.deepcopy(result["components"]), copy.deepcopy(result["comparisons"])
    comparisons["ordinary"]["late_Q_difference"] = dict(mean=0., se=0., n=4)
    require(not decisions(components, comparisons)["passed"], "stationary deterioration alone cannot establish benefit")
    panels = [copy.deepcopy(result) for _ in range(3)]
    require(replicated_gates(panels)["timed_exploration"]["passed"], "complete three-lineage evidence")
    times = np.asarray([[100., 1000., 100.], [1., 1.5, 1.], [1., 1.5, 1.], [1., 1.5, 1.]])
    require(operational_cost(times)["passed"], "cold compilation excluded and ratio boundary inclusive")
    require("jax" not in sys.modules and not any(name.startswith("seed.") for name in sys.modules),
            "self-check imports no simulator or PRNG runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=7, simulator_imported=False,
        coverage=["birth and previous realized feedback timing", "independent phases and one pulse per complete live block",
                  "pulse versus actual-action change and incoming motor feedback", "counter/feedback/effector corruption",
                  "all-assigned latencies and paired family uncertainty", "direct advantage and all-three behavior gate",
                  "separate inclusive warm-runtime admission"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2, allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required")
    started, auditor = time.monotonic(), None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        dev_folder = folders[0] if args.development else auditor.completed_bundle(True)[0]
        dev_prepared = auditor.metadata(dev_folder, 0, True)
        prepared = [dev_prepared] if args.development else [auditor.metadata(folder, index, False) for index, folder in enumerate(folders)]
        dev_panel = auditor.panel(dev_folder, 0, True, dev_prepared)
        cost = dev_panel.pop("operational_cost")
        if args.development:
            panels, gates = [dev_panel], None
        else:
            require(cost["passed"], "development runtime admission failed; no final outcomes opened")
            panels = [auditor.panel(folder, index, False, prepared[index]) for index, folder in enumerate(folders)]
            gates = replicated_gates([panel["measurements"] for panel in panels])
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, replicated_gates=gates, operational_cost=cost,
            adoption=None if args.development else dict(behavioral_passed=gates["timed_exploration"]["passed"],
                operational_passed=cost["passed"], passed=gates["timed_exploration"]["passed"] and cost["passed"]),
            limitations=["Recorded energy, eligibility, phased quotas, executed and next-tick motor feedback, events and family statistics are independently reconstructed; neural weights and full grid transitions are not replayed.",
                "The ordinary sampled action is source-verified by historical parity tests; logits are not retained, so sampled action draws are not independently regenerated.",
                "Only the pinned JAX PRNG is used for assignment, phase, direction, offset and native action-key regeneration; no simulator is imported.",
                "Synchronized wall timings are operational observations, with raw/journal/status linkage and a separate admission gate; the audit cannot independently remeasure a past clock interval.",
                "This tests engineered exploration timing and may change feedback exposure. It does not establish better learning from identical experience or an evolved exploration mechanism."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), replicated_gates=None)
        code = 2
    except (base.AuditError, OSError, KeyError, ValueError, TypeError, IndexError, zipfile.BadZipFile, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", replicated_gates=None)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        elapsed, reconstruction = time.monotonic() - started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed - reconstruction,
            reconstruction_seconds=reconstruction, family_seconds=auditor.family_seconds, completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
