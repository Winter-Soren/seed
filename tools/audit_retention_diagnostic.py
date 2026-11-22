"""Independent E128 retention reconstruction from immutable traces; no simulator."""
import argparse
import copy
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
import audit_encounter_diagnostic as encounter

base, prior, source_bytes = encounter.base, encounter.prior, encounter.source_bytes
require, same, subset, typed = base.require, base.same, base.subset, base.typed
exact, fingerprint, prefixed, stat = encounter.exact, encounter.fingerprint, encounter.prefixed, encounter.stat
PROTOCOL = "retention-diagnostic-v1"
ARMS = ("plastic_stationary", "plastic_reversal", "direct_stationary", "direct_reversal")
DELAYS = (0, 1, 33)
KEY_ROLES = ("script", "birth", "chamber", "grid", "action", "probe", "retention")
LOCAL_FIELDS = ("p_food", "p_poison", "approach", "other", "S")
GRID_FIELDS = ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")
SOURCE_PATHS = {"seed/seed.py", "seed/direct.py", "seed/encounter_diagnostic.py", "seed/retention_diagnostic.py",
                "scripts/run_encounter_diagnostic.py", "scripts/run_retention_diagnostic.py"}
CFG, MOVES, BODY, STATE_FIELDS = encounter.CFG, encounter.MOVES, encounter.BODY, encounter.STATE_FIELDS


def local_by_rule(probabilities, placements):
    values = np.asarray(probabilities, np.float64)
    require(values.ndim == 4 and values.shape[0] == 2 and values.shape[-2:] == (12, 5), "all rules, placements and five actions")
    require(placements.shape == (12, 2) and set(map(tuple, placements.tolist()))
            == {(a, b) for a in range(1, 5) for b in range(1, 5) if a != b}, "complete ordered probe placements")
    rows = []
    for rule in (0, 1):
        food, poison = [np.take_along_axis(values[rule], placements[None, :, color, None], axis=-1)[..., 0].mean(-1)
                        for color in (1 - rule, rule)]
        rows.append(np.stack((food, poison, food + poison, 1 - food - poison, food - poison), axis=-1))
    return np.stack(rows)


def change(values):
    result = stat(values)
    result["lower_2se"] = None if result["se"] is None else result["mean"] - 2 * result["se"]
    return result


def retention_gate(local, grid):
    local_times, grid_times = local["by_delay"], grid["by_delay"]
    evidence = encounter.beyond
    local_criteria = dict(S0=bool(evidence(local_times["0"]["signed_preference"], .10)),
        S33=bool(evidence(local_times["33"]["signed_preference"], .10)),
        C0=bool(evidence(local_times["0"]["reversal_minus_stationary"], .10)),
        C33=bool(evidence(local_times["33"]["reversal_minus_stationary"], .10)),
        retention=local["change_33_minus_0"]["lower_2se"] is not None and local["change_33_minus_0"]["lower_2se"] > -.05)
    grid_criteria = dict(Y0=bool(evidence(grid_times["0"]["food_minus_poison"])),
        Y33=bool(evidence(grid_times["33"]["food_minus_poison"])),
        C33=bool(evidence(grid_times["33"]["reversal_minus_stationary"], .10)),
        retention=grid["change_33_minus_0"]["lower_2se"] is not None and grid["change_33_minus_0"]["lower_2se"] > -.05)
    for result, criteria in ((local, local_criteria), (grid, grid_criteria)):
        result.update(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, passed in criteria.items() if not passed])
    return bool(local["passed"] and grid["passed"])


def summarize(rows, *, delays=DELAYS):
    require(len(rows) >= 2, "at least two independent families for sample SE")
    pre = {arm: np.stack([row[arm + "_pre_local_by_rule"] for row in rows]).astype(np.float64) for arm in ARMS}
    local = {arm: np.stack([row[arm + "_local_by_rule"] for row in rows]).astype(np.float64) for arm in ARMS}
    grid = {arm: np.stack([row[arm + "_grid_by_rule"] for row in rows]).astype(np.float64) for arm in ARMS}
    for arm in ARMS:
        require(pre[arm].shape == (len(rows), 2, 5) and local[arm].shape == (len(rows), 2, len(delays), 5)
                and grid[arm].shape == (len(rows), 2, len(delays), 7), "family/rule/delay statistical units")
    result = dict(families=len(rows), initial_rule_replays=2 * len(rows), retention_ticks=list(delays),
                  arms={}, architectures={}, by_initial_rule={"0": {}, "1": {}})
    for arm in ARMS:
        by_delay = {str(delay): dict(local={field: stat(local[arm][:, :, point, column].mean(1)) for column, field in enumerate(LOCAL_FIELDS)},
            grid={field: stat(grid[arm][:, :, point, column].mean(1)) for column, field in enumerate(GRID_FIELDS)})
            for point, delay in enumerate(delays)}
        result["arms"][arm] = dict(pre_old=stat(-pre[arm][:, :, 4].mean(1)),
            pre_local={field: stat(pre[arm][:, :, column].mean(1)) for column, field in enumerate(LOCAL_FIELDS)}, by_delay=by_delay)
    for architecture in ("plastic", "direct"):
        rev, steady = architecture + "_reversal", architecture + "_stationary"
        signed, contrast = local[rev][..., 4], local[rev][..., 4] - local[steady][..., 4]
        good, net, grid_contrast = grid[rev][..., 0], grid[rev][..., 3], grid[rev][..., 0] - grid[steady][..., 0]
        signed_family, good_family = signed.mean(1), good.mean(1)
        contrast_family = signed_family - local[steady][..., 4].mean(1)
        grid_contrast_family = good_family - grid[steady][..., 0].mean(1)
        local_result = dict(by_delay={str(delay): dict(signed_preference=stat(signed_family[:, point]),
            reversal_minus_stationary=stat(contrast_family[:, point])) for point, delay in enumerate(delays)},
            change_33_minus_0=change(signed_family[:, -1] - signed_family[:, 0]))
        grid_result = dict(by_delay={str(delay): dict(first_food=stat(good_family[:, point]),
            food_minus_poison=stat(net[:, :, point].mean(1)), reversal_minus_stationary=stat(grid_contrast_family[:, point]))
            for point, delay in enumerate(delays)}, change_33_minus_0=change(good_family[:, -1] - good_family[:, 0]))
        combined = retention_gate(local_result, grid_result)
        result["architectures"][architecture] = dict(local=local_result, grid=grid_result, combined_passed=combined)
        for rule in (0, 1):
            result["by_initial_rule"][str(rule)][architecture] = dict(pre_old=stat(-pre[rev][:, rule, 4]),
                by_delay={str(delay): {field: stat(values[:, rule, point])
                    for field, values in (("S", signed), ("C", contrast), ("G", good), ("Y", net), ("grid_difference", grid_contrast))}
                    for point, delay in enumerate(delays)})
    return result


def replicated_gates(panels):
    require(len(panels) == 3, "all three complete lineages required")
    result = {}
    for architecture in ("plastic", "direct"):
        local = [index for index, panel in enumerate(panels) if panel["architectures"][architecture]["local"]["passed"]]
        combined = [index for index in local if panels[index]["architectures"][architecture]["grid"]["passed"]]
        result[architecture] = {name: dict(count=len(indices), seeds=indices, passed=len(indices) == 3)
                                for name, indices in (("local", local), ("combined", combined))}
    return result


def state_hashes(state):
    for field in STATE_FIELDS:
        require(typed(state, field + "_sha256", (32,), "u").dtype == np.uint8, "actual complete neural-state fingerprints")


def state_at(state, trace, initial, stop, clock_start=0):
    """Reconstruct the physical carry and activity at a saved trajectory prefix."""
    state_hashes(state)
    for field, recorded in (("grid", "grid"), ("pos", "position"), ("energy", "energy"), ("alive", "alive_after"),
                            ("sig", "signal"), ("denergy", "denergy"), ("key", "key_after")):
        exact(state[field], trace[recorded][stop - 1], "saved state is the actual trajectory prefix " + field)
    counts = initial["counts"].copy()
    counts[0, 0] += np.asarray([trace[field][:stop].sum() for field in ("food", "poison")], counts.dtype)
    exact(state["counts"], counts, "all actual contacts retained in the saved carry")
    require(int(state["t"]) == clock_start + stop, "saved episode clock is continuous")
    exact(state["x_sha256"], fingerprint(trace["activity"][stop - 1]), "saved recurrent state follows the actual activity")


def probe(probe, state, placements, keys):
    for field in STATE_FIELDS:
        exact(typed(probe, "initial_" + field + "_sha256", (12, 32), "u"),
              np.broadcast_to(state[field + "_sha256"], (12, 32)), "discarded probe clones the complete stored brain")
    for field in ("W", "Wp", "mask"):
        exact(probe["initial_" + field + "_sha256"], probe["final_" + field + "_sha256"], "probe never changes fixed neural bytes")
    for field in ("hebbian_enabled", "predictor_enabled", "feedback_applied"):
        require(not typed(probe, field, (12,), "biu", binary=True).any(), "discarded readout never updates or supplies feedback")
    for field in BODY:
        exact(probe["initial_" + field], probe["final_" + field], "probe returns no physical or random-stream feedback")
    for field, expected in (("energy", 1), ("alive", 1), ("denergy", 0), ("sig", 0), ("t", 0), ("counts", 0)):
        require(np.all(probe["initial_" + field] == expected), "matched fresh probe body " + field)
    require(np.all(probe["initial_pos"] == 8), "local probe uses the registered neutral hub")
    exact(probe["initial_key"], keys, "every history and delay uses the same discarded probe streams")
    logits = typed(probe, "logits", (12, 5), "f")
    probabilities = typed(probe, "probabilities", (12, 5), "f")
    encounter.softmax(logits, probabilities)
    require(np.all(np.abs(probe["decision_motor"]) <= 1), "bounded unmasked probe motors")
    exact(logits, np.float32(3) * probe["decision_motor"], "original five-action logits")
    for index, placement in enumerate(placements):
        scene = encounter.pair_grid(placement)
        exact(probe["initial_grid"][index], scene, "the declared two-color scene")
        exact(probe["observation"][index], encounter.observation(scene, np.array([8, 8]), 1, 0)[None],
              "probe has only visible items and standardized body feedback")


def active_interval(acquired, states, retention, key, plastic, rule, *, hebbian=None):
    require(len(states) == 3, "all three registered retention states")
    for state in states:
        state_hashes(state)
    for field in STATE_FIELDS:
        exact(states[0][field + "_sha256"], acquired[field + "_sha256"], "empty-world entry preserves the entire acquired brain")
    for field in BODY:
        if field not in ("grid", "key"):
            exact(states[0][field], acquired[field], "empty-world entry preserves living body carry " + field)
    exact(states[0]["grid"], np.zeros((16, 16), np.int32), "only the scene is cleared")
    exact(states[0]["key"], key, "independent matched retention action stream")
    ticks = int(acquired["t"])
    exact(retention["t"], np.arange(ticks, ticks + 33, dtype=np.int32), "one uninterrupted 33-turn active interval")
    for field in ("grid_before", "grid", "food", "poison", "item"):
        require(not retention[field].any(), "no resource or consumption anywhere during retention")
    normalized = dict(retention, t=retention["t"] - np.int32(ticks))
    encounter.trace_events(normalized, states[0], np.full(33, rule, np.int32),
        hebbian=np.full(33, plastic if hebbian is None else hebbian), predictor=plastic)
    for point, delay in enumerate(DELAYS[1:], 1):
        state_at(states[point], retention, states[0], delay, ticks)


def audit_family(raw, index, pre, plastic, direct, probe_keys, *, arms=ARMS, delays=DELAYS, interval_checks=None):
    require(int(raw["base_index"]) == index and int(raw["pre_contacts"]) == pre and pre in (8, 16), "registered family identity")
    for field, expected in (("initial_rules", [0, 1]), ("target_rules", [1, 0]), ("retention_ticks", delays)):
        exact(raw[field], np.asarray(expected, np.int32), "registered rules and retention intervals")
    require(int(raw["grid_steps"]) == 50, "all fifty autonomous readout turns")
    keys = typed(raw, "keys", (7, 2), "u")
    exact(raw["probe_keys"], probe_keys, "independently derived probe role")
    for architecture, genome in (("plastic", plastic), ("direct", direct)):
        exact(raw[architecture + "_genome_sha256"], fingerprint(genome), "unchanged original controller")
    script = typed(raw, "script", (pre + 16, 5), "iu")
    require(np.isin(script[:, :2], np.arange(1, 5)).all() and np.all(script[:, 0] != script[:, 1])
            and np.isin(script[:, 2], (1, 2)).all(), "registered visible contact alternatives")
    exact(script[:, 3], script[np.arange(len(script)), script[:, 2] - 1], "actual outward contact targets the assigned color")
    exact(MOVES[script[:, 4]], -MOVES[script[:, 3]], "physical return after the real outcome")
    require(np.all(np.sort(script[:, 2].reshape(-1, 2), axis=1) == [1, 2]), "every contact pair includes both colors")
    for block in script.reshape(-1, 8, 5):
        require(set(map(tuple, block[:, [2, 3]].tolist())) == {(color, action) for color in (1, 2) for action in range(1, 5)},
                "each contact block balances color and direction")
    placements = typed(raw, "probe_placements", (12, 2), "iu")
    ticks, forced = 2 * (pre + 16), script[:, [3, 4]].reshape(-1)
    fixed_direct = dict(W=fingerprint(prior.direct_weights(direct)[None]),
        Wp=fingerprint(np.zeros((1, 103, 48), np.float32)),
        mask=fingerprint(((np.arange(157)[:, None] >= 103) & ~np.eye(157, dtype=bool)).astype(np.float32)[None]))
    rows, lives, paired_grid = {}, {}, None
    for arm in arms:
        data, plastic_arm = prefixed(raw, arm + "_"), arm.startswith("plastic")
        rows[arm + "_pre_local_by_rule"] = local_by_rule(data["baseline_probe_probabilities"][:, None], placements)[:, 0]
        rows[arm + "_local_by_rule"] = local_by_rule(data["probe_probabilities"], placements)
        grid_by_rule = []
        for rule in (0, 1):
            life = {field: value[rule] for field, value in data.items()}
            lives[arm, rule] = life
            birth, before, acquired = (prefixed(life, stem) for stem in ("birth_", "pre_", "acquired_"))
            for state in (birth, before, acquired):
                state_hashes(state)
            exact(birth["grid"], np.zeros((16, 16), np.int32), "empty fresh chamber")
            exact(birth["pos"], np.asarray([[8, 8]], np.int32), "fresh chamber hub")
            exact(birth["key"], keys[2], "separate chamber stream")
            exact(birth["x_sha256"], fingerprint(np.zeros((1, 157), np.float32)), "fresh initial activity")
            exact(birth["Wp_sha256"], fingerprint(np.zeros((1, 103, 48), np.float32)), "fresh predictor")
            for field, expected in (("energy", 1), ("alive", 1), ("denergy", 0), ("sig", 0), ("t", 0), ("counts", 0)):
                require(np.all(birth[field] == expected), "fresh acquisition body " + field)
            rule_after = rule if arm.endswith("stationary") else 1 - rule
            rules = np.full(ticks, rule_after, np.int32)
            rules[:2 * pre] = rule
            guided = prefixed(life, "guided_")
            encounter.trace_events(guided, birth, rules, forced=forced, script=script,
                hebbian=np.full(ticks, plastic_arm), predictor=plastic_arm)
            require(np.all(guided["alive_after"] == 1) and np.all((guided["unclipped_energy"] > 0)
                    & (guided["unclipped_energy"] < 2)), "live unclipped guided acquisition")
            state_at(before, guided, birth, 2 * pre)
            state_at(acquired, guided, birth, ticks)
            probe(prefixed(life, "baseline_probe_"), before, placements, probe_keys)
            delay_evidence = prefixed(life, "delay_")
            states = [{field: value[point] for field, value in delay_evidence.items()} for point in range(len(delays))]
            retention = prefixed(life, "retention_")
            interval_check = active_interval if interval_checks is None else interval_checks.get(arm, active_interval)
            interval_check(acquired, states, retention, keys[6], plastic_arm, rule_after)
            grid_values, checked_states = [], [birth, before, acquired, *states]
            for point in range(len(states)):
                state = states[point]
                sample = {field: value[point] for field, value in prefixed(life, "probe_").items()}
                probe(sample, state, placements, probe_keys)
                grid = {field: value[point] for field, value in prefixed(life, "grid_").items()}
                initial, final = prefixed(grid, "initial_"), prefixed(grid, "final_")
                state_hashes(initial)
                for field in STATE_FIELDS:
                    exact(initial[field + "_sha256"], state[field + "_sha256"], "fresh grid gets the unprobed complete delayed brain")
                for field, expected in (("energy", 1), ("alive", 1), ("denergy", 0), ("sig", 0), ("t", 0), ("counts", 0)):
                    require(np.all(initial[field] == expected), "matched fresh grid body " + field)
                exact(initial["key"], keys[4], "separate matched grid action stream")
                require(np.sum(initial["grid"] == 1) == np.sum(initial["grid"] == 2) == 24, "ordinary target resource counts")
                if paired_grid is None:
                    paired_grid = initial["grid"], initial["pos"]
                exact(initial["grid"], paired_grid[0], "same fresh layout across histories, delays, architectures and rules")
                exact(initial["pos"], paired_grid[1], "same fresh readout position")
                trace = {field: value for field, value in grid.items() if not field.startswith(("initial_", "final_"))}
                outcome = encounter.trace_events(trace, initial, np.full(50, 1 - rule, np.int32),
                    hebbian=np.full(50, plastic_arm), predictor=plastic_arm)
                grid_values.append([outcome[field] for field in GRID_FIELDS])
                state_at(final, trace, initial, 50)
                checked_states.extend((initial, final))
            grid_by_rule.append(grid_values)
            for state in checked_states:
                exact(state["mask_sha256"], birth["mask_sha256"], "fixed inherited structural mask")
                if not plastic_arm:
                    for field, expected in fixed_direct.items():
                        exact(state[field + "_sha256"], expected, "independently decoded unbounded direct weights and fixed predictor")
        rows[arm + "_grid_by_rule"] = np.asarray(grid_by_rule, np.float64)
    reference = lives[arms[0], 0]
    for (arm, rule), life in lives.items():
        for field, value in life.items():
            if field.startswith("birth_"):
                exact(value, lives[arm, 0][field], "birth independent of hidden rule")
        for stem in ("guided", "retention"):
            for field in ("key_before", "key_after", "action_key", "regrowth_key", "regrowth_cell"):
                exact(life[stem + "_" + field], reference[stem + "_" + field], "same assigned random stream for every treatment")
        for field in ("key_before", "key_after", "action_key", "regrowth_key", "regrowth_cell"):
            exact(life["grid_" + field], np.broadcast_to(reference["grid_" + field][0], life["grid_" + field].shape), "grid random streams paired across all delays")
        exact(life["baseline_probe_action_key"], reference["baseline_probe_action_key"], "matched discarded baseline readouts")
        exact(life["probe_action_key"], np.broadcast_to(reference["baseline_probe_action_key"], life["probe_action_key"].shape),
              "local randomness is matched across histories and every retention delay")
    for controller in dict.fromkeys(arm.rsplit("_", 1)[0] for arm in arms):
        for rule in (0, 1):
            steady, changed = (lives[controller + "_" + history, rule] for history in ("stationary", "reversal"))
            for field, value in steady.items():
                if field.startswith(("birth_", "pre_", "baseline_probe_")):
                    exact(changed[field], value, "matched histories clone identical pre-boundary state and readout")
                elif field.startswith("guided_"):
                    exact(changed[field][:2 * pre], value[:2 * pre], "histories have the same actual pre-boundary experience")
    return rows


class Auditor(prior.chain.Auditor):
    RUN_PREFIX = "E128"
    MASTER = 2026120700
    PROTOCOL = PROTOCOL
    ARMS = ARMS
    DELAYS = DELAYS
    ASSIGNMENT_TAG = 0xE128
    SOURCE_PATHS = SOURCE_PATHS
    MAX_SECONDS = {"development": 120, "final": 300}
    reconstruct_family = staticmethod(audit_family)
    summarize_rows = staticmethod(summarize)

    def __init__(self, root, source):
        require(re.fullmatch(r"[0-9a-f]{40}", source), "full immutable source commit required")
        super().__init__(root, source)
        self.code, self.family_seconds = None, []

    def completed_bundle(self, development=False):
        count, names = (4, [self.RUN_PREFIX + "-benchmark"]) if development else (128, [f"{self.RUN_PREFIX}-A{index}" for index in range(3)])
        folders, pending = [], []
        for name in names:
            found = list(self.path("runs").glob("*_" + name))
            require(len(found) <= 1, "duplicate registered panel " + name)
            if not found or not (found[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = found[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            filenames = {file.name for file in (found[0] / "families").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_families") != count
                    or status.get("expected_families") != count or not (found[0] / "summary.json").is_file()
                    or filenames != {f"family_{index:04d}.npz" for index in range(count)}):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return folders

    def arrays(self, relative):
        require(self.path(relative).stat().st_size <= 16 << 20, "compressed artifact exceeds 16 MiB")
        content = self.read(relative)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            require(0 < len(entries) <= 2048 and sum(entry.file_size for entry in entries) <= 64 << 20,
                    "retention archive exceeds entry or expanded-byte bound")
            require(len({entry.filename for entry in entries}) == len(entries)
                    and all(entry.filename.endswith(".npy") and "/" not in entry.filename and "\\" not in entry.filename for entry in entries),
                    "unique numeric archive entry names")
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            values = {field: archive[field] for field in archive.files}
        require(all(value.dtype.kind in "biuf" and np.isfinite(value).all() for value in values.values()), "finite numeric evidence only")
        return values

    def artifact(self, folder, manifest):
        path = self.path(folder + "/" + manifest["path"])
        require(path.is_relative_to(self.path(folder)), "artifact leaves its own run")
        relative = path.relative_to(self.root).as_posix()
        values = self.arrays(relative)
        same(self.hashes[relative], manifest["sha256"], "exact saved artifact digest")
        return values

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master, name = (4, self.MASTER, self.RUN_PREFIX + "-benchmark") if development else (128, self.MASTER + 1 + index, f"{self.RUN_PREFIX}-A{index}")
        subset(config, dict(protocol=self.PROTOCOL, condition_split="development" if development else "final", cfg=CFG,
            cfg_overrides={}, generation=0, eval_seed=master, expected_families=count, statistical_unit="independent paired base family",
            arms=list(self.ARMS), key_roles=list(KEY_ROLES), initial_rules=[0, 1], pre_contact_choices=[8, 16], post_contacts=16,
            retention_ticks=list(self.DELAYS), grid_steps=50, probe_placements=12, raw_schema=self.PROTOCOL,
            raw_pattern="families/family_{index:04d}.npz", assignment_stream_tag=self.ASSIGNMENT_TAG,
            max_seconds=self.MAX_SECONDS["development" if development else "final"], scientific_result=False), "registered retention design")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=self.PROTOCOL,
            condition_split=config["condition_split"], allow_dirty=False, resume=None, cfg=None, no_plastic=False,
            objective=None, memory_condition=None), "registered fresh invocation")
        require(config.get("retry_of") is None and config.get("recovery_of") is None, "no unregistered recovery")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
                and self.code_trees(commit) == self.expected_trees, "actual clean source matches registered scientific trees")
        require(set(config["source_code"]) == self.SOURCE_PATHS, "complete transitive source manifest")
        context = type("SourceContext", (), {"root": self.root, "source": commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
        for path, digest in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        if self.code is None:
            self.code = config["source_code"]
        same(config["source_code"], self.code, "development and every final use identical scientific source bytes")
        genomes = []
        for field, pinned, copy, shape in (("source", base.ORIGINALS[index], "plastic_genome.npy", (155,)),
                                         ("direct_source", prior.DIRECT[index], "direct_genome.npy", (8424,))):
            path, digest = pinned
            subset(config[field], dict(path=path, sha256=digest, training_seed=index), "original saved controller")
            content = self.read(path)
            same(hashlib.sha256(content).hexdigest(), digest, "pinned original controller digest")
            require(self.read(folder + "/" + copy) == content, "unchanged copied original controller")
            genome = np.load(io.BytesIO(content), allow_pickle=False)
            require(genome.shape == shape and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite genome encoding")
            genomes.append(genome)
        same(config["args"]["start"], "file:" + base.ORIGINALS[index][0], "original acquisition source, no inherited archive")
        same(config["assignments"]["path"], "assignments.npz", "registered assignment archive")
        schedule = self.artifact(folder, config["assignments"])
        require(set(schedule) == {"key", "permutation", "pre_contacts"}, "exact assignment evidence")
        jax, _ = source_bytes.random_runtime()
        same(config["runtime"]["jax"], jax.__version__, "same archived PRNG runtime")
        key = jax.random.fold_in(jax.random.PRNGKey(master), self.ASSIGNMENT_TAG)
        permutation = np.asarray(jax.random.permutation(key, count), np.int32)
        exact(schedule["key"], np.asarray(key), "new independent assignment stream")
        exact(schedule["permutation"], permutation, "independent balanced assignment order")
        exact(schedule["pre_contacts"], np.repeat(np.array([8, 16], np.int32), count // 2)[permutation], "balanced acquisition lengths")
        self.register_keys(schedule["key"][None], "assignment stream")
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "all assigned families appear exactly once")
        probe_keys = []
        for family, record in enumerate(records):
            subset(record, dict(family=family, pre_contacts=int(schedule["pre_contacts"][family])), "ordered assigned family")
            keys = np.asarray(record["keys"])
            require(keys.shape == (7, 2) and keys.dtype.kind in "iu" and np.all(keys >= 0)
                    and np.all(keys <= np.iinfo(np.uint32).max), "seven unsigned separate roles")
            keys = keys.astype(np.uint32)
            expected = np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(master), family), 7))
            exact(keys, expected, "new independently derived family roles")
            self.register_keys(keys, "independent family")
            probe_keys.append(np.asarray(jax.random.split(keys[5], 12)))
        return config, schedule, records, genomes, probe_keys

    def panel(self, folder, index, development, prepared):
        config, schedule, records, genomes, probe_keys = prepared
        count = config["expected_families"]
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=self.PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, evaluation_seed=config["eval_seed"], seed_index=index,
            artifact_validity="passed", scientific_result=False, gate_status="not_applicable" if development else "undecided"), "complete bounded panel")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"], "complete run within its registered cap")
        rows = []
        for family, record in enumerate(records):
            started = time.monotonic()
            manifest = summary["raw_files"][family]
            subset(manifest, dict(index=family, path=f"families/family_{family:04d}.npz"), "ordered raw family")
            same(record["raw_file"], manifest, "journal and saved-family linkage")
            raw = self.artifact(folder, manifest)
            exact(raw["keys"], np.asarray(record["keys"], np.uint32), "actual assigned role keys")
            rows.append(self.reconstruct_family(raw, family, int(schedule["pre_contacts"][family]), *genomes, probe_keys[family]))
            self.family_seconds.append(time.monotonic() - started)
        measured = self.summarize_rows(rows)
        same(summary["measurements"], measured, "independent complete outcomes, paired family SE and retention decisions")
        return dict(index=index, run=folder, measurements=measured)


def self_check():
    placements = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
    uniform = np.full((2, 3, 12, 5), .2, np.float32)
    encounter.softmax(np.zeros_like(uniform), uniform)
    expected = np.broadcast_to([.2, .2, .4, .6, 0], (2, 3, 5))
    prior.prior.close(local_by_rule(uniform, placements), expected, "all five actions and placements retained")
    rows = []
    for value in (.2, .4, .6, .8):
        row = {}
        for arm in ARMS:
            row[arm + "_pre_local_by_rule"] = np.broadcast_to([.2, .2, .4, .6, 0], (2, 5)).copy()
            local = np.zeros((2, 3, 5), np.float64)
            signed = np.asarray([value, -1, value + .01]) if arm.endswith("reversal") else np.zeros(3)
            local[..., 0], local[..., 1], local[..., 2], local[..., 4] = (1 + signed) / 2, (1 - signed) / 2, 1, signed
            row[arm + "_local_by_rule"] = local
            result = [1, 0, 0, 1, 1, 1, 1] if arm.endswith("reversal") else [0, 1, 0, -1, 1, -1, 1]
            row[arm + "_grid_by_rule"] = np.broadcast_to(result, (2, 3, 7)).copy()
        rows.append(row)
    measured = summarize(rows)
    local = measured["architectures"]["plastic"]["local"]
    require(local["passed"] and local["change_33_minus_0"]["se"] < 1e-15
            and local["by_delay"]["0"]["signed_preference"]["se"] > .1, "paired change has its own family SE")
    require(local["by_delay"]["1"]["signed_preference"]["mean"] == -1 and local["passed"], "one-turn result is descriptive")
    require(encounter.beyond(dict(mean=.1, se=0), .1) and not encounter.beyond(dict(mean=.1, se=.05), .1), "inclusive effect floor and strict evidence boundary")
    altered = copy.deepcopy(measured["architectures"]["plastic"])
    altered["local"]["change_33_minus_0"]["lower_2se"] = -.05
    require(not retention_gate(altered["local"], altered["grid"]) and not altered["local"]["criteria"]["retention"], "retention bound is strict")
    panels = [copy.deepcopy(measured) for _ in range(3)]
    require(replicated_gates(panels)["plastic"]["combined"]["passed"], "three matching successful lineages")
    panels[2]["architectures"]["plastic"]["grid"]["passed"] = False
    require(replicated_gates(panels)["plastic"]["local"]["passed"]
            and not replicated_gates(panels)["plastic"]["combined"]["passed"], "combined requires the same all-three intersection")
    absent = np.zeros(50, bool)
    missing = encounter.grid_outcomes(absent, absent, absent)
    require(missing["food_minus_poison"] == 0 and missing["no_contact"] == 1 and missing["latency"] == 51, "dead and no-contact lives stay in the denominator")
    last = absent.copy()
    last[-1] = True
    require(encounter.grid_outcomes(last, absent, ~absent)["first_food"] == 1, "the fiftieth turn is included")
    require("seed.seed" not in sys.modules and "seed.retention_diagnostic" not in sys.modules and "jax" not in sys.modules,
            "self-check imports no simulator or random runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=6, simulator_imported=False,
        coverage=["all original action alternatives and within-family placements", "paired change SE and family denominator",
                  "one-turn descriptive snapshot", "inclusive effect floors and strict evidence/retention bounds",
                  "all-three same-lineage replication", "assigned no-contact/death and fifty-turn horizon"])


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
        if not args.development:
            auditor.metadata(auditor.completed_bundle(True)[0], 0, True)
        prepared = [auditor.metadata(folder, index, args.development) for index, folder in enumerate(folders)]
        panels = [auditor.panel(folder, index, args.development, prepared[index]) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, replicated_gates=None if args.development else replicated_gates([panel["measurements"] for panel in panels]),
            limitations=["Physical events, observations, clone fingerprints and probabilities are independently reconstructed; neural weight dynamics are not replayed.",
                "Direct weights are independently decoded and checked at saved endpoints; hashes cannot exclude an unrecorded transient change between endpoints.",
                "Role/probe key derivation is replayed using the pinned JAX PRNG; categorical sampling and developmental wiring are not rerun.",
                "The active empty interval includes scene, energy, activity and continuing plastic updates; any change cannot be assigned to time alone."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), replicated_gates=None)
        code = 2
    except (base.AuditError, OSError, KeyError, ValueError, TypeError, IndexError, zipfile.BadZipFile, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", replicated_gates=None)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        elapsed, reconstructed = time.monotonic() - started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed - reconstructed,
            reconstruction_seconds=reconstructed, family_seconds=auditor.family_seconds, completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
