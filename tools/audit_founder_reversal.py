"""Independent E130 event reconstruction and family-paired evolutionary recovery audit."""
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
import audit_retention_diagnostic as prior

base, source_bytes = prior.base, prior.source_bytes
require, same, subset, typed = prior.require, prior.same, prior.subset, prior.typed
exact, fingerprint, stat = prior.exact, prior.fingerprint, prior.stat
PROTOCOL = "founder-reversal-v1"
ARMS = ("gen0_stationary", "gen0_reversal", "gen200_stationary", "gen200_reversal")
EDGES = (0, 50, 100, 150, 200, 250, 300)
CFG = dict(prior.CFG, M=4)
START_HASHES = ("621e5a735e408e487653f7ffc15561be773fbd1053af847ef1bb633fd5243e07",
    "3ec510cab0205915614a7df2d3b217eca201972897842592fcef419b215e5d99",
    "c62051a478c75f3211645968e5a15bb64bcfcdb8f61c366357020f4cff8294b9")
STARTS = tuple((path.rsplit("/", 1)[0] + "/genome_start.npy", digest) for (path, _), digest in zip(base.ORIGINALS, START_HASHES))
SOURCE_PATHS = {"seed/seed.py", "seed/founder_reversal.py", "scripts/run_founder_reversal.py",
    "scripts/run_retention_diagnostic.py", "scripts/run_encounter_diagnostic.py", "scripts/run_cold_donor.py",
    "scripts/run_shelter_evolution.py"}


def event_outcomes(food, poison, before, after):
    for field, value in (("food", food), ("poison", poison), ("alive_before", before), ("alive_after", after)):
        require(value.shape == (2, 300, 4) and value.dtype == np.bool_, "complete original-agent boolean events: " + field)
    require(not (food & poison).any(), "a first item cannot be both food and poison")
    require(before[:, 0].all() and np.array_equal(before[:, 1:], after[:, :-1]) and not (after & ~before).any(),
            "all original agents are born alive and death cannot be undone")
    require(not ((food | poison) & ~before).any(), "dead agents cannot generate contact events")
    f, p = food.reshape(2, 6, 50, 4), poison.reshape(2, 6, 50, 4)
    contact = f | p
    present, first = contact.any(2), contact.argmax(2)
    good = np.take_along_axis(f, first[:, :, None, :], axis=2)[:, :, 0] & present
    bad = np.take_along_axis(p, first[:, :, None, :], axis=2)[:, :, 0] & present
    return dict(Q=good.astype(np.int8) - bad.astype(np.int8), first_food=good, first_poison=bad,
                no_contact=~present, survival=after.reshape(2, 6, 50, 4)[:, :, -1], end_survival=after[:, -1])


def audit_family(raw, index, genomes, keys, births):
    fields = {"base_index", "keys", "birth_keys", "initial_rules", "bin_edges", "flip_step", "gen0_genome_sha256", "gen200_genome_sha256"}
    fields.update(arm + "_" + field for arm in ARMS for field in
                  ("food", "poison", "alive_before", "alive_after", "native_counts", "native_fitness", "native_survived"))
    require(set(raw) == fields, "complete registered event archive without substituted outcomes")
    exact(raw["base_index"], np.asarray(index, np.int32), "all assigned family indices retained in order")
    exact(raw["keys"], keys, "paired world/action and development roles")
    exact(raw["birth_keys"], births, "the four original development streams")
    exact(raw["initial_rules"], np.asarray([0, 1], np.int32), "both initial rules retained")
    exact(raw["bin_edges"], np.asarray(EDGES, np.int32), "fixed end-exclusive fifty-turn windows")
    exact(raw["flip_step"], np.asarray(150, np.int32), "hidden flip before zero-indexed event150")
    for name, genome in zip(("gen0", "gen200"), genomes):
        exact(raw[name + "_genome_sha256"], fingerprint(genome), "fixed original evolution endpoint")
    result, traces = {}, {}
    for arm in ARMS:
        values = {field: typed(raw, arm + "_" + field, (2, 300, 4), "b")
                  for field in ("food", "poison", "alive_before", "alive_after")}
        result[arm] = event_outcomes(values["food"], values["poison"], values["alive_before"], values["alive_after"])
        traces[arm] = values
        counts = np.stack([np.stack([values[field][:, start:stop].sum(axis=(1, 2)) for field in ("food", "poison")], axis=-1)
                           for start, stop in ((0, 150), (150, 300))], axis=1).astype(np.float32)
        exact(typed(raw, arm + "_native_counts", (2, 2, 2), "f"), counts, "native total contacts across original agents and halves")
        fitness = (counts[:, :, 0].sum(1) - counts[:, :, 1].sum(1)) / np.float32(4)
        exact(typed(raw, arm + "_native_fitness", (2,), "f"), fitness, "native per-agent food-minus-poison score")
        exact(typed(raw, arm + "_native_survived", (2,), "f"), values["alive_after"][:, -1].mean(1).astype(np.float32),
              "native all-assigned final survival")
    for genotype in ("gen0", "gen200"):
        steady, changed = traces[genotype + "_stationary"], traces[genotype + "_reversal"]
        for field in traces[genotype + "_stationary"]:
            exact(changed[field][:, :150], steady[field][:, :150],
                  "counterfactual event histories agree before the hidden switch")
        exact(changed["alive_before"][:, 150], steady["alive_before"][:, 150], "same living agents at the switch")
        exact(changed["food"][:, 150], steady["poison"][:, 150], "turn150 relabels the same contact before eating")
        exact(changed["poison"][:, 150], steady["food"][:, 150], "turn150 reverses both item labels")
    return result


def bounded_difference(values):
    estimate = stat(values)
    estimate["lower_2se"] = None if estimate["se"] is None else estimate["mean"] - 2 * estimate["se"]
    return estimate


def decisions(components, safeguards):
    def positive(value, minimum=0.):
        return value["se"] is not None and value["mean"] >= minimum and value["mean"] > 2 * value["se"]
    def retained(name):
        return safeguards[name]["lower_2se"] is not None and safeguards[name]["lower_2se"] > -.05
    criteria = dict(descendant_pre_positive=positive(components["descendant_pre"]),
        descendant_late_positive=positive(components["descendant_reversal_late"]),
        descendant_recovers=positive(components["descendant_recovery"], .05),
        descendant_beats_founder=positive(components["descendant_vs_founder_late"], .05),
        recovery_interaction=positive(components["recovery_interaction"], .05),
        pre_noninferiority=retained("pre_Q"), stationary_late_noninferiority=retained("stationary_late_Q"),
        reversal_survival_noninferiority=retained("reversal_end_survival"),
        stationary_survival_noninferiority=retained("stationary_end_survival"))
    return dict(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, passed in criteria.items() if not passed])


def population_summary(values, rule=None):
    def family(array):
        return array.mean(axis=(1, 3)) if rule is None and array.ndim == 4 else (
            array.mean(axis=(1, 2)) if rule is None else array[:, rule].mean(axis=-1))
    reduced = {arm: {field: family(array.astype(np.float64)) for field, array in fields.items()} for arm, fields in values.items()}
    arms = {}
    for arm in ARMS:
        arms[arm] = dict(windows=[dict(index=index, start=start, end=stop,
            **{field: stat(reduced[arm][field][:, index]) for field in ("Q", "first_food", "first_poison", "no_contact", "survival")})
            for index, (start, stop) in enumerate(zip(EDGES[:-1], EDGES[1:]))], end_survival=stat(reduced[arm]["end_survival"]))
    f, fs, d, ds = (reduced[arm]["Q"] for arm in ("gen0_reversal", "gen0_stationary", "gen200_reversal", "gen200_stationary"))
    recovery = d[:, 5] - d[:, 3]
    interaction = (recovery - (ds[:, 5] - ds[:, 3])) - ((f[:, 5] - f[:, 3]) - (fs[:, 5] - fs[:, 3]))
    components = {name: stat(array) for name, array in dict(descendant_pre=d[:, 2], descendant_reversal_late=d[:, 5],
        descendant_recovery=recovery, descendant_vs_founder_late=d[:, 5] - f[:, 5], recovery_interaction=interaction).items()}
    safeguard_values = dict(pre_Q=d[:, 2] - f[:, 2], stationary_late_Q=ds[:, 5] - fs[:, 5],
        reversal_end_survival=reduced["gen200_reversal"]["end_survival"] - reduced["gen0_reversal"]["end_survival"],
        stationary_end_survival=reduced["gen200_stationary"]["end_survival"] - reduced["gen0_stationary"]["end_survival"])
    return dict(arms=arms, components=components, safeguards={name: bounded_difference(value) for name, value in safeguard_values.items()})


def summarize(rows):
    require(len(rows) >= 2, "sample uncertainty requires independent assigned families")
    values = {arm: {field: np.stack([row[arm][field] for row in rows]) for field in rows[0][arm]} for arm in ARMS}
    for arm in ARMS:
        for field, value in values[arm].items():
            expected = (len(rows), 2, 4) if field == "end_survival" else (len(rows), 2, 6, 4)
            require(value.shape == expected and np.isfinite(value).all(), "all original agents/rules/windows retained")
        q = values[arm]["Q"]
        require(np.isin(q, (-1, 0, 1)).all(), "signed first items have only three possible values")
        for field, expected in (("first_food", q == 1), ("first_poison", q == -1), ("no_contact", q == 0)):
            require(np.array_equal(values[arm][field], expected), "all-assigned first-item partition")
        require(np.isin(values[arm]["survival"], (0, 1)).all()
                and np.array_equal(values[arm]["end_survival"], values[arm]["survival"][:, :, -1]), "all-assigned survival partition")
    pooled = population_summary(values)
    return dict(families=len(rows), initial_rule_replays=2 * len(rows), genotype_history_rollouts=8 * len(rows), agents_per_rule=4,
        bin_edges=list(EDGES), flip_step=150, **pooled, gate=decisions(pooled["components"], pooled["safeguards"]),
        by_initial_rule={str(rule): population_summary(values, rule) for rule in (0, 1)})


def replicated_gates(panels):
    require(len(panels) == 3, "all three matched original lineages required")
    passed = []
    for index, panel in enumerate(panels):
        gate = decisions(panel["components"], panel["safeguards"])
        same(panel["gate"], gate, "every registered criterion applies within the same lineage")
        if gate["passed"]:
            passed.append(index)
    return dict(evolutionary_recovery=dict(count=len(passed), seeds=passed, passed=len(passed) == 3))


class Auditor(prior.Auditor):
    """Reuse bounded archive reading, exact artifact hashes and tracked-text provenance."""
    RUN_PREFIX = "E130"
    MASTER = 2026120900
    PROTOCOL = PROTOCOL
    SOURCE_PATHS = SOURCE_PATHS
    MAX_SECONDS = {"development": 180, "final": 450}

    def completed_bundle(self, development=False):
        count, names = (4, ["E130-benchmark"]) if development else (64, [f"E130-A{index}" for index in range(3)])
        folders, pending = [], []
        for name in names:
            found = list(self.path("runs").glob("*_" + name))
            require(len(found) <= 1, "duplicate registered panel " + name)
            if not found or not (found[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = found[0].relative_to(self.root).as_posix()
            state = self.json(folder + "/status.json")
            filenames = {file.name for file in (found[0] / "families").glob("*.npz")}
            if (state.get("status") != "finished" or state.get("completed_families") != count
                    or state.get("expected_families") != count or not (found[0] / "summary.json").is_file()
                    or filenames != {f"family_{index:04d}.npz" for index in range(count)}):
                pending.append(name)
            else:
                folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no final outcomes opened")
        return folders

    def metadata(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master, name = (4, self.MASTER, "E130-benchmark") if development else (64, self.MASTER + 1 + index, f"E130-A{index}")
        split = "development" if development else "final"
        subset(config, dict(protocol=PROTOCOL, condition_split=split, cfg=CFG, cfg_overrides={}, generation=0,
            seed_index=index, eval_seed=master, expected_families=count, source_generations=[0, 200],
            statistical_unit="independent family averaging four original animals and both rules",
            arms=list(ARMS), key_roles=["world_action", "development"], initial_rules=[0, 1], bin_edges=list(EDGES),
            flip_step=150, raw_schema=PROTOCOL, raw_pattern="families/family_{index:04d}.npz",
            max_seconds=self.MAX_SECONDS[split], scientific_result=False), "fixed native founder comparison")
        subset(config["args"], dict(name=name, seed=master, generations=0, eval_worlds=count, eval_protocol=PROTOCOL,
            condition_split=split, allow_dirty=False, resume=None, cfg=None, no_plastic=False,
            quick=False, objective=None, memory_condition=None, max_train_seconds=None,
            pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10), "registered fresh invocation")
        require(config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off")
                and not config["args"].get("persistence", False)
                and config.get("retry_of") is None and config.get("recovery_of") is None, "no unregistered intervention or recovery")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
                and self.code_trees(commit) == self.expected_trees, "actual clean source matches registered scientific trees")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete transitive source manifest")
        context = type("SourceContext", (), {"root": self.root, "source": commit})()
        attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
        for path, digest in config["source_code"].items():
            blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                capture_output=True, check=True, timeout=15).stdout
            source_bytes.source_hash_representation(blob, digest, attributes[path])
        if self.code is None:
            self.code = config["source_code"]
        same(config["source_code"], self.code, "development and all final panels use identical scientific source bytes")
        genomes = []
        for field, pinned, copy_name in (("source_start", STARTS[index], "genome_start.npy"),
                                        ("source_final", base.ORIGINALS[index], "genome_final.npy")):
            path, digest = pinned
            subset(config[field], dict(path=path, sha256=digest, training_seed=index), "original paired E103 endpoint")
            content = self.read(path)
            same(hashlib.sha256(content).hexdigest(), digest, "pinned original genome file")
            require(self.read(folder + "/" + copy_name) == content, "unchanged copied original genome")
            genome = np.load(io.BytesIO(content), allow_pickle=False)
            require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), "original finite rules-v0 encoding")
            genomes.append(genome)
        same(config["args"]["start"], "file:" + STARTS[index][0], "correct original founder lineage")
        same(config["assignments"]["path"], "assignments.npz", "registered assignment archive")
        schedule = self.artifact(folder, config["assignments"])
        require(set(schedule) == {"family_keys", "initial_rules"}, "complete predeclared assignment evidence")
        exact(schedule["initial_rules"], np.asarray([0, 1], np.int32), "both hidden-rule assignments")
        jax, _ = source_bytes.random_runtime()
        same(config["runtime"]["jax"], jax.__version__, "same archived PRNG runtime")
        stream = jax.random.fold_in(jax.random.PRNGKey(master), 0xE130)
        expected = np.stack([np.asarray(jax.random.split(jax.random.fold_in(stream, family), 2)) for family in range(count)])
        exact(schedule["family_keys"], expected, "new tagged independent paired family streams")
        self.register_keys(expected.reshape(-1, 2), "world/action and development roles")
        births = [np.asarray(jax.random.split(keys[1], 4)) for keys in expected]
        self.register_keys(np.stack(births).reshape(-1, 2), "four original animals in every independent family")
        records = [json.loads(line) for line in self.read(folder + "/metrics.jsonl").decode().splitlines()]
        require(len(records) == count, "every assigned family is journaled exactly once")
        for family, record in enumerate(records):
            subset(record, dict(family=family, keys=expected[family].tolist()), "ordered full assignment journal")
            require(np.isfinite(record["family_seconds"]) and record["family_seconds"] >= 0
                    and np.isfinite(record["elapsed_seconds"]) and 0 < record["elapsed_seconds"] <= self.MAX_SECONDS[split],
                    "finite bounded family timing")
        return config, schedule, records, genomes, births

    def panel(self, folder, index, development, prepared):
        config, schedule, records, genomes, births = prepared
        count = config["expected_families"]
        summary = self.json(folder + "/summary.json")
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            completed_families=count, expected_families=count, evaluation_seed=config["eval_seed"], seed_index=index,
            artifact_validity="passed", scientific_result=False, gate_status="not_applicable" if development else "undecided"),
            "complete immutable founder comparison")
        require(len(summary["raw_files"]) == count and 0 < summary["elapsed_seconds"] <= config["max_seconds"],
                "complete panel within its registered cap")
        for name in ("family_seconds", "family_wall_seconds"):
            values = np.asarray(summary[name], np.float64)
            require(values.shape == (count,) and np.isfinite(values).all() and (values >= 0).all(), "complete nonnegative timing evidence")
        same(summary["family_seconds"], [row["family_seconds"] for row in records], "journal duration linkage")
        rows = []
        for family, record in enumerate(records):
            started = time.monotonic()
            manifest = summary["raw_files"][family]
            subset(manifest, dict(index=family, path=f"families/family_{family:04d}.npz"), "every original raw family in order")
            same(record["raw_file"], manifest, "journal and raw artifact linkage")
            raw = self.artifact(folder, manifest)
            rows.append(audit_family(raw, family, genomes, schedule["family_keys"][family], births[family]))
            self.family_seconds.append(time.monotonic() - started)
        measured = summarize(rows)
        same(summary["measurements"], measured, "independent first-item outcomes, family SE and all registered criteria")
        return dict(index=index, run=folder, measurements=measured)


def synthetic_family(index=0):
    """Tiny event-only fixture; no simulator, randomness or saved research inputs."""
    keys = np.arange(4, dtype=np.uint32).reshape(2, 2)
    births = np.arange(8, 16, dtype=np.uint32).reshape(4, 2)
    genomes = (np.zeros(155, np.float32), np.ones(155, np.float32))
    raw = dict(base_index=np.asarray(index, np.int32), keys=keys, birth_keys=births,
        initial_rules=np.asarray([0, 1], np.int32), bin_edges=np.asarray(EDGES, np.int32), flip_step=np.asarray(150, np.int32),
        gen0_genome_sha256=fingerprint(genomes[0]), gen200_genome_sha256=fingerprint(genomes[1]))
    for arm in ARMS:
        food, poison = np.zeros((2, 300, 4), bool), np.zeros((2, 300, 4), bool)
        food[:, 100, :1 if arm.startswith("gen0_") else 4] = True
        if arm.endswith("stationary"):
            food[:, 150] = food[:, 299] = True
        else:
            poison[:, 150] = True
            if arm.startswith("gen0_"):
                poison[:, 299] = True
            else:
                food[:, 299] = True
        values = dict(food=food, poison=poison, alive_before=np.ones_like(food), alive_after=np.ones_like(food))
        # Separate scalar loops deliberately avoid the vectorized reconstruction.
        counts = np.zeros((2, 2, 2), np.float32)
        for rule in (0, 1):
            for step in range(300):
                for agent in range(4):
                    counts[rule, step // 150, 0] += int(food[rule, step, agent])
                    counts[rule, step // 150, 1] += int(poison[rule, step, agent])
        values.update(native_counts=counts, native_fitness=(counts[:, :, 0] - counts[:, :, 1]).sum(1) / np.float32(4),
                      native_survived=np.ones(2, np.float32))
        raw.update({arm + "_" + name: value for name, value in values.items()})
    return raw, genomes, keys, births


def self_check():
    raw, genomes, keys, births = synthetic_family()
    row = audit_family(raw, 0, genomes, keys, births)
    result = summarize([copy.deepcopy(row) for _ in range(4)])
    require(result["gate"]["passed"] and result["components"]["descendant_recovery"]["mean"] == 2,
            "full paired founder/descendant recovery fixture")
    for field in ("gen200_reversal_native_counts", "gen0_genome_sha256", "birth_keys"):
        altered = copy.deepcopy(raw)
        altered[field].flat[0] += 1
        try:
            audit_family(altered, 0, genomes, keys, births)
        except base.AuditError:
            continue
        raise base.AuditError("corrupted event accounting or provenance accepted: " + field)
    food, poison = np.zeros((2, 300, 4), bool), np.zeros((2, 300, 4), bool)
    before, after = np.ones_like(food), np.ones_like(food)
    after[:, 0:, 3], before[:, 1:, 3] = False, False
    food[:, 49, 0] = food[:, 50, 0] = food[:, 299, 0] = True
    poison[:, 1, 1], food[:, 2, 1] = True, True
    outcomes = event_outcomes(food, poison, before, after)
    require(outcomes["Q"][0, 0].tolist() == [1, -1, 0, 0]
            and outcomes["Q"][0, 1, 0] == 1 and outcomes["Q"][0, 5, 0] == 1, "first item, exact bins and death/no-contact denominator")
    # A positive interaction produced only by founder deterioration cannot pass.
    components, safeguards = copy.deepcopy(result["components"]), copy.deepcopy(result["safeguards"])
    components["descendant_recovery"] = dict(mean=0., se=0., n=4)
    require(not decisions(components, safeguards)["criteria"]["descendant_recovers"], "descendant must itself recover")
    components["descendant_recovery"] = dict(mean=.05, se=0., n=4)
    require(decisions(components, safeguards)["passed"], "inclusive effect floor")
    components["descendant_recovery"]["se"] = .025
    require(not decisions(components, safeguards)["passed"], "strict two-SE evidence")
    components["descendant_recovery"]["se"] = 0.
    safeguards["pre_Q"]["lower_2se"] = -.05
    require(not decisions(components, safeguards)["passed"], "strict noninferiority boundary")
    panels = [copy.deepcopy(result) for _ in range(3)]
    require(replicated_gates(panels)["evolutionary_recovery"]["passed"], "all three qualifying original lineages")
    panels[-1]["components"]["descendant_recovery"] = dict(mean=0., se=0., n=4)
    panels[-1]["gate"] = decisions(panels[-1]["components"], panels[-1]["safeguards"])
    require(not replicated_gates(panels)["evolutionary_recovery"]["passed"], "one failed lineage blocks replication")
    require("jax" not in sys.modules and not any(name.startswith("seed.") for name in sys.modules),
            "synthetic self-check imports neither simulator nor random runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=6, simulator_imported=False,
        coverage=["four original animals and both rule assignments", "first item, death and exact fifty-turn bins",
                  "native count and provenance corruption rejection", "descendant recovery separate from interaction",
                  "inclusive effect floor and strict evidence/noninferiority boundaries", "all-three lineage replication"])


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
            limitations=["All recorded individual events, native totals, first-item outcomes, paired family errors and decisions are independently reconstructed; physical and neural trajectories are not retained or replayed.",
                "The source-verified native rollout supplies food/poison labels. The exact pre-switch prefix and the first changed contact labels at turn150 are checked in saved events.",
                "World/development/birth keys are regenerated with the pinned JAX PRNG only; movement sampling and developmental wiring are not rerun.",
                "Four animals share a world; resource competition, feedback access and within-life learning can all influence the endpoint comparison. Two endpoints cannot establish monotonic evolution."])
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
