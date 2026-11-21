"""Read-only final E116 audit. No Seed/JAX import or simulation.

Reuse the independent E115 event/statistics auditor, with explicit E116 training
and v6 expectations. Refuse incomplete bundles before reading scientific arrays.
Require the coordinator's registered clean source commit; never infer it from
the current checkout. JSON is emitted only to stdout. Exit 2 means incomplete,
exit 1 a mismatch, exit 0 a passed artifact audit (not necessarily a passed gate).
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "tools" / "audit_e115.py"
SPEC = importlib.util.spec_from_file_location("independent_e115_checks", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
AuditError, Incomplete = base.AuditError, base.Incomplete
require, compare, subset = base.require, base.compare, base.subset
stats, ratio, distribution = base.stats, base.ratio, base.distribution
CFG, ORIGINALS = base.CFG, base.ORIGINALS
REFERENCES, CONDITIONS, EVENTS, PAIRING, METRICS = (
    base.REFERENCES, base.CONDITIONS, base.EVENTS, base.PAIRING, base.METRICS)
ARMS = ("control", "candidate")
GENOMES = tuple(f"{arm}{i}" for i in range(3) for arm in ("start", *ARMS))
PANELS = GENOMES + REFERENCES
TRAINING_PROTOCOL = "paired-train-v2"
EVAL_SEED = 2026112001


def comparisons(vectors):
    """Independent reconstruction of the 30 registered within-world contrasts."""
    contrasts, per_seed = [], []
    for i in range(3):
        parts = {}
        for control in ("control", "start"):
            comparison = "candidate-minus-" + control
            for metric in METRICS:
                delta = vectors[f"candidate{i}"][metric] - vectors[f"{control}{i}"][metric]
                estimate = stats(delta)
                contrasts.append(dict(seed=i, comparison=comparison, metric=metric,
                    minuend=f"candidate{i}", subtrahend=f"{control}{i}", **estimate,
                    n_worlds=len(delta), per_world_difference=delta.tolist()))
                if metric == "individual_min_return":
                    limit = 2 * estimate["se"]
                    parts[comparison + "_minimum_return"] = dict(
                        **estimate, threshold=limit, operator=">", passed=estimate["mean"] > limit)
                if metric == "joint_survival" and control == "start":
                    limit = -2 * estimate["se"]
                    parts["candidate-minus-start_joint_survival"] = dict(
                        **estimate, threshold=limit, operator=">=", passed=estimate["mean"] >= limit)
        per_seed.append(dict(seed=i, **parts, passed=all(x["passed"] for x in parts.values())))
    improvement = all(row[key]["passed"] for row in per_seed for key in (
        "candidate-minus-control_minimum_return", "candidate-minus-start_minimum_return"))
    survival = all(row["candidate-minus-start_joint_survival"]["passed"] for row in per_seed)
    return contrasts, dict(passed=improvement and survival, all_minimum_improvements=improvement,
                          no_joint_survival_regression=survival, per_seed=per_seed)


class Auditor(base.Auditor):
    def canonical(self, name):
        folders = list(self.path("runs").glob("*_" + name))
        require(len(folders) <= 1, "duplicate registered attempts: " + name)
        if not folders:
            raise Incomplete("registered folder does not exist: " + name)
        return folders[0].relative_to(self.root).as_posix()

    def completed_bundle(self, requested):
        """Inspect completion metadata only before any E116 scientific checks."""
        folder = self.canonical("E116-v6")
        require(self.path(requested) == self.path(folder), "evaluation is not the unique registered E116-v6 folder")
        summary = self.json(folder + "/summary.json")
        if (summary.get("status") != "finished" or summary.get("completed_panels") != 14
                or len(summary.get("panels", {})) != 14
                or any(panel.get("status") != "finished" for panel in summary.get("panels", {}).values())):
            raise Incomplete("the complete 14-panel final evaluation is required; no partial audit")
        for i in range(3):
            for arm in ARMS:
                name = f"E116-{arm}-s{i}"
                run = self.canonical(name)
                final = self.json(run + "/summary.json")
                if (final.get("status") != "finished" or final.get("generation") != 20
                        or final.get("training_complete") is not True):
                    raise Incomplete("all six finished generation-20 runs are required: " + name)
        return folder

    def training(self, i, arm, original):
        name = f"E116-{arm}-s{i}"
        folder = self.canonical(name)
        config = self.json(folder + "/config.json")
        expected_args = dict(name=name, eval_protocol=TRAINING_PROTOCOL, objective="paired-min",
            condition_split="final", controller="rules-v0", seed=2000+i, generations=20, pop=64,
            worlds=4, sigma=.1, lr=.1 if arm == "control" else .02, checkpoint_every=10,
            eval_worlds=256, no_plastic=False, quick=False, allow_dirty=False,
            structural_plasticity="off", start="file:" + ORIGINALS[i][0] + "/genome_final.npy",
            max_train_seconds=120 if i == 0 and arm == "control" else 900, resume=None)
        subset(config["args"], expected_args, name + " arguments")
        subset(config, dict(cfg=CFG, cfg_overrides={}, controller="rules-v0", genome_size=155,
            eval_protocol=TRAINING_PROTOCOL, objective="paired-min", evaluation_status="deferred",
            start_genome_hash=original["metadata"]["genome_hash"],
            input_genome=original["input_metadata"]), name + " config")
        provenance = self.provenance(config, pinned=True)
        for resume in config.get("resumes", []):
            require(resume.get("max_train_seconds") == 900, name + " registered resume bound")
        require(self.read(folder + "/genome_start.npy") == original["data"], name + " starting bytes")
        optimizer = config["initial_optimizer"]
        subset(optimizer, dict(method="es_init", generation=0, adam_m_zero=True, adam_v_zero=True, l2=.005), name + " optimizer")
        require(len(optimizer["key"]) == 2 and all(type(k) is int and 0 <= k < 2**32 for k in optimizer["key"]), name + " initial RNG")
        summary = self.json(folder + "/summary.json")
        data, flat, genome_hash = self.genome(folder + "/genome_final.npy")
        digest = hashlib.sha256(data).hexdigest()
        subset(summary, dict(status="finished", generation=20, training_complete=True, evaluation_status="deferred",
            controller="rules-v0", condition_split="final", eval_protocol=TRAINING_PROTOCOL, objective="paired-min",
            eval_worlds=256, final_genome_hash=genome_hash, final_genome_sha256=digest), name + " summary")
        require(not any(field in summary for field in ("evaluations", "final_learning", "final_frozen", "start_learning"))
                and not self.path(folder + "/events.npz").exists(), name + " unexpected held-out evaluation")
        checkpoint = self.arrays(folder + "/checkpoint.npz")
        require(set(checkpoint) == {"mu", "m", "v", "t", "key"}, name + " checkpoint fields")
        for field in ("mu", "m", "v"):
            value = checkpoint[field]
            require(value.shape == (155,) and np.issubdtype(value.dtype, np.floating)
                    and np.isfinite(value).all(), name + " checkpoint " + field)
        require(np.all(checkpoint["v"] >= 0) and np.array_equal(checkpoint["mu"], flat), name + " final state")
        require(checkpoint["t"].shape == () and np.issubdtype(checkpoint["t"].dtype, np.integer)
                and int(checkpoint["t"]) == 20, name + " checkpoint generation")
        require(checkpoint["key"].shape == (2,) and checkpoint["key"].dtype == np.uint32, name + " checkpoint RNG")
        text = self.read(folder + "/metrics.jsonl").decode("utf-8")
        require(text.endswith("\n"), name + " incomplete history line")
        history = [json.loads(line) for line in text.splitlines()]
        require(len(history) == 20, name + " history length")
        for generation, row in enumerate(history):
            require(set(row) == {"gen", "fit_mean", "fit_best", "fit_std", "acc_first", "acc_second", "survived", "seconds"}
                    and row["gen"] == generation
                    and all(type(v) in (int, float) and math.isfinite(v) for v in row.values()), name + " history row")
            require(row["fit_std"] >= 0 and row["seconds"] >= 0 and row["fit_best"] >= row["fit_mean"]
                    and all(0 <= row[key] <= 1 for key in ("acc_first", "acc_second", "survived")), name + " history range")
        metadata = dict(label=arm+str(i), controller="rules-v0", seed=i, training_seed=2000+i, objective="paired-min",
            run=folder, sha256=digest, genome_hash=genome_hash, genome_size=155, input_genome=original["input_metadata"],
            config_sha256=self.hashes[folder + "/config.json"], summary_sha256=self.hashes[folder + "/summary.json"],
            source_commits=provenance)
        return dict(name=name, run=folder, status="finished", generation=20, metadata=metadata,
            initial_optimizer=optimizer, final_key=checkpoint["key"].tolist(), history_rows=20,
            first_training_row=history[0], final_training_row=history[-1], arguments=config["args"], data=data)

    def matched_training(self, sources):
        pairs = []
        for i in range(3):
            control, candidate = (sources[f"{arm}{i}"] for arm in ARMS)
            compare(control["initial_optimizer"], candidate["initial_optimizer"], f"seed {i} initial optimizer/RNG")
            compare(control["final_key"], candidate["final_key"], f"seed {i} generation-20 RNG position")
            excluded = {"name", "lr", "max_train_seconds"}
            a, b = ({k: v for k, v in row["arguments"].items() if k not in excluded} for row in (control, candidate))
            compare(a, b, f"seed {i} all shared training arguments")
            a, b = ({k: v for k, v in row["first_training_row"].items() if k != "seconds"} for row in (control, candidate))
            compare(a, b, f"seed {i} first evaluated population before learning-rate update")
            pairs.append(dict(seed=i, training_seed=2000+i, starting_genome_identical=True,
                initial_optimizer_identical=True, final_rng_identical=True, exposure_arguments_identical=True,
                first_population_metrics_identical=True, complete_contiguous_history_rows=20))
        return pairs

    def evaluate(self, folder, sources):
        config, summary = self.json(folder + "/config.json"), self.json(folder + "/summary.json")
        self.provenance(config, pinned=True)
        subset(config, dict(cfg=CFG, eval_protocol="v6", eval_seed=EVAL_SEED, eval_worlds=256,
            reference_policies=list(REFERENCES), internal_seconds_limit=1140), "v6 config")
        subset(config["args"], dict(name="E116-v6", eval_protocol="v6", condition_split="final", generations=0,
            eval_worlds=256, allow_dirty=False, quick=False, resume=None, no_plastic=False), "v6 arguments")
        subset(summary, dict(status="finished", generation=0, eval_protocol="v6", condition_split="final",
            eval_seed=EVAL_SEED, eval_worlds=256, expected_panels=14, completed_panels=14, artifact_validity="passed"), "v6 summary")
        require(tuple(summary["panels"]) == PANELS, "v6 panel order")
        require([record["label"] for record in config["input_genomes"]] == list(GENOMES), "v6 input manifest order")
        for record in config["input_genomes"]:
            label = record["label"]
            compare(record, dict(sources[label]["metadata"], copied_genome=f"inputs/{label}.npy"), f"v6 {label} manifest")
            require(self.read(folder + "/" + record["copied_genome"]) == sources[label]["data"], f"v6 {label} copied bytes")
        first = self.arrays(folder + "/start0/stationary_r0.npz")
        common = {key: first[key] for key in PAIRING}
        for key in ("wkeys", "dkeys"):
            require(common[key].shape == (256, 2) and common[key].dtype == np.uint32
                    and len(np.unique(common[key], axis=0)) == 256, "v6 key shape/type/unique worlds: " + key)
        require(common["initial_grid"].shape == (256, 16, 16) and np.isin(common["initial_grid"], (0, 1, 2)).all()
                and all(np.all((common["initial_grid"] == kind).sum((1, 2)) == 24) for kind in (1, 2)), "v6 initial grids")
        require(common["initial_pos"].shape == (256, 4, 2) and np.issubdtype(common["initial_pos"].dtype, np.integer)
                and np.all((common["initial_pos"] >= 0) & (common["initial_pos"] < 16)), "v6 initial positions")
        require(np.array_equal(common["agent_ids"], np.broadcast_to(np.arange(4), (256, 4))), "v6 individual identities")
        vectors, panels, fitness_vectors = {}, {}, {}
        for label in PANELS:
            entry = summary["panels"][label]
            controller = "rules-v0" if label in GENOMES else label
            subset(entry, dict(status="finished", controller=controller), label + " status/controller")
            require(tuple(entry["conditions"]) == CONDITIONS, label + " conditions")
            compare(entry["raw_files"], {key: f"{label}/{key}.npz" for key in (*CONDITIONS, "paired")}, label + " raw paths")
            data, condition_report, fitness_vectors[label] = {}, {}, {}
            for condition in CONDITIONS:
                arrays = self.arrays(folder + "/" + entry["raw_files"][condition])
                data[condition], condition_report[condition] = self.condition(
                    arrays, condition, controller, entry["conditions"][condition], common)
                fitness_vectors[label][condition] = data[condition]["returns"].mean(1)
            for rule in (0, 1):
                for event in EVENTS:
                    require(np.array_equal(data[f"stationary_r{rule}"]["events"][event][:, :150],
                        data[f"reversal_r{rule}"]["events"][event][:, :150]), label + " preflip identity")
            r0, r1 = data["stationary_r0"], data["stationary_r1"]
            q0, q1 = data["reversal_r0"], data["reversal_r1"]
            minimum = np.minimum(r0["returns"], r1["returns"])
            joint = r0["final_alive"] & r1["final_alive"]
            post = [q["food"][:, 150:].sum(1, dtype=np.int64)-q["poison"][:, 150:].sum(1, dtype=np.int64) for q in (q0, q1)]
            values = dict(stationary_fitness=((r0["returns"]+r1["returns"])/2).mean(1),
                individual_min_return=minimum.mean(1), joint_survival=joint.mean(1),
                post_flip_return=((post[0]+post[1])/2).mean(1),
                reversal_final_survival=((q0["final_alive"].astype(float)+q1["final_alive"])/2).mean(1),
                reversal_fitness=((q0["returns"]+q1["returns"])/2).mean(1))
            expected = {"paired_" + key: value for key, value in values.items()}
            expected.update(paired_individual_minimum=minimum.astype(float), paired_individual_joint_survival=joint)
            saved = self.arrays(folder + "/" + entry["raw_files"]["paired"])
            require(set(saved) == set(expected), label + " paired fields")
            for key, value in expected.items():
                require(np.array_equal(saved[key], value), label + " paired array " + key)
            estimates = {key: stats(value) for key, value in values.items()}
            estimates["individual_min_return"]["distribution"] = distribution(minimum.astype(float))
            compare(entry["paired"], estimates, label + " paired summary")
            vectors[label] = values
            panels[label] = dict(paired=estimates, conditions=condition_report)
        require(len(list(self.path(folder).glob("*/*.npz"))) == 70, "v6 archive count")
        contrasts, gate = comparisons(vectors)
        require(len(contrasts) == 30, "v6 registered contrast count")
        compare(summary["contrasts"], contrasts, "v6 all 30 contrasts")
        compare(summary["gate"], gate, "v6 numerical gate")
        for label in PANELS:
            for condition in CONDITIONS:
                numerator, denominator = fitness_vectors[label][condition], fitness_vectors["greedy"][condition]
                estimate = ratio(numerator, denominator)
                if estimate["mean"] is not None:
                    # An independent covariance expression checks the paired delta-method SE.
                    mx, my = numerator.mean(), denominator.mean()
                    covariance = np.cov(numerator, denominator, ddof=1)
                    variance = (covariance[0, 0]/my**2 + mx**2*covariance[1, 1]/my**4
                                - 2*mx*covariance[0, 1]/my**3) / 256
                    require(math.isclose(estimate["se"], math.sqrt(max(0., variance)), rel_tol=1e-9, abs_tol=1e-9),
                            label + "/" + condition + " greedy ratio SE")
                    if label == "greedy":
                        compare(estimate, dict(mean=1., se=0.), condition + " greedy self ratio")
                panels[label]["conditions"][condition]["fitness_percent_of_greedy"] = {
                    key: None if value is None else 100*value for key, value in estimate.items()}
        return dict(run=folder, archive_count=70, panels=panels, contrasts=contrasts, gate=gate,
            eval_seconds=summary["eval_seconds"], first_panel_seconds=summary["first_panel_seconds"],
            pairing_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in common.items()})


def self_check():
    """Tiny arithmetic and boundary checks only; no run data or engine import."""
    compare(stats([0, 2]), dict(mean=1., se=1.), "hand-calculated world SE")
    compare(ratio([1, 3], [2, 2]), dict(mean=1., se=.5), "hand-calculated ratio SE")
    boundary = {label: {metric: np.zeros(2) for metric in METRICS} for label in GENOMES}
    for i in range(3):
        boundary[f"candidate{i}"]["individual_min_return"][:] = [1., 3.]
    _, boundary_gate = comparisons(boundary)
    item = boundary_gate["per_seed"][0]["candidate-minus-control_minimum_return"]
    compare(item, dict(mean=2., se=1., threshold=2., operator=">", passed=False),
            "exact nonzero two-SE boundary must fail")
    for i in range(3):
        boundary[f"candidate{i}"]["individual_min_return"] += .001
    _, boundary_gate = comparisons(boundary)
    require(boundary_gate["passed"], "strictly above the nonzero two-SE boundary must pass")
    vectors = {label: {metric: np.zeros(256) for metric in METRICS} for label in GENOMES}
    rows, gate = comparisons(vectors)
    require(len(rows) == 30 and not gate["passed"] and not gate["all_minimum_improvements"]
            and gate["no_joint_survival_regression"], "equal-return fails, equal-survival passes")
    for i in range(3):
        vectors[f"candidate{i}"]["individual_min_return"][:] = 1
    _, gate = comparisons(vectors)
    require(gate["passed"], "positive zero-variance improvements pass")
    vectors["candidate2"]["joint_survival"][:] = -.1
    _, gate = comparisons(vectors)
    require(not gate["passed"] and gate["all_minimum_improvements"]
            and not gate["no_joint_survival_regression"], "one survival failure blocks the gate")
    require(not any(name == "seed" or name.startswith(("seed.", "jax", "scripts.eval")) for name in sys.modules),
            "audit must not import simulation/evaluation code")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", help="required full registered clean E116 source SHA")
    parser.add_argument("--evaluation", help="required unique completed E116-v6 folder")
    parser.add_argument("--self-check", action="store_true", help="synthetic arithmetic only; no scientific input")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print(json.dumps(dict(status="self-check-passed", scientific_inputs_opened=False)))
        return 0
    if (not args.source_commit or len(args.source_commit) != 40
            or any(ch not in "0123456789abcdef" for ch in args.source_commit) or not args.evaluation):
        parser.error("provide --source-commit with the full registered SHA and --evaluation with the complete final folder")
    report = dict(status="not-audited", mode="full", protocol="v6",
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_helper_sha256=hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(), training=[])
    audit = None
    try:
        audit = Auditor(args.root, args.source_commit)
        report.update(registered_source=audit.source, registered_code_trees=audit.expected_trees)
        evaluation = audit.completed_bundle(args.evaluation)
        sources = {}
        for i in range(3):
            original = audit.original(i)
            sources[f"start{i}"] = original
            for arm in ARMS:
                result = audit.training(i, arm, original)
                sources[f"{arm}{i}"] = result
                report["training"].append({key: value for key, value in result.items() if key != "data"})
        report["matched_training"] = audit.matched_training(sources)
        report["evaluation"] = audit.evaluate(evaluation, sources)
        report.update(status="passed", final_evaluation_audited=True, limitations=[
            "Read-only reconstruction from saved artifacts; not an optimizer or physics replay.",
            "Equal initial/final RNG positions and fixed exposure arguments are checked; streams are not regenerated from seeds.",
            "Zero initial Adam moments are recorded metadata; final states and continuous histories are checked independently.",
            "Operational wall-time, budget and resource-guard evidence are reviewed separately by the coordinator.",
            "World-family SEs describe fixed genomes, not variability across evolution seeds.",
            "The registered numerical gate establishes neither causal learning nor reversal competence."])
        code = 0
    except Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if audit is not None:
        report["checked_file_sha256"] = audit.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
