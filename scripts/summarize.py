#!/usr/bin/env python3
"""Rebuild results/RESULTS.md (one row per run) from runs/*/config.json and summary.json.

Only the generated table is overwritten; nothing in runs/ is changed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    diagnostics = []
    v3_rows = []
    v4_rows = []
    v5_rows = []
    v5_gates = []
    v6_rows = []
    v6_gates = []
    paired_training_rows = []
    ecology_fixture_rows = []
    cold_donor_rows = []
    shelter_evolution_rows = []
    paired_v2_training_rows = []
    direct_rows = []
    reversals = []
    structural_rows = []
    broader_rows = []
    n_runs = 0
    for d in sorted((ROOT / "runs").glob("*/")):
        cfg_p, sum_p = d / "config.json", d / "summary.json"
        if not cfg_p.exists():
            continue
        n_runs += 1
        cfg = json.loads(cfg_p.read_text())
        s = json.loads(sum_p.read_text()) if sum_p.exists() else {"status": "no summary"}
        a = cfg["args"]
        commit = (cfg.get("git", {}).get("commit") or "")[:7] + ("*" if cfg.get("git", {}).get("dirty") else "")
        if a.get("eval_protocol") == "shelter-evolution-v1":
            split = a.get("condition_split", "")
            expected = (10, 128) if split == "final" else (1, 4) if split == "development" else (None, None)
            complete = (s.get("status") == "finished" and s.get("artifact_validity") == "passed"
                        and s.get("generation") == s.get("expected_generations") == expected[0]
                        and s.get("completed_families") == s.get("expected_families") == expected[1])
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            gate = s.get("gate", {})
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and type(gate.get("passed")) is bool)
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            shelter_evolution_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('generation', '?')}/{cfg.get('expected_generations', '?')}",
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                str(s.get("changed_parameters", "undefined")),
                estimate(measurements.get("parent", {}).get("latency", {})),
                estimate(measurements.get("descendant", {}).get("latency", {})),
                estimate(measurements.get("latency_improvement", {})),
                estimate(measurements.get("survival_difference", {})), outcome, commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") == "cold-donor-v1":
            split = a.get("condition_split", "")
            expected = 256 if split == "final" else 4 if split == "development" else None
            complete = (expected is not None and s.get("status") == "finished"
                        and s.get("artifact_validity") == "passed"
                        and s.get("completed_families") == s.get("expected_families") == expected)
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            gate = s.get("gate", {})
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and type(gate.get("passed")) is bool)
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            cold_donor_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                cfg.get("source", {}).get("label", ""), str(a.get("seed", "")),
                estimate(measurements.get("inherited", {}).get("latency", {})),
                estimate(measurements.get("fresh", {}).get("latency", {})),
                estimate(measurements.get("latency_improvement", {})),
                estimate(measurements.get("inherited", {}).get("survival", {})),
                estimate(measurements.get("fresh", {}).get("survival", {})),
                estimate(measurements.get("survival_difference", {})), outcome, commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") == "ecology-lineage-v1":
            expected = s.get("expected_cohorts", len(cfg.get("schedule", [])) or "?")
            ecology_fixture_rows.append("| " + " | ".join([
                d.name, s.get("status", ""), f"{s.get('completed_cohorts', '?')}/{expected}",
                s.get("condition_split", cfg.get("condition_split", a.get("condition_split", ""))), commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") in ("paired-train-v1", "paired-train-v2"):
            learning_rate_study = a["eval_protocol"] == "paired-train-v2"
            target = paired_v2_training_rows if learning_rate_study else paired_training_rows
            target.append("| " + " | ".join([
                d.name, s.get("status", ""), a.get("objective", ""), a.get("condition_split", "final"),
                str(a.get("seed", "")), f"{s.get('generation', '')}/{a.get('generations', '')}",
                str(a.get("pop", "")), str(a.get("worlds", "")),
                *([str(a.get("sigma", "")), str(a.get("lr", ""))] if learning_rate_study else []),
                s.get("evaluation_status", cfg.get("evaluation_status", "deferred")), commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") in ("v4", "v5", "v6"):
            if a["eval_protocol"] in ("v5", "v6"):
                decided = s.get("status") == "finished" and s.get("artifact_validity") == "passed" and "gate" in s
                if a["eval_protocol"] == "v6":
                    panels = s.get("panels", {})
                    decided = (decided and s.get("completed_panels") == s.get("expected_panels") == 14
                               and len(panels) == 14 and all(panel.get("status") == "finished"
                                   and len(panel.get("conditions", {})) == 4 for panel in panels.values()))
                outcome = ("PASS" if s["gate"]["passed"] else "FAIL") if decided else "undecided"
                target = v5_gates if a["eval_protocol"] == "v5" else v6_gates
                target.append(f"| {d.name} | {a.get('condition_split', 'final')} | {s.get('status', '')} | {outcome} | {commit} |")
            for label, panel in s.get("panels", {}).items() or [("pending", {})]:
                paired = panel.get("paired", {})
                estimates = []
                for key in ("stationary_fitness", "individual_min_return", "joint_survival",
                            "post_flip_return", "reversal_final_survival"):
                    metric = paired.get(key, {})
                    estimates.append("undefined" if metric.get("mean") is None or metric.get("se") is None
                                     else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
                target = {"v4": v4_rows, "v5": v5_rows, "v6": v6_rows}[a["eval_protocol"]]
                target.append("| " + " | ".join([
                    d.name, s.get("status", ""), label, panel.get("controller", ""), panel.get("status", "pending"),
                    f"{len(panel.get('conditions', {}))}/4", a.get("condition_split", "final"), *estimates,
                    str(s.get("eval_seed", cfg.get("eval_seed", ""))),
                    str(s.get("eval_worlds", cfg.get("eval_worlds", a.get("eval_worlds", "")))), commit,
                ]) + " |")
            continue
        if a.get("eval_protocol") == "direct-v1":
            number = lambda value: "undefined" if value is None else f"{value:.6f}"
            for condition, controllers in s.get("evaluations", {"pending": {"genome": {}}}).items():
                metric = controllers.get("genome", {})
                direct_rows.append("| " + " | ".join([
                    d.name, s.get("status", ""), cfg.get("controller", a.get("controller", "")),
                    a.get("condition_split", "final"), f"{s.get('generation', '')}/{a['generations']}",
                    str(a["seed"]), condition, number(metric.get("fitness")), number(metric.get("fitness_se")),
                    number(metric.get("greedy_fraction")), number(metric.get("greedy_fraction_se")),
                    str(s.get("eval_seed", cfg.get("eval_seed", ""))), str(a["eval_worlds"]), commit,
                ]) + " |")
            continue
        if a.get("eval_protocol") == "v3":
            number = lambda value: "undefined" if value is None else f"{value:.6f}"
            evaluations = s.get("evaluations", {"pending": {"pending": {}}})
            for condition, controllers in evaluations.items():
                for controller, metric in controllers.items():
                    v3_rows.append("| " + " | ".join([
                        d.name, s.get("status", ""), condition, controller,
                        number(metric.get("fitness")), number(metric.get("fitness_se")),
                        number(metric.get("greedy_fraction")), number(metric.get("greedy_fraction_se")),
                        str(s.get("eval_seed", cfg.get("eval_seed", ""))),
                        str(s.get("eval_worlds", cfg.get("eval_worlds", a.get("eval_worlds", "")))), commit,
                    ]) + " |")
            continue
        if a.get("eval_protocol") == "reversal-v1":
            for label, row in s.get("evaluations", {"pending": {}}).items():
                metric = row.get("metrics", {})
                reversals.append("| " + " | ".join([
                    d.name, s.get("status", ""), a["condition_split"], str(cfg["eval_seed"]),
                    str(a["eval_worlds"]), label, str(metric.get("post_fitness", "")),
                    str(metric.get("recovered_agents", "")), str(metric.get("agents_at_risk", "")),
                    commit,
                ]) + " |")
            continue
        if a.get("eval_protocol", "v1") == "v2":
            ev = s.get("evaluations", {})
            b, h, f = (ev.get(k, {}).get("fitness") for k in ("both", "hebbian_only", "frozen"))
            differences = [b - h, h - f] if None not in (b, h, f) else [None, None]
            ratio = (h - f) / (b - f) if None not in (b, h, f) and b > f else None
            number = lambda value: "" if value is None else f"{value:.6f}"
            diagnostics.append("| " + " | ".join([
                d.name, s.get("status", ""), a["start"],
                *map(number, (b, h, f, *differences)), "undefined" if ratio is None else number(ratio),
                str(s.get("eval_seed", cfg.get("eval_seed", ""))),
                str(s.get("eval_worlds", cfg.get("eval_worlds", ""))),
                commit,
            ]) + " |")
            continue
        if a.get("eval_protocol", "v1") == "structural-v1":
            if not s.get("conditions"):
                structural_rows.append("| " + " | ".join([d.name, s.get("status", ""), *[""] * 11, commit]) + " |")
            for condition, result in s.get("conditions", {}).items():
                stats = result.get("structure", {})
                structural_rows.append("| " + " | ".join(map(str, [
                    d.name, s.get("status", ""), condition, result["fitness"], result["acc_first"],
                    result["acc_second"], result["survived"], result["eaten"],
                    stats.get("added", ""), stats.get("removed", ""), stats.get("initial_edges", ""),
                    stats.get("final_edges", ""), stats.get("edge_cap", ""), commit,
                ])) + " |")
            continue
        if a.get("eval_protocol", "v1") == "broader-v1":
            protocol = cfg["eval_protocol"]
            for label, result in s.get("condition_evaluations", {"pending": {}}).items():
                learning, frozen = result.get("learning", {}), result.get("frozen", {})
                metric = lambda r, k: f"{r[k]:.4f}" if k in r else ""
                broader_rows.append("| " + " | ".join([
                    d.name, s.get("status", ""), protocol["split"], str(protocol["seed"]),
                    str(protocol["n_worlds"]), label, metric(learning, "fitness"), metric(frozen, "fitness"),
                    metric(result, "learning_minus_frozen_fitness"), commit,
                ]) + " |")
            continue
        fl, ff, sl = s.get("final_learning"), s.get("final_frozen"), s.get("start_learning")
        f = lambda r, k, p=2: f"{r[k]:.{p}f}" if r else ""
        acc = lambda r: f"{r['acc_first']:.2f} → {r['acc_second']:.2f}" if r else ""
        rows.append("| " + " | ".join([
            d.name, s.get("status", ""), a["start"], "no" if a["no_plastic"] else "yes",
            f"{s.get('generation', '')}/{a['generations']}", f"{a['pop']}×{a['worlds']}",
            f"{a['sigma']}/{a['lr']}", ", ".join(f"{k}={v}" for k, v in cfg.get("cfg_overrides", {}).items()),
            f(sl, "fitness"), f(fl, "fitness"), f(ff, "fitness"), acc(fl),
            commit,
        ]) + " |")
    head = ("| Run | Status | Start | Plastic | Gens | Pop×worlds | σ/lr | Cfg overrides | Start fitness | "
            "Final learning fitness | Final frozen | Learning acc 1st→2nd | Commit |\n" + "|---" * 13 + "|")
    out = ["# Results (generated by scripts/summarize.py, do not edit by hand)", "",
           "Fitness = food minus poison per agent in unseen evaluation worlds (seed 123). "
           "Frozen = same genome with plasticity disabled (lesion test). "
           "For C (`--no-plastic`), `final_learning` re-enables updates and is diagnostic; "
           "`final_frozen` is the control score. * = run from uncommitted code.", "",
           head, *rows, ""]
    if ecology_fixture_rows:
        out += ["## Ecology/lineage development fixtures", "",
                "Engineering exercises of a fixed cohort schedule and saved memory. Completion records mechanism "
                "execution, not competence, a scientific gate, or an EVAL v1 score. Partial attempts retain their "
                "completed cohort evidence; paired observations remain in each run folder.", "",
                "| Run | Status | Cohorts completed/expected | Split | Commit |", "|---" * 5 + "|",
                *ecology_fixture_rows, ""]
    if shelter_evolution_rows:
        out += ["## Shelter evolution (shelter-evolution-v1)", "",
                "E118 evolves each original A genome for ten generations and compares its final descendant "
                "with its parent on 128 paired unseen worlds. Both use the same shelter ecology and persistence "
                "trait. Births outside shelter contribute to the capped first living arrival time. Death and "
                "nonarrival count as 301 turns. The reported reduction is parent minus descendant, with a "
                "family-cluster standard error. The survival difference is descendant minus parent among all "
                "assigned creatures. Each seed passes if latency improves by more than two paired standard "
                "errors and survival falls by at most two paired standard errors. All three seeds must pass. "
                "The development benchmark and incomplete runs have no scientific decision. Genome changes "
                "describe mutation and selection; the paired test measures their outcome.", "",
                "| Run | Status | Split | Generations | Families | Changed parameters | Parent latency ± SE | "
                "Descendant latency ± SE | Latency reduction ± SE | Survival difference ± SE | Seed decision | Commit |",
                "|---" * 12 + "|", *shelter_evolution_rows, ""]
    if cold_donor_rows:
        out += ["## Cold donor inheritance (cold-donor-v1)", "",
                "E117 uses fixed original A genomes in the same full ecology for both splits. Each independent "
                "family has one cold donor life and paired newborn recipient arms, with persistence off. "
                "Restricted first-shelter-arrival latency excludes agents born inside shelter; nonarrival, "
                "including death before a living arrival, counts as 301 steps. Latency reduction is fresh minus "
                "inherited, and survival difference is inherited minus fresh. Estimates are means ± paired "
                "family SE; pooled outside-born latency uses a family-cluster ratio SE. Each seed requires "
                "latency improvement strictly above two SE and survival difference at least minus two SE. "
                "All three final seed decisions must pass for the study gate. The four-family development "
                "benchmark has no gate; incomplete attempts have no pooled estimates or decision. Other "
                "registered metrics and compressed family evidence remain in each run folder.", "",
                "| Run | Status | Split | Families completed/expected | Source | Evaluation seed | Inherited latency ± SE | "
                "Fresh latency ± SE | Latency reduction ± SE | Inherited survival ± SE | Fresh survival ± SE | "
                "Survival difference ± SE | Seed decision | Commit |", "|---" * 14 + "|", *cold_donor_rows, ""]
    if paired_training_rows:
        out += ["## Paired objective continuation (paired-train-v1)", "",
                "Both arms continue the same saved plastic A genomes with fresh zero Adam moments and matched "
                "training keys. Paired mean averages both rule returns; paired minimum takes each individual's "
                "minimum before averaging. Training completion saves the genome and defers evaluation to the "
                "separate EVAL v5 aggregate. Training population variation is not held-out uncertainty; no "
                "scientific fitness estimate or SE is reported here.", "",
                "| Run | Status | Objective | Split | Training seed | Gens | Population | Base-world pairs | Evaluation | Commit |",
                "|---" * 10 + "|", *paired_training_rows, ""]
    if paired_v2_training_rows:
        out += ["## Paired-min learning-rate continuation (paired-train-v2)", "",
                "E116 compares control learning rate 0.1 with candidate 0.02, both using paired individual-minimum "
                "selection from the same original A genomes. Fresh optimizer states and matched seeds 2000/2001/2002 "
                "give both arms the same 20-generation exposure. The run name identifies the arm; sigma and learning "
                "rate are shown explicitly. Evaluation remains deferred until all six final genomes are valid. "
                "Training variation is not held-out uncertainty or a scientific improvement estimate.", "",
                "| Run | Status | Objective | Split | Training seed | Gens | Population | Base-world pairs | Sigma | Learning rate | Evaluation | Commit |",
                "|---" * 12 + "|", *paired_v2_training_rows, ""]
    if v6_rows:
        out += ["## EVAL v6 paired learning-rate comparison", "",
                "Original starts, six fixed E116 final genomes and five references share fresh v6 worlds. Values "
                "are independent base-world means ± SE, never pooled across training seeds. The gate requires "
                "candidate individual-minimum return to exceed both control and unchanged start by strictly more "
                "than two paired SE in every seed, with candidate-minus-start joint survival at least minus two "
                "paired SE. Equality fails return checks and is allowed at the survival boundary; this does not "
                "prove equivalent survival. Incomplete or invalid attempts remain undecided. All 30 registered "
                "contrasts and per-rule survival/encounter data remain in the aggregate summary and raw files. "
                "Reversal outcomes cannot substitute for the gate, and E115's failed gate is unchanged.", "",
                "| Run | Split | Status | E116 gate | Commit |", "|---" * 5 + "|", *v6_gates, "",
                "| Run | Run status | Panel | Controller | Panel status | Conditions | Split | Stationary fitness ± SE | "
                "Individual minimum return ± SE | Joint survival ± SE | Post-flip return ± SE | Reversal survival ± SE | "
                "Eval seed | Base worlds | Commit |", "|---" * 15 + "|", *v6_rows, ""]
    if v5_rows:
        out += ["## EVAL v5 paired objective comparison", "",
                "Original A starts and all six registered paired-mean/paired-min continuations are evaluated "
                "together only after training completes. Every value is a base-world mean ± SE on fresh v5 "
                "worlds; development uses seed 41 only. The fixed gate requires minimum-return improvements "
                "over both start and paired mean beyond two paired SE in all three seeds, with no clear "
                "joint-survival regression against start. The complete gate decision, 30 contrasts, "
                "per-rule results and raw observations remain in the aggregate summary and panel files. "
                "E113's failed gate remains unchanged.", "",
                "| Run | Split | Status | E115 gate | Commit |", "|---" * 5 + "|", *v5_gates, "",
                "| Run | Run status | Panel | Controller | Panel status | Conditions | Split | Stationary fitness ± SE | "
                "Individual minimum return ± SE | Joint survival ± SE | Post-flip return ± SE | Reversal survival ± SE | "
                "Eval seed | Base worlds | Commit |", "|---" * 15 + "|", *v5_rows, ""]
    if v4_rows:
        out += ["## EVAL v4 paired-world diagnostics", "",
                "Saved controllers and fixed reference policies; no training. Each base world is evaluated under "
                "both initial food rules and their mid-life reversals. Means and SEs use independent base-world "
                "families; minimum return and joint survival pair each original individual across the two stationary rules "
                "before averaging within its family. Survival values are fractions. Development rows use tiny seed-41 "
                "fixtures only. These descriptive measurements do not replace the failed E113 gate. Per-rule accuracy, "
                "deaths, encounters, all 15 planned A-minus-direct contrasts and raw events remain in the run folder. "
                "Incomplete attempts retain their completed conditions and show undefined paired estimates.", "",
                "| Run | Run status | Panel | Controller | Panel status | Conditions | Split | Stationary fitness ± SE | "
                "Individual minimum return ± SE | Joint survival ± SE | Post-flip return ± SE | Reversal survival ± SE | "
                "Eval seed | Base worlds | Commit |", "|---" * 15 + "|", *v4_rows, ""]
    if v3_rows:
        out += ["## EVAL v3 diagnostics", "",
                "Saved genomes only; 256 fresh worlds per panel. SE treats each world-family as one cluster. "
                "Greedy fractions use a paired world-level delta-method SE. Greedy knows the current rule "
                "but sees only the local patch; it is not an optimal-policy bound. Repeated policy rows "
                "use the same worlds and are not independent replications. Meal counts, survivor-conditioned "
                "bins and raw events are preserved in each run's summary.json and events.npz.", "",
                "| Run | Status | Condition | Controller | Fitness | Fitness SE | Fraction of greedy | Fraction SE | Seed | Worlds | Commit |",
                "|---" * 11 + "|", *v3_rows, ""]
    if direct_rows:
        out += ["## Direct recurrent baseline (direct-v1)", "",
                "The direct-rnn-v1 genome encodes fixed recurrent weights; hidden states still change during life. "
                "No lifetime synaptic, predictor or structural updates occur. Final rows use the same 256 EVAL v3 "
                "worlds and references; development rows use tiny fixtures and seed 41, never held-out worlds. "
                "Each uncertainty is a world-family SE for that fixed genome, not variability across training seeds. "
                "Raw events, full diagnostics and initial/final weight magnitudes remain in each run folder.", "",
                "| Run | Status | Controller | Split | Gens | Training seed | Condition | Fitness | Fitness SE | Fraction of greedy | Fraction SE | Eval seed | Worlds | Commit |",
                "|---" * 14 + "|", *direct_rows, ""]
    if diagnostics:
        out += ["## EVAL v2 diagnostics", "",
                "Saved genomes; no evolution. Both enables Hebbian and predictor updates; Hebbian-only disables "
                "predictor updates; Frozen disables both. Differences are signed fitness changes. "
                "Retained benefit = (Hebbian-only − Frozen) / (Both − Frozen), defined only when Both > Frozen.", "",
                "| Run | Status | Source genome | Both | Hebbian-only | Frozen | Both − Hebbian-only | "
                "Hebbian-only − Frozen | Retained benefit | Eval seed | Eval worlds | Commit |",
                "|---" * 12 + "|", *diagnostics, ""]
    if structural_rows:
        out += ["## Structural v1 (separate diagnostic)", "",
                "Same worlds/seed 123; Weight-only retains both local weight-learning rules. "
                "Frozen disables weight and structural updates. Edge counts are means per agent; "
                "turnover is capability evidence, not evidence of a fitness benefit. No EVAL v1 rows changed.", "",
                "| Run | Status | Condition | Fitness | Accuracy first | Accuracy second | Survival | Eaten | "
                "Added | Removed | Initial edges | Final edges | Edge cap | Commit |", "|---" * 14 + "|",
                *structural_rows, ""]
    if broader_rows:
        out += ["## BROADER v1 (separate parameter-shift protocol)", "",
                "Development rows are operational checks, not scientific generalization results. "
                "Split/seed/world count are explicit; these are not EVAL v1 seed-123 scores. "
                "Full accuracy, survival, encounter counts and condition shifts are in each summary.json.", "",
                "| Run | Status | Split | Eval seed | Worlds | Condition | Learning | Frozen | Gap | Commit |",
                "|---" * 10 + "|", *broader_rows, ""]
    if reversals:
        out += ["## REVERSAL v1 prototype", "",
                "Development rows are software checks, not adaptation results. Window hits use "
                "complete 25-step windows, >=5 encounters, >=70% food choices and survival through window end; "
                "all censored agents and full counts remain in summary.json.", "",
                "| Run | Status | Split | Seed | Worlds | Condition | Post fitness | Window hits | At risk | Commit |",
                "|---" * 10 + "|", *reversals, ""]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(out))
    print(f"wrote results/RESULTS.md with {n_runs} runs")


if __name__ == "__main__":
    main()
