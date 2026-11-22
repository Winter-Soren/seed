#!/usr/bin/env python3
"""Rebuild results/RESULTS.md (one row per run) from runs/*/config.json and summary.json.

Only the generated table is overwritten; nothing in runs/ is changed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    inherited_evolution_rows = []
    social_learning_rows = []
    random_reversal_gates = []
    random_reversal_windows = []
    random_reversal_survival = []
    random_reversal_pairs = []
    random_reversal_controls = []
    random_reversal_subgroups = []
    reversal_training_rows = []
    reversal_endpoint_rows = []
    reversal_window_rows = []
    encounter_status_rows = []
    retention_status_rows = []
    retention_intervention_status_rows = []
    founder_status_rows = []
    exploration_status_rows = []
    offer_status_rows = []
    contact_gate_status_rows = []
    replay_selection_rows = []
    scene_transfer_rows = []
    background_selection_rows = []
    social_observer_rows = []
    social_belief_gates = []
    social_belief_arms = []
    social_belief_contrasts = []
    social_belief_calibrations = []
    social_belief_losses = []
    social_transmission_gates = []
    social_transmission_arms = []
    social_transmission_contrasts = []
    social_channel_rows = []
    social_channel_contrasts = []
    social_channel_gates = []
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
        if a.get("eval_protocol") == "background-selection-v1":
            background_selection_rows.append("| " + " | ".join([d.name, cfg["mode"], cfg.get("view") or "—", s.get("status", "no summary"),
                f"{s.get('completed_attempts', '?')}/{cfg['expected_attempts']}",
                f"{s.get('completed_families', '?')}/{cfg['expected_families']}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "scene-transfer-v1":
            scene_transfer_rows.append("| " + " | ".join([d.name, cfg["mode"], s.get("status", "no summary"),
                f"{s.get('completed_families', '?')}/{cfg['expected_families']}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "replay-selection-v1":
            replay_selection_rows.append("| " + " | ".join([d.name, cfg["mode"], s.get("status", "no summary"),
                f"{s.get('completed_attempts', '?')}/{cfg['expected_attempts']}",
                f"{s.get('completed_families', '?')}/{cfg['expected_families']}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "contact-gate-v1":
            contact_gate_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "voluntary-offers-v1":
            offer_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "timed-exploration-v1":
            exploration_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "founder-reversal-v1":
            founder_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}", commit]) + " |")
            continue
        if a.get("eval_protocol") == "retention-intervention-v1":
            retention_intervention_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                commit]) + " |")
            continue
        if a.get("eval_protocol") == "retention-diagnostic-v1":
            retention_status_rows.append("| " + " | ".join([d.name, s.get("status", "no summary"),
                a.get("condition_split", ""), f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                commit]) + " |")
            continue
        if a.get("eval_protocol") == "encounter-diagnostic-v1":
            role = ("preserved partial; excluded" if a.get("name") == "E127-A0" else
                    "registered full-panel recovery" if a.get("name") == "E127-A0-recovery1" else
                    "development only" if a.get("condition_split") == "development" else "registered final panel")
            encounter_status_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), a.get("condition_split", ""),
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}", role, commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") == "reversal-evolution-v1":
            split, stage = a.get("condition_split", ""), cfg.get("stage", "")
            if stage == "training":
                reversal_training_rows.append("| " + " | ".join([
                    d.name, s.get("status", "no summary"), split, str(cfg.get("target", "?")),
                    str(cfg.get("controller", "?")), str(cfg.get("start_attempt", "?")),
                    f"{s.get('attempted_rounds', '?')}/{cfg.get('expected_attempts', '?')}",
                    str(s.get("optimizer_updates", "?")), "undecided", commit,
                ]) + " |")
            elif stage == "evaluation":
                complete = (s.get("status") == "finished" and s.get("artifact_validity") == "passed"
                            and s.get("completed_worlds") == s.get("expected_worlds") == 128
                            and len(s.get("raw_files", [])) == 128)
                measurement = s.get("measurements", {}) if complete else {}
                estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                           else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
                arms = measurement.get("arms", {})
                contrast = measurement.get("plastic_vs_direct", {})
                comparison = ("undefined" if contrast.get("difference") is None or contrast.get("combined_se") is None
                              else f"{contrast['difference']:.6f} ± {contrast['combined_se']:.6f}")
                reversal_endpoint_rows.append("| " + " | ".join([
                    d.name, s.get("status", "no summary"), str(cfg.get("seed_index", "?")),
                    f"{s.get('completed_worlds', '?')}/128",
                    estimate(arms.get("plastic_reversal", {}).get("late_min_return", {})),
                    estimate(arms.get("direct_reversal", {}).get("late_min_return", {})), comparison,
                    estimate(contrast.get("paired_difference", {})),
                    estimate(measurement.get("plastic_vs_lesion", {}).get("difference", {})),
                    estimate(arms.get("plastic_reversal", {}).get("late_first_food", {})),
                    estimate(arms.get("plastic_reversal", {}).get("joint_survival", {})),
                    estimate(measurement.get("survival", {}).get("difference", {})), commit,
                ]) + " |")
                for arm, metrics in arms.items():
                    for window, values in metrics.get("windows", {}).items():
                        reversal_window_rows.append("| " + " | ".join([
                            d.name, arm, window, estimate(values.get("first_food", {})),
                            estimate(values.get("no_meal", {})), estimate(values.get("fitness", {})),
                            estimate(metrics.get("whole_return", {})), estimate(metrics.get("joint_survival", {})),
                        ]) + " |")
            continue
        if a.get("eval_protocol") == "reversal-random-v1":
            split = a.get("condition_split", "")
            expected = 128 if split == "final" else 4 if split == "development" else None
            complete = (expected is not None and s.get("status") == "finished"
                        and s.get("artifact_validity") == "passed" and s.get("primary") == "observed"
                        and s.get("completed_families") == s.get("expected_families") == expected)
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None else
                                       f"{metric['mean']:.6f} ± " +
                                       ("undefined" if metric.get("se") is None else f"{metric['se']:.6f}"))
            gate = s.get("gate", {})
            gate_fields = ("passed", "late_level_passed", "late_chance_passed", "recovery_passed",
                           "accuracy_noninferiority_passed", "survival_noninferiority_passed")
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and all(type(gate.get(field)) is bool for field in gate_fields))
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            primary = measurements.get("arms", {}).get("observed", {})
            paired = primary.get("reversal_minus_stationary", {})
            random_reversal_gates.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                estimate(primary.get("conditions", {}).get("reversal", {}).get("windows", {}).get("late", {}).get("first_food", {})),
                *[estimate(paired.get(field, {})) for field in
                  ("recovery_difference", "late_first_food_difference", "end_survival_difference")],
                outcome, "undecided", commit,
            ]) + " |")
            if complete:
                for arm in ("observed", "belief_reset", "birth_parent", "direct_baseline"):
                    panel = measurements.get("arms", {}).get(arm, {})
                    differences = panel.get("reversal_minus_stationary", {})
                    random_reversal_pairs.append("| " + " | ".join([
                        d.name, split, arm, *[estimate(differences.get(field, {})) for field in
                            ("pre_first_food_difference", "early_first_food_difference", "late_first_food_difference",
                             "recovery_difference", "end_survival_difference")],
                    ]) + " |")
                    for condition in ("stationary", "reversal"):
                        life = panel.get("conditions", {}).get(condition, {})
                        random_reversal_survival.append("| " + " | ".join([
                            d.name, split, arm, condition, estimate(life.get("fitness", {})),
                            *[estimate(life.get("survival", {}).get(point, {})) for point in
                              ("at_flip", "after_early", "after_late", "end")],
                        ]) + " |")
                        for window in ("pre", "early", "late"):
                            metric = life.get("windows", {}).get(window, {})
                            random_reversal_windows.append("| " + " | ".join([
                                d.name, split, arm, condition, window,
                                *[estimate(metric.get(field, {})) for field in ("first_food", "no_meal", "fitness", "accuracy")],
                                str(metric.get("food_encounters", "?")), str(metric.get("poison_encounters", "?")),
                                str(metric.get("encounters", "?")),
                            ]) + " |")
                        for group in ("alive_at_flip_only", "end_survivor_only"):
                            subgroup = life.get(group, {})
                            random_reversal_subgroups.append("| " + " | ".join([
                                d.name, split, arm, condition, group, str(subgroup.get("families", "?")),
                                *[estimate(subgroup.get("windows", {}).get(window, {}).get("first_food", {}))
                                  for window in ("pre", "early", "late")],
                            ]) + " |")
                for comparison in ("within_social", "within_birth"):
                    for condition in ("stationary", "reversal"):
                        paired = measurements.get(comparison, {}).get(condition, {})
                        for window in ("pre", "early", "late"):
                            metric = paired.get("windows", {}).get(window, {})
                            random_reversal_controls.append("| " + " | ".join([
                                d.name, split, comparison, condition, window,
                                estimate(metric.get("first_food_difference", {})), estimate(metric.get("fitness_difference", {})),
                                estimate(paired.get("end_survival_difference", {})),
                            ]) + " |")
            continue
        if a.get("eval_protocol") == "social-transmission-v1":
            split = a.get("condition_split", "")
            expected = 128 if split == "final" else 4 if split == "development" else None
            frozen = s.get("frozen_calibration") or {}
            complete = (expected is not None and s.get("status") == "finished"
                        and s.get("artifact_validity") == "passed"
                        and s.get("completed_chains") == s.get("expected_chains") == expected
                        and len(s.get("raw_files", [])) == expected
                        and frozen.get("path") == "calibration/selection.json"
                        and len(frozen.get("sha256", "")) == 64)
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            interval = lambda metric: ("undefined" if metric.get("lower_2se") is None or metric.get("upper_2se") is None
                                       else f"[{metric['lower_2se']:.6f}, {metric['upper_2se']:.6f}]")
            verdicts = []
            fields = ("passed", "first_food_passed", "survival_passed")
            for decision in ("retained_efficacy", "cumulative_improvement"):
                gate = s.get(decision, {})
                decided = (complete and split == "final" and s.get("scientific_result") is True
                           and all(type(gate.get(field)) is bool for field in fields))
                verdicts.append(("PASS" if gate["passed"] else "FAIL") if decided
                                else "not applicable" if complete and split == "development" else "undecided")
            social_transmission_gates.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_chains', '?')}/{cfg.get('expected_chains', '?')}",
                str(s.get("selected_lambda", "undefined")), *verdicts, commit,
            ]) + " |")
            if complete:
                for cohort in (1, 2, 3):
                    cohort_data = measurements.get("cohorts", {}).get(str(cohort), {})
                    for arm in ("predecessor", "repeated_founder", "yoked", "belief_reset"):
                        metrics = cohort_data.get("arms", {}).get(arm, {})
                        social_transmission_arms.append("| " + " | ".join([
                            d.name, split, str(cohort), arm,
                            *[estimate(metrics.get(metric, {})) for metric in
                              ("first_food", "no_meal", "survival", "fitness", "decoder_coverage")],
                        ]) + " |")
                    for control in ("repeated_founder", "yoked", "belief_reset"):
                        metrics = cohort_data.get("contrasts", {}).get(control, {})
                        social_transmission_contrasts.append("| " + " | ".join([
                            d.name, split, str(cohort), "predecessor minus " + control,
                            estimate(metrics.get("first_food_difference", {})), interval(metrics.get("first_food_difference", {})),
                            estimate(metrics.get("survival_difference", {})), interval(metrics.get("survival_difference", {})),
                        ]) + " |")
                metrics = measurements.get("cross_cohort", {}).get("c3_vs_c1", {})
                social_transmission_contrasts.append("| " + " | ".join([
                    d.name, split, "3 versus 1", "c3 versus c1 predecessor",
                    estimate(metrics.get("first_food_difference", {})), interval(metrics.get("first_food_difference", {})),
                    estimate(metrics.get("survival_difference", {})), interval(metrics.get("survival_difference", {})),
                ]) + " |")
            continue
        if a.get("eval_protocol") == "social-belief-readout-v1":
            split = a.get("condition_split", "")
            expected = (32, 128) if split == "final" else (4, 4) if split == "development" else (None, None)
            calibration = s.get("calibration") or {}
            frozen = s.get("frozen_calibration") or {}
            calibrated = (expected[0] is not None and calibration.get("status") == "frozen"
                          and calibration.get("completed_families") == calibration.get("expected_families") == expected[0]
                          and s.get("completed_calibration_families") == s.get("expected_calibration_families") == expected[0]
                          and frozen.get("path") == "calibration/selection.json"
                          and len(frozen.get("sha256", "")) == 64)
            complete = (calibrated and s.get("status") == "finished" and s.get("artifact_validity") == "passed"
                        and s.get("completed_families") == s.get("expected_families") == expected[1])
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None else
                                       f"{metric['mean']:.6f} ± " +
                                       ("undefined" if metric.get("se") is None else f"{metric['se']:.6f}"))
            gate = s.get("gate", {})
            gate_fields = ("passed", "first_food_passed", "survival_passed")
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and all(type(gate.get(field)) is bool for field in gate_fields))
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            social_belief_gates.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_calibration_families', '?')}/{cfg.get('expected_calibration_families', '?')}",
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                *[("PASS" if gate[field] else "FAIL") if decided else "not applicable" if complete and split == "development"
                  else "undecided" for field in gate_fields[1:]], outcome, commit,
            ]) + " |")
            if calibrated:
                social_belief_calibrations.append("| " + " | ".join([
                    d.name, split, str(calibration.get("selected_lambda", "undefined")),
                    f"{calibration.get('contacts', '?')}/{calibration.get('families', '?')}",
                    estimate(calibration.get("contact_coverage", {})), estimate(calibration.get("vote_coverage", {})),
                ]) + " |")
                for loss in calibration.get("losses", []):
                    social_belief_losses.append("| " + " | ".join([
                        d.name, str(loss.get("lambda", "undefined")),
                        estimate(dict(mean=loss.get("mean_log_loss"), se=loss.get("se_log_loss"))),
                    ]) + " |")
            if complete:
                for arm in ("observed", "yoked", "scene_only", "no_demo", "belief_reset", "birth_parent"):
                    metrics = measurements.get("arms", {}).get(arm, {})
                    social_belief_arms.append("| " + " | ".join([
                        d.name, split, arm, *[estimate(metrics.get(metric, {})) for metric in
                                            ("first_food", "no_meal", "survival", "fitness", "first_meal_step")],
                    ]) + " |")
                for control in ("yoked", "scene_only", "no_demo", "belief_reset", "birth_parent"):
                    metrics = measurements.get("contrasts", {}).get(control, {})
                    social_belief_contrasts.append("| " + " | ".join([
                        d.name, split, control, estimate(metrics.get("first_food_difference", {})),
                        estimate(metrics.get("survival_difference", {})),
                    ]) + " |")
            continue
        if a.get("eval_protocol") == "social-observer-train-v1":
            split = a.get("condition_split", "")
            expected = (10, 128) if split == "final" else (1, 4) if split == "development" else (None, None)
            complete = (s.get("status") == "finished" and s.get("artifact_validity") == "passed"
                        and s.get("attempted_rounds") == s.get("expected_attempts") == expected[0]
                        and s.get("completed_families") == s.get("expected_families") == expected[1])
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            gate = s.get("gate", {})
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and all(type(gate.get(name)) is bool for name in ("passed", "first_food_passed", "survival_passed")))
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            social_observer_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('attempted_rounds', '?')}/{cfg.get('expected_attempts', '?')}",
                str(s.get("optimizer_updates", "?")),
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                estimate(measurements.get("arms", {}).get("observed", {}).get("first_food", {})),
                estimate(measurements.get("arms", {}).get("starting_observed", {}).get("first_food", {})),
                *[estimate(measurements.get("contrasts", {}).get(control, {}).get("first_food_difference", {}))
                  for control in ("yoked", "no_demo", "scene_only", "starting_observed")], outcome, commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") == "social-channel-v1":
            split = a.get("condition_split", "")
            expected = 128 if split == "final" else 4 if split == "development" else None
            complete = (expected is not None and s.get("status") == "finished"
                        and s.get("artifact_validity") == "passed"
                        and s.get("completed_families") == s.get("expected_families") == expected)
            measurements = s.get("measurements", {}) if complete else {}
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            gate = s.get("gate", {})
            gate_fields = ("passed", "coverage_passed", "accuracy_passed", "chance_passed", "yoked_passed")
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and all(type(gate.get(field)) is bool for field in gate_fields))
            decision = lambda field: (("PASS" if gate[field] else "FAIL") if decided else
                                      "not applicable" if complete and split == "development" else "undecided")
            social_channel_gates.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                *[decision(field) for field in gate_fields[1:]], decision("passed"), commit,
            ]) + " |")
            if complete:
                for method in ("track", "scene"):
                    panel = measurements.get(method, {})
                    for arm in ("observed", "yoked", "no_demo", "scene_only"):
                        metrics = panel.get("arms", {}).get(arm, {})
                        social_channel_rows.append("| " + " | ".join([
                            d.name, split, method, arm,
                            *[estimate(metrics.get(metric, {})) for metric in (
                                "accuracy", "coverage", "no_vote_fraction", "tie_fraction", "votes", "identifiable_steps")],
                        ]) + " |")
                    social_channel_contrasts.append("| " + " | ".join([
                        d.name, split, method, estimate(panel.get("chance_difference", {})),
                        *[estimate(panel.get("comparisons", {}).get(control, {}))
                          for control in ("yoked", "no_demo", "scene_only")],
                    ]) + " |")
            continue
        if a.get("eval_protocol") == "social-learning-v1":
            split = a.get("condition_split", "")
            expected = 128 if split == "final" else 4 if split == "development" else None
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
            social_learning_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                estimate(measurements.get("arms", {}).get("observed", {}).get("first_food", {})),
                *[estimate(measurements.get("contrasts", {}).get(control, {}).get("first_food_difference", {}))
                  for control in ("yoked", "no_demo", "scene_only")],
                outcome, commit,
            ]) + " |")
            continue
        if a.get("eval_protocol") == "inherited-evolution-v1":
            split = a.get("condition_split", "")
            expected = (10, 128) if split == "final" else (1, 4) if split == "development" else (None, None)
            complete = (s.get("status") == "finished" and s.get("artifact_validity") == "passed"
                        and s.get("generation") == s.get("expected_generations") == expected[0]
                        and s.get("completed_families") == s.get("expected_families") == expected[1])
            measurements = s.get("measurements", {}) if complete else {}
            comparison = measurements.get("contrasts", {}).get("child_vs_parent", {})
            estimate = lambda metric: ("undefined" if metric.get("mean") is None or metric.get("se") is None
                                       else f"{metric['mean']:.6f} ± {metric['se']:.6f}")
            gate = s.get("gate", {})
            decided = (complete and split == "final" and s.get("scientific_result") is True
                       and type(gate.get("passed")) is bool)
            outcome = ("PASS" if gate["passed"] else "FAIL") if decided else "undecided"
            if complete and split == "development":
                outcome = "not applicable"
            inherited_evolution_rows.append("| " + " | ".join([
                d.name, s.get("status", "no summary"), split,
                f"{s.get('generation', '?')}/{cfg.get('expected_generations', '?')}",
                f"{s.get('completed_families', '?')}/{cfg.get('expected_families', '?')}",
                estimate(measurements.get("parent_previous", {}).get("latency", {})),
                estimate(measurements.get("child_latest", {}).get("latency", {})),
                estimate(comparison.get("latency_improvement", {})),
                estimate(comparison.get("survival_difference", {})), outcome, commit,
            ]) + " |")
            continue
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
    if background_selection_rows:
        out += ["## Paired selection on masked and ordinary backgrounds (background-selection-v1)", "",
            "Both selection arms computed the same two first-view scores, with one preassigned scalar per arm. "
            "The [protocol](../experiments/E136_PROTOCOL.md) gives separate stationary and reversal decisions.", "",
            "| Run | Mode | Selected view | Status | ES attempts completed/assigned | Families completed/assigned | Commit |",
            "|---" * 7 + "|", *background_selection_rows, ""]
        if (ROOT / "results/diagnostics/E136-independent-audit.json").is_file():
            out += ["[Independent reconstruction and paired lineage results](diagnostics/E136-independent-audit.json).", ""]
    if scene_transfer_rows:
        from scripts import run_scene_transfer
        study = run_scene_transfer.build_comparison(ROOT)
        out += ["## First-choice transfer across visual scenes (scene-transfer-v1)", "",
            "E135 compares each frozen E134 attempt-ten descendant with its original E103 parent, with zero "
            "new training. After identical stationary or reversed histories, discarded clones choose among "
            "the same adjacent A/B items in the same ordinary-density physical scene. Only the first sensory "
            "view changes: extra items are masked or fully visible. Both learning processes are frozen for "
            "that decision. A randomly assigned placement continues freely with ordinary learning, and a "
            "separate ordinary-grid reference tests the same acquired state.", "",
            "The primary gate concerns stationary experience. First replicate at least a 0.05 descendant "
            "advantage in the masked view, then require at least 0.05 attenuation of that advantage in the "
            "full view. Each effect must have a strictly positive lower two-SE bound in all three lineages. "
            "Failure to establish attenuation does not exclude it; excluding a loss of 0.05 requires the "
            "registered upper-bound criterion. Reversed histories and subsequent movement are reported separately.", "",
            "| Run | Mode | Status | Families completed/assigned | Commit |", "|---" * 5 + "|", *scene_transfer_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated first-choice attenuation: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E135-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")

            def scene_estimate(value):
                result = ("null (no contacts)" if value["mean"] is None else
                          f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})")
                if "contact_count" in value:
                    result += f"; {value['food_count']}/{value['contact_count']} food/contact, {value['assigned_lives']} assigned lives"
                return result

            def scene_estimates(value, prefix=""):
                if isinstance(value, dict) and {"mean", "se", "n"}.issubset(value):
                    yield prefix, value
                elif isinstance(value, dict):
                    for key, child in value.items():
                        yield from scene_estimates(child, prefix + ("." if prefix else "") + key)

            verdict = study["replicated_gates"]["passed"]
            out += [f"Replicated first-choice attenuation: **{'PASS' if verdict['passed'] else 'FAIL'}** ({verdict['count']}/3 lineages).", "",
                "A pass identifies a first-observation context effect on the inherited advantage. It does not "
                "by itself establish that the advantage vanishes: the full-view bounds distinguish erasure, "
                "partial attenuation and unresolved cases. A failed masked-view replication leaves that "
                "prerequisite open. Every assigned family remains in the analysis; no-contact lives score "
                "zero in signed first-item Y. Neither a first-choice deficit nor a later solo deficit alone "
                "establishes forgetting.", ""]
            classifications = ["| " + " | ".join([name, str(value["count"]) + "/3",
                ", ".join("A" + str(index) for index in value["seeds"]) or "none"]) + " |"
                for name, value in study["replicated_gates"].items()]
            contrasts, decisions, estimates = [], [], []
            for panel in study["panels"]:
                label, values = "A" + str(panel["index"]), panel["measurements"]
                decisions.append("| " + " | ".join([label, ", ".join(name for name, passed in panel["gate"].items() if passed) or "none"]) + " |")
                for history, measured in values["histories"].items():
                    contrasts.append("| " + " | ".join([label, history, *[scene_estimate(measured[name])
                        for name in ("masked_gain", "full_gain", "attenuation", "reference_gain")]]) + " |")
                for name, value in scene_estimates(values):
                    lower = "—" if "lower_2se" not in value else f"{value['lower_2se']:.6f}"
                    upper = "—" if "upper_2se" not in value else f"{value['upper_2se']:.6f}"
                    estimates.append("| " + " | ".join([label, name, scene_estimate(value), lower, upper]) + " |")
            out += ["Each ± is one sample SE across 128 independent families. First-decision measures average "
                    "both initial rules and all twelve placements within each family before estimates and "
                    "paired contrasts. Continuation and reference measures average the two initial-rule "
                    "lives within a family. Contact accuracy is descriptive, reports its food/contact denominator and is "
                    "null without contacts. Its ratio uncertainty uses family clusters. The approach and "
                    "conditional-choice decomposition is descriptive. Zero-distractor scenes remain included.", "",
                "| Classification | Lineages meeting criterion | Lineages |", "|---" * 3 + "|", *classifications, "",
                "| Lineage | Classifications established |", "|---" * 2 + "|", *decisions, "",
                "| Lineage | History | Masked descendant gain | Full descendant gain | Attenuation | Ordinary-reference solo gain |",
                "|---" * 6 + "|", *contrasts, "",
                "| Lineage | Measure | Mean ± family SE | Lower two-SE bound | Upper two-SE bound |",
                "|---" * 5 + "|", *estimates, ""]
    if replay_selection_rows:
        out += ["## Selection on independent choice after matched replay (replay-selection-v1)", "",
            "E134 selects the native genome on expected signed choice after matched experience. "
            "Each endpoint is frozen before separate evaluation against its own E103 parent. "
            "See the [protocol](../experiments/E134_PROTOCOL.md) for the feedback ablation and decision rules.", "",
            "| Run | Mode | Status | ES attempts completed/assigned | Families completed/assigned | Commit |",
            "|---" * 6 + "|", *replay_selection_rows, ""]
        audit = ROOT / "results/diagnostics/E134-independent-audit.json"
        if audit.is_file():
            out += ["[Independent held-out reconstruction and gate decisions](diagnostics/E134-independent-audit.json).", ""]
    if contact_gate_status_rows:
        from scripts import run_contact_gate
        study = run_contact_gate.build_comparison(ROOT)
        out += ["## Contact-timed Hebbian updating (contact-gate-v1)", "",
            "E133 uses the three original saved controllers with zero training. Each family replays both initial "
            "rules with identical observations, imposed actions and received feedback across native updating, "
            "a locally observed contact gate, equally sparse randomly timed updates and a reopened-delay ablation. "
            "After sixteen common contacts and eight stationary or reversed contacts, local choice and fresh "
            "fifty-turn solo behavior are measured immediately and after 33 empty turns. All solo readouts "
            "use ordinary learning, so this measures transfer of the carried state, not live use of the gate.", "",
            "Every assigned life counts. Two rule replays are averaged within each of 64 independent families "
            "before sample standard errors and paired contrasts. A dead exposure state gets no fresh body: "
            "local and signed solo choice are zero, with no contact retained. Solo Y is first food minus first "
            "poison. The sparse schedule draws one offset independently in each four-turn block and can "
            "coincide with contact feedback by chance; its overlap is descriptive.", "",
            "Retention requires verified immediate acquisition plus the registered delayed comparison. "
            "Every acquisition, retention, transfer, stationary and survival criterion must pass in each "
            "of the three lineages. Stationary-adjusted solo differences are descriptive.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|", *contact_gate_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated contact gate: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E133-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value: f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})"
            verdict = study["replicated_gates"]["contact_gate"]
            out += [f"Replicated contact gate: **{'PASS' if verdict['passed'] else 'FAIL'}** ({verdict['count']}/3 lineages).", "",
                "A full pass would justify a separately registered autonomous deployment check before bounded "
                "equal-budget selection of a heritable between-contact learning gain. Failed immediate "
                "acquisition points to feedback use. A local retention effect without transfer does not "
                "advance this design into evolutionary training. Failure to distinguish equal sparsity or "
                "the reopened delay retires this setting without tuning on these assignments.", ""]
            gates, components, safeguards, endpoints = [], [], [], []
            for panel in study["panels"]:
                values, label = panel["measurements"], str(panel["index"])
                gates.append("| " + " | ".join([label, "PASS" if values["gate"]["passed"] else "FAIL",
                    ", ".join(values["gate"]["failed_criteria"]) or "none"]) + " |")
                for name, value in values["components"].items():
                    components.append("| " + " | ".join([label, name, estimate(value)]) + " |")
                for name, value in values["safeguards"].items():
                    safeguards.append("| " + " | ".join([label, name, estimate(value), f"{value['lower_2se']:.6f}"]) + " |")
                for arm, fields in values["arms"].items():
                    for name, pair in fields.items():
                        endpoints.append("| " + " | ".join([label, arm, name, estimate(pair[0]), estimate(pair[1])]) + " |")
            out += ["One ± is one family SE, paired for comparisons. All rates use fractions. Immediate and "
                    "delayed solo probes are separate clones; neither changes the source state.", "",
                "| Lineage | Joint decision | Failed criteria |", "|---" * 3 + "|", *gates, "",
                "| Lineage | Component | Mean ± family SE |", "|---" * 3 + "|", *components, "",
                "| Lineage | Safeguard | Mean ± paired SE | Lower two-SE bound |", "|---" * 4 + "|", *safeguards, "",
                "| Lineage | Condition | Endpoint | Immediate mean ± family SE | Delayed mean ± family SE |", "|---" * 5 + "|", *endpoints, ""]
    if offer_status_rows:
        from scripts import run_voluntary_offers
        study = run_voluntary_offers.build_comparison(ROOT)
        out += ["## Voluntary access to both alternatives (voluntary-offers-v1)", "",
            "E132 uses the three original saved controllers with one creature per life. Every family replays "
            "both initial rules. After eight common nearby eight-turn offers, copies receive eight further "
            "offers crossing nearby versus random-distance placement with stationary versus reversed rules. "
            "The creature chooses every action. Local choice and a fresh fifty-turn solo probe measure what "
            "transfers beyond the presentation apparatus.", "",
            "Every initially assigned birth counts. Two rule replays are averaged inside each of 64 independent "
            "families before means, paired differences and sample standard errors. Both-color exposure counts "
            "actual contacts during the continuation, including a contact that causes death. The survival "
            "safeguard is measured at exposure tick 128; survival after the empty feedback tick 129 is shown "
            "separately. Dead readout states score zero for local and signed solo choice, with no-contact "
            "retained in the assigned denominator. Solo Y is first food minus first poison.", "",
            "All three lineages must pass the registered preference, access, solo-transfer and survival "
            "criteria. There is no policy-time ratio and no new training.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|", *offer_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated voluntary access: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E132-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value: f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})"
            verdict = study["replicated_gates"]["voluntary_access"]
            out += [f"Replicated voluntary access: **{'PASS' if verdict['passed'] else 'FAIL'}** ({verdict['count']}/3 lineages).", "",
                "A pass would support a contribution from easier access to the alternatives in this apparatus. "
                "It would leave calorie intake, visual stimulation and feedback contributions unresolved. "
                "When access improves but transfer fails, the result points to examining feedback use; a "
                "failed access manipulation cannot establish a memory defect.", ""]
            gates, components, safeguards, endpoints = [], [], [], []
            for panel in study["panels"]:
                values, label = panel["measurements"], str(panel["index"])
                gates.append("| " + " | ".join([label, "PASS" if values["gate"]["passed"] else "FAIL",
                    ", ".join(values["gate"]["failed_criteria"]) or "none"]) + " |")
                for rule, measured in [("both", values), *values["by_initial_rule"].items()]:
                    for name, value in measured["components"].items():
                        components.append("| " + " | ".join([label, rule, name, estimate(value)]) + " |")
                    for name, value in measured["safeguards"].items():
                        safeguards.append("| " + " | ".join([label, rule, name, estimate(value), f"{value['lower_2se']:.6f}"]) + " |")
                    for arm, fields in measured["arms"].items():
                        for name, value in fields.items():
                            endpoints.append("| " + " | ".join([label, rule, arm, name, estimate(value)]) + " |")
            out += ["All rates use fractions; one ± is one family SE. Initial-rule views are descriptive. "
                    "Intake counts each consumed color per assigned life, and realized energy is the sum "
                    "of signed actual energy changes during the 64 continuation turns, including clipped gains.", "",
                "| Lineage | Joint decision | Failed criteria |", "|---" * 3 + "|", *gates, "",
                "| Lineage | Initial rule | Component | Mean ± family SE |", "|---" * 4 + "|", *components, "",
                "| Lineage | Initial rule | Near-minus-sham safeguard | Mean ± paired SE | Lower two-SE bound |", "|---" * 5 + "|", *safeguards, "",
                "| Lineage | Initial rule | Condition | Endpoint | Mean ± family SE |", "|---" * 5 + "|", *endpoints, ""]
    if exploration_status_rows:
        from scripts import run_timed_exploration
        study = run_timed_exploration.build_comparison(ROOT)
        out += ["## Feedback-timed exploration (timed-exploration-v1)", "",
            "E131 compares the same saved generation-200 controller with unchanged actions, drought-timed "
            "exploration, and outcome-independent exploration with the same nominal quota. Independent "
            "random phase delays avoid alignment with the hidden switch at turn 150. All four original "
            "animals and both initial rules count inside each of 64 independent families per lineage.", "",
            "First-food latency uses offsets 0–49 in [150,200), or 50 without food, including death. "
            "Each fifty-turn Q is the first item: food +1, poison -1, no contact zero. Every lineage must "
            "improve latency and direct late reversal Q over both controls above two paired SE, retain "
            "positive adaptive late Q, and improve history-adjusted late Q by at least 0.05 and above "
            "two paired SE. Stationary late Q and survival in both histories must retain lower paired "
            "two-SE bounds above -0.05. The development runtime ceiling is a separate adoption condition.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|", *exploration_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated timed exploration: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E131-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value: f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})"
            verdict, cost = study["replicated_gates"]["timed_exploration"], study["operational_cost"]
            out += [f"Replicated timed exploration: **{'PASS' if verdict['passed'] else 'FAIL'}** ({verdict['count']}/3 lineages). "
                f"Adoption: **{'PASS' if study['adoption']['passed'] else 'FAIL'}**.", "",
                f"Warm development medians: ordinary {cost['ordinary_median_seconds']:.6f} s, "
                f"adaptive {cost['adaptive_median_seconds']:.6f} s, quota {cost['quota_median_seconds']:.6f} s. "
                f"Adaptive/ordinary {cost['ratio']:.6f}; ceiling 1.5; **{'PASS' if cost['passed'] else 'FAIL'}**. "
                "All four policy timings, including compilation, remain in the raw development record.", "",
                "No genotype was evolved or replaced. Differences isolate this action schedule on these "
                "worlds. Independent checks reconstruct events, policy counters and paired uncertainty; "
                "they do not recompute every neural or world transition.", ""]
            gates, components, safeguards, windows, endpoints = [], [], [], [], []
            for panel in study["panels"]:
                values, lineage = panel["measurements"], str(panel["index"])
                gates.append("| " + " | ".join([lineage, "PASS" if values["gate"]["passed"] else "FAIL",
                    ", ".join(values["gate"]["failed_criteria"]) or "none"]) + " |")
                for view, measurements in [("both", values), *values["by_initial_rule"].items()]:
                    for name, value in measurements["components"].items():
                        components.append("| " + " | ".join([lineage, view, "absolute", name, estimate(value)]) + " |")
                    for control, comparison in measurements["comparisons"].items():
                        for name in ("latency_saved", "late_Q_difference", "adjusted_late_Q"):
                            components.append("| " + " | ".join([lineage, view, control, name, estimate(comparison[name])]) + " |")
                        for name, value in comparison["safeguards"].items():
                            safeguards.append("| " + " | ".join([lineage, view, control, name, estimate(value), f"{value['lower_2se']:.6f}"]) + " |")
                    for arm, data in measurements["arms"].items():
                        endpoints.append("| " + " | ".join([lineage, view, arm, *[estimate(data[name]) for name in
                            ("first_food_latency", "first_any_latency", "end_survival", "override_count", "override_live_fraction", "changed_action_count", "changed_action_live_fraction")]]) + " |")
                        for window in data["windows"]:
                            windows.append("| " + " | ".join([lineage, view, arm, f"[{window['start']},{window['end']})",
                                *[estimate(window[name]) for name in ("Q", "first_food", "first_poison", "no_contact", "survival")]]) + " |")
            out += ["All rates below use fractions; one ± is one family SE. Initial-rule views are descriptive. "
                    "Override count is per assigned animal; live fraction is each family's overrides divided "
                    "by its alive-turn count, then summarized across families. Changed-action counts exclude "
                    "overrides that happen to match the ordinary sampled action.", "",
                "| Lineage | Behavioral decision | Failed criteria |", "|---" * 3 + "|", *gates, "",
                "| Lineage | Initial rule | Control | Component | Mean ± paired family SE |", "|---" * 5 + "|", *components, "",
                "| Lineage | Initial rule | Control | Safeguard | Mean ± paired SE | Lower two-SE bound |", "|---" * 6 + "|", *safeguards, "",
                "| Lineage | Initial rule | Policy/history | First-food latency | First-item latency | End survival | Overrides/assigned | Overrides/live turn | Changed actions/assigned | Changed actions/live turn |",
                "|---" * 10 + "|", *endpoints, "",
                "| Lineage | Initial rule | Policy/history | Window | Q | First food | First poison | No contact | Alive at window end |",
                "|---" * 9 + "|", *windows, ""]
    if founder_status_rows:
        from scripts import run_founder_reversal
        study = run_founder_reversal.build_comparison(ROOT)
        out += ["## Founder and descendant autonomous reversal (founder-reversal-v1)", "",
            "E130 pairs each original E103 founder with its generation-200 descendant in stationary and "
            "unannounced reversal worlds. Four original animals choose every action for 300 turns, with the "
            "food rule changing before consumption at turn 150. Each family replays both initial rules with "
            "the same world and development randomness across both genotypes and histories.", "",
            "Q scores each animal's first item in each fifty-turn window as +1 for food, -1 for poison and "
            "zero for no item, including death before contact. Average all four animals and both initial rules "
            "inside each of 64 independent families before computing a mean or sample SE. Pre, early and late "
            "windows are [100,150), [150,200) and [250,300). Endpoints and lineages are never pooled into extra samples.", "",
            "Every lineage must have positive descendant pre-switch and late reversal Q above two SE. The "
            "descendant's reversal improvement from early to late, its late advantage over the founder, and "
            "the genotype-by-history recovery interaction K must each be at least 0.05 and strictly above "
            "two paired SE. Each descendant-minus-founder lower two-SE bound must exceed -0.05 for pre Q, "
            "stationary late Q and final survival in both histories. All three lineages must pass every criterion.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|", *founder_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated evolutionary recovery: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E130-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value, percent=False: (f"{100*value['mean']:.2f}% ± {100*value['se']:.2f} pp (n={value['n']})"
                if percent else f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})")
            verdict = study["replicated_gates"]["evolutionary_recovery"]
            out += [f"Replicated evolutionary recovery: **{'PASS' if verdict['passed'] else 'FAIL'}** "
                    f"({verdict['count']}/3 lineages). Individual positive effects remain visible below.", ""]
            components, safeguards, gates, windows, survival, subgroups, subgroup_components, subgroup_safeguards = [], [], [], [], [], [], [], []
            for panel in study["panels"]:
                label, values = str(panel["index"]), panel["measurements"]
                gates.append("| " + " | ".join([label, "PASS" if values["gate"]["passed"] else "FAIL",
                    ", ".join(values["gate"]["failed_criteria"]) or "none"]) + " |")
                for name, metric in values["components"].items():
                    components.append("| " + " | ".join([label, name, estimate(metric),
                        "yes" if metric["mean"] > 2*metric["se"] else "no"]) + " |")
                for name, metric in values["safeguards"].items():
                    safeguards.append("| " + " | ".join([label, name, estimate(metric), f"{metric['lower_2se']:.6f}",
                        "PASS" if metric["lower_2se"] > -.05 else "FAIL"]) + " |")
                for arm, data in values["arms"].items():
                    survival.append("| " + " | ".join([label, arm, estimate(data["end_survival"], True)]) + " |")
                    for window in data["windows"]:
                        windows.append("| " + " | ".join([label, arm, f"[{window['start']},{window['end']})", estimate(window["Q"]),
                            *[estimate(window[name], True) for name in ("first_food", "first_poison", "no_contact", "survival")]]) + " |")
                for rule, data in values["by_initial_rule"].items():
                    for arm, outcome in data["arms"].items():
                        for window in outcome["windows"]:
                            subgroups.append("| " + " | ".join([label, rule, arm, f"[{window['start']},{window['end']})", estimate(window["Q"]),
                                *[estimate(window[name], True) for name in ("first_food", "first_poison", "no_contact", "survival")]]) + " |")
                    for name, metric in data["components"].items():
                        subgroup_components.append("| " + " | ".join([label, rule, name, estimate(metric)]) + " |")
                    for name, metric in data["safeguards"].items():
                        subgroup_safeguards.append("| " + " | ".join([label, rule, name, estimate(metric), f"{metric['lower_2se']:.6f}"]) + " |")
            out += ["The independent check reconstructs event-based outcomes and uncertainty. It does not "
                    "recompute every neural or world transition. Comparing generation zero with generation "
                    "200 cannot establish steady improvement across intervening generations.", "",
                "| Lineage | Joint decision | Failed criteria |", "|---" * 3 + "|", *gates, "",
                "| Lineage | Component | Mean ± family SE | Positive above two SE |", "|---" * 4 + "|", *components, "",
                "| Lineage | Descendant-minus-founder safeguard | Mean ± paired SE | Lower two-SE bound | Above -0.05 |", "|---" * 5 + "|", *safeguards, "",
                "| Lineage | Genotype/history | Window | Q ± SE | First food | First poison | No contact | Alive at window end |", "|---" * 8 + "|", *windows, "",
                "| Lineage | Genotype/history | End-of-life survival |", "|---" * 3 + "|", *survival, "",
                "Initial-rule views are descriptive and use the same family denominator:", "",
                "| Lineage | Initial rule | Genotype/history | Window | Q ± SE | First food | First poison | No contact | Alive at window end |",
                "|---" * 9 + "|", *subgroups, "",
                "| Lineage | Initial rule | Component | Mean ± family SE |", "|---" * 4 + "|", *subgroup_components, "",
                "| Lineage | Initial rule | Safeguard | Mean ± paired SE | Lower two-SE bound |", "|---" * 5 + "|", *subgroup_safeguards, ""]
    if retention_intervention_status_rows:
        from scripts import run_retention_intervention
        study = run_retention_intervention.build_comparison(ROOT)
        out += ["## Hebbian updates during the empty interval (retention-intervention-v1)", "",
            "E129 compares normal and temporarily Hebbian-off retention in the original plastic controllers, "
            "with paired reversed and stationary histories on fresh families. The immediate snapshot is shared. "
            "Only Hebbian changes during the 33 empty turns are disabled in the intervention; prediction updates "
            "continue and Hebbian updates resume in the fifty-turn free grid. The fixed-weight controls have "
            "only their normal trajectory and remain descriptive.", "",
            "The primary effect requires a Hebbian-off minus normal signed-preference gain of at least 0.025 "
            "and strictly more than two paired SE in all three lineages. Local rescue additionally requires "
            "every original E128 local criterion in the intervention. Transfer rescue additionally requires "
            "the E128 grid criterion and a positive first-food gain above two paired SE in those same three "
            "lineages. The original 0.05 retention-loss margin is unchanged. All uncertainties average both "
            "initial rules and local placements inside independent families before paired statistics.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|",
            *retention_intervention_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated intervention decisions: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E129-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value: f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})"
            decisions, effects, baseline, changes, primary, local, grids, prior, subgroups = [], [], [], [], [], [], [], [], []
            for name, verdict in study["replicated_gates"].items():
                decisions.append("| " + " | ".join([name, f"{verdict['count']}/3",
                    ", ".join(map(str, verdict["seeds"])) or "none", "PASS" if verdict["passed"] else "FAIL"]) + " |")
            for panel in study["panels"]:
                label, measures = str(panel["index"]), panel["measurements"]
                intervention = measures["intervention"]
                effects.append("| " + " | ".join([label, estimate(intervention["local_difference_33"]),
                    estimate(intervention["grid_difference_33"]), *[
                        "PASS" if intervention[name]["passed"] else "FAIL: " + ", ".join(intervention[name]["failed_criteria"])
                        for name in ("primary", "local_rescue", "transfer_rescue")]]) + " |")
                for condition in ("normal", "frozen"):
                    data = measures[condition]
                    for architecture, values in data["architectures"].items():
                        baseline.append("| " + " | ".join([label, condition, architecture,
                            "PASS" if values["local"]["passed"] else "FAIL: " + ", ".join(values["local"]["failed_criteria"]),
                            "PASS" if values["grid"]["passed"] else "FAIL: " + ", ".join(values["grid"]["failed_criteria"])]) + " |")
                        for endpoint in ("local", "grid"):
                            value = values[endpoint]["change_33_minus_0"]
                            changes.append("| " + " | ".join([label, condition, architecture,
                                "S" if endpoint == "local" else "G", estimate(value), f"{value['lower_2se']:.6f}"]) + " |")
                        for delay in ("0", "33"):
                            choice, grid = values["local"]["by_delay"][delay], values["grid"]["by_delay"][delay]
                            primary.append("| " + " | ".join([label, condition, architecture, delay,
                                estimate(choice["signed_preference"]), estimate(choice["reversal_minus_stationary"]),
                                estimate(grid["first_food"]), estimate(grid["reversal_minus_stationary"]),
                                estimate(grid["food_minus_poison"])]) + " |")
                    for arm, values in data["arms"].items():
                        prior.append("| " + " | ".join([label, arm, estimate(values["pre_old"])]) + " |")
                        for delay, state in values["by_delay"].items():
                            local.append("| " + " | ".join([label, arm, delay,
                                *[estimate(state["local"][name]) for name in run_retention_intervention.LOCAL_FIELDS]]) + " |")
                            grids.append("| " + " | ".join([label, arm, delay,
                                *[estimate(state["grid"][name]) for name in run_retention_intervention.GRID_FIELDS]]) + " |")
                for rule, value in intervention["by_initial_rule"].items():
                    subgroups.append("| " + " | ".join([label, rule, estimate(value["local_difference_33"]),
                        estimate(value["grid_difference_33"])]) + " |")
            out += ["Independent three-panel reconstruction passed. This is the single registered retention "
                    "intervention; the primary effect and the two rescue claims are reported separately.", "",
                "| Milestone | Qualifying lineages | Seed indices | Decision |", "|---" * 4 + "|", *decisions, "",
                "| Seed | Hebbian-off minus normal S at 33 ± paired SE | Hebbian-off minus normal G at 33 ± paired SE | Primary | Local rescue | Transfer rescue |",
                "|---" * 6 + "|", *effects, "",
                "| Seed | Retention condition | Architecture | Original local criteria | Original grid criteria |", "|---" * 5 + "|", *baseline, "",
                "| Seed | Retention condition | Architecture | Endpoint | 33-minus-0 change ± paired SE | Lower two-SE bound |", "|---" * 6 + "|", *changes, "",
                "| Seed | Retention condition | Architecture | Active turns | S ± SE | C ± paired SE | G ± SE | G history contrast ± paired SE | Y ± SE |",
                "|---" * 9 + "|", *primary, "",
                "| Seed | History | Old-food preference before reversal ± SE |", "|---" * 3 + "|", *prior, "",
                "| Seed | History | Active turns | P(food) ± SE | P(poison) ± SE | Approach ± SE | Other ± SE | Signed preference ± SE |", "|---" * 8 + "|", *local, "",
                "| Seed | History | Active turns | First food ± SE | First poison ± SE | No contact ± SE | Y ± SE | Latency ± SE | Net return ± SE | Survival ± SE |",
                "|---" * 10 + "|", *grids, "",
                "Initial-rule intervention differences are descriptive:", "",
                "| Seed | Initial rule | Local gain at 33 ± paired SE | First-food gain at 33 ± paired SE |", "|---" * 4 + "|", *subgroups, ""]
    if retention_status_rows:
        from scripts import run_retention_diagnostic
        study = run_retention_diagnostic.build_comparison(ROOT)
        out += ["## Retention after guided evidence (retention-diagnostic-v1)", "",
            "E128 compares the same original E103 plastic and E113 fixed-weight controllers on new families. "
            "After sixteen changed or stationary encounters, the complete living state continues for 0, 1 or "
            "33 active turns on an empty torus. Ordinary body feedback, recurrence and lifetime updates continue; "
            "no items appear. Discarded local probes and a fresh-body, fifty-turn grid readout use paired layouts "
            "and randomness across histories and delays. There is no new training or brain mechanism.", "",
            "All errors are one sample SE across independent families, after averaging twelve local placements "
            "and both initial rules inside each family. S is the reversal-history signed food preference; C is "
            "its paired reversal-minus-stationary contrast. G is first-food success across all assigned lives; "
            "Y is first-food minus first-poison success, with no contact or prior death scoring zero.", "",
            "Local retention requires S and C at both 0 and 33 turns to be at least 0.10 and strictly above "
            "two SE. Grid retention requires positive Y above two SE at both times and a 33-turn paired G "
            "advantage of at least 0.10 and above two SE. Each reversal-history change from 0 to 33 must also "
            "have a lower two-SE bound strictly above -0.05. All three lineages must pass local retention; "
            "the same three must pass local and grid criteria for the combined milestone. The one-turn "
            "snapshot and old-food preference before reversal are descriptive. Fixed-weight controls are secondary.", "",
            "| Run | Status | Split | Families completed/assigned | Commit |", "|---" * 5 + "|",
            *retention_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated retention decisions: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E128-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimate = lambda value: f"{value['mean']:.6f} ± {value['se']:.6f} (n={value['n']})"
            decisions, criteria, changes, primary, prior, local, grids, subgroups = [], [], [], [], [], [], [], []
            for architecture, effects in study["replicated_gates"].items():
                for effect, verdict in effects.items():
                    decisions.append("| " + " | ".join([architecture, effect, f"{verdict['count']}/3",
                        ", ".join(map(str, verdict["seeds"])) or "none", "PASS" if verdict["passed"] else "FAIL"]) + " |")
            for panel in study["panels"]:
                label, measures = str(panel["index"]), panel["measurements"]
                for architecture, values in measures["architectures"].items():
                    criteria.append("| " + " | ".join([label, architecture,
                        "PASS" if values["local"]["passed"] else "FAIL", ", ".join(values["local"]["failed_criteria"]) or "none",
                        "PASS" if values["grid"]["passed"] else "FAIL", ", ".join(values["grid"]["failed_criteria"]) or "none",
                        "PASS" if values["combined_passed"] else "FAIL"]) + " |")
                    for endpoint in ("local", "grid"):
                        change = values[endpoint]["change_33_minus_0"]
                        changes.append("| " + " | ".join([label, architecture, "S" if endpoint == "local" else "G",
                            estimate(change), f"{change['lower_2se']:.6f}",
                            "PASS" if values[endpoint]["criteria"]["retention"] else "FAIL"]) + " |")
                    for delay in ("0", "1", "33"):
                        choice, grid = values["local"]["by_delay"][delay], values["grid"]["by_delay"][delay]
                        primary.append("| " + " | ".join([label, architecture, delay,
                            estimate(choice["signed_preference"]), estimate(choice["reversal_minus_stationary"]),
                            estimate(grid["first_food"]), estimate(grid["reversal_minus_stationary"]),
                            estimate(grid["food_minus_poison"])]) + " |")
                for arm, values in measures["arms"].items():
                    prior.append("| " + " | ".join([label, arm, estimate(values["pre_old"])]) + " |")
                    for delay, state in values["by_delay"].items():
                        local.append("| " + " | ".join([label, arm, delay,
                            *[estimate(state["local"][name]) for name in run_retention_diagnostic.LOCAL_FIELDS]]) + " |")
                        grids.append("| " + " | ".join([label, arm, delay,
                            *[estimate(state["grid"][name]) for name in run_retention_diagnostic.GRID_FIELDS]]) + " |")
                for rule, architectures in measures["by_initial_rule"].items():
                    for architecture, values in architectures.items():
                        for delay, state in values["by_delay"].items():
                            subgroups.append("| " + " | ".join([label, rule, architecture, delay,
                                estimate(values["pre_old"]), *[estimate(state[name]) for name in ("S", "C", "G", "Y", "grid_difference")]]) + " |")
            out += ["Independent three-panel reconstruction passed. Threshold failures leave retention "
                    "unestablished in this assay; an absent immediate preference is not evidence of forgetting. "
                    "The empty interval changes scene, energy and activity, so a loss cannot identify its cause.", "",
                "| Architecture | Milestone | Qualifying lineages | Seed indices | Decision |", "|---" * 5 + "|", *decisions, "",
                "| Seed | Architecture | Local | Failed local criteria | Grid | Failed grid criteria | Combined |",
                "|---" * 7 + "|", *criteria, "",
                "| Seed | Architecture | Reversal-history endpoint | 33-minus-0 change ± paired SE | Lower two-SE bound | Above -0.05 |",
                "|---" * 6 + "|", *changes, "",
                "| Seed | Architecture | Active turns | S ± SE | C ± paired SE | G ± SE | G history contrast ± paired SE | Y ± SE |",
                "|---" * 8 + "|", *primary, "",
                "| Seed | History | Old-food preference before reversal ± SE |", "|---" * 3 + "|", *prior, "",
                "| Seed | History | Active turns | P(food) ± SE | P(poison) ± SE | Approach ± SE | Other ± SE | Signed preference ± SE |",
                "|---" * 8 + "|", *local, "",
                "| Seed | History | Active turns | First food ± SE | First poison ± SE | No contact ± SE | Y ± SE | Latency ± SE | Net return ± SE | Survival ± SE |",
                "|---" * 10 + "|", *grids, "",
                "Initial-rule breakdowns are descriptive, not extra independent replications:", "",
                "| Seed | Initial rule | Architecture | Active turns | Old-food preference ± SE | S ± SE | C ± paired SE | G ± SE | Y ± SE | G history contrast ± paired SE |",
                "|---" * 10 + "|", *subgroups, ""]
    if encounter_status_rows:
        from scripts import run_encounter_diagnostic
        study = run_encounter_diagnostic.build_comparison(ROOT)
        out += ["## Guided evidence and fresh-grid transfer (encounter-diagnostic-v1)", "",
                "E127 uses the original three E103 plastic controllers and matched E113 recurrent controllers. "
                "There is no training or fitted readout. Each independent family replays both initial rules on "
                "the same counterbalanced guided-contact script. The boundary follows eight or sixteen contacts; "
                "sixteen further contacts provide stationary or reversed feedback. Choice probes use discarded "
                "clones at zero, two, eight and sixteen post-boundary contacts. The grid readout preserves the "
                "unprobed brain in a fresh body and ordinary grid for fifty turns.", "",
                "All uncertainties use independent families after averaging both rule replays and, for local "
                "probes, twelve ordered color placements. S is target-food action probability minus other-color "
                "action probability. G is all-assigned first-food success in fifty turns. Y is +1 for first food, "
                "-1 for first poison and zero for no contact or death before contact; its SE retains covariance "
                "within each family. Net return is a separate whole-grid endpoint. A weak grid readout can reflect "
                "context, fresh body, search or memory effects and cannot isolate navigation alone.", "",
                "Development, incomplete panels and panels awaiting independent verification have no replicated "
                "scientific decision. Local reversal, transfer in the same lineages, and Hebbian contribution "
                "are separate decisions. A failed threshold means this assay did not establish that effect.", "",
                "The registered comparison uses E127-A0-recovery1, E127-A1 and E127-A2. The failed original "
                "A0 prefix remains preserved and is excluded from inference; its eleven families are not added "
                "to the recovery's 128-family denominator. The corrected four-family rehearsal is for "
                "verification and timing only.", "",
                "| Run | Status | Split | Families completed/assigned | Role | Commit |", "|---" * 6 + "|",
                *encounter_status_rows, ""]
        if study["status"] != "complete":
            out += ["Replicated diagnostic decisions: **undecided**; " + study["reason"] + ".", ""]
        else:
            destination = ROOT / "results/diagnostics/E127-comparison.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            estimates = lambda value: (f"undefined (n={value.get('n', '?')})" if value.get("mean") is None or value.get("se") is None
                                       else f"{value['mean']:.6f} ± {value['se']:.6f} (n={value.get('n', '?')})")
            decisions, primary, probes, grids, lesions, subgroups = [], [], [], [], [], []
            for architecture, effects in study["replicated_gates"].items():
                for effect, decision in effects.items():
                    decisions.append("| " + " | ".join([architecture, effect, f"{decision['count']}/3",
                        ", ".join(map(str, decision["seeds"])) or "none", "PASS" if decision["passed"] else "FAIL"]) + " |")
            for panel in study["panels"]:
                label, measures = str(panel["index"]), panel["measurements"]
                for architecture, values in measures["architectures"].items():
                    primary.append("| " + " | ".join([label, architecture,
                        *[estimates(values["local"][name]) for name in ("pre_old", "post_target", "reversal_minus_stationary")],
                        estimates(values["transfer"]["reversal_minus_stationary"]), estimates(values["transfer"]["food_minus_poison"]),
                        "PASS" if values["local_pass"] else "FAIL", "PASS" if values["transfer_pass"] else "FAIL"]) + " |")
                lesions.append("| " + " | ".join([label, estimates(measures["hebbian"]["local_difference"]),
                    estimates(measures["hebbian"]["transfer_difference"]),
                    "PASS" if measures["hebbian"]["hebbian_local_pass"] else "FAIL"]) + " |")
                for rule, architectures in measures.get("by_initial_rule", {}).items():
                    for architecture, values in architectures.items():
                        subgroups.append("| " + " | ".join([label, rule, architecture,
                            *[estimates(values[name]) for name in ("S", "delta_S", "G", "delta_G")]]) + " |")
                for arm, values in measures["arms"].items():
                    for contacts, reading in zip((0, 2, 8, 16), values["local"]):
                        for metric, value in reading.items():
                            if isinstance(value, dict) and {"mean", "se", "n"} <= set(value):
                                probes.append("| " + " | ".join([label, arm, str(contacts), metric, estimates(value)]) + " |")
                    grids.append("| " + " | ".join([label, arm, *[estimates(values["grid"][name]) for name in
                        ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")]]) + " |")
            out += ["Independent three-panel verification passed. Decisions use the frozen thresholds:", "",
                "| Architecture | Effect | Qualifying lineages | Seed indices | Replicated decision |", "|---" * 5 + "|", *decisions, "",
                "| Seed | Architecture | Old-food preference before reversal ± SE | S after sixteen contacts ± SE | "
                "Reversed minus stationary S ± paired SE | Reversed minus stationary G ± paired SE | Reversal Y ± SE | "
                "Local decision | Transfer decision |", "|---" * 9 + "|", *primary, "",
                "| Seed | Intact minus frozen S ± paired SE | Intact minus frozen G ± paired SE | Local Hebbian decision |",
                "|---" * 4 + "|", *lesions, "",
                "| Seed | Arm | Post-boundary contacts | Choice statistic | Mean ± family SE (denominator) |",
                "|---" * 5 + "|", *probes, "",
                "| Seed | Arm | G first food ± SE | First poison ± SE | No contact ± SE | Y ± SE | "
                "Latency ± SE | Net return ± SE | Survival ± SE |", "|---" * 9 + "|", *grids, ""]
            if subgroups:
                out += ["Initial-rule contrasts are descriptive and do not create extra independent replications:", "",
                        "| Seed | Initial rule | Architecture | S after reversed feedback ± SE | "
                        "Reversed minus stationary S ± paired SE | G after reversed feedback ± SE | "
                        "Reversed minus stationary G ± paired SE |", "|---" * 7 + "|", *subgroups, ""]
    if reversal_training_rows or reversal_endpoint_rows:
        from scripts.run_reversal_evolution import build_comparison
        study = build_comparison(ROOT)
        outcome = ("PASS" if study["passed"] else "FAIL") if study["status"] == "complete" else "undecided"
        out += ["## Matched from-blank reversal evolution (reversal-evolution-v1)", "",
                "E126 targets 200 attempted rounds in each of three plastic and three direct recurrent lineages. "
                "Both use population 128 and four base worlds replayed under both initial rules: eight actual lives "
                "per candidate. The objective is mean individual minimum late return across those rule replays. "
                "Genomes start from their original blank initializers. There is no social readout or new mechanism. "
                "Equal scores receive average ranks; all-tied attempts consume worlds and advance RNG without "
                "changing the genome, Adam moments or informative-update counter. Immutable segments preserve "
                "both counters. Development benchmarks measure throughput only.", "",
                "No held-out world opens before all six targets reach attempt 200. Final panels use 128 independent "
                "base worlds per seed pair, each under both rules and stationary/reversal interventions. Errors use "
                "base-world units. The formal architecture comparison uses combined standard errors; the paired "
                "difference is reported separately. At least two same seeds must pass the architecture, positive "
                "late return, late first-food and postflip Hebbian-lesion criteria. All three plastic seeds must "
                "pass the joint-survival floor and relative lower-two-SE screen. Partial panels and short training "
                "segments cannot decide the rung or authorize an added mechanism.", "",
                f"Complete three-seed rung decision: **{outcome}**.", "",
                "| Run | Status | Split | Target | Controller | Start attempt | Attempted / target | "
                "Informative updates | Scientific decision | Commit |", "|---" * 10 + "|",
                *reversal_training_rows, ""]
        if study["status"] == "complete":
            diagnostic = ROOT / "results/diagnostics/E126-comparison.json"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n")
            out += [f"Conjunctive performance criteria passed in {study['passing_seeds']}/3 seed pairs; "
                    f"all-seed survival safeguard: {'PASS' if study['all_seed_survival_passed'] else 'FAIL'}.", ""]
        if reversal_endpoint_rows:
            out += ["| Run | Status | Seed pair | Base worlds | Plastic late-min return ± SE | "
                    "Direct late-min return ± SE | Plastic minus direct ± combined SE | Paired difference ± SE | "
                    "Intact minus lesion ± paired SE | Plastic late first food ± SE | Plastic joint survival ± SE | "
                    "Joint-survival difference ± paired SE | Commit |", "|---" * 13 + "|",
                    *reversal_endpoint_rows, "",
                    "| Run | Arm | Window | First food ± SE | No contact ± SE | Window return ± SE | "
                    "Whole-life return ± SE | Joint survival ± SE |", "|---" * 8 + "|",
                    *reversal_window_rows, ""]
    if ecology_fixture_rows:
        out += ["## Ecology/lineage development fixtures", "",
                "Engineering exercises of a fixed cohort schedule and saved memory. Completion records mechanism "
                "execution, not competence, a scientific gate, or an EVAL v1 score. Partial attempts retain their "
                "completed cohort evidence; paired observations remain in each run folder.", "",
                "| Run | Status | Cohorts completed/expected | Split | Commit |", "|---" * 5 + "|",
                *ecology_fixture_rows, ""]
    if social_transmission_gates:
        out += ["## Three public-behavior handoffs (social-transmission-v1)", "",
                "E124 freezes each original E103 genotype and its E123 training-calibrated scalar. "
                "A solitary founder supplies its last 150 public frames to a newborn. Three successive "
                "cohorts watch, act alone, and supply their last 150 public frames to the next newborn. "
                "Only the predecessor arm teaches the next cohort. All assigned chains remain, including "
                "dead teachers and empty late footage. Fresh bodies, brains and solo layouts are drawn "
                "for each cohort. No weights, internal beliefs, rewards or hidden labels cross the handoff.", "",
                "First-food accuracy scores one for food and zero for first poison or no meal. Each mean "
                "and standard error uses independent chains; paired comparisons keep entire chains intact. "
                "Intervals show mean minus/plus two paired standard errors. Retained efficacy requires "
                "first-food gains over yoked and belief-reset movies strictly above two SE in every cohort. "
                "The survival lower two-SE bound must strictly exceed -0.10 against every same-cohort control, "
                "and for cohort 3 versus cohort 1. This ten-percentage-point tolerance is a confidence screen; "
                "it does not prove equal survival.", "",
                "Cumulative improvement additionally requires cohort 3 to beat both cohort 1 and the "
                "cohort 3 repeated-founder control by more than two paired SE. All gates must hold in all "
                "three lineages. Retention alone supports persistence through these three handoffs. "
                "Both gates support net improvement over these three cohorts, not perpetual or monotonic "
                "improvement. The genotype and engineered readout remain fixed. Development and incomplete "
                "attempts have no scientific decision.", "",
                "| Run | Status | Split | Chains | Frozen lambda | Retained efficacy | Cumulative improvement | Commit |",
                "|---" * 8 + "|", *social_transmission_gates, ""]
        if social_transmission_arms:
            out += ["| Run | Split | Cohort | Arm | First food ± SEM | No-meal fraction ± SEM | "
                    "Survival ± SEM | Fitness ± SEM | Decoder coverage ± SEM |",
                    "|---" * 9 + "|", *social_transmission_arms, "",
                    "| Run | Split | Cohort | Paired comparison | First-food difference ± SEM | "
                    "First-food two-SE interval | Survival difference ± SEM | Survival two-SE interval |",
                    "|---" * 8 + "|", *social_transmission_contrasts, ""]
    if social_belief_gates:
        out += ["## Calibrated engineered social readout (social-belief-readout-v1)", "",
                "E123 retains the original E103 genomes and adds an engineered scalar belief with a fixed "
                "local action readout. There are zero ES generations and no genome mutation. One scalar lambda "
                "is selected from nine fixed values using public movie contacts and an unassisted pupil's own "
                "first outcome on calibration worlds. The complete calibration evidence and selection are "
                "frozen before opening disjoint evaluation worlds. Calibration losses and coverage below are "
                "fitting diagnostics, not held-out performance.", "",
                "Each held-out family has six paired arms. First food is one for food and zero for poison or "
                "no meal, across all assigned families. No-meal first-choice latency is capped at step 301. "
                "Means and ± SEM use families as the independent unit; "
                "contrasts use paired family differences. The observed arm must beat every one of the five "
                "controls by strictly more than two paired SEM on first food, with survival differences "
                "at least minus two paired SEM. All three lineages must pass the complete intersection gate. "
                "A parent-only benefit cannot override a failed control or survival comparison. Development "
                "and incomplete attempts have no scientific verdict.", "",
                "The belief-reset control changes only the added belief; birth-parent uses the original "
                "unassisted birth state. A pass supports useful public information through this calibrated "
                "engineered readout; it does not show the unchanged hidden network invented social learning. "
                "A failed gate ends this version without retuning on its evaluation worlds.", "",
                "| Run | Status | Split | Calibration families | Evaluation families | All first-food contrasts | "
                "All survival safeguards | Lineage decision | Commit |", "|---" * 9 + "|",
                *social_belief_gates, ""]
        if social_belief_calibrations:
            out += ["Calibration diagnostics (fresh fitting worlds only):", "",
                    "| Run | Split | Frozen lambda | Own contacts / assigned | Contact coverage ± SEM | "
                    "Decoder vote coverage ± SEM |", "|---" * 6 + "|", *social_belief_calibrations, "",
                    "| Run | Candidate lambda | Calibration log loss ± SEM |", "|---" * 3 + "|",
                    *social_belief_losses, ""]
        if social_belief_arms:
            out += ["Held-out family outcomes:", "",
                    "| Run | Split | Arm | First food ± SEM | No-meal fraction ± SEM | Survival ± SEM | "
                    "Fitness ± SEM | First-meal step ± SEM |", "|---" * 8 + "|", *social_belief_arms, "",
                    "| Run | Split | Control | Observed first-food difference ± paired SEM | "
                    "Observed survival difference ± paired SEM |", "|---" * 5 + "|", *social_belief_contrasts, ""]
    if random_reversal_gates:
        out += ["## Saved-controller transfer to randomized rule reversal (reversal-random-v1)", "",
                "E125 changes only the hidden food interpretation at time 100, 125, 150 or 175, with 16 "
                "families per initial-rule/time cell in each final lineage. Every arm has paired stationary "
                "and reversal copies; the stationary copy uses the same shadow boundary for analysis. "
                "No new genome optimization or scalar calibration occurs. The fixed primary is the E123 "
                "prepared observer, with its original genome and pinned calibration. Its belief is not "
                "reset at the hidden flip.", "",
                "Windows relative to the boundary are pre [-50,0), early [0,50), and late [50,100). "
                "First contact is food scores one; poison, death or no contact scores zero across all "
                "assigned families. Means and ± SEM use independent families; all contrasts use paired "
                "family differences. Contact-conditioned accuracy includes its explicit encounter "
                "denominator and is undefined when no items were eaten.", "",
                "The primary passes only if late reversal success is at least 0.60 and exceeds 0.50 by "
                "more than two SEM; recovery (late minus early in reversal, minus the same change in "
                "stationary) exceeds two paired SEM; and the lower two-SEM bounds for late-success and "
                "end-survival reversal-minus-stationary differences both exceed -0.10. Every criterion "
                "must pass in all three lineages. These bounds are a transfer screen, not proof of "
                "equivalence. A good immediate response without the registered recovery contrast still "
                "fails this gate. Development and incomplete attempts have no scientific verdict.", "",
                "This measures saved-controller transfer, Phase 3(a). The formal rung remains undecided: "
                "it still requires registered evolution from blank across three seeds, a matched fair "
                "baseline comparison, and a mechanism lesion. The neutral selected birth state is exactly "
                "birth_parent and is reused, not another replicate. Only the birth_parent versus "
                "direct_baseline panel has matched birth information. Prepared-versus-direct scores "
                "have unequal prior information and cannot establish architecture superiority.", "",
                "| Run | Status | Split | Families | Primary late success ± SEM | Recovery contrast ± paired SEM | "
                "Late gap ± paired SEM | End-survival gap ± paired SEM | Transfer decision | Formal rung | Commit |",
                "|---" * 11 + "|", *random_reversal_gates, ""]
        if random_reversal_windows:
            out += ["All assigned families:", "",
                    "| Run | Split | Arm | World | Window | First food ± SEM | No contact ± SEM | Fitness ± SEM | "
                    "Contact accuracy ± SEM | Food count | Poison count | Accuracy denominator |",
                    "|---" * 12 + "|", *random_reversal_windows, "",
                    "| Run | Split | Arm | World | Lifetime fitness ± SEM | Survival at flip ± SEM | "
                    "Survival after 50 ± SEM | Survival after 100 ± SEM | End survival ± SEM |",
                    "|---" * 9 + "|", *random_reversal_survival, "",
                    "Paired reversal minus stationary differences:", "",
                    "| Run | Split | Arm | Pre first-food gap ± SEM | Early gap ± SEM | Late gap ± SEM | "
                    "Recovery contrast ± SEM | End-survival gap ± SEM |", "|---" * 8 + "|", *random_reversal_pairs, "",
                    "Matched comparisons: within_social is observed minus belief_reset; within_birth is "
                    "birth_parent minus direct_baseline. All ± values below are paired SEM.", "",
                    "| Run | Split | Comparison | World | Window | First-food gap ± SEM | Fitness gap ± SEM | "
                    "End-survival gap ± SEM |", "|---" * 8 + "|", *random_reversal_controls, "",
                    "Secondary subgroups only. End-survivor selection cannot establish recovery or replace "
                    "the assigned-family gate; subgroup sizes are shown explicitly.", "",
                    "| Run | Split | Arm | World | Subgroup | Families | Pre first food ± SEM | Early ± SEM | Late ± SEM |",
                    "|---" * 9 + "|", *random_reversal_subgroups, ""]
    if social_channel_gates:
        out += ["## Information in public demonstrations (social-channel-v1)", "",
                "E121 uses fixed readers of 150 public frames after a teacher's 150 practice turns. "
                "The track reader uses consecutive visible teacher positions; the scene reader uses "
                "item-image motion only. Both predict the most frequently contacted visible item color "
                "as food. Ties and no votes earn expected accuracy of one half. Every family is retained. "
                "Means use independent families; all ± values are standard errors of the mean (SEM), "
                "and control contrasts use paired within-family differences. Fractions are proportions, "
                "while votes and identifiable steps are mean counts per family. The tie fraction "
                "includes movies with no votes.", "",
                "The channel gate uses only the track reader's observed arm: coverage at least 0.50, "
                "expected accuracy at least 0.60, accuracy minus chance (0.50) strictly greater than "
                "two family SEM, and observed-minus-yoked accuracy strictly greater than two paired SEM. "
                "All four criteria must pass separately in all three lineages. A pass shows this fixed "
                "reader can use the public stimulus; it does not show an agent learned socially. "
                "A failure does not establish that every possible reader would fail. The scene reader "
                "and other control comparisons are diagnostics. Development and incomplete attempts "
                "have no scientific verdict.", "",
                "| Run | Status | Split | Families | Coverage ≥ 0.50 | Accuracy ≥ 0.60 | Above chance > 2 SEM | "
                "Above yoked > 2 paired SEM | Lineage decision | Commit |", "|---" * 10 + "|",
                *social_channel_gates, ""]
        if social_channel_rows:
            out += ["| Run | Split | Reader | Movie | Expected accuracy ± SEM | Coverage ± SEM | "
                    "No-vote fraction ± SEM | Tie fraction ± SEM | Votes ± SEM | Identifiable steps ± SEM |",
                    "|---" * 10 + "|", *social_channel_rows, "",
                    "| Run | Split | Reader | Observed minus chance ± SEM | Observed minus yoked ± paired SEM | "
                    "Observed minus no demo ± paired SEM | Observed minus scene only ± paired SEM |",
                    "|---" * 7 + "|", *social_channel_contrasts, ""]
    if social_observer_rows:
        out += ["## Observer evolution with fixed teachers (social-observer-train-v1)", "",
                "E122 branch O attempts ten observer ES rounds from each original genome while its demonstrator "
                "stays fixed. Equal scores receive average ranks. All-tied rounds consume their worlds and "
                "advance the random stream while preserving genome, moments and optimizer-update count. The "
                "table reports informative updates separately from attempted rounds. Held-out gains are observed "
                "minus each comparator, in first-food probability with paired family standard errors. A lineage "
                "passes only if all four gains exceed two SE and every survival difference is at least minus "
                "two SE. All assigned families contribute, including no-meal outcomes. All three lineages must "
                "pass. Development and incomplete records have no scientific verdict.", "",
                "| Run | Status | Split | Attempted rounds | Informative updates | Families | Evolved observed ± SE | "
                "Original observed ± SE | Gain over yoked ± SE | Gain over no demo ± SE | Gain over scene only ± SE | "
                "Gain over original ± SE | Decision | Commit |",
                "|---" * 14 + "|", *social_observer_rows, ""]
    if social_learning_rows:
        out += ["## Observational learning and solo transfer (social-learning-v1)", "",
                "E120 tests one newborn per independent world family after 150 observation frames from "
                "an experienced teacher. Its solo test lasts 300 turns in a fresh world. The primary measure "
                "is first solo meal being food per assigned newborn; poison and no meal both count as zero. "
                "The observed movie is compared with yoked item colors, a time-matched blank movie and a "
                "scene-only movie. Each lineage must exceed every control by more than two paired family "
                "standard errors, with each all-assigned survival difference at least minus two SE. "
                "The table shows proportions and observed-minus-control differences. All three lineages "
                "must pass; development and incomplete attempts have no scientific verdict.", "",
                "| Run | Status | Split | Families | Observed first-food ± SE | Difference vs yoked ± SE | "
                "Difference vs no demo ± SE | Difference vs scene only ± SE | Seed decision | Commit |",
                "|---" * 10 + "|", *social_learning_rows, ""]
    if inherited_evolution_rows:
        out += ["## Evolution with inherited experience (inherited-evolution-v1)", "",
                "E119 continues each E118 optimizer for ten updates. Before each update, the current mean genome "
                "adds its training donors' acquired sensory-to-output changes to a bounded archive. All candidates "
                "then receive that archive. The final child with its latest archive is compared with its immediate "
                "parent and previous archive on 128 fresh paired worlds. Death and nonarrival count as 301 turns; "
                "inside births are excluded from latency and retained in survival. A seed passes when saved turns "
                "exceed two paired family-cluster standard errors and survival changes by at least minus two SE. "
                "All three seeds must pass. Additional controls hold the genome or archive fixed and remain in "
                "each summary. Development and incomplete attempts have no scientific decision.", "",
                "| Run | Status | Split | Additional generations | Families | Parent latency ± SE | "
                "Child latency ± SE | Latency reduction ± SE | Survival difference ± SE | Seed decision | Commit |",
                "|---" * 11 + "|", *inherited_evolution_rows, ""]
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
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote results/RESULTS.md with {n_runs} runs")


if __name__ == "__main__":
    main()
