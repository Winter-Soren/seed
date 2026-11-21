"""E116 version boundaries, fixed gate, and summary presentation; seed 41 only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import jax
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import eval_v4, eval_v5, eval_v6 as panel, summarize
from seed import seed
from tests.test_eval_v5 import synthetic_panels


def candidate_panels():
    rename = {f"{old}{i}": f"{new}{i}" for i in range(3) for old, new in (("mean", "control"), ("min", "candidate"))}
    return {rename.get(label, label): raw for label, raw in synthetic_panels().items()}


def historical_module(path):
    source = subprocess.run(["git", "show", "27dfb88f25185c891e6bac57d2fcd86079ea65a0:"+path],
                            capture_output=True, check=True, timeout=15).stdout
    module = types.ModuleType("historical_"+Path(path).stem)
    module.__file__ = str(Path(path).resolve())
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class LearningRateGateTests(unittest.TestCase):
    def test_candidate_contrasts_have_registered_names_and_hand_calculated_paired_errors(self):
        result = panel.compare_panels(candidate_panels())
        self.assertEqual(len(result["contrasts"]), 30)
        self.assertEqual({row["comparison"] for row in result["contrasts"]},
                         {"candidate-minus-control", "candidate-minus-start"})
        self.assertTrue(result["gate"]["passed"])
        for row in result["contrasts"]:
            self.assertEqual(row["minuend"], f"candidate{row['seed']}")
            if row["metric"] == "individual_min_return":
                expected = (4., 1., [3., 5.]) if row["comparison"] == "candidate-minus-control" else (5., 1., [4., 6.])
                self.assertEqual((row["mean"], row["se"], row["per_world_difference"]), expected)
        for row in result["gate"]["per_seed"]:
            self.assertEqual(set(row), {"seed", "passed", "candidate-minus-control_minimum_return",
                "candidate-minus-start_minimum_return", "candidate-minus-start_joint_survival"})
        with self.assertRaisesRegex(ValueError, "14 registered"):
            panel.compare_panels(synthetic_panels())

    def test_each_seed_start_safeguard_and_exact_gate_boundaries_remain_required(self):
        raw = candidate_panels()
        raw["candidate2"]["paired_individual_min_return"] = np.array([2., 6.])
        gate = panel.compare_panels(raw)["gate"]
        self.assertEqual([row["passed"] for row in gate["per_seed"]], [True, True, False])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["per_seed"][2]["candidate-minus-control_minimum_return"]["threshold"], 2.)
        raw = candidate_panels()
        raw["control1"]["paired_individual_min_return"] = np.array([-5., -3.])
        raw["candidate1"]["paired_individual_min_return"] = np.array([-1., 1.])
        gate = panel.compare_panels(raw)["gate"]
        self.assertTrue(gate["per_seed"][1]["candidate-minus-control_minimum_return"]["passed"])
        self.assertFalse(gate["per_seed"][1]["candidate-minus-start_minimum_return"]["passed"])
        self.assertFalse(gate["passed"])
        raw = candidate_panels()
        raw["candidate0"]["paired_joint_survival"] = np.array([0., .5])
        self.assertTrue(panel.compare_panels(raw)["gate"]["passed"])
        raw["candidate0"]["paired_joint_survival"] = np.array([.5, .5])
        gate = panel.compare_panels(raw)["gate"]
        self.assertTrue(gate["all_minimum_improvements"])
        self.assertFalse(gate["no_joint_survival_regression"])
        self.assertFalse(gate["passed"])

    def test_missing_reference_or_unpaired_worlds_cannot_produce_gate(self):
        raw = candidate_panels()
        raw.pop("greedy")
        with self.assertRaisesRegex(ValueError, "14 registered"):
            panel.compare_panels(raw)
        raw = candidate_panels()
        raw["candidate0"]["initial_pos"] = np.ones((2, 2, 2))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            panel.compare_panels(raw)

    def test_v5_contrast_values_schema_and_decision_are_exact_against_committed_source(self):
        historical = historical_module("scripts/eval_v5.py")
        for failure in (False, True):
            raw = synthetic_panels()
            if failure:
                raw["min2"]["paired_individual_min_return"] = np.array([2., 6.])
            self.assertEqual(eval_v5.compare_panels(raw), historical.compare_panels(raw))


class EvalVersionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = seed.Cfg(G=2, M=2, T=4, R=0, H=3, n_items=1)

    def test_final_default_and_whitelists_are_fixed_without_evaluating_heldout(self):
        self.assertEqual(panel.SPLIT_SEEDS, {"development": 41, "final": 2026112001})
        self.assertEqual(panel.PANELS[:9], ("start0", "control0", "candidate0", "start1", "control1",
                                         "candidate1", "start2", "control2", "candidate2"))
        with patch.object(eval_v4, "_evaluate_controller", return_value="sentinel") as execute:
            self.assertEqual(panel.evaluate_controller(None, seed.Cfg(), controller="random_walk"), "sentinel")
            self.assertEqual(execute.call_args.args[2:], (256, 2026112001))
        cases = ((panel, eval_v4.EVAL_SEED), (panel, eval_v5.EVAL_SEED),
                 (panel, 42), (eval_v4, panel.EVAL_SEED), (eval_v5, panel.EVAL_SEED))
        for module, invalid in cases:
            with self.subTest(version=module.PROTOCOL, seed=invalid), patch.object(eval_v4, "_evaluate_controller") as execute:
                with self.assertRaisesRegex(ValueError, "seed"):
                    module.evaluate_controller(None, self.cfg, 2, invalid, controller="random_walk")
                execute.assert_not_called()
        with patch.object(eval_v4, "_evaluate_controller") as execute:
            with self.assertRaisesRegex(ValueError, "256 base worlds"):
                panel.evaluate_controller(None, self.cfg, 2, panel.EVAL_SEED, controller="random_walk")
            with self.assertRaisesRegex(ValueError, "registered controller"):
                panel.evaluate_controller(None, self.cfg, 2, 41, controller="direct-rnn-v1")
            with self.assertRaisesRegex(ValueError, "tiny fixtures"):
                panel.evaluate_controller(None, self.cfg, 5, 41, controller="random_walk")
            execute.assert_not_called()

    def test_shared_execution_and_callbacks_match_immutable_v5_in_tiny_lives(self):
        historical = historical_module("scripts/eval_v5.py")
        genome = ravel_pytree(seed.init_genome(self.cfg, jax.random.PRNGKey(41)))[0]
        for controller in ("rules-v0", "mixture"):
            mu = genome if controller == "rules-v0" else None
            expected, expected_raw = historical.evaluate_controller(mu, self.cfg, 2, 41, controller=controller)
            for module in (eval_v5, panel):
                seen = []
                actual, raw = module.evaluate_controller(mu, self.cfg, 2, 41, controller=controller,
                    on_condition=lambda label, metrics, arrays: seen.append((label, metrics, arrays)))
                self.assertEqual(actual, expected)
                self.assertEqual(set(raw), set(expected_raw))
                for key in raw:
                    np.testing.assert_array_equal(raw[key], expected_raw[key])
                self.assertEqual([item[0] for item in seen], list(panel.CONDITIONS))
                for label, metrics, arrays in seen:
                    self.assertEqual(metrics, expected["conditions"][label])
                    for key, value in arrays.items():
                        np.testing.assert_array_equal(value, expected_raw[key])


class SummaryVersionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="seed-eval-v6-summary-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def add_run(self, name, args, summary=None):
        folder = self.root/"runs"/name
        folder.mkdir(parents=True)
        config = dict(args=args, git=dict(commit="a"*40, dirty=False), evaluation_status="deferred")
        (folder/"config.json").write_text(json.dumps(config), encoding="utf-8")
        if summary is not None:
            (folder/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return folder

    def render(self, module=summarize):
        with patch.object(module, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        return (self.root/"results/RESULTS.md").read_text()

    def complete_summary(self, module, passed):
        return dict(status="finished", artifact_validity="passed", gate=dict(passed=passed), completed_panels=14,
            expected_panels=14, eval_seed=module.EVAL_SEED, eval_worlds=256,
            panels={label: dict(status="finished", controller="rules-v0" if label not in module.REFERENCES else label,
                conditions={condition: {} for condition in module.CONDITIONS},
                paired={metric: dict(mean=1., se=.125) for metric in module.CONTRAST_METRICS}) for label in module.PANELS})

    def test_old_summary_unchanged_and_new_training_rates_and_protocol_are_explicit(self):
        self.add_run("old_E115-v5", dict(eval_protocol="v5", condition_split="final"), self.complete_summary(eval_v5, False))
        self.assertEqual(self.render(), self.render(historical_module("scripts/summarize.py")))
        for arm, rate in (("control", .1), ("candidate", .02)):
            self.add_run("new_E116-"+arm+"-s0", dict(eval_protocol="paired-train-v2", objective="paired-min",
                condition_split="final", seed=2000, generations=20, pop=64, worlds=4, sigma=.1, lr=rate),
                dict(status="finished", generation=20, evaluation_status="deferred"))
        self.add_run("new_E116-v6", dict(eval_protocol="v6", condition_split="final"), self.complete_summary(panel, True))
        text = self.render()
        self.assertIn("## EVAL v5 paired objective comparison", text)
        self.assertIn("| old_E115-v5 | final | finished | FAIL | aaaaaaa |", text)
        self.assertIn("## EVAL v6 paired learning-rate comparison", text)
        self.assertIn("| new_E116-v6 | final | finished | PASS | aaaaaaa |", text)
        self.assertIn("| new_E116-control-s0 | finished | paired-min | final | 2000 | 20/20 | 64 | 4 | 0.1 | 0.1 | deferred |", text)
        self.assertIn("| new_E116-candidate-s0 | finished | paired-min | final | 2000 | 20/20 | 64 | 4 | 0.1 | 0.02 | deferred |", text)
        self.assertIn("1.000000 ± 0.125000", text)
        self.assertIn("2026112001 | 256", text)

    def test_config_only_partial_or_invalid_v6_cannot_be_displayed_as_decided(self):
        self.add_run("E116-no-summary", dict(eval_protocol="v6", condition_split="final"))
        valid = self.complete_summary(panel, True)
        incomplete = copy.deepcopy(valid)
        incomplete["status"] = "interrupted"
        incomplete["completed_panels"] = 13
        incomplete["panels"].pop("greedy")
        self.add_run("E116-interrupted", dict(eval_protocol="v6", condition_split="final"), incomplete)
        incomplete["status"] = "finished"
        self.add_run("E116-missing-panel", dict(eval_protocol="v6", condition_split="final"), incomplete)
        invalid = copy.deepcopy(valid)
        invalid["artifact_validity"] = "failed"
        self.add_run("E116-invalid", dict(eval_protocol="v6", condition_split="final"), invalid)
        missing_condition = copy.deepcopy(valid)
        missing_condition["panels"]["candidate0"]["conditions"].pop("reversal_r1")
        self.add_run("E116-missing-condition", dict(eval_protocol="v6", condition_split="final"), missing_condition)
        text = self.render()
        for name, status in (("E116-no-summary", "no summary"), ("E116-interrupted", "interrupted"),
                             ("E116-missing-panel", "finished"), ("E116-invalid", "finished"),
                             ("E116-missing-condition", "finished")):
            self.assertIn(f"| {name} | final | {status} | undecided |", text)
        self.assertIn("undefined", text)


if __name__ == "__main__":
    unittest.main()
