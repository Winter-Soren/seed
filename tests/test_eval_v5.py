"""Hand-computed gate checks and seed-41 fixtures; never run held-out worlds."""
import subprocess
import types
import unittest
from unittest.mock import patch

import jax
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import eval_v4, eval_v5 as panel
from seed import seed


def synthetic_panels():
    """Two paired worlds with known differences, SEs and survival rates."""
    result = {}
    for label in panel.PANELS:
        raw = dict(wkeys=np.array([[1, 2], [3, 4]], np.uint32),
                   dkeys=np.array([[5, 6], [7, 8]], np.uint32),
                   initial_grid=np.zeros((2, 2, 2), np.int32),
                   initial_pos=np.zeros((2, 2, 2), np.int32),
                   agent_ids=np.array([[0, 1], [0, 1]], np.int32))
        for metric in panel.CONTRAST_METRICS:
            raw["paired_" + metric] = np.zeros(2)
        raw["paired_joint_survival"] = np.array([.75, .75])
        raw["paired_reversal_final_survival"] = np.array([.25, .25])
        for arm, values in (("start", [0., 2.]), ("mean", [1., 3.]), ("min", [4., 8.])):
            if label.startswith(arm):
                raw["paired_individual_min_return"] = np.array(values)
        result[label] = raw
    return result


class ContinuationGateTests(unittest.TestCase):
    def test_thirty_fixed_contrasts_and_all_seed_gate_use_paired_world_se(self):
        result = panel.compare_panels(synthetic_panels())
        self.assertEqual(len(result["contrasts"]), 30)
        self.assertEqual({r["metric"] for r in result["contrasts"]}, set(panel.CONTRAST_METRICS))
        self.assertEqual({r["comparison"] for r in result["contrasts"]},
                         {"min-minus-mean", "min-minus-start"})
        self.assertTrue(result["gate"]["passed"])
        self.assertTrue(result["gate"]["all_minimum_improvements"])
        self.assertTrue(result["gate"]["no_joint_survival_regression"])
        self.assertEqual(len(result["gate"]["per_seed"]), 3)
        for row in result["contrasts"]:
            if row["metric"] != "individual_min_return":
                self.assertEqual((row["mean"], row["se"]), (0., 0.))
                continue
            if row["comparison"] == "min-minus-mean":
                self.assertEqual((row["mean"], row["se"]), (4., 1.))
                self.assertEqual(row["per_world_difference"], [3., 5.])
            else:
                self.assertEqual((row["mean"], row["se"]), (5., 1.))
                self.assertEqual(row["per_world_difference"], [4., 6.])
            self.assertEqual(row["n_worlds"], 2)
            self.assertEqual(row["minuend"], f"min{row['seed']}")
        for row in result["gate"]["per_seed"]:
            self.assertTrue(row["passed"])
            self.assertEqual(row["min-minus-mean_minimum_return"]["threshold"], 2.)

    def test_one_seed_at_two_se_fails_strict_improvement_despite_other_two_passing(self):
        raw = synthetic_panels()
        raw["min2"]["paired_individual_min_return"] = np.array([2., 6.])
        gate = panel.compare_panels(raw)["gate"]
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_minimum_improvements"])
        self.assertTrue(gate["no_joint_survival_regression"])
        self.assertEqual([row["passed"] for row in gate["per_seed"]], [True, True, False])
        component = gate["per_seed"][2]["min-minus-mean_minimum_return"]
        self.assertEqual((component["mean"], component["se"], component["threshold"]), (2., 1., 2.))
        self.assertFalse(component["passed"])

    def test_zero_change_with_zero_se_does_not_pass(self):
        raw = synthetic_panels()
        raw["min0"]["paired_individual_min_return"] = raw["mean0"]["paired_individual_min_return"].copy()
        component = panel.compare_panels(raw)["gate"]["per_seed"][0]["min-minus-mean_minimum_return"]
        self.assertEqual((component["mean"], component["se"]), (0., 0.))
        self.assertFalse(component["passed"])

    def test_degraded_mean_control_cannot_hide_decline_from_unchanged_start(self):
        raw = synthetic_panels()
        raw["mean1"]["paired_individual_min_return"] = np.array([-5., -3.])
        raw["min1"]["paired_individual_min_return"] = np.array([-1., 1.])
        gate = panel.compare_panels(raw)["gate"]
        self.assertFalse(gate["passed"])
        component = gate["per_seed"][1]
        self.assertTrue(component["min-minus-mean_minimum_return"]["passed"])
        self.assertEqual(component["min-minus-start_minimum_return"]["mean"], -1.)
        self.assertFalse(component["min-minus-start_minimum_return"]["passed"])

    def test_survival_regression_blocks_positive_return_improvements(self):
        raw = synthetic_panels()
        raw["min1"]["paired_joint_survival"] = np.array([.5, .5])
        gate = panel.compare_panels(raw)["gate"]
        self.assertTrue(gate["all_minimum_improvements"])
        self.assertFalse(gate["no_joint_survival_regression"])
        self.assertFalse(gate["passed"])
        component = gate["per_seed"][1]["min-minus-start_joint_survival"]
        self.assertEqual((component["mean"], component["se"]), (-.25, 0.))

    def test_survival_boundary_at_minus_two_se_is_allowed(self):
        raw = synthetic_panels()
        raw["min0"]["paired_joint_survival"] = np.array([0., .5])
        gate = panel.compare_panels(raw)["gate"]
        component = gate["per_seed"][0]["min-minus-start_joint_survival"]
        self.assertEqual((component["mean"], component["se"], component["threshold"]), (-.5, .25, -.5))
        self.assertTrue(component["passed"])
        self.assertTrue(gate["passed"])

    def test_incomplete_unpaired_nonfinite_or_wrong_length_panels_cannot_decide_gate(self):
        mutations = (
            lambda raw: raw.pop("greedy"),
            lambda raw: raw.update(unregistered=raw["start0"]),
            lambda raw: raw["fixed_A"].update(initial_pos=np.ones((2, 2, 2))),
            lambda raw: raw["min2"].pop("dkeys"),
            lambda raw: raw["start0"].pop("wkeys"),
            lambda raw: raw["min0"].update(paired_individual_min_return=np.array([np.nan, 4.])),
            lambda raw: raw["greedy"].update(paired_joint_survival=np.zeros(3)),
            lambda raw: raw["mean0"].pop("paired_post_flip_return"),
        )
        for index, mutate in enumerate(mutations):
            raw = synthetic_panels()
            mutate(raw)
            with self.subTest(case=index), self.assertRaises(ValueError):
                panel.compare_panels(raw)


class VersionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.c = seed.Cfg(G=2, M=2, T=4, R=0, H=3, n_items=1)

    def test_public_versions_keep_distinct_final_defaults_and_seed_whitelists(self):
        for module, expected in ((eval_v4, 2026111902), (panel, 2026111903)):
            self.assertEqual(module.EVAL_SEED, expected)
            with patch.object(eval_v4, "_evaluate_controller", return_value="sentinel") as execute:
                self.assertEqual(module.evaluate_controller(None, seed.Cfg(), controller="random_walk"), "sentinel")
                self.assertEqual(execute.call_args.args[2:], (256, expected))
            for invalid in (2026111901, 42, panel.EVAL_SEED if module is eval_v4 else eval_v4.EVAL_SEED):
                with self.subTest(protocol=module.PROTOCOL, seed=invalid):
                    with patch.object(eval_v4, "_evaluate_controller") as execute:
                        with self.assertRaisesRegex(ValueError, "seed"):
                            module.evaluate_controller(None, self.c, 2, invalid, controller="random_walk")
                        execute.assert_not_called()
            with patch.object(eval_v4, "_evaluate_controller") as execute:
                with self.assertRaisesRegex(ValueError, "256 base worlds"):
                    module.evaluate_controller(None, self.c, 2, expected, controller="random_walk")
                execute.assert_not_called()

    def test_v5_rejects_unregistered_controller_and_large_development_worlds(self):
        invalid = ((self.c, 2, "direct-rnn-v1"), (self.c, 5, "random_walk"),
                   (self.c._replace(T=32), 2, "random_walk"))
        with patch.object(eval_v4, "_evaluate_controller") as execute:
            for c, worlds, controller in invalid:
                with self.subTest(controller=controller), self.assertRaises(ValueError):
                    panel.evaluate_controller(None, c, worlds, 41, controller=controller)
            execute.assert_not_called()

    def test_v5_and_refactored_v4_match_immutable_v4_on_tiny_development_lives(self):
        source = subprocess.run(["git", "show",
            "5596c1a7746069548bc533931834ead9e3904b97:scripts/eval_v4.py"],
            capture_output=True, check=True).stdout
        historical = types.ModuleType("historical_eval_v4")
        exec(compile(source, "historical_eval_v4.py", "exec"), historical.__dict__)
        genome = ravel_pytree(seed.init_genome(self.c, jax.random.PRNGKey(41)))[0]
        for controller in panel.CONTROLLERS:
            mu = genome if controller == "rules-v0" else None
            expected, expected_raw = historical.evaluate_controller(mu, self.c, 2, 41, controller=controller)
            for module in (eval_v4, panel):
                seen = []
                actual, raw = module.evaluate_controller(mu, self.c, 2, 41, controller=controller,
                    on_condition=lambda label, summary, arrays: seen.append((label, summary, arrays)))
                self.assertEqual(actual, expected)
                self.assertEqual(set(raw), set(expected_raw))
                for key in expected_raw:
                    np.testing.assert_array_equal(raw[key], expected_raw[key])
                self.assertEqual([x[0] for x in seen], list(panel.CONDITIONS))
                for label, summary, arrays in seen:
                    self.assertEqual(summary, actual["conditions"][label])
                    for key, value in arrays.items():
                        np.testing.assert_array_equal(value, expected_raw[key])


if __name__ == "__main__":
    unittest.main()
