"""EVAL v3 fixtures use development seeds only; no held-out worlds are inspected."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import eval_v3 as panel
from seed import seed


def events(worlds=2, steps=20, agents=2):
    shape = (worlds, steps, agents)
    return dict(food=np.zeros(shape, bool), poison=np.zeros(shape, bool),
                alive_before=np.ones(shape, bool), alive_after=np.ones(shape, bool))


class MeasurementTests(unittest.TestCase):
    def test_world_standard_error_and_paired_ratio(self):
        self.assertEqual(panel.mean_se([1, 3]), dict(mean=2., se=1.))
        self.assertEqual(panel.ratio_se([1, 3], [2, 2]), dict(mean=1., se=.5))
        self.assertEqual(panel.ratio_se([0, 0], [0, 0]), dict(mean=None, se=None))
        self.assertIsNone(panel.mean_se([1])["se"])
        for invalid in ([np.nan, 1], [], [[1, 2]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                panel.mean_se(invalid)
        with self.assertRaises(ValueError):
            panel.ratio_se([1, 2], [1])

    def test_meals_are_per_agent_with_censoring_and_matched_gain(self):
        raw = events()
        raw["poison"][0, 0, 0] = True
        raw["food"][0, 1:, 0] = True
        raw["food"][1, 0, 0] = True
        raw["poison"][1, 1:, 0] = True
        raw["food"][1, 0, 1] = True
        result = panel.summarize_events(raw)
        first, last = result["meal_accuracy"][0], result["meal_accuracy"][-1]
        self.assertEqual((first["eligible_agents"], first["accuracy"]), (3, 2/3))
        self.assertEqual((last["eligible_agents"], last["accuracy"]), (2, .5))
        self.assertEqual(result["meal20_minus_meal1_matched"],
                         dict(difference=0., se=1., eligible_agents=2, contributing_worlds=2))
        self.assertEqual((result["fitness"], result["fitness_se"]), (.25, 8.75))
        self.assertEqual(result["eaten"], 41)

    def test_survivor_bins_exclude_dead_agents_in_every_bin(self):
        raw = events(steps=100)
        raw["food"][:, 20, :] = True
        raw["poison"][:, 60, 0] = True
        raw["alive_after"][:, 49:, 1] = False
        raw["alive_before"][:, 50:, 1] = False
        result = panel.summarize_events(raw)
        self.assertEqual(result["bins"][0]["food"], 4)
        self.assertEqual(result["survivor_bins"][0]["food"], 2)
        self.assertEqual(result["survivor_bins"][1]["poison"], 2)
        self.assertEqual(result["survived"], .5)
        self.assertEqual([(b["start"], b["stop"]) for b in result["bins"]], [(0, 50), (50, 100)])
        self.assertTrue(all(b["complete"] for b in result["survivor_bins"]))

    def test_absent_meals_and_survivors_are_missing_not_zero(self):
        raw = events(steps=2)
        raw["alive_after"][:, 0:] = False
        raw["alive_before"][:, 1:] = False
        result = panel.summarize_events(raw)
        self.assertIsNone(result["meal_accuracy"][0]["accuracy"])
        self.assertIsNone(result["survivor_bins"][0]["accuracy_se"])
        self.assertIsNone(result["meal20_minus_meal1_matched"]["difference"])
        self.assertEqual(result["fitness_se"], 0.)

    def test_invalid_survival_and_eating_events_are_rejected(self):
        raw = events()
        raw["food"][0, 0, 0] = raw["poison"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            panel.summarize_events(raw)
        raw = events()
        raw["alive_after"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            panel.summarize_events(raw)

    def test_final_request_cannot_change_registered_settings(self):
        good = dict(generations=0, start="file:genome.npy", resume=None, quick=False,
                    no_plastic=False, structural_plasticity="off", eval_worlds=256)
        panel.validate_request(SimpleNamespace(**good), seed.Cfg())
        for key, value in (("generations", 1), ("start", "blank"), ("resume", "old"),
                           ("quick", True), ("no_plastic", True), ("eval_worlds", 128),
                           ("structural_plasticity", "on"), ("condition_split", "final"),
                           ("cfg", ["T=300"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                panel.validate_request(SimpleNamespace(**dict(good, **{key: value})), seed.Cfg())
        with self.assertRaises(ValueError):
            panel.validate_request(SimpleNamespace(**good), seed.Cfg(T=10))


class ReferencePolicyTests(unittest.TestCase):
    def test_random_policy_matches_engine_with_zero_logits(self):
        c = seed.Cfg(G=4, M=2, T=12, R=1, H=3, n_items=3)
        g = seed.init_genome(c, jax.random.PRNGKey(42))
        g = dict(g, conn=jnp.full_like(g["conn"], -100.))
        wkey, dkey = jax.random.PRNGKey(41), jax.random.PRNGKey(42)
        for flip in (None, 6):
            for rule in (0, 1):
                original = seed.rollout(g, wkey, dkey, rule, c, plastic=False,
                                        predictor_plastic=False, diagnostics=True, flip_step=flip)
                reference = panel.reference_rollout(wkey, rule, c, policy="random_walk", flip_step=flip)
                np.testing.assert_array_equal(original["fitness"], reference["fitness"])
                for key in panel.EVENT_KEYS:
                    np.testing.assert_array_equal(original["events"][key], reference["events"][key],
                                                  err_msg=f"{flip} {rule} {key}")

    def test_greedy_moves_towards_food_avoids_poison_and_uses_current_rule(self):
        c = seed.Cfg(G=5, M=1, R=1, H=3)
        grid = jnp.zeros((5, 5), jnp.int32).at[1, 2].set(1).at[2, 3].set(2)
        obs = seed.observe(grid, jnp.array([[2, 2]]), jnp.ones(1), jnp.zeros(1),
                           jnp.ones(1), jnp.zeros(1), c)
        logits = np.asarray(panel.greedy_logits(obs, 0, c))[0]
        np.testing.assert_array_equal(np.isfinite(logits), [False, True, False, False, False])
        logits = np.asarray(panel.greedy_logits(obs, 1, c))[0]
        np.testing.assert_array_equal(np.isfinite(logits), [False, False, False, False, True])
        obs = jnp.zeros_like(obs)
        np.testing.assert_array_equal(panel.greedy_logits(obs, 0, c), np.zeros((1, 5)))

    def test_panel_raw_worlds_support_every_reported_fitness(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.init_genome(c, jax.random.PRNGKey(42)))
        summary, raw = panel.evaluate_conditions(flat, c, n_worlds=2, eval_seed=41)
        self.assertEqual(set(summary), {"stationary", "reversal"})
        np.testing.assert_array_equal(raw["rules"], [0, 1])
        for condition, controllers in summary.items():
            self.assertEqual(set(controllers), {"genome", "random_walk", "greedy"})
            for controller, metrics in controllers.items():
                prefix = f"{condition}_{controller}_"
                estimate = panel.mean_se(raw[prefix + "fitness"])
                self.assertEqual(metrics["fitness"], estimate["mean"])
                self.assertEqual(metrics["fitness_se"], estimate["se"])
                self.assertEqual(raw[prefix + "food"].shape, (2, 6, 1))
            if controllers["greedy"]["fitness"] != 0:
                self.assertEqual(controllers["greedy"]["greedy_fraction"], 1.)
                self.assertEqual(controllers["greedy"]["greedy_fraction_se"], 0.)
        for controller in ("genome", "random_walk", "greedy"):
            for key in panel.EVENT_KEYS:
                np.testing.assert_array_equal(raw[f"stationary_{controller}_{key}"][:, :3],
                                              raw[f"reversal_{controller}_{key}"][:, :3])
        with self.assertRaisesRegex(ValueError, "reserved"):
            panel.evaluate_conditions(flat, c, n_worlds=2, eval_seed=panel.EVAL_SEED)


if __name__ == "__main__":
    unittest.main()
