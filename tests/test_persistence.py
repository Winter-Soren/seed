"""Tiny mechanism checks for the optional persistence rule, not performance claims."""
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from seed import seed
from seed.ecology import EcologyV1
from seed.persistence import PersistenceV1, bias_logits, initial_memory


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.trait = PersistenceV1()
        self.logits = jnp.array([[.2, -.3, .4, -.5, .6]])
        self.memory = initial_memory(1)
        self.empty = jnp.zeros((1, 5, 5), dtype=bool)

    def decide(self, shelter, memory=None, *, cold=True, alive=True, trait=None):
        return bias_logits(self.logits, shelter, jnp.array([cold]), jnp.array([alive]),
                           self.memory if memory is None else memory,
                           self.trait if trait is None else trait, seed.MOVES)

    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)

    def test_config_and_legacy_rejection(self):
        self.assertEqual(self.trait.version, "persistence-v1")
        for kwargs in ({"memory_steps": 0}, {"memory_steps": 17}, {"memory_steps": True},
                       {"memory_steps": 1.5}, {"stall_steps": 0}, {"stall_steps": False},
                       {"stall_steps": 5}, {"stall_steps": "2"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PersistenceV1(**kwargs)
        c = seed.Cfg(G=4, M=1, T=4, R=1, H=3, n_items=0)
        key = jax.random.PRNGKey(41)
        genome = seed.init_genome(c, key)
        for ecology, trait in ((None, self.trait), (EcologyV1(), True), (True, self.trait)):
            with self.subTest(ecology=ecology, trait=trait), self.assertRaises(ValueError):
                seed.rollout(genome, key, key, 0, c, ecology=ecology, persistence=trait)
        with self.assertRaises(ValueError):
            seed.make_evaluator(c, 2, persistence=self.trait)
        with self.assertRaises(TypeError):
            seed.test_genome(jnp.zeros(155), c, persistence=self.trait)

    def test_off_preserves_previous_legacy_and_ecology_trajectories(self):
        source = subprocess.check_output(
            ["git", "show", "5d31e5b:seed/seed.py"],
            cwd=Path(__file__).resolve().parents[1], text=True, timeout=10)
        previous = types.ModuleType("before_persistence")
        exec(compile(source, "before_persistence.py", "exec"), previous.__dict__)
        c = seed.Cfg(G=4, M=1, T=4, R=1, H=3, n_items=1)
        wk, dk = jax.random.PRNGKey(41), jax.random.PRNGKey(42)
        g = seed.init_genome(c, dk)
        for ecology in (None, EcologyV1(.01, .02)):
            kwargs = dict(record=True, diagnostics=True, ecology=ecology)
            expected = previous.rollout(g, wk, dk, 0, c, **kwargs)
            self.assert_tree_equal(seed.rollout(g, wk, dk, 0, c, **kwargs), expected)
            self.assert_tree_equal(seed.rollout(g, wk, dk, 0, c, persistence=None, **kwargs), expected)

    def test_recent_goal_survives_cue_loss_using_only_own_action(self):
        east = self.empty.at[0, 2, 4].set(True)
        logits, memory, events = self.decide(east)
        self.assertTrue(bool(events["active"][0]))
        self.assertFalse(bool(events["remembered"][0]))
        np.testing.assert_allclose(logits - self.logits, [[0, 0, 0, 0, 1]])
        memory["last_action"] = jnp.array([4])
        logits, memory, events = self.decide(self.empty, memory)
        np.testing.assert_array_equal(memory["offset"], [[0, 1]])
        self.assertTrue(bool(events["remembered"][0]))
        self.assertFalse(bool(events["progress_observed"][0]))
        self.assertFalse(bool(events["stalled"][0]))
        np.testing.assert_allclose(logits - self.logits, [[0, 0, 0, 0, 1]])

    def test_unseen_goal_expires_after_bounded_memory(self):
        _, memory, _ = self.decide(self.empty.at[0, 0, 2].set(True))
        for _ in range(self.trait.memory_steps):
            _, memory, events = self.decide(self.empty, memory)
            self.assertTrue(bool(events["remembered"][0]))
        logits, memory, events = self.decide(self.empty, memory)
        self.assertFalse(bool(events["active"][0]))
        np.testing.assert_array_equal(logits, self.logits)
        self.assertEqual(int(memory["remaining"][0]), 0)
        np.testing.assert_array_equal(memory["offset"], [[0, 0]])

    def test_visible_stall_reweights_tactic_with_bounded_attempts(self):
        north = self.empty.at[0, 1, 2].set(True)
        _, memory, _ = self.decide(north)
        attempts = 0
        for _ in range(10):
            memory["last_action"] = jnp.array([1])
            logits, memory, events = self.decide(north, memory)
            self.assertTrue(bool(events["stalled"][0]))
            self.assertFalse(bool(events["progress_observed"][0]))
            if bool(events["tactic_attempted"][0]):
                attempts += 1
                self.assertLess(float(logits[0, 1]), float(self.logits[0, 1]))
                self.assertTrue(bool(jnp.any(logits[0, 3:] > self.logits[0, 3:])))
        self.assertEqual(attempts, 2)
        _, memory, _ = self.decide(north)
        for _ in range(self.trait.stall_steps):
            logits, memory, events = self.decide(north, memory)
        self.assertTrue(bool(events["tactic_attempted"][0]))
        self.assertLess(float(logits[0, 0]), float(self.logits[0, 0]))
        self.assertGreater(float(logits[0, 1]), float(self.logits[0, 1]))
        _, memory, _ = self.decide(self.empty.at[0, 0, 2].set(True))
        _, memory, events = self.decide(north, memory)
        self.assertTrue(bool(events["progress_observed"][0]))
        self.assertFalse(bool(events["stalled"][0]))
        self.assertEqual(int(memory["stalls"][0]), 0)

    def test_warm_sheltered_dead_and_no_goal_keep_policy_and_clear_memory(self):
        north = self.empty.at[0, 1, 2].set(True)
        _, remembered, _ = self.decide(north)
        for shelter, kwargs in ((north, {"cold": False}), (north, {"alive": False}),
                                (north.at[0, 2, 2].set(True), {})):
            logits, memory, events = self.decide(shelter, remembered, **kwargs)
            np.testing.assert_array_equal(logits, self.logits)
            self.assert_tree_equal(memory, initial_memory(1))
            self.assertTrue(all(not bool(value[0]) for value in events.values()))
        logits, _, events = self.decide(self.empty)
        np.testing.assert_array_equal(logits, self.logits)
        self.assertFalse(bool(events["active"][0]))

    def test_rollout_records_attempt_separately_from_actual_switch(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=0, decay=0.)
        wk, dk = jax.random.PRNGKey(41), jax.random.PRNGKey(42)
        g = seed.init_genome(c, dk)
        g = dict(g, conn=jnp.full_like(g["conn"], -100.))
        grid, pos = jnp.zeros((4, 4), jnp.int32), jnp.array([[2, 2]])
        shelter = jnp.zeros((4, 4), dtype=bool).at[1, 2].set(True)
        for final_action in (0, 1):
            actions = [jnp.array([0])] * 5 + [jnp.array([final_action])]
            with (jax.disable_jit(), patch.object(seed, "init_world", return_value=(grid, pos)),
                  patch.object(seed, "make_shelter", return_value=shelter),
                  patch.object(jax.random, "categorical", side_effect=actions) as choose):
                result = seed.rollout(g, wk, dk, 0, c, ecology=EcologyV1(),
                                      persistence=self.trait, record=True, diagnostics=True)
            events = result["persistence"]
            np.testing.assert_array_equal(events["active"][:, 0], [False] * 3 + [True] * 3)
            np.testing.assert_array_equal(events["tactic_attempted"][:, 0], [False] * 5 + [True])
            self.assertEqual(int(events["previous_action"][-1, 0]), 0)
            self.assertEqual(int(events["selected_action"][-1, 0]), final_action)
            self.assertEqual(bool(events["action_switched"][-1, 0]), final_action != 0)
            np.testing.assert_array_equal(choose.call_args_list[-1].args[1], [[-1., 1., 0., 0., 0.]])
            np.testing.assert_array_equal(result["traj"]["energy"], np.ones((6, 1)))
        dead = seed.rollout(g, wk, dk, 0, c, ecology=EcologyV1(), persistence=self.trait,
                            record=True, diagnostics=True, _initial_alive=jnp.zeros(1))
        np.testing.assert_array_equal(dead["traj"]["alive"], np.zeros((6, 1)))
        np.testing.assert_array_equal(dead["traj"]["energy"], np.zeros((6, 1)))
        for name, values in dead["persistence"].items():
            if name not in ("previous_action", "selected_action"):
                self.assertFalse(np.asarray(values).any())


if __name__ == "__main__":
    unittest.main()
