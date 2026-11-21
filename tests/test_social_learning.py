"""Development checks for the observation boundary and matched solo controls."""
import copy
from pathlib import Path
import subprocess
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from scripts import social_learning as social
from seed import seed


def teacher_fixture():
    c = seed.Cfg(G=6, M=1, T=6, R=1, H=3, n_items=1)
    grid = np.zeros((c.G, c.G), np.int32)
    grid[2, 2], grid[3, 2] = 1, 2
    raw = dict(teacher_initial_grid=grid, teacher_initial_position=np.array([[2, 2]], np.int32),
               teacher_grid=np.tile(grid, (c.T, 1, 1)),
               teacher_position=np.array([[[2, 2]], [[2, 3]], [[2, 4]], [[3, 4]], [[3, 4]], [[3, 4]]]),
               teacher_signal=np.full((c.T, 1), .5, np.float32),
               teacher_alive_before=np.array([[1], [1], [1], [1], [0], [0]], bool),
               teacher_energy=np.full((c.T, 1), .7, np.float32))
    return c, raw


class SocialLearningTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(value, reference)

    def test_movie_uses_previous_camera_and_hides_teacher_internal_rewards(self):
        c, raw = teacher_fixture()
        movie, camera = social.make_movie(raw, c)
        np.testing.assert_array_equal(camera, [[3, 3], [3, 4], [4, 4]])
        area = 9
        self.assertEqual(movie[0, 2 * area:3 * area].sum(), 1)
        np.testing.assert_array_equal(movie[1:, 2 * area:4 * area], 0)
        np.testing.assert_array_equal(movie[:, -3:], np.tile([0, 0, 1], (3, 1)))
        changed = copy.deepcopy(raw)
        changed["teacher_energy"][:] = 1000
        changed["teacher_reward"] = np.arange(c.T)
        changed["teacher_terminal_weights"] = np.ones((3, 3))
        self.assert_tree_equal(social.make_movie(changed, c), (movie, camera))

    def test_movie_controls_preserve_exposure_and_change_only_their_public_channels(self):
        c, raw = teacher_fixture()
        movie, _ = social.make_movie(raw, c)
        movie[0, 0], movie[0, 9] = 1, 0
        arms = social.movie_conditions(movie, True, c)
        np.testing.assert_array_equal(arms["yoked"][:, :9], movie[:, 9:18])
        np.testing.assert_array_equal(arms["yoked"][:, 9:18], movie[:, :9])
        np.testing.assert_array_equal(arms["yoked"][:, 18:], movie[:, 18:])
        np.testing.assert_array_equal(arms["scene_only"][:, :18], movie[:, :18])
        np.testing.assert_array_equal(arms["scene_only"][:, 18:36], 0)
        np.testing.assert_array_equal(arms["no_demo"][:, :36], 0)
        np.testing.assert_array_equal(arms["no_demo"][:, -3:], movie[:, -3:])
        np.testing.assert_array_equal(social.movie_conditions(movie, False, c)["yoked"], movie)

    def test_watch_learns_in_own_brain_with_stay_action_and_zero_reward(self):
        c, raw = teacher_fixture()
        movie, _ = social.make_movie(raw, c)
        genome = seed.biology_prior(c)
        flat, _ = ravel_pytree(genome)
        result = jax.tree.map(np.asarray, social.watch(flat, jax.random.PRNGKey(47), movie, c))
        np.testing.assert_array_equal(result["trace"]["motor"], np.tile([1, 0, 0, 0, 0], (3, 1)))
        np.testing.assert_array_equal(result["trace"]["denergy"], 0)
        np.testing.assert_array_equal(result["initial"]["x"], 0)
        np.testing.assert_array_equal(result["initial"]["Wp"], 0)
        self.assertTrue(np.any(result["final"]["W"] != result["initial"]["W"]))
        self.assertTrue(np.all(result["final"]["W"][result["mask"] == 0] == 0))

    def test_optional_hook_preserves_exact_legacy_trajectory(self):
        source = subprocess.check_output(["git", "show", "1e864e1:seed/seed.py"],
                                         cwd=Path(__file__).resolve().parents[1], text=True, timeout=10)
        old = types.ModuleType("before_social_learning")
        exec(compile(source, "before_social_learning.py", "exec"), old.__dict__)
        c = seed.Cfg(G=4, M=1, T=4, R=1, H=3, n_items=2)
        wk, dk = jax.random.PRNGKey(47), jax.random.PRNGKey(48)
        g = seed.biology_prior(c)
        kwargs = dict(record=True, diagnostics=True, _return_weights=True)
        expected = old.rollout(g, wk, dk, 0, c, **kwargs)
        self.assert_tree_equal(seed.rollout(g, wk, dk, 0, c, **kwargs), expected)
        self.assert_tree_equal(seed.rollout(g, wk, dk, 0, c, _initial_brain=None, **kwargs), expected)

    def test_neural_continuation_preserves_own_state_and_resets_body(self):
        c = seed.Cfg(G=4, M=1, T=4, R=1, H=3, n_items=2)
        wk, dk = jax.random.PRNGKey(47), jax.random.PRNGKey(48)
        g = seed.biology_prior(c)
        _, inputs, _, neurons, _ = seed.dims(c)
        weights = jnp.zeros((1, neurons, neurons))
        brain = dict(x=jnp.full((1, neurons), .2), Wp=jnp.full((1, inputs, c.H), .1))
        result = seed.rollout(g, wk, dk, 1, c, diagnostics=True, _return_weights=True,
                              _initial_weights=weights, _initial_brain=brain)
        np.testing.assert_array_equal(result["_weight_state"]["initial_W"], weights)
        for name, value in brain.items():
            np.testing.assert_array_equal(result["_initial_brain"][name], value)
        np.testing.assert_array_equal(result["_initial_brain"]["energy"], [c.e0])
        np.testing.assert_array_equal(result["_initial_brain"]["alive"], [1])
        np.testing.assert_array_equal(result["_initial_brain"]["denergy"], [0])
        with self.assertRaises(ValueError):
            seed.rollout(g, wk, dk, 0, c, _initial_brain=brain)
        with self.assertRaises(ValueError):
            seed.rollout(g, wk, dk, 0, c, _initial_weights=weights,
                          _initial_brain=dict(x=brain["x"], Wp=jnp.zeros((1, 2, 3))))

    def test_first_meal_and_gate_keep_missed_meals_and_strict_boundaries(self):
        food = np.zeros((4, 1), bool)
        poison = food.copy()
        result = social.first_meal(food, poison)
        self.assertTrue(result["no_meal"])
        self.assertEqual(result["first_meal_step"], 5)
        poison[1] = True
        food[2] = True
        self.assertTrue(social.first_meal(food, poison)["first_poison"])
        self.assertFalse(social._gate(dict(mean=.2, se=.1), dict(mean=0., se=0.))["passed"])
        self.assertTrue(social._gate(dict(mean=.201, se=.1), dict(mean=-.2, se=.1))["passed"])
        self.assertFalse(social._gate(dict(mean=.201, se=.1), dict(mean=-.201, se=.1))["passed"])

    def test_development_family_has_four_matched_solo_conditions_and_validates_corruption(self):
        c = seed.Cfg(G=6, M=1, T=6, R=1, H=3, n_items=3)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        row = social.evaluate_family(flat, social.family_keys(47, 0), 0, c)
        social.validate_family(row, c)
        for event in social.EVENTS:
            np.testing.assert_array_equal(row["observed_" + event], row["yoked_" + event])
        for arm in social.ARMS:
            for name in ("W", "x", "Wp"):
                np.testing.assert_array_equal(row[arm + "_watch_final_" + name + "_sha256"],
                                              row[arm + "_solo_initial_" + name + "_sha256"])
        report = social.summarize([row])
        self.assertIsNone(report["gate"]["passed"])
        self.assertEqual(report["assigned_newborns"], 1)
        for key in ("observed_frames", "observed_watch_motor", "observed_solo_initial_energy",
                    "observed_solo_initial_W_sha256"):
            damaged = copy.deepcopy(row)
            damaged[key].flat[0] += 1
            with self.assertRaises(ValueError, msg=key):
                social.validate_family(damaged, c)
        with self.assertRaises(ValueError):
            social.summarize([row, row])


if __name__ == "__main__":
    unittest.main()
