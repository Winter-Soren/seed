"""Synthetic E131 policy and native compatibility tests; no final data is read."""
import subprocess
import types
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed, exploration, timed_exploration as study

HISTORICAL_COMMIT = "6ed8cb06b93f8c2f0d43a6414754043872088cb3"
SYNTHETIC_MASTER = 0xE131F0


def same_tree(test, actual, expected, *, extra=False):
    if isinstance(expected, dict):
        test.assertTrue(set(expected) <= set(actual) if extra else set(expected) == set(actual))
        for name in expected:
            same_tree(test, actual[name], expected[name], extra=extra)
    else:
        actual, expected = np.asarray(actual), np.asarray(expected)
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.tobytes(), expected.tobytes())


def historical_engine():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(["git", "show", HISTORICAL_COMMIT + ":seed/seed.py"], cwd=root).decode("utf-8")
    module = types.ModuleType("historical_e130_seed")
    module.__file__ = HISTORICAL_COMMIT + ":seed/seed.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class ExplorationPolicyTests(unittest.TestCase):
    def test_phase_counter_quota_and_partial_blocks(self):
        schedule = dict(phase_delay=jnp.asarray([0, 5, 24, 3], jnp.int32),
            directions=jnp.tile(jnp.asarray([[1, 2, 3, 4]], jnp.int32), (3, 1)),
            quota_offsets=jnp.tile(jnp.asarray([[7, 12, 24, 0]], jnp.int32), (3, 1)))
        age = jnp.arange(75, dtype=jnp.int32)
        observed = jnp.zeros((75, 4), jnp.float32).at[:, 1].set(.01)
        alive = jnp.ones((75, 4), jnp.float32).at[10:, 3].set(0)

        def replay(mode):
            def tick(state, inputs):
                t, feedback, living = inputs
                action, state, trace = exploration.choose_action(mode, state, schedule, t, feedback, living,
                                                                  jnp.zeros(4, jnp.int32))
                return state, dict(trace, action=action)
            return jax.jit(lambda: jax.lax.scan(tick, exploration.initial_state(4), (age, observed, alive)))()[1]

        adaptive, quota = replay("adaptive"), replay("random_quota")
        for animal, expected in enumerate(([12, 25, 50], [29, 54], [24, 49, 74], [])):
            np.testing.assert_array_equal(np.flatnonzero(np.asarray(adaptive["override"])[:, animal]), expected)
        for animal, expected in enumerate(([7, 32, 57], [17, 42, 67], [48, 73], [3])):
            np.testing.assert_array_equal(np.flatnonzero(np.asarray(quota["override"])[:, animal]), expected)
        self.assertEqual(int(adaptive["counter"][0, 0]), 0)
        self.assertEqual(int(adaptive["counter"][11, 0]), 11)
        self.assertEqual(int(adaptive["counter"][12, 0]), 12)
        np.testing.assert_array_equal(np.asarray(adaptive["counter"])[10:, 3], 9)
        # The phase-delayed final block of animal1 is partial: there is no
        # artificial final-horizon pulse when its counter stays zero.
        self.assertFalse(bool(np.asarray(adaptive["override"])[74, 1]))
        for trace in (adaptive, quota):
            np.testing.assert_array_equal(trace["action"], np.where(trace["override"], schedule["directions"][0], 0))

    def test_positive_realized_feedback_resets_but_zero_does_not(self):
        schedule = dict(phase_delay=jnp.zeros(1, jnp.int32), directions=jnp.ones((2, 1), jnp.int32),
                        quota_offsets=jnp.zeros((2, 1), jnp.int32))
        state = dict(counter=jnp.asarray([11], jnp.int32), spent=jnp.asarray([False]))
        decide = jax.jit(lambda feedback: exploration.choose_action("adaptive", state, schedule, jnp.int32(12),
            feedback, jnp.ones(1), jnp.zeros(1, jnp.int32)))
        zero, positive = decide(jnp.asarray([0.], jnp.float32)), decide(jnp.asarray([.001], jnp.float32))
        self.assertEqual(int(zero[1]["counter"][0]), 12)
        self.assertTrue(bool(zero[2]["override"][0]))
        self.assertEqual(int(positive[1]["counter"][0]), 0)
        self.assertFalse(bool(positive[2]["override"][0]))
        # A food label is not an argument. A clipped zero gain follows the zero
        # path; a small truly positive realized gain follows the reset path.

    def test_preassigned_streams_are_reproducible_and_separate(self):
        keys = study.evaluation_keys(SYNTHETIC_MASTER, 0)
        first = exploration.make_schedule(keys[2], 4, 300)
        second = exploration.make_schedule(keys[2], 4, 300)
        same_tree(self, first, second)
        self.assertEqual(len({key.tobytes() for key in first["stream_keys"]}), 3)
        for name, low, high in (("phase_delay", 0, 25), ("directions", 1, 5), ("quota_offsets", 0, 25)):
            self.assertTrue(np.all((first[name] >= low) & (first[name] < high)))
        self.assertEqual(first["directions"].shape, (12, 4))
        self.assertEqual(first["phase_delay"].shape, (4,))
        self.assertFalse(set(key.tobytes() for key in keys[:2]) & set(key.tobytes() for key in first["stream_keys"]))


class ExplorationNativeTests(unittest.TestCase):
    def test_disabled_and_observed_baseline_match_complete_e130_trajectory(self):
        old = historical_engine()
        c, genome = seed.Cfg(), seed.biology_prior(seed.Cfg())
        world, birth, _ = study.evaluation_keys(SYNTHETIC_MASTER, 0)
        args = (genome, jnp.asarray(world), jnp.asarray(birth), jnp.int32(0), c)
        options = dict(plastic=True, predictor_plastic=True, record=True, diagnostics=True,
                       flip_step=150, _world_protocol="reversal-random-v1")
        expected = old.rollout(*args, **options)
        disabled = seed.rollout(*args, **options)
        observed = seed.rollout(*args, **options, _exploration_record=True)
        # Includes every grid/body/neural/observation/action array and terminal
        # W/x/Wp returned by the historical engine, on the same runtime.
        same_tree(self, disabled, expected)
        same_tree(self, observed, expected, extra=True)
        proof = observed["exploration"]
        np.testing.assert_array_equal(proof["override"], False)
        np.testing.assert_array_equal(proof["action"], proof["ordinary_action"])
        np.testing.assert_array_equal(proof["incoming_efference"][1:], proof["efference"][:-1])

    def test_active_policy_keeps_real_energy_feedback_and_executed_efference(self):
        c = seed.Cfg(T=50)
        genome = seed.biology_prior(c)
        keys = study.evaluation_keys(SYNTHETIC_MASTER, 1)
        proof = exploration.make_schedule(keys[2], c.M, c.T)
        schedule = {name: jnp.asarray(proof[name]) for name in ("phase_delay", "directions", "quota_offsets")}
        for mode in exploration.MODES:
            raw = seed.rollout(genome, jnp.asarray(keys[0]), jnp.asarray(keys[1]), jnp.int32(0), c,
                plastic=True, predictor_plastic=True, record=False, diagnostics=True,
                exploration=mode, _exploration_schedule=schedule, _exploration_record=True)
            trace = jax.tree.map(np.asarray, raw["exploration"])
            before = np.asarray(raw["events"]["alive_before"])
            same_tree(self, trace["observed_denergy"][1:], trace["denergy"][:-1])
            same_tree(self, trace["denergy"], (trace["energy_after"] - trace["energy_before"]) * before)
            same_tree(self, trace["efference"], np.eye(5, dtype=np.float32)[trace["action"]])
            same_tree(self, trace["incoming_efference"][1:], trace["efference"][:-1])
            same_tree(self, trace["action"], np.where(trace["override"], trace["direction"], trace["ordinary_action"]))
            self.assertFalse(np.any(trace["override"] & ~(before > 0)))
            self.assertTrue(np.all((trace["energy_after"] >= 0) & (trace["energy_after"] <= c.e_max)))
            self.assertTrue(np.array_equal(np.clip(trace["unclipped_energy"], 0., c.e_max), trace["energy_after"]))


if __name__ == "__main__":
    unittest.main()
