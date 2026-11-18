"""Small development fixtures for the fixed-weight direct recurrent controller."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from seed import direct, seed


class DirectEncodingTests(unittest.TestCase):
    def test_default_encoding_has_exactly_8424_channel_specific_weights(self):
        c = seed.Cfg()
        self.assertEqual(direct.FORMAT, "direct-rnn-v1")
        self.assertEqual(direct.genome_size(c), 8424)
        flat = jnp.arange(1, 8425, dtype=jnp.float32)
        weights = np.asarray(direct.decode(flat, c))
        _, sensors, _, neurons, _ = seed.dims(c)
        self.assertEqual(weights.shape, (157, 157))
        np.testing.assert_array_equal(weights[:sensors], 0.)
        np.testing.assert_array_equal(weights.diagonal(), 0.)
        legal = (np.arange(neurons)[:, None] >= sensors) & ~np.eye(neurons, dtype=bool)
        np.testing.assert_array_equal(weights[legal], flat)
        self.assertNotEqual(weights[sensors, 0], weights[sensors, 25])

    def test_initializer_uses_exact_registered_fanin_scale(self):
        c = seed.Cfg(G=3, M=1, T=4, R=0, H=3, n_items=1)
        key = jax.random.PRNGKey(41)
        expected = jax.random.normal(key, (direct.genome_size(c),)) / jnp.sqrt(float(seed.dims(c)[3]-1))
        np.testing.assert_array_equal(direct.init_genome(c, key), expected)
        with self.assertRaisesRegex(ValueError, "registered size"):
            direct.decode(jnp.zeros(155), c)
        with self.assertRaisesRegex(ValueError, "floating-point"):
            direct.decode(jnp.zeros(direct.genome_size(c), dtype=jnp.int32), c)

    def test_es_keeps_training_world_streams_aligned_with_rule_encoding(self):
        c = seed.Cfg(G=3, M=1, T=4, R=0, H=3, n_items=1)
        _, k0 = jax.random.split(jax.random.PRNGKey(41))
        state = seed.es_init(c, direct.init_genome(c, k0), seed=41)
        old = seed.es_init(c, seed=41)
        np.testing.assert_array_equal(state['key'], old['key'])
        seen = []

        def evaluate(population, worlds, development):
            seen.append((np.asarray(worlds), np.asarray(development)))
            pop, count = len(population), len(worlds)
            return dict(fitness=jnp.zeros((pop, count)), counts=jnp.ones((pop, count, 2, 2)),
                        survived=jnp.ones((pop, count)))

        direct_next, _ = seed.es_step(state, evaluate, pop=4, E=2, sigma=.1, lr=.1)
        old_next, _ = seed.es_step(old, evaluate, pop=4, E=2, sigma=.1, lr=.1)
        for a, b in zip(seen[0], seen[1]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(direct_next['key'], old_next['key'])


class DirectLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.c = seed.Cfg(G=3, M=2, T=6, R=0, H=3, n_items=2)
        self.wkey = jax.random.PRNGKey(41)
        self.dkey = jax.random.PRNGKey(42)

    def run_direct(self, weights, *, c=None, dkey=None, **kwargs):
        return seed.rollout(None, self.wkey, self.dkey if dkey is None else dkey, 0,
                            self.c if c is None else c, plastic=False, predictor_plastic=False,
                            direct_weights=weights, record=True, diagnostics=True, **kwargs)

    def test_weights_above_plastic_clip_bound_are_exactly_fixed(self):
        flat = direct.init_genome(self.c, self.dkey).at[0].set(8.)
        weights = direct.decode(flat, self.c)
        result = self.run_direct(weights)
        expected = np.broadcast_to(np.asarray(weights), (self.c.M, *weights.shape))
        np.testing.assert_array_equal(result['direct_state']['W'], expected)
        self.assertEqual(float(result['direct_state']['W'].max()), 8.)
        np.testing.assert_array_equal(result['direct_state']['Wp'], 0.)
        self.assertTrue(np.isfinite(np.asarray(result['direct_state']['x'])).all())

    def test_development_key_cannot_change_a_direct_controller(self):
        weights = direct.decode(direct.init_genome(self.c, self.dkey), self.c)
        a = self.run_direct(weights)
        b = self.run_direct(weights, dkey=self.wkey)
        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(left, right)

    def test_frozen_rule_controller_with_same_weights_is_exactly_equivalent(self):
        c = self.c._replace(M=1)
        g = seed.biology_prior(c)
        weights, _ = seed.develop(g, jax.random.split(self.dkey, 1)[0], c)
        self.assertLess(float(jnp.abs(weights).max()), c.wmax)
        for flip in (None, c.T//2):
            existing = seed.rollout(g, self.wkey, self.dkey, 0, c, plastic=False,
                                    predictor_plastic=False, record=True, diagnostics=True, flip_step=flip)
            direct_result = self.run_direct(weights, c=c, flip_step=flip)
            for key in existing:
                for left, right in zip(jax.tree.leaves(existing[key]), jax.tree.leaves(direct_result[key])):
                    np.testing.assert_array_equal(left, right, err_msg=key)

    def test_sensor_a_and_b_can_produce_opposite_responses(self):
        c = seed.Cfg(G=1, M=1, T=2, R=0, H=3, n_items=0, decay=0.)
        n = seed.dims(c)[3]
        weights = jnp.zeros((n, n)).at[n-1, 0].set(2.).at[n-1, 1].set(-2.)
        signals = []
        for item in (1, 2):
            fixed = (jnp.full((1, 1), item, dtype=jnp.int32), jnp.zeros((1, 2), dtype=jnp.int32))
            with patch.object(seed, 'init_world', return_value=fixed):
                # Clear this fixture's trace so its patched initial world is not reused.
                seed.rollout.clear_cache()
                result = self.run_direct(weights, c=c)
                signals.append(float(result['traj']['sig'][0, 0]))
        seed.rollout.clear_cache()
        self.assertGreater(signals[0], .9)
        self.assertLess(signals[1], -.9)

    def test_recurrent_state_retains_history_under_identical_current_observations(self):
        c = seed.Cfg(G=1, M=1, T=3, R=0, H=3, n_items=0, micro=1)
        _, sensors, _, n, _ = seed.dims(c)
        h0, h1, signal = sensors, sensors+1, n-1
        weights = (jnp.zeros((n, n)).at[h0, 0].set(1.).at[h1, h0].set(1.)
                   .at[signal, h1].set(1.))

        def observation(grid, pos, alive, sig, energy, denergy, cfg):
            return jnp.zeros((cfg.M, sensors)).at[:, 0].set(energy == cfg.e0)

        with patch.object(seed, 'observe', side_effect=observation):
            seed.rollout.clear_cache()
            result = self.run_direct(weights, c=c)
        seed.rollout.clear_cache()
        signal_trace = np.asarray(result['traj']['sig'])[:, 0]
        self.assertEqual(signal_trace[1], 0.)
        self.assertGreater(signal_trace[2], .5)
        np.testing.assert_array_equal(result['direct_state']['W'][0], weights)

    def test_learning_modes_and_wrong_shapes_are_rejected(self):
        n = seed.dims(self.c)[3]
        weights = jnp.zeros((n, n))
        for kwargs in (dict(plastic=True), dict(plastic=False, predictor_plastic=True),
                       dict(plastic=False, structural=True)):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'frozen'):
                seed.rollout(None, self.wkey, self.dkey, 0, self.c, direct_weights=weights, **kwargs)
        with self.assertRaisesRegex(ValueError, 'matrix'):
            self.run_direct(jnp.zeros((self.c.M, n, n)))

    def test_population_evaluator_reuses_world_rollout(self):
        flat = direct.init_genome(self.c, self.dkey)
        evaluate, decode = direct.make_evaluator(self.c, 2)
        worlds = jax.random.split(self.wkey, 2)
        development = jax.random.split(self.dkey, 2)[None]
        result = evaluate(flat[None], worlds, development)
        for index in range(2):
            expected = seed.rollout(None, worlds[index], development[0, index], index,
                                    self.c, plastic=False, predictor_plastic=False, direct_weights=decode(flat))
            for key, value in expected.items():
                np.testing.assert_array_equal(result[key][0, index], value, err_msg=key)


if __name__ == '__main__':
    unittest.main()
