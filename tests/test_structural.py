"""Direct capability fixtures, not evidence of a fitness advantage."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from seed.structural import birth_caps, rewire, validate_structure


class RewiringTests(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.legal = (jnp.arange(self.n)[:, None] >= 2) & ~jnp.eye(self.n, dtype=bool)
        self.mask = jnp.zeros((self.n, self.n))
        self.weights = jnp.zeros_like(self.mask)
        self.scale = jnp.ones_like(self.mask)
        self.draws = jnp.array([1., 1., -0.5, 0.25, 0.5])

    def tick(self, weights=None, mask=None, activity=None, caps=None, **kwargs):
        mask = self.mask if mask is None else mask
        return rewire(self.weights if weights is None else weights, mask,
                      jnp.array([1., 0., 0., 0., 0.]) if activity is None else activity,
                      self.legal, birth_caps(mask, self.legal) if caps is None else caps,
                      self.scale, self.draws, wmax=3., **kwargs)

    def assert_invariants(self, weights, mask, caps):
        m, w = np.asarray(mask), np.asarray(weights)
        self.assertTrue(np.isin(m, [0, 1]).all())
        self.assertFalse(np.any(m[~np.asarray(self.legal)]))
        self.assertTrue(np.all(m.sum(1) <= np.asarray(caps)))
        self.assertLessEqual(m.sum(), np.asarray(caps).sum())
        self.assertTrue(np.all(w[m == 0] == 0))
        self.assertTrue(np.isfinite(w).all())
        self.assertLessEqual(np.abs(w).max(), 3.)

    def test_growth_without_pruning_recruits_an_empty_row(self):
        caps = jnp.array([0, 0, 1, 0, 0])
        w, m, d = self.tick(caps=caps)
        self.assertEqual(np.asarray(m).sum(), 1)
        self.assertEqual(float(m[2, 0]), 1.)
        self.assertEqual(float(w[2, 0]), -0.5)
        self.assertEqual(int(d['added']), 1)
        self.assertEqual(int(d['removed']), 0)
        self.assert_invariants(w, m, caps)

    def test_pruning_without_growth_zeros_the_weight(self):
        m = self.mask.at[2, 0].set(1)
        w = self.weights.at[2, 0].set(0.01)
        w1, m1, d = self.tick(w, m, jnp.zeros(self.n))
        self.assertEqual(float(m1.sum()), 0.)
        self.assertEqual(float(w1[2, 0]), 0.)
        self.assertEqual((int(d['added']), int(d['removed'])), (0, 1))
        self.assert_invariants(w1, m1, birth_caps(m, self.legal))

    def test_pruning_ties_use_lowest_presynaptic_index(self):
        m = self.mask.at[2, 0].set(1).at[2, 1].set(1)
        w = m * 0.01
        w1, m1, d = self.tick(w, m, jnp.zeros(self.n))
        self.assertEqual((float(m1[2, 0]), float(m1[2, 1])), (0., 1.))
        self.assertEqual((float(w1[2, 0]), float(w1[2, 1])), (0., float(w[2, 1])))
        self.assertEqual(int(d['removed']), 1)

    def test_activity_changes_selected_growth_edge_and_ties_use_lowest_index(self):
        caps = jnp.array([0, 0, 1, 0, 0])
        for x, column in (([1, 0, 0, 0, 0], 0), ([0, 1, 0, 0, 0], 1), ([1, 1, 0, 0, 0], 0)):
            with self.subTest(activity=x):
                w, m, _ = self.tick(activity=jnp.array(x, dtype=float), caps=caps)
                self.assertEqual(np.argwhere(np.asarray(m)).tolist(), [[2, column]])
                self.assertNotEqual(float(w[2, column]), 0.)

    def test_nonempty_row_uses_coactivity_and_does_not_readd_just_pruned_edge(self):
        m = self.mask.at[2, 0].set(1)
        w = self.weights.at[2, 0].set(0.001)
        _, m1, d = self.tick(w, m, jnp.array([1., 0., 0., 0., 0.]), caps=jnp.array([0, 0, 2, 0, 0]))
        self.assertEqual(float(m1.sum()), 0.)
        self.assertEqual((int(d['added']), int(d['removed'])), (0, 1))
        _, m2, _ = self.tick(w, m, jnp.array([1., 0.5, 1., 0., 0.]), caps=jnp.array([0, 0, 2, 0, 0]))
        self.assertEqual(float(m2[2, 1]), 1.)

    def test_fixed_birth_caps_do_not_drift_over_repeated_events(self):
        caps = birth_caps(self.mask, self.legal)
        w, m = self.weights, self.mask
        for _ in range(12):
            w, m, _ = self.tick(w, m, jnp.ones(self.n), caps=caps)
            self.assert_invariants(w, m, caps)
        np.testing.assert_array_equal(np.asarray(m).sum(1), [0, 0, 1, 1, 1])

    def test_full_rows_self_edges_and_sensor_rows_have_no_candidates(self):
        m = self.legal.astype(float)
        w = m * 0.5
        w1, m1, d = self.tick(w, m, jnp.ones(self.n))
        np.testing.assert_array_equal(m1, m)
        np.testing.assert_array_equal(w1, w)
        self.assertEqual((int(d['added']), int(d['removed'])), (0, 0))
        caps = birth_caps(m, self.legal)
        np.testing.assert_array_equal(caps, [0, 0, 4, 4, 4])

    def test_dead_frozen_disabled_and_nonevent_are_exact_noops(self):
        m = self.mask.at[2, 0].set(1)
        w = self.weights.at[2, 0].set(0.001)
        for flags in ({'alive': False}, {'plastic': False}, {'enabled': False}, {'event': False}):
            with self.subTest(flags=flags):
                w1, m1, d = self.tick(w, m, jnp.ones(self.n), **flags)
                np.testing.assert_array_equal(w1, w)
                np.testing.assert_array_equal(m1, m)
                self.assertEqual((int(d['added']), int(d['removed'])), (0, 0))

    def test_zero_draw_skips_growth_nonfinite_draw_is_a_validation_failure(self):
        self.draws = jnp.zeros(self.n)
        _, m, d = self.tick()
        self.assertEqual(float(m.sum()), 0.)
        validate_structure(d)
        for value in (jnp.nan, jnp.inf):
            self.draws = jnp.full(self.n, value)
            _, _, d = self.tick()
            with self.assertRaisesRegex(ValueError, 'invalid_proposals'):
                validate_structure(d)
        self.draws = jnp.ones(self.n)
        self.scale = self.scale.at[2, 0].set(jnp.inf)
        _, _, d = self.tick()
        with self.assertRaisesRegex(ValueError, 'invalid_proposals'):
            validate_structure(d)

    def test_new_weights_are_clipped_and_counters_equal_actual_mask_differences(self):
        self.draws = jnp.full(self.n, 1e6)
        m = self.mask.at[2, 1].set(1)
        w = self.weights.at[2, 1].set(0.001)
        w1, m1, d = self.tick(w, m, jnp.array([1., 0., 0., 0., 0.]))
        self.assertEqual(int(d['added']), int(jnp.sum((m1 > 0) & (m == 0))))
        self.assertEqual(int(d['removed']), int(jnp.sum((m > 0) & (m1 == 0))))
        self.assertEqual((int(d['added']), int(d['removed'])), (3, 1))
        self.assert_invariants(w1, m1, birth_caps(m, self.legal))

    def test_jit_and_eager_are_identical(self):
        args = (self.weights, self.mask, jnp.ones(self.n), self.legal,
                birth_caps(self.mask, self.legal), self.scale, self.draws)
        eager = rewire(*args, wmax=3.)
        compiled = jax.jit(lambda *a: rewire(*a, wmax=3.))(*args)
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
