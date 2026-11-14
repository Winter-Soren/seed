"""Tiny real-world integration and original-engine compatibility fixtures."""
import inspect
import json
from pathlib import Path
import subprocess
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from seed import seed as engine
from seed.structural import rewire, StructuralValidationError


class StructuralIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.c = engine.Cfg(G=4, M=1, T=22, R=0, H=3, n_items=2)
        self.g = engine.init_genome(self.c, jax.random.PRNGKey(7))
        self.keys = (jax.random.PRNGKey(3), jax.random.PRNGKey(5))

    def assert_tree_equal(self, a, b):
        self.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)

    def test_isolated_hidden_neuron_is_recruited_only_on_the_next_tick(self):
        c = self.c
        _, D, _, n, _ = engine.dims(c)
        mask = jnp.zeros((n, n))
        W = jnp.zeros_like(mask)
        x = jnp.zeros(n).at[D - 1].set(1.)
        obs = x[:D]
        Wp = jnp.zeros((D, c.H))
        hebb = jnp.zeros((n, n, 4))
        tick = lambda w, activity, m: engine.brain_step(w, activity, Wp, obs, 0., m, hebb, 0., 0., jnp.zeros(3), 1., c)
        W0, current_x, _, _ = tick(W, x, mask)
        self.assertEqual(float(current_x[D]), 0.)
        legal = (jnp.arange(n)[:, None] >= D) & ~jnp.eye(n, dtype=bool)
        caps = jnp.zeros(n, dtype=int).at[D].set(1)
        W1, m1, d = rewire(W0, mask, x, legal, caps, jnp.ones_like(W), jnp.ones(n), wmax=c.wmax)
        self.assertEqual((int(d['added']), int(d['removed'])), (1, 0))
        self.assertEqual(float(m1[D, D - 1]), 1.)
        self.assertEqual(float(current_x[D]), 0.)
        _, next_x, _, _ = tick(W1, current_x, m1)
        self.assertGreater(abs(float(next_x[D])), 0.)
        # A removed edge's zero weight must have no influence on later activity.
        W2, m2, d2 = rewire(W1 * 0.001, m1, jnp.zeros(n), legal, caps,
                            jnp.ones_like(W), jnp.ones(n), wmax=c.wmax)
        self.assertEqual(int(d2['removed']), 1)
        _, pruned_x, _, _ = tick(W2, x, m2)
        self.assertEqual(float(pruned_x[D]), 0.)
        print('DIRECT_CAPABILITY ' + json.dumps(dict(
            grown_edges=np.argwhere(np.asarray(m1 - mask) > 0).tolist(),
            new_weight=float(W1[D, D - 1]), current_activation=float(current_x[D]),
            next_activation=float(next_x[D]), growth_net=int(m1.sum() - mask.sum()),
            pruned_edges=np.argwhere(np.asarray(m1 - m2) > 0).tolist(),
            prune_net=int(m2.sum() - m1.sum()), pruned_weight=float(W2[D, D - 1]),
            pruned_activation=float(pruned_x[D]))))

    def test_disabled_rollout_matches_original_including_final_internal_state(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.run(['git', 'show', 'a0cb4a7:seed/seed.py'], cwd=root,
                                capture_output=True, text=True, check=True).stdout
        legacy = types.ModuleType('legacy_seed')
        exec(compile(source, 'legacy_seed.py', 'exec'), legacy.__dict__)

        def probe(source, namespace):
            source = source.replace('def rollout(', 'def probe_rollout(')
            source = source.replace('    return res\n',
                '    res["probe_state"] = dict(W=s["W"], x=s["x"], Wp=s["Wp"], key=s["key"], mask=mask)\n    return res\n')
            exec(compile(source, 'probe_rollout.py', 'exec'), namespace)
            return namespace['probe_rollout']

        old_source = source[source.index('@partial(jax.jit,'):source.index('def make_evaluator(')]
        old = probe(old_source, dict(legacy.__dict__))
        new = probe(inspect.getsource(engine.rollout), dict(engine.__dict__))
        for plastic in (False, True):
            a = old(self.g, *self.keys, 0, self.c, plastic, True)
            b = new(self.g, *self.keys, 0, self.c, plastic, True)
            self.assert_tree_equal(a, b)
            self.assert_tree_equal(b, new(self.g, *self.keys, 0, self.c, plastic, True, structural=False))
        mu = ravel_pytree(self.g)[0]
        self.assertEqual(legacy.test_genome(mu, self.c, n_worlds=2), engine.test_genome(mu, self.c, n_worlds=2))
        self.assertEqual(engine.genome_size(engine.Cfg()), 155)

    def test_enabled_rollout_repeats_exactly_and_tracks_real_counts(self):
        # Zero initial fan-in forces recruitment; no initial edge can be swapped.
        g = dict(self.g, conn=jnp.full_like(self.g['conn'], -100.))
        a = engine.rollout(g, *self.keys, 0, self.c, record=True, structural=True)
        b = engine.rollout(g, *self.keys, 0, self.c, record=True, structural=True)
        self.assert_tree_equal(a, b)
        stats = a['structure']
        self.assertEqual(float(stats['initial_edges']), 0.)
        self.assertGreater(float(stats['added']), 0.)
        self.assertEqual(float(stats['final_edges']), float(stats['added'] - stats['removed']))
        self.assertLessEqual(float(stats['max_edges']), float(stats['edge_cap']))
        for k in ('invalid_proposals', 'invalid_weights', 'cap_violations', 'illegal_edges'):
            self.assertEqual(float(stats[k]), 0.)
        self.assertEqual(float(a['structural_state']['mask'].sum()), float(stats['final_edges']))
        self.assertTrue(np.all(np.asarray(a['structural_state']['W'])[np.asarray(a['structural_state']['mask']) == 0] == 0))
        # The first event is step index 10, and uses previous activity.
        self.assertTrue(np.all(np.asarray(a['traj']['structural_added'])[:10] == 0))
        self.assertGreater(float(a['traj']['structural_added'][10]), 0.)

    def test_frozen_structural_mode_preserves_behavior_and_reports_no_changes(self):
        a = engine.rollout(self.g, *self.keys, 0, self.c, False, True)
        b = dict(engine.rollout(self.g, *self.keys, 0, self.c, False, True, structural=True))
        stats = b.pop('structure')
        b.pop('structural_state')
        b['traj'] = {k: v for k, v in b['traj'].items() if not k.startswith('structural_')}
        self.assert_tree_equal(a, b)
        self.assertEqual((float(stats['added']), float(stats['removed'])), (0., 0.))

    def test_dead_agents_never_rewire_and_nonfinite_scales_reject_evaluation(self):
        g = dict(self.g, conn=jnp.full_like(self.g['conn'], -100.))
        dead = engine.rollout(g, *self.keys, 0, self.c._replace(e0=0., n_items=0), structural=True)
        self.assertEqual((float(dead['structure']['added']), float(dead['structure']['removed'])), (0., 0.))
        invalid = dict(self.g, wscale=jnp.full_like(self.g['wscale'], 1000.))
        with self.assertRaisesRegex(StructuralValidationError, 'invalid_proposals|invalid_weights'):
            engine.test_genome(ravel_pytree(invalid)[0], self.c, n_worlds=1, structural=True)


if __name__ == '__main__':
    unittest.main()
