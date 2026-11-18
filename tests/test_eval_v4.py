"""Synthetic and tiny seed-41 checks; never execute the EVAL v4 held-out seed."""
import subprocess
import types
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import eval_v3, eval_v4 as panel
from seed import direct, seed


def empty_events(worlds=2, steps=4, agents=2):
    shape = (worlds, steps, agents)
    return dict(food=np.zeros(shape, bool), poison=np.zeros(shape, bool),
                alive_before=np.ones(shape, bool), alive_after=np.ones(shape, bool))


class PairedAccountingTests(unittest.TestCase):
    def synthetic_pairs(self):
        raw = {}
        for label in panel.CONDITIONS:
            r = np.array([[10., -4.], [10., -4.]]) if label.endswith('r0') else np.array([[-4., 10.], [-4., 10.]])
            raw[label+'_agent_return'] = r
            raw[label+'_final_alive'] = r > 0
            raw[label+'_food'] = np.zeros((2, 4, 2), bool)
            raw[label+'_poison'] = np.zeros((2, 4, 2), bool)
        return raw

    def test_individual_minimum_precedes_averaging_and_prevents_mixture_masking(self):
        raw = self.synthetic_pairs()
        metrics, arrays = panel.summarize_pairs(raw)
        self.assertEqual(metrics['stationary_fitness'], dict(mean=3., se=0.))
        self.assertEqual(metrics['individual_min_return']['mean'], -4.)
        self.assertEqual(metrics['individual_min_return']['se'], 0.)
        self.assertEqual(metrics['joint_survival'], dict(mean=0., se=0.))
        np.testing.assert_array_equal(arrays['paired_individual_minimum'], -4.)
        self.assertEqual(metrics['individual_min_return']['distribution'], dict(values=[-4.], counts=[4]))
        # Taking the minimum after the family mean would incorrectly return +3.
        self.assertEqual(np.minimum(raw['stationary_r0_agent_return'].mean(1),
                                    raw['stationary_r1_agent_return'].mean(1)).mean(), 3.)

    def test_two_rule_mean_clusters_by_base_world_and_postflip_excludes_prefix(self):
        raw = self.synthetic_pairs()
        for label in panel.CONDITIONS:
            raw[label+'_agent_return'] = np.array([[1., 1.], [3., 3.]])
            raw[label+'_final_alive'] = np.ones((2, 2), bool)
            raw[label+'_food'][:, :2] = True
        raw['reversal_r0_poison'][:, 2:] = True
        metrics, arrays = panel.summarize_pairs(raw)
        self.assertEqual(metrics['stationary_fitness'], dict(mean=2., se=1.))
        self.assertEqual(metrics['post_flip_return'], dict(mean=-1., se=0.))
        self.assertEqual(metrics['reversal_final_survival'], dict(mean=1., se=0.))
        self.assertEqual(arrays['paired_stationary_fitness'].shape, (2,))

    def test_missingness_deaths_and_survival_boundary_use_all_initial_agents(self):
        raw = empty_events()
        raw['poison'][0, 1, 0] = True
        raw['alive_after'][0, 1:, 0] = False
        raw['alive_before'][0, 2:, 0] = False
        summary, individuals = panel.summarize_condition(raw, 2)
        self.assertEqual(summary['deaths'], 1)
        self.assertEqual(summary['zero_encounter_agents'], 3)
        self.assertEqual(summary['survival_at_flip'], .75)
        self.assertEqual(summary['survival_at_flip_se'], .25)
        self.assertEqual(individuals['death_step'][0, 0], 1)
        self.assertEqual(individuals['death_step'][0, 1], -1)
        self.assertEqual(summary['bins'][0]['zero_encounter_agents'], 3)
        self.assertIsNone(summary['survivor_bins'][0]['accuracy'])
        raw['alive_before'][1, 0, 0] = False
        raw['alive_after'][1, :, 0] = False
        raw['alive_before'][1, :, 0] = False
        with self.assertRaisesRegex(ValueError, 'starting'):
            panel.summarize_condition(raw, 2)

    def test_fixed_fifteen_contrasts_use_world_pairing(self):
        pairs = {}
        for i in range(3):
            for label, values in ((f'A{i}', [1., 3.]), (f'direct{i}', [0., 0.])):
                pairs[label] = {key: np.zeros((2, 1)) for key in panel.PAIRING_KEYS}
                pairs[label].update({'paired_'+metric: np.array(values) for metric in panel.CONTRAST_METRICS})
        result = panel.compare_panels(pairs)
        self.assertEqual(len(result), 15)
        self.assertEqual({x['metric'] for x in result}, set(panel.CONTRAST_METRICS))
        self.assertTrue(all(x['mean']==2. and x['se']==1. and x['n_worlds']==2 for x in result))
        pairs['direct1']['initial_pos'] = np.ones((2, 1))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            panel.compare_panels(pairs)


class SpecialistTests(unittest.TestCase):
    def test_fixed_targets_and_mixture_identities_follow_existing_greedy_navigation(self):
        c = seed.Cfg(G=3, M=4, R=1, H=3, n_items=1)
        grid = jnp.zeros((3, 3), jnp.int32).at[0, 1].set(1).at[1, 2].set(2)
        pos = jnp.tile(jnp.array([[1, 1]]), (4, 1))
        obs = seed.observe(grid, pos, jnp.ones(4), jnp.zeros(4), jnp.ones(4), jnp.zeros(4), c)
        a, b = eval_v3.greedy_logits(obs, 0, c), eval_v3.greedy_logits(obs, 1, c)
        np.testing.assert_array_equal(eval_v3.specialist_logits(obs, c, 'fixed_A'), a)
        np.testing.assert_array_equal(eval_v3.specialist_logits(obs, c, 'fixed_B'), b)
        mixture = eval_v3.specialist_logits(obs, c, 'mixture')
        np.testing.assert_array_equal(mixture[:2], a[:2])
        np.testing.assert_array_equal(mixture[2:], b[2:])
        self.assertFalse(np.array_equal(a, b))
        with self.assertRaisesRegex(ValueError, 'even family'):
            eval_v3.specialist_logits(obs[:1], c._replace(M=1), 'mixture')

    def test_old_reference_semantics_are_exact_against_immutable_v3_source(self):
        source = subprocess.run(['git', 'show',
            'c224f91e20d9ea12a09613d70320cfe17abea8cb:scripts/eval_v3.py'],
            capture_output=True, check=True).stdout
        historical = types.ModuleType('historical_eval_v3')
        exec(compile(source, 'historical_eval_v3.py', 'exec'), historical.__dict__)
        c = seed.Cfg(G=3, M=2, T=4, R=1, H=3, n_items=2)
        for policy in ('random_walk', 'greedy'):
            for flip in (None, 2):
                args = (jax.random.PRNGKey(41), 0, c)
                before = historical.reference_rollout(*args, policy=policy, flip_step=flip)
                after = eval_v3.reference_rollout(*args, policy=policy, flip_step=flip)
                for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
                    np.testing.assert_array_equal(a, b)


class PairedEngineTests(unittest.TestCase):
    def setUp(self):
        self.c = seed.Cfg(G=3, M=2, T=4, R=1, H=3, n_items=2)

    def test_all_conditions_share_initial_state_and_keep_prefix_and_callbacks(self):
        seen = []
        metrics, raw = panel.evaluate_controller(None, self.c, 2, 41, controller='mixture',
            on_condition=lambda label, summary, arrays: seen.append((label, summary, arrays)))
        self.assertEqual([x[0] for x in seen], list(panel.CONDITIONS))
        self.assertEqual(set(metrics['conditions']), set(panel.CONDITIONS))
        np.testing.assert_array_equal(raw['specialist_targets'], [0, 1])
        for key in panel.PAIRING_KEYS:
            for _, _, arrays in seen:
                np.testing.assert_array_equal(arrays[key], raw[key])
        for i in range(2):
            grid, pos = seed.init_world(jnp.asarray(raw['wkeys'][i]), self.c)
            np.testing.assert_array_equal(raw['initial_grid'][i], grid)
            np.testing.assert_array_equal(raw['initial_pos'][i], pos)
        for rule in (0, 1):
            for key in eval_v3.EVENT_KEYS:
                np.testing.assert_array_equal(raw[f'stationary_r{rule}_{key}'][:, :2],
                                              raw[f'reversal_r{rule}_{key}'][:, :2])
        for _, summary, arrays in seen:
            self.assertEqual(summary['total_agents'], 4)
            self.assertFalse(any(key.startswith('paired_') for key in arrays))

    def test_genome_events_match_original_engine_for_both_formats(self):
        for controller in ('rules-v0', direct.FORMAT):
            if controller == 'rules-v0':
                g = seed.init_genome(self.c, jax.random.PRNGKey(41)); mu = ravel_pytree(g)[0]
                kwargs = dict(plastic=True, predictor_plastic=True)
            else:
                mu = direct.init_genome(self.c, jax.random.PRNGKey(41)); g = None
                kwargs = dict(plastic=False, predictor_plastic=False, direct_weights=direct.decode(mu,self.c))
            _, raw = panel.evaluate_controller(mu, self.c, 2, 41, controller=controller)
            for label in panel.CONDITIONS:
                rule = int(label[-1]); flip = 2 if label.startswith('reversal') else None
                expected = seed.rollout(g, jnp.asarray(raw['wkeys'][0]), jnp.asarray(raw['dkeys'][0]),
                                        rule,self.c,diagnostics=True,flip_step=flip,**kwargs)
                for key in eval_v3.EVENT_KEYS:
                    np.testing.assert_array_equal(raw[label+'_'+key][0],expected['events'][key])
                self.assertEqual(raw[label+'_agent_return'].shape,(2,self.c.M))

    def test_callback_failure_preserves_only_completed_condition_boundary(self):
        seen=[]
        def stop(label, summary, arrays):
            seen.append(label)
            raise RuntimeError('fixture stop after first completed condition')
        with self.assertRaisesRegex(RuntimeError,'fixture stop'):
            panel.evaluate_controller(None,self.c,2,41,controller='random_walk',on_condition=stop)
        self.assertEqual(seen,['stationary_r0'])

    def test_protocol_seed_and_source_validation_rejects_before_evaluation(self):
        invalid = ((None, self.c,2,panel.EVAL_SEED,'random_walk'),
                   (None,self.c,2,42,'random_walk'),
                   (np.zeros(155),self.c,2,41,'random_walk'),
                   (None,self.c,2,41,'unknown'),
                   (None,self.c._replace(M=1),2,41,'mixture'))
        for mu,c,n,evseed,controller in invalid:
            with self.subTest(controller=controller), self.assertRaises(ValueError):
                panel.evaluate_controller(mu,c,n,evseed,controller=controller)
        mu=np.full(seed.genome_size(self.c),np.nan,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'finite'):
            panel.evaluate_controller(mu,self.c,2,41)


if __name__ == '__main__':
    unittest.main()
