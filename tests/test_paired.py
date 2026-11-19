"""Paired training objectives: synthetic accounting and tiny seed-41 fixtures only."""
import io
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import paired, seed


class PairedEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.c = seed.Cfg(G=3, M=2, T=6, R=0, H=3, n_items=2)
        kg, kw, kd = jax.random.split(jax.random.PRNGKey(41), 3)
        self.genome = seed.init_genome(self.c, kg)
        self.flat = ravel_pytree(self.genome)[0]
        self.wkeys = jax.random.split(kw, 2)
        self.dkeys = jax.random.split(kd, 4).reshape(2, 2, 2)

    def test_minimum_precedes_family_mean_and_keeps_dead_and_zero_encounter_agents(self):
        for agents in (2, 4):
            c = self.c._replace(M=agents, T=10)

            def synthetic_rollout(genome, wkey, dkey, rule, cfg, **kwargs):
                food, poison = np.zeros((10, agents), bool), np.zeros((10, agents), bool)
                food[:, rule] = True
                poison[:4, 1-rule] = True
                alive_after = np.ones((10, agents), bool)
                alive_after[3:, 1-rule] = False
                if agents == 4:
                    alive_after[:, 2] = False
                alive_before = np.concatenate([np.ones((1, agents), bool), alive_after[:-1]])
                counts = [[food[:5].sum(), poison[:5].sum()],
                          [food[5:].sum(), poison[5:].sum()]]
                return dict(fitness=jnp.array(999.), counts=jnp.asarray(counts),
                            survived=jnp.asarray(alive_after[-1].mean(), jnp.float32),
                            events=dict(food=jnp.asarray(food), poison=jnp.asarray(poison),
                                        alive_before=jnp.asarray(alive_before),
                                        alive_after=jnp.asarray(alive_after)))

            for objective, expected in (("paired-mean", 6/agents), ("paired-min", -8/agents)):
                with self.subTest(agents=agents, objective=objective):
                    with patch.object(paired, "rollout", side_effect=synthetic_rollout) as mocked:
                        evaluate, _ = paired.make_evaluator(c, 1, objective)
                        result = evaluate(self.flat[None], self.wkeys[:1], self.dkeys[:1, :1])
                        self.assertEqual(float(result['fitness'][0, 0]), expected)
                    self.assertEqual(mocked.call_count, 2)
                    first, second = mocked.call_args_list
                    self.assertEqual([first.args[3], second.args[3]], [0, 1])
                    self.assertIs(first.args[0], second.args[0])
                    self.assertIs(first.args[1], second.args[1])
                    self.assertIs(first.args[2], second.args[2])
                    for call in (first, second):
                        self.assertEqual(call.kwargs, dict(plastic=True, predictor_plastic=True,
                                                          structural=False, flip_step=None, diagnostics=True))
                    np.testing.assert_array_equal(result['counts'][0, 0], [[10, 8], [10, 0]])
                    self.assertEqual(float(result['survived'][0, 0]), .5)
                    # Both family means are positive; each specialist's worse return is -4.
                    self.assertGreater(6/agents, 0)
                    self.assertLess(-8/agents, 0)

    def test_paired_scores_and_counts_exactly_match_manual_fresh_rule_events(self):
        prior = seed.biology_prior(self.c, jax.random.PRNGKey(41))
        population = jnp.stack([self.flat, ravel_pytree(prior)[0]])
        manual = {}
        for candidate, genome in enumerate((self.genome, prior)):
            for world in range(2):
                replays = [seed.rollout(
                    genome, self.wkeys[world], self.dkeys[candidate, world], rule, self.c,
                    plastic=True, predictor_plastic=True, structural=False, flip_step=None,
                    diagnostics=True) for rule in (0, 1)]
                returns = []
                for replay in replays:
                    food = np.asarray(replay['events']['food'], dtype=np.int32)
                    poison = np.asarray(replay['events']['poison'], dtype=np.int32)
                    self.assertEqual(food.shape, (self.c.T, self.c.M))
                    self.assertEqual(poison.shape, (self.c.T, self.c.M))
                    returns.append((food-poison).sum(0))
                    boundary = self.c.T//2
                    event_counts = [[food[:boundary].sum(), poison[:boundary].sum()],
                                    [food[boundary:].sum(), poison[boundary:].sum()]]
                    np.testing.assert_array_equal(replay['counts'], event_counts)
                manual[candidate, world] = replays, returns

        for objective in paired.OBJECTIVES:
            evaluate, unravel = paired.make_evaluator(self.c, 2, objective)
            self.assertEqual(jax.tree.structure(unravel(self.flat)), jax.tree.structure(self.genome))
            for actual, expected in zip(jax.tree.leaves(unravel(self.flat)), jax.tree.leaves(self.genome)):
                np.testing.assert_array_equal(actual, expected)
            result = evaluate(population, self.wkeys, self.dkeys)
            self.assertEqual(set(result), {'fitness', 'counts', 'survived'})
            self.assertEqual(result['fitness'].shape, (2, 2))
            self.assertEqual(result['counts'].shape, (2, 2, 2, 2))
            self.assertEqual(result['survived'].shape, (2, 2))
            for (candidate, world), (replays, returns) in manual.items():
                with self.subTest(objective=objective, candidate=candidate, world=world):
                    expected = (np.minimum(*returns).mean() if objective == 'paired-min'
                                else np.mean(returns))
                    self.assertEqual(float(result['fitness'][candidate, world]), expected)
                    np.testing.assert_array_equal(result['counts'][candidate, world],
                                                  replays[0]['counts'] + replays[1]['counts'])
                    self.assertEqual(float(result['survived'][candidate, world]),
                                     float((replays[0]['survived'] + replays[1]['survived']) / 2))

    def test_es_arms_share_sampling_start_with_zero_adam_and_resume_exactly(self):
        records = {}
        for objective in paired.OBJECTIVES:
            evaluate, _ = paired.make_evaluator(self.c, 2, objective)
            state = seed.es_init(self.c, mean=self.flat, seed=41)
            np.testing.assert_array_equal(state['mu'], self.flat)
            np.testing.assert_array_equal(state['m'], np.zeros(self.flat.shape))
            np.testing.assert_array_equal(state['v'], np.zeros(self.flat.shape))
            self.assertEqual(state['t'], 0)
            records[objective] = []
            for generation in range(2):
                next_key, ke, kw, kd = jax.random.split(state['key'], 4)
                half_noise = jax.random.normal(ke, (2, self.flat.size))
                noise = jnp.concatenate([half_noise, -half_noise])
                expected_population = state['mu'] + .1*noise
                expected_worlds = jax.random.split(kw, 2)
                half_development = jax.random.split(kd, 4).reshape(2, 2, 2)
                expected_development = jnp.concatenate([half_development, half_development])

                def capture(population, worlds, development):
                    np.testing.assert_array_equal(population, expected_population)
                    np.testing.assert_array_equal(worlds, expected_worlds)
                    np.testing.assert_array_equal(development, expected_development)
                    records[objective].append(tuple(np.asarray(x) for x in
                                                    (noise, worlds, development, state['key'])))
                    return evaluate(population, worlds, development)

                following, row = seed.es_step(state, capture, pop=4, E=2, sigma=.1, lr=.1, l2=.005)
                self.assertEqual(row['gen'], generation)
                self.assertTrue(all(np.isfinite(value) for value in row.values()))
                np.testing.assert_array_equal(following['key'], next_key)
                if generation == 1:
                    checkpoint = io.BytesIO()
                    np.savez(checkpoint, **{name: np.asarray(value) for name, value in state.items()})
                    checkpoint.seek(0)
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        restored = {name: jnp.asarray(saved[name]) for name in saved.files}
                    restored['t'] = int(restored['t'])
                    resumed, resumed_row = seed.es_step(restored, evaluate, pop=4, E=2,
                                                       sigma=.1, lr=.1, l2=.005)
                    self.assertEqual(resumed_row, row)
                    self.assertEqual(set(resumed), set(following))
                    for name in following:
                        np.testing.assert_array_equal(resumed[name], following[name], err_msg=name)
                state = following
        for mean_record, min_record in zip(records['paired-mean'], records['paired-min']):
            for mean_value, min_value in zip(mean_record, min_record):
                np.testing.assert_array_equal(mean_value, min_value)

    def test_invalid_objective_world_count_and_config_are_rejected(self):
        for objective in ('mean', 'min', '', None):
            with self.subTest(objective=objective), self.assertRaisesRegex(ValueError, 'objective'):
                paired.make_evaluator(self.c, 2, objective)
        for worlds in (0, -1, True, 1.5):
            with self.subTest(worlds=worlds), self.assertRaisesRegex(ValueError, 'positive integer'):
                paired.make_evaluator(self.c, worlds, 'paired-min')
        for change in (dict(G=0), dict(M=0), dict(T=1), dict(R=-1), dict(n_items=5),
                       dict(H=4), dict(n_htypes=0), dict(micro=0), dict(M=2.5)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'dimensions'):
                paired.make_evaluator(self.c._replace(**change), 2, 'paired-min')
        for change in (dict(e0=0), dict(e0=3), dict(e_max=np.nan), dict(wmax=0),
                       dict(gain=-.1), dict(decay=-.1)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'bounds'):
                paired.make_evaluator(self.c._replace(**change), 2, 'paired-min')
        with self.assertRaisesRegex(ValueError, 'Cfg'):
            paired.make_evaluator(None, 2, 'paired-min')

    def test_population_and_key_batch_mismatches_are_rejected(self):
        evaluate, _ = paired.make_evaluator(self.c, 2, 'paired-min')
        for population in (self.flat, self.flat[None, :-1], jnp.zeros((0, self.flat.size)),
                           jnp.zeros((1, self.flat.size), dtype=jnp.int32)):
            with self.subTest(shape=population.shape), self.assertRaisesRegex(ValueError, 'population'):
                evaluate(population, self.wkeys, self.dkeys[:1])
        for worlds, development in ((self.wkeys[:1], self.dkeys[:1]),
                                    (self.wkeys, self.dkeys),
                                    (self.wkeys, self.dkeys[:1, :1])):
            with self.subTest(worlds=worlds.shape, development=development.shape):
                with self.assertRaisesRegex(ValueError, 'key batches'):
                    evaluate(self.flat[None], worlds, development)


if __name__ == '__main__':
    unittest.main()
