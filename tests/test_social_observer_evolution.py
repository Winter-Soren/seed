"""Tie handling, shared demonstrations and paired observer comparison checks."""
import copy
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import social_observer_evolution as study
from scripts import social_learning as social
from seed import seed


def teacher_fixture(c):
    grid = np.zeros((c.G, c.G), np.uint8)
    return dict(teacher_initial_grid=grid, teacher_grid=np.tile(grid, (c.T, 1, 1)),
                teacher_initial_position=np.array([[2, 2]], np.int32),
                teacher_position=np.tile(np.array([[[2, 2]]], np.int32), (c.T, 1, 1)),
                teacher_signal=np.zeros((c.T, 1), np.float32),
                teacher_energy=np.full((c.T, 1), c.e0, np.float32),
                teacher_food=np.zeros((c.T, 1), bool), teacher_poison=np.zeros((c.T, 1), bool),
                teacher_alive_before=np.ones((c.T, 1), bool), teacher_alive_after=np.ones((c.T, 1), bool),
                teacher_birth_sha256=np.zeros(32, np.uint8), teacher_terminal_sha256=np.zeros(32, np.uint8))


def family(index=0, starting_food=False, observed_survives=True):
    c = seed.Cfg(G=6, M=1, T=6, R=1, H=3)
    flat = np.zeros(seed.genome_size(c), np.float32)
    row = dict(keys=np.arange(index * 8 + 1, index * 8 + 9, dtype=np.uint32).reshape(4, 2),
               family_index=np.asarray(index), rule=np.asarray(index % 2),
               yoked_swap=np.asarray(bool((index // 2) % 2)), steps=np.asarray(c.T),
               watch_start=np.asarray(c.T // 2), agents=np.asarray(1), radius=np.asarray(c.R),
               grid_size=np.asarray(c.G), hidden_units=np.asarray(c.H), initial_energy=np.asarray(c.e0, np.float32),
               teacher_genome=flat.copy(), starting_genome=flat.copy(), evolved_genome=flat.copy(),
               solo_initial_grid=np.zeros((c.G, c.G), np.uint8),
               solo_initial_position=np.array([[0, 0]], np.int32), **teacher_fixture(c))
    movie, row["camera_positions"] = social.make_movie(row, c)
    conditions = social.movie_conditions(movie, bool(row["yoked_swap"]), c)
    conditions["starting_observed"] = movie
    _, inputs, outputs, neurons, _ = seed.dims(c)
    hashes = dict(W=social._hash(np.zeros((neurons, neurons), np.float32)),
                  x=social._hash(np.zeros(neurons, np.float32)),
                  Wp=social._hash(np.zeros((inputs, c.H), np.float32)))
    for arm, frames in conditions.items():
        prefix = arm + "_"
        food = np.zeros((c.T, 1), bool)
        food[0, 0] = arm == "observed" or (arm == "starting_observed" and starting_food)
        poison = np.zeros_like(food)
        after = np.ones_like(food)
        if arm == "observed":
            after[-1, 0] = observed_survives
        before = np.concatenate((np.ones((1, 1), bool), after[:-1]))
        motor = np.zeros((c.T // 2, outputs - 1), np.float32)
        motor[:, 0] = 1
        fields = dict(frames=frames, watch_motor=motor, watch_denergy=np.zeros(c.T // 2, np.float32),
                      birth_sha256=hashes["W"].copy(), solo_initial_energy=np.array([c.e0], np.float32),
                      solo_initial_alive=np.ones(1, np.float32), solo_initial_denergy=np.zeros(1, np.float32),
                      food=food, poison=poison, alive_before=before, alive_after=after,
                      fitness=np.asarray(float(food.sum())), weight_change_count=np.asarray(0),
                      **social.first_meal(food, poison))
        for name, digest in hashes.items():
            for stage in ("watch_initial", "watch_final", "solo_initial"):
                fields[stage + "_" + name + "_sha256"] = digest.copy()
        row.update({prefix + key: value for key, value in fields.items()})
    return c, row


class ObserverEvolutionTests(unittest.TestCase):
    def test_average_ranks_treat_ties_identically(self):
        actual = np.asarray(study.average_centered_ranks(jnp.array([0., 0., .5, 1.])))
        np.testing.assert_allclose(actual, [-1 / 3, -1 / 3, 1 / 6, .5], atol=3e-8)
        np.testing.assert_array_equal(study.average_centered_ranks(jnp.ones(16)), np.zeros(16))
        order = np.array([3, 0, 2, 1])
        np.testing.assert_array_equal(study.average_centered_ranks(jnp.array([0., 0., .5, 1.])[order]), actual[order])

    def test_all_ties_advance_rng_without_mutation_momentum_or_regularization(self):
        state = dict(mu=jnp.array([1., 2.]), m=jnp.array([.2, -.1]), v=jnp.array([.4, .3]),
                     t=7, key=jax.random.PRNGKey(49))
        seen = {}
        def equal(candidates, world_keys, birth_keys):
            seen.update(candidates=np.asarray(candidates), worlds=np.asarray(world_keys), births=np.asarray(birth_keys))
            return dict(fitness=np.ones((4, 2), np.float32), counts=np.zeros((4, 2, 2, 2)),
                        survived=np.ones((4, 2)))
        updated, metrics = study.es_step(state, equal, pop=4, E=2, sigma=.05, lr=.02)
        for key in ("mu", "m", "v", "t"):
            np.testing.assert_array_equal(updated[key], state[key])
        self.assertFalse(np.array_equal(updated["key"], state["key"]))
        self.assertFalse(metrics["informative"])
        self.assertEqual(metrics["t"], 7)
        np.testing.assert_array_equal(seen["births"][:2], seen["births"][2:])
        np.testing.assert_allclose(seen["candidates"][:2] + seen["candidates"][2:],
                                   np.broadcast_to(2 * np.asarray(state["mu"]), (2, 2)), atol=1e-7)

    def test_informative_round_updates_with_average_rank_gradient(self):
        state = dict(mu=jnp.array([.5, -.5]), m=jnp.zeros(2), v=jnp.zeros(2), t=0,
                     key=jax.random.PRNGKey(49))
        def score(candidates, world_keys, birth_keys):
            return dict(fitness=jnp.array([[0., 0.], [0., 0.], [0., 1.], [1., 1.]]),
                        counts=np.zeros((4, 2, 2, 2)), survived=np.ones((4, 2)))
        updated, metrics = study.es_step(state, score, pop=4, E=2, sigma=.05, lr=.02)
        _, noise_key, _, _ = jax.random.split(state["key"], 4)
        half = jax.random.normal(noise_key, (2, 2))
        noise = jnp.concatenate((half, -half))
        ranks = jnp.array([-1 / 3, -1 / 3, 1 / 6, .5])
        gradient = (ranks[:, None] * noise).sum(0) / (.05 * 4) - .005 * state["mu"]
        np.testing.assert_allclose(updated["m"], .1 * gradient, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(updated["v"], .001 * gradient ** 2, rtol=1e-6, atol=1e-7)
        self.assertTrue(metrics["informative"])
        self.assertEqual(updated["t"], 1)
        self.assertEqual(metrics["distinct_scores"], 3)

    def test_teacher_movie_generated_once_per_family_and_shared(self):
        c = seed.Cfg(G=6, M=1, T=6, R=1, H=3)
        flat = np.zeros(seed.genome_size(c), np.float32)
        roots = np.asarray(jax.random.split(jax.random.PRNGKey(49), 4))
        with patch.object(social, "_record_teacher", side_effect=lambda *args: teacher_fixture(c)) as record:
            batch = study.make_training_batch(flat, roots, c)
        self.assertEqual(record.call_count, 4)
        self.assertEqual(batch["movies"].shape, (4, 3, seed.dims(c)[1]))
        self.assertEqual(batch["role_keys"].shape, (4, 3, 2))
        np.testing.assert_array_equal(batch["solo_world_keys"], batch["role_keys"][:, 2])
        for call in record.call_args_list:
            np.testing.assert_array_equal(call.args[0], flat)
        np.testing.assert_array_equal(batch["rules"], [0, 1, 0, 1])

    def test_combined_gate_requires_beating_original_observer(self):
        rows = [family(i, starting_food=True)[1] for i in range(2)]
        result = study.summarize(rows)
        for control in social.CONTROLS:
            self.assertTrue(result["contrasts"][control]["gate"]["passed"])
        self.assertFalse(result["contrasts"]["starting_observed"]["gate"]["passed"])
        self.assertFalse(result["gate"]["passed"])
        passed = study.summarize([family(i)[1] for i in range(2)])
        self.assertTrue(passed["gate"]["passed"])
        failed = study.summarize([family(i, observed_survives=False)[1] for i in range(2)])
        self.assertFalse(failed["gate"]["survival_passed"])
        self.assertIsNone(study.summarize([rows[0]])["gate"]["passed"])

    def test_starting_observer_state_and_movie_corruption_are_rejected(self):
        c, reference = family()
        for field in ("starting_observed_frames", "starting_observed_solo_initial_W_sha256",
                      "starting_observed_solo_initial_energy", "teacher_genome"):
            row = copy.deepcopy(reference)
            row[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.validate_family(row, c)


class ObserverEvolutionDevelopmentTests(unittest.TestCase):
    def test_compiled_scorer_matches_recorded_own_state_path_and_equal_observers(self):
        c = seed.Cfg(G=6, M=1, T=6, R=1, H=3, n_items=3)
        flat = np.asarray(ravel_pytree(seed.biology_prior(c))[0])
        roots = np.asarray(jax.random.split(jax.random.PRNGKey(49), 2))
        batch = study.make_training_batch(flat, roots, c)
        births = np.asarray(jax.random.split(jax.random.PRNGKey(50), 2))[None]
        score = jax.tree.map(np.asarray, study.make_evaluator(c)(flat[None], batch["movies"],
            batch["solo_world_keys"], births, batch["rules"]))
        for index in range(2):
            reference = social.record_pupil(flat, batch["solo_world_keys"][index], births[0, index],
                                            index % 2, batch["movies"][index], c)
            for event in social.EVENTS:
                np.testing.assert_array_equal(score["candidate_" + event][0, index], reference[event])
            self.assertEqual(score["fitness"][0, index], float(reference["first_food"]))
        row = study.evaluate_family(flat, flat, flat, social.family_keys(49, 0), 0, c)
        for field in social.EVENTS + ("frames", "first_food", "birth_sha256", "solo_initial_W_sha256"):
            np.testing.assert_array_equal(row["observed_" + field], row["starting_observed_" + field])


if __name__ == "__main__":
    unittest.main()
