"""E126 synthetic accounting and development-only timing/compatibility fixtures."""
import copy
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct, seed
from scripts import reversal_evolution as evolution


def projected_world(index, *, plastic=2, baseline=0, lesion=0, survive=True):
    row = dict(base_index=np.asarray(index, np.int32), flip_step=np.asarray(evolution.FLIP_TIMES[index % 4], np.int32),
               keys=np.arange(index * 4, index * 4 + 4, dtype=np.uint32).reshape(2, 2),
               plastic_genome_sha256=np.zeros(32, np.uint8), direct_genome_sha256=np.ones(32, np.uint8))
    for arm in evolution.ARMS:
        net = plastic if arm.startswith("plastic") else baseline if arm.startswith("direct") else lesion
        food = np.full((2, 3), max(0, net), np.int32)
        poison = np.full((2, 3), max(0, -net), np.int32)
        fields = dict(window_food_count=food, window_poison_count=poison, window_fitness=food - poison,
            window_first_food=food > 0, window_first_poison=poison > 0, window_no_meal=food + poison == 0,
            window_first_meal_step=np.full((2, 3), 1 if net else 51, np.int32),
            boundary_survival=np.full((2, 4), survive if arm.startswith("plastic") else True, bool),
            fitness=np.full(2, net * 3, np.float32))
        row.update({arm + "_" + name: value for name, value in fields.items()})
    return row


class ReversalAccountingTests(unittest.TestCase):
    def test_paired_min_window_excludes_early_and_postwindow_events(self):
        events = {name: np.zeros((300, 1), bool) for name in evolution.EVENTS}
        events["alive_before"][:] = events["alive_after"][:] = True
        events["food"][[99, 100, 149, 150, 199, 200], 0] = True
        events["poison"][151, 0] = True
        result = evolution._windows_jax(dict(events={name: jnp.asarray(value) for name, value in events.items()}), 100)
        np.testing.assert_array_equal(result["window_counts"], [[1, 0], [2, 0], [2, 1]])
        np.testing.assert_array_equal(result["window_first_food"], True)
        self.assertEqual(int(result["whole_return"]), 5)

    def test_pack_and_validate_every_assigned_dead_or_empty_candidate(self):
        shape = (2, 4, 2, 300, 1)
        events = {name: np.zeros(shape, bool) for name in evolution.EVENTS}
        events["alive_before"][..., :90, :] = True
        events["alive_after"][..., :89, :] = True
        result = dict(events, fitness=np.zeros((2, 4), np.float32),
                      rule_late_return=np.zeros((2, 4, 2), np.int32), survived=np.zeros((2, 4), np.float32))
        packed = evolution.pack_evidence(result)
        self.assertNotIn("food", packed)
        for name, value in evolution.unpack_events(packed).items():
            np.testing.assert_array_equal(value, events[name])
        evolution.validate_training_evidence(packed, evolution.FLIP_TIMES)
        changed = copy.deepcopy(packed)
        changed["fitness"][0, 0] = 1
        with self.assertRaises(ValueError):
            evolution.validate_training_evidence(changed, evolution.FLIP_TIMES)

    def test_combined_se_is_not_replaced_by_paired_difference_se(self):
        rows = [projected_world(index, plastic=2 if index % 2 else 40, baseline=1 if index % 2 else 39)
                for index in range(4)]
        result = evolution.summarize(rows)
        self.assertEqual(result["plastic_vs_direct"]["paired_difference"]["se"], 0.)
        self.assertEqual(result["plastic_vs_direct"]["difference"], 1.)
        self.assertGreater(result["plastic_vs_direct"]["combined_se"], 1.)
        self.assertFalse(result["plastic_vs_direct"]["passed"])

    def test_gate_requires_positive_useful_return_and_causal_lesion(self):
        good = evolution.summarize([projected_world(index) for index in range(128)])
        self.assertTrue(good["gate"]["performance_passed"])
        avoided = evolution.summarize([projected_world(index, plastic=0, baseline=-2, lesion=-1) for index in range(128)])
        self.assertTrue(avoided["plastic_vs_direct"]["passed"])
        self.assertFalse(avoided["positive_return_passed"])
        self.assertFalse(avoided["late_accuracy_passed"])
        self.assertFalse(avoided["gate"]["performance_passed"])
        no_mechanism = evolution.summarize([projected_world(index, lesion=2) for index in range(128)])
        self.assertFalse(no_mechanism["plastic_vs_lesion"]["passed"])
        self.assertFalse(no_mechanism["gate"]["performance_passed"])

    def test_formal_aggregate_requires_two_same_performance_seeds_and_all_survival(self):
        good = evolution.summarize([projected_world(index) for index in range(128)])
        panels = [copy.deepcopy(good) for _ in range(3)]
        for index, panel in enumerate(panels):
            panel["world_keys_sha256"] = str(index + 1) * 64
        panels[2]["gate"]["performance_passed"] = False
        self.assertTrue(evolution.aggregate(panels)["passed"])
        panels[2]["gate"]["survival_passed"] = False
        self.assertFalse(evolution.aggregate(panels)["passed"])
        with self.assertRaises(ValueError):
            evolution.aggregate([good, good, good])
        with self.assertRaises(ValueError):
            evolution.aggregate(panels[:2])

    def test_cluster_unit_is_world_not_two_rule_replays(self):
        rows = [projected_world(index, plastic=index % 2 + 1) for index in range(4)]
        result = evolution.summarize(rows)
        self.assertAlmostEqual(result["arms"]["plastic_reversal"]["late_min_return"]["se"],
                               np.asarray([1, 2, 1, 2]).std(ddof=1) / np.sqrt(4))
        self.assertEqual(result["base_worlds"], 4)
        self.assertEqual(result["lives_per_arm"], 8)
        self.assertIsNone(evolution.summarize(rows[:1])["gate"]["passed"])


class ReversalSamplingTests(unittest.TestCase):
    def test_both_encodings_share_roots_and_compact_noise_regenerates_exactly(self):
        c = seed.Cfg(G=4, M=1, T=300, R=1, H=3, n_items=2)
        states = [evolution.initialize_state(controller, 53, c) for controller in evolution.CONTROLLERS]
        draws = [evolution.sampling(state, pop=4) for state in states]
        for name in ("next_key", "noise_key", "world_keys", "development_keys", "assignment_key", "permutation", "flip_steps"):
            np.testing.assert_array_equal(draws[0][name], draws[1][name])
        for state, recorded in zip(states, draws):
            half = jax.random.normal(jnp.asarray(recorded["noise_key"]), (2, state["mu"].size))
            np.testing.assert_array_equal(evolution.array_hash(half), recorded["half_eps_sha256"])
            candidates = state["mu"] + .1 * jnp.concatenate((half, -half))
            np.testing.assert_array_equal(evolution.candidate_hashes(candidates), recorded["candidate_sha256"])
            self.assertNotIn("half_eps", recorded)
            self.assertNotIn("candidates", recorded)
            np.testing.assert_array_equal(recorded["development_keys"][:2], recorded["development_keys"][2:])
        self.assertIs(evolution.make_evaluator("rules-v0", c), evolution.make_evaluator("rules-v0", c))

    def test_all_ties_keep_nonzero_moments_and_advance_attempt_stream(self):
        c = seed.Cfg(M=1)
        state = evolution.initialize_state("rules-v0", 53, c)
        state.update(m=jnp.ones_like(state["mu"]), v=jnp.ones_like(state["mu"]), t=2)
        captured = {}

        def evaluate(candidates, worlds, births):
            captured.update(candidate_sha256=evolution.candidate_hashes(candidates), world_keys=np.asarray(worlds),
                            development_keys=np.asarray(births))
            return dict(fitness=np.zeros((4, 4)), counts=np.zeros((4, 4, 2, 2)), survived=np.zeros((4, 4)))

        draw = evolution.sampling(state, pop=4)
        after, row = evolution.es_step(state, evaluate, 4, 4, .1, .1)
        self.assertTrue(row["all_tied"])
        for name in ("mu", "m", "v", "t"):
            np.testing.assert_array_equal(state[name], after[name])
        for name, value in captured.items():
            np.testing.assert_array_equal(value, draw[name])
        np.testing.assert_array_equal(after["key"], draw["next_key"])
        self.assertFalse(np.array_equal(state["key"], after["key"]))

    def test_evaluation_schedule_balances_times_without_exposing_rule(self):
        schedule = evolution.evaluation_schedule(53, 128)
        self.assertEqual(set(schedule), {"key", "permutation", "flip_steps"})
        for flip in evolution.FLIP_TIMES:
            self.assertEqual(int((schedule["flip_steps"] == flip).sum()), 32)
        self.assertEqual(len(set(schedule["permutation"].tolist())), 128)
        with self.assertRaises(ValueError):
            evolution.evaluation_schedule(53, 127)


class ReversalEngineTests(unittest.TestCase):
    """Prepared for the coordinator's released development checks; no final masters."""

    def assert_tree_equal(self, left, right):
        self.assertEqual(jax.tree.structure(left), jax.tree.structure(right))
        for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
            np.testing.assert_array_equal(actual, expected)

    def test_lesion_starts_after_first_postflip_outcome_and_keeps_predictor(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        genome = seed.init_genome(c, jax.random.PRNGKey(53))
        keys = evolution.evaluation_keys(54, 0)
        for flip in (None, 2):
            common = dict(record=True, diagnostics=True, flip_step=flip,
                          _world_protocol=evolution.PROTOCOL, _reversal_probe_step=3)
            intact = jax.tree.map(np.asarray, seed.rollout(genome, *keys, 0, c, **common))
            lesion = jax.tree.map(np.asarray, seed.rollout(genome, *keys, 0, c, _hebbian_off_step=3, **common))
            for section in ("events", "_world"):
                for name, value in intact[section].items():
                    np.testing.assert_array_equal(value[:3], lesion[section][name][:3], err_msg=name)
            self.assert_tree_equal(intact["_reversal_probe"], lesion["_reversal_probe"])
            np.testing.assert_array_equal(lesion["_world_final"]["W"], lesion["_reversal_probe"]["W"])
            np.testing.assert_array_equal(lesion["_world"]["hebbian_enabled"], [1, 1, 1, 0, 0, 0])
            np.testing.assert_array_equal(lesion["_world"]["predictor_enabled"], True)
            self.assertFalse(np.array_equal(lesion["_world_final"]["Wp"], lesion["_reversal_probe"]["Wp"]))
        plain = seed.rollout(genome, *keys, 0, c, record=True, diagnostics=True)
        self.assert_tree_equal(plain, seed.rollout(genome, *keys, 0, c, record=True, diagnostics=True,
                                                  _hebbian_off_step=None, _reversal_probe_step=None))
        with self.assertRaises(ValueError):
            seed.rollout(genome, *keys, 0, c, flip_step=2, _world_protocol=evolution.PROTOCOL, _hebbian_off_step=2)

    def test_compiled_pair_objective_matches_actual_both_rule_events(self):
        c = seed.Cfg(G=4, M=1, T=300, R=1, H=3, n_items=2)
        for controller in evolution.CONTROLLERS:
            state = evolution.initialize_state(controller, 53, c)
            draw = evolution.sampling(state, pop=2)
            half = jax.random.normal(jnp.asarray(draw["noise_key"]), (1, state["mu"].size))
            population = state["mu"] + .1 * jnp.concatenate((half, -half))
            result = evolution.make_evaluator(controller, c)(population, draw["world_keys"], draw["development_keys"])
            evidence = evolution.pack_evidence(result)
            evolution.validate_training_evidence(evidence, draw["flip_steps"], c)
            self.assertEqual(result["fitness"].shape, (2, 4))
            self.assertEqual(result["rule_late_return"].shape, (2, 4, 2))

    def test_complete_pair_world_carries_actual_private_and_public_evidence(self):
        c = seed.Cfg(G=4, M=1, T=300, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        fixed = direct.init_genome(c, jax.random.PRNGKey(53))
        row = evolution.evaluate_world(np.asarray(flat), np.asarray(fixed), evolution.evaluation_keys(54, 0), 0, 100, c)
        compact = evolution.summary_input(row)
        self.assertEqual(evolution.summarize([compact])["lives_per_arm"], 2)
        self.assertLess(sum(np.asarray(value).nbytes for value in compact.values()), 4096)
        for name in ("lesion_reversal_hebbian_enabled", "plastic_reversal_window_first_food",
                     "plastic_reversal_prelesion_sha256", "lesion_stationary_probe_W_sha256"):
            altered = copy.deepcopy(row)
            if altered[name].dtype == bool:
                altered[name].flat[0] = not altered[name].flat[0]
            else:
                altered[name].flat[0] ^= 1
            with self.subTest(field=name), self.assertRaises(ValueError):
                evolution.validate_world(altered, c)


if __name__ == "__main__":
    unittest.main()
