"""Synthetic development checks for the versioned changing-world measurement."""
import copy
from pathlib import Path
import subprocess
import types
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import social_reversal as reversal
from seed import direct, seed


def compact_family(index):
    row = dict(keys=np.arange(8 * index, 8 * index + 8, dtype=np.uint32).reshape(4, 2),
               family_index=np.asarray(index), rule=np.asarray(index % 2),
               flip_step=np.asarray(reversal.FLIP_TIMES[index % 4]), lambda_value=np.asarray(4.))
    for arm in reversal.ARMS:
        row[arm + "_preflip_identical"] = np.asarray(True)
        for condition in reversal.CONDITIONS:
            prefix = arm + "_" + condition + "_"
            choices = [1, 0, 1] if condition == "reversal" else [1, 1, 1]
            food = np.asarray(choices, bool)
            row.update({prefix + "window_first_food": food,
                        prefix + "window_first_poison": ~food,
                        prefix + "window_no_meal": np.zeros(3, bool),
                        prefix + "window_food_count": food.astype(np.int32),
                        prefix + "window_poison_count": (~food).astype(np.int32),
                        prefix + "window_fitness": food.astype(np.int32) - (~food).astype(np.int32),
                        prefix + "window_first_meal_step": np.ones(3, np.int32),
                        prefix + "boundary_survival": np.ones(4, bool),
                        prefix + "fitness": np.asarray(2 * food.sum() - 3, np.float32)})
    return row


class SocialReversalTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(reversal._same_bytes(left, right))

    def test_schedule_balances_cells_before_independent_role_permutation(self):
        master = 2026120300  # Development only; no final worlds are constructed.
        manifest = reversal.assignments_manifest(master, 128)
        self.assertEqual(set(manifest), {"key", "permutation", "table"})
        np.testing.assert_array_equal(np.sort(manifest["permutation"]), np.arange(128))
        for rule in (0, 1):
            for step in reversal.FLIP_TIMES:
                self.assertEqual(np.all(manifest["table"] == [rule, step], axis=1).sum(), 16)
        self.assert_tree_equal(manifest, reversal.assignments_manifest(master, 128))
        development = reversal.schedule(master, 4)
        self.assertEqual(sorted(development[:, 1]), list(reversal.FLIP_TIMES))
        self.assertEqual(np.count_nonzero(development[:, 0]), 2)
        self.assertFalse(any(np.array_equal(key, manifest["key"]) for key in reversal.family_keys(master, 0)))
        with self.assertRaises(ValueError):
            reversal.schedule(master, 5)

    def test_windows_are_half_open_and_keep_poison_and_no_contact_failures(self):
        food = np.zeros((300, 1), bool)
        poison = np.zeros_like(food)
        # At boundary100: step99 remains pre;100 starts early;150 starts late.
        food[[99, 101, 150], 0] = True
        poison[100, 0] = True
        alive = np.ones_like(food)
        scored = reversal.window_outcomes(food, poison, alive, alive, 100)
        np.testing.assert_array_equal(scored["window_first_food"], [True, False, True])
        np.testing.assert_array_equal(scored["window_food_count"], [1, 1, 1])
        np.testing.assert_array_equal(scored["window_poison_count"], [0, 1, 0])
        np.testing.assert_array_equal(scored["window_first_meal_step"], [50, 1, 1])
        np.testing.assert_array_equal(scored["boundary_survival"], True)
        dead = np.zeros_like(alive)
        empty = reversal.window_outcomes(np.zeros_like(food), np.zeros_like(poison), dead, dead, 100)
        np.testing.assert_array_equal(empty["window_first_food"], False)
        np.testing.assert_array_equal(empty["window_no_meal"], True)
        np.testing.assert_array_equal(empty["window_first_meal_step"], 51)

    def test_survival_uses_the_registered_boundary_states(self):
        food = np.zeros((300, 1), bool)
        before = np.ones_like(food)
        after = np.ones_like(food)
        after[149:] = False
        before[150:] = False
        scored = reversal.window_outcomes(food, food, before, after, 100)
        np.testing.assert_array_equal(scored["boundary_survival"], [True, False, False, False])

    def test_intersection_gate_preserves_strict_and_inclusive_boundaries(self):
        stats = lambda mean, se=0.: dict(mean=mean, se=se)
        passing = reversal.transfer_gate(stats(.6), stats(.2), stats(0), stats(0))
        self.assertTrue(passing["passed"])
        self.assertFalse(reversal.transfer_gate(stats(.6), stats(.2, .1), stats(0), stats(0))["recovery_passed"])
        self.assertFalse(reversal.transfer_gate(stats(.6), stats(.2), stats(-.1), stats(0))["accuracy_noninferiority_passed"])
        self.assertFalse(reversal.transfer_gate(stats(.6), stats(.2), stats(0), stats(-.1))["survival_noninferiority_passed"])
        self.assertFalse(reversal.transfer_gate(stats(.599), stats(.2), stats(0), stats(0))["late_level_passed"])
        self.assertIsNone(reversal.transfer_gate(stats(1., None), stats(1., None), stats(0., None), stats(0., None))["passed"])

    def test_summary_uses_family_paired_recovery_and_keeps_panels_separate(self):
        rows = [compact_family(index) for index in range(4)]
        summary = reversal.summarize(rows)
        self.assertTrue(summary["gate"]["passed"])
        contrast = summary["arms"]["observed"]["reversal_minus_stationary"]
        self.assertEqual(contrast["recovery_difference"], dict(mean=1., se=0.))
        self.assertEqual(contrast["late_first_food_difference"], dict(mean=0., se=0.))
        for panel in ("within_social", "within_birth"):
            self.assertEqual(summary[panel]["reversal"]["windows"]["late"]["first_food_difference"], dict(mean=0., se=0.))
        self.assertEqual(summary["neutral_selected_birth_duplicates"], "birth_parent")
        with self.assertRaises(ValueError):
            reversal.summarize(rows + [rows[0]])
        changed = copy.deepcopy(rows)
        changed[0]["observed_reversal_window_food_count"][0] += 1
        with self.assertRaises(ValueError):
            reversal.summarize(changed)

    def test_survivor_only_empty_denominator_stays_missing(self):
        rows = [compact_family(index) for index in range(4)]
        for row in rows:
            for condition in reversal.CONDITIONS:
                row["birth_parent_" + condition + "_boundary_survival"][-1] = False
        result = reversal.summarize(rows)["arms"]["birth_parent"]["conditions"]["reversal"]
        self.assertEqual(result["families"], 4)
        self.assertEqual(result["end_survivor_only"]["families"], 0)
        self.assertEqual(result["end_survivor_only"]["windows"]["late"]["accuracy"], dict(mean=None, se=None))

    def test_default_and_versioned_stationary_paths_preserve_old_trajectories(self):
        source = subprocess.check_output(["git", "show", "271d3869f531fd556554015249c02d08b0f2cbd5:seed/seed.py"],
                                         cwd=Path(__file__).resolve().parents[1], text=True, timeout=10)
        old = types.ModuleType("before_randomized_reversal")
        exec(compile(source, "before_randomized_reversal.py", "exec"), old.__dict__)
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        wk, dk = jax.random.PRNGKey(55), jax.random.PRNGKey(56)
        genome = seed.biology_prior(c)
        for q in (None, jnp.array([.75])):
            options = dict(record=True, diagnostics=True, _belief_q=q)
            baseline = old.rollout(genome, wk, dk, 0, c, **options)
            self.assert_tree_equal(seed.rollout(genome, wk, dk, 0, c, **options), baseline)
            versioned = seed.rollout(genome, wk, dk, 0, c, _world_protocol=reversal.PROTOCOL, **options)
            versioned = {key: value for key, value in versioned.items() if not key.startswith("_world")}
            versioned["traj"] = {key: value for key, value in versioned["traj"].items() if key != "_world"}
            self.assert_tree_equal(versioned, baseline)

    def test_social_reversal_requires_explicit_world_version_and_preserves_preflip_actions(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        wk, dk = jax.random.PRNGKey(55), jax.random.PRNGKey(56)
        genome = seed.biology_prior(c)
        options = dict(record=True, diagnostics=True, _belief_q=jnp.array([.75]))
        with self.assertRaises(ValueError):
            seed.rollout(genome, wk, dk, 0, c, flip_step=3, **options)
        with self.assertRaises(ValueError):
            seed.rollout(genome, wk, dk, 0, c, _world_protocol="unregistered", **options)
        stationary = seed.rollout(genome, wk, dk, 0, c, _world_protocol=reversal.PROTOCOL, **options)
        changed = seed.rollout(genome, wk, dk, 0, c, _world_protocol=reversal.PROTOCOL, flip_step=3, **options)
        for name in ("observation", "action", "action_key", "activity", "efference"):
            self.assertTrue(reversal._same_bytes(stationary["_world"][name][:4], changed["_world"][name][:4]))
        self.assertEqual(changed["_world"]["observation"].shape[-1], seed.dims(c)[1])
        np.testing.assert_array_equal(changed["_belief"]["q"], .75)

    def test_direct_baseline_remains_fixed_under_the_world_extension(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        weights = direct.decode(direct.init_genome(c, jax.random.PRNGKey(55)), c)
        options = dict(direct_weights=weights, plastic=False, predictor_plastic=False, record=True, diagnostics=True)
        result = seed.rollout(None, jax.random.PRNGKey(55), jax.random.PRNGKey(56), 0, c,
                              flip_step=3, _world_protocol=reversal.PROTOCOL, **options)
        np.testing.assert_array_equal(result["_world_initial"]["W"][0], weights)
        np.testing.assert_array_equal(result["_world_final"]["W"][0], weights)
        np.testing.assert_array_equal(result["_world_final"]["Wp"], 0)

    def test_development_family_preserves_pairing_and_compacts_every_assigned_endpoint(self):
        c = seed.Cfg(G=4, M=1, T=300, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        weights = np.asarray(direct.decode(direct.init_genome(c, jax.random.PRNGKey(55)), c))
        row = reversal.evaluate_family(np.asarray(flat), weights, 2., reversal.family_keys(2026120300, 0),
                                        0, 1, 100, c)
        compact = reversal.summary_input(row)
        self.assertLess(sum(value.nbytes for value in compact.values()), 10000)
        self.assertTrue(all(bool(row[arm + "_preflip_identical"]) for arm in reversal.ARMS))
        self.assertEqual(int(row["rule"]), 1)  # Assignment is not silently derived from family index.
        self.assertIsNone(reversal.summarize([compact])["gate"]["passed"])
        changed = copy.deepcopy(row)
        changed["observed_reversal_action"][99, 0] = (int(changed["observed_reversal_action"][99, 0]) + 1) % 5
        with self.assertRaises(ValueError):
            reversal.validate_family(changed, c)


if __name__ == "__main__":
    unittest.main()
