"""Synthetic E134 scoring and matched-history contracts."""
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed, encounter_diagnostic as encounter, replay_selection as study


class ReplaySelectionAccountingTests(unittest.TestCase):
    def test_uniform_and_fixed_specialist_are_zero_when_rules_balance(self):
        uniform = np.full((12, 5), .2)
        self.assertEqual(float(study.signed_local(uniform, 0)), 0.)
        specialist = np.zeros((12, 5))
        specialist[np.arange(12), encounter.PLACEMENTS[:, 0]] = 1.
        self.assertEqual(float(study.signed_local(specialist, 0)), 1.)
        self.assertEqual(float(study.signed_local(specialist, 1)), -1.)
        current = np.zeros((12, 5))
        current[np.arange(12), encounter.PLACEMENTS[:, 1]] = 1.
        self.assertEqual(float(study.signed_local(current, 1) - study.signed_local(specialist, 1)), 2.)
        self.assertEqual(float(study.signed_local(current, 1) - study.signed_local(current, 1)), 0.)

    def test_balanced_script_energy_bound_includes_sham_and_longest_gap(self):
        for pre in (8, 16):
            script = encounter.contact_script(jax.random.PRNGKey(17), pre)
            for initial in (0, 1):
                for later in (0, 1):
                    energy = 1.
                    energies = []
                    for n, row in enumerate(script):
                        rule = initial if n < pre else later
                        energy += (.25 if row[2] == 1 + rule else -.25) - .003
                        energies.append(energy)
                        energy -= .003
                        energies.append(energy)
                    for _ in range(32):
                        energy -= .003
                        energies.append(energy)
                    self.assertGreaterEqual(min(energies), .462 - 1e-12)
                    self.assertLessEqual(max(energies), 1.25)

    def test_training_batch_preserves_all_schedule_cells(self):
        keys = jax.random.split(jax.random.PRNGKey(31), 4)
        batch = study.training_batch(keys)
        self.assertEqual(sorted(zip(batch["pre_contacts"], batch["gap"])), list(map(tuple, study.TRAIN_CELLS)))
        for i, pre in enumerate(batch["pre_contacts"]):
            actual = batch["scripts"][i, :int(pre) + 16]
            expected = encounter.contact_script(batch["role_keys"][i, 0], int(pre))
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            study.training_batch(np.zeros((4, 2), np.uint32))

    def test_first_item_counts_lethal_poison_and_missing_contact(self):
        row = dict(base_index=0, old_probabilities=np.full((2, 2, 12, 5), .2),
                   local_probabilities=np.full((2, 2, 2, 2, 12, 5), .2),
                   solo_food=np.zeros((2, 2, 2, 50, 1), bool),
                   solo_poison=np.zeros((2, 2, 2, 50, 1), bool),
                   solo_alive_after=np.ones((2, 2, 2, 50, 1)))
        row["solo_poison"][1, 0, 1, 3, 0] = True
        row["solo_alive_after"][1, 0, 1, 3:, 0] = 0
        row["solo_food"][0, 0, 0, 7, 0] = True
        result = study.summary_input(row)
        self.assertEqual(result["solo_Y"][1, 0, 1], -1)
        self.assertEqual(result["solo_Y"][0, 0, 0], 1)
        self.assertEqual(result["solo_Y"][0, 1, 1], 0)
        self.assertEqual(result["contact_latency"][0, 1, 1], 50)


class ReplaySelectionDynamicsTests(unittest.TestCase):
    def test_complete_family_pairs_yoke_and_preserves_free_actions(self):
        flat = np.asarray(ravel_pytree(seed.biology_prior(study.CFG))[0])
        keys = np.asarray(jax.random.split(jax.random.PRNGKey(0xE134F2), 6))
        row = study.evaluate_family(flat, flat, keys, 0, 8, 16, 1)
        self.assertEqual(row["local_probabilities"].shape, (2, 2, 2, 2, 12, 5))
        self.assertEqual(row["exposure_energy"].shape, (2, 2, 2, 2, 64, 1))
        self.assertEqual(row["target_grid"].shape, (16, 16))
        self.assertEqual(row["solo_grid"].shape, (2, 2, 2, 50, 16, 16))
        self.assertEqual(row["local_probabilities"][0].tobytes(), row["local_probabilities"][1].tobytes())
        np.testing.assert_array_equal(row["solo_forced_action"], -1)
        np.testing.assert_array_equal(row["solo_action"], row["solo_sampled_action"])

    def test_training_score_matches_native_traced_reference(self):
        flat = ravel_pytree(seed.biology_prior(study.CFG))[0]
        keys = jax.random.split(jax.random.PRNGKey(0xE134F0), 4)
        script = jnp.asarray(encounter.contact_script(keys[0], 8))
        fast = jax.jit(lambda value: study._score_family(value, script, keys[1], keys[2], keys[3], 8, 0))(flat)
        references = []
        probe_keys = jax.random.split(jax.random.fold_in(keys[3], 1), 12)
        for initial in (0, 1):
            birth = encounter._birth(flat, "plastic", keys[1], keys[2], study.CFG)
            old, _ = encounter._chamber("plastic", study.CFG)(flat, birth, script[:8], initial)
            histories = []
            for history in (0, 1):
                final, _ = encounter._chamber("plastic", study.CFG)(flat, old, script[8:], initial ^ history)
                result = encounter._readouts("plastic", study.CFG)(flat, final, probe_keys)
                histories.append(np.asarray(result["probabilities"][:, 0]))
            references.append(np.stack(histories))
        expected = np.stack(references)
        self.assertEqual(np.asarray(fast["probabilities"]).tobytes(), expected.tobytes())
        score = np.mean([study.signed_local(expected[i, h], i ^ h) for i in (0, 1) for h in (0, 1)])
        self.assertAlmostEqual(float(fast["fitness"]), float(score), places=6)
        np.testing.assert_array_equal(np.asarray(fast["counts"]), [[4, 4], [8, 8]])
        self.assertEqual(float(fast["survived"]), 1.)

    def test_empty_gap_clears_items_and_uses_native_empty_stays(self):
        flat = ravel_pytree(seed.biology_prior(study.CFG))[0]
        keys = jax.random.split(jax.random.PRNGKey(0xE134F1), 2)
        state = encounter._birth(flat, "plastic", keys[0], keys[1], study.CFG)
        state["grid"] = state["grid"].at[8, 8].set(1)
        final, trace = study._empty_gap(2)(flat, state, 0)
        self.assertTrue(np.all(np.asarray(trace["grid_before"]) == 0))
        self.assertTrue(np.all(np.asarray(trace["forced_action"]) == 0))
        self.assertFalse(np.asarray(trace["food"]).any())
        self.assertFalse(np.asarray(trace["poison"]).any())
        self.assertAlmostEqual(float(final["energy"][0]), .994, places=6)


if __name__ == "__main__":
    unittest.main()
