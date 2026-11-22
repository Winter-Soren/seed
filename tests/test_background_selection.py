"""Synthetic contracts for E136; registered final assignments are never opened."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import background_selection as study
from seed import encounter_diagnostic as encounter
from seed import scene_transfer as scene
from seed import seed


class BackgroundSelectionAccountingTests(unittest.TestCase):
    def test_current_safe_score_and_all_assigned_first_contact(self):
        shape = (3, 2, 2, 2, 12, 1, 5)
        probabilities = np.full(shape, .2)
        actions = np.zeros(shape[:-1], np.int32)
        for initial in (0, 1):
            for history in (0, 1):
                target = initial ^ history
                for k, placement in enumerate(encounter.PLACEMENTS):
                    probabilities[2, initial, history, 1, k, 0] = 0
                    probabilities[2, initial, history, 1, k, 0, placement[target]] = 1
                    actions[2, initial, history, 1, k, 0] = placement[target]
        row = dict(base_index=0, background=np.zeros((16, 16), np.int32),
                   first_probabilities=probabilities, first_action=actions,
                   solo_food=np.zeros((3, 2, 2, 50, 1), bool),
                   solo_poison=np.zeros((3, 2, 2, 50, 1), bool),
                   solo_alive_after=np.ones((3, 2, 2, 50, 1)))
        row["solo_food"][2, 0, 0, 7, 0] = True
        row["solo_poison"][2, 0, 1, 3, 0] = True
        row["solo_alive_after"][2, 0, 1, 3:, 0] = 0
        result = study.summary_input(row)
        np.testing.assert_array_equal(result["local"][2, ..., 1], 1)
        np.testing.assert_array_equal(result["sampled"][2, ..., 1], 1)
        np.testing.assert_allclose(sum(result["decomposition"].values()), result["local"][..., 0] - result["local"][..., 1])
        self.assertEqual(result["solo_Y"][2, 0, 0], 1)
        self.assertEqual(result["solo_Y"][2, 0, 1], -1)
        self.assertEqual(result["solo_survival"][2, 0, 1], 0)
        self.assertEqual(result["solo_Y"][2, 1, 1], 0)
        self.assertTrue(result["no_contact"][2, 1, 1])
        self.assertEqual(result["contact_latency"][2, 1, 1], 50)

    def test_unrecognized_objective_and_reused_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            study.make_evaluator("selected-after-results")
        with self.assertRaisesRegex(ValueError, "distinct"):
            study.training_batch(np.zeros((4, 2), np.uint32))

    def test_reversed_view_ordering_is_routed_without_a_rollout(self):
        # Tied groups reverse ordering between views. The unused objective must
        # never be averaged into the scalar consumed by rank/ES.
        candidates = np.zeros((4, 155), np.float32)
        births = np.zeros((4, 4, 2), np.uint32)
        batch = dict(pre_contacts=study.TRAIN_CELLS[:, 0], gap=study.TRAIN_CELLS[:, 1],
                     role_keys=np.zeros((4, 4, 2), np.uint32),
                     scripts=np.zeros((4, 32, 5), np.int32),
                     backgrounds=np.zeros((4, 16, 16), np.int32))
        both = np.asarray([[-1, 1], [-1, 1], [1, -1], [1, -1]], np.float32)
        raw = dict(view_fitness=jnp.asarray(both), survived=jnp.ones(4),
                   exposure_energy_bounds=jnp.tile(jnp.asarray([.5, 1.25]), (4, 1)))
        with patch.object(study, "_population_cell", return_value=lambda *args: raw):
            isolated = study.make_evaluator("isolated")(candidates, batch, births)
            background = study.make_evaluator("background")(candidates, batch, births)
        np.testing.assert_array_equal(isolated["view_fitness"], background["view_fitness"])
        np.testing.assert_array_equal(isolated["fitness"], np.broadcast_to(both[:, :1], (4, 4)))
        np.testing.assert_array_equal(background["fitness"], -isolated["fitness"])


class BackgroundSelectionDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat = np.asarray(ravel_pytree(seed.biology_prior(study.CFG))[0])

    def test_training_reuses_e135_views_and_native_isolated_readout(self):
        keys = jax.random.split(jax.random.PRNGKey(0xE136F0), 5)
        script = jnp.asarray(encounter.contact_script(keys[0], 8))
        background = scene._background()(keys[3])
        fast = jax.jit(lambda value: study._score_family(
            value, script, keys[1], keys[2], background, keys[4], 8, 0))(jnp.asarray(self.flat))
        probe_keys = jax.random.split(keys[4], 12)
        references = []
        expected_actions = []
        for initial in (0, 1):
            birth = encounter._birth(self.flat, "plastic", keys[1], keys[2], study.CFG)
            boundary, _ = encounter._chamber("plastic", study.CFG)(self.flat, birth, script[:8], initial)
            histories, actions = [], []
            for history in (0, 1):
                target = initial ^ history
                acquired, _ = encounter._chamber("plastic", study.CFG)(self.flat, boundary, script[8:], target)
                acquired, _ = scene.replay._empty_gap(0)(self.flat, acquired, target)
                _, decision = scene._first_decisions()(self.flat, acquired, background, probe_keys, target)
                legacy = encounter._readouts("plastic", study.CFG)(self.flat, acquired, probe_keys)
                np.testing.assert_array_equal(decision["observation"][0], legacy["observation"])
                np.testing.assert_allclose(decision["probabilities"][0], legacy["probabilities"], rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(decision["logits"][0], legacy["logits"], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(decision["action"][0], legacy["sampled_action"])
                histories.append(np.asarray(decision["probabilities"][..., 0, :]))
                actions.append(np.asarray(decision["action"][..., 0]))
            references.append(np.stack(histories))
            expected_actions.append(np.stack(actions))
        probabilities = np.stack(references)
        np.testing.assert_array_equal(fast["probabilities"], probabilities)
        np.testing.assert_array_equal(fast["sampled_action"], np.stack(expected_actions))
        scores = np.mean([study.signed_local(probabilities[i, h], i ^ h) for i in (0, 1) for h in (0, 1)], axis=0)
        np.testing.assert_allclose(fast["view_fitness"], scores, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(fast["counts"], [[4, 4], [8, 8]])
        self.assertEqual(float(fast["survived"]), 1.)

    def test_objective_selection_changes_only_the_selected_fitness(self):
        keys = jax.random.split(jax.random.PRNGKey(0xE136F1), 8)
        batch = study.training_batch(np.asarray(keys[:4]))
        candidates = np.stack((self.flat, self.flat))
        births = np.broadcast_to(np.asarray(keys[4:]), (2, 4, 2)).copy()
        isolated = study.make_evaluator("isolated")(candidates, batch, births)
        background = study.make_evaluator("background")(candidates, batch, births)
        self.assertEqual(isolated["probabilities"].shape, (2, 4, 2, 2, 2, 12, 5))
        self.assertEqual(isolated["probe_action_keys"].shape, (2, 4, 2, 2, 2, 12, 2))
        for name in isolated:
            if name != "fitness":
                np.testing.assert_array_equal(isolated[name], background[name], err_msg=name)
        np.testing.assert_array_equal(isolated["fitness"], isolated["view_fitness"][..., 0])
        np.testing.assert_array_equal(background["fitness"], background["view_fitness"][..., 1])
        self.assertEqual(sorted(zip(batch["pre_contacts"], batch["gap"])), list(map(tuple, study.TRAIN_CELLS)))
        for i, pre in enumerate(batch["pre_contacts"]):
            np.testing.assert_array_equal(batch["scripts"][i, :int(pre) + 16],
                                          encounter.contact_script(batch["role_keys"][i, 0], int(pre)))
            np.testing.assert_array_equal(batch["backgrounds"][i], scene._background()(batch["role_keys"][i, 2]))

    def test_final_clones_share_histories_and_keep_every_solo_move_free(self):
        keys = np.asarray(jax.random.split(jax.random.PRNGKey(0xE136F2), 7))
        row = study.evaluate_family(self.flat, self.flat, self.flat, keys, 0, 8, 16)
        self.assertEqual(row["first_probabilities"].shape, (3, 2, 2, 2, 12, 1, 5))
        self.assertEqual(row["exposure_energy"].shape, (3, 2, 2, 64, 1))
        self.assertEqual(row["solo_grid"].shape, (3, 2, 2, 50, 16, 16))
        for name, values in row.items():
            if name.startswith(("exposure_", "acquired_", "first_", "solo_")):
                np.testing.assert_array_equal(values[0], values[1], err_msg=name)
                np.testing.assert_array_equal(values[0], values[2], err_msg=name)
        np.testing.assert_array_equal(row["first_action"], row["first_sampled_action"])
        np.testing.assert_array_equal(row["solo_action"], row["solo_sampled_action"])
        np.testing.assert_array_equal(row["solo_forced_action"], -1)
        for name in encounter.NEURAL:
            np.testing.assert_array_equal(row["solo_initial_" + name + "_sha256"], row["acquired_" + name + "_sha256"])
        corrupted = dict(row, solo_forced_action=row["solo_forced_action"].copy())
        corrupted["solo_forced_action"][0, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "imposed"):
            study.validate_family(corrupted)


if __name__ == "__main__":
    unittest.main()
