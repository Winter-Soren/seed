"""Synthetic E138 contracts with no registered family assignment or outcome reads."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import encounter_diagnostic as encounter
from seed import replay_selection as replay
from seed import scene_transfer as scene
from seed import seed
from seed import solo_selection as study


def cold_trace_check():
    """Trace real population shapes from uninitialized native factories."""
    assert not scene._first_decisions.cache_info().currsize
    assert not encounter._unravel.cache_info().currsize
    cells = ((8, 0), (16, 8))
    args = (jax.ShapeDtypeStruct((16, 155), np.float32),
            jax.ShapeDtypeStruct((32, 5), np.int32),
            jax.ShapeDtypeStruct((16, 2), np.uint32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((16, 16), np.int32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((16, 16), np.int32),
            jax.ShapeDtypeStruct((1, 2), np.int32),
            jax.ShapeDtypeStruct((2,), np.uint32))
    for pre, gap in cells:
        with jax.checking_leaks():
            found = jax.eval_shape(study._population_cell(pre, gap), *args)
        assert found["probabilities"].shape == (16, 2, 2, 12, 5)
        assert found["view_fitness"].shape == (16, 2)
        assert found["solo_logits"].shape == (16, 2, 2, 50, 5)
        for name in study.SOLO_AGENT_FIELDS + study.SOLO_FLAGS:
            assert found["solo_" + name].shape == (16, 2, 2, 50)
    return dict(status="passed", cells=cells, cold_cache=True, abstract_trace_only=True,
                executed_rollouts=0, eval_keys_created=False)


class SoloSelectionTests(unittest.TestCase):
    def test_cold_process_traces_both_schedule_shapes(self):
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--cold-trace"],
            cwd=ROOT, text=True, capture_output=True, check=True, timeout=60)
        proof = json.loads(result.stdout)
        self.assertEqual(proof["status"], "passed")
        self.assertEqual(proof["cells"], [[8, 0], [16, 8]])
        self.assertTrue(proof["cold_cache"])
        self.assertEqual(proof["executed_rollouts"], 0)

    def test_first_contact_preserves_lethal_contact_and_no_contact(self):
        food = np.zeros((3, 50), bool)
        poison = np.zeros_like(food)
        poison[0, 4] = True
        food[1, 9] = True
        poison[1, 12] = True
        # Alive-after is deliberately absent. Death at the first contact must
        # never erase the food/poison event already earned by that action.
        actual = study.first_item_score(jnp.asarray(food), jnp.asarray(poison))
        np.testing.assert_array_equal(actual, [-1, 1, 0])

    def test_invalid_objective_and_duplicate_family_roots_fail(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            study.make_evaluator("pick-after-results")
        with self.assertRaisesRegex(ValueError, "distinct"):
            study.training_batch(np.zeros((4, 2), np.uint32))

    def test_only_assigned_score_enters_reversed_objective_ordering(self):
        candidates = np.zeros((4, 155), np.float32)
        births = np.zeros((4, 4, 2), np.uint32)
        batch = dict(pre_contacts=study.TRAIN_CELLS[:, 0], gap=study.TRAIN_CELLS[:, 1],
            role_keys=np.zeros((4, 6, 2), np.uint32), scripts=np.zeros((4, 32, 5), np.int32),
            backgrounds=np.zeros((4, 16, 16), np.int32), solo_grids=np.zeros((4, 16, 16), np.int32),
            solo_positions=np.zeros((4, 1, 2), np.int32))
        both = np.asarray([[-1, 1], [-1, 1], [1, -1], [1, -1]], np.float32)
        scalar_shape = (4, 2, 2, 50)
        raw = dict(view_fitness=jnp.asarray(both), survived=jnp.ones(4),
            exposure_energy_bounds=jnp.tile(jnp.asarray([.5, 1.25]), (4, 1)),
            solo_action=jnp.zeros(scalar_shape, jnp.int32), solo_sampled_action=jnp.zeros(scalar_shape, jnp.int32),
            solo_forced_action=jnp.full(scalar_shape, -1, jnp.int32),
            solo_hebbian_enabled=jnp.ones(scalar_shape, bool), solo_predictor_enabled=jnp.ones(scalar_shape, bool))
        with patch.object(study, "_population_cell", return_value=lambda *args: raw):
            background = study.make_evaluator("background")(candidates, batch, births)
            solo = study.make_evaluator("solo")(candidates, batch, births)
        for name in raw:
            np.testing.assert_array_equal(background[name], solo[name], err_msg=name)
        np.testing.assert_array_equal(background["fitness"], np.broadcast_to(both[:, :1], (4, 4)))
        np.testing.assert_array_equal(solo["fitness"], -background["fitness"])

    def test_training_materialization_keeps_grid_and_action_roles_separate(self):
        roots = np.asarray(jax.random.split(jax.random.PRNGKey(0xE138F1), 4))
        batch = study.training_batch(roots)
        self.assertEqual(batch["role_keys"].shape, (4, 6, 2))
        self.assertEqual(len({key.tobytes() for key in batch["role_keys"].reshape(-1, 2)}), 24)
        self.assertEqual(sorted(zip(batch["pre_contacts"], batch["gap"])), list(map(tuple, study.TRAIN_CELLS)))
        for family, pre in enumerate(batch["pre_contacts"]):
            roles = jax.random.split(roots[family], 6)
            np.testing.assert_array_equal(batch["role_keys"][family], roles)
            grid, position = seed.init_world(roles[4], study.CFG)
            np.testing.assert_array_equal(batch["solo_grids"][family], grid)
            np.testing.assert_array_equal(batch["solo_positions"][family], position)
            np.testing.assert_array_equal(batch["scripts"][family, :int(pre) + 16],
                                          encounter.contact_script(roles[0], int(pre)))

    def test_dense_probe_and_solo_trace_match_native_unprobed_clones(self):
        flat = np.asarray(ravel_pytree(seed.biology_prior(study.CFG))[0])
        keys = jax.random.split(jax.random.PRNGKey(0xE138F2), 7)
        script = encounter.contact_script(keys[0], 8)
        padded = np.zeros((32, 5), np.int32)
        padded[:len(script)] = script
        background = scene._background()(keys[3])
        grid, position = seed.init_world(keys[5], study.CFG)
        compiled = study._population_cell(8, 0)(jnp.asarray(flat[None]), jnp.asarray(padded), keys[1][None],
            keys[2], background, keys[4], grid, position, keys[6])
        actual = jax.tree.map(lambda value: np.asarray(value)[0], compiled)
        local_scores, solo_scores = [], []
        for initial in (0, 1):
            birth = encounter._birth(flat, "plastic", keys[1], keys[2], study.CFG)
            boundary, _ = encounter._chamber("plastic", study.CFG)(flat, birth, jnp.asarray(script[:8]), initial)
            for history in (0, 1):
                target = initial ^ history
                delayed, _, _ = replay._training_history(flat, boundary, jnp.asarray(script[8:]), target, 0)
                _, first = scene._first_decisions()(flat, delayed, background, jax.random.split(keys[4], 12), target)
                np.testing.assert_array_equal(actual["probabilities"][initial, history], first["probabilities"][1, :, 0, :])
                np.testing.assert_array_equal(actual["logits"][initial, history], first["logits"][1, :, 0, :])
                np.testing.assert_array_equal(actual["sampled_action"][initial, history], first["sampled_action"][1, :, 0])
                np.testing.assert_array_equal(actual["probe_action_keys"][initial, history], first["action_key"][1])
                fresh = encounter._fresh_body(delayed, grid, position, keys[6], study.CFG)
                _, trace = encounter._free_grid("plastic", study.CFG, steps=50)(flat, fresh, target)
                for name in study.SOLO_AGENT_FIELDS:
                    np.testing.assert_array_equal(actual["solo_" + name][initial, history], trace[name][:, 0], err_msg=name)
                for name in study.SOLO_FLAGS:
                    np.testing.assert_array_equal(actual["solo_" + name][initial, history], trace[name], err_msg=name)
                np.testing.assert_array_equal(actual["solo_logits"][initial, history], trace["logits"][:, 0, :])
                local_scores.append(replay.signed_local(np.asarray(first["probabilities"])[1, :, 0, :], target))
                food, poison = np.asarray(trace["food"][:, 0]), np.asarray(trace["poison"][:, 0])
                contacts = np.flatnonzero(food | poison)
                solo_scores.append(0 if not len(contacts) else int(food[contacts[0]]) - int(poison[contacts[0]]))
        np.testing.assert_allclose(actual["view_fitness"][0], np.mean(local_scores), rtol=1e-6, atol=1e-7)
        self.assertEqual(actual["view_fitness"][1], np.mean(solo_scores))
        np.testing.assert_array_equal(actual["counts"], [[4, 4], [8, 8]])
        self.assertEqual(float(actual["survived"]), 1.)
        np.testing.assert_array_equal(actual["solo_forced_action"], -1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--cold-trace"]:
        print(json.dumps(cold_trace_check()))
    else:
        unittest.main()
