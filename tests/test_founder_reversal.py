"""E130 accounting and native-rollout fixtures, with no experimental data reads."""
import subprocess
import types
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed, founder_reversal as assessment


HISTORICAL_COMMIT = "2d2586735061ac20481d6904f86791affb58b587"
SYNTHETIC_MASTER = 0xE130F0


def same_bytes(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.tobytes(), expected.tobytes())


def empty_events():
    return dict(food=np.zeros((2, 300, 4), bool), poison=np.zeros((2, 300, 4), bool),
                alive_before=np.ones((2, 300, 4), bool), alive_after=np.ones((2, 300, 4), bool))


def historical_engine():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(["git", "show", HISTORICAL_COMMIT + ":seed/seed.py"], cwd=root).decode("utf-8")
    module = types.ModuleType("historical_seed_e130")
    module.__file__ = HISTORICAL_COMMIT + ":seed/seed.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class FounderReversalAccountingTests(unittest.TestCase):
    def test_first_item_is_not_net_count_and_dead_animals_keep_their_slots(self):
        events = empty_events()
        # Agent0's first item is poison despite a positive total return in bin3.
        events["poison"][0, 150, 0] = True
        events["food"][0, [151, 152], 0] = True
        # Agent1 dies in the preceding bin: later zero-contact bins still count.
        events["poison"][0, 149, 1] = True
        events["alive_after"][0, 149:, 1] = False
        events["alive_before"][0, 150:, 1] = False
        # The boundary belongs to [150,200), not the preceding window.
        events["food"][0, 149, 2] = True
        events["poison"][0, 150, 2] = True
        events["food"][0, 299, 3] = True
        events["poison"][1, 0, 3] = True
        events["alive_after"][1, :, 3] = False
        events["alive_before"][1, 1:, 3] = False
        result = assessment.bin_events(events)
        self.assertEqual(result["Q_by_rule"].shape, (2, 6, 4))
        self.assertEqual(result["Q_by_rule"].dtype, np.int8)
        self.assertEqual(result["Q_by_rule"][0, 3, 0], -1)
        self.assertEqual(result["food_count_by_rule"][0, 3, 0], 2)
        self.assertEqual(result["poison_count_by_rule"][0, 3, 0], 1)
        np.testing.assert_array_equal(result["Q_by_rule"][0, 3:, 1], 0)
        np.testing.assert_array_equal(result["no_contact_by_rule"][0, 3:, 1], True)
        np.testing.assert_array_equal(result["alive_start_by_rule"][0, 3:, 1], False)
        self.assertEqual(result["Q_by_rule"][0, 2, 2], 1)
        self.assertEqual(result["Q_by_rule"][0, 3, 2], -1)
        self.assertEqual(result["Q_by_rule"][0, 5, 3], 1)
        self.assertEqual(result["Q_by_rule"][1, 0, 3], -1)
        np.testing.assert_array_equal(result["Q_by_rule"][1, 1:, 3], 0)
        # The eight assigned animal/rule replays remain in this family average.
        self.assertEqual(result["Q_by_rule"][:, 3].mean(), -.25)
        self.assertEqual(result["final_survival_by_rule"].shape, (2, 4))

    def test_event_validation_rejects_replacement_conflicting_and_postdeath_meals(self):
        conflict = empty_events()
        conflict["food"][0, 10, 0] = conflict["poison"][0, 10, 0] = True
        with self.assertRaises(ValueError):
            assessment.bin_events(conflict)
        resurrection = empty_events()
        resurrection["alive_after"][0, 10, 0] = False
        resurrection["alive_before"][0, 11, 0] = False
        with self.assertRaises(ValueError):
            assessment.bin_events(resurrection)
        after_death = empty_events()
        after_death["alive_after"][0, 10:, 0] = False
        after_death["alive_before"][0, 11:, 0] = False
        after_death["food"][0, 50, 0] = True
        with self.assertRaises(ValueError):
            assessment.bin_events(after_death)
        omitted = {name: value[:, :, :3] for name, value in empty_events().items()}
        with self.assertRaises(ValueError):
            assessment.bin_events(omitted)

    def test_family_key_pairing_is_deterministic_and_dimensions_are_native(self):
        manifest = assessment.assignments_manifest(SYNTHETIC_MASTER, 4)
        self.assertEqual(set(manifest), {"family_keys", "initial_rules"})
        self.assertEqual(manifest["family_keys"].shape, (4, 2, 2))
        self.assertEqual(len({key.tobytes() for family in manifest["family_keys"] for key in family}), 8)
        for index, family in enumerate(manifest["family_keys"]):
            same_bytes(self, family, assessment.evaluation_keys(SYNTHETIC_MASTER, index))
        for index in (-1, True, assessment.ASSIGNMENT_TAG):
            with self.assertRaises(ValueError):
                assessment.evaluation_keys(SYNTHETIC_MASTER, index)
        with self.assertRaises(ValueError):
            assessment._inputs(manifest["family_keys"][0], 0, seed.Cfg(M=1))
        duplicate = manifest["family_keys"][0].copy()
        duplicate[1] = duplicate[0]
        with self.assertRaises(ValueError):
            assessment._inputs(duplicate, 0, assessment.CFG)


class FounderReversalNativeTests(unittest.TestCase):
    def test_native_lifetimes_match_pinned_engine_and_physical_flip_labels(self):
        c = assessment.CFG
        old = historical_engine()
        source_key = jax.random.PRNGKey(SYNTHETIC_MASTER)
        # Synthetic encodings exercise both interfaces; these are not saved
        # founders/descendants and supply no scientific comparison or tuning.
        founders = np.asarray(ravel_pytree(seed.init_genome(c, source_key))[0])
        descendants = np.asarray(ravel_pytree(seed.biology_prior(c))[0])
        keys = assessment.evaluation_keys(SYNTHETIC_MASTER, 0)
        row = assessment.evaluate_world(founders, descendants, keys, 0)
        for stage, flat in (("gen0", founders), ("gen200", descendants)):
            genome = assessment._unravel(c)(jnp.asarray(flat))
            for condition in ("stationary", "reversal"):
                for rule in (0, 1):
                    raw = old.rollout(genome, jnp.asarray(keys[0]), jnp.asarray(keys[1]), jnp.asarray(rule, jnp.int32),
                        c, plastic=True, record=False, predictor_plastic=True, structural=False,
                        flip_step=150 if condition == "reversal" else None, diagnostics=True)
                    prefix = stage + "_" + condition + "_"
                    for event in assessment.EVENTS:
                        same_bytes(self, row[prefix + event][rule], np.asarray(raw["events"][event], bool))
                    for name, field in (("native_counts", "counts"), ("native_fitness", "fitness"), ("native_survived", "survived")):
                        same_bytes(self, row[prefix + name][rule], raw[field])
        compact = assessment.summary_input(row)
        for arm in assessment.ARMS:
            self.assertEqual(compact[arm + "_Q_by_rule"].shape, (2, 6, 4))
        # Separate historical recorded life makes the labeling check physical:
        # previous grid + actual destination + prior alive state determine eating.
        # Recording is not needed or stored by the production assessment.
        raw = old.rollout(assessment._unravel(c)(jnp.asarray(descendants)), jnp.asarray(keys[0]), jnp.asarray(keys[1]),
            jnp.asarray(0, jnp.int32), c, plastic=True, record=True, predictor_plastic=True,
            structural=False, flip_step=150, diagnostics=True)
        initial_grid, _ = old.init_world(jnp.asarray(keys[0]), c)
        grids = np.concatenate((np.asarray(initial_grid)[None], np.asarray(raw["traj"]["grid"])[:-1]))
        positions = np.asarray(raw["traj"]["pos"])
        item = grids[np.arange(c.T)[:, None], positions[..., 0], positions[..., 1]]
        before = np.asarray(raw["events"]["alive_before"], bool)
        current_rule = (np.arange(c.T) >= 150).astype(np.int32)[:, None]
        np.testing.assert_array_equal(raw["events"]["food"], before & (item == 1 + current_rule))
        np.testing.assert_array_equal(raw["events"]["poison"], before & (item == 2 - current_rule))


if __name__ == "__main__":
    unittest.main()
