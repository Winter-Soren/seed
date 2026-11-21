"""Trait plumbing and checkpoint compatibility without additional simulated lives."""
import unittest
from unittest.mock import patch

import numpy as np

from seed import lineage, seed
from seed.ecology import EcologyV1
from seed.persistence import PersistenceV1


class LineagePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.c = seed.Cfg(G=4, M=2, T=4, R=0, H=3, n_items=0)
        self.ecology = EcologyV1()
        _, _, _, n, K = seed.dims(self.c, ecology=self.ecology)
        self.genome = dict(conn=np.zeros((K, K), np.float32), wscale=np.zeros((K, K), np.float32),
                          hebb=np.zeros((K, K, 4), np.float32), log_eta=np.array(0., np.float32),
                          mod=np.zeros(3, np.float32), log_plr=np.array(0., np.float32))
        self.worlds = np.array([[41, 1], [41, 2]], np.uint32)
        self.development = np.array([[42, 1], [42, 2]], np.uint32)
        self.birth = np.zeros((self.c.M, n, n), np.float32)
        self.mask = np.ones_like(self.birth)
        self.trait = PersistenceV1()

    def rollout(self, g, wkey, dkey, rule, c, **kwargs):
        initial = self.birth if kwargs["_initial_weights"] is None else kwargs["_initial_weights"]
        alive = np.ones(c.M, np.float32)
        return dict(fitness=np.array(0.), counts=np.zeros((2, 2)), survived=np.array(1.),
                    pursuit_state=dict(target=np.full((c.M, 2), 7), stalled=np.full(c.M, 2)),
                    _weight_state=dict(birth_W=self.birth, initial_W=initial.copy(),
                                       terminal_W=initial.copy(), mask=self.mask, alive=alive))

    def cohort(self, *, index=0, **kwargs):
        return lineage.collective_lives(self.genome, self.worlds[index:index + 1],
                                          self.development[index:index + 1], [0], self.c,
                                          ecologies=[self.ecology], **kwargs)

    def test_fixed_trait_reaches_both_arms_with_identical_keys_and_no_transient_state(self):
        with patch.object(lineage, "_birth", return_value=(self.birth, self.mask)), \
                patch.object(seed, "rollout", side_effect=self.rollout) as simulation:
            off = self.cohort()
            off_calls = simulation.call_args_list[:]
            simulation.reset_mock()
            on = self.cohort(persistence=self.trait)
            for empty, enabled in zip(off_calls, simulation.call_args_list):
                self.assertNotIn("persistence", empty.kwargs)
                self.assertIs(enabled.kwargs["persistence"], self.trait)
                np.testing.assert_array_equal(empty.args[1], enabled.args[1])
                np.testing.assert_array_equal(empty.args[2], enabled.args[2])
            simulation.reset_mock()
            continued = lineage.same_individual(self.genome, self.worlds, self.development[0],
                                                 [0, 0], self.c,
                                                 ecologies=[self.ecology, self.ecology],
                                                 persistence=self.trait)
            self.assertEqual(simulation.call_count, 3)
            allowed = {"plastic", "diagnostics", "ecology", "_initial_weights", "_initial_alive",
                       "_return_weights", "persistence"}
            for call in simulation.call_args_list:
                self.assertEqual(set(call.kwargs), allowed)
                self.assertIs(call.kwargs["persistence"], self.trait)
        self.assertNotIn("persistence", off)
        self.assertNotIn("persistence", off["archive_state"])
        expected = dict(version="persistence-v1", memory_steps=4, stall_steps=2)
        self.assertEqual(on["archive_state"]["persistence"], expected)
        self.assertEqual(on["lives"][0]["persistence"], expected)
        self.assertTrue(all(row["persistence"] == expected for row in continued["lives"]))

    def test_old_off_archive_and_enabled_archive_resume_without_mixing_settings(self):
        with patch.object(lineage, "_birth", return_value=(self.birth, self.mask)), \
                patch.object(seed, "rollout", side_effect=self.rollout) as simulation:
            old = self.cohort()["archive_state"]
            enabled = self.cohort(persistence=self.trait)["archive_state"]
            self.assertNotIn("persistence", old)
            self.assertEqual(self.cohort(index=1, archive_state=old)["archive_state"]["next_life"], 2)
            resumed = self.cohort(index=1, archive_state=enabled, persistence=self.trait)
            self.assertEqual(resumed["archive_state"]["persistence"], enabled["persistence"])
            for state, setting in ((old, self.trait), (enabled, None),
                                   (enabled, PersistenceV1(memory_steps=5, stall_steps=2))):
                before = simulation.call_count
                with self.assertRaisesRegex(ValueError, "persistence setting"):
                    self.cohort(index=1, archive_state=state, persistence=setting)
                self.assertEqual(simulation.call_count, before)


if __name__ == "__main__":
    unittest.main()
