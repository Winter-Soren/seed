"""Small development fixtures for inheritance and explicit matched controls."""
import copy
import io
import json
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from seed import lineage, seed
from seed.ecology import EcologyV1


class LineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = seed.Cfg(G=4, M=2, T=4, R=0, H=3, n_items=0, decay=0.)
        cls.keys = jax.random.split(jax.random.PRNGKey(41), 8)
        base = seed.init_genome(cls.c, cls.keys[0])
        cls.genome = dict(base, conn=jnp.full_like(base["conn"], 2.),
                          hebb=jnp.zeros_like(base["hebb"]).at[..., 3].set(.2),
                          mod=jnp.array([0., 0., 1.]), log_eta=jnp.array(-2.))

    def collective(self, c=None, **kwargs):
        return lineage.collective_lives(self.genome, self.keys[:3], self.keys[3:6],
                                        [0, 1, 0], self.c if c is None else c, **kwargs)

    def test_newborns_share_dead_donors_and_accumulate_actual_changes(self):
        c = self.c._replace(decay=1.)
        result = self.collective(c)
        self.assertEqual(result["completed_lives"], 3)
        previous = np.zeros_like(result["final_archive"])
        for index, row in enumerate(result["lives"]):
            np.testing.assert_array_equal(row["agent_ids"], [[index, 0], [index, 1]])
            np.testing.assert_array_equal(row["archive_before"], previous)
            for arm in ("inherited", "fresh"):
                np.testing.assert_array_equal(row[arm]["events"]["alive_before"][0], [1, 1])
                np.testing.assert_array_equal(row[arm]["weights"]["final_alive"], [0, 0])
            self.assertTrue(np.any(row["donor_delta"] != 0))
            expected = np.clip(previous + row["donor_delta"].sum(0)
                               / np.maximum(row["donor_counts"], 1), -c.wmax, c.wmax)
            np.testing.assert_array_equal(row["archive_after"], expected)
            previous = expected
        self.assertNotEqual(result["lives"][1]["inherited"]["weights"]["initial_W_sha256"],
                            result["lives"][1]["fresh"]["weights"]["initial_W_sha256"])
        np.testing.assert_array_equal(result["final_archive"], previous)

    def test_frozen_genome_is_null_transfer_and_reproducible(self):
        result = self.collective(plastic=False)
        replay = self.collective(plastic=False)
        np.testing.assert_array_equal(result["final_archive"], 0.)
        for row, again in zip(result["lives"], replay["lives"]):
            np.testing.assert_array_equal(row["donor_delta"], 0.)
            self.assertEqual(row["inherited"]["weights"]["initial_W_sha256"],
                             row["fresh"]["weights"]["initial_W_sha256"])
            for field in ("food", "poison", "alive_before", "alive_after"):
                np.testing.assert_array_equal(row["inherited"]["events"][field],
                                              row["fresh"]["events"][field])
                np.testing.assert_array_equal(row["inherited"]["events"][field],
                                              again["inherited"]["events"][field])

    def test_same_individual_transfers_full_weights_and_keeps_agent_identity(self):
        result = lineage.same_individual(self.genome, self.keys[:2], self.keys[3],
                                         [0, 1], self.c)
        first, second = result["lives"]
        self.assertEqual(first["first"]["weights"]["terminal_W_sha256"],
                         second["inherited"]["weights"]["initial_W_sha256"])
        self.assertEqual(second["inherited"]["weights"]["birth_W_sha256"],
                         second["fresh"]["weights"]["initial_W_sha256"])
        np.testing.assert_array_equal(first["agent_ids"], second["agent_ids"])
        np.testing.assert_array_equal(second["eligibility"], [1, 1])

    def test_same_individual_keeps_dead_agents_in_both_outcomes(self):
        result = lineage.same_individual(self.genome, self.keys[:2], self.keys[3],
                                         [0, 1], self.c._replace(decay=1.))
        second = result["lives"][1]
        np.testing.assert_array_equal(second["eligibility"], [0, 0])
        for arm in ("inherited", "fresh"):
            np.testing.assert_array_equal(second[arm]["events"]["alive_before"], 0.)
            self.assertEqual(float(second[arm]["fitness"]), 0.)
            self.assertEqual(float(second[arm]["survived"]), 0.)

    def test_masked_mean_counts_zero_donors_excludes_absent_edges_and_does_not_recount(self):
        _, D, O, n, _ = seed.dims(self.c)
        birth = np.zeros((self.c.M, n, n), np.float32)
        birth[:, n - O, :D] = -0.
        mask = birth.copy()
        mask[:, n - O, 0] = 1.
        mask[1, n - O, 1] = 1.
        calls = []

        def life(g, wk, dk, rule, c, ecology, plastic, *, weights=None, alive=None):
            initial = weights.copy()
            terminal = initial.copy()
            if len(calls) == 0:
                terminal[0, n - O, 0] += .5
                terminal[1, n - O, 1] += .75
            if len(calls) == 4:
                terminal[:, n - O, 0] += 5.
            calls.append(initial)
            return {}, dict(initial_W=initial, terminal_W=terminal,
                            mask=mask, alive=np.array([0., 1.]))

        with patch.object(lineage, "_birth", return_value=(birth, mask)), \
                patch.object(lineage, "_life", side_effect=life):
            result = self.collective()
        first, second, third = result["lives"]
        self.assertEqual(calls[0].tobytes(), birth.tobytes())
        self.assertEqual(float(first["archive_after"][0, 0]), .25)
        self.assertEqual(float(first["archive_after"][0, 1]), .75)
        np.testing.assert_array_equal(second["archive_after"], first["archive_after"])
        self.assertEqual(float(calls[2][0, n - O, 0]), .25)
        self.assertEqual(float(calls[2][0, n - O, 1]), 0.)
        self.assertEqual(float(calls[2][1, n - O, 1]), .75)
        np.testing.assert_array_equal(calls[3], birth)
        self.assertEqual(float(third["archive_after"][0, 0]), self.c.wmax)
        self.assertEqual(first["archive_after"].shape, (O, D))
        self.assertTrue(all(np.count_nonzero(call[:, :n - O]) == 0 for call in calls))

    def test_callback_streams_complete_pairs_and_failure_stops_later_cohorts(self):
        captured = []
        result = self.collective(plastic=False, on_life=captured.append)
        self.assertEqual(len(captured), 3)
        self.assertEqual(result["lives"], [])
        self.assertTrue(all("inherited" in row and "fresh" in row for row in captured))
        seen = []

        def stop(row):
            seen.append(row["life"])
            raise RuntimeError("checkpoint writer failed")

        with self.assertRaisesRegex(RuntimeError, "checkpoint writer failed"):
            self.collective(plastic=False, on_life=stop)
        self.assertEqual(seen, [0])

    def test_boundaries_reject_mixed_layouts_bad_genomes_and_reused_keys_before_running(self):
        invalid_genome = dict(self.genome, mod=np.array([np.nan, 0., 0.]))
        cases = [
            dict(g=invalid_genome),
            dict(g=dict(self.genome, extra=np.array(0.))),
            dict(wkeys=[self.keys[0]] * 3),
            dict(dkeys=[self.keys[3]] * 3),
            dict(wkeys=[np.array([1, 2], np.int32)] * 3),
            dict(rules=[0, 2, 0]),
            dict(rules=[0, True, 0]),
            dict(ecologies=[None, EcologyV1(), EcologyV1()]),
            dict(ecologies=[None]),
            dict(on_life="writer"),
        ]
        for overrides in cases:
            kwargs = dict(g=self.genome, wkeys=self.keys[:3], dkeys=self.keys[3:6],
                          rules=[0, 1, 0], c=self.c)
            kwargs.update(overrides)
            with self.subTest(overrides=list(overrides)), \
                    patch.object(seed, "rollout") as rollout, self.assertRaises(ValueError):
                lineage.collective_lives(**kwargs)
            rollout.assert_not_called()

    def test_ecology_curriculum_keeps_layout_and_paired_world_geometry(self):
        pressure = [EcologyV1(), EcologyV1(.01, 0.), EcologyV1(.01, .02)]
        result = self.collective(ecologies=pressure, plastic=False)
        expected_shape = (seed.dims(self.c, ecology=pressure[0])[2],
                          seed.dims(self.c, ecology=pressure[0])[1])
        self.assertEqual(result["final_archive"].shape, expected_shape)
        for row, ecology in zip(result["lives"], pressure):
            self.assertEqual(row["ecology"]["movement_energy_cost"], ecology.movement_energy_cost)
            np.testing.assert_array_equal(row["inherited"]["ecology"]["shelter"],
                                          row["fresh"]["ecology"]["shelter"])
            np.testing.assert_array_equal(row["inherited"]["events"]["alive_before"],
                                          row["fresh"]["events"]["alive_before"])

    def test_portable_checkpoint_continuation_matches_continuous_lives(self):
        continuous = self.collective()
        first = lineage.collective_lives(self.genome, self.keys[:1], self.keys[3:4],
                                          [0], self.c)
        checkpoint = first["archive_state"]
        array_names = ("archive", "used_world_keys", "used_development_keys")
        metadata = {name: value for name, value in checkpoint.items() if name not in array_names}
        buffer = io.BytesIO()
        np.savez(buffer, metadata=np.array(json.dumps(metadata)),
                 **{name: checkpoint[name] for name in array_names})
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            restored = json.loads(str(saved["metadata"]))
            restored.update({name: saved[name].copy() for name in array_names})
        resumed = lineage.collective_lives(self.genome, self.keys[1:3], self.keys[4:6],
                                            [1, 0], self.c, archive_state=restored)
        np.testing.assert_array_equal(resumed["final_archive"], continuous["final_archive"])
        self.assertEqual(resumed["archive_state"]["next_life"], 3)
        self.assertEqual(restored["next_life"], 1)
        np.testing.assert_array_equal(restored["archive"], checkpoint["archive"])
        for actual, expected in zip(resumed["lives"], continuous["lives"][1:]):
            self.assertEqual(actual["life"], expected["life"])
            for field in ("agent_ids", "archive_before", "archive_after", "donor_delta"):
                np.testing.assert_array_equal(actual[field], expected[field])
            for arm in ("inherited", "fresh"):
                self.assertEqual(actual[arm]["weights"]["initial_W_sha256"],
                                 expected[arm]["weights"]["initial_W_sha256"])
                for field in ("food", "poison", "alive_before", "alive_after"):
                    np.testing.assert_array_equal(actual[arm]["events"][field],
                                                  expected[arm]["events"][field])

    def test_checkpoint_rejects_incompatible_or_reused_state_before_running(self):
        complete = self.collective(plastic=False)
        checkpoint = complete["lives"][0]["archive_state"]
        invalid = []
        for name, value in (("state_version", 2), ("method", "other"),
                            ("genome_sha256", "other"), ("next_life", 0),
                            ("archive", np.full_like(checkpoint["archive"], np.nan)),
                            ("archive", np.full_like(checkpoint["archive"], self.c.wmax + 1.)),
                            ("used_world_keys", np.zeros((2, 2), np.uint32))):
            invalid.append(dict(checkpoint, **{name: value}))
        wrong_layout = copy.deepcopy(checkpoint)
        wrong_layout["layout"]["ecology_version"] = "ecology-v1"
        invalid.append(wrong_layout)
        for saved in invalid:
            with self.subTest(state_version=saved["state_version"]), \
                    patch.object(seed, "rollout") as rollout, self.assertRaises(ValueError):
                lineage.collective_lives(self.genome, self.keys[1:3], self.keys[4:6],
                                          [1, 0], self.c, archive_state=saved)
            rollout.assert_not_called()
        for worlds, development in ((self.keys[:2], self.keys[4:6]),
                                    (self.keys[1:3], self.keys[3:5])):
            with patch.object(seed, "rollout") as rollout, self.assertRaises(ValueError):
                lineage.collective_lives(self.genome, worlds, development, [1, 0], self.c,
                                          archive_state=checkpoint)
            rollout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
