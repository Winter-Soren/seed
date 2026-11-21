"""Development-only event fixtures; held-out E117 worlds are never used here."""
import copy
import unittest
from unittest.mock import patch

import jax
import numpy as np

from scripts import cold_donor
from seed import lineage, seed
from seed.ecology import EcologyV1


def family(index=0, inherited=(1, 1, 1, 1), fresh=(4, 4, 4, 4), inside=(False,) * 4,
           inherited_survival=(True,) * 4, fresh_survival=(True,) * 4):
    """Four-agent/four-step hand-authored outcomes, without invoking the engine."""
    m, t = 4, 4
    shape = (6, 9)
    row = dict(keys=np.arange(8 * index + 1, 8 * index + 9, dtype=np.uint32).reshape(4, 2),
               rule=np.asarray(index % 2, np.int32), steps=np.asarray(t, np.int32),
               agents=np.asarray(m, np.int32), cold_start=np.asarray(2, np.int32),
               weight_bound=np.asarray(3.), archive_before=np.zeros(shape, np.float32),
               archive_after=np.zeros(shape, np.float32),
               donor_delta=np.zeros((m,) + shape, np.float32),
               donor_mask=np.ones((m,) + shape, bool), donor_counts=np.full(shape, m, np.float32),
               recipient_mask=np.ones((m,) + shape, bool),
               recipient_birth=np.zeros((m,) + shape, np.float32),
               recipient_initial_delta=np.zeros((m,) + shape, np.float32))
    for arm, arrivals, initially, survival in (
            ("donor", (5,) * m, (False,) * m, (True,) * m),
            ("inherited", inherited, inside, inherited_survival),
            ("fresh", fresh, inside, fresh_survival)):
        before, after = np.ones((t, m), bool), np.ones((t, m), bool)
        after[1:, ~np.asarray(survival)] = False
        before[1:] = after[:-1]
        occupied = np.zeros((t, m), bool)
        for agent, arrival in enumerate(arrivals):
            if 1 <= arrival <= t:
                occupied[arrival - 1, agent] = True
        shelter = np.zeros((4, 4), bool)
        shelter[0, 0] = True
        position = np.where(np.asarray(initially)[:, None], np.zeros((m, 2), np.int32),
                            np.ones((m, 2), np.int32))
        prefix = arm + "_"
        row.update({prefix + "food": np.zeros((t, m), bool),
                    prefix + "poison": np.zeros((t, m), bool),
                    prefix + "alive_before": before, prefix + "alive_after": after,
                    prefix + "sheltered": occupied, prefix + "initial_sheltered": np.asarray(initially, bool),
                    prefix + "energy_lost": np.zeros((t, m), np.float32),
                    prefix + "shelter": shelter, prefix + "initial_position": position,
                    prefix + "fitness": np.asarray(0., np.float32),
                    prefix + "first_entry": cold_donor.first_entry(initially, occupied, after),
                    prefix + "changed_connections": np.zeros(m, np.int32),
                    prefix + "mean_abs_weight_change": np.zeros(m, np.float32)})
        for name in ("birth", "initial", "terminal"):
            row[prefix + name + "_sha256"] = np.zeros((m, 32), np.uint8)
    return row


class ColdDonorMetricsTests(unittest.TestCase):
    def test_arrival_requires_living_transition_and_never_drops_deaths(self):
        occupied = np.zeros((4, 4), bool)
        occupied[0, 1] = True
        occupied[1, 2] = True
        alive = np.ones((4, 4), bool)
        alive[1:, 2:] = False
        np.testing.assert_array_equal(
            cold_donor.first_entry([True, False, False, False], occupied, alive), [0, 1, 5, 5])

    def test_pooled_outside_ratio_and_cluster_influence_not_world_average(self):
        first = family(0, inherited=(1, 0, 0, 0), fresh=(5, 0, 0, 0),
                       inside=(False, True, True, True))
        second = family(1, inherited=(4,) * 4, fresh=(4,) * 4)
        result = cold_donor.summarize([first, second])
        self.assertEqual(result["assigned_agents"], 8)
        self.assertEqual(result["outside_born_agents"], 5)
        self.assertAlmostEqual(result["latency_improvement"]["mean"], .8)
        self.assertAlmostEqual(result["latency_improvement"]["se"], 1.28)
        self.assertAlmostEqual(result["inherited"]["latency"]["mean"], 3.4)
        self.assertAlmostEqual(result["fresh"]["latency"]["mean"], 4.2)
        self.assertEqual(result["rule_counts"], {"0": 1, "1": 1})

    def test_gate_latency_is_strict_at_exactly_two_standard_errors(self):
        result = cold_donor.summarize([
            family(0, fresh=(2,) * 4), family(1, fresh=(4,) * 4)])
        self.assertEqual(result["latency_improvement"], dict(mean=2., se=1.))
        self.assertFalse(result["gate"]["latency_passed"])
        self.assertFalse(result["gate"]["passed"])

    def test_gate_survival_includes_all_assigned_and_allows_exact_boundary(self):
        first = family(0, fresh=(3,) * 4, inherited_survival=(True, False, False, False))
        second = family(1, fresh=(3,) * 4, inherited_survival=(True, True, True, False))
        result = cold_donor.summarize([first, second])
        self.assertEqual(result["survival_difference"], dict(mean=-.5, se=.25))
        self.assertEqual(result["inherited"]["deaths"], 4)
        self.assertTrue(result["gate"]["passed"])

    def test_no_arrival_keeps_dead_slots_at_t_plus_one(self):
        row = family(inherited=(5,) * 4, fresh=(5,) * 4,
                     inherited_survival=(False,) * 4)
        result = cold_donor.summarize([row])
        self.assertEqual(result["inherited"]["latency"]["mean"], 5.)
        self.assertEqual(result["inherited"]["survival"]["mean"], 0.)
        self.assertEqual(result["inherited"]["living_arrivals"], 0)
        self.assertEqual(result["latency_improvement"], dict(mean=0., se=None))
        self.assertIsNone(result["gate"]["passed"])

    def test_empty_outside_population_is_unevaluable(self):
        rows = [family(i, inherited=(0,) * 4, fresh=(0,) * 4, inside=(True,) * 4) for i in range(2)]
        result = cold_donor.summarize(rows)
        self.assertEqual(result["outside_born_agents"], 0)
        self.assertEqual(result["latency_improvement"], dict(mean=None, se=None))
        self.assertIsNone(result["gate"]["passed"])

    def test_donor_update_includes_zero_changes_and_bounds_recipient_overlay(self):
        row = family()
        row["donor_mask"][2:, 0, 0] = False
        row["donor_counts"][0, 0] = 2
        row["donor_delta"][0, 0, 0] = 1.
        row["archive_after"][0, 0] = .5
        row["recipient_birth"][:, 0, 0] = 2.9
        row["recipient_initial_delta"][:, 0, 0] = np.float32(3.) - np.float32(2.9)
        cold_donor.validate_family(row)
        row["donor_counts"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "masked original donors"):
            cold_donor.validate_family(row)

    def test_corrupt_events_pairing_arrivals_and_keys_are_rejected(self):
        for field, edit in (
                ("inherited_alive_before", lambda x: x.__setitem__((0, 0), False)),
                ("inherited_first_entry", lambda x: x.__setitem__(0, 4)),
                ("inherited_initial_position", lambda x: x.__setitem__((0, 0), 2)),
                ("inherited_birth_sha256", lambda x: x.__setitem__((0, 0), 1)),
                ("donor_energy_lost", lambda x: x.__setitem__((0, 0), np.nan))):
            with self.subTest(field=field):
                row = family()
                edit(row[field])
                with self.assertRaises(ValueError):
                    cold_donor.validate_family(row)
        row = family()
        with self.assertRaisesRegex(ValueError, "reuse random keys"):
            cold_donor.summarize([row, copy.deepcopy(row)])

    def test_cold_exposure_uses_alive_before_while_shelter_occupancy_uses_alive_after(self):
        row = family(inherited=(5,) * 4, fresh=(5,) * 4)
        row["donor_alive_after"][2:, 0] = False
        row["donor_alive_before"][3:, 0] = False
        # Agent zero dies on the first cold transition while outside.
        result = cold_donor.summarize([row])
        self.assertEqual(result["donor"]["cold_exposed_agents"], 4)
        self.assertEqual(result["donor"]["cold_exposed_agent_steps"], 7)
        self.assertEqual(result["donor"]["deaths"], 1)
        self.assertEqual(result["donor"]["cold_sheltered_fraction"]["mean"], 0.)


class ColdDonorEngineTests(unittest.TestCase):
    def test_development_family_matches_public_lineage_and_resets_between_calls(self):
        c = seed.Cfg(G=4, M=2, T=4, R=0, H=3, n_items=0, decay=.1)
        keys = cold_donor.family_keys(cold_donor.DEVELOPMENT_SEED, 0)
        genome = seed.biology_prior(c)
        ecology = EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02)
        row = cold_donor.evaluate_family(genome, keys, 0, c, ecology)
        reference = lineage.collective_lives(genome, keys[[0, 2]], keys[[1, 3]], [0, 0], c,
                                             ecologies=(ecology, ecology), persistence=None)
        for arm in ("inherited", "fresh"):
            raw = reference["lives"][1][arm]
            for event in ("food", "poison", "alive_before", "alive_after"):
                np.testing.assert_array_equal(row[arm + "_" + event], raw["events"][event])
            np.testing.assert_array_equal(row[arm + "_sheltered"], raw["ecology"]["sheltered"])
        np.testing.assert_array_equal(row["archive_after"], reference["lives"][0]["archive_after"])
        # Engine birth_W precedes the lineage's initial-weight override. A
        # paired counterfactual birth fingerprint need not equal the actual
        # eager-developed fresh initial fingerprint, even on one machine.
        alternate = copy.deepcopy(reference)
        alternate_birth = ["0" * 64] * c.M
        for arm in ("inherited", "fresh"):
            alternate["lives"][1][arm]["weights"]["birth_W_sha256"] = alternate_birth
        with patch.object(lineage, "collective_lives", return_value=alternate):
            alternate_row = cold_donor.evaluate_family(genome, keys, 0, c, ecology)
        np.testing.assert_array_equal(alternate_row["inherited_birth_sha256"], 0)
        np.testing.assert_array_equal(alternate_row["fresh_birth_sha256"], 0)
        np.testing.assert_array_equal(alternate_row["fresh_initial_sha256"], row["fresh_initial_sha256"])
        # Actual initial weights must still match the reconstructed overlay.
        alternate["lives"][1]["inherited"]["weights"]["initial_W_sha256"] = alternate_birth
        with patch.object(lineage, "collective_lives", return_value=alternate):
            with self.assertRaisesRegex(ValueError, "exact archive overlay"):
                cold_donor.evaluate_family(genome, keys, 0, c, ecology)
        replay = cold_donor.evaluate_family(genome, keys, 0, c, ecology)
        for name in row:
            np.testing.assert_array_equal(row[name], replay[name], err_msg=name)
        next_keys = cold_donor.family_keys(cold_donor.DEVELOPMENT_SEED, 1)
        self.assertEqual(len({tuple(key) for key in np.concatenate((keys, next_keys))}), 8)
        self.assertEqual(keys.dtype, np.uint32)

    def test_key_arguments_reject_boolean_negative_and_nonintegral_values(self):
        for master, index in ((True, 0), (42, -1), (42, 1.5), (2 ** 32, 0)):
            with self.subTest(master=master, index=index):
                with self.assertRaises(ValueError):
                    cold_donor.family_keys(master, index)


if __name__ == "__main__":
    unittest.main()
