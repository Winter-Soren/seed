"""E124 synthetic checks and a small development-only chain fixture."""
import copy
import unittest

import jax
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import social_belief as belief, social_learning as social, social_transmission as transmission
from seed import seed


def projected_chain(index, *, improving=True):
    """Synthetic scalar outcomes only; these are not simulated evidence."""
    row = dict(chain_index=np.asarray(index, np.int32), family_index=np.asarray(index, np.int32),
               rule=np.asarray(index % 2, np.int32), yoked_swap=np.asarray(bool((index // 2) % 2)),
               keys=np.arange(index * 16, index * 16 + 16, dtype=np.uint32).reshape(4, 2, 2),
               lambda_value=np.asarray(2., np.float64), steps=np.asarray(6, np.int32),
               genome_sha256=np.zeros(32, np.uint8))

    def add(prefix, first):
        row.update({prefix + key: value for key, value in dict(
            first_food=np.asarray(first, bool), first_poison=np.asarray(not first, bool),
            no_meal=np.asarray(False), first_meal_step=np.asarray(1, np.int32),
            survived=np.asarray(True), fitness=np.asarray(1. if first else -1., np.float32),
            weight_change_count=np.asarray(0, np.int32), q=np.asarray(.5, np.float32),
            track_votes=np.asarray([1, 0], np.int32)).items()})

    add("founder_", index < 40)
    for cohort in range(1, 4):
        successful = index < ((40, 48, 64)[cohort - 1] if improving else 64)
        for arm in transmission.ARMS:
            first = successful if arm == "predecessor" else (index < 40 if improving else True) if arm == "repeated_founder" else False
            add(f"c{cohort}_{arm}_", first)
    return row


class TransmissionMathTests(unittest.TestCase):
    def test_persistence_and_cumulative_improvement_have_distinct_gates(self):
        rows = [projected_chain(index, improving=False) for index in range(64)]
        result = transmission.summarize(rows)
        self.assertTrue(result["retained_efficacy"]["passed"])
        self.assertFalse(result["cumulative_improvement"]["passed"])
        self.assertEqual(result["cross_cohort"]["c3_vs_c1"]["first_food_difference"]["mean"], 0.)
        improved = transmission.summarize([projected_chain(index) for index in range(64)])
        self.assertTrue(improved["retained_efficacy"]["passed"])
        self.assertTrue(improved["cumulative_improvement"]["passed"])

    def test_survival_uses_strict_negative_tenth_screen(self):
        values = np.full(16, -.10)
        contrast = transmission._contrast(np.ones(16), values)
        self.assertFalse(contrast["gate"]["survival_passed"])
        self.assertTrue(transmission._contrast(np.ones(16), np.full(16, -.09))["gate"]["survival_passed"])
        self.assertEqual(contrast["survival_difference"]["lower_2se"], -.10)

    def test_failed_cross_cohort_survival_prevents_both_gates(self):
        rows = [projected_chain(index) for index in range(64)]
        for row in rows:
            for arm in transmission.ARMS:
                row[f"c3_{arm}_survived"] = np.asarray(False)
        result = transmission.summarize(rows)
        self.assertTrue(result["retained_efficacy"]["first_food_passed"])
        self.assertFalse(result["retained_efficacy"]["survival_passed"])
        self.assertFalse(result["retained_efficacy"]["passed"])
        self.assertFalse(result["cumulative_improvement"]["passed"])

    def test_one_bad_cohort_cannot_be_hidden_by_later_success(self):
        rows = [projected_chain(index) for index in range(64)]
        for row in rows:
            row["c2_yoked_first_food"] = row["c2_predecessor_first_food"].copy()
            row["c2_yoked_first_poison"] = row["c2_predecessor_first_poison"].copy()
        result = transmission.summarize(rows)
        self.assertFalse(result["retained_efficacy"]["passed"])
        self.assertFalse(result["cumulative_improvement"]["passed"])

    def test_chain_is_the_unit_for_paired_se_and_undefined_single_chain(self):
        rows = [projected_chain(index) for index in range(64)]
        result = transmission.summarize(rows)
        delta = np.asarray([float(row["c3_predecessor_first_food"]) - float(row["c1_predecessor_first_food"])
                            for row in rows])
        self.assertAlmostEqual(result["cross_cohort"]["c3_vs_c1"]["first_food_difference"]["se"],
                               delta.std(ddof=1) / np.sqrt(64))
        self.assertIsNone(transmission.summarize(rows[:1])["retained_efficacy"]["passed"])
        self.assertIsNone(transmission.summarize(rows[:1])["cumulative_improvement"]["passed"])

    def test_projection_validation_keeps_missing_outcomes_and_rejects_reused_keys(self):
        rows = [projected_chain(index) for index in range(4)]
        changed = copy.deepcopy(rows)
        changed[1]["keys"] = changed[0]["keys"].copy()
        with self.assertRaises(ValueError):
            transmission.summarize(changed)
        changed = copy.deepcopy(rows)
        changed[1]["lambda_value"] = np.asarray(4.)
        with self.assertRaises(ValueError):
            transmission.summarize(changed)
        changed = copy.deepcopy(rows)
        for name, value in (("first_food", False), ("first_poison", False), ("no_meal", True), ("first_meal_step", 7)):
            changed[0]["c3_predecessor_" + name] = np.asarray(value)
        result = transmission.summarize(changed)
        self.assertEqual(result["cohorts"]["3"]["arms"]["predecessor"]["no_meal_count"], 1)
        self.assertEqual(result["chains"], 4)

    def test_public_movie_preserves_actual_signal_and_dead_late_footage(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=1)
        life = dict(solo_initial_grid=np.zeros((4, 4), np.uint8),
                    solo_initial_position=np.asarray([[1, 1]], np.int32),
                    solo_grid=np.zeros((6, 4, 4), np.uint8),
                    solo_position=np.full((6, 1, 2), 1, np.int32),
                    solo_signal=np.full((6, 1), .75, np.float32), alive_before=np.ones((6, 1), bool))
        movie, _ = transmission.life_movie(life, c)
        self.assertEqual(float(movie[0, 27:36].max()), .75)
        life["alive_before"][4:] = False
        dead, _ = transmission.life_movie(life, c)
        np.testing.assert_array_equal(dead[1:, 18:36], 0)
        self.assertEqual(dead.shape[0], 3)


class TransmissionEngineTests(unittest.TestCase):
    """Run only when the coordinator releases simulator checks; uses no final seeds."""

    @classmethod
    def setUpClass(cls):
        cls.c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(cls.c))
        cls.flat = np.asarray(flat)
        cls.row = transmission.evaluate_chain(cls.flat, transmission.chain_keys(52, 2), 2, 2., cls.c)

    def test_three_fresh_cohorts_and_actual_previous_solo_source(self):
        row, c = self.row, self.c
        self.assertEqual(row["keys"].shape, (4, 2, 2))
        predecessor = transmission._life(row, "founder_")
        for cohort in range(1, 4):
            movie, _ = transmission.life_movie(predecessor, c)
            np.testing.assert_array_equal(row[f"c{cohort}_predecessor_frames"], movie)
            np.testing.assert_array_equal(row[f"c{cohort}_repeated_founder_frames"], row["founder_movie"])
            predecessor = transmission._life(row, f"c{cohort}_predecessor_")
        for suffix, value in transmission._life(row, "c1_repeated_founder_").items():
            np.testing.assert_array_equal(row["c1_predecessor_" + suffix], value)
        self.assertEqual(len({key.tobytes() for key in row["keys"].reshape(8, 2)}), 8)

    def test_neutral_birth_and_reset_preserves_only_own_acquired_brain(self):
        self.assertEqual(self.row["founder_frames"].shape[0], 0)
        for cohort in range(1, 4):
            for arm in transmission.ARMS:
                self.assertEqual(float(self.row[f"c{cohort}_{arm}_q_birth"]), .5)
            self.assertEqual(float(self.row[f"c{cohort}_belief_reset_q"]), .5)
            for name in ("W", "x", "Wp"):
                np.testing.assert_array_equal(self.row[f"c{cohort}_belief_reset_solo_initial_{name}_sha256"],
                                              self.row[f"c{cohort}_predecessor_solo_initial_{name}_sha256"])
        compact = transmission.summary_input(self.row)
        self.assertNotIn("c1_predecessor_solo_grid", compact)
        self.assertLess(sum(np.asarray(value).nbytes for value in compact.values()), 4096)
        self.assertEqual(transmission.summarize([compact])["assigned_agents"], 13)

    def test_validator_rejects_handoff_and_state_corruption(self):
        for field in ("c2_predecessor_movie", "c1_repeated_founder_solo_signal",
                      "c3_belief_reset_solo_initial_W_sha256", "c2_predecessor_q_birth",
                      "c1_predecessor_frames"):
            changed = copy.deepcopy(self.row)
            changed[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                transmission.validate_chain(changed, self.c)


if __name__ == "__main__":
    unittest.main()
