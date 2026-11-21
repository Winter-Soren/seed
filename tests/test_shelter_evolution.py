"""Synthetic outcomes and tiny development checks for E118 shelter evolution."""
import unittest

import jax
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import cold_donor, shelter_evolution as study
from seed import seed


def family(index=0, parent=(4, 4, 4, 4), descendant=(1, 1, 1, 1),
           inside=(False,) * 4, descendant_survival=(True,) * 4):
    steps, agents = 4, 4
    initial = np.asarray(inside, bool)
    position = np.where(initial[:, None], [0, 0], [1, 1]).astype(np.int32)
    shelter = np.zeros((4, 4), bool)
    shelter[0, 0] = True
    row = dict(keys=np.arange(4 * index + 1, 4 * index + 5, dtype=np.uint32).reshape(2, 2),
               rule=np.asarray(index % 2, np.int32), steps=np.asarray(steps, np.int32),
               agents=np.asarray(agents, np.int32), initial_position=position)
    for arm, arrivals in (("parent", parent), ("descendant", descendant)):
        after = np.ones((steps, agents), bool)
        if arm == "descendant":
            after[-1] = descendant_survival
        before = np.concatenate((np.ones((1, agents), bool), after[:-1]))
        sheltered = np.zeros((steps, agents), bool)
        for agent, turn in enumerate(arrivals):
            if 0 < turn <= steps:
                sheltered[turn - 1, agent] = True
        prefix = arm + "_"
        row.update({prefix + "food": np.zeros_like(after), prefix + "poison": np.zeros_like(after),
                    prefix + "alive_before": before, prefix + "alive_after": after,
                    prefix + "sheltered": sheltered, prefix + "initial_sheltered": initial.copy(),
                    prefix + "shelter": shelter.copy(), prefix + "fitness": np.asarray(0.),
                    prefix + "first_entry": cold_donor.first_entry(initial, sheltered, after),
                    prefix + "energy_lost": np.zeros((steps, agents), np.float32),
                    prefix + "birth_connections": np.full(agents, 8 if arm == "parent" else 7, np.int32),
                    prefix + "birth_sha256": np.full((agents, 32), int(arm == "descendant"), np.uint8)})
        for name in ("active", "remembered", "tactic_attempted"):
            row[prefix + "persistence_" + name] = np.zeros_like(after)
    return row


class ShelterScoreTests(unittest.TestCase):
    def test_arrival_reward_excludes_inside_and_fatal_arrival(self):
        initial = np.array([True, False, False, False])
        sheltered = np.array([[False, True, True, False], [False, False, False, True]])
        alive = np.array([[True, True, False, True], [True, False, False, True]])
        # The second agent reaches alive then dies and retains its arrival reward.
        self.assertEqual(float(study.shelter_score(initial, sheltered, alive)), 1.)
        self.assertEqual(float(study.shelter_score(np.ones(4, bool), sheltered, alive)), 0.)

    def test_weighted_latency_uses_world_cluster_uncertainty(self):
        rows = [family(), family(1, descendant=(1, 1, 1, 3), inside=(True, True, True, False))]
        result = study.summarize(rows)
        self.assertEqual(result["outside_born_agents"], 5)
        self.assertEqual(result["assigned_agents"], 8)
        self.assertAlmostEqual(result["parent"]["latency"]["mean"], 4.)
        self.assertAlmostEqual(result["descendant"]["latency"]["mean"], 1.4)
        self.assertAlmostEqual(result["latency_improvement"]["mean"], 2.6)
        self.assertAlmostEqual(result["latency_improvement"]["se"], .64)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["descendant"]["before_cold_arrivals"], 4)

    def test_survival_veto_and_unevaluable_samples(self):
        rows = [family(i, descendant_survival=(False,) * 4) for i in range(2)]
        result = study.summarize(rows)
        self.assertTrue(result["gate"]["latency_passed"])
        self.assertFalse(result["gate"]["survival_passed"])
        self.assertFalse(result["gate"]["passed"])
        self.assertIsNone(study.summarize([family()])["gate"]["passed"])
        inside = study.summarize([family(i, inside=(True,) * 4) for i in range(2)])
        self.assertIsNone(inside["latency_improvement"]["mean"])
        self.assertIsNone(inside["gate"]["passed"])

    def test_corrupt_evidence_and_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            study.summarize([family(), family()])
        for field, value in (("descendant_first_entry", np.zeros(4, np.int32)),
                             ("descendant_initial_sheltered", np.ones(4, bool)),
                             ("descendant_fitness", np.asarray(1.))):
            row = family()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.validate_pair(row)
        row = family()
        row["parent_alive_after"][0, 0] = False
        with self.assertRaises(ValueError):
            study.validate_pair(row)

    def test_valid_genotypes_can_develop_different_wiring(self):
        row = family()
        study.validate_pair(row)
        self.assertFalse(np.array_equal(row["parent_birth_sha256"], row["descendant_birth_sha256"]))


class ShelterDevelopmentTests(unittest.TestCase):
    def test_invalid_seeds_and_unbalanced_training_worlds_are_rejected(self):
        for value in (True, -1, .5, 2 ** 32):
            with self.subTest(value=value), self.assertRaises(ValueError):
                study.evaluation_keys(value, 0)
        for worlds in (True, 0, 1, 3, 2.5):
            with self.subTest(worlds=worlds), self.assertRaises(ValueError):
                study.make_evaluator(worlds=worlds)

    def test_existing_es_changes_genome_and_equal_parents_are_paired(self):
        c = seed.Cfg(G=4, M=2, T=4, R=0, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        evaluate, _ = study.make_evaluator(c, worlds=2)
        state = seed.es_init(c, mean=flat, seed=43)
        changed, metrics = seed.es_step(state, evaluate, pop=4, E=2, sigma=.05, lr=.02)
        self.assertEqual(changed["t"], 1)
        self.assertTrue(np.isfinite(changed["mu"]).all())
        self.assertFalse(np.array_equal(changed["mu"], flat))
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        row = study.evaluate_pair(flat, flat, 43, 0, c)
        study.validate_pair(row, c)
        for name in study.EVENTS + ("birth_sha256", "birth_connections", "first_entry"):
            np.testing.assert_array_equal(row["parent_" + name], row["descendant_" + name])
        invalid = np.asarray(flat).copy()
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            study.evaluate_pair(invalid, flat, 43, 0, c)


if __name__ == "__main__":
    unittest.main()
