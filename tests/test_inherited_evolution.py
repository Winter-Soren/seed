"""Synthetic archive accounting and small development checks for E119."""
import copy
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import cold_donor, inherited_evolution as study
from seed import seed


def family(index=0, child_turn=1, child_survival=True, inside=(False,) * 4):
    steps, agents = 4, 4
    initial = np.asarray(inside, bool)
    position = np.where(initial[:, None], [0, 0], [1, 1]).astype(np.int32)
    shelter_map = np.zeros((4, 4), bool)
    shelter_map[0, 0] = True
    row = dict(keys=np.arange(index * 4 + 1, index * 4 + 5, dtype=np.uint32).reshape(2, 2),
               rule=np.asarray(index % 2, np.int32), steps=np.asarray(steps), agents=np.asarray(agents),
               weight_bound=np.asarray(3., np.float32), initial_position=position)
    for arm in study.ARMS:
        group = 2 if arm.startswith("child") else int(arm.startswith("parent"))
        archive = np.full((2, 3), .2 if arm.endswith("latest") else .1, np.float32)
        if arm.endswith("empty"):
            archive.fill(0)
        after = np.ones((steps, agents), bool)
        if arm == "child_latest":
            after[-1] = child_survival
        before = np.concatenate((np.ones((1, agents), bool), after[:-1]))
        occupied = np.zeros_like(after)
        turn = child_turn if arm == "child_latest" else steps
        if turn <= steps:
            occupied[turn - 1] = True
        prefix = arm + "_"
        fresh = np.zeros((agents, 2, 3), np.float32)
        initial_weights = fresh + archive
        values = dict(food=np.zeros_like(after), poison=np.zeros_like(after), alive_before=before,
                      alive_after=after, sheltered=occupied, initial_sheltered=initial.copy(),
                      shelter=shelter_map.copy(), fitness=np.asarray(0.),
                      first_entry=cold_donor.first_entry(initial, occupied, after),
                      energy_lost=np.zeros((steps, agents), np.float32),
                      birth_connections=np.full(agents, 6, np.int32),
                      birth_sha256=np.full((agents, 32), group, np.uint8),
                      fresh_sha256=np.full((agents, 32), group, np.uint8),
                      initial_sha256=np.full((agents, 32), group + 3, np.uint8),
                      terminal_sha256=np.full((agents, 32), group + 4, np.uint8),
                      fresh_sensor_weights=fresh, initial_sensor_weights=initial_weights,
                      terminal_sensor_weights=initial_weights.copy(), sensor_mask=np.ones_like(fresh, bool),
                      genome=np.full(155, group, np.float32), archive=archive)
        values.update({"persistence_" + name: np.zeros_like(after)
                       for name in ("active", "remembered", "tactic_attempted")})
        row.update({prefix + key: value for key, value in values.items()})
    return row


class InheritanceArithmeticTests(unittest.TestCase):
    def test_archive_includes_zero_deltas_and_uses_counts_per_edge(self):
        old = np.array([[.5, -.5]], np.float32)
        delta = np.array([[[[2., 0.]], [[0., -2.]]], [[[0., 0.]], [[0., 0.]]]], np.float32)
        mask = np.array([[[[1, 0]], [[1, 1]]], [[[0, 0]], [[0, 1]]]], bool)
        after, counts = study.update_archive(old, delta, mask, 3.)
        np.testing.assert_array_equal(counts, [[2, 2]])
        np.testing.assert_array_equal(after, [[1.5, -1.5]])
        # Later inherited initialization is subtracted; it is not a second acquisition.
        unchanged, _ = study.update_archive(after, np.zeros_like(delta), mask, 3.)
        np.testing.assert_array_equal(unchanged, after)

    def test_overlay_respects_mask_clipping_and_empty_signed_zeros(self):
        fresh = np.array([[[2.9, -2.9, -0.0]]], np.float32)
        mask = np.array([[[True, True, False]]])
        archive = np.array([[1., -1., 2.]], np.float32)
        actual = study.overlay_archive(fresh, mask, archive, 3.)
        np.testing.assert_array_equal(actual, [[[3., -3., -0.0]]])
        self.assertEqual(study.overlay_archive(fresh, mask, np.zeros_like(archive), 3.).tobytes(), fresh.tobytes())

    def test_all_six_arms_and_immediate_parent_gate(self):
        rows = [family(0), family(1, child_turn=3, inside=(True, True, True, False))]
        measured = study.summarize(rows)
        primary = measured["contrasts"]["child_vs_parent"]
        self.assertEqual(measured["outside_born_agents"], 5)
        self.assertEqual(measured["assigned_agents"], 8)
        self.assertAlmostEqual(primary["latency_improvement"]["mean"], 2.6)
        self.assertAlmostEqual(primary["latency_improvement"]["se"], .64)
        self.assertTrue(measured["gate"]["passed"])
        self.assertEqual(measured["gate"], primary["gate"])
        self.assertEqual(set(measured["contrasts"]), set(study.CONTRASTS))
        failed = study.summarize([family(i, child_survival=False) for i in range(2)])
        self.assertFalse(failed["gate"]["survival_passed"])
        self.assertFalse(failed["gate"]["passed"])

    def test_corrupt_memory_controls_and_duplicate_worlds_are_rejected(self):
        for field in ("child_latest_initial_sensor_weights", "child_latest_fresh_sensor_weights",
                      "child_previous_genome", "parent_latest_archive"):
            row = family()
            row[field] = row[field].copy()
            row[field].flat[0] += .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.validate_world(row)
        with self.assertRaises(ValueError):
            study.summarize([family(), family()])
        self.assertIsNone(study.summarize([family()])["gate"]["passed"])

    def test_invalid_inputs_fail_before_lifetime(self):
        c = seed.Cfg()
        valid = np.zeros(seed.genome_size(c), np.float32)
        for bad in (np.full((6, 129), np.nan, np.float32), np.ones((1, 1), np.float32),
                    np.full((6, 129), 4., np.float32)):
            with self.assertRaises(ValueError):
                study.donor_update(valid, bad, 45, 0, c)
        for value in (True, -1, .5, 2 ** 32):
            with self.assertRaises(ValueError):
                study.donor_keys(value, 0)
        for count in (0, 1, 3, True):
            with self.assertRaises(ValueError):
                study.make_evaluator(worlds=count)


class InheritanceDevelopmentTests(unittest.TestCase):
    def test_dynamic_archive_reaches_candidates_and_donors_preserve_accounting(self):
        c = seed.Cfg(G=6, M=2, T=4, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        flat = np.asarray(flat)
        archive = study.empty_archive(c)
        evaluate, _ = study.make_evaluator(c, worlds=2)
        keys = study.donor_keys(45, 0, worlds=2)
        population = jnp.stack((flat, flat))
        dkeys = jnp.stack((keys[:, 1], keys[:, 1]))
        baseline = jax.tree.map(np.asarray, evaluate(population, keys[:, 0], dkeys, archive))
        archive.fill(.25)
        inherited = jax.tree.map(np.asarray, evaluate(population, keys[:, 0], dkeys, archive))
        np.testing.assert_array_equal(baseline["candidate_birth"], inherited["candidate_birth"])
        np.testing.assert_array_equal(inherited["candidate_initial"], study.overlay_archive(
            inherited["candidate_birth"], inherited["candidate_mask"], archive, c.wmax))
        self.assertTrue(np.any(inherited["candidate_initial"] != baseline["candidate_initial"]))
        donor = study.donor_update(flat, archive, 45, 1, c, worlds=2)
        self.assertEqual(donor["donor_delta"].shape[:2], (2, 2))
        np.testing.assert_array_equal(donor["donor_delta"],
                                      donor["donor_terminal_sensor_weights"] - donor["donor_initial_sensor_weights"])
        corrupt = copy.deepcopy(donor)
        corrupt["donor_counts"].flat[0] += 1
        with self.assertRaises(ValueError):
            study.validate_donors(corrupt, c)

    def test_identical_genomes_and_empty_archives_share_actual_lives(self):
        c = seed.Cfg(G=6, M=2, T=4, R=1, H=3, n_items=2)
        flat = np.asarray(ravel_pytree(seed.biology_prior(c))[0])
        arms = {arm: (flat, study.empty_archive(c)) for arm in study.ARMS}
        row = study.evaluate_world(arms, 46, 0, c)
        study.validate_world(row, c)
        for arm in study.ARMS:
            for name in study.shelter.EVENTS + ("first_entry", "initial_sensor_weights", "initial_sha256"):
                np.testing.assert_array_equal(row["founder_empty_" + name], row[arm + "_" + name])


if __name__ == "__main__":
    unittest.main()
