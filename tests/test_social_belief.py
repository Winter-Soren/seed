"""Synthetic E123 checks; no final calibration or evaluation worlds belong here."""
import copy
from pathlib import Path
import subprocess
import types
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from scripts import social_belief as belief, social_learning as social
from seed import seed


def tracked_movie():
    movie = np.zeros((3, 103), np.float32)
    movie[:, -1] = 1
    visual = movie[:, :100].reshape(3, 4, 5, 5)
    visual[0, 2, 1, 2] = 1
    visual[1:, 2, 1, 3] = 1
    visual[0, 0, 1, 3] = 1
    return movie


def fit_row(votes=(3, 1), color=0, outcome=1):
    votes = np.asarray(votes, np.int32)
    return dict(votes=votes, z=float(votes[0] - votes[1]) / max(1, int(votes.sum())),
                contact_color=color, contact_outcome=outcome)


class SocialBeliefTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(value, reference)

    def test_public_decoder_swaps_symmetrically_and_masked_occupancy_abstains(self):
        movie = tracked_movie()
        arms = social.movie_conditions(movie, True)
        decoded = {arm: belief.decode_movie(frames, 2.) for arm, frames in arms.items()}
        self.assertEqual(float(decoded["observed"]["z"]), 1.)
        self.assertEqual(float(decoded["yoked"]["z"]), -1.)
        self.assertAlmostEqual(float(decoded["observed"]["q"]) + float(decoded["yoked"]["q"]), 1., places=7)
        for arm in ("no_demo", "scene_only"):
            self.assertEqual(float(decoded[arm]["q"]), .5)
            np.testing.assert_array_equal(decoded[arm]["track_votes"], 0)
        movie[:, 75:] = 99  # Signal and private-looking scalar content are not decoder inputs.
        self.assert_tree_equal(belief.decode_movie(movie, 2.), decoded["observed"])

    def test_calibrator_uses_own_contact_color_and_retains_negative_reliability(self):
        rows = [fit_row(color=0, outcome=1), fit_row(color=1, outcome=-1)]
        fitted = belief.calibrate(rows)
        self.assertEqual(fitted["selected_lambda"], 4.)
        self.assertEqual([row["lambda"] for row in fitted["losses"]], list(belief.LAMBDA_GRID))
        self.assertEqual(fitted["contact_coverage"], dict(mean=1., se=0.))
        for row in rows:
            row["contact_outcome"] *= -1
        self.assertEqual(belief.calibrate(rows)["selected_lambda"], -4.)

    def test_calibrator_excludes_missing_contacts_and_uses_exact_neutral_ties(self):
        neutral = belief.calibrate([fit_row((1, 1)), fit_row((0, 0), -1, 0)])
        self.assertEqual(neutral["selected_lambda"], 0.)
        self.assertEqual(neutral["contacts"], 1)
        self.assertEqual(neutral["contact_coverage"]["mean"], .5)
        self.assertEqual(neutral["vote_coverage"]["mean"], .5)
        self.assertTrue(all(row["se_log_loss"] is None for row in neutral["losses"]))
        absent = belief.calibrate([fit_row(color=-1, outcome=0)])
        self.assertEqual(absent["selected_lambda"], 0.)
        self.assertTrue(all(row["mean_log_loss"] is None for row in absent["losses"]))

    def test_fitter_rejects_hidden_labels_and_projection_excludes_them(self):
        public = fit_row()
        for forbidden in ("rule", "teacher_energy", "teacher_food", "teacher_terminal_W"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                belief.calibrate([{**public, forbidden: 1}])
        raw = dict(z=.5, track_votes=np.array([3, 1]), contact_color=0, contact_outcome=1,
                   rule=1, teacher_energy=-999, teacher_food=np.ones(300))
        self.assertEqual(set(belief.calibration_input(raw)), belief.FIT_FIELDS)
        projected = belief.calibrate([belief.calibration_input(raw)])
        raw.update(rule=0, teacher_energy=1234, teacher_food=np.zeros(300))
        self.assertEqual(belief.calibrate([belief.calibration_input(raw)]), projected)
        with self.assertRaises(ValueError):
            belief.calibrate([{**public, "z": -.5}])

    def test_first_contact_reads_prior_grid_and_own_energy_not_regrown_item(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=1)
        initial = np.zeros((4, 4), np.int32)
        initial[1, 1] = 1
        grids = np.zeros((6, 4, 4), np.int32)
        grids[0, 1, 1] = 2  # Opposite color after collection/regrowth.
        positions = np.full((6, 1, 2), 1, np.int32)
        energies = np.full((6, 1), 1.247, np.float32)
        alive = np.ones((6, 1), bool)
        first = belief.first_contact(initial, grids, positions, energies, alive, c)
        self.assertEqual(int(first["contact_color"]), 0)
        self.assertEqual(int(first["contact_outcome"]), 1)
        self.assertEqual(int(first["contact_step"]), 1)
        self.assertEqual(float(first["contact_denergy"]), float(energies[0, 0] - np.float32(c.e0)))
        energies[:] = .747
        self.assertEqual(int(belief.first_contact(initial, grids, positions, energies, alive, c)["contact_outcome"]), -1)
        missed = belief.first_contact(np.zeros_like(initial), np.zeros_like(grids), positions, energies, alive, c)
        self.assertEqual(int(missed["contact_color"]), -1)
        self.assertEqual(int(missed["contact_outcome"]), 0)
        self.assertEqual(int(missed["contact_step"]), c.T + 1)

    def test_readout_uses_all_five_actual_local_destinations_and_is_bounded(self):
        c = seed.Cfg(M=1, R=1)
        observation = np.zeros((1, seed.dims(c)[1]), np.float32)
        # Action order stay, north, south, west, east; use contrasting items.
        observation[0, [4, 7, 5]] = 1
        observation[0, 9 + np.array([1, 3])] = 1
        np.testing.assert_array_equal(seed.social_action_bias(jnp.asarray(observation), jnp.array([1.]), c),
                                      [[3, -3, 3, -3, 3]])
        np.testing.assert_array_equal(seed.social_action_bias(jnp.asarray(observation), jnp.array([0.]), c),
                                      [[-3, 3, -3, 3, -3]])
        for q in (belief.belief_probability(0, 4.), belief.belief_probability(1., 0.)):
            np.testing.assert_array_equal(seed.social_action_bias(jnp.asarray(observation), jnp.array([q]), c), 0)

    def test_default_and_neutral_paths_match_legacy_actual_trajectory(self):
        source = subprocess.check_output(["git", "show", "839846f57186174a91bc33abe901fdab3b04a96f:seed/seed.py"],
                                         cwd=Path(__file__).resolve().parents[1], text=True, timeout=10)
        old = types.ModuleType("before_social_belief")
        exec(compile(source, "before_social_belief.py", "exec"), old.__dict__)
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        wk, dk = jax.random.PRNGKey(51), jax.random.PRNGKey(52)
        genome = seed.biology_prior(c)
        kwargs = dict(record=True, diagnostics=True, _return_weights=True)
        baseline = jax.tree.map(np.asarray, old.rollout(genome, wk, dk, 0, c, **kwargs))
        self.assert_tree_equal(seed.rollout(genome, wk, dk, 0, c, **kwargs), baseline)
        self.assert_tree_equal(seed.rollout(genome, wk, dk, 0, c, _belief_q=None, **kwargs), baseline)
        neutral = jax.tree.map(np.asarray, seed.rollout(genome, wk, dk, 0, c, _belief_q=jnp.array([.5]), **kwargs))
        np.testing.assert_array_equal(neutral.pop("_belief")["bias"], 0)
        neutral["traj"].pop("_belief")
        self.assert_tree_equal(neutral, baseline)

    def test_paired_six_arm_state_reset_and_action_evidence(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        keys = belief.family_keys(52, 2)
        row = belief.evaluate_family(np.asarray(flat), keys, 2, 2., c)
        for arm in belief.ARMS:
            np.testing.assert_array_equal(row[arm + "_birth_sha256"], row["observed_birth_sha256"])
        for name in ("W", "x", "Wp"):
            np.testing.assert_array_equal(row["observed_solo_initial_" + name + "_sha256"],
                                          row["belief_reset_solo_initial_" + name + "_sha256"])
        self.assertEqual(row["birth_parent_frames"].shape[0], 0)
        self.assertFalse(bool(row["birth_parent_readout_enabled"]))
        unassisted = seed.rollout(seed.biology_prior(c), keys[2], keys[3], 0, c,
                                  record=True, diagnostics=True, _return_weights=True)
        birth = belief._birth_observer(flat, keys[3], c)
        normal_birth = unassisted["_weight_state"]["birth_W"][0]
        np.testing.assert_array_equal(birth["initial"]["W"], normal_birth)
        np.testing.assert_array_equal(birth["mask"], unassisted["_weight_state"]["mask"][0])
        np.testing.assert_array_equal(row["observed_birth_sha256"], social._hash(normal_birth))
        for name in ("W", "x", "Wp"):
            np.testing.assert_array_equal(birth["final"][name], birth["initial"][name])
        for name in social.EVENTS:
            np.testing.assert_array_equal(row["birth_parent_" + name], unassisted["events"][name])
        for name, source_name in (("position", "pos"), ("grid", "grid"), ("energy", "energy")):
            np.testing.assert_array_equal(row["birth_parent_solo_" + name], unassisted["traj"][source_name])
        self.assertEqual(set(belief.summarize([row])["contrasts"]), set(belief.CONTROLS))
        for field in ("belief_reset_q", "belief_reset_solo_initial_x_sha256",
                      "observed_belief_bias", "observed_belief_efference"):
            changed = copy.deepcopy(row)
            changed[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                belief.validate_family(changed, c)

    def test_calibration_family_is_unexposed_and_keeps_own_outcome_evidence(self):
        c = seed.Cfg(G=4, M=1, T=6, R=1, H=3, n_items=2)
        flat, _ = ravel_pytree(seed.biology_prior(c))
        row = belief.calibration_family(np.asarray(flat), belief.family_keys(51, 0), 0, c)
        self.assertEqual(row["pupil_frames"].shape[0], 0)
        for name in ("W", "x", "Wp"):
            np.testing.assert_array_equal(row["pupil_watch_initial_" + name + "_sha256"],
                                          row["pupil_solo_initial_" + name + "_sha256"])
        self.assertEqual(belief.calibrate([belief.calibration_input(row)])["families"], 1)
        changed = copy.deepcopy(row)
        changed["contact_outcome"] = np.asarray(1 - int(row["contact_outcome"]), np.int32)
        with self.assertRaises(ValueError):
            belief.validate_calibration_family(changed, c)


if __name__ == "__main__":
    unittest.main()
