"""E127 harness/compatibility fixtures; no saved endpoint or final world is opened."""
import copy
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct, seed, encounter_diagnostic as diagnostic


def same_bytes(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.tobytes(), expected.tobytes())


def compact_family(index, *, pre_s=-.4, post_s=.4, grid_success=True):
    row = dict(base_index=np.asarray(index, np.int32),
        keys=np.arange(index * 12, index * 12 + 12, dtype=np.uint32).reshape(6, 2),
        pre_contacts=np.asarray(8 if index % 2 == 0 else 16, np.int32),
        plastic_genome_sha256=np.zeros(32, np.uint8), direct_genome_sha256=np.ones(32, np.uint8))
    for arm in diagnostic.ARMS:
        signed = np.full((2, 4), pre_s, np.float64)
        if arm.endswith("reversal"):
            signed[:, 1:] = post_s
        elif arm.endswith("frozen"):
            signed[:, 1:] = 0.
        local = np.stack((.4 + signed / 2, .4 - signed / 2,
                          np.full_like(signed, .8), np.full_like(signed, .2), signed), axis=-1)
        good = arm.endswith("reversal") and grid_success
        grid = np.broadcast_to(np.asarray([1, 0, 0, 1, 2, 1, 1] if good else
                                         [0, 0, 1, 0, 51, 0, 1], np.float64), (2, 7)).copy()
        row.update({arm + "_local_by_rule": local, arm + "_local": local.mean(0),
                    arm + "_grid_by_rule": grid, arm + "_grid": grid.mean(0)})
    return row


class EncounterAccountingTests(unittest.TestCase):
    def test_five_way_uniform_and_current_obsolete_rule_readers(self):
        uniform = diagnostic.local_outcomes(np.full((2, 4, 12, 5), .2, np.float64))
        np.testing.assert_allclose(uniform, np.broadcast_to([.2, .2, .4, .6, 0.], uniform.shape), atol=1e-15)
        for correct in (False, True):
            values = np.zeros((2, 4, 12, 5), np.float64)
            for rule in (0, 1):
                actions = diagnostic.PLACEMENTS[:, 1 - rule if correct else rule]
                for time in range(4):
                    values[rule, time, np.arange(12), actions] = 1.
            outcome = diagnostic.local_outcomes(values)
            np.testing.assert_array_equal(outcome[:, :, 4], 1 if correct else -1)
            np.testing.assert_array_equal(outcome[:, :, 2], 1)

    def test_local_gate_requires_old_preference_and_actual_reversal(self):
        good = diagnostic.summarize([compact_family(i) for i in range(4)])
        self.assertTrue(good["architectures"]["plastic"]["local_pass"])
        self.assertTrue(good["architectures"]["plastic"]["transfer_pass"])
        self.assertTrue(good["hebbian"]["hebbian_local_pass"])
        no_old = diagnostic.summarize([compact_family(i, pre_s=0.) for i in range(4)])
        self.assertFalse(no_old["architectures"]["plastic"]["local_pass"])
        modest = diagnostic.summarize([compact_family(i, post_s=.05) for i in range(4)])
        self.assertFalse(modest["architectures"]["plastic"]["local_pass"])

    def test_no_contact_is_zero_not_conditional_success(self):
        result = diagnostic.summarize([compact_family(i, grid_success=False) for i in range(4)])
        self.assertEqual(result["arms"]["plastic_reversal"]["grid"]["first_food"]["mean"], 0.)
        self.assertEqual(result["arms"]["plastic_reversal"]["grid"]["food_minus_poison"]["mean"], 0.)
        self.assertFalse(result["architectures"]["plastic"]["transfer_pass"])

    def test_cluster_se_and_direction_description(self):
        rows = [compact_family(0, post_s=.2), compact_family(1, post_s=.8)]
        result = diagnostic.summarize(rows)
        stat = result["architectures"]["plastic"]["local"]["post_target"]
        self.assertEqual(stat["n"], 2)
        self.assertAlmostEqual(stat["mean"], .5)
        self.assertAlmostEqual(stat["se"], .3)
        self.assertEqual(result["by_initial_rule"]["0"]["plastic"]["S"]["n"], 2)
        with self.assertRaises(ValueError):
            diagnostic.summarize([rows[0], copy.deepcopy(rows[0])])
        self.assertIsNone(diagnostic.summarize(rows[:1])["architectures"]["plastic"]["local_pass"])

    def test_signed_first_outcome_uses_its_own_family_se(self):
        rows = [compact_family(i) for i in range(2)]
        # Each family's two replays have opposite outcomes, so family Y is exactly zero.
        for row in rows:
            by_rule = np.asarray([[1, 0, 0, 1, 1, 1, 1], [0, 1, 0, -1, 1, -1, 1]], np.float64)
            row["plastic_reversal_grid_by_rule"] = by_rule
            row["plastic_reversal_grid"] = by_rule.mean(0)
        result = diagnostic.summarize(rows)
        preference = result["architectures"]["plastic"]["transfer"]["food_minus_poison"]
        self.assertEqual(preference["mean"], 0.)
        self.assertEqual(preference["se"], 0.)
        self.assertFalse(result["architectures"]["plastic"]["transfer_pass"])


class EncounterDynamicsTests(unittest.TestCase):
    def test_scripts_balance_colors_directions_and_boundary(self):
        assignment = diagnostic.assignments_manifest(2026120600, 4)
        self.assertEqual(sorted(assignment["pre_contacts"].tolist()), [8, 8, 16, 16])
        for index, pre in enumerate(assignment["pre_contacts"]):
            script = diagnostic.contact_script(diagnostic.evaluation_keys(2026120600, index)[0], int(pre))
            for pair in script.reshape(-1, 2, 5):
                self.assertEqual(sorted(pair[:, 2].tolist()), [1, 2])
            for block in script.reshape(-1, 8, 5):
                self.assertEqual(set(map(tuple, block[:, [2, 3]].tolist())),
                                 {(color, action) for color in (1, 2) for action in range(1, 5)})
            np.testing.assert_array_equal(script[:, 4], diagnostic.OPPOSITE[script[:, 3]])

    def test_unforced_transition_matches_original_rollout_for_both_encodings(self):
        c = diagnostic.CFG._replace(T=8)
        keys = jax.random.split(jax.random.PRNGKey(2026120600), 4)
        for controller in ("plastic", "direct"):
            with self.subTest(controller=controller):
                flat = ravel_pytree(seed.biology_prior(c))[0] if controller == "plastic" else direct.init_genome(c, keys[2])
                born = diagnostic._birth(flat, controller, keys[1], keys[0], c)
                grid, pos = seed.init_world(keys[0], c)
                initial = diagnostic._fresh_body(born, grid, pos, keys[0], c)
                final, trace = diagnostic._free_grid(controller, c, steps=c.T)(flat, initial, 1)
                original = seed.rollout(diagnostic._unravel(c)(flat) if controller == "plastic" else None,
                    keys[0], keys[1], 1, c, plastic=controller == "plastic", predictor_plastic=controller == "plastic",
                    direct_weights=direct.decode(flat, c) if controller == "direct" else None,
                    record=True, diagnostics=True, _world_protocol="reversal-random-v1")
                for name, prior in (("grid", "grid"), ("position", "pos"), ("energy", "energy"),
                                    ("signal", "sig"), ("alive_after", "alive"), ("surprise", "surprise")):
                    same_bytes(self, trace[name], original["traj"][prior])
                for name in ("food", "poison", "alive_before", "alive_after"):
                    same_bytes(self, trace[name], original["events"][name])
                for name in ("observation", "action", "action_key", "efference", "activity", "denergy"):
                    same_bytes(self, trace[name], original["_world"][name])
                for name in ("W", "x", "Wp"):
                    same_bytes(self, final[name], original["_world_final"][name])

    def test_forced_action_efference_and_actual_next_tick_feedback(self):
        c = diagnostic.CFG
        flat = ravel_pytree(seed.biology_prior(c))[0]
        keys = jax.random.split(jax.random.PRNGKey(2026120600), 2)
        initial = diagnostic._birth(flat, "plastic", keys[0], keys[1], c)
        script = jnp.asarray([[4, 1, 1, 4, 3], [1, 4, 2, 4, 3]], jnp.int32)
        _, trace = diagnostic._chamber("plastic", c)(flat, initial, script, 0)
        np.testing.assert_array_equal(trace["action"][:, 0], [4, 3, 4, 3])
        np.testing.assert_array_equal(trace["efference"][:, 0], np.eye(5, dtype=np.float32)[[4, 3, 4, 3]])
        np.testing.assert_array_equal(trace["food"][:, 0], [True, False, False, False])
        np.testing.assert_array_equal(trace["poison"][:, 0], [False, False, True, False])
        same_bytes(self, trace["incoming_denergy"][1:], trace["denergy"][:-1])
        np.testing.assert_array_equal(trace["observation"][1:, :, -2], 4 * np.asarray(trace["denergy"][:-1]))
        np.testing.assert_array_equal(trace["incoming_activity"][1:, :, -6:-1], trace["efference"][:-1])
        np.testing.assert_array_equal(np.asarray(trace["position"])[[1, 3]], np.broadcast_to(np.asarray(initial["pos"]), (2, 1, 2)))

    def test_hebbian_lesion_starts_when_changed_feedback_arrives(self):
        c = diagnostic.CFG
        flat = ravel_pytree(seed.biology_prior(c))[0]
        keys = jax.random.split(jax.random.PRNGKey(2026120600), 2)
        initial = diagnostic._birth(flat, "plastic", keys[0], keys[1], c)
        script = jnp.asarray([[4, 1, 1, 4, 3], [1, 4, 2, 4, 3]], jnp.int32)
        _, intact = diagnostic._chamber("plastic", c)(flat, initial, script, 1)
        _, frozen = diagnostic._chamber("plastic", c, 1)(flat, initial, script, 1)
        np.testing.assert_array_equal(frozen["hebbian_enabled"], [True, False, False, False])
        np.testing.assert_array_equal(frozen["predictor_enabled"], True)
        for name in ("observation", "action", "activity", "food", "poison", "energy", "denergy"):
            same_bytes(self, intact[name][0], frozen[name][0])

    def test_unbounded_direct_weights_survive_chamber_probe_and_grid(self):
        c = diagnostic.CFG
        keys = jax.random.split(jax.random.PRNGKey(2026120600), 5)
        flat = direct.init_genome(c, keys[0]).at[0].set(7.25).at[1].set(-6.5)
        initial = diagnostic._birth(flat, "direct", keys[1], keys[2], c)
        self.assertGreater(float(jnp.abs(initial["W"]).max()), c.wmax)
        script = jnp.asarray([[4, 1, 1, 4, 3], [1, 4, 2, 4, 3]], jnp.int32)
        snapshot, _ = diagnostic._chamber("direct", c)(flat, initial, script, 0)
        snapshot_hashes = {name: diagnostic.array_hash(snapshot[name]) for name in diagnostic.NEURAL}
        probes = diagnostic._readouts("direct", c)(flat, snapshot, jax.random.split(keys[3], 12))
        grid, pos = seed.init_world(keys[4], c)
        fresh = diagnostic._fresh_body(snapshot, grid, pos, keys[3], c)
        final, _ = diagnostic._free_grid("direct", c, steps=4)(flat, fresh, 1)
        for state in (snapshot, fresh, final):
            same_bytes(self, state["W"], initial["W"])
            same_bytes(self, state["Wp"], initial["Wp"])
        for phase in ("initial", "final"):
            for orientation in range(12):
                same_bytes(self, probes[phase]["W"][orientation], initial["W"])
                same_bytes(self, probes[phase]["Wp"][orientation], initial["Wp"])
        for name, digest in snapshot_hashes.items():
            same_bytes(self, diagnostic.array_hash(snapshot[name]), digest)
        same_bytes(self, fresh["x"], snapshot["x"])
        np.testing.assert_array_equal(fresh["denergy"], 0)
        np.testing.assert_array_equal(fresh["energy"], c.e0)


if __name__ == "__main__":
    unittest.main()
