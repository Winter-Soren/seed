import unittest
from unittest.mock import patch

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from seed import contact_gate as gate
from seed import encounter_diagnostic as encounter
from seed import seed
from tools import audit_contact_gate as independent


class ContactGateTests(unittest.TestCase):
    def test_uses_previous_item_under_actual_move_and_not_energy_sign(self):
        width = 2 * gate.CFG.R + 1
        area = width**2
        center = area // 2
        obs = np.zeros(103, np.float32)
        obs[area + center - 1] = 1
        # A clipped gain can be zero at the cap, an empty turn costs energy.
        obs[-2] = 0
        self.assertTrue(bool(gate.contact_due(obs, jnp.eye(5)[3], True)))
        self.assertFalse(bool(gate.contact_due(obs, jnp.eye(5)[4], True)))
        self.assertFalse(bool(gate.contact_due(obs, np.zeros(5, np.float32), True)))
        self.assertFalse(bool(gate.contact_due(obs, jnp.eye(5)[3], False)))
        obs[area + center - 1] = 0
        obs[-2] = 1
        self.assertFalse(bool(gate.contact_due(obs, jnp.eye(5)[3], True)))
        obs[center] = 1
        self.assertTrue(bool(gate.contact_due(obs, jnp.eye(5)[0], True)))

    def test_zero_clipped_gain_triggers_on_feedback_tick_only(self):
        c = gate.CFG
        g = seed.biology_prior(c, jax.random.PRNGKey(14))
        flat = ravel_pytree(g)[0]
        state = encounter._birth(flat, "plastic", jax.random.PRNGKey(1), jax.random.PRNGKey(2), c)
        state = dict(state, energy=jnp.asarray([c.e_max], jnp.float32),
                     grid=encounter._pair_grid(state["pos"], jnp.asarray([3, 1]), c))
        after, first = gate.replay_tick(state, g, 0, 3, gate.CONTACT, False)
        following, second = gate.replay_tick(after, g, 0, 4, gate.CONTACT, False)
        _, third = gate.replay_tick(dict(following, grid=jnp.zeros_like(state["grid"])),
                                  g, 0, 0, gate.CONTACT, False)
        self.assertFalse(bool(first["hebbian"]))
        self.assertTrue(bool(first["weight_same"]))
        self.assertTrue(bool(second["hebbian"]))
        self.assertTrue(bool(second["predictor"]))
        self.assertEqual(float(second["incoming_denergy"]), 0.0)
        self.assertFalse(bool(third["hebbian"]))
        self.assertTrue(bool(third["weight_same"]))

    def test_equal_dose_no_lookahead_and_same_immediate_ablation(self):
        c = gate.CFG
        g = seed.biology_prior(c, jax.random.PRNGKey(19))
        flat = np.asarray(ravel_pytree(g)[0])
        row = gate.evaluate_world(flat, gate.evaluation_keys(42, 0), 0)
        acquisition = np.arange(32)
        scheduled = acquisition % 4 == row["sparse_offsets"][acquisition // 4]
        masks = {"native": np.ones(65, bool),
                 "contact": np.r_[acquisition % 4 == 3, np.zeros(33, bool)],
                 "sparse": np.r_[scheduled, np.zeros(33, bool)],
                 "reopened": np.r_[acquisition % 4 == 3, np.ones(33, bool)]}
        self.assertEqual(scheduled.sum(), 8)
        for history in ("stationary", "reversal"):
            for policy, mask in masks.items():
                prefix = f"{policy}_{history}_"
                self.assertTrue(np.all(row[prefix + "hebbian"] == mask))
                self.assertTrue(np.all(row[prefix + "weight_same"][:, ~mask]))
                for field in ("observation", "action", "incoming_denergy"):
                    np.testing.assert_array_equal(row[prefix + field], row[f"native_{history}_" + field])
            contact, reopened = (f"{policy}_{history}_" for policy in ("contact", "reopened"))
            np.testing.assert_array_equal(row[contact + "checkpoint_hashes"][:, 0],
                                          row[reopened + "checkpoint_hashes"][:, 0])
        endpoints = gate.summary_input(row)
        self.assertEqual(endpoints["arms"]["contact_reversal"]["solo_Y"].shape, (2, 2))
        self.assertEqual(endpoints["sparse_feedback_overlap"], (row["sparse_offsets"] == 3).mean())
        _, regenerate = independent.random_helpers()
        randomness = jax.tree.map(np.asarray, regenerate(row["keys"]))
        audited = independent.audit_family(row, 0, flat, row["keys"], randomness)
        np.testing.assert_array_equal(audited["pre_old"], endpoints["pre_old"])
        for arm, values in audited["arms"].items():
            for outcome, value in values.items():
                np.testing.assert_array_equal(value, endpoints["arms"][arm][outcome])

    def test_dead_probe_returns_no_events_and_never_creates_fresh_body(self):
        c = gate.CFG
        g = seed.biology_prior(c, jax.random.PRNGKey(20))
        flat = ravel_pytree(g)[0]
        key = jax.random.PRNGKey(7)
        state = encounter._birth(flat, "plastic", key, key, c)
        dead = dict(state, alive=jnp.zeros(1), energy=jnp.zeros(1))
        with patch.object(encounter, "_fresh_body", side_effect=AssertionError("revival")):
            proof = gate.probe(flat, dead, dead["grid"], dead["pos"], key,
                               jax.random.split(key, 12), 0)
        self.assertFalse(proof["probe_alive"])
        self.assertFalse(np.any(proof["grid_alive_before"]))
        self.assertFalse(np.any(proof["grid_food"]))
        self.assertFalse(np.any(proof["probe_probabilities"]))


if __name__ == "__main__":
    unittest.main()
