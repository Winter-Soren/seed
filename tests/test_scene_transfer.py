"""Protocol checks use synthetic scenes and do not open registered assignments."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from seed import encounter_diagnostic as encounter
from seed import scene_transfer as scene
from seed import seed


class FirstViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = scene.CFG
        keys = jax.random.split(jax.random.PRNGKey(7), 4)
        cls.flat, _ = ravel_pytree(seed.init_genome(cls.c, keys[0]))
        cls.state = encounter._birth(cls.flat, "plastic", keys[1], keys[2], cls.c)
        cls.probe_keys = jax.random.split(keys[3], 12)
        cls.empty = jnp.zeros((cls.c.G, cls.c.G), jnp.int32)

    def test_mask_is_legacy_readout_and_only_hides_nonfocal_pixels(self):
        background = self.empty.at[6, 6].set(1).at[10, 10].set(2).at[0, 0].set(2)
        legacy = encounter._readouts("plastic", self.c)(self.flat, self.state, self.probe_keys)
        _, trace = scene._first_decisions()(self.flat, self.state, background, self.probe_keys, 0)
        np.testing.assert_array_equal(trace["observation"][0], legacy["observation"])
        np.testing.assert_allclose(trace["probabilities"][0], legacy["probabilities"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(trace["logits"][0], legacy["logits"], rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(trace["action"][0], legacy["sampled_action"])
        for field in ("grid_before", "position_before", "action_key", "regrowth_key", "incoming_denergy"):
            np.testing.assert_array_equal(trace[field][0], trace[field][1])
        for i, placement in enumerate(encounter.PLACEMENTS):
            grid = np.asarray(trace["grid_before"][0, i])
            destinations = 8 + np.asarray(seed.MOVES)
            items = grid[destinations[:, 0], destinations[:, 1]]
            expected = np.zeros(5, np.int32)
            expected[placement] = [1, 2]
            np.testing.assert_array_equal(items, expected)
        actual = np.asarray(trace["observation"])
        np.testing.assert_array_equal(actual[1, :, :, 50:], actual[0, :, :, 50:])
        delta = actual[1] - actual[0]
        np.testing.assert_array_equal(np.flatnonzero(delta[0, 0]), [0, 49])
        np.testing.assert_array_equal(delta.sum(axis=-1), np.full((12, 1), 2))
        self.assertTrue(np.all(np.asarray(trace["forced_action"]) == -1))
        self.assertFalse(np.any(np.asarray(trace["hebbian_enabled"])))
        self.assertFalse(np.any(np.asarray(trace["predictor_enabled"])))

    def test_hidden_distant_item_does_not_change_trajectory(self):
        states, traces = scene._first_decisions()(self.flat, self.state, self.empty.at[0, 0].set(1), self.probe_keys, 1)
        for values in (*states.values(), *traces.values()):
            np.testing.assert_array_equal(values[0], values[1])

    def test_ordinary_tick_after_mask_uses_full_observation_and_delivers_feedback(self):
        background = self.empty.at[6, 6].set(1).at[10, 10].set(2)
        states, trace = scene._first_decisions()(self.flat, self.state, background, self.probe_keys, 0)
        first = jax.tree.map(lambda value: value[0, 0], states)
        _, rest = encounter._free_grid("plastic", self.c, steps=1)(self.flat, first, 0)
        expected = seed.observe(first["grid"], first["pos"], first["alive"], first["sig"], first["energy"], first["denergy"], self.c)
        np.testing.assert_allclose(rest["observation"][0], expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(rest["incoming_denergy"][0], trace["denergy"][0, 0])
        self.assertTrue(bool(rest["hebbian_enabled"][0]))
        self.assertTrue(bool(rest["predictor_enabled"][0]))

        from tools import audit_scene_transfer as auditor
        from tools import audit_scene_transfer_physics as physics

        turn_zero = {name: values[0, 0:1] for name, values in scene._compact(trace).items()}
        full_trace = scene.replay._join_trace(turn_zero, scene._compact(rest))
        hub = np.asarray([[8, 8]], np.int32)
        focal = np.asarray(encounter._pair_grid(hub, jnp.asarray(encounter.PLACEMENTS[0]), self.c))
        body = auditor.neutral_body(np.asarray(background) + focal, hub, self.probe_keys[0])
        tape = auditor.random_helpers()[2]
        options = dict(learning=np.asarray([False, True]), masked_first=focal)
        physics.verify(full_trace, body, [0, 0], tape, **options)
        changed = full_trace["observation"].copy()
        changed[1, 0, 0] = 1 - changed[1, 0, 0]
        with self.assertRaisesRegex(auditor.base.AuditError, "first view masks"):
            physics.verify(dict(full_trace, observation=changed), body, [0, 0], tape, **options)


class ContrastTests(unittest.TestCase):
    def test_pass_and_retirement_are_distinct(self):
        def panel(masked, full, attenuation, reference):
            def estimate(mean, se=.005):
                return dict(mean=mean, lower_2se=mean - 2 * se, upper_2se=mean + 2 * se)
            return scene.gate({"histories": {"stationary": dict(masked_gain=estimate(masked), full_gain=estimate(full),
                                    attenuation=estimate(attenuation), reference_gain=estimate(reference))}})

        partial = panel(.15, .08, .07, -.15)
        self.assertTrue(partial["passed"] and partial["partial_attenuation"])
        erased = panel(.15, -.02, .17, -.15)
        self.assertTrue(erased["passed"] and erased["advantage_erased"])
        retired = panel(.15, .14, .01, -.15)
        self.assertTrue(retired["practical_loss_excluded"] and retired["reference_deficit"])
        self.assertFalse(retired["passed"])
        unknown = panel(.04, .03, .01, -.15)
        self.assertFalse(unknown["practical_loss_excluded"])

    def test_first_score_respects_history_and_approach_split(self):
        probabilities = np.full((2, 2, 2, 2, 12, 1, 5), .2)
        actions = np.zeros((2, 2, 2, 2, 12, 1), np.int32)
        for initial in (0, 1):
            for history in (0, 1):
                rule = initial ^ history
                for k, placement in enumerate(encounter.PLACEMENTS):
                    probabilities[1, initial, history, 0, k, 0] = 0
                    probabilities[1, initial, history, 0, k, 0, placement[rule]] = 1
                    actions[1, initial, history, 0, k, 0] = placement[rule]
        row = dict(base_index=0, background=np.zeros((16, 16)), first_probabilities=probabilities, first_action=actions)
        for prefix, lead in (("scene_", (2, 2, 2, 2)), ("reference_", (2, 2, 2))):
            row.update({prefix + name: np.zeros(lead + (50, 1)) for name in ("food", "poison", "alive_after")})
        scores = scene.summary_input(row)
        np.testing.assert_array_equal(scores["local"][1, ..., 0], 1)
        np.testing.assert_array_equal(scores["sampled"][1, ..., 0], 1)
        np.testing.assert_allclose(sum(scores["decomposition"].values()), scores["local"][..., 0] - scores["local"][..., 1])
        np.testing.assert_array_equal(scores["scene"]["no_contact"], True)
        np.testing.assert_array_equal(scores["scene"]["latency"], 50)
        panel = scene.summarize_rows([scores, dict(scores, base_index=1)])
        self.assertEqual(panel["histories"]["stationary"]["reference"]["parent"]["contact_accuracy"]["mean"], None)


if __name__ == "__main__":
    unittest.main()
