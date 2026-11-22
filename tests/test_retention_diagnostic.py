"""E128 synthetic contracts; no saved genome, run, or held-out seed is opened."""
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct, seed, encounter_diagnostic as encounter, retention_diagnostic as retention


SYNTHETIC_MASTER = 0xE128F0


def same_bytes(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.tobytes(), expected.tobytes())


def acquired_fixture(controller):
    c = retention.CFG
    keys = jax.random.split(jax.random.PRNGKey(SYNTHETIC_MASTER), 6)
    flat = (ravel_pytree(seed.biology_prior(c))[0] if controller == "plastic"
            else direct.init_genome(c, keys[0]).at[0].set(7.25).at[1].set(-6.5))
    state = encounter._birth(flat, controller, keys[1], keys[2], c)
    _, inputs, _, neurons, _ = seed.dims(c)
    state = dict(state,
        grid=encounter._pair_grid(state["pos"], jnp.asarray([1, 4], jnp.int32), c),
        x=jnp.asarray(np.linspace(-.6, .6, neurons, dtype=np.float32)[None]),
        Wp=jnp.asarray(np.linspace(-.02, .02, inputs * c.H, dtype=np.float32).reshape(1, inputs, c.H)),
        energy=jnp.asarray([.808], jnp.float32), denergy=jnp.asarray([np.float32(.808) - np.float32(.811)]),
        sig=jnp.asarray([.25], jnp.float32), t=jnp.asarray(64, jnp.int32),
        counts=jnp.asarray([[[16., 16.], [0., 0.]]], jnp.float32))
    return flat, state, keys


class RetentionDynamicsTests(unittest.TestCase):
    def test_active_retention_carries_state_and_one_action_stream(self):
        for controller in ("plastic", "direct"):
            with self.subTest(controller=controller):
                flat, acquired, keys = acquired_fixture(controller)
                before = jax.tree.map(lambda value: np.asarray(value).copy(), acquired)
                original_free_grid = encounter._free_grid
                with mock.patch.object(encounter, "_free_grid", wraps=original_free_grid) as calls:
                    states, trace = retention.retention_states(flat, acquired, keys[3], 1, controller)
                self.assertEqual([call.kwargs["steps"] for call in calls.call_args_list], [1, 32])
                for field in acquired:
                    same_bytes(self, acquired[field], before[field])
                    if field not in ("grid", "key"):
                        same_bytes(self, states[0][field], acquired[field])
                np.testing.assert_array_equal(states[0]["grid"], 0)
                same_bytes(self, states[0]["key"], keys[3])
                # Independent direct use of the unchanged operators establishes
                # that neither continuation nor the intermediate readout resets carry.
                first, first_trace = original_free_grid(controller, retention.CFG, steps=1)(flat, states[0], 1)
                last, last_trace = original_free_grid(controller, retention.CFG, steps=32)(flat, first, 1)
                for field in acquired:
                    same_bytes(self, states[1][field], first[field])
                    same_bytes(self, states[2][field], last[field])
                for field in trace:
                    same_bytes(self, trace[field], np.concatenate((np.asarray(first_trace[field]), np.asarray(last_trace[field]))))
                for field in ("grid_before", "grid", "food", "poison", "item"):
                    np.testing.assert_array_equal(trace[field], 0)
                np.testing.assert_array_equal(trace["forced_action"], -1)
                same_bytes(self, trace["action"], trace["sampled_action"])
                same_bytes(self, trace["efference"], np.eye(5, dtype=np.float32)[trace["action"]])
                same_bytes(self, trace["incoming_activity"][0], acquired["x"])
                same_bytes(self, trace["incoming_activity"][1:], trace["activity"][:-1])
                same_bytes(self, trace["incoming_denergy"][0], acquired["denergy"])
                same_bytes(self, trace["incoming_denergy"][1:], trace["denergy"][:-1])
                same_bytes(self, trace["key_before"][1:], trace["key_after"][:-1])
                np.testing.assert_array_equal(trace["observation"][..., :-3], 0)
                same_bytes(self, trace["observation"][..., -3], trace["energy_before"] - np.float32(1))
                same_bytes(self, trace["observation"][..., -2], np.float32(4) * trace["incoming_denergy"])
                np.testing.assert_array_equal(trace["observation"][..., -1], 1)
                same_bytes(self, trace["energy"], trace["energy_before"] - np.float32(retention.CFG.decay))
                for state, elapsed in zip(states, retention.RETENTION_TICKS):
                    self.assertEqual(int(state["t"]), int(acquired["t"]) + elapsed)
                    same_bytes(self, state["counts"], acquired["counts"])
                    same_bytes(self, state["mask"], acquired["mask"])
                    if controller == "direct":
                        same_bytes(self, state["W"], acquired["W"])
                        same_bytes(self, state["Wp"], acquired["Wp"])
                np.testing.assert_array_equal(trace["hebbian_enabled"], controller == "plastic")
                np.testing.assert_array_equal(trace["predictor_enabled"], controller == "plastic")

    def test_zero_delay_readout_matches_acquired_state_and_all_delays_get_fresh_body(self):
        for controller in ("plastic", "direct"):
            with self.subTest(controller=controller):
                flat, acquired, keys = acquired_fixture(controller)
                states, _ = retention.retention_states(flat, acquired, keys[3], 0, controller)
                probe_keys = jax.random.split(keys[4], 12)
                original = encounter._readouts(controller, retention.CFG)(flat, acquired, probe_keys)
                immediate = encounter._readouts(controller, retention.CFG)(flat, states[0], probe_keys)
                for field in ("observation", "decision_motor", "logits", "probabilities", "sampled_action", "action_key"):
                    same_bytes(self, immediate[field], original[field])
                grid, pos = seed.init_world(keys[5], retention.CFG)
                old_grid_start = encounter._fresh_body(acquired, grid, pos, keys[4], retention.CFG)
                zero_grid_start = encounter._fresh_body(states[0], grid, pos, keys[4], retention.CFG)
                for field in old_grid_start:
                    same_bytes(self, zero_grid_start[field], old_grid_start[field])
                for state in states:
                    readout = encounter._fresh_body(state, grid, pos, keys[4], retention.CFG)
                    for field in encounter.NEURAL:
                        same_bytes(self, readout[field], state[field])
                    for field in encounter.BODY:
                        same_bytes(self, readout[field], old_grid_start[field])
                    probes = encounter._readouts(controller, retention.CFG)(flat, state, probe_keys)
                    for field in ("W", "Wp", "mask"):
                        expected = np.broadcast_to(np.asarray(state[field]), np.asarray(probes["initial"][field]).shape)
                        same_bytes(self, probes["initial"][field], expected)
                        same_bytes(self, probes["final"][field], expected)
                    expected_x = np.broadcast_to(np.asarray(state["x"]), np.asarray(probes["initial"]["x"]).shape)
                    same_bytes(self, probes["initial"]["x"], expected_x)


class RetentionAccountingTests(unittest.TestCase):
    def test_family_streams_and_boundary_assignment_are_reproducible(self):
        manifest = retention.assignments_manifest(SYNTHETIC_MASTER, 4)
        repeat = retention.assignments_manifest(SYNTHETIC_MASTER, 4)
        for name in manifest:
            same_bytes(self, manifest[name], repeat[name])
        self.assertEqual(sorted(manifest["pre_contacts"].tolist()), [8, 8, 16, 16])
        all_keys = [retention.evaluation_keys(SYNTHETIC_MASTER, index) for index in range(4)]
        self.assertEqual(len({key.tobytes() for family in all_keys for key in family}), 28)
        for index, keys in enumerate(all_keys):
            self.assertEqual(keys.shape, (7, 2))
            same_bytes(self, keys, retention.evaluation_keys(SYNTHETIC_MASTER, index))
        self.assertFalse(np.array_equal(manifest["key"], encounter.assignments_manifest(SYNTHETIC_MASTER, 4)["key"]))
        for index in (-1, True, retention.ASSIGNMENT_TAG):
            with self.assertRaises(ValueError):
                retention.evaluation_keys(SYNTHETIC_MASTER, index)
        bad = all_keys[0].copy()
        bad[-1] = bad[0]
        with self.assertRaises(ValueError):
            retention._inputs(bad, 0, 8, retention.CFG)

    def test_compact_projection_preserves_rules_delays_and_no_contact(self):
        row = dict(base_index=np.asarray(0, np.int32), keys=retention.evaluation_keys(SYNTHETIC_MASTER, 0),
            pre_contacts=np.asarray(8, np.int32), initial_rules=np.asarray([0, 1], np.int32),
            retention_ticks=np.asarray(retention.RETENTION_TICKS, np.int32),
            plastic_genome_sha256=np.zeros(32, np.uint8), direct_genome_sha256=np.ones(32, np.uint8))
        probabilities = np.empty((2, 4, 12, 5), np.float32)
        for rule in (0, 1):
            for time, (food, poison) in enumerate(((.1, .6), (.3, .1), (.4, .1), (.5, .1))):
                for orientation, pair in enumerate(encounter.PLACEMENTS):
                    probabilities[rule, time, orientation] = np.float32((1 - food - poison) / 3)
                    probabilities[rule, time, orientation, pair[1 - rule]] = np.float32(food)
                    probabilities[rule, time, orientation, pair[rule]] = np.float32(poison)
        food, poison = np.zeros((2, 3, 50, 1), bool), np.zeros((2, 3, 50, 1), bool)
        poison[0, 0, 3, 0], food[0, 0, 4, 0], food[1, 0, 0, 0], food[1, 2, 49, 0] = True, True, True, True
        alive = np.ones((2, 3, 50, 1), np.float32)
        alive[0, 2] = 0  # A dead, no-contact assignment remains in every denominator.
        for arm in retention.ARMS:
            row.update({arm + "_baseline_probe_probabilities": probabilities[:, 0],
                arm + "_probe_probabilities": probabilities[:, 1:], arm + "_grid_food": food,
                arm + "_grid_poison": poison, arm + "_grid_alive_after": alive})
        # This fixture isolates the projection from the full raw physical-schema
        # checker; no fake simulation or source-state integrity is asserted.
        with mock.patch.object(retention, "validate_world"):
            compact = retention.summary_input(row)
        for arm in retention.ARMS:
            local, grid = compact[arm + "_local_by_rule"], compact[arm + "_grid_by_rule"]
            self.assertEqual(local.shape, (2, 3, 5))
            self.assertEqual(grid.shape, (2, 3, 7))
            self.assertEqual(local.dtype, np.float64)
            np.testing.assert_allclose(local[:, :, 4], [[.2, .3, .4], [.2, .3, .4]], atol=3e-8, rtol=0)
            np.testing.assert_allclose(compact[arm + "_pre_old_by_rule"], .5, atol=3e-8, rtol=0)
            np.testing.assert_array_equal(grid[0, 0], [0, 1, 0, -1, 4, 0, 1])
            np.testing.assert_array_equal(grid[1, 0], [1, 0, 0, 1, 1, 1, 1])
            np.testing.assert_array_equal(grid[0, 2], [0, 0, 1, 0, 51, 0, 0])
            np.testing.assert_array_equal(grid[1, 2], [1, 0, 0, 1, 50, 1, 1])
            self.assertEqual(grid[:, 0, 3].mean(), 0.)
            self.assertEqual(grid[:, 2, 0].mean(), .5)
        self.assertFalse(any("activity" in key or "probabilities" in key or "grid_initial" in key for key in compact))


if __name__ == "__main__":
    unittest.main()
