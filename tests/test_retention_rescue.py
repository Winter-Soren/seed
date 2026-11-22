"""E129 synthetic compatibility checks; no archived family or genome is loaded."""
import subprocess
import types
import unittest
from unittest import mock
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

import seed as seed_package
from seed import direct, seed, encounter_diagnostic as encounter
from seed import retention_diagnostic as retention, retention_rescue as rescue


HISTORICAL_COMMIT = "2d2586735061ac20481d6904f86791affb58b587"
SYNTHETIC_MASTER = 0xE129F0


def same_bytes(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.tobytes(), expected.tobytes())


def synthetic_state():
    c = rescue.CFG
    keys = jax.random.split(jax.random.PRNGKey(SYNTHETIC_MASTER), 7)
    flat = ravel_pytree(seed.biology_prior(c))[0]
    state = encounter._birth(flat, "plastic", keys[1], keys[2], c)
    _, inputs, _, neurons, _ = seed.dims(c)
    mask = ((np.arange(neurons)[:, None] >= inputs) & ~np.eye(neurons, dtype=bool)).astype(np.float32)[None]
    weights = np.linspace(-.125, .125, neurons * neurons, dtype=np.float32).reshape(1, neurons, neurons) * mask
    predictor = np.linspace(-.02, .02, inputs * c.H, dtype=np.float32).reshape(1, inputs, c.H)
    patterns = np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                           0x00800000, 0x80800000, 0x3d800000, 0xbd800000], np.uint32)
    weights.view(np.uint32)[0, inputs, :len(patterns)] = patterns
    state = dict(state, W=jnp.asarray(weights), mask=jnp.asarray(mask), Wp=jnp.asarray(predictor),
        x=jnp.asarray(np.linspace(-.6, .6, neurons, dtype=np.float32)[None]),
        energy=jnp.asarray([.808], jnp.float32), denergy=jnp.asarray([np.float32(.808) - np.float32(.811)]),
        sig=jnp.asarray([.25], jnp.float32), t=jnp.asarray(64, jnp.int32),
        counts=jnp.asarray([[[16., 16.], [0., 0.]]], jnp.float32))
    return flat, state, keys


def historical_modules():
    """Bind old retention to its own old kernel, from source only at a full pin."""
    root = Path(__file__).resolve().parents[1]

    def load(relative, name):
        source = subprocess.check_output(["git", "show", HISTORICAL_COMMIT + ":" + relative], cwd=root).decode("utf-8")
        module = types.ModuleType(name)
        module.__file__ = HISTORICAL_COMMIT + ":" + relative
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module

    old_encounter = load("seed/encounter_diagnostic.py", "historical_e127_kernel")
    with mock.patch.object(seed_package, "encounter_diagnostic", old_encounter):
        old_retention = load("seed/retention_diagnostic.py", "historical_e128_retention")
    if old_retention.encounter is not old_encounter:
        raise AssertionError("historical retention did not bind the historical kernel")
    return old_encounter, old_retention


class RetentionRescueDynamicsTests(unittest.TestCase):
    def test_frozen_interval_preserves_every_weight_bit_but_not_predictor_learning(self):
        flat, acquired, keys = synthetic_state()
        original = jax.tree.map(lambda value: np.asarray(value).copy(), acquired)
        states, trace = rescue.frozen_retention_states(flat, acquired, keys[6], 1)
        for state in states:
            same_bytes(self, state["W"], original["W"])
            same_bytes(self, state["mask"], original["mask"])
        for field in acquired:
            same_bytes(self, acquired[field], original[field])
        np.testing.assert_array_equal(trace["W_input_matches_acquired"], np.ones(33, bool))
        np.testing.assert_array_equal(trace["W_output_matches_acquired"], np.ones(33, bool))
        np.testing.assert_array_equal(trace["hebbian_enabled"], False)
        np.testing.assert_array_equal(trace["predictor_enabled"], True)
        self.assertNotEqual(np.asarray(states[-1]["Wp"]).tobytes(), original["Wp"].tobytes())
        same_bytes(self, trace["incoming_denergy"][0], acquired["denergy"])
        same_bytes(self, trace["incoming_denergy"][1:], trace["denergy"][:-1])
        same_bytes(self, trace["incoming_activity"][1:], trace["activity"][:-1])
        same_bytes(self, trace["key_before"][1:], trace["key_after"][:-1])
        same_bytes(self, trace["action"], trace["sampled_action"])
        np.testing.assert_array_equal(trace["forced_action"], -1)
        # Instrumentation must preserve the established frozen-Hebbian/active-
        # predictor calculation, including all decisions and actual feedback.
        one, first = encounter._free_grid("plastic", rescue.CFG, hebbian=False, steps=1)(flat, states[0], 1)
        last, rest = encounter._free_grid("plastic", rescue.CFG, hebbian=False, steps=32)(flat, one, 1)
        for field in last:
            same_bytes(self, states[1][field], one[field])
            same_bytes(self, states[2][field], last[field])
        for field, expected in retention._trace_join(first, rest).items():
            same_bytes(self, trace[field], expected)
        # The standalone transfer restores both normal updates. Its initial
        # neural state remains the delayed clone, not a newborn or a probe.
        grid, position = seed.init_world(keys[3], rescue.CFG)
        readout = retention._readout_evidence(flat, (states[2],), grid, position, keys[4],
            jax.random.split(keys[5], 12), 1, "plastic", rescue.CFG)
        np.testing.assert_array_equal(readout["grid_hebbian_enabled"], True)
        np.testing.assert_array_equal(readout["grid_predictor_enabled"], True)
        same_bytes(self, readout["grid_initial_W_sha256"][0], encounter.array_hash(states[2]["W"]))
        same_bytes(self, readout["grid_initial_Wp_sha256"][0], encounter.array_hash(states[2]["Wp"]))

    def test_weight_observer_detects_a_single_signed_zero_bit_without_changing_dynamics(self):
        flat, initial, keys = synthetic_state()
        initial = dict(initial, key=keys[6])
        reference = np.asarray(initial["W"]).copy()
        altered = reference.copy()
        # A false reference differing only in zero's sign must fail. Numeric
        # float equality would miss this deliberately constructed corruption.
        reference.view(np.uint32)[0, 0, 0] = 0
        initial = dict(initial, W=jnp.asarray(reference))
        altered[:] = reference
        altered.view(np.uint32)[0, 0, 0] = np.uint32(0x80000000)
        self.assertTrue(np.array_equal(reference, altered))
        observed = encounter._free_grid("plastic", rescue.CFG, hebbian=False, steps=1, verify_weight_identity=True)
        good_state, good = observed(flat, initial, 0, jnp.asarray(reference))
        bad_state, bad = observed(flat, initial, 0, jnp.asarray(altered))
        self.assertTrue(bool(np.asarray(good["W_input_matches_acquired"])[0]))
        self.assertTrue(bool(np.asarray(good["W_output_matches_acquired"])[0]))
        self.assertFalse(bool(np.asarray(bad["W_input_matches_acquired"])[0]))
        self.assertFalse(bool(np.asarray(bad["W_output_matches_acquired"])[0]))
        for field in good_state:
            same_bytes(self, good_state[field], bad_state[field])
        for field in good:
            if not field.startswith("W_"):
                same_bytes(self, good[field], bad[field])

    def test_historical_defaults_and_extracted_e128_family_remain_bit_identical(self):
        old_encounter, old_retention = historical_modules()
        flat, state, keys = synthetic_state()
        # Explicit nonzero recurrent/predictor/body carry, not a trivial newborn.
        for controller in ("plastic", "direct"):
            current, trace = encounter._free_grid(controller, rescue.CFG, steps=3)(flat, state, 1)
            old, old_trace = old_encounter._free_grid(controller, rescue.CFG, steps=3)(flat, state, 1)
            self.assertEqual(set(trace), set(old_trace))
            self.assertNotIn("W_input_matches_acquired", trace)
            for field in current:
                same_bytes(self, current[field], old[field])
            for field in trace:
                same_bytes(self, trace[field], old_trace[field])
        # One synthetic family exercises the entire extracted acquisition,
        # continuation, default readouts, raw schema and compact projection.
        # Only source is read from the historical pin; all inputs are new here.
        plastic = np.asarray(flat)
        recurrent = np.asarray(direct.init_genome(rescue.CFG, keys[0]).at[0].set(7.25).at[1].set(-6.5))
        family_keys = retention.evaluation_keys(SYNTHETIC_MASTER, 0)
        original = old_retention.evaluate_world(plastic, recurrent, family_keys, 0, 8)
        current = retention.evaluate_world(plastic, recurrent, family_keys, 0, 8)
        self.assertEqual(set(current), set(original))
        for field in current:
            with self.subTest(field=field):
                same_bytes(self, current[field], original[field])
        before, after = old_retention.summary_input(original), retention.summary_input(current)
        self.assertEqual(set(before), set(after))
        for field in before:
            same_bytes(self, after[field], before[field])


class RetentionRescueAccountingTests(unittest.TestCase):
    def test_new_assignment_and_two_endpoint_projection_keep_all_rule_replays(self):
        manifest = rescue.assignments_manifest(SYNTHETIC_MASTER, 4)
        self.assertEqual(sorted(manifest["pre_contacts"].tolist()), [8, 8, 16, 16])
        self.assertFalse(np.array_equal(manifest["key"], retention.assignments_manifest(SYNTHETIC_MASTER, 4)["key"]))
        row = dict(base_index=np.asarray(0, np.int32), keys=rescue.evaluation_keys(SYNTHETIC_MASTER, 0),
            pre_contacts=np.asarray(8, np.int32), initial_rules=np.asarray([0, 1], np.int32),
            retention_ticks=np.asarray([0, 33], np.int32), plastic_genome_sha256=np.zeros(32, np.uint8),
            direct_genome_sha256=np.ones(32, np.uint8))
        for arm in rescue.ARMS:
            food, poison = np.zeros((2, 2, 50, 1), bool), np.zeros((2, 2, 50, 1), bool)
            food[0, 1, 1, 0], poison[1, 1, 0, 0] = True, True
            row.update({arm + "_baseline_probe_probabilities": np.full((2, 12, 5), .2, np.float32),
                arm + "_probe_probabilities": np.full((2, 2, 12, 5), .2, np.float32),
                arm + "_grid_food": food, arm + "_grid_poison": poison,
                arm + "_grid_alive_after": np.ones((2, 2, 50, 1), np.float32)})
        with mock.patch.object(rescue, "validate_world"):
            compact = rescue.summary_input(row)
        for arm in rescue.ARMS:
            self.assertEqual(compact[arm + "_local_by_rule"].shape, (2, 2, 5))
            self.assertEqual(compact[arm + "_grid_by_rule"].shape, (2, 2, 7))
            np.testing.assert_array_equal(compact[arm + "_local_by_rule"][..., 4], 0.)
            np.testing.assert_array_equal(compact[arm + "_grid_by_rule"][:, 0, 2], 1.)
            self.assertEqual(compact[arm + "_grid_by_rule"][:, 1, 3].mean(), 0.)


if __name__ == "__main__":
    unittest.main()
