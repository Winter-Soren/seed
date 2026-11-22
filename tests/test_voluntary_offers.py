"""E132 synthetic contracts; no saved genome, run, or held-out key is opened."""
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed, encounter_diagnostic as encounter, voluntary_offers as offers


SYNTHETIC_MASTER = 0xE132F0


def same_bytes(test, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.tobytes(), expected.tobytes())


def fixture():
    c = offers.CFG
    keys = offers.evaluation_keys(SYNTHETIC_MASTER, 0)
    flat = ravel_pytree(seed.biology_prior(c))[0]
    state = encounter._birth(flat, "plastic", jnp.asarray(keys[0]), jnp.asarray(keys[2]), c)
    _, inputs, _, neurons, _ = seed.dims(c)
    state = dict(state, x=jnp.asarray(np.linspace(-.7, .7, neurons, dtype=np.float32)[None]),
        Wp=jnp.asarray(np.linspace(-.02, .02, inputs * c.H, dtype=np.float32).reshape(1, inputs, c.H)),
        energy=jnp.asarray([1.31], jnp.float32), denergy=jnp.asarray([-.253], jnp.float32),
        sig=jnp.asarray([.3], jnp.float32), t=jnp.asarray(64, jnp.int32))
    return flat, state, keys


class VoluntaryOfferDynamicsTests(unittest.TestCase):
    def test_alive_tick_is_native_voluntary_transition(self):
        flat, state, _ = fixture()
        c, genome = offers.CFG, encounter._unravel(offers.CFG)(flat)
        state = dict(state, grid=encounter._pair_grid(state["pos"], jnp.asarray([1, 4]), c))
        native = jax.jit(lambda s: encounter._step(s, genome, 1, -1, "plastic", c, regrowth=False))
        guarded = jax.jit(lambda s: offers._tick(s, genome, 1, 64, c, False))
        expected, old = native(state)
        actual, new = guarded(state)
        for name in state:
            same_bytes(self, actual[name], expected[name])
        for name in offers.TRACE_FIELDS:
            same_bytes(self, new[name], old[name])
        same_bytes(self, new["action"], new["sampled_action"])
        same_bytes(self, new["efference"], np.eye(5, dtype=np.float32)[new["action"]])
        same_bytes(self, new["incoming_efference"], state["x"][:, -6:-1])
        same_bytes(self, new["carried_denergy"], actual["denergy"])
        self.assertFalse(bool(new["skipped"]))

    def test_offers_persist_until_consumed_and_never_force_actions(self):
        flat, state, keys = fixture()
        c = offers.CFG
        state = dict(state, energy=jnp.asarray([c.e_max], jnp.float32))
        offsets = offers.geometry(keys[1])["geometry_near_offsets"][:2]
        _, trace = offers._offers(c)(flat, state, jnp.asarray(offsets), 0, 64)
        trace = jax.tree.map(np.asarray, trace)
        self.assertTrue(np.all(trace["alive_after"] == 1))
        same_bytes(self, trace["action"], trace["sampled_action"])
        np.testing.assert_array_equal(trace["external_tick"], np.arange(64, 80))
        np.testing.assert_array_equal(trace["hebbian_enabled"], True)
        np.testing.assert_array_equal(trace["predictor_enabled"], True)
        for tick in range(16):
            if tick % 8 == 0:
                cells = (trace["position_before"][tick, 0] + offsets[tick // 8]) % c.G
                expected = np.zeros((c.G, c.G), np.int32)
                expected[cells[:, 0], cells[:, 1]] = [1, 2]
            same_bytes(self, trace["grid_before"][tick], expected)
            if bool(trace["food"][tick, 0] or trace["poison"][tick, 0]):
                expected[tuple(trace["position"][tick, 0])] = 0
            same_bytes(self, trace["grid"][tick], expected)
        same_bytes(self, trace["incoming_denergy"][1:], trace["carried_denergy"][:-1])

    def test_discarded_flush_processes_feedback_without_mutating_split(self):
        flat, split, keys = fixture()
        before = jax.tree.map(lambda value: np.asarray(value).copy(), split)
        feedback, trace = offers._flush(flat, split, 1, 64, offers.CFG)
        same_bytes(self, trace["observation"][0, :, -2], np.float32(4) * split["denergy"])
        np.testing.assert_array_equal(trace["grid_before"], 0)
        np.testing.assert_array_equal(trace["food"], False)
        np.testing.assert_array_equal(trace["poison"], False)
        self.assertEqual(int(feedback["t"]), 65)
        for name in split:
            same_bytes(self, split[name], before[name])
        offsets = offers.geometry(keys[1])["geometry_near_offsets"][:1]
        _, continued = offers._offers(offers.CFG)(flat, split, jnp.asarray(offsets), 1, 64)
        same_bytes(self, continued["incoming_denergy"][0], split["denergy"])
        same_bytes(self, continued["key_before"][0], split["key"])
        self.assertFalse(encounter._identical(continued["key_before"][0], feedback["key"]))

    def test_death_freezes_all_bits_and_never_starts_a_readout(self):
        flat, state, keys = fixture()
        state = dict(state, alive=jnp.zeros(1), energy=jnp.zeros(1),
            grid=jnp.ones((16, 16), jnp.int32), denergy=jnp.asarray([-.017], jnp.float32))
        for name in ("W", "x", "Wp"):
            bits = np.asarray(state[name]).copy().view(np.uint32).reshape(-1)
            bits[:4] = [0x80000000, 1, 0x80000001, 0]
            state[name] = jnp.asarray(bits.view(np.float32).reshape(state[name].shape))
        saved = jax.tree.map(lambda value: np.asarray(value).copy(), state)
        offsets = offers.geometry(keys[1])["geometry_sham_offsets"][:1]
        end, trace = offers._offers(offers.CFG)(flat, state, jnp.asarray(offsets), 0, 64)
        flushed, flush = offers._flush(flat, end, 0, 128, offers.CFG)
        with mock.patch.object(offers, "_compact_readouts", side_effect=AssertionError("dead readout compiled")):
            probe = offers._local_probe(flat, flushed, jax.random.split(jnp.asarray(keys[5]), 12), offers.CFG)
        with mock.patch.object(encounter, "_fresh_body", side_effect=AssertionError("dead body revived")):
            initial, final, solo, performed = offers._solo(flat, flushed, jnp.zeros((16, 16), jnp.int32),
                jnp.asarray([[0, 0]], jnp.int32), jnp.asarray(keys[4]), 0, offers.CFG)
        for after in (end, flushed, initial, final):
            for name in state:
                same_bytes(self, after[name], saved[name])
        for record in (trace, flush, solo):
            np.testing.assert_array_equal(record["skipped"], True)
            np.testing.assert_array_equal(record["action"], -1)
            np.testing.assert_array_equal(record["denergy"], 0)
            np.testing.assert_array_equal(record["food"], False)
            np.testing.assert_array_equal(record["poison"], False)
            np.testing.assert_array_equal(record["t"], 64)
            same_bytes(self, record["carried_denergy"], np.full((len(record["skipped"]), 1), -.017, np.float32))
        self.assertFalse(bool(performed))
        self.assertFalse(bool(probe["performed"]))
        np.testing.assert_array_equal(probe["probabilities"], 0)

    def test_last_feedback_tick_can_kill_without_revival(self):
        flat, state, keys = fixture()
        state = dict(state, energy=jnp.asarray([.001], jnp.float32), t=jnp.asarray(128, jnp.int32))
        self.assertEqual(float(state["alive"][0]), 1.)
        after, trace = offers._flush(flat, state, 1, 128, offers.CFG)
        self.assertEqual(float(after["alive"][0]), 0.)
        self.assertEqual(int(after["t"]), 129)
        self.assertEqual(float(trace["alive_before"][0, 0]), 1.)
        with mock.patch.object(offers, "_compact_readouts", side_effect=AssertionError("dead readout compiled")):
            probe = offers._local_probe(flat, after, jax.random.split(jnp.asarray(keys[5]), 12), offers.CFG)
        self.assertFalse(bool(probe["performed"]))

    def test_compact_local_readout_keeps_native_responses_and_frozen_proof(self):
        flat, state, keys = fixture()
        probe_keys = jax.random.split(jnp.asarray(keys[5]), 12)
        expected = encounter._readouts("plastic", offers.CFG)(flat, state, probe_keys)
        actual = offers._local_probe(flat, state, probe_keys, offers.CFG)
        for field in offers.PROBE_FIELDS:
            value = expected[field]
            if field in ("decision_motor", "logits", "probabilities", "sampled_action"):
                value = value[:, 0]
            same_bytes(self, actual[field], value)
        self.assertTrue(bool(actual["performed"]))
        np.testing.assert_array_equal(actual["initial_neural_matches"], True)
        np.testing.assert_array_equal(actual["final_frozen_matches"], True)
        for name in encounter.NEURAL:
            same_bytes(self, actual["source_" + name + "_sha256"], encounter.array_hash(state[name]))
        self.assertFalse(any(np.asarray(value).shape[-2:] == (157, 157) for value in actual.values()))
        encounter.validate_probabilities(actual["probabilities"], actual["logits"])


class VoluntaryOfferAccountingTests(unittest.TestCase):
    def test_independent_assignment_and_rule_blind_geometry(self):
        manifest = offers.assignments_manifest(SYNTHETIC_MASTER, 4)
        self.assertEqual(manifest["family_keys"].shape, (4, 6, 2))
        self.assertEqual(len({key.tobytes() for family in manifest["family_keys"] for key in family}), 24)
        np.testing.assert_array_equal(manifest["initial_rules"], [0, 1])
        for index, keys in enumerate(manifest["family_keys"]):
            same_bytes(self, keys, offers.evaluation_keys(SYNTHETIC_MASTER, index))
            schedule = offers.geometry(keys[1])
            for name, value in schedule.items():
                same_bytes(self, value, offers.geometry(keys[1])[name])
            for near, sham, block_key in zip(schedule["geometry_near_offsets"], schedule["geometry_sham_offsets"],
                                           schedule["geometry_block_keys"]):
                near_key, sham_key = jax.random.split(jnp.asarray(block_key))
                near_indices = np.asarray(jax.random.permutation(near_key, 4))[:2] + 1
                cells = np.asarray(jax.random.permutation(sham_key, 255))[:2] + 1
                np.testing.assert_array_equal(near, np.asarray(seed.MOVES)[near_indices] % 16)
                np.testing.assert_array_equal(sham, np.stack((cells // 16, cells % 16), axis=1))
                self.assertFalse(np.array_equal(near[0], near[1]))
                self.assertFalse(np.array_equal(sham[0], sham[1]))
                self.assertTrue(np.all(np.any(sham != 0, axis=1)))
        for count in (0, 3, 128):
            with self.assertRaises(ValueError):
                offers.assignments_manifest(SYNTHETIC_MASTER, count)
        for index in (True, -1, offers.ASSIGNMENT_TAG):
            with self.assertRaises(ValueError):
                offers.evaluation_keys(SYNTHETIC_MASTER, index)

    def test_projection_retains_fatal_contacts_actual_energy_and_paired_rules(self):
        row = dict(base_index=np.asarray(0, np.int32), keys=offers.evaluation_keys(SYNTHETIC_MASTER, 0),
            initial_rules=np.asarray([0, 1], np.int32), genome_sha256=np.zeros(32, np.uint8))
        old = np.full((2, 12, 5), .1, np.float32)
        for rule in (0, 1):
            old[rule, np.arange(12), encounter.PLACEMENTS[:, rule]] = .6
        row["common_probe_probabilities"] = old
        for arm in offers.ARMS:
            rule = np.asarray([0, 1] if arm.endswith("stationary") else [1, 0], np.int32)
            local = np.full((2, 12, 5), .1, np.float32)
            local[0, np.arange(12), encounter.PLACEMENTS[:, rule[0]]] = .6
            local[1] = 0  # Death on the feedback tick: no readout or revival.
            food, poison = np.zeros((2, 50, 1), bool), np.zeros((2, 50, 1), bool)
            poison[0, 3, 0], food[0, 4, 0] = True, True
            ef, ep = np.zeros((2, 64, 1), bool), np.zeros((2, 64, 1), bool)
            item = np.zeros((2, 64, 1), np.int32)
            ef[0, 1, 0], ep[0, 2, 0], ep[1, 63, 0] = True, True, True
            item[0, 1, 0], item[0, 2, 0], item[1, 63, 0] = 1, 2, 2
            de = np.zeros((2, 64, 1), np.float32)
            de[0, 1, 0], de[0, 2, 0], de[1, 63, 0] = .05, -.253, -.1
            values = dict(rule=rule, probe_probabilities=local, solo_food=food, solo_poison=poison,
                exposure_food=ef, exposure_poison=ep, exposure_item=item, exposure_denergy=de,
                exposure_end_alive=np.asarray([[1.], [1.]], np.float32),
                feedback_alive=np.asarray([[1.], [0.]], np.float32), solo_final_alive=np.asarray([[1.], [0.]], np.float32))
            row.update({arm + "_" + name: value for name, value in values.items()})
        # This tests only endpoint arithmetic, not a fabricated physical archive.
        with mock.patch.object(offers, "validate_world"):
            compact = offers.summary_input(row)
        np.testing.assert_allclose(compact["pre_old_S_by_rule"], .5, atol=3e-8, rtol=0)
        for arm in offers.ARMS:
            np.testing.assert_allclose(compact[arm + "_local_S_by_rule"], [.5, 0], atol=3e-8, rtol=0)
            np.testing.assert_array_equal(compact[arm + "_both_colors_by_rule"], [1, 0])
            np.testing.assert_array_equal(compact[arm + "_intake_B_by_rule"], [1, 1])
            np.testing.assert_array_equal(compact[arm + "_solo_Y_by_rule"], [-1, 0])
            np.testing.assert_array_equal(compact[arm + "_no_contact_by_rule"], [0, 1])
            np.testing.assert_array_equal(compact[arm + "_exposure_survival_by_rule"], [1, 1])
            np.testing.assert_array_equal(compact[arm + "_readout_survival_by_rule"], [1, 0])
            np.testing.assert_allclose(compact[arm + "_realized_energy_by_rule"], [-.203, -.1], atol=3e-8, rtol=0)


class VoluntaryOfferFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flat = np.asarray(ravel_pytree(seed.biology_prior(offers.CFG))[0])
        cls.row = offers.evaluate_world(flat, offers.evaluation_keys(SYNTHETIC_MASTER, 1), 1)

    def test_full_family_clones_unflushed_common_state_and_never_censors_death(self):
        offers.validate_world(self.row)
        compact = offers.summary_input(self.row)
        for arm in offers.ARMS:
            np.testing.assert_array_equal(self.row[arm + "_exposure_external_tick"], np.broadcast_to(np.arange(64, 128), (2, 64)))
            np.testing.assert_array_equal(self.row[arm + "_flush_external_tick"], [[128], [128]])
            same_bytes(self, self.row[arm + "_exposure_incoming_denergy"][:, 0], self.row["common_split_denergy"])
            same_bytes(self, self.row[arm + "_exposure_key_before"][:, 0], self.row["common_split_key"])
            self.assertEqual(compact[arm + "_solo_Y_by_rule"].shape, (2,))
            self.assertEqual(self.row[arm + "_solo_action"].shape, (2, 50, 1))

    def test_validation_rejects_lost_clone_and_frozen_probe_proof(self):
        for field in ("near_reversal_split_W_sha256", "near_reversal_probe_initial_neural_matches",
                      "near_reversal_probe_final_frozen_matches"):
            damaged = dict(self.row)
            damaged[field] = self.row[field].copy()
            damaged[field].flat[0] = not damaged[field].flat[0] if damaged[field].dtype == bool else damaged[field].flat[0] ^ 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                offers.validate_world(damaged)


if __name__ == "__main__":
    unittest.main()
