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


def synthetic_frozen_snapshot():
    """Normal, signed-zero and subnormal state; never load an experimental genome."""
    c = diagnostic.CFG
    _, inputs, _, neurons, _ = seed.dims(c)
    genome = seed.biology_prior(c)
    flat = ravel_pytree(genome)[0]
    mask = ((np.arange(neurons)[:, None] >= inputs) & ~np.eye(neurons, dtype=bool)).astype(np.float32)[None]
    W = np.linspace(-.125, .125, neurons * neurons, dtype=np.float32).reshape(1, neurons, neurons) * mask
    Wp = np.linspace(-.05, .05, inputs * c.H, dtype=np.float32).reshape(1, inputs, c.H)
    patterns = np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                           0x00800000, 0x80800000, 0x3d800000, 0xbd800000], np.uint32)
    W.view(np.uint32)[0, inputs, :len(patterns)] = patterns
    Wp.view(np.uint32)[0, 0, :len(patterns)] = patterns
    grid = np.zeros((c.G, c.G), np.int32)
    grid[8, 7], grid[8, 9] = 1, 2
    host = dict(W=W, Wp=Wp, mask=mask, x=np.linspace(-.8, .8, neurons, dtype=np.float32)[None],
        grid=grid, pos=np.asarray([[8, 8]], np.int32), energy=np.ones(1, np.float32), alive=np.ones(1, np.float32),
        sig=np.zeros(1, np.float32), denergy=np.asarray([.125], np.float32), counts=np.zeros((1, 2, 2), np.float32),
        t=np.asarray(0, np.int32), key=np.asarray(jax.random.PRNGKey(0xE127F0)))
    return c, genome, flat, host, jax.tree.map(jnp.asarray, host)


def legacy_plastic_tick(state, genome, observation, c, hebbian=True, predictor=True):
    """Unchanged legacy API as a reference for active learning and prediction."""
    types = seed.neuron_types(c)
    hebb = genome["hebb"][types[:, None], types[None, :]]
    result = jax.vmap(lambda w, x, wp, obs, de, mask: seed.brain_step(w, x, wp, obs, de,
        mask, hebb, jnp.exp(genome["log_eta"]), jnp.exp(genome["log_plr"]), genome["mod"],
        float(hebbian), c, predictor_plastic=float(predictor)))(state["W"], state["x"], state["Wp"],
        observation, state["denergy"], state["mask"])
    return (*result, 3. * result[1][:, -6:-1])


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
    def test_captured_development_vector_uses_float64_validation_sum(self):
        # Original e096... family0 capture, selected solely by maximum mass error.
        p = np.asarray([.49940401315689087, .0012383188586682081, .0012383188586682081,
                        .0012383188586682081, .4968811571598053], np.float32)
        logits = np.asarray([2.9996604919433594, -3., -3., -3., 2.994596004486084], np.float32)
        before = p.tobytes(), logits.tobytes()
        self.assertGreater(abs(float(p.sum(dtype=np.float32)) - 1), 2e-7)
        self.assertLess(abs(float(p.sum(dtype=np.float64)) - 1), 2e-7)
        diagnostic.validate_probabilities(p, logits)
        self.assertEqual(before, (p.tobytes(), logits.tobytes()))
        redistributed = p.copy()
        redistributed[0] += np.float32(.0001)
        redistributed[4] -= np.float32(.0001)
        self.assertLess(abs(redistributed.sum(dtype=np.float64) - 1), diagnostic.PROBABILITY_MASS_BOUND)
        with self.assertRaises(ValueError):
            diagnostic.validate_probabilities(redistributed, logits)

    def test_mass_bound_is_derived_before_data_and_rejects_excess(self):
        unit = 2. ** -24
        gamma = 4 * unit / (1 - 4 * unit)
        expected = (gamma + unit) / (1 - gamma) + 8 * np.finfo(np.float64).eps
        self.assertEqual(diagnostic.PROBABILITY_MASS_BOUND, expected)
        self.assertLess(diagnostic.PROBABILITY_MASS_BOUND, 3e-7)
        # Contract fixture inside the derived bound but outside the old cutoff.
        p = np.full(5, .20000004768371582, np.float32)
        self.assertGreater(abs(p.sum(dtype=np.float64) - 1), 2e-7)
        diagnostic.validate_probabilities(p, np.zeros(5, np.float32))
        with self.assertRaises(ValueError):
            diagnostic.validate_probabilities(np.full(5, .20000008, np.float32), np.zeros(5, np.float32))

    def test_probability_contract_rejects_invalid_or_masked_distributions(self):
        p, logits = np.full(5, .2, np.float32), np.zeros(5, np.float32)
        cases = [(p[:4], logits[:4]), (p.astype(np.float64), logits),
                 (np.asarray([.5, .5, 0, 0, 0], np.float32), logits),
                 (p, np.full(5, 3.1, np.float32))]
        for bad in (np.nan, np.inf, -.1):
            q = p.copy()
            q[0] = bad
            cases.append((q, logits))
        bad_logits = logits.copy()
        bad_logits[0] = np.nan
        cases.append((p, bad_logits))
        for probability, logit in cases:
            with self.subTest(probability=probability, logits=logit), self.assertRaises(ValueError):
                diagnostic.validate_probabilities(probability, logit)

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
    def test_frozen_probe_preserves_ieee_state_and_uses_original_forward(self):
        c, _, flat, host, snapshot = synthetic_frozen_snapshot()
        keys = jax.random.split(jax.random.PRNGKey(0xE127F0), 12)
        for name in ("W", "Wp", "mask"):
            same_bytes(self, snapshot[name], host[name])
        actual = diagnostic._readouts("plastic", c)(flat, snapshot, keys)
        # The existing fixed-weight forward is an independent reference for the
        # recurrent response to the same original W/x and twelve visible scenes.
        fixed = diagnostic._readouts("direct", c)(flat, snapshot, keys)
        for phase in ("initial", "final"):
            for name in ("W", "Wp", "mask"):
                expected = np.broadcast_to(host[name], np.asarray(actual[phase][name]).shape)
                same_bytes(self, actual[phase][name], expected)
        same_bytes(self, actual["final"]["x"], fixed["final"]["x"])
        for field in ("observation", "decision_motor", "logits", "probabilities", "sampled_action", "action_key"):
            same_bytes(self, actual[field], fixed[field])
        self.assertFalse(np.array_equal(np.asarray(actual["final"]["x"]), np.asarray(actual["initial"]["x"])))
        for name in diagnostic.NEURAL:
            same_bytes(self, snapshot[name], host[name])

    def test_dynamic_frozen_hebbian_preserves_w_and_active_predictor(self):
        c, genome, _, host, snapshot = synthetic_frozen_snapshot()
        observation = seed.observe(snapshot["grid"], snapshot["pos"], snapshot["alive"], snapshot["sig"],
                                   snapshot["energy"], snapshot["denergy"], c)
        dynamic = jax.jit(lambda s, g, o, enabled: diagnostic._think(s, g, o, "plastic", c, enabled, True))
        actual = dynamic(snapshot, genome, observation, jnp.asarray(False))
        legacy = jax.jit(lambda s, g, o: legacy_plastic_tick(s, g, o, c, False, True))(snapshot, genome, observation)
        same_bytes(self, actual[0], host["W"])
        same_bytes(self, actual[2], legacy[2])
        same_bytes(self, actual[3], legacy[3])
        self.assertFalse(np.array_equal(np.asarray(actual[2]), host["Wp"]))
        fixed = jax.jit(lambda s, o: diagnostic._think(s, None, o, "direct", c, False, False))(snapshot, observation)
        same_bytes(self, actual[1], fixed[1])
        same_bytes(self, actual[4], fixed[4])
        # The same conditional executable must retain the original active arm.
        active = dynamic(snapshot, genome, observation, jnp.asarray(True))
        original = jax.jit(lambda s, g, o: legacy_plastic_tick(s, g, o, c))(snapshot, genome, observation)
        for actual_field, expected_field in zip(active, original):
            same_bytes(self, actual_field, expected_field)

    def test_already_disabled_chamber_keeps_frozen_w_bits(self):
        c, _, flat, host, snapshot = synthetic_frozen_snapshot()
        script = jnp.asarray([[4, 1, 1, 4, 3], [1, 4, 2, 4, 3]], jnp.int32)
        # This is a synthetic continuation at/after the registered cutoff. The
        # ordinary timing fixture below separately verifies its first live tick.
        snapshot = dict(snapshot, t=jnp.asarray(1, jnp.int32))
        final, trace = diagnostic._chamber("plastic", c, 1)(flat, snapshot, script, 1)
        same_bytes(self, final["W"], host["W"])
        same_bytes(self, final["mask"], host["mask"])
        np.testing.assert_array_equal(trace["hebbian_enabled"], False)
        np.testing.assert_array_equal(trace["predictor_enabled"], True)

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
