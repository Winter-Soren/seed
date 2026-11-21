"""Tiny development fixtures for opt-in ecology; no registered evaluation worlds."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import direct, seed
from seed.ecology import EcologyV1, apply_costs, make_shelter
from scripts import eval_v3, run_experiment


class EcologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = seed.Cfg(G=4, M=2, T=8, R=0, H=3, n_items=0, decay=0.)
        cls.wkey, cls.dkey = jax.random.PRNGKey(41), jax.random.PRNGKey(42)
        genome = seed.init_genome(cls.c, cls.dkey)
        cls.genome = dict(genome, conn=jnp.full_like(genome["conn"], -100.))

    def lifetime(self, ecology=None, **kwargs):
        return seed.rollout(self.genome, self.wkey, self.dkey, 0, self.c,
                            plastic=False, predictor_plastic=False, record=True,
                            diagnostics=True, ecology=ecology, **kwargs)

    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)

    def test_cost_validation_and_action_time_alive_boundaries(self):
        self.assertEqual(EcologyV1().version, "ecology-v1")
        for field in ("movement_energy_cost", "seasonal_energy_cost"):
            for value in (-1., np.nan, np.inf, -np.inf, True, "0.1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    EcologyV1(**{field: value})
        energy = np.ones(3)
        action, alive = np.array([0, 1, 1]), np.array([1., 1., 0.])
        pressure = EcologyV1(.125, .25)
        np.testing.assert_array_equal(apply_costs(energy, action, alive, 3, 4, pressure), [1., .875, 1.])
        np.testing.assert_array_equal(apply_costs(energy, action, alive, 4, 4, pressure), [.75, .625, 1.])
        np.testing.assert_array_equal(apply_costs(energy, action, alive, 4, 4, pressure,
                                                exposed=np.array([False, True, False])), [1., .625, 1.])
        self.assertIs(apply_costs(energy, action, alive, 4, 4, None), energy)
        self.assertIs(apply_costs(energy, action, alive, 4, 4, EcologyV1()), energy)
        with self.assertRaises(ValueError):
            apply_costs(energy, action, alive, 4, 4, (.1, .2))

    def test_default_matches_pinned_original_trajectories(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.run(["git", "show", "f9135ba:seed/seed.py"], cwd=root,
                                capture_output=True, check=True, timeout=10).stdout
        legacy = types.ModuleType("legacy_ecology")
        exec(compile(source, "legacy_ecology.py", "exec"), legacy.__dict__)
        self.assertEqual(seed.Cfg._fields, legacy.Cfg._fields)
        c = self.c._replace(R=1, n_items=3, decay=.003)
        genome = seed.init_genome(c, self.dkey)
        for plastic in (False, True):
            expected = legacy.rollout(genome, self.wkey, self.dkey, 0, legacy.Cfg(**c._asdict()),
                                      plastic=plastic, record=True, diagnostics=True)
            for kwargs in ({}, {"ecology": None}):
                actual = seed.rollout(genome, self.wkey, self.dkey, 0, c,
                                      plastic=plastic, record=True, diagnostics=True, **kwargs)
                self.assert_tree_equal(actual, expected)

    def test_ecology_layout_and_shelter_are_visible_stable_and_seeded(self):
        c = self.c._replace(R=1)
        grid, pos = seed.init_world(self.wkey, c)
        shelter = make_shelter(self.wkey, c)
        self.assertEqual(int(shelter.sum()), 1)
        maps = [np.asarray(make_shelter(jax.random.PRNGKey(key), c)) for key in range(41, 46)]
        self.assertTrue(any(not np.array_equal(maps[0], other) for other in maps[1:]))
        ordinary = seed.observe(grid, pos, jnp.ones(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2), c)
        ecology_obs = []
        for pressure in (EcologyV1(), EcologyV1(.1, .2)):
            self.assertEqual(seed.dims(c, ecology=pressure)[1], seed.dims(c)[1] + 10)
            obs = seed.observe(grid, pos, jnp.ones(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2), c,
                               ecology=pressure, shelter=shelter, step=4)
            np.testing.assert_array_equal(obs[:, :ordinary.shape[1]], ordinary)
            np.testing.assert_array_equal(obs[:, -1], np.ones(2))
            for agent in range(2):
                row, column = np.asarray(pos[agent])
                patch_rows = (row + np.arange(-1, 2)) % c.G
                patch_cols = (column + np.arange(-1, 2)) % c.G
                expected = np.asarray(shelter)[patch_rows[:, None], patch_cols[None, :]].reshape(-1)
                np.testing.assert_array_equal(obs[agent, ordinary.shape[1]:-1], expected)
            ecology_obs.append(obs)
        np.testing.assert_array_equal(*ecology_obs)
        np.testing.assert_array_equal(self.lifetime(EcologyV1())["traj"]["energy"], np.ones((8, 2)))
        legacy_weights = jnp.zeros(direct.genome_size(c))
        with self.assertRaises(ValueError):
            direct.decode(legacy_weights, c, ecology=EcologyV1())

    def test_movement_cost_follows_nonstay_actions_in_actual_rollout(self):
        result = self.lifetime(EcologyV1(movement_energy_cost=.0625))
        _, initial_position = seed.init_world(self.wkey, self.c)
        positions = np.concatenate([np.asarray(initial_position)[None], np.asarray(result["traj"]["pos"])])
        moved = np.any(positions[1:] != positions[:-1], axis=-1)
        self.assertTrue(moved.any())
        expected = 1. - .0625 * np.cumsum(moved, axis=0)
        np.testing.assert_array_equal(result["traj"]["energy"], expected)
        np.testing.assert_array_equal(result["traj"]["pos"], self.lifetime(EcologyV1())["traj"]["pos"])

    def test_visible_shelter_makes_moving_better_than_staying_in_cold(self):
        c = self.c._replace(M=1, T=2, R=1)
        pressure = EcologyV1(movement_energy_cost=.125, seasonal_energy_cost=.25)
        grid = jnp.zeros((4, 4), dtype=jnp.int32)
        pos = jnp.array([[2, 2]])
        shelter = jnp.zeros((4, 4), dtype=bool).at[1, 2].set(True)
        obs = seed.observe(grid, pos, jnp.ones(1), jnp.zeros(1), jnp.ones(1), jnp.zeros(1), c,
                           ecology=pressure, shelter=shelter, step=1)
        self.assertEqual(float(obs[0, seed.dims(c)[1] + 1]), 1.)
        self.assertEqual(float(obs[0, -1]), 1.)
        outcomes = []
        for action in (0, 1):
            with (jax.disable_jit(), patch.object(seed, "init_world", return_value=(grid, pos)),
                  patch.object(seed, "make_shelter", return_value=shelter),
                  patch.object(jax.random, "categorical", side_effect=[jnp.array([0]), jnp.array([action])])):
                outcomes.append(seed.rollout(self.genome, self.wkey, self.dkey, 0, c,
                                              plastic=False, predictor_plastic=False, record=True,
                                              diagnostics=True, ecology=pressure))
        self.assertEqual(float(outcomes[0]["traj"]["energy"][-1, 0]), .75)
        self.assertEqual(float(outcomes[1]["traj"]["energy"][-1, 0]), .875)
        self.assertFalse(bool(outcomes[1]["ecology"]["initial_sheltered"][0]))
        self.assertFalse(bool(outcomes[0]["ecology"]["sheltered"][-1, 0]))
        self.assertTrue(bool(outcomes[1]["ecology"]["sheltered"][-1, 0]))

    def test_seasonal_cost_starts_at_second_half_and_can_cause_death(self):
        result = self.lifetime(EcologyV1(seasonal_energy_cost=.125))
        exposed = ~np.asarray(result["ecology"]["sheltered"])
        cold = (np.arange(8) >= 4)[:, None]
        expected = 1. - .125 * np.cumsum(exposed & cold, axis=0)
        np.testing.assert_array_equal(result["traj"]["energy"], expected)
        death = self.lifetime(EcologyV1(seasonal_energy_cost=.5))
        alive = np.asarray(death["events"]["alive_after"])
        self.assertTrue(np.any(alive[-1] == 0))
        np.testing.assert_array_equal(np.asarray(death["traj"]["energy"])[alive == 0], 0.)
        np.testing.assert_array_equal(death["ecology"]["energy_lost"][:4], np.zeros((4, 2)))

    def test_reference_policies_and_direct_controller_pay_same_costs(self):
        pressure = EcologyV1(movement_energy_cost=.25, seasonal_energy_cost=.5)
        expected = self.lifetime(pressure)
        for policy in ("random_walk", "greedy", "fixed_A", "fixed_B", "mixture"):
            reference = eval_v3.reference_rollout(self.wkey, 0, self.c, policy=policy, ecology=pressure)
            self.assert_tree_equal(reference["events"], expected["events"])
            self.assert_tree_equal(reference["ecology"], expected["ecology"])
            np.testing.assert_array_equal(reference["fitness"], expected["fitness"])
        flat = jnp.zeros(direct.genome_size(self.c, ecology=pressure))
        evaluate, _ = direct.make_evaluator(self.c, 2, ecology=pressure)
        world_keys = jnp.stack([self.wkey, self.wkey])
        development_keys = jnp.stack([self.dkey, self.dkey])[None]
        actual = evaluate(flat[None], world_keys, development_keys)
        np.testing.assert_array_equal(actual["survived"], np.full((1, 2), expected["survived"]))

    def test_saved_flat_genome_connects_to_existing_es_without_new_encoding(self):
        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            path = Path(temp) / "genome.npy"
            flat = ravel_pytree(self.genome)[0]
            np.save(path, np.asarray(flat))
            state = seed.es_init(self.c, mean=np.load(path, allow_pickle=False), seed=41)
        pressure = EcologyV1(seasonal_energy_cost=.5)
        evaluate, decode = seed.make_evaluator(self.c, 2, ecology=pressure)
        self.assert_tree_equal(decode(state["mu"]), self.genome)
        world_keys = jnp.stack([self.wkey, self.wkey])
        development_keys = jnp.stack([self.dkey, self.dkey])[None]
        np.testing.assert_array_equal(evaluate(flat[None], world_keys, development_keys)["survived"],
                                      np.full((1, 2), self.lifetime(pressure)["survived"]))
        updated, row = seed.es_step(state, evaluate, 2, 2, .1, .02)
        self.assertEqual((updated["t"], row["gen"]), (1, 0))
        self.assertTrue(np.isfinite(np.asarray(updated["mu"])).all())

    def test_registered_protocols_cannot_accept_custom_ecology(self):
        flat = ravel_pytree(self.genome)[0]
        with self.assertRaises(TypeError):
            seed.test_genome(flat, self.c, ecology=EcologyV1(.1, .1))
        with self.assertRaises(TypeError):
            eval_v3.evaluate_conditions(flat, self.c, ecology=EcologyV1(.1, .1))
        for flag in ("--movement-energy-cost", "--seasonal-energy-cost"):
            with (patch.object(run_experiment.sys, "argv", ["runner", flag, "0.1"]),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
                run_experiment.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
