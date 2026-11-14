"""Tiny functional fixtures for rule inversion and auditable per-agent accounting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from seed import seed
from scripts import reversal_eval as panel, run_experiment as runner, summarize


class ReversalEngineTests(unittest.TestCase):
    def test_weight_only_learning_has_exact_unannounced_preboundary_prefix(self):
        c = seed.Cfg(G=4, M=1, T=22, R=0, H=3, n_items=8)
        g = seed.biology_prior(c)
        args = (g, jax.random.PRNGKey(8210), jax.random.PRNGKey(23), 0, c)
        control = seed.rollout(*args, plastic=True, predictor_plastic=True, structural=False,
                               record=True, diagnostics=True)
        flipped = seed.rollout(*args, plastic=True, predictor_plastic=True, structural=False,
                               flip_step=11, record=True, diagnostics=True)
        self.assertGreater(float(control["counts"][0].sum()), 0.)
        for key in control["traj"]:
            np.testing.assert_array_equal(control["traj"][key][:11], flipped["traj"][key][:11], err_msg=key)
        np.testing.assert_array_equal(control["traj"]["pos"][11], flipped["traj"]["pos"][11])
        np.testing.assert_array_equal(control["counts"][0], flipped["counts"][0])

    def test_fixed_item_inverts_exactly_without_announcing_flip(self):
        c = seed.Cfg(G=1, M=1, T=6, R=0, H=3, n_items=0, e0=10., e_max=20., decay=0.)
        g = seed.init_genome(c, jax.random.PRNGKey(17))
        fixed = (jnp.ones((1, 1), jnp.int32), jnp.zeros((1, 2), jnp.int32))
        with patch.object(seed, "init_world", return_value=fixed):
            for rule in (0, 1):
                args = (g, jax.random.PRNGKey(8210), jax.random.PRNGKey(42), rule, c)
                control = seed.rollout(*args, plastic=False, record=True, predictor_plastic=False, diagnostics=True)
                flipped = seed.rollout(*args, plastic=False, record=True, predictor_plastic=False,
                                       flip_step=3, diagnostics=True)
                expected = [[3, 0], [0, 3]] if rule == 0 else [[0, 3], [3, 0]]
                np.testing.assert_array_equal(flipped["counts"], expected)
                for key in control["traj"]:
                    np.testing.assert_array_equal(control["traj"][key][:3], flipped["traj"][key][:3], err_msg=key)
                self.assertNotEqual(float(control["traj"]["energy"][3, 0]), float(flipped["traj"]["energy"][3, 0]))
                np.testing.assert_array_equal(flipped["events"]["food"], flipped["traj"]["food"])

    def test_diagnostics_preserve_default_rollout_and_predictor_switch(self):
        c = seed.Cfg(G=3, M=1, T=8, R=0, H=3, n_items=2)
        g = seed.init_genome(c, jax.random.PRNGKey(5))
        g["mod"] = jnp.array([1., .3, .1])
        args = (g, jax.random.PRNGKey(8210), jax.random.PRNGKey(23), 0, c)
        for plastic, predictor in ((True, True), (True, False), (False, False)):
            normal = seed.rollout(*args, plastic=plastic, predictor_plastic=predictor, record=True)
            diagnostic = seed.rollout(*args, plastic=plastic, predictor_plastic=predictor, record=True, diagnostics=True)
            self.assertNotIn("events", normal)
            for key in ("fitness", "counts", "survived", "surprise_first", "surprise_last"):
                np.testing.assert_array_equal(normal[key], diagnostic[key], err_msg=key)
            for key in normal["traj"]:
                np.testing.assert_array_equal(normal["traj"][key], diagnostic["traj"][key], err_msg=key)

    def test_enabled_reversal_replays_exactly(self):
        c = seed.Cfg(G=3, M=1, T=8, R=0, H=3, n_items=2)
        g = seed.init_genome(c, jax.random.PRNGKey(5))
        args = (g, jax.random.PRNGKey(8210), jax.random.PRNGKey(23), 1, c)
        a = seed.rollout(*args, flip_step=4, diagnostics=True, record=True, predictor_plastic=True)
        b = seed.rollout(*args, flip_step=4, diagnostics=True, record=True, predictor_plastic=True)
        for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(left, right)

    def test_flip_boundary_must_be_an_interior_integer(self):
        c = seed.Cfg(G=3, M=1, T=8, R=0, H=3, n_items=2)
        g = seed.init_genome(c, jax.random.PRNGKey(5))
        for boundary in (-1, 0, 8, 1.5, True):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "flip_step"):
                seed.rollout(g, jax.random.PRNGKey(8210), jax.random.PRNGKey(23), 0, c, flip_step=boundary)

    def test_structural_and_predictor_apis_compose_with_reversal(self):
        c = seed.Cfg(G=4, M=1, T=22, R=0, H=3, n_items=2)
        g = seed.init_genome(c, jax.random.PRNGKey(7))
        g = dict(g, conn=jnp.full_like(g["conn"], -100.))
        args = (g, jax.random.PRNGKey(3), jax.random.PRNGKey(5), 0, c)
        control = seed.rollout(*args, structural=True, predictor_plastic=True, record=True, diagnostics=True)
        flipped = seed.rollout(*args, structural=True, predictor_plastic=True, record=True,
                               flip_step=11, diagnostics=True)
        for key in control["traj"]:
            np.testing.assert_array_equal(control["traj"][key][:11], flipped["traj"][key][:11], err_msg=key)
        self.assertGreater(float(flipped["traj"]["structural_added"][10]), 0.)
        for key in ("cap_violations", "illegal_edges", "invalid_weights", "invalid_proposals"):
            self.assertEqual(float(flipped["structure"][key]), 0.)
        frozen = seed.rollout(*args, plastic=False, predictor_plastic=False, structural=True,
                              flip_step=11, diagnostics=True)
        static = seed.rollout(*args, plastic=False, predictor_plastic=False, flip_step=11, diagnostics=True)
        for key in ("fitness", "counts", "survived"):
            np.testing.assert_array_equal(frozen[key], static[key], err_msg=key)
        for key in static["events"]:
            np.testing.assert_array_equal(frozen["events"][key], static["events"][key], err_msg=key)
        self.assertEqual((float(frozen["structure"]["added"]), float(frozen["structure"]["removed"])), (0., 0.))


class RecoveryAccountingTests(unittest.TestCase):
    def events(self, steps=100, agents=6):
        shape = (1, steps, agents)
        return dict(food=np.zeros(shape, bool), poison=np.zeros(shape, bool),
                    alive_before=np.ones(shape, bool), alive_after=np.ones(shape, bool))

    def test_per_agent_windows_and_all_censor_reasons_are_preserved(self):
        events = self.events()
        events["food"][0, 50:55, 0] = True
        events["poison"][0, 50:55, 1] = True
        events["alive_after"][0, 39:, 2] = False
        events["alive_before"][0, 40:, 2] = False
        events["food"][0, 60, 4] = True
        events["food"][0, 51:56, 5] = True
        events["alive_after"][0, 60:, 5] = False
        events["alive_before"][0, 61:, 5] = False
        result = panel.summarize_events(events, 50)
        self.assertEqual((result["recovered_agents"], result["agents_at_risk"], result["total_agents"]), (1, 5, 6))
        self.assertEqual(result["window_hit_fraction_all"], 1/6)
        self.assertEqual(result["window_hit_fraction_at_risk"], 1/5)
        self.assertEqual(result["agents"][0]["window_hit_latency"], 25)
        self.assertEqual([row["censor_reason"] for row in result["agents"]],
                         [None, "criterion_not_met", "dead_at_boundary", "no_post_encounters",
                          "insufficient_window_encounters", "died_before_window_hit"])
        self.assertIsNone(result["pre"]["accuracy"])
        self.assertIsNone(result["post_windows"][1]["accuracy"])
        self.assertEqual((result["eaten"], result["fitness"], result["post_fitness"]), (16, 1., 1.))
        self.assertLess(result["post_windows"][0]["accuracy"], panel.ACCURACY)
        json.dumps(result, allow_nan=False)

    def test_partial_last_window_retains_counts_but_never_counts_as_recovery(self):
        events = self.events(steps=60, agents=1)
        events["food"][0, 55:60, 0] = True
        result = panel.summarize_events(events, 30)
        self.assertIsNone(result["agents"][0]["window_hit_latency"])
        self.assertEqual(result["agents"][0]["censor_reason"], "insufficient_window_encounters")
        self.assertFalse(result["post_windows"][-1]["complete"])
        self.assertEqual(result["post_windows"][-1]["food"], 5)
        self.assertEqual(result["post_windows"][-1]["stop"], 60)

    def test_exact_seventy_percent_and_minimum_encounter_boundaries(self):
        events = self.events(agents=3)
        events["food"][0, 50:57, 0] = True
        events["poison"][0, 57:60, 0] = True
        events["food"][0, 50:59, 1] = True
        events["poison"][0, 59:63, 1] = True
        events["food"][0, 50:54, 2] = True
        result = panel.summarize_events(events, 50)
        self.assertEqual(result["agents"][0]["post"]["accuracy"], 0.7)
        self.assertEqual(result["agents"][0]["window_hit_latency"], 25)
        self.assertLess(result["agents"][1]["post"]["accuracy"], 0.7)
        self.assertIsNone(result["agents"][1]["window_hit_latency"])
        self.assertEqual(result["agents"][1]["censor_reason"], "criterion_not_met")
        self.assertIsNone(result["agents"][2]["window_hit_latency"])
        self.assertEqual(result["agents"][2]["censor_reason"], "insufficient_window_encounters")

    def test_short_lifetime_has_explicit_no_complete_window_censor(self):
        events = self.events(steps=12, agents=1)
        events["food"][0, 6:12, 0] = True
        result = panel.summarize_events(events, 6)
        self.assertEqual(result["agents"][0]["censor_reason"], "no_complete_window")
        self.assertEqual(result["recovered_agents"], 0)

    def test_zero_at_risk_is_undefined_and_never_fake_recovery(self):
        events = self.events()
        events["alive_after"][:, 19:] = False
        events["alive_before"][:, 20:] = False
        result = panel.summarize_events(events, 50)
        self.assertEqual(result["agents_at_risk"], 0)
        self.assertIsNone(result["window_hit_fraction_at_risk"])
        self.assertIsNone(result["median_window_hit_latency"])
        self.assertIsNone(result["post"]["accuracy"])
        self.assertEqual(result["censored"], {"dead_at_boundary": 6})

    def test_impossible_dead_agent_encounters_are_rejected(self):
        events = self.events()
        events["alive_after"][:, 19:] = False
        events["alive_before"][:, 20:] = False
        events["food"][0, 50, 0] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            panel.summarize_events(events, 50)


class ReversalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / ".tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.npy"
        np.save(self.source, np.zeros(155, np.float32))
        self.base = ["--eval-protocol", "reversal-v1", "--condition-split", "development",
                     "--start", f"file:{self.source}", "--generations", "0", "--eval-worlds", "1",
                     "--cfg", "G=3", "--cfg", "M=1", "--cfg", "T=12", "--cfg", "R=0", "--cfg", "H=3", "--cfg", "n_items=1"]

    def invoke(self, *extra, bad=False):
        def evaluate(*args):
            shape = (1, 12, 1)
            events = dict(food=np.zeros(shape, bool), poison=np.zeros(shape, bool),
                          alive_before=np.ones(shape, bool), alive_after=np.ones(shape, bool))
            result = {}
            for (name, flip, plastic), fitness in zip(panel.CONDITIONS, (3., 1., -2., -3.)):
                metrics = panel.summarize_events(events, 6)
                metrics["post_fitness"] = float("nan") if bad else fitness
                result[name] = dict(flip_step=6 if flip else None, hebbian=plastic, predictor=plastic, metrics=metrics)
            return result
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="fixture", dirty=False)),
              patch.object(panel, "evaluate_conditions", side_effect=evaluate) as call,
              patch.object(runner, "make_evaluator") as training,
              patch.object(runner.sys, "argv", ["runner", *self.base, *extra]),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()
            training.assert_not_called()
        path = next((self.root / "runs").iterdir())
        return path, json.loads((path/"config.json").read_text()), json.loads((path/"summary.json").read_text()), call

    def test_dispatch_provenance_signed_metrics_and_separate_summary_table(self):
        path, config, summary, call = self.invoke()
        self.assertEqual(summary["status"], "finished")
        self.assertEqual((summary["eval_seed"], summary["boundary"]), (8210, 6))
        self.assertEqual(summary["post_fitness_differences"],
                         dict(noflip_plasticity=2., flip_plasticity=1., reversal_learning=-5., reversal_frozen=-4.))
        source = self.source.read_bytes()
        self.assertEqual(config["source_genome_sha256"], hashlib.sha256(source).hexdigest())
        for name in ("genome_start.npy", "genome_final.npy"):
            self.assertEqual((path/name).read_bytes(), source)
        self.assertEqual(call.call_args.args[-2:], (1, "development"))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root/"results/RESULTS.md").read_text()
        self.assertIn("REVERSAL v1 prototype", table)
        self.assertIn("development", table)
        self.assertIn("flip_frozen", table)

    def test_invalid_modes_fail_before_creating_artifacts(self):
        for extra in (("--generations", "1"), ("--quick",), ("--no-plastic",),
                      ("--cfg", "T=300"), ("--cfg", "n_items=5"), ("--cfg", "e0=nan"), ("--eval-worlds", "128"),
                      ("--condition-split", "final"), ("--structural-plasticity", runner.MODE)):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(*extra)
            self.assertFalse((self.root/"runs").exists())

    def test_invalid_source_uses_correct_protocol_label(self):
        np.save(self.source, np.zeros(3, np.float32))
        with self.assertRaisesRegex(SystemExit, "REVERSAL v1 requires"):
            self.invoke()
        self.assertFalse((self.root/"runs").exists())

    def test_nonfinite_results_preserve_failed_artifacts(self):
        with self.assertRaises(SystemExit):
            self.invoke(bad=True)
        path = next((self.root/"runs").iterdir())
        result = json.loads((path/"summary.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse((path/"genome_final.npy").exists())


if __name__ == "__main__":
    unittest.main()
