"""Mechanism isolation and tracked diagnostic dispatch, without full research runs."""
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
from jax.flatten_util import ravel_pytree
import numpy as np

from seed import seed
from scripts import run_experiment as runner, summarize


class EngineCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = Path(__file__).parent / "fixtures"
        cls.meta = json.loads((fixtures / "eval_v1_compatibility.json").read_text())
        cls.expected = np.load(fixtures / "eval_v1_compatibility.npz")
        cls.c = seed.Cfg(**cls.meta["cfg"])
        cls.g = seed.init_genome(cls.c, jax.random.PRNGKey(cls.meta["genome_seed"]))
        cls.g["mod"] = jnp.array(cls.meta["mod"])
        cls.mu = ravel_pytree(cls.g)[0]

    def rollout(self, plastic, **kwargs):
        m = self.meta
        return seed.rollout(self.g, jax.random.PRNGKey(m["world_seed"]),
                            jax.random.PRNGKey(m["development_seed"]), m["rule"],
                            self.c, plastic, True, **kwargs)

    def test_default_and_explicit_modes_match_prechange_v1_fixture(self):
        for name, plastic in (("both", True), ("frozen", False)):
            for kwargs in ({}, {"predictor_plastic": plastic}):
                with self.subTest(name=name, kwargs=kwargs):
                    result = self.rollout(plastic, **kwargs)
                    for key, value in result.items():
                        leaves = value.items() if isinstance(value, dict) else [(None, value)]
                        for child, array in leaves:
                            path = f"{name}__{key}" + (f"__{child}" if child else "")
                            np.testing.assert_array_equal(array, self.expected[path], err_msg=path)
                    self.assertEqual(seed.test_genome(self.mu, self.c, n_worlds=4, plastic=plastic,
                                                      seed=123, **kwargs), self.meta["evaluations"][name])

    def test_predictor_only_is_behaviorally_identical_to_frozen(self):
        frozen = self.rollout(False, predictor_plastic=False)
        predictor = self.rollout(False, predictor_plastic=True)
        for key in ("fitness", "counts", "survived"):
            np.testing.assert_array_equal(frozen[key], predictor[key], err_msg=key)
        for key in ("grid", "pos", "alive", "sig", "energy"):
            np.testing.assert_array_equal(frozen["traj"][key], predictor["traj"][key], err_msg=key)
        self.assertTrue(np.any(np.asarray(frozen["traj"]["surprise"]) !=
                               np.asarray(predictor["traj"]["surprise"])))

    def test_brain_update_switches_are_independent(self):
        _, d, _, n, _ = seed.dims(self.c)
        w, wp = jnp.zeros((n, n)), jnp.zeros((d, self.c.H))
        x = jnp.linspace(0.1, 0.7, n)
        hebb = jnp.zeros((n, n, 4)).at[..., 0].set(1.0)
        args = (w, x, wp, jnp.ones(d), 0.0, jnp.ones_like(w), hebb, 0.1, 0.2, jnp.array([0., 0., 1.]))
        hebbian = seed.brain_step(*args, 1.0, self.c, predictor_plastic=0.0)
        predictor = seed.brain_step(*args, 0.0, self.c, predictor_plastic=1.0)
        np.testing.assert_array_equal(hebbian[2], wp)
        np.testing.assert_array_equal(predictor[0], w)
        self.assertTrue(np.any(np.asarray(hebbian[0]) != np.asarray(w)))
        self.assertTrue(np.any(np.asarray(predictor[2]) != np.asarray(wp)))


class DiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / ".tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.npy"
        np.save(self.source, np.zeros(seed.genome_size(seed.Cfg()), dtype=np.float32))
        self.base = ["--eval-protocol", "v2", "--start", f"file:{self.source}",
                     "--generations", "0", "--eval-worlds", "128"]

    def invoke(self, *args, fitness=(4.0, 3.0, 1.0)):
        results = [dict(fitness=f, acc_first=.5, acc_second=.6, survived=.75, eaten=20) for f in fitness]
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="test", branch="test", dirty=False)),
              patch.object(runner, "make_evaluator", return_value=(None, None)),
              patch.object(runner, "test_genome", side_effect=results) as evaluate,
              patch.object(runner.sys, "argv", ["run_experiment.py", *args]),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()
        return evaluate

    def directory(self, name):
        return next((self.root / "runs").glob(f"*_{name}"))

    def snapshot(self, directory):
        return {p.name: p.read_bytes() for p in directory.iterdir()}

    def test_v2_dispatch_and_source_provenance(self):
        evaluate = self.invoke("--name", "v2", *self.base)
        directory = self.directory("v2")
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["eval_protocol"], "v2")
        self.assertEqual((summary["generation"], summary["eval_seed"], summary["eval_worlds"]), (0, 123, 128))
        self.assertEqual(list(summary["evaluations"]), ["both", "hebbian_only", "frozen"])
        self.assertNotIn("final_learning", summary)
        self.assertNotIn("start_learning", summary)
        self.assertEqual([c.kwargs for c in evaluate.call_args_list], [
            dict(n_worlds=128, seed=123, plastic=True, predictor_plastic=True),
            dict(n_worlds=128, seed=123, plastic=True, predictor_plastic=False),
            dict(n_worlds=128, seed=123, plastic=False, predictor_plastic=False)])
        self.assertEqual(config["source_genome_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual((directory / "genome_start.npy").read_bytes(), self.source.read_bytes())
        np.testing.assert_array_equal(np.load(directory / "genome_final.npy"), np.load(self.source))
        self.assertEqual((directory / "metrics.jsonl").read_bytes(), b"")

    def test_invalid_v2_settings_fail_before_creating_a_run(self):
        for override in (("--generations", "1"), ("--start", "blank"), ("--eval-worlds", "64"),
                         ("--cfg", "n_items=24"), ("--quick",), ("--no-plastic",)):
            with self.subTest(override=override), self.assertRaisesRegex(SystemExit, "EVAL v2 requires"):
                self.invoke(*self.base, *override)
        self.assertFalse((self.root / "runs").exists())

    def test_invalid_source_genomes_fail_before_creating_a_run(self):
        for values in (np.zeros(3, np.float32), np.full(155, np.nan, np.float32),
                       np.ones(155, np.complex64), np.zeros((1, 155), np.float32)):
            np.save(self.source, values)
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaisesRegex(SystemExit, "EVAL v2 requires"):
                self.invoke(*self.base)
        self.assertFalse((self.root / "runs").exists())

    def test_old_config_defaults_to_v1_and_resume_cannot_switch_protocol(self):
        self.invoke("--name", "old", "--generations", "0")
        old = self.directory("old")
        config = json.loads((old / "config.json").read_text())
        del config["args"]["eval_protocol"]
        (old / "config.json").write_text(json.dumps(config))
        evaluate = self.invoke("--resume", str(old))
        self.assertEqual([c.kwargs for c in evaluate.call_args_list], [dict(n_worlds=128, plastic=p)
                                                                     for p in (True, False, True)])
        self.invoke("--name", "new", *self.base)
        for directory, protocol in ((old, "v2"), (self.directory("new"), "v1")):
            before = self.snapshot(directory)
            with self.assertRaisesRegex(SystemExit, "Cannot change evaluation protocol"):
                self.invoke("--resume", str(directory), "--eval-protocol", protocol)
            self.assertEqual(before, self.snapshot(directory))

    def test_v2_resume_uses_saved_source_and_rejects_changed_saved_state(self):
        self.invoke("--name", "resume", *self.base)
        directory = self.directory("resume")
        self.source.unlink()
        self.invoke("--resume", str(directory))
        saved = directory / "genome_start.npy"
        source_bytes = saved.read_bytes()
        saved.write_bytes(source_bytes + b"changed")
        before = self.snapshot(directory)
        with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
            self.invoke("--resume", str(directory))
        self.assertEqual(before, self.snapshot(directory))
        saved.write_bytes(source_bytes)
        state = runner.load_checkpoint(directory)
        state["mu"] = state["mu"].at[0].set(1.0)
        runner.save_checkpoint(directory, state)
        before = self.snapshot(directory)
        with self.assertRaisesRegex(SystemExit, "checkpoint differs"):
            self.invoke("--resume", str(directory))
        self.assertEqual(before, self.snapshot(directory))

    def test_nonfinite_evaluation_cannot_be_published_as_finished(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite EVAL v2"):
            self.invoke("--name", "nan", *self.base, fitness=(4.0, float("nan"), 1.0))
        summary = json.loads((self.directory("nan") / "summary.json").read_text())
        self.assertNotEqual(summary["status"], "finished")

    def test_summary_separates_protocols_and_preserves_signed_differences(self):
        self.invoke("--name", "v1", "--generations", "0")
        self.invoke("--name", "v2", *self.base, fitness=(1., 3., -1.))
        self.invoke("--name", "zero-denominator", *self.base, fitness=(1., 3., 1.))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        v1, v2 = (self.root / "results" / "RESULTS.md").read_text().split("## EVAL v2 diagnostics")
        self.assertIn(self.directory("v1").name, v1)
        self.assertNotIn(self.directory("v2").name, v1)
        self.assertNotIn(self.directory("v1").name, v2)
        self.assertIn("| 1.000000 | 3.000000 | -1.000000 | -2.000000 | 4.000000 | 2.000000 | 123 | 128 |", v2)
        self.assertIn("| 1.000000 | 3.000000 | 1.000000 | -2.000000 | 2.000000 | undefined |", v2)


if __name__ == "__main__":
    unittest.main()
