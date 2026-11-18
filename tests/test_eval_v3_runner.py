"""Tracked diagnostic dispatch, immutable input and failure evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import eval_v3, run_experiment as runner, summarize


class V3RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / ".tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.npy"
        np.save(self.source, np.zeros(155, np.float32))
        self.argv = ["runner", "--eval-protocol", "v3", "--generations", "0",
                     "--eval-worlds", "256", "--start", f"file:{self.source}"]

    def invoke(self, extra=(), bad=False, error=None):
        metric = dict(fitness=float("nan") if bad else 2., fitness_se=.25,
                      greedy_fraction=.2, greedy_fraction_se=.03)
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="fixture", dirty=False)),
              patch.object(eval_v3, "evaluate_conditions", return_value=(
                  {"stationary": {"genome": metric}}, {"fixture": np.array([1])}),
                  side_effect=error) as evaluate,
              patch.object(runner, "make_evaluator") as training,
              patch.object(runner.sys, "argv", self.argv + list(extra)),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()
            training.assert_not_called()
        return next((self.root / "runs").iterdir()), evaluate

    def test_dispatch_preserves_source_and_reports_uncertainty(self):
        folder, call = self.invoke()
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(call.call_args.args[-2:], (256, eval_v3.EVAL_SEED))
        source = self.source.read_bytes()
        self.assertEqual(config["source_genome_sha256"], hashlib.sha256(source).hexdigest())
        for name in ("genome_start.npy", "genome_final.npy"):
            self.assertEqual((folder / name).read_bytes(), source)
        with np.load(folder / "events.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["fixture"], [1])
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("EVAL v3 diagnostics", table)
        self.assertIn("2.000000 | 0.250000 | 0.200000 | 0.030000", table)

    def test_missing_summary_keeps_incomplete_panel_visible(self):
        folder = self.root / "runs" / "incomplete"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(dict(
            args=dict(eval_protocol="v3", eval_worlds=256), eval_seed=eval_v3.EVAL_SEED,
            eval_worlds=256, git=dict(commit="fixture", dirty=False))))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("no summary", table)
        self.assertIn(str(eval_v3.EVAL_SEED), table)
        self.assertIn("incomplete", table)

    def test_invalid_requests_create_no_folder(self):
        for extra in (("--generations", "1"), ("--quick",), ("--no-plastic",),
                      ("--eval-worlds", "128"), ("--cfg", "T=12"),
                      ("--condition-split", "final"), ("--resume", "unused"),
                      ("--structural-plasticity", runner.MODE), ("--start", "biology")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(extra)
            self.assertFalse((self.root / "runs").exists())

    def test_invalid_genome_rejected_before_folder(self):
        np.save(self.source, np.full(155, np.nan, np.float32))
        with self.assertRaisesRegex(SystemExit, "finite genome"):
            self.invoke()
        self.assertFalse((self.root / "runs").exists())

    def test_nonfinite_summary_retains_failed_status_and_raw_evidence(self):
        with self.assertRaises(SystemExit):
            self.invoke(bad=True)
        folder = next((self.root / "runs").iterdir())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertNotIn("evaluations", summary)
        self.assertTrue((folder / "events.npz").exists())
        self.assertFalse((folder / "genome_final.npy").exists())

    def test_interrupt_is_preserved(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(error=KeyboardInterrupt())
        folder = next((self.root / "runs").iterdir())
        self.assertEqual(json.loads((folder / "summary.json").read_text())["status"], "interrupted")
        self.assertFalse((folder / "genome_final.npy").exists())

    def test_implicit_resume_of_v3_rejected_without_mutation(self):
        folder, _ = self.invoke()
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with (patch.object(runner, "git_info", return_value=dict(commit="fixture", dirty=False)),
              patch.object(runner.sys, "argv", ["runner", "--resume", str(folder)]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            runner.main()
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
