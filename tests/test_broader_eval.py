"""Small deterministic fixtures; real rollouts run separately through the tracked runner."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import broader_eval as panel, run_experiment as runner, summarize
from seed.seed import Cfg


class BroaderEvaluationTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "saved.npy"
        np.save(self.source, np.zeros(155, dtype=np.float32))
        self.before = self.source.read_bytes()
        self.calls = []

    def evaluate(self, mu, c, n_worlds, plastic, seed):
        self.calls.append((np.asarray(mu).copy(), c, n_worlds, plastic, seed))
        return dict(fitness=float(c.n_items + 100 * c.decay + 3 * plastic),
                    acc_first=0.5, acc_second=0.6 if plastic else 0.55, survived=0.5, eaten=10)

    def invoke(self, *args, evaluator=None):
        argv = ["run_experiment.py", "--name", "fixture", "--eval-protocol", panel.PROTOCOL,
                "--generations", "0", "--start", f"file:{self.source}", *args]
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="fixture", branch="fixture", dirty=False)),
              patch.object(panel, "test_genome", side_effect=evaluator or self.evaluate),
              patch.object(runner, "make_evaluator") as training,
              patch.object(runner.sys, "argv", argv),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()
            training.assert_not_called()
        directory = next((self.root / "runs").iterdir())
        return directory, json.loads((directory / "config.json").read_text()), json.loads((directory / "summary.json").read_text())

    def test_conditions_change_exactly_one_field_and_keep_interface(self):
        self.assertEqual(panel.CONDITIONS["control"], {})
        for label, shift in panel.CONDITIONS.items():
            changed = Cfg()._replace(**shift)
            differences = [field for field in Cfg()._fields if getattr(Cfg(), field) != getattr(changed, field)]
            self.assertEqual(set(differences), set(shift), label)
            self.assertEqual(len(shift), 0 if label == "control" else 1)
            self.assertEqual((changed.R, changed.H, changed.n_htypes, changed.M, changed.T), (2, 48, 3, 4, 300))

    def test_development_dispatch_hashes_and_matched_controls(self):
        directory, config, summary = self.invoke("--eval-worlds", "1", "--seed", "2")
        self.assertEqual((summary["eval_protocol"], summary["condition_split"], summary["eval_seed"]),
                         ("broader-v1", "development", 8108))
        self.assertEqual(config["eval_protocol"]["rule_counts"], {"0": 1, "1": 0})
        self.assertEqual(config["input_genome"]["sha256"], hashlib.sha256(self.before).hexdigest())
        self.assertEqual(self.source.read_bytes(), self.before)
        for name in ("genome_start.npy", "genome_final.npy"):
            self.assertEqual((directory / name).read_bytes(), self.before)
        self.assertEqual(summary["status"], "finished")
        self.assertNotIn("final_learning", summary)
        self.assertEqual(len(self.calls), 10)
        for i, label in enumerate(panel.CONDITIONS):
            learning, frozen = self.calls[2*i:2*i+2]
            np.testing.assert_array_equal(learning[0], frozen[0])
            self.assertEqual(learning[1:], (Cfg()._replace(**panel.CONDITIONS[label]), 1, True, 8108))
            self.assertEqual(frozen[1:], (learning[1], 1, False, 8108))
            row = summary["condition_evaluations"][label]
            self.assertEqual(row["learning_minus_frozen_fitness"], 3.0)
            self.assertEqual(row["learning_advantage_shift"], 0.0)
            expected = row["learning"]["fitness"] - summary["condition_evaluations"]["control"]["learning"]["fitness"]
            self.assertEqual(row["learning_fitness_shift"], expected)
            self.assertAlmostEqual(row["learning_accuracy_gain"], 0.1)

    def test_final_dispatch_uses_reserved_seed_with_synthetic_evaluator(self):
        _, config, summary = self.invoke("--condition-split", "final")
        self.assertEqual(summary["eval_seed"], 8109)
        self.assertEqual(config["eval_protocol"]["rule_counts"], {"0": 64, "1": 64})
        self.assertTrue(all(call[2] == 128 and call[4] == 8109 for call in self.calls))

    def test_invalid_requests_fail_before_creating_artifacts(self):
        cases = (("--generations", "1"), ("--start", "blank"), ("--quick",), ("--no-plastic",),
                 ("--eval-worlds", "0"), ("--cfg", "n_items=12"), ("--cfg", "decay=0.004"),
                 ("--cfg", "G=4"), ("--cfg", "n_htypes=0"),
                 ("--condition-split", "final", "--eval-worlds", "2"),
                 ("--condition-split", "final", "--cfg", "T=4"))
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke(*args)
            self.assertFalse((self.root / "runs").exists())
        self.assertEqual(self.calls, [])

    def test_malformed_input_fails_before_artifacts_and_never_changes_source(self):
        for invalid in (np.zeros(154, np.float32), np.zeros((1, 155), np.float32),
                        np.full(155, np.nan, np.float32), np.zeros(155, np.float64)):
            np.save(self.source, invalid)
            original = self.source.read_bytes()
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(SystemExit):
                self.invoke()
            self.assertFalse((self.root / "runs").exists())
            self.assertEqual(self.source.read_bytes(), original)

    def test_nonfinite_evaluation_preserves_failed_run(self):
        def bad(*args, **kwargs):
            return dict(fitness=float("nan"), acc_first=0.5, acc_second=0.5, survived=0.5, eaten=10)
        with self.assertRaises(SystemExit):
            self.invoke(evaluator=bad)
        directory = next((self.root / "runs").iterdir())
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("nonfinite", summary["error"])
        self.assertEqual((directory / "genome_start.npy").read_bytes(), self.before)
        self.assertFalse((directory / "genome_final.npy").exists())

    def test_resume_refused_without_modifying_existing_panel(self):
        directory, _, _ = self.invoke()
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(directory))
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_legacy_resume_route_also_refuses_broader_panel(self):
        directory, _, _ = self.invoke()
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with self.assertRaises(SystemExit):
            self.invoke("--eval-protocol", "v1", "--resume", str(directory))
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_default_v1_preserves_exact_evaluation_call_contract(self):
        metrics = dict(fitness=1., acc_first=0.5, acc_second=0.6, survived=0.5, eaten=10)
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="fixture", dirty=False)),
              patch.object(runner, "make_evaluator", return_value=(None, None)),
              patch.object(runner, "test_genome", return_value=metrics) as evaluate,
              patch.object(panel, "evaluate_conditions") as broader,
              patch.object(runner.sys, "argv", ["runner", "--generations", "0"]),
              contextlib.redirect_stdout(io.StringIO())):
            runner.main()
        self.assertEqual([call.kwargs for call in evaluate.call_args_list],
                         [dict(n_worlds=128, plastic=True), dict(n_worlds=128, plastic=False),
                          dict(n_worlds=128, plastic=True)])
        broader.assert_not_called()
        directory = next((self.root / "runs").iterdir())
        summary = json.loads((directory / "summary.json").read_text())
        self.assertEqual(summary["final_learning"], metrics)
        self.assertNotIn("condition_evaluations", summary)

    def test_generated_table_separates_broader_protocol_and_labels_development(self):
        directory, _, _ = self.invoke("--eval-worlds", "1")
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results" / "RESULTS.md").read_text()
        self.assertIn("BROADER v1", table)
        self.assertIn("development", table)
        self.assertIn("8108", table)
        self.assertIn("high-cost", table)
        self.assertEqual(table.count(directory.name), 5)


if __name__ == "__main__":
    unittest.main()
