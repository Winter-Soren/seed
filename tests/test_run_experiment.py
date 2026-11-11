"""Runner scheduling checks; tiny fitness fixture replaces expensive world rollouts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_experiment as runner


class RunnerSchedulingTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 0.0
        self.real_step = runner.es_step

    def evaluate(self, population, world_keys, development_keys):
        p, worlds = len(population), len(world_keys)
        fitness = -np.square(np.asarray(population) - 0.2).mean(axis=1)
        return dict(fitness=np.broadcast_to(fitness[:, None], (p, worlds)),
                    counts=np.ones((p, worlds, 2, 2)), survived=np.full((p, worlds), 0.5))

    def step(self, *args):
        result = self.real_step(*args)
        self.now += 2.0
        return result

    def invoke(self, *args, dirty=False):
        result = dict(fitness=1.0, acc_first=0.5, acc_second=0.6, survived=0.5, eaten=10)
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="test", branch="test", dirty=dirty)),
              patch.object(runner, "make_evaluator", return_value=(self.evaluate, None)),
              patch.object(runner, "es_step", side_effect=self.step),
              patch.object(runner, "test_genome", return_value=result),
              patch.object(runner.time, "monotonic", side_effect=lambda: self.now),
              patch.object(runner.sys, "argv", ["run_experiment.py", *args]),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()

    def run_dir(self, name):
        return next((self.root / "runs").glob(f"*_{name}"))

    def read_json(self, directory, name):
        return json.loads((directory / name).read_text())

    def test_bounded_resume_matches_uninterrupted_state_and_metrics(self):
        self.invoke("--name", "whole", "--generations", "5", "--pop", "4", "--worlds", "2")
        self.invoke("--name", "split", "--generations", "5", "--pop", "4", "--worlds", "2",
                    "--max-train-seconds", "3")
        split = self.run_dir("split")
        summary = self.read_json(split, "summary.json")
        self.assertEqual((summary["status"], summary["generation"]), ("interrupted", 2))
        self.assertNotIn("final_learning", summary)
        self.assertFalse((split / "genome_final.npy").exists())
        self.assertEqual(runner.load_checkpoint(split)["t"], 2)
        self.assertEqual(self.read_json(split, "config.json")["args"]["generations"], 5)

        # A new invocation gets its own limit, rather than the saved initial limit.
        self.invoke("--resume", str(split), "--max-train-seconds", "1")
        self.assertEqual(self.read_json(split, "summary.json")["generation"], 3)
        config = self.read_json(split, "config.json")
        self.assertEqual(config["resumes"][-1]["max_train_seconds"], 1.0)
        self.invoke("--resume", str(split))
        self.assertEqual(self.read_json(split, "summary.json")["status"], "finished")
        whole = self.run_dir("whole")
        for key, value in runner.load_checkpoint(whole).items():
            np.testing.assert_array_equal(value, runner.load_checkpoint(split)[key], err_msg=key)
        rows = lambda d: [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([r["gen"] for r in rows(split)], [0, 1, 2, 3, 4])
        self.assertEqual(rows(split), rows(whole))
        np.testing.assert_array_equal(np.load(split / "genome_final.npy"),
                                      np.load(whole / "genome_final.npy"))

    def test_default_quick_and_zero_generation_runs_still_finish(self):
        for name, args, target in (("default", ["--generations", "4"], 4),
                                   ("quick", ["--quick"], 3),
                                   ("reference", ["--generations", "0"], 0)):
            with self.subTest(name=name):
                self.invoke("--name", name, "--pop", "4", "--worlds", "2", *args)
                summary = self.read_json(self.run_dir(name), "summary.json")
                self.assertEqual((summary["status"], summary["generation"]), ("finished", target))
                self.assertEqual(summary["eval_worlds"], 4 if name == "quick" else 128)
                self.assertIn("final_learning", summary)
                self.assertIn("final_frozen", summary)

    def test_invalid_limits_fail_before_creating_run_artifacts(self):
        for limit in ("0", "-1", "nan", "inf"):
            with self.subTest(limit=limit), self.assertRaises(SystemExit) as error:
                self.invoke("--max-train-seconds", limit)
            self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "runs").exists())

    def test_dirty_resume_is_rejected_before_modifying_artifacts(self):
        self.invoke("--name", "clean", "--generations", "0")
        directory = self.run_dir("clean")
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with self.assertRaisesRegex(SystemExit, "Uncommitted code"):
            self.invoke("--resume", str(directory), dirty=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})


if __name__ == "__main__":
    unittest.main()
