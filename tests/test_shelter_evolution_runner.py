"""E118 CLI integrity, immutable evolution evidence and incomplete-run handling."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_experiment as runner
from scripts import run_shelter_evolution as entry
from scripts import summarize


class ShelterEvolutionRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="shelter-evolution-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E118-benchmark", "--seed", "43", "--generations", "1",
                       "--pop", "16", "--worlds", "4", "--sigma", "0.05", "--lr", "0.02",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.training_calls = 0
        self.pair_calls = 0
        self.fail_pair = None
        self.nonfinite = False
        self.expire_training = False
        self.now = 0.

    def evaluator(self, c, worlds):
        def evaluate(candidates, world_keys, development_keys):
            self.training_calls += 1
            population = np.asarray(candidates)
            if self.expire_training:
                self.now = 901.
            return dict(fitness=np.broadcast_to(population[:, :1], (len(population), worlds)),
                        counts=np.ones((len(population), worlds, 2, 2), np.float32),
                        survived=np.ones((len(population), worlds), np.float32))
        return evaluate, None

    def pair(self, parent, descendant, master_seed, index, c):
        if index == self.fail_pair:
            raise RuntimeError("intentional evaluation interruption")
        self.pair_calls += 1
        self.assertEqual(parent.shape, (155,))
        self.assertEqual(descendant.shape, (155,))
        self.assertFalse(np.array_equal(parent, descendant))
        return dict(keys=entry.shelter_evolution.evaluation_keys(master_seed, index),
                    rule=np.asarray(index % 2, np.int32),
                    observation=np.asarray(np.nan if self.nonfinite else index, np.float32))

    @staticmethod
    def measurements(rows):
        metric = dict(mean=10., se=.5)
        arm = dict(latency=dict(metric), survival=dict(mean=.5, se=.01), fitness=dict(metric))
        return dict(families=len(rows), agents_per_family=4, assigned_agents=len(rows) * 4,
                    outside_born_agents=len(rows) * 3, parent=arm, descendant=arm,
                    latency_improvement=dict(metric), survival_difference=dict(mean=0., se=.01),
                    gate=dict(passed=True, latency_passed=True, survival_passed=True))

    def invoke(self, *extra, dirty=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *self.common, *extra]))
            stack.enter_context(patch.object(entry.shelter_evolution, "make_evaluator", side_effect=self.evaluator))
            stack.enter_context(patch.object(entry.shelter_evolution, "evaluate_pair", side_effect=self.pair))
            stack.enter_context(patch.object(entry.shelter_evolution, "validate_pair"))
            stack.enter_context(patch.object(entry.shelter_evolution, "summarize", side_effect=self.measurements))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E118-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_cli_saves_real_es_transitions_and_paired_development_evidence(self):
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((config["seed"], config["evaluation_seed"]), (43, 44))
        self.assertEqual(config["max_seconds"], 900)
        self.assertEqual(config["persistence"], dict(version="persistence-v1", memory_steps=4, stall_steps=2))
        self.assertEqual((result["status"], result["generation"], result["completed_families"]), ("finished", 1, 4))
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertNotIn("gate", result)
        self.assertEqual((self.training_calls, self.pair_calls), (1, 4))
        trajectory = np.load(folder / "genome_trajectory.npy", allow_pickle=False)
        self.assertEqual(trajectory.shape, (2, 155))
        np.testing.assert_array_equal(trajectory[0], np.load(folder / "genome_start.npy", allow_pickle=False))
        np.testing.assert_array_equal(trajectory[-1], np.load(folder / "genome_final.npy", allow_pickle=False))
        self.assertGreater(result["changed_parameters"], 0)
        for index, artifact in enumerate(result["generations"]):
            path = folder / artifact["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            with np.load(path, allow_pickle=False) as state:
                self.assertEqual(int(state["t"]), index)
                np.testing.assert_array_equal(state["mu"], trajectory[index])
        metric = json.loads((folder / "metrics.jsonl").read_text())
        with np.load(folder / metric["samples"]["path"], allow_pickle=False) as samples:
            self.assertEqual(samples["fitness"].shape, (16, 4))
            np.testing.assert_array_equal(samples["eps"][:8], -samples["eps"][8:])
            np.testing.assert_array_equal(samples["development_keys"][:8], samples["development_keys"][8:])
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(before, self.snapshot(folder))
        with patch.object(runner, "ROOT", self.root), \
                patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=False)), \
                patch.object(runner.sys, "argv", ["runner", "--resume", str(folder)]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        self.assertEqual(before, self.snapshot(folder))

    def test_final_identity_maps_training_evaluation_and_source_for_three_seeds(self):
        for index, source in enumerate(entry.SOURCES):
            with self.subTest(index=index):
                self.invoke("--condition-split", "final", "--name", "E118-" + source["label"],
                            "--seed", str(2026112201 + index), "--generations", "10", "--eval-worlds", "128",
                            "--start", "file:" + source["path"])
                folder = self.folder("E118-" + source["label"])
                config = json.loads((folder / "config.json").read_text())
                result = json.loads((folder / "summary.json").read_text())
                self.assertEqual(config["evaluation_seed"], 2026112301 + index)
                self.assertEqual(config["source"], source)
                self.assertEqual((result["generation"], result["completed_families"]), (10, 128))
                self.assertEqual(len(result["generations"]), 11)
                self.assertTrue(result["gate"]["passed"])
                self.assertTrue(result["scientific_result"])

    def test_registration_mismatches_fail_before_any_run_folder(self):
        cases = [("--seed", "123"), ("--generations", "2"), ("--pop", "32"), ("--sigma", "0.1"),
                 ("--no-plastic",), ("--persistence",), ("--allow-dirty",), ("--resume", "somewhere"),
                 ("--cfg", "T=3"), ("--max-train-seconds", "3"), ("--name", "new-name"),
                 ("--start", "file:" + entry.SOURCES[1]["path"])]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(*arguments)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(list((self.root / "runs").glob("*_E118-*")))
        source = self.root / entry.SOURCES[0]["path"]
        source.write_bytes(source.read_bytes() + b"tampered")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E118-*")))

    def test_evaluation_interruption_preserves_generations_and_completed_families(self):
        self.fail_pair = 2
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["generation"], result["completed_families"]), ("failed", 1, 2))
        self.assertEqual(len(result["generations"]), 2)
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertNotIn("measurements", result)
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(before, self.snapshot(folder))

    def test_deadline_stops_after_durable_generation_before_evaluation(self):
        self.expire_training = True
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now):
            with self.assertRaises(TimeoutError):
                self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["generation"], result["completed_families"]), ("interrupted", 1, 0))
        self.assertEqual(len(result["generations"]), 2)
        self.assertEqual(self.pair_calls, 0)
        self.assertNotIn("gate", result)

    def test_nonfinite_raw_evidence_is_rejected_before_file_creation(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_families"], 0)
        self.assertFalse(list((self.folder() / "families").glob("*.npz")))

    def test_results_index_keeps_development_undecided_and_suppresses_partial_estimates(self):
        self.invoke()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results" / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("## Shelter evolution", text)
        self.assertIn("not applicable", text)
        self.assertIn("10.000000", text)
        result_path = self.folder() / "summary.json"
        result = json.loads(result_path.read_text())
        result.update(status="failed", gate=dict(passed=True), scientific_result=True)
        result_path.write_text(json.dumps(result))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results" / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("undecided", text)
        self.assertNotIn("10.000000", text)
        self.assertNotIn("| PASS |", text)


if __name__ == "__main__":
    unittest.main()
