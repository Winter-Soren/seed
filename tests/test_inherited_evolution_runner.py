"""E119 continuation provenance, training archive flow and immutable evidence."""
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
from scripts import run_inherited_evolution as entry
from scripts import summarize


class InheritedEvolutionRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="inherited-evolution-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (entry.DEVELOPMENT_SOURCE, *entry.SOURCES):
            for name in ("config.json", "status.json", "genome_final.npy",
                         f"generations/generation_{source['initial_generation']:04d}.npz"):
                path = self.root / source["run"] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((runner.ROOT / source["run"] / name).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E119-benchmark", "--seed", "45", "--generations", "1",
                       "--pop", "16", "--worlds", "4", "--sigma", "0.05", "--lr", "0.02",
                       "--eval-worlds", "4", "--start", "file:" + entry.DEVELOPMENT_SOURCE["run"] + "/genome_final.npy"]
        self.training_calls = 0
        self.donor_calls = 0
        self.world_calls = 0
        self.fail_world = None
        self.corrupt_key = False
        self.nonfinite = False
        self.expire_training = False
        self.now = 0.

    def donor(self, flat, archive, master_seed, round_index, c, worlds=4):
        self.donor_calls += 1
        self.assertEqual(flat.shape, (155,))
        return dict(keys=entry.inherited_evolution.donor_keys(master_seed, round_index, worlds),
                    genome=np.asarray(flat), archive_before=np.asarray(archive).copy(),
                    archive_after=np.asarray(archive + np.float32(.01), np.float32))

    def evaluator(self, c, worlds):
        def evaluate(candidates, world_keys, development_keys, archive):
            self.training_calls += 1
            self.assertTrue(np.all(archive > 0))
            population = np.asarray(candidates)
            if self.expire_training:
                self.now = 901.
            shape = (len(population), worlds, c.M, *archive.shape)
            return dict(fitness=np.broadcast_to(population[:, :1], (len(population), worlds)),
                        counts=np.ones((len(population), worlds, 2, 2), np.float32),
                        survived=np.ones((len(population), worlds), np.float32),
                        candidate_birth=np.zeros(shape, np.float32),
                        candidate_initial=np.broadcast_to(archive, shape),
                        candidate_mask=np.ones(shape, bool))
        return evaluate, None

    def world(self, arms, master_seed, index, c):
        if index == self.fail_world:
            raise RuntimeError("intentional evaluation interruption")
        self.world_calls += 1
        self.assertEqual(tuple(arms), entry.inherited_evolution.ARMS)
        parent, child = arms["parent_previous"], arms["child_latest"]
        self.assertFalse(np.array_equal(parent[0], child[0]))
        np.testing.assert_array_equal(arms["parent_latest"][0], parent[0])
        np.testing.assert_array_equal(arms["parent_latest"][1], child[1])
        np.testing.assert_array_equal(arms["child_previous"][0], child[0])
        np.testing.assert_array_equal(arms["child_previous"][1], parent[1])
        self.assertFalse(np.any(arms["founder_empty"][1]))
        self.assertFalse(np.any(arms["child_empty"][1]))
        np.testing.assert_allclose(child[1] - parent[1], .01, atol=1e-7, rtol=0)
        keys = entry.inherited_evolution.evaluation_keys(master_seed, index)
        if self.corrupt_key:
            keys = entry.inherited_evolution.evaluation_keys(master_seed + 1, index)
        return dict(keys=keys, rule=np.asarray(index % 2, np.int32),
                    observation=np.asarray(np.nan if self.nonfinite else index, np.float32))

    @staticmethod
    def measurements(rows):
        metric = dict(mean=10., se=.5)
        arm = dict(latency=dict(metric), survival=dict(mean=.5, se=.01), fitness=dict(metric))
        gate = dict(passed=True, latency_passed=True, survival_passed=True)
        comparison = dict(latency_improvement=dict(metric), survival_difference=dict(mean=0., se=.01), gate=gate)
        return dict(families=len(rows), agents_per_family=4, assigned_agents=len(rows) * 4,
                    outside_born_agents=len(rows) * 3,
                    **{name: arm for name in entry.inherited_evolution.ARMS},
                    contrasts={name: comparison for name in entry.inherited_evolution.CONTRASTS}, gate=gate)

    def invoke(self, *extra, dirty=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *self.common, *extra]))
            stack.enter_context(patch.object(entry.inherited_evolution, "make_evaluator", side_effect=self.evaluator))
            stack.enter_context(patch.object(entry.inherited_evolution, "donor_update", side_effect=self.donor))
            stack.enter_context(patch.object(entry.inherited_evolution, "validate_donors"))
            stack.enter_context(patch.object(entry.inherited_evolution, "evaluate_world", side_effect=self.world))
            stack.enter_context(patch.object(entry.inherited_evolution, "validate_world"))
            stack.enter_context(patch.object(entry.inherited_evolution, "summarize", side_effect=self.measurements))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E119-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_cli_continues_optimizer_and_passes_new_donor_archive_into_es(self):
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((config["donor_seed"], config["evaluation_seed"], config["initial_generation"]), (45, 46, 1))
        self.assertEqual((result["generation"], result["optimizer_generation"], result["completed_families"]), (1, 2, 4))
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertEqual((self.donor_calls, self.training_calls, self.world_calls), (1, 1, 4))
        source = self.root / entry.DEVELOPMENT_SOURCE["run"] / "generations/generation_0001.npz"
        self.assertEqual((folder / "checkpoint_start.npz").read_bytes(), source.read_bytes())
        with np.load(source, allow_pickle=False) as prior, np.load(folder / "generations/generation_0000.npz", allow_pickle=False) as first:
            for name in ("mu", "m", "v", "t", "key"):
                np.testing.assert_array_equal(first[name], prior[name])
            self.assertTrue(np.any(first["m"]))
        trajectory = np.load(folder / "genome_trajectory.npy", allow_pickle=False)
        memory = np.load(folder / "archive_trajectory.npy", allow_pickle=False)
        self.assertEqual(trajectory.shape, (2, 155))
        self.assertFalse(np.any(memory[0]))
        np.testing.assert_allclose(memory[1], .01, atol=1e-9, rtol=0)
        metric = json.loads((folder / "metrics.jsonl").read_text())
        for name in ("checkpoint", "archive", "donor", "samples"):
            artifact = metric[name]
            self.assertEqual(hashlib.sha256((folder / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        with np.load(folder / metric["samples"]["path"], allow_pickle=False) as samples:
            np.testing.assert_array_equal(samples["archive"], memory[1])
            np.testing.assert_array_equal(samples["candidate_initial"][0, 0, 0], memory[1])
            np.testing.assert_array_equal(samples["eps"][:8], -samples["eps"][8:])
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(before, self.snapshot(folder))

    def test_final_identity_continues_each_source_to_optimizer_generation_twenty(self):
        for index, source in enumerate(entry.SOURCES):
            with self.subTest(index=index):
                self.invoke("--condition-split", "final", "--name", "E119-" + source["label"],
                            "--seed", str(2026112401 + index), "--generations", "10", "--eval-worlds", "128",
                            "--start", "file:" + source["run"] + "/genome_final.npy")
                folder = self.folder("E119-" + source["label"])
                result = json.loads((folder / "summary.json").read_text())
                config = json.loads((folder / "config.json").read_text())
                self.assertEqual((result["generation"], result["optimizer_generation"], result["completed_families"]), (10, 20, 128))
                self.assertEqual(config["evaluation_seed"], 2026112501 + index)
                self.assertEqual(len(result["generations"]), 11)
                self.assertEqual(len(result["archives"]), 11)
                self.assertEqual(len(result["donors"]), 10)
                self.assertTrue(result["gate"]["passed"])

    def test_source_and_cli_mismatches_fail_before_any_run_folder(self):
        cases = [("--seed", "123"), ("--generations", "2"), ("--pop", "32"), ("--sigma", "0.1"),
                 ("--no-plastic",), ("--persistence",), ("--allow-dirty",), ("--resume", "somewhere"),
                 ("--cfg", "T=3"), ("--max-train-seconds", "3"), ("--name", "new-name"),
                 ("--start", "file:" + entry.SOURCES[0]["run"] + "/genome_final.npy")]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(*arguments)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(list((self.root / "runs").glob("*_E119-*")))
        checkpoint = self.root / entry.DEVELOPMENT_SOURCE["run"] / "generations/generation_0001.npz"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E119-*")))

    def test_interruption_preserves_archives_and_checkpoints_without_a_decision(self):
        self.fail_world = 2
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["generation"], result["completed_families"]), ("failed", 1, 2))
        self.assertEqual(len(result["archives"]), 2)
        self.assertNotIn("measurements", result)
        self.assertNotIn("gate", result)
        before = self.snapshot(folder)
        with patch.object(runner, "ROOT", self.root), \
                patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=False)), \
                patch.object(runner.sys, "argv", ["runner", "--resume", str(folder)]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        self.assertEqual(before, self.snapshot(folder))

    def test_deadline_preserves_completed_update_and_does_not_open_evaluation(self):
        self.expire_training = True
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now):
            with self.assertRaises(TimeoutError):
                self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["generation"], result["completed_families"]), ("interrupted", 1, 0))
        self.assertEqual(len(result["archives"]), 2)
        self.assertEqual(self.world_calls, 0)

    def test_bad_evaluation_assignment_and_nonfinite_evidence_are_rejected(self):
        self.corrupt_key = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual(result["completed_families"], 0)
        self.assertFalse(result["scientific_result"])
        with self.assertRaises(ValueError):
            entry._save_arrays(self.folder(), "bad.npz", dict(x=np.nan))
        self.assertFalse((self.folder() / "bad.npz").exists())

    def test_results_table_suppresses_partial_comparisons(self):
        self.invoke()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("## Evolution with inherited experience", text)
        self.assertIn("not applicable", text)
        self.assertIn("10.000000", text)
        result_path = self.folder() / "summary.json"
        result = json.loads(result_path.read_text())
        result.update(status="failed", gate=dict(passed=True), scientific_result=True)
        result_path.write_text(json.dumps(result))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("undecided", text)
        self.assertNotIn("10.000000", text)


if __name__ == "__main__":
    unittest.main()
