"""E122 fixed-teacher provenance, tie skipping and complete paired evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from scripts import run_social_observer_evolution as entry, run_experiment as runner, summarize


class SocialObserverRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="observer-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E122-benchmark", "--seed", "49", "--generations", "1",
                       "--pop", "16", "--worlds", "4", "--sigma", "0.05", "--lr", "0.02",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.teacher_vectors = []
        self.training_calls = 0
        self.family_calls = 0
        self.all_tied = False
        self.alternate_ties = False
        self.fail_at = None
        self.bad_teacher = False
        self.bad_key = False
        self.expire = False
        self.now = 0.

    def teacher_batch(self, flat, world_keys, c):
        self.teacher_vectors.append(flat.copy())
        keys = np.stack([np.asarray(jax.random.split(key, 3)) for key in world_keys])
        return dict(teacher_genome=flat + 1 if self.bad_teacher else flat.copy(),
                    world_keys=np.asarray(world_keys), role_keys=keys, solo_world_keys=keys[:, 2],
                    rules=np.arange(len(world_keys), dtype=np.int32) % 2,
                    movies=np.zeros((len(world_keys), c.T // 2, 103), np.float32))

    def evaluator(self, c):
        def evaluate(population, movies, solo_world_keys, birth_keys, rules):
            self.training_calls += 1
            count, worlds = len(population), len(movies)
            tied = self.all_tied or (self.alternate_ties and self.training_calls % 2 == 0)
            score = np.zeros(count, np.float32) if tied else np.arange(count, dtype=np.float32) % 2
            fitness = np.broadcast_to(score[:, None], (count, worlds)).copy()
            if self.expire:
                self.now = 901.
            return dict(fitness=fitness, counts=np.ones((count, worlds, 2, 2), np.float32),
                        survived=np.ones((count, worlds), np.float32))
        return evaluate

    def family(self, teacher, starting, evolved, keys, index, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate observer evaluation interruption")
        self.family_calls += 1
        np.testing.assert_array_equal(teacher, starting)
        self.assertTrue(all(np.array_equal(item, teacher) for item in self.teacher_vectors))
        return dict(keys=keys + 1 if self.bad_key else keys.copy(), family_index=np.asarray(index),
                    rule=np.asarray(index % 2), yoked_swap=np.asarray((index // 2) % 2),
                    teacher_genome=teacher.copy(), starting_genome=starting.copy(), evolved_genome=evolved.copy())

    @staticmethod
    def measurements(rows):
        estimate = dict(mean=.75, se=.05)
        arm = dict(first_food=estimate, survival=dict(mean=.5, se=.1))
        gate = dict(passed=True, first_food_passed=True, survival_passed=True)
        contrast = dict(first_food_difference=dict(mean=.2, se=.05),
                        survival_difference=dict(mean=0., se=.1), gate=gate)
        return dict(families=len(rows), assigned_newborns=len(rows),
                    arms={name: arm for name in entry.observer.ARMS},
                    contrasts={name: contrast for name in entry.observer.CONTROLS}, gate=gate)

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry.observer, "make_training_batch", side_effect=self.teacher_batch))
            stack.enter_context(patch.object(entry.observer, "make_evaluator", side_effect=self.evaluator))
            stack.enter_context(patch.object(entry.observer, "evaluate_family", side_effect=self.family))
            stack.enter_context(patch.object(entry.observer, "validate_family"))
            stack.enter_context(patch.object(entry.observer, "summarize", side_effect=self.measurements))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E122-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_cli_saves_fixed_teacher_attempt_and_full_rank_evidence(self):
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((config["seed"], config["evaluation_seed"]), (49, 50))
        self.assertEqual(config["source"], config["teacher_source"])
        self.assertEqual((result["attempted_rounds"], result["optimizer_updates"], result["completed_families"]), (1, 1, 4))
        self.assertEqual((len(self.teacher_vectors), self.training_calls, self.family_calls), (1, 1, 4))
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertEqual((folder / "teacher_genome.npy").read_bytes(), (folder / "genome_start.npy").read_bytes())
        record = json.loads((folder / "metrics.jsonl").read_text())
        for field in ("checkpoint", "samples", "teacher"):
            artifact = record[field]
            self.assertEqual(hashlib.sha256((folder / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        with np.load(folder / record["samples"]["path"], allow_pickle=False) as samples:
            np.testing.assert_array_equal(samples["eps"][:8], -samples["eps"][8:])
            np.testing.assert_array_equal(samples["development_keys"][:8], samples["development_keys"][8:])
            expected = np.tile(np.array([-4 / 15, 4 / 15], np.float32), 8)
            np.testing.assert_allclose(samples["rank_weights"], expected, atol=1e-7, rtol=0)
        self.assertGreater(result["changed_parameters"], 0)

    def test_all_tied_batch_preserves_endpoint_and_zero_optimizer_but_advances_key(self):
        self.all_tied = True
        self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["attempted_rounds"], result["optimizer_updates"], result["changed_parameters"]), (1, 0, 0))
        self.assertEqual(result["status"], "finished")
        with np.load(folder / "generations/attempt_0000.npz", allow_pickle=False) as first, \
                np.load(folder / "generations/attempt_0001.npz", allow_pickle=False) as last:
            for name in ("mu", "m", "v", "t"):
                np.testing.assert_array_equal(last[name], first[name])
            self.assertFalse(np.array_equal(last["key"], first["key"]))
        record = json.loads((folder / "metrics.jsonl").read_text())
        self.assertFalse(record["informative"])
        self.assertTrue(record["all_tied"])

    def test_three_final_lineages_record_ten_attempts_and_only_informative_updates(self):
        self.alternate_ties = True
        for index, source in enumerate(entry.SOURCES):
            self.teacher_vectors.clear()
            self.invoke("--condition-split", "final", "--name", "E122-" + source["label"],
                        "--seed", str(2026112801 + index), "--eval-worlds", "128", "--generations", "10",
                        "--start", "file:" + source["path"])
            folder = self.folder("E122-" + source["label"])
            result = json.loads((folder / "summary.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["evaluation_seed"], 2026112901 + index)
            self.assertEqual((result["attempted_rounds"], result["optimizer_updates"], len(result["attempts"])), (10, 5, 11))
            self.assertEqual(result["completed_families"], 128)
            self.assertTrue(result["gate"]["passed"])
            records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(sum(record["informative"] for record in records), 5)

    def test_unregistered_settings_and_source_edits_fail_before_creation(self):
        for flags in [("--seed", "48"), ("--generations", "2"), ("--eval-worlds", "8"),
                      ("--pop", "32"), ("--cfg", "M=2"), ("--persistence",), ("--start", "biology"),
                      ("--resume", "old"), ("--allow-dirty",), ("--max-train-seconds", "30")]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        source = self.root / entry.SOURCES[0]["path"]
        source.write_bytes(source.read_bytes() + b"changed")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E122-*")))

    def test_partial_evaluation_preserves_attempts_and_refuses_rerun_or_generic_resume(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["attempted_rounds"], result["completed_families"]), ("failed", 1, 2))
        self.assertNotIn("measurements", result)
        self.assertNotIn("gate", result)
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), common=False)
        self.assertEqual(before, self.snapshot(folder))

    def test_deadline_preserves_completed_attempt_without_evaluation(self):
        self.expire = True
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now), self.assertRaises(TimeoutError):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["attempted_rounds"], result["completed_families"]), ("interrupted", 1, 0))
        self.assertEqual(self.family_calls, 0)

    def test_fixed_teacher_violation_is_rejected_before_candidate_selection(self):
        self.bad_teacher = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["attempted_rounds"], result["optimizer_updates"]), (0, 0))
        self.assertEqual(self.training_calls, 0)

    def test_result_table_reports_informative_counts_and_suppresses_incomplete_scores(self):
        self.all_tied = True
        self.invoke()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("Informative updates", text)
        self.assertIn("| 1/1 | 0 | 4/4 |", text)
        self.assertIn("not applicable", text)
        result_path = self.folder() / "summary.json"
        result = json.loads(result_path.read_text())
        result.update(status="failed", gate=dict(passed=True), scientific_result=True)
        result_path.write_text(json.dumps(result))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("undecided", text)
        self.assertNotIn("0.750000", text)


if __name__ == "__main__":
    unittest.main()
