"""Tracked E120 boundaries, immutable records and incomplete-result suppression."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_social_learning as entry, run_experiment as runner, summarize
from seed.seed import Cfg


class SocialLearningRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="social-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E120-benchmark", "--seed", "47", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls = []
        self.fail_at = None
        self.bad_key = False
        self.bad_swap = False
        self.nonfinite = False
        self.expire = False
        self.now = 0.

    @staticmethod
    def keys(master, index):
        return np.arange(index * 8, index * 8 + 8, dtype=np.uint32).reshape(4, 2)

    def family(self, flat, keys, index, c):
        self.assertEqual((flat.shape, flat.dtype), ((155,), np.dtype("float32")))
        self.assertEqual(c, Cfg(M=1))
        if index == self.fail_at:
            raise RuntimeError("deliberate teacher/observer interruption")
        self.calls.append(index)
        if self.expire:
            self.now = 901.
        return dict(keys=keys + 1 if self.bad_key else keys.copy(), family_index=np.asarray(index), rule=np.asarray(index % 2),
                    yoked_swap=np.asarray(1 - (index // 2) % 2 if self.bad_swap else (index // 2) % 2),
                    observation=np.asarray(np.nan if self.nonfinite else index, np.float32))

    @staticmethod
    def measurements(rows):
        estimate = dict(mean=.75, se=.05)
        arm = dict(first_food=estimate, survival=dict(mean=.5, se=.1))
        gate = dict(passed=True, first_food_passed=True, survival_passed=True)
        contrast = dict(first_food_difference=dict(mean=.2, se=.05),
                        survival_difference=dict(mean=0., se=.1), gate=gate)
        return dict(families=len(rows), assigned_newborns=len(rows),
                    arms={name: arm for name in ("observed", "yoked", "no_demo", "scene_only")},
                    contrasts={name: contrast for name in entry.CONTROLS}, gate=gate)

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry.social_learning, "family_keys", side_effect=self.keys))
            stack.enter_context(patch.object(entry.social_learning, "evaluate_family", side_effect=self.family))
            stack.enter_context(patch.object(entry.social_learning, "validate_family"))
            stack.enter_context(patch.object(entry.social_learning, "summarize", side_effect=self.measurements))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E120-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_cli_records_matched_assignment_and_immutable_numeric_evidence(self):
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(self.calls, list(range(4)))
        self.assertEqual(config["cfg"], Cfg(M=1)._asdict())
        self.assertEqual((config["practice_steps"], config["watch_steps"], config["solo_steps"]), (150, 150, 300))
        self.assertEqual(config["key_order"], list(entry.KEY_ORDER))
        self.assertEqual(config["controls"], list(entry.CONTROLS))
        self.assertEqual((result["status"], result["completed_families"]), ("finished", 4))
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertNotIn("gate", result)
        records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([(row["rule"], row["yoked_swap"]) for row in records], [(0, 0), (1, 0), (0, 1), (1, 1)])
        for index, (artifact, record) in enumerate(zip(result["raw_files"], records)):
            path = folder / artifact["path"]
            self.assertEqual(artifact, record["raw_file"])
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["keys"], self.keys(47, index))
        self.assertEqual((folder / "genome_start.npy").read_bytes(),
                         (self.root / entry.SOURCES[0]["path"]).read_bytes())

    def test_all_final_identities_have_the_registered_complete_gate(self):
        for index, source in enumerate(entry.SOURCES):
            self.calls.clear()
            self.invoke("--condition-split", "final", "--name", "E120-" + source["label"],
                        "--seed", str(2026112601 + index), "--eval-worlds", "128",
                        "--start", "file:" + source["path"])
            result = json.loads((self.folder("E120-" + source["label"]) / "summary.json").read_text())
            self.assertEqual(self.calls, list(range(128)))
            self.assertEqual(result["source_label"], source["label"])
            self.assertTrue(result["scientific_result"])
            self.assertEqual(result["gate"], dict(passed=True, first_food_passed=True, survival_passed=True))
            self.assertEqual(result["gate_status"], "passed")

    def test_unregistered_settings_dirty_state_and_source_changes_fail_before_creation(self):
        bad_flags = [("--seed", "48"), ("--eval-worlds", "8"), ("--generations", "1"),
                     ("--pop", "16"), ("--cfg", "M=2"), ("--persistence",),
                     ("--start", "biology"), ("--resume", "old"), ("--allow-dirty",)]
        for flags in bad_flags:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        path = self.root / entry.SOURCES[0]["path"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E120-*")))

    def test_existing_attempt_and_generic_resume_preserve_all_saved_bytes(self):
        self.invoke()
        folder = self.folder()
        original = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), common=False)
        self.assertEqual(self.snapshot(folder), original)

    def test_partial_failure_preserves_checkpointed_families_without_aggregate_claims(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "undecided")
        self.assertNotIn("measurements", result)
        self.assertEqual(len(list((folder / "families").glob("*.npz"))), 2)
        self.assertEqual(len((folder / "metrics.jsonl").read_text().splitlines()), 2)

    def test_deadline_preserves_completed_world_and_stops_further_worlds(self):
        self.expire = True
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now), self.assertRaises(TimeoutError):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("interrupted", 1))
        self.assertEqual(self.calls, [0])
        self.assertEqual(result["gate_status"], "undecided")

    def test_bad_assignments_nonfinite_data_and_partial_table_values_are_rejected(self):
        for flag in ("bad_key", "bad_swap", "nonfinite"):
            with self.subTest(flag=flag):
                setattr(self, flag, True)
                with tempfile.TemporaryDirectory(prefix="social-invalid-", dir=runner.ROOT / ".tmp") as temp:
                    invalid_root = Path(temp)
                    source_path = invalid_root / entry.SOURCES[0]["path"]
                    source_path.parent.mkdir(parents=True)
                    source_path.write_bytes((self.root / entry.SOURCES[0]["path"]).read_bytes())
                    with patch.object(self, "root", invalid_root), self.assertRaises(SystemExit):
                        self.invoke()
                    folder = next((invalid_root / "runs").glob("*_E120-*"))
                    self.assertFalse(list((folder / "families").glob("*.npz")))
                setattr(self, flag, False)
        self.invoke()
        folder = self.folder()
        path = folder / "summary.json"
        saved = json.loads(path.read_text())
        saved.update(status="failed", gate_status="passed", gate=dict(passed=True))
        path.write_text(json.dumps(saved))
        (self.root / "results").mkdir()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = (self.root / "results" / "RESULTS.md").read_text()
        row = next(line for line in text.splitlines() if folder.name in line)
        self.assertIn("undecided", row)
        self.assertIn("undefined", row)
        self.assertNotIn("PASS", row)


if __name__ == "__main__":
    unittest.main()
