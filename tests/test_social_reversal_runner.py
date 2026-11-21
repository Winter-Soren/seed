"""E125 provenance, paired schedules, streamed evidence and transfer-only reporting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_social_reversal as entry, run_experiment as runner, summarize
from seed.seed import Cfg


class SocialReversalRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="reversal-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root, self.pins = Path(self.temp.name), {}
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        for source in entry.SOURCES:
            self.calibration_fixture(source, "final")
        self.calibration_fixture(entry.SOURCES[0], "development")
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E125-benchmark", "--seed", "2026120300", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.projected = [], []
        self.fail_at, self.tamper, self.expire = None, False, False
        self.reuse_keys, self.bad_flip, self.bad_schedule, self.bad_gate = False, False, False, False
        self.nonfinite, self.passed = False, True
        self.now = 0.

    def calibration_fixture(self, source, split):
        name = "E123-benchmark" if split == "development" else "E123-" + source["label"]
        folder = self.root / "runs" / ("20261121-000000_" + name)
        (folder / "calibration").mkdir(parents=True)
        count, seed = (4, 51) if split == "development" else (32, 2026113001 + source["training_seed"])
        artifacts = []
        for index in range(count):
            path = folder / f"calibration/family_{index:04d}.npz"
            np.savez_compressed(path, keys=np.asarray([[seed, index * 4 + role] for role in range(4)], np.uint32))
            artifacts.append(dict(index=index, path=path.relative_to(folder).as_posix(),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selection = dict(families=count, contacts=count, status="frozen", calibration_seed=seed,
            completed_families=count, expected_families=count, selected_lambda=1.,
            contact_coverage=dict(mean=1., se=0.), vote_coverage=dict(mean=1., se=0.), raw_files=artifacts,
            losses=[{"lambda": value, "mean_log_loss": (value - 1.) ** 2 + .2}
                    for value in (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)])
        config = dict(protocol="social-belief-readout-v1", condition_split=split, source=source,
            calibration_seed=seed, expected_calibration_families=count, calibration_worlds=count,
            cfg=Cfg(M=1)._asdict(), generation=0, args=dict(name=name), git=dict(commit="b" * 40, dirty=False))
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "calibration/selection.json").write_text(json.dumps(selection))
        (folder / "summary.json").write_text("forbidden prior evaluation")
        (folder / "families").mkdir()
        (folder / "families/family_0000.npz").write_bytes(b"forbidden prior world")
        self.pins[name] = dict(folder=folder.relative_to(self.root).as_posix(), commit="b" * 40,
            config_sha256=hashlib.sha256((folder / "config.json").read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256((folder / "calibration/selection.json").read_bytes()).hexdigest())

    @staticmethod
    def assignment_key(master):
        return np.asarray([master, 0xE125], np.uint32)

    def assignments_manifest(self, master, count):
        base = np.asarray([(0, 100), (1, 125), (0, 150), (1, 175)] if count == 4 else
                          [(rule, step) for rule in (0, 1) for step in entry.FLIP_STEPS for _ in range(16)], np.int32)
        permutation = np.arange(count - 1, -1, -1, dtype=np.int32)
        table = base[permutation]
        if self.bad_schedule:
            table[0] = table[1]
        return dict(key=self.assignment_key(master), permutation=permutation, table=table)

    def family_keys(self, master, index):
        keys = np.asarray([[master, index * 4 + role] for role in range(4)], np.uint32)
        if self.reuse_keys:
            keys[0] = np.asarray([51, 0], np.uint32)
        return keys

    def evaluate_family(self, flat, weights, selected_lambda, keys, index, rule, flip_step, c):
        self.assertEqual(c, Cfg(M=1))
        self.assertEqual((flat.shape, weights.shape, selected_lambda), ((155,), (157, 157), 1.))
        if index == self.fail_at:
            raise RuntimeError("deliberate family interruption")
        if self.expire:
            self.now = 3601.
        if self.tamper and index == 0:
            (self.root / self.pins["E123-benchmark"]["folder"] / "calibration/selection.json").write_text("{}")
        self.calls.append((int(keys[0, 0]), index, rule, flip_step))
        return dict(keys=keys.copy(), family_index=np.asarray(index), rule=np.asarray(rule),
            flip_step=np.asarray(flip_step + 1 if self.bad_flip else flip_step), lambda_value=np.asarray(selected_lambda),
            original_genome=flat.copy(), direct_weights=weights.copy(),
            score=np.asarray(np.nan if self.nonfinite else .75), large_trace=np.zeros((8, 300, 20), np.float32))

    @staticmethod
    def summary_input(row):
        return {field: row[field] for field in ("family_index", "rule", "flip_step", "score")}

    def measurements(self, rows):
        self.projected = rows
        self.assertTrue(all(set(row) == {"family_index", "rule", "flip_step", "score"} for row in rows))
        estimate = dict(mean=.7, se=.02)
        delta = dict(mean=.1, se=.02, lower_2se=.06, upper_2se=.14)
        window = {metric: estimate for metric in ("first_food", "fitness", "no_meal", "accuracy")}
        condition = dict(windows={name: window for name in ("pre", "early", "late")})
        contrast = dict(late_first_food_difference=delta, recovery_difference=delta, end_survival_difference=delta)
        arm = dict(conditions={name: condition for name in entry.CONDITIONS}, reversal_minus_stationary=contrast)
        gate = {name: True for name in entry.GATE_FIELDS}
        gate.update(passed=True if self.bad_gate else self.passed, recovery_passed=self.passed)
        return dict(families=len(rows), primary="observed", arms={name: arm for name in entry.ARMS}, gate=gate)

    def invoke(self, *extra, dirty=False, common=True, timed=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry.calibration, "CALIBRATION_SOURCES", self.pins))
            stack.enter_context(patch.object(entry, "_runtime", return_value={"test": "mock backend"}))
            for name, method in (("assignment_key", self.assignment_key), ("assignments_manifest", self.assignments_manifest),
                                 ("family_keys", self.family_keys), ("evaluate_family", self.evaluate_family),
                                 ("summary_input", self.summary_input), ("summarize", self.measurements)):
                stack.enter_context(patch.object(entry.social_reversal, name, side_effect=method))
            if timed:
                stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        return ["--condition-split", "final", "--name", f"E125-A{index}", "--seed", str(2026120301 + index),
                "--eval-worlds", "128", "--start", "file:" + entry.SOURCES[index]["path"]]

    def folder(self, name="E125-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E125-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    def test_benchmark_preserves_sources_schedule_hashes_and_compact_summary_inputs(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual((config["max_seconds"], config["primary"], config["selected_neutral_birth_reuses"]),
                         (900, "observed", "birth_parent"))
        self.assertEqual(config["formal_ladder_status"], "not_decided")
        self.assertEqual(result["completed_families"], 4)
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertTrue(all("large_trace" not in row for row in self.projected))
        expected = self.assignments_manifest(2026120300, 4)
        self.assertEqual([[rule, step] for _, _, rule, step in self.calls], expected["table"].tolist())
        with np.load(folder / "assignments.npz", allow_pickle=False) as arrays:
            for name in expected:
                np.testing.assert_array_equal(arrays[name], expected[name])
        self.assertEqual((folder / "direct_genome.npy").read_bytes(),
                         (self.root / entry.DIRECT_SOURCES[0]["path"]).read_bytes())
        records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        for artifact, record in zip(result["raw_files"], records):
            path = folder / artifact["path"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["assignments_sha256"], config["assignments"]["sha256"])
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["assignments_sha256"].tobytes().hex(), config["assignments"]["sha256"])
                self.assertEqual(arrays["calibration_sha256"].tobytes().hex(), config["frozen_calibration"]["sha256"])
                self.assertEqual(arrays["direct_genome"].shape, (8424,))

    def test_three_final_identities_have_balanced_cells_and_transfer_only_decisions(self):
        for index in range(3):
            self.calls.clear()
            self.invoke(*self.final_args(index))
            result = self.result(f"E125-A{index}")
            self.assertEqual(result["completed_families"], 128)
            self.assertTrue(result["scientific_result"])
            self.assertTrue(result["gate"]["passed"])
            self.assertEqual(result["formal_ladder_status"], "not_decided")
            cells = [(rule, step) for _, _, rule, step in self.calls]
            self.assertTrue(all(cells.count((rule, step)) == 16 for rule in (0, 1) for step in entry.FLIP_STEPS))

    def test_bad_flags_dirty_state_unbalanced_schedule_and_source_changes_precede_creation(self):
        for flags in (("--seed", "51"), ("--eval-worlds", "8"), ("--generations", "1"),
                      ("--cfg", "M=2"), ("--pop", "16"), ("--allow-dirty",), ("--resume", "old")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.bad_schedule = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.bad_schedule = False
        direct_path = self.root / entry.DIRECT_SOURCES[0]["path"]
        direct_path.write_bytes(direct_path.read_bytes() + b"changed")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E125-*")))

    def test_existing_attempt_and_generic_resume_preserve_all_evidence(self):
        self.invoke()
        folder = self.folder()
        snapshot = lambda: {path.relative_to(folder).as_posix(): path.read_bytes()
                            for path in folder.rglob("*") if path.is_file()}
        before = snapshot()
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), common=False)
        self.assertEqual(snapshot(), before)

    def test_partial_failure_preserves_prefix_without_gate_or_aggregate_outcomes(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertNotIn("measurements", result)
        self.assertNotIn("gate", result)
        self.assertEqual(len(list((self.folder() / "families").glob("*.npz"))), 2)

    def test_source_calibration_tamper_stops_and_preserves_completed_family(self):
        self.tamper = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual((self.result()["status"], self.result()["completed_families"]), ("failed", 1))

    def test_calibration_key_reuse_is_rejected_before_any_world(self):
        self.reuse_keys = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)
        self.assertFalse(self.calls)

    def test_boundary_mismatch_rejected_before_saving_its_family(self):
        self.bad_flip = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)
        self.assertFalse(list((self.folder() / "families").glob("*.npz")))

    def test_time_cap_preserves_completed_family(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke(*self.final_args(0), timed=True)
        self.assertEqual((self.result("E125-A0")["status"], self.result("E125-A0")["completed_families"]),
                         ("interrupted", 1))

    def test_inconsistent_intersection_gate_is_not_reported_as_complete(self):
        self.passed, self.bad_gate = False, True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["status"], "failed")
        self.assertNotIn("gate", self.result())

    def test_partial_report_suppresses_gate_and_fair_ladder_claim(self):
        self.invoke()
        folder = self.folder()
        result = self.result()
        result.update(status="failed", gate={field: True for field in entry.GATE_FIELDS}, gate_status="passed")
        (folder / "summary.json").write_text(json.dumps(result))
        (self.root / "results").mkdir()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        report = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        row = next(line for line in report.splitlines() if folder.name in line)
        self.assertIn("undecided", row)
        self.assertNotIn("PASS", row)
        self.assertIn("formal rung", report)


if __name__ == "__main__":
    unittest.main()
