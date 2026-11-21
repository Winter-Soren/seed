"""E124 source isolation, immutable chain evidence and separate decisions."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_social_transmission as entry, run_experiment as runner, summarize
from seed.seed import Cfg


class SocialTransmissionRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="transmission-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pins = {}
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
            self.calibration(source, "final")
        self.calibration(entry.SOURCES[0], "development")
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E124-benchmark", "--seed", "2026120200", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.summarized_rows = [], []
        self.fail_at, self.tamper, self.expire = None, False, False
        self.bad_key, self.bad_lambda, self.nonfinite, self.reuse_keys = False, False, False, False
        self.retained, self.improved, self.bad_gate = True, False, False
        self.now = 0.

    def calibration(self, source, split):
        name = "E123-benchmark" if split == "development" else "E123-" + source["label"]
        parent = self.root / "runs" / ("20261121-000000_" + name)
        (parent / "calibration").mkdir(parents=True)
        count, seed = (4, 51) if split == "development" else (32, 2026113001 + source["training_seed"])
        artifacts = []
        for index in range(count):
            path = parent / f"calibration/family_{index:04d}.npz"
            keys = np.asarray([[seed, index * 4 + role] for role in range(4)], np.uint32)
            np.savez_compressed(path, keys=keys)
            artifacts.append(dict(index=index, path=path.relative_to(parent).as_posix(),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selection = dict(families=count, contacts=count, status="frozen", calibration_seed=seed,
            completed_families=count, expected_families=count, selected_lambda=1.,
            contact_coverage=dict(mean=1., se=0.), vote_coverage=dict(mean=1., se=0.), raw_files=artifacts,
            losses=[dict(lambda_value=value, mean_log_loss=(value - 1.) ** 2 + .2) for value in (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)])
        for loss in selection["losses"]:
            loss["lambda"] = loss.pop("lambda_value")
        config = dict(protocol="social-belief-readout-v1", condition_split=split, source=source,
            calibration_seed=seed, expected_calibration_families=count, calibration_worlds=count,
            cfg=Cfg(M=1)._asdict(), generation=0,
            args=dict(name=name, eval_protocol="social-belief-readout-v1", condition_split=split),
            git=dict(commit="b" * 40, dirty=False))
        (parent / "config.json").write_text(json.dumps(config))
        (parent / "calibration/selection.json").write_text(json.dumps(selection))
        # Access to prior outcomes is deliberately unusable; only frozen training records are sources.
        (parent / "summary.json").write_text("not a readable result")
        (parent / "families").mkdir()
        (parent / "families/family_0000.npz").write_bytes(b"not evaluation evidence")
        self.pins[name] = dict(folder=parent.relative_to(self.root).as_posix(), commit="b" * 40,
            config_sha256=hashlib.sha256((parent / "config.json").read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256((parent / "calibration/selection.json").read_bytes()).hexdigest())

    def keys(self, master, index):
        keys = np.asarray([[master, index * 8 + role] for role in range(8)], np.uint32).reshape(4, 2, 2)
        if self.reuse_keys:
            keys[0, 0] = np.asarray([51, 0], np.uint32)
        return keys

    def evaluate_chain(self, flat, keys, index, selected_lambda, c):
        self.assertEqual((flat.shape, flat.dtype), ((155,), np.dtype("float32")))
        self.assertEqual(c, Cfg(M=1))
        self.assertEqual(selected_lambda, 1.)
        if index == self.fail_at:
            raise RuntimeError("deliberate chain interruption")
        if self.expire:
            self.now = 3601.
        self.calls.append((int(keys[0, 0, 0]), index))
        if self.tamper and index == 0:
            (self.root / self.pins["E123-benchmark"]["folder"] / "calibration/selection.json").write_text("{}")
        return dict(keys=keys + 1 if self.bad_key else keys.copy(), chain_index=np.asarray(index),
            original_genome=flat.copy(),
            rule=np.asarray(index % 2), yoked_swap=np.asarray((index // 2) % 2),
            lambda_value=np.asarray(selected_lambda + 1 if self.bad_lambda else selected_lambda),
            score=np.asarray(np.nan if self.nonfinite else .75), large_evidence=np.zeros((12, 100), np.float32))

    @staticmethod
    def summary_input(row):
        return {field: row[field] for field in ("chain_index", "score")}

    def measurements(self, rows):
        self.summarized_rows = rows
        self.assertTrue(all(set(row) == {"chain_index", "score"} for row in rows))
        arm = {metric: dict(mean=.75, se=.05) for metric in
               ("first_food", "no_meal", "survival", "fitness", "first_meal_step", "decoder_coverage", "accuracy")}
        delta = dict(mean=.2, se=.05, lower_2se=.1, upper_2se=.3)
        contrast = dict(first_food_difference=delta, survival_difference=delta)
        cohort = dict(arms={name: arm for name in entry.ARMS},
                      contrasts={name: contrast for name in entry.CONTROLS})
        return dict(chains=len(rows), cohorts={str(i): cohort for i in (1, 2, 3)},
            cross_cohort=dict(c3_vs_c1=contrast, c3_vs_repeated_founder=contrast),
            retained_efficacy=dict(passed=self.retained, first_food_passed=self.retained, survival_passed=True),
            cumulative_improvement=dict(passed=True if self.bad_gate else self.retained and self.improved,
                                        first_food_passed=self.improved, survival_passed=True),
            gate=dict(passed=self.retained))

    def invoke(self, *extra, dirty=False, common=True, timed=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry, "CALIBRATION_SOURCES", self.pins))
            stack.enter_context(patch.object(entry, "_runtime", return_value={"test": "mock backend"}))
            for name, method in (("chain_keys", self.keys), ("evaluate_chain", self.evaluate_chain),
                                 ("summary_input", self.summary_input), ("summarize", self.measurements)):
                stack.enter_context(patch.object(entry.social_transmission, name, side_effect=method))
            stack.enter_context(patch.object(entry.social_transmission, "validate_chain"))
            if timed:
                stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        source = entry.SOURCES[index]
        return ["--condition-split", "final", "--name", "E124-" + source["label"],
                "--seed", str(2026120201 + index), "--eval-worlds", "128", "--start", "file:" + source["path"]]

    def folder(self, name="E124-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E124-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_benchmark_reuses_only_calibration_and_streams_immutable_raw_chains(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.calls, [(2026120200, index) for index in range(4)])
        self.assertEqual((config["max_seconds"], config["cohorts"], config["selected_lambda"]), (900, 3, 1.))
        self.assertEqual(config["arms"], list(entry.ARMS))
        self.assertEqual(result["completed_chains"], 4)
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["retained_efficacy_status"], "not_applicable")
        self.assertEqual(result["cumulative_improvement_status"], "not_applicable")
        self.assertNotIn("retained_efficacy", result)
        self.assertNotIn("cumulative_improvement", result)
        self.assertNotIn("gate", result)
        self.assertEqual(len(result["chain_seconds"]), 4)
        source_selection = self.root / config["calibration_source"]["selection"]["path"]
        frozen = folder / result["frozen_calibration"]["path"]
        self.assertEqual(frozen.read_bytes(), source_selection.read_bytes())
        for artifact in result["raw_files"]:
            path = folder / artifact["path"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["calibration_sha256"].tobytes().hex(), result["frozen_calibration"]["sha256"])
                self.assertIn("large_evidence", arrays)
        self.assertTrue(all("large_evidence" not in row for row in self.summarized_rows))

    def test_three_final_sources_keep_retention_and_improvement_separate(self):
        for index, source in enumerate(entry.SOURCES):
            self.calls.clear()
            self.invoke(*self.final_args(index))
            result = self.result("E124-" + source["label"])
            self.assertEqual(self.calls, [(2026120201 + index, chain) for chain in range(128)])
            self.assertEqual(result["calibration_source"]["name"], "E123-" + source["label"])
            self.assertEqual(result["completed_chains"], 128)
            self.assertTrue(result["scientific_result"])
            self.assertEqual(result["retained_efficacy_status"], "passed")
            self.assertEqual(result["cumulative_improvement_status"], "failed")
            self.assertTrue(result["retained_efficacy"]["passed"])
            self.assertFalse(result["cumulative_improvement"]["passed"])

    def test_unregistered_flags_dirty_state_and_missing_pins_fail_before_output(self):
        for flags in [("--seed", "51"), ("--eval-worlds", "128"), ("--generations", "1"),
                      ("--pop", "16"), ("--cfg", "M=2"), ("--persistence",), ("--start", "biology"),
                      ("--resume", "old"), ("--allow-dirty",)]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.pins.clear()
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E124-*")))

    def test_source_hash_tampering_fails_before_output(self):
        parent = self.root / self.pins["E123-benchmark"]["folder"]
        path = parent / "calibration/family_0000.npz"
        path.write_bytes(path.read_bytes() + b"altered")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)
        self.assertFalse(list((self.root / "runs").glob("*_E124-*")))

    def test_prior_evaluation_path_cannot_be_substituted_for_calibration(self):
        pin = self.pins["E123-benchmark"]
        path = self.root / pin["folder"] / "calibration/selection.json"
        selection = json.loads(path.read_text())
        selection["raw_files"][0]["path"] = "families/family_0000.npz"
        path.write_text(json.dumps(selection))
        pin["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)

    def test_existing_attempt_and_generic_resume_leave_all_bytes_intact(self):
        self.invoke()
        before = self.snapshot(self.folder())
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(before, self.snapshot(self.folder()))

    def test_interruption_keeps_full_prefix_without_scientific_decisions(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_chains"]), ("failed", 2))
        self.assertEqual(len(result["raw_files"]), 2)
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["retained_efficacy_status"], "undecided")
        self.assertEqual(result["cumulative_improvement_status"], "undecided")
        self.assertNotIn("measurements", result)

    def test_midrun_calibration_tampering_is_preserved_as_failure(self):
        self.tamper = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual((self.result()["status"], self.result()["completed_chains"]), ("failed", 1))

    def test_key_reuse_stops_before_the_first_chain(self):
        self.reuse_keys = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)
        self.assertEqual(self.result()["completed_chains"], 0)

    def test_nonfinite_evidence_is_not_saved(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_chains"], 0)
        self.assertFalse(list((self.folder() / "chains").glob("*.npz")))

    def test_wrong_frozen_lambda_is_rejected(self):
        self.bad_lambda = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_chains"], 0)

    def test_invalid_improvement_verdict_cannot_override_retention(self):
        self.retained, self.bad_gate = False, True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertNotIn("measurements", self.result())
        self.assertEqual(self.result()["retained_efficacy_status"], "undecided")

    def test_hard_wall_cap_preserves_completed_chain_and_no_partial_result(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke(timed=True)
        result = self.result()
        self.assertEqual((result["status"], result["completed_chains"]), ("interrupted", 1))
        self.assertNotIn("measurements", result)

    def test_result_table_reports_both_decisions_and_all_cohorts(self):
        self.invoke(*self.final_args(0))
        (self.root / "results").mkdir()
        for pin in self.pins.values():
            (self.root / pin["folder"] / "summary.json").write_text("{}")
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("social-transmission-v1", table)
        self.assertIn("Retained efficacy", table)
        self.assertIn("Cumulative improvement", table)
        self.assertIn("| PASS | FAIL |", table)
        self.assertIn("c3 versus c1", table)
        self.assertIn("lower two-SE", table)
        for cohort in (1, 2, 3):
            for arm in entry.ARMS:
                self.assertIn(f"| {cohort} | {arm} |", table)

    def test_result_table_suppresses_partial_gates_and_measurements(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke(*self.final_args(0))
        path = self.folder("E124-A0") / "summary.json"
        result = json.loads(path.read_text())
        result.update(scientific_result=True, retained_efficacy=dict(passed=True, first_food_passed=True, survival_passed=True),
                      cumulative_improvement=dict(passed=True, first_food_passed=True, survival_passed=True))
        path.write_text(json.dumps(result))
        for pin in self.pins.values():
            (self.root / pin["folder"] / "summary.json").write_text("{}")
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("| undecided | undecided |", table)
        self.assertNotIn("| PASS | PASS |", table)
        self.assertNotIn("| 1 | predecessor |", table)


if __name__ == "__main__":
    unittest.main()
