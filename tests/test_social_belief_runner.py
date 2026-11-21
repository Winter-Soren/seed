"""E123 runner isolation, frozen calibration, immutable evidence and result boundaries."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_social_belief as entry, run_experiment as runner, summarize
from seed.seed import Cfg


class SocialBeliefRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="belief-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E123-benchmark", "--seed", "51", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calibration_calls, self.evaluation_calls = [], []
        self.fail_phase, self.fail_at, self.expire_phase = None, None, None
        self.bad_key, self.bad_lambda, self.nonfinite, self.bad_selection = False, False, False, False
        self.reuse_keys, self.tamper, self.no_contacts = False, False, False
        self.now = 0.
        self.selected = None

    def keys(self, master, index):
        return np.asarray([[51 if self.reuse_keys else master, index * 4 + role]
                           for role in range(4)], np.uint32)

    def raw(self, flat, keys, index, c, phase):
        self.assertEqual((flat.shape, flat.dtype), ((155,), np.dtype("float32")))
        self.assertEqual(c, Cfg(M=1))
        if self.fail_phase == phase and index == self.fail_at:
            raise RuntimeError("deliberate stage interruption")
        if self.expire_phase == phase:
            self.now = 901. if phase == "calibration" else 3601.
        return dict(keys=keys + 1 if self.bad_key else keys.copy(), family_index=np.asarray(index),
                    rule=np.asarray(index % 2), yoked_swap=np.asarray((index // 2) % 2),
                    observation=np.asarray(np.nan if self.nonfinite else index, np.float32),
                    z=np.asarray(.5), votes=np.asarray([3, 1]), contact_color=np.asarray(0),
                    contact_outcome=np.asarray(1))

    def calibration_family(self, flat, keys, index, c):
        row = self.raw(flat, keys, index, c, "calibration")
        if self.no_contacts:
            row.update(contact_color=np.asarray(-1), contact_outcome=np.asarray(0))
        self.calibration_calls.append((int(keys[0, 0]), index))
        return row

    @staticmethod
    def calibration_input(row):
        return {field: row[field] for field in ("z", "votes", "contact_color", "contact_outcome")}

    def calibrate(self, rows):
        self.assertFalse(self.evaluation_calls)
        self.assertTrue(all(set(row) == {"z", "votes", "contact_color", "contact_outcome"} for row in rows))
        contacts = 0 if self.no_contacts else len(rows)
        selected = 0. if self.no_contacts else 1.
        self.selected = selected
        return dict(families=len(rows), contacts=contacts,
            selected_lambda=2. if self.bad_selection else selected,
            contact_coverage=dict(mean=contacts / len(rows), se=0.), vote_coverage=dict(mean=1., se=0.),
            losses=[{"lambda": value, "mean_log_loss": None if self.no_contacts else (value - 1.) ** 2 + .4,
                     "se_log_loss": None if self.no_contacts else .01} for value in entry.LAMBDA_GRID])

    def evaluate_family(self, flat, keys, index, selected_lambda, c):
        row = self.raw(flat, keys, index, c, "evaluation")
        master = int(keys[0, 0])
        name = "E123-benchmark" if master == 52 else "E123-A" + str(master - 2026120101)
        folder = self.folder(name)
        frozen = folder / "calibration" / "selection.json"
        selection = json.loads(frozen.read_text())
        self.assertEqual((selection["status"], selection["selected_lambda"]), ("frozen", self.selected))
        self.assertEqual(selected_lambda, self.selected)
        records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        witnesses = [record for record in records if record["phase"] == "calibration_frozen"]
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(witnesses[0]["sha256"], hashlib.sha256(frozen.read_bytes()).hexdigest())
        self.assertEqual(len(selection["raw_files"]), len(self.calibration_calls))
        row["lambda_value"] = np.asarray(selected_lambda + 1 if self.bad_lambda else selected_lambda)
        if self.tamper and index == 0:
            frozen.write_text("{}")
        self.evaluation_calls.append((int(keys[0, 0]), index))
        return row

    @staticmethod
    def measurements(rows):
        gate = dict(passed=True, first_food_passed=True, survival_passed=True)
        arm = {metric: dict(mean=.75, se=.05) for metric in
               ("first_food", "no_meal", "survival", "fitness", "first_meal_step")}
        contrast = dict(first_food_difference=dict(mean=.2, se=.05),
                        survival_difference=dict(mean=0., se=.1), gate=gate)
        return dict(families=len(rows), assigned_newborns=len(rows),
                    arms={name: arm for name in entry.ARMS},
                    contrasts={name: contrast for name in entry.CONTROLS}, gate=gate)

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_runtime", return_value={"test": "mock backend"}))
            for name in ("family_keys", "calibration_family", "calibration_input", "calibrate", "evaluate_family", "summarize"):
                mock = {"family_keys": self.keys, "summarize": self.measurements}.get(name, getattr(self, name, None))
                stack.enter_context(patch.object(entry.social_belief, name, side_effect=mock))
            stack.enter_context(patch.object(entry.social_belief, "validate_calibration_family"))
            stack.enter_context(patch.object(entry.social_belief, "validate_family"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        source = entry.SOURCES[index]
        return ["--condition-split", "final", "--name", "E123-" + source["label"],
                "--seed", str(2026113001 + index), "--eval-worlds", "128", "--start", "file:" + source["path"]]

    def folder(self, name="E123-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_benchmark_freezes_calibration_before_disjoint_six_arm_evaluation(self):
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(self.calibration_calls, [(51, index) for index in range(4)])
        self.assertEqual(self.evaluation_calls, [(52, index) for index in range(4)])
        self.assertEqual((config["calibration_seed"], config["eval_seed"]), (51, 52))
        self.assertEqual((config["calibration_max_seconds"], config["max_seconds"]), (900, 900))
        self.assertEqual(config["cfg"], Cfg(M=1)._asdict())
        self.assertEqual(config["arms"], list(entry.ARMS))
        self.assertEqual(config["controls"], list(entry.CONTROLS))
        self.assertEqual(result["status"], "finished")
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertNotIn("gate", result)
        self.assertEqual(len(result["calibration"]["losses"]), 9)
        frozen = result["frozen_calibration"]
        self.assertEqual(frozen["sha256"], hashlib.sha256((folder / frozen["path"]).read_bytes()).hexdigest())
        records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["phase"] for row in records], ["calibration"] * 4 + ["calibration_frozen"] + ["evaluation"] * 4)
        for row in records[5:]:
            self.assertEqual(row["calibration_sha256"], frozen["sha256"])
        for artifact in result["calibration_raw_files"] + result["raw_files"]:
            path = folder / artifact["path"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for artifact in result["raw_files"]:
            with np.load(folder / artifact["path"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["calibration_sha256"].tobytes().hex(), frozen["sha256"])
        self.assertEqual((folder / "genome_start.npy").read_bytes(),
                         (self.root / entry.SOURCES[0]["path"]).read_bytes())

    def test_three_final_identities_have_separate_calibration_and_complete_gates(self):
        for index, source in enumerate(entry.SOURCES):
            self.calibration_calls.clear()
            self.evaluation_calls.clear()
            self.invoke(*self.final_args(index))
            folder = self.folder("E123-" + source["label"])
            result = json.loads((folder / "summary.json").read_text())
            self.assertEqual(self.calibration_calls, [(2026113001 + index, family) for family in range(32)])
            self.assertEqual(self.evaluation_calls, [(2026120101 + index, family) for family in range(128)])
            self.assertEqual((result["source_label"], result["completed_calibration_families"], result["completed_families"]),
                             (source["label"], 32, 128))
            self.assertEqual(result["gate"], dict(passed=True, first_food_passed=True, survival_passed=True))
            self.assertTrue(result["scientific_result"])

    def test_unregistered_flags_dirty_state_and_altered_source_fail_before_creation(self):
        bad = [("--seed", "52"), ("--eval-worlds", "32"), ("--generations", "1"),
               ("--pop", "16"), ("--cfg", "M=2"), ("--persistence",), ("--start", "biology"),
               ("--resume", "old"), ("--allow-dirty",), ("--calibration-seed", "51")]
        for flags in bad:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        source = self.root / entry.SOURCES[0]["path"]
        source.write_bytes(source.read_bytes() + b"altered")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(list((self.root / "runs").glob("*_E123-*")))

    def test_existing_attempt_and_generic_resume_preserve_all_saved_bytes(self):
        self.invoke()
        folder = self.folder()
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), common=False)
        self.assertEqual(before, self.snapshot(folder))

    def test_calibration_failure_retains_prefix_and_never_opens_evaluation(self):
        self.fail_phase, self.fail_at = "calibration", 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_calibration_families"], result["completed_families"]),
                         ("failed", 2, 0))
        self.assertEqual(len(result["calibration_raw_files"]), 2)
        self.assertIsNone(result["calibration"])
        self.assertFalse(self.evaluation_calls)
        self.assertNotIn("measurements", result)

    def test_invalid_selection_stops_before_freeze_or_evaluation(self):
        self.bad_selection = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual(result["completed_calibration_families"], 4)
        self.assertFalse((self.folder() / "calibration" / "selection.json").exists())
        self.assertFalse(self.evaluation_calls)

    def test_key_reuse_cannot_open_an_evaluation_world(self):
        self.reuse_keys = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertIsNotNone(result["frozen_calibration"])
        self.assertEqual(result["completed_families"], 0)
        self.assertFalse(self.evaluation_calls)

    def test_changed_frozen_selection_stops_after_preserving_evaluated_prefix(self):
        self.tamper = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 1))
        self.assertNotIn("measurements", result)
        self.assertEqual(result["gate_status"], "undecided")

    def test_calibration_subcap_stops_before_evaluation(self):
        self.expire_phase = "calibration"
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now), self.assertRaises(TimeoutError):
            self.invoke(*self.final_args(0))
        result = json.loads((self.folder("E123-A0") / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_calibration_families"]), ("interrupted", 1))
        self.assertFalse(self.evaluation_calls)

    def test_full_invocation_cap_preserves_frozen_calibration_and_final_prefix(self):
        self.expire_phase = "evaluation"
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now), self.assertRaises(TimeoutError):
            self.invoke(*self.final_args(0))
        result = json.loads((self.folder("E123-A0") / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_calibration_families"], result["completed_families"]),
                         ("interrupted", 32, 1))
        self.assertIsNotNone(result["frozen_calibration"])
        self.assertNotIn("measurements", result)

    def test_no_contact_fitting_retains_zero_lambda_and_null_losses(self):
        self.no_contacts = True
        self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual(result["calibration"]["selected_lambda"], 0.)
        self.assertTrue(all(loss["mean_log_loss"] is None for loss in result["calibration"]["losses"]))

    def test_bad_assignments_nonfinite_data_and_wrong_frozen_lambda_are_not_saved(self):
        for flag in ("bad_key", "nonfinite", "bad_lambda"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(prefix="belief-invalid-", dir=runner.ROOT / ".tmp") as temp:
                invalid_root = Path(temp)
                source = invalid_root / entry.SOURCES[0]["path"]
                source.parent.mkdir(parents=True)
                source.write_bytes((self.root / entry.SOURCES[0]["path"]).read_bytes())
                setattr(self, flag, True)
                self.calibration_calls.clear()
                self.evaluation_calls.clear()
                with patch.object(self, "root", invalid_root), self.assertRaises(SystemExit):
                    self.invoke()
                folder = next((invalid_root / "runs").glob("*_E123-*"))
                self.assertFalse(list((folder / "families").glob("*.npz")))
                if flag != "bad_lambda":
                    self.assertFalse(list((folder / "calibration").glob("*.npz")))
                setattr(self, flag, False)

    def test_partial_summary_keeps_calibration_separate_from_a_scientific_gate(self):
        self.invoke()
        folder = self.folder()
        path = folder / "summary.json"
        result = json.loads(path.read_text())
        result.update(status="failed", gate_status="passed", gate=dict(passed=True))
        path.write_text(json.dumps(result))
        (self.root / "results").mkdir()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        report = (self.root / "results" / "RESULTS.md").read_text(encoding="utf-8")
        gate_row = next(line for line in report.splitlines() if folder.name in line)
        self.assertIn("undecided", gate_row)
        self.assertNotIn("PASS", gate_row)
        self.assertIn("Calibration diagnostics", report)
        self.assertNotIn("Held-out family outcomes", report)


if __name__ == "__main__":
    unittest.main()
