"""E127 source isolation, immutable paired evidence and independently audited replication."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_encounter_diagnostic as entry, run_experiment as runner, summarize


class EncounterDiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="encounter-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E127-benchmark", "--seed", "2026120600", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.projected = [], []
        self.fail_at, self.reuse, self.bad_source, self.nonfinite, self.expire = None, False, False, False, False
        self.now = 0.

    @staticmethod
    def assignments(master, count):
        return dict(key=np.asarray([master, 1000000], np.uint32), permutation=np.arange(count, dtype=np.int32),
                    pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2))

    def keys(self, master, index):
        index = 0 if self.reuse else index
        return np.asarray([[master, index * 6 + role] for role in range(6)], np.uint32)

    def evaluate(self, plastic, direct, keys, index, pre_contacts, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate diagnostic interruption")
        if self.expire:
            self.now = 181.
        self.calls.append((int(keys[0, 0]), index, pre_contacts))
        return dict(keys=keys.copy(), base_index=np.asarray(index), pre_contacts=np.asarray(pre_contacts),
            initial_rules=np.asarray([0, 1]), plastic_genome_sha256=entry._array_sha(plastic),
            direct_genome_sha256=np.zeros(32, np.uint8) if self.bad_source else entry._array_sha(direct),
            outcome=np.asarray(np.nan if self.nonfinite else .5), large_trace=np.zeros((2, 32, 50), np.float32))

    def projection(self, row):
        compact = {key: row[key] for key in ("keys", "base_index", "outcome")}
        self.projected.append(compact)
        return compact

    @staticmethod
    def measurements(rows):
        master = int(rows[0]["keys"][0, 0])
        index, n = max(0, master - 2026120601), len(rows)
        metric = dict(mean=.25, se=.02, n=n)
        architecture = lambda local, transfer: dict(local_pass=local, transfer_pass=transfer,
            local={name: metric for name in ("pre_old", "post_target", "reversal_minus_stationary")},
            transfer={name: metric for name in ("reversal_minus_stationary", "food_minus_poison")})
        return dict(families=n, initial_rule_replays=2*n,
            arms={arm: dict(local=[], grid={name: metric for name in
                ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")})
                  for arm in entry.ARMS},
            architectures=dict(plastic=architecture(index in (0, 2), index in (1, 2)),
                               direct=architecture(index in (0, 1), index in (0, 1))),
            hebbian=dict(hebbian_local_pass=True, local_difference=metric, transfer_difference=metric))

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b" * 64}))
            stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys),
                                   ("evaluate_world", self.evaluate), ("summary_input", self.projection),
                                   ("summarize", self.measurements)):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        return ["--condition-split", "final", "--name", f"E127-A{index}", "--seed", str(2026120601 + index),
                "--eval-worlds", "128", "--start", "file:" + entry.SOURCES[index]["path"]]

    def folder(self, name="E127-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E127-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def finish_panels(self):
        for index in range(3):
            self.invoke(*self.final_args(index))

    def write_audit(self):
        checked, panels = {}, []
        for index in range(3):
            folder = self.folder(f"E127-A{index}")
            result = self.result(f"E127-A{index}")
            panels.append(dict(index=index, run=folder.relative_to(self.root).as_posix(), measurements=result["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            checked[source["path"]] = source["sha256"]
        report = dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return path

    def test_development_uses_frozen_sources_and_saves_complete_assignments(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.calls, [(2026120600, 0, 8), (2026120600, 1, 8), (2026120600, 2, 16), (2026120600, 3, 16)])
        self.assertEqual((config["grid_steps"], config["post_contacts"], config["probe_placements"]), (50, 16, 12))
        self.assertEqual(config["arms"], list(entry.ARMS))
        self.assertEqual(config["key_roles"], list(entry.KEY_ROLES))
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertTrue(all("large_trace" not in row for row in self.projected))
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((folder / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        with np.load(folder / "families/family_0000.npz") as data:
            self.assertIn("large_trace", data)

    def test_final_panels_and_independent_audit_are_both_required(self):
        for index in range(3):
            self.invoke(*self.final_args(index))
            self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        self.write_audit()
        decision = entry.build_comparison(self.root)
        self.assertEqual(decision["status"], "complete")
        self.assertEqual(decision["replicated_gates"]["plastic"]["local"], dict(count=2, seeds=[0, 2], passed=True))
        self.assertEqual(decision["replicated_gates"]["plastic"]["transfer"], dict(count=1, seeds=[2], passed=False))
        self.assertTrue(decision["replicated_gates"]["plastic"]["hebbian_local"]["passed"])
        self.assertTrue(decision["replicated_gates"]["direct"]["transfer"]["passed"])

    def test_no_e126_data_or_calibration_is_needed_or_read(self):
        for name in ("unused_E126-plastic-s0-part001", "unused_E123-A0"):
            folder = self.root / "runs" / name
            folder.mkdir()
            (folder / "config.json").write_text("forbidden prior experiment")
            (folder / "summary.json").write_text("forbidden prior outcomes")
        self.invoke()
        self.assertEqual(self.result()["status"], "finished")

    def test_overrides_wrong_identity_and_dirty_code_fail_before_creation(self):
        for flags in (("--seed", "2026120501"), ("--generations", "1"), ("--eval-worlds", "8"),
                      ("--start", "biology"), ("--pop", "16"), ("--cfg", "T=300"), ("--allow-dirty",),
                      ("--controller", "direct-rnn-v1"), ("--resume", "unused")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(list((self.root / "runs").glob("*_E127-*")))

    def test_source_digest_tamper_blocks_before_creation(self):
        path = self.root / entry.DIRECT_SOURCES[0]["path"]
        path.write_bytes(path.read_bytes() + b"wrong source")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)

    def test_existing_attempt_and_generic_resume_preserve_evidence(self):
        self.invoke()
        before = self.snapshot(self.folder())
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(before, self.snapshot(self.folder()))

    def test_failure_keeps_prefix_without_result_or_decision(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertNotIn("measurements", result)
        self.assertEqual(result["gate_status"], "undecided")

    def test_repeated_assignment_keys_and_nonfinite_events_are_rejected(self):
        self.reuse = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 1)

    def test_nonfinite_evidence_is_rejected_before_save(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_raw_controller_digest_must_match_selected_source(self):
        self.bad_source = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_deadline_preserves_complete_family_without_partial_scientific_result(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("interrupted", 1))
        self.assertNotIn("measurements", result)

    def test_stale_audit_and_disagreeing_replication_are_rejected(self):
        self.finish_panels()
        audit_path = self.write_audit()
        report = json.loads(audit_path.read_text())
        report["replicated_gates"]["plastic"]["transfer"]["passed"] = True
        audit_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "disagree"):
            entry.build_comparison(self.root)
        self.write_audit()
        path = self.folder("E127-A1") / "families/family_0000.npz"
        path.write_bytes(path.read_bytes() + b"changed since audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)

    def test_generated_report_keeps_decisions_locked_until_verified(self):
        self.finish_panels()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        path = self.root / "results/RESULTS.md"
        self.assertIn("Replicated diagnostic decisions: **undecided**", path.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "results/diagnostics/E127-comparison.json").exists())
        self.write_audit()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = path.read_text(encoding="utf-8")
        self.assertIn("| plastic | transfer | 1/3 | 2 | FAIL |", text)
        self.assertIn("n=128", text)
        self.assertTrue((self.root / "results/diagnostics/E127-comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
