"""E128 independent-family math, immutable evidence and audited reporting."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_retention_diagnostic as entry, run_experiment as runner, summarize


def compact_family(index, *, delayed_loss=False):
    row = dict(base_index=np.asarray(index))
    for arm in entry.ARMS:
        reversal = arm.endswith("reversal")
        local = np.tile([.6, .3, .9, .1, .3] if reversal else [.2, .6, .8, .2, -.4], (2, 3, 1))
        grid = np.tile([1., 0., 0., 1., 1., 1., 1.] if reversal else [0., 1., 0., -1., 1., -1., 1.], (2, 3, 1))
        if delayed_loss and arm == "plastic_reversal":
            local[:, 2, 4] = .19
        row[arm + "_pre_old_by_rule"] = np.full(2, .6)
        row[arm + "_pre_local_by_rule"] = np.tile([.1, .7, .8, .2, -.6], (2, 1))
        row[arm + "_local_by_rule"] = local
        row[arm + "_grid_by_rule"] = grid
    return row


class RetentionStatisticsTests(unittest.TestCase):
    def test_rule_pairing_preserves_covariance_in_retention_changes(self):
        rows = [compact_family(i) for i in range(4)]
        for row, offset in zip(rows, (.125, .25, .0625, .1875)):
            local = row["plastic_reversal_local_by_rule"]
            local[:, 0, 4] = [.5-offset, .5+offset]
            local[:, 2, 4] = [.5+offset, .5-offset]
            grid = row["plastic_reversal_grid_by_rule"]
            grid[:, 0, 0] = [0., 1.]
            grid[:, 2, 0] = [1., 0.]
        result = entry.summarize(rows)
        for endpoint in ("local", "grid"):
            self.assertEqual(result["architectures"]["plastic"][endpoint]["change_33_minus_0"],
                             dict(mean=0., se=0., n=4, lower_2se=0.))
        self.assertEqual(result["arms"]["plastic_reversal"]["by_delay"]["33"]["local"]["S"]["n"], 4)
        self.assertGreater(result["by_initial_rule"]["0"]["plastic"]["by_delay"]["33"]["S"]["se"], 0)

    def test_sample_se_and_strict_evidence_boundaries(self):
        self.assertAlmostEqual(entry._estimate([0., 1., 1., 0.])["se"], np.sqrt(1./3.) / 2.)
        self.assertTrue(entry._advantage(dict(mean=.1, se=0.), .1))
        self.assertFalse(entry._advantage(dict(mean=.1, se=.05), .1))
        rows = [compact_family(i) for i in range(4)]
        for row in rows:
            row["plastic_reversal_local_by_rule"][:, 0, 4] = 0.
            row["plastic_reversal_local_by_rule"][:, 2, 4] = -.05
        local = entry.summarize(rows)["architectures"]["plastic"]["local"]
        self.assertEqual(local["change_33_minus_0"]["lower_2se"], -.05)
        self.assertFalse(local["criteria"]["retention"])
        self.assertIn("retention", local["failed_criteria"])

    def test_one_turn_snapshot_and_old_preference_are_descriptive(self):
        rows = [compact_family(i) for i in range(4)]
        for row in rows:
            for arm in entry.ARMS:
                row[arm + "_pre_old_by_rule"][:] = -1.
                row[arm + "_local_by_rule"][:, 1, 4] = -1.
                row[arm + "_grid_by_rule"][:, 1, 0] = 0.
                row[arm + "_grid_by_rule"][:, 1, 3] = -1.
        result = entry.summarize(rows)
        self.assertTrue(result["architectures"]["plastic"]["combined_passed"])
        self.assertTrue(result["architectures"]["direct"]["combined_passed"])

    def test_three_lineage_replication_and_same_lineage_intersection(self):
        measurements = entry.summarize([compact_family(i) for i in range(4)])
        panels = [dict(index=i, measurements=copy.deepcopy(measurements)) for i in range(3)]
        for i, panel in enumerate(panels):
            values = panel["measurements"]["architectures"]["plastic"]
            values["local"]["passed"], values["grid"]["passed"] = i in (0, 1), i in (1, 2)
            values["combined_passed"] = i == 1
        gates = entry._replicated_gates(panels)
        self.assertEqual(gates["plastic"]["local"], dict(count=2, seeds=[0, 1], passed=False))
        self.assertEqual(gates["plastic"]["combined"], dict(count=1, seeds=[1], passed=False))
        self.assertTrue(gates["direct"]["combined"]["passed"])
        panels[0]["measurements"]["architectures"]["plastic"]["combined_passed"] = True
        with self.assertRaisesRegex(ValueError, "same lineage"):
            entry._replicated_gates(panels)

    def test_invalid_compact_shape_nonfinite_or_missing_family_is_rejected(self):
        rows = [compact_family(i) for i in range(4)]
        for invalid in (rows[:-1] + [rows[0]], [rows[0]]):
            with self.assertRaises(ValueError):
                entry.summarize(invalid)
        rows[0]["plastic_reversal_local_by_rule"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            entry.summarize(rows)


class RetentionDiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="retention-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E128-benchmark",
                       "--seed", "2026120700", "--generations", "0", "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.projected = [], []
        self.fail_at, self.reuse, self.bad_source, self.nonfinite, self.expire = None, False, False, False, False
        self.now = 0.

    @staticmethod
    def assignments(master, count):
        return dict(key=np.asarray([master, 1000000], np.uint32), permutation=np.arange(count, dtype=np.int32),
                    pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2))

    def keys(self, master, index):
        index = 0 if self.reuse else index
        return np.asarray([[master, 7*index+role] for role in range(7)], np.uint32)

    def evaluate(self, plastic, fixed, keys, index, pre_contacts, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate retention interruption")
        if self.expire:
            self.now = 121.
        self.calls.append((int(keys[0, 0]), index, pre_contacts))
        row = compact_family(index, delayed_loss=int(keys[0, 0]) == 2026120703)
        row.update(keys=keys.copy(), pre_contacts=np.asarray(pre_contacts), initial_rules=np.asarray([0, 1]),
            retention_ticks=np.asarray(entry.RETENTION_TICKS), plastic_genome_sha256=entry._array_sha(plastic),
            direct_genome_sha256=np.zeros(32, np.uint8) if self.bad_source else entry._array_sha(fixed),
            large_trace=np.full((2, 33, 10), np.nan if self.nonfinite else 0., np.float32))
        return row

    def projection(self, row):
        compact = {name: value for name, value in row.items() if name != "large_trace"}
        self.projected.append(compact)
        return compact

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a"*40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b"*64}))
            stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys),
                                   ("evaluate_world", self.evaluate), ("summary_input", self.projection)):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        return ["--condition-split", "final", "--name", f"E128-A{index}", "--seed", str(2026120701+index),
                "--eval-worlds", "128", "--start", "file:" + entry.SOURCES[index]["path"]]

    def folder(self, name="E128-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E128-benchmark"):
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
            folder, result = self.folder(f"E128-A{index}"), self.result(f"E128-A{index}")
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

    def test_development_records_pinned_inputs_all_delays_and_full_evidence(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.calls, [(2026120700, 0, 8), (2026120700, 1, 8), (2026120700, 2, 16), (2026120700, 3, 16)])
        self.assertEqual((config["retention_ticks"], config["key_roles"]), (list(entry.RETENTION_TICKS), list(entry.KEY_ROLES)))
        self.assertEqual((config["assignment_stream_tag"], config["max_seconds"]), (0xE128, 120))
        self.assertEqual((result["status"], result["gate_status"]), ("finished", "not_applicable"))
        self.assertFalse(result["scientific_result"])
        self.assertTrue(all("large_trace" not in row for row in self.projected))
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((folder / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        with np.load(folder / "families/family_0000.npz") as data:
            self.assertIn("large_trace", data)

    def test_partial_and_unaudited_results_stay_locked(self):
        for index in range(3):
            self.invoke(*self.final_args(index))
            self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        self.write_audit()
        result = entry.build_comparison(self.root)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["replicated_gates"]["plastic"]["local"], dict(count=2, seeds=[0, 1], passed=False))
        self.assertTrue(result["replicated_gates"]["direct"]["combined"]["passed"])

    def test_prior_e127_outcomes_are_not_inputs(self):
        folder = self.root / "runs" / "unused_E127-A0-recovery1"
        folder.mkdir()
        (folder / "config.json").write_text("forbidden previous experiment")
        (folder / "summary.json").write_text("forbidden held-out outcome")
        self.invoke()
        self.assertEqual(self.result()["completed_families"], 4)

    def test_overrides_dirty_inputs_and_unregistered_recovery_names_fail_before_creation(self):
        for flags in (("--seed", "2026120600"), ("--generations", "1"), ("--eval-worlds", "8"),
                      ("--start", "biology"), ("--cfg", "T=300"), ("--pop", "16"), ("--allow-dirty",),
                      ("--resume", "unused"), ("--name", "E128-benchmark-retry1"), ("--name", "E128-A0-recovery1")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(self.calls)
        self.assertFalse(list((self.root / "runs").glob("*_E128-*")))

    def test_changed_source_or_duplicate_attempt_and_generic_resume_are_rejected(self):
        path = self.root / entry.DIRECT_SOURCES[0]["path"]
        before = path.read_bytes()
        path.write_bytes(before + b"tampered")
        with self.assertRaises(SystemExit):
            self.invoke()
        path.write_bytes(before)
        self.invoke()
        snapshot = self.snapshot(self.folder())
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(snapshot, self.snapshot(self.folder()))

    def test_interruption_keeps_prefix_without_scientific_result(self):
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertEqual(len(result["family_wall_seconds"]), 2)
        self.assertNotIn("measurements", result)

    def test_duplicate_keys_and_invalid_source_evidence_fail_closed(self):
        self.reuse = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 1)

    def test_nonfinite_evidence_is_not_committed_as_complete_family(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_raw_source_hash_must_match_original_controller(self):
        self.bad_source = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_absolute_deadline_preserves_saved_prefix(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("interrupted", 1))
        self.assertNotIn("measurements", result)

    def test_wall_timing_includes_persistence_and_matches_actual_boundary_order(self):
        schedule = self.assignments(2026120700, 4)
        schedule["permutation"] = np.asarray([2, 0, 3, 1], np.int32)
        schedule["pre_contacts"] = np.asarray([16, 8, 16, 8], np.int32)
        self.assignments = lambda master, count: schedule
        writer, fsync = entry._write_status, entry.os.fsync

        def timed_writer(*args, **kwargs):
            writer(*args, **kwargs)
            self.now += .5

        def timed_sync(*args, **kwargs):
            fsync(*args, **kwargs)
            self.now += .25

        with patch.object(entry, "_write_status", side_effect=timed_writer), patch.object(entry.os, "fsync", side_effect=timed_sync):
            self.invoke()
        result = self.result()
        self.assertEqual(result["family_pre_contacts"], [16, 8, 16, 8])
        self.assertEqual(len(result["family_wall_seconds"]), 4)
        self.assertTrue(all(wall >= 0 and wall >= inner+.75 for wall, inner in zip(result["family_wall_seconds"], result["family_seconds"])))

    def test_audit_selection_gate_disagreement_and_stale_artifacts_are_rejected(self):
        self.finish_panels()
        path = self.write_audit()
        original = path.read_bytes()
        report = json.loads(original)
        report["panels"][0]["run"] = "runs/wrong_E128-A0"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "provenance"):
            entry.build_comparison(self.root)
        report = json.loads(original)
        report["replicated_gates"]["plastic"]["local"]["passed"] = True
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "disagree"):
            entry.build_comparison(self.root)
        path.write_bytes(original)
        raw = self.folder("E128-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes() + b"changed after audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)

    def test_report_publishes_measurements_only_after_complete_independent_verification(self):
        self.finish_panels()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        path = self.root / "results/RESULTS.md"
        self.assertIn("Replicated retention decisions: **undecided**", path.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "results/diagnostics/E128-comparison.json").exists())
        self.write_audit()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        text = path.read_text(encoding="utf-8")
        self.assertIn("| plastic | local | 2/3 | 0, 1 | FAIL |", text)
        self.assertIn("Lower two-SE bound", text)
        self.assertIn("n=128", text)
        self.assertTrue((self.root / "results/diagnostics/E128-comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
