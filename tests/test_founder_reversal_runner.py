"""E130 all-assigned family statistics and immutable tracked panels."""
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

from scripts import run_founder_reversal as entry, run_experiment as runner, summarize


def compact_family(index, *, recovery=True):
    row = dict(base_index=np.asarray(index))
    for arm in entry.ARMS:
        q = np.zeros((2, 6, 4), np.int8)
        if arm.startswith("gen200"):
            q[:] = 1
            if arm.endswith("reversal") and recovery:
                q[:, 3] = -1
        row[arm + "_Q_by_rule"] = q
        row[arm + "_no_contact_by_rule"] = q == 0
        row[arm + "_alive_end_by_rule"] = np.ones_like(q, bool)
        row[arm + "_final_survival_by_rule"] = np.ones((2, 4), bool)
    return row


class FounderReversalStatisticsTests(unittest.TestCase):
    def test_full_paired_recovery_and_descriptive_initial_rule_views(self):
        result = entry.summarize([compact_family(i) for i in range(4)])
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["components"]["descendant_recovery"], dict(mean=2., se=0., n=4))
        self.assertEqual(result["components"]["recovery_interaction"], dict(mean=2., se=0., n=4))
        self.assertEqual((result["initial_rule_replays"], result["genotype_history_rollouts"], result["agents_per_rule"]), (8, 32, 4))
        self.assertEqual(set(result["by_initial_rule"]["0"]), {"arms", "components", "safeguards"})

    def test_interaction_cannot_substitute_for_descendant_recovery(self):
        rows = [compact_family(i, recovery=False) for i in range(4)]
        for row in rows:
            row["gen0_reversal_Q_by_rule"][:, 3] = 1
            row["gen0_reversal_Q_by_rule"][:, 5] = -1
            row["gen0_reversal_no_contact_by_rule"] = row["gen0_reversal_Q_by_rule"] == 0
        result = entry.summarize(rows)
        self.assertEqual(result["components"]["recovery_interaction"]["mean"], 2.)
        self.assertFalse(result["gate"]["passed"])
        self.assertIn("descendant_recovers", result["gate"]["failed_criteria"])

    def test_standard_error_uses_families_after_rules_and_animals(self):
        rows = [compact_family(i) for i in range(4)]
        for i, row in enumerate(rows):
            row["gen200_reversal_Q_by_rule"][:, 5] = 1 if i < 2 else -1
        value = entry.summarize(rows)["arms"]["gen200_reversal"]["windows"][5]["Q"]
        self.assertEqual((value["mean"], value["n"]), (0., 4))
        self.assertAlmostEqual(value["se"], np.std([1., 1., -1., -1.], ddof=1)/2.)

    def test_absent_and_dead_animals_remain_in_the_denominator(self):
        rows = [compact_family(i) for i in range(4)]
        for arm in entry.ARMS:
            rows[-1][arm + "_Q_by_rule"][:, 3:] = 0
            rows[-1][arm + "_no_contact_by_rule"] = rows[-1][arm + "_Q_by_rule"] == 0
            rows[-1][arm + "_alive_end_by_rule"][:, 3:] = False
            rows[-1][arm + "_final_survival_by_rule"][:] = False
        result = entry.summarize(rows)["arms"]["gen200_reversal"]
        late = result["windows"][5]
        self.assertEqual((late["Q"]["mean"], late["first_food"]["mean"], late["no_contact"]["mean"]), (.75, .75, .25))
        self.assertEqual(result["end_survival"]["mean"], .75)
        self.assertEqual(late["Q"]["n"], 4)

    def test_strict_bounds_and_three_separate_lineages(self):
        measurement = entry.summarize([compact_family(i) for i in range(4)])
        component, safeguard = copy.deepcopy(measurement["components"]), copy.deepcopy(measurement["safeguards"])
        component["descendant_vs_founder_late"] = dict(mean=.05, se=0., n=4)
        self.assertTrue(entry._gate(component, safeguard)["passed"])
        component["descendant_vs_founder_late"]["se"] = .025
        self.assertFalse(entry._gate(component, safeguard)["passed"])
        safeguard["pre_Q"]["lower_2se"] = -.05
        self.assertFalse(entry._gate(measurement["components"], safeguard)["criteria"]["pre_noninferiority"])
        panels = [dict(index=i, measurements=entry.summarize([compact_family(j, recovery=i < 2) for j in range(4)])) for i in range(3)]
        self.assertEqual(entry._replicated_gates(panels)["evolutionary_recovery"], dict(count=2, seeds=[0, 1], passed=False))

    def test_pre_switch_pairing_and_compact_dimensions_are_checked(self):
        rows = [compact_family(i) for i in range(4)]
        rows[0]["gen200_reversal_Q_by_rule"][0, 2, 0] = -1
        with self.assertRaisesRegex(ValueError, "before the switch"):
            entry.summarize(rows)
        rows[0]["gen200_reversal_Q_by_rule"] = np.zeros((2, 6, 3), np.int8)
        with self.assertRaises(ValueError):
            entry.summarize(rows)


class FounderReversalRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="founder-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (*entry.START_SOURCES, *entry.FINAL_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E130-benchmark",
            "--seed", "2026120900", "--generations", "0", "--eval-worlds", "4", "--start", "file:" + entry.START_SOURCES[0]["path"]]
        self.calls, self.fail_at, self.now, self.expire = [], None, 0., False

    @staticmethod
    def keys(master, index):
        return np.asarray([[master, 2*index], [master, 2*index+1]], np.uint32)

    def assignments(self, master, count):
        return dict(family_keys=np.asarray([self.keys(master, i) for i in range(count)]), initial_rules=np.asarray([0, 1], np.int32))

    def evaluate(self, first, last, keys, index, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate founder interruption")
        if self.expire:
            self.now = 181.
        self.calls.append((int(keys[0, 0]), index))
        row = compact_family(index, recovery=int(keys[0, 0]) != 2026120903)
        row.update(keys=keys.copy(), initial_rules=np.asarray([0, 1]), bin_edges=np.asarray(entry.BIN_EDGES), flip_step=np.asarray(150),
            gen0_genome_sha256=entry._array_sha(first), gen200_genome_sha256=entry._array_sha(last), trace=np.zeros((2, 300, 4)))
        return row

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a"*40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.args if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b"*64}))
            stack.enter_context(patch.object(entry, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys), ("evaluate_world", self.evaluate),
                                   ("summary_input", lambda row: {key: value for key, value in row.items() if key != "trace"})):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E130-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E130-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    def finals(self):
        for i in range(3):
            self.invoke("--condition-split", "final", "--name", f"E130-A{i}", "--seed", str(2026120901+i),
                        "--eval-worlds", "64", "--start", "file:" + entry.START_SOURCES[i]["path"])

    def audit(self):
        checked, panels = {}, []
        for i in range(3):
            folder = self.folder(f"E130-A{i}")
            panels.append(dict(index=i, run=folder.relative_to(self.root).as_posix(), measurements=self.result(f"E130-A{i}")["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for source in (*entry.START_SOURCES, *entry.FINAL_SOURCES):
            checked[source["path"]] = source["sha256"]
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))))

    def test_development_pins_both_original_endpoints_and_assignments(self):
        self.invoke()
        config, result = json.loads((self.folder() / "config.json").read_text()), self.result()
        self.assertEqual(self.calls, [(2026120900, i) for i in range(4)])
        self.assertEqual((config["cfg"]["M"], config["cfg"]["T"], config["max_seconds"]), (4, 300, 180))
        self.assertEqual(config["source_start"], entry.START_SOURCES[0])
        self.assertEqual(config["source_final"], entry.FINAL_SOURCES[0])
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((self.folder() / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])

    def test_invalid_settings_dirty_code_and_replays_fail_before_evaluation(self):
        for flags in (("--seed", "2026120800"), ("--eval-worlds", "128"), ("--generations", "1"), ("--cfg", "M=1"),
                      ("--no-plastic",), ("--name", "E130-benchmark-retry1"), ("--allow-dirty",), ("--start", "file:" + entry.FINAL_SOURCES[0]["path"])):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(self.calls)
        self.invoke()
        before = (self.folder() / "summary.json").read_bytes()
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(before, (self.folder() / "summary.json").read_bytes())

    def test_source_tampering_and_partial_failure_are_preserved(self):
        path = self.root / entry.START_SOURCES[0]["path"]
        raw = path.read_bytes()
        path.write_bytes(raw + b"wrong")
        with self.assertRaises(SystemExit):
            self.invoke()
        path.write_bytes(raw)
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertNotIn("measurements", result)

    def test_deadline_preserves_completed_prefix(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        self.assertEqual((self.result()["status"], self.result()["completed_families"]), ("interrupted", 1))

    def test_family_wall_time_includes_journal_and_status_persistence(self):
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
        self.assertEqual(len(result["family_wall_seconds"]), 4)
        self.assertTrue(all(wall >= inner+.75 for wall, inner in zip(result["family_wall_seconds"], result["family_seconds"])))

    def test_all_three_panels_and_independent_audit_lock_reporting(self):
        self.finals()
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = self.root / "results/RESULTS.md"
        self.assertIn("Replicated evolutionary recovery: **undecided**", table.read_text(encoding="utf-8"))
        self.audit()
        result = entry.build_comparison(self.root)
        self.assertEqual(result["replicated_gates"]["evolutionary_recovery"], dict(count=2, seeds=[0, 1], passed=False))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        self.assertIn("Replicated evolutionary recovery: **FAIL** (2/3 lineages)", table.read_text(encoding="utf-8"))
        self.assertIn("n=64", table.read_text(encoding="utf-8"))
        raw = self.folder("E130-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes() + b"changed after audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
