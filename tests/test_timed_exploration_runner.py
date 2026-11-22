"""E131 paired gates, cost separation and immutable panel integration without simulation."""
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

from scripts import run_experiment as dispatcher, run_timed_exploration as entry, summarize as report


def family(index, *, improve=True):
    row = dict(base_index=np.asarray(index))
    for arm in entry.ARMS:
        q = np.zeros((2, 6, 4), np.int8)
        if arm == "adaptive_reversal" and improve:
            q[:, 5] = 1
        row.update({arm + "_Q_by_rule": q, arm + "_no_contact_by_rule": q == 0,
            arm + "_alive_end_by_rule": np.ones_like(q, bool), arm + "_final_survival_by_rule": np.ones((2, 4), bool),
            arm + "_latency_first_food_by_rule": np.full((2, 4), 10 if arm.startswith("adaptive_") and improve else 50),
            arm + "_latency_first_any_by_rule": np.zeros((2, 4), np.int32),
            arm + "_override_count_by_rule": np.full((2, 4), 0 if arm.startswith("unchanged_") else 12),
            arm + "_changed_action_count_by_rule": np.full((2, 4), 0 if arm.startswith("unchanged_") else 9),
            arm + "_alive_turns_by_rule": np.full((2, 4), 300)})
    return row


class TimedExplorationStatisticsTests(unittest.TestCase):
    def test_paired_both_controls_and_all_three_lineages(self):
        values = entry.summarize([family(i) for i in range(4)])
        self.assertTrue(values["gate"]["passed"])
        self.assertEqual(values["comparisons"]["ordinary"]["latency_saved"], dict(mean=40., se=0., n=4))
        self.assertEqual(values["policy_history_rollouts"], 48)
        self.assertEqual(set(values["by_initial_rule"]["0"]), {"arms", "components", "comparisons"})
        panels = [dict(index=i, measurements=entry.summarize([family(j, improve=i < 2) for j in range(4)])) for i in range(3)]
        self.assertEqual(entry._replicated_gates(panels)["timed_exploration"], dict(count=2, seeds=[0, 1], passed=False))

    def test_adjustment_cannot_replace_direct_late_advantage(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            for arm in ("unchanged_reversal", "random_quota_reversal", "unchanged_stationary", "random_quota_stationary"):
                row[arm + "_Q_by_rule"][:, 5] = 1
                row[arm + "_no_contact_by_rule"] = row[arm + "_Q_by_rule"] == 0
        result = entry.summarize(rows)
        self.assertEqual(result["comparisons"]["ordinary"]["adjusted_late_Q"]["mean"], 1.)
        self.assertFalse(result["gate"]["criteria"]["ordinary_late_Q_improves"])

    def test_no_food_and_death_remain_censored_at_fifty(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            for arm in entry.ARMS:
                row[arm + "_alive_end_by_rule"][:, 3:, 0] = False
                row[arm + "_final_survival_by_rule"][:, 0] = False
                row[arm + "_Q_by_rule"][:, 3:, 0] = 0
                row[arm + "_no_contact_by_rule"] = row[arm + "_Q_by_rule"] == 0
                row[arm + "_latency_first_food_by_rule"][:, 0] = 50
        value = entry.summarize(rows)["arms"]["adaptive_reversal"]
        self.assertEqual(value["first_food_latency"], dict(mean=20., se=0., n=4))
        self.assertEqual(value["end_survival"]["mean"], .75)
        self.assertEqual(value["windows"][5]["Q"]["mean"], .75)

    def test_family_pairing_preserves_rule_covariance_and_override_ratio(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            row["adaptive_reversal_Q_by_rule"][0, 5] = -1
            row["adaptive_reversal_Q_by_rule"][1, 5] = 1
            row["adaptive_reversal_alive_turns_by_rule"][0] = 100
        result = entry.summarize(rows)
        self.assertEqual(result["arms"]["adaptive_reversal"]["windows"][5]["Q"], dict(mean=0., se=0., n=4))
        self.assertEqual(result["by_initial_rule"]["0"]["arms"]["adaptive_reversal"]["windows"][5]["Q"]["mean"], -1.)
        self.assertAlmostEqual(result["arms"]["adaptive_reversal"]["override_live_fraction"]["mean"], .06)
        self.assertAlmostEqual(result["arms"]["adaptive_reversal"]["changed_action_live_fraction"]["mean"], .045)

    def test_inclusive_effect_strict_evidence_and_noninferiority_bound(self):
        result = entry.summarize([family(i) for i in range(4)])
        c = copy.deepcopy(result["comparisons"])
        c["ordinary"]["adjusted_late_Q"] = dict(mean=.05, se=0., n=4)
        self.assertTrue(entry._gate(result["components"], c)["passed"])
        c["ordinary"]["adjusted_late_Q"]["se"] = .025
        c["quota"]["safeguards"]["stationary_late_Q"]["lower_2se"] = -.05
        gate = entry._gate(result["components"], c)
        self.assertFalse(gate["criteria"]["ordinary_adjusted_Q_improves"])
        self.assertFalse(gate["criteria"]["quota_stationary_late_noninferiority"])

    def test_cost_uses_all_warm_observations_and_is_separate(self):
        status = dict(status="finished", completed_families=4, expected_families=4, policy_order=list(entry.POLICIES),
            family_policy_seconds=[[100, 300, 100], [2, 2, 4], [4, 6, 8], [10, 40, 20]])
        cost = entry.operational_cost(status)
        self.assertEqual((cost["ordinary_median_seconds"], cost["adaptive_median_seconds"], cost["ratio"], cost["passed"]), (4., 6., 1.5, True))
        status["family_policy_seconds"][2][1] = 6.01
        self.assertFalse(entry.operational_cost(status)["passed"])
        self.assertTrue(entry.summarize([family(i) for i in range(4)])["gate"]["passed"])
        status["family_policy_seconds"][0][0] = 0
        with self.assertRaises(ValueError):
            entry.operational_cost(status)

    def test_rejects_changed_pre_switch_history_and_invalid_endpoint(self):
        rows = [family(i) for i in range(4)]
        rows[0]["adaptive_reversal_Q_by_rule"][0, 0, 0] = 1
        rows[0]["adaptive_reversal_no_contact_by_rule"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "before the switch"):
            entry.summarize(rows)
        rows = [family(i) for i in range(4)]
        rows[0]["adaptive_reversal_latency_first_food_by_rule"][0, 0] = 51
        with self.assertRaises(ValueError):
            entry.summarize(rows)


class TimedExplorationRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = dispatcher.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="timed-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((dispatcher.ROOT / source["path"]).read_bytes())
        self.args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E131-benchmark",
            "--seed", "2026121000", "--generations", "0", "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.fail_at, self.now, self.expire, self.bad_timing = [], None, 0., False, False

    @staticmethod
    def keys(master, index):
        return np.asarray([[master, 3*index+i] for i in range(3)], np.uint32)

    def assignments(self, master, count):
        return dict(family_keys=np.asarray([self.keys(master, i) for i in range(count)]), initial_rules=np.asarray([0, 1], np.int32))

    def evaluate(self, genome, keys, index, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate immutable interruption")
        self.calls.append((int(keys[0, 0]), index))
        if self.expire:
            self.now = 201.
        row = family(index, improve=int(keys[0, 0]) != 2026121003)
        row.update(keys=keys.copy(), initial_rules=np.asarray([0, 1]), bin_edges=np.asarray(entry.BIN_EDGES), flip_step=np.asarray(150),
            genome_sha256=entry.base._array_sha(genome), policy_seconds=np.asarray([0. if self.bad_timing else 1., 1.25, 1.1]), trace=np.zeros((2, 300, 4)))
        return row

    def invoke(self, *flags, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(dispatcher, "ROOT", self.root))
            stack.enter_context(patch.object(dispatcher, "git_info", return_value=dict(commit="a"*40, dirty=dirty)))
            stack.enter_context(patch.object(dispatcher.sys, "argv", ["runner", *(self.args if common else []), *flags]))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b"*64}))
            stack.enter_context(patch.object(entry.base, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry.base.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys), ("evaluate_world", self.evaluate),
                                   ("summary_input", lambda row: {key: value for key, value in row.items() if key != "trace"})):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            dispatcher.main()

    def folder(self, name="E131-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E131-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    def finals(self):
        for i in range(3):
            self.invoke("--condition-split", "final", "--name", f"E131-A{i}", "--seed", str(2026121001+i),
                "--eval-worlds", "64", "--start", "file:" + entry.SOURCES[i]["path"])

    def audit(self):
        panels, checked = [], {}
        for i in range(3):
            folder = self.folder(f"E131-A{i}")
            panels.append(dict(index=i, run=folder.relative_to(self.root).as_posix(), measurements=self.result(f"E131-A{i}")["measurements"]))
        for folder in [self.folder(), *[self.folder(f"E131-A{i}") for i in range(3)]]:
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.update({source["path"]: source["sha256"] for source in entry.SOURCES})
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels), operational_cost=entry.operational_cost(self.result()))))

    def test_pins_single_original_genome_and_preserves_all_policy_timings(self):
        self.invoke()
        config, result = json.loads((self.folder() / "config.json").read_text()), self.result()
        self.assertEqual(self.calls, [(2026121000, i) for i in range(4)])
        self.assertEqual((config["cfg"]["M"], config["cfg"]["T"], config["max_seconds"]), (4, 300, 200))
        self.assertEqual(config["source_genome"], entry.SOURCES[0])
        self.assertFalse((self.folder() / "genome_start.npy").exists())
        self.assertEqual(result["family_policy_seconds"], [[1., 1.25, 1.1]]*4)
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertFalse(result["scientific_result"])
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((self.folder() / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        journal = [json.loads(line) for line in (self.folder() / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["policy_seconds"] for row in journal], result["family_policy_seconds"])

    def test_fixed_names_sources_settings_dirty_code_and_duplicates_fail_closed(self):
        for flags in (("--seed", "2026120900"), ("--eval-worlds", "128"), ("--generations", "1"), ("--cfg", "M=1"),
                      ("--no-plastic",), ("--name", "E131-benchmark-retry1"), ("--allow-dirty",), ("--start", "file:" + entry.SOURCES[1]["path"])):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(self.calls)
        self.invoke()
        original = (self.folder() / "summary.json").read_bytes()
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(original, (self.folder() / "summary.json").read_bytes())

    def test_partial_failure_preserves_evidence_and_never_summarizes_measurements(self):
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertEqual(len(result["family_policy_seconds"]), 2)
        self.assertNotIn("measurements", result)

    def test_zero_timing_rejected_before_any_family_is_saved(self):
        self.bad_timing = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_deadline_preserves_the_completed_prefix(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        self.assertEqual((self.result()["status"], self.result()["completed_families"]), ("interrupted", 1))

    def test_full_persistence_timing(self):
        writer = entry.base._write_status
        def timed_write(*args, **kwargs):
            writer(*args, **kwargs)
            self.now += .5
        with patch.object(entry.base, "_write_status", side_effect=timed_write):
            self.invoke()
        result = self.result()
        self.assertEqual(len(result["family_wall_seconds"]), 4)
        self.assertTrue(all(wall >= inner+.5 for wall, inner in zip(result["family_wall_seconds"], result["family_seconds"])))

    def test_complete_triplet_and_audit_lock_report_and_cost_separately(self):
        self.invoke()
        self.finals()
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        self.audit()
        result = entry.build_comparison(self.root)
        self.assertEqual(result["replicated_gates"]["timed_exploration"], dict(count=2, seeds=[0, 1], passed=False))
        self.assertEqual(result["adoption"], dict(behavioral_passed=False, operational_passed=True, passed=False))
        with patch.object(report, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            report.main()
        table = (self.root / "results/RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("Replicated timed exploration: **FAIL** (2/3 lineages)", table)
        self.assertIn("n=64", table)
        raw = self.folder("E131-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
