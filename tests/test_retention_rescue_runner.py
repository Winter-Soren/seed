"""E129 gate arithmetic and thin tracked-runner compatibility."""
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

from scripts import run_experiment as runner, run_retention_diagnostic as base
from scripts import run_retention_intervention as entry, summarize


def compact_family(index, *, local_gain=.15):
    row = dict(base_index=np.asarray(index))
    for arm in entry.ARMS:
        reversal = arm.endswith("reversal")
        local = np.tile([.5, .25, .75, .25, .25] if reversal else [.1, .5, .6, .4, -.4], (2, 2, 1))
        grid = np.tile([1., 0., 0., 1., 1., 1., 1.] if reversal else [0., 1., 0., -1., 1., -1., 1.], (2, 2, 1))
        if arm in ("plastic_reversal", "plastic_hebbian_off_reversal"):
            score = .1 + (local_gain if "hebbian_off" in arm else 0.)
            local[:, 1] = [score+.25, .25, score+.5, .5-score, score]
        if arm == "plastic_reversal":
            grid[:, 1] = [0., 1., 0., -1., 1., -1., 1.]
        row[arm + "_pre_old_by_rule"] = np.full(2, .6)
        row[arm + "_pre_local_by_rule"] = np.tile([.1, .7, .8, .2, -.6], (2, 1))
        row[arm + "_local_by_rule"], row[arm + "_grid_by_rule"] = local, grid
    return row


class RetentionInterventionStatisticsTests(unittest.TestCase):
    def test_paired_primary_and_original_rescue_gates_are_distinct(self):
        result = entry.summarize([compact_family(i) for i in range(4)])
        effect = result["intervention"]
        self.assertAlmostEqual(effect["local_difference_33"]["mean"], .15)
        self.assertEqual(effect["local_difference_33"]["n"], 4)
        self.assertTrue(all(effect[name]["passed"] for name in ("primary", "local_rescue", "transfer_rescue")))
        self.assertEqual(set(result["frozen"]["arms"]), {"plastic_hebbian_off_stationary", "plastic_hebbian_off_reversal"})
        self.assertEqual(set(result["frozen"]["architectures"]), {"plastic"})
        self.assertFalse(result["normal"]["architectures"]["plastic"]["local"]["passed"])

    def test_inclusive_effect_size_and_strict_evidence_boundaries(self):
        frozen = dict(local=dict(passed=True), grid=dict(passed=True))
        zero_grid = dict(mean=0., se=0.)
        decision = entry._gates(dict(mean=.025, se=0.), zero_grid, frozen)
        self.assertTrue(decision["primary"]["passed"])
        self.assertTrue(decision["local_rescue"]["passed"])
        self.assertFalse(decision["transfer_rescue"]["passed"])
        self.assertFalse(entry._gates(dict(mean=.025, se=.0125), dict(mean=1., se=0.), frozen)["primary"]["passed"])
        frozen["local"]["passed"] = False
        self.assertFalse(entry._gates(dict(mean=.5, se=0.), dict(mean=1., se=0.), frozen)["local_rescue"]["passed"])

    def test_primary_uncertainty_averages_rules_before_paired_contrast(self):
        rows = [compact_family(i) for i in range(4)]
        for row, variation in zip(rows, (.125, .25, .0625, .1875)):
            row["plastic_reversal_local_by_rule"][:, 1, 4] = [.125, .125]
            row["plastic_hebbian_off_reversal_local_by_rule"][:, 1, 4] = [.15625-variation, .15625+variation]
        result = entry.summarize(rows)
        self.assertEqual(result["intervention"]["local_difference_33"], dict(mean=.03125, se=0., n=4))
        self.assertGreater(result["intervention"]["by_initial_rule"]["0"]["local_difference_33"]["se"], 0)

    def test_common_immediate_snapshot_is_byte_exact(self):
        rows = [compact_family(i) for i in range(4)]
        rows[0]["plastic_reversal_pre_old_by_rule"][0] = 0.
        rows[0]["plastic_hebbian_off_reversal_pre_old_by_rule"][0] = -0.
        with self.assertRaisesRegex(ValueError, "exact immediate"):
            entry.summarize(rows)

    def test_all_three_lineages_and_same_lineage_rescue_are_required(self):
        panels = [dict(index=i, measurements=entry.summarize([compact_family(j, local_gain=.15 if i < 2 else 0.)
                                                             for j in range(4)])) for i in range(3)]
        gates = entry._replicated_gates(panels)
        self.assertEqual(gates["primary"], dict(count=2, seeds=[0, 1], passed=False))
        self.assertFalse(gates["local_rescue"]["passed"])
        self.assertFalse(gates["transfer_rescue"]["passed"])
        panels[2]["measurements"]["intervention"]["primary"]["passed"] = True
        with self.assertRaisesRegex(ValueError, "same lineage"):
            entry._replicated_gates(panels)

    def test_e128_default_schema_and_calculations_remain_unchanged(self):
        rows = [compact_family(i) for i in range(4)]
        old_rows = copy.deepcopy(rows)
        for row in old_rows:
            for arm in base.ARMS:
                for suffix in ("local_by_rule", "grid_by_rule"):
                    row[arm + "_" + suffix] = row[arm + "_" + suffix][:, [0, 0, 1]]
        old = base.summarize(old_rows)
        self.assertEqual(old, base.summarize(old_rows, retention_ticks=(0, 1, 33)))
        self.assertEqual(old["retention_ticks"], [0, 1, 33])
        self.assertEqual(set(old), {"families", "initial_rule_replays", "retention_ticks", "arms", "architectures", "by_initial_rule"})
        normal = entry.summarize(rows)["normal"]
        for architecture in ("plastic", "direct"):
            for endpoint in ("local", "grid"):
                previous = copy.deepcopy(old["architectures"][architecture][endpoint])
                del previous["by_delay"]["1"]
                self.assertEqual(previous, normal["architectures"][architecture][endpoint])


class RetentionInterventionRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="retention-intervention-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (*base.SOURCES, *base.DIRECT_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E129-benchmark",
                     "--seed", "2026120800", "--generations", "0", "--eval-worlds", "4", "--start", "file:" + base.SOURCES[0]["path"]]
        self.calls, self.fail_at = [], None

    @staticmethod
    def assignments(master, count):
        return dict(key=np.asarray([master, 1000000], np.uint32), permutation=np.arange(count, dtype=np.int32),
                    pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2))

    @staticmethod
    def keys(master, index):
        return np.asarray([[master, index*7+role] for role in range(7)], np.uint32)

    def evaluate(self, plastic, fixed, keys, index, pre_contacts, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate intervention interruption")
        self.calls.append((int(keys[0, 0]), index, pre_contacts))
        row = compact_family(index, local_gain=0. if int(keys[0, 0]) == 2026120803 else .15)
        row.update(keys=keys.copy(), pre_contacts=np.asarray(pre_contacts), initial_rules=np.asarray([0, 1]),
            retention_ticks=np.asarray(entry.RETENTION_TICKS), plastic_genome_sha256=base._array_sha(plastic),
            direct_genome_sha256=base._array_sha(fixed), continuation_tick=np.asarray(1), trace=np.zeros((2, 33, 2)))
        return row

    def invoke(self, *extra, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a"*40, dirty=False)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.args if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b"*64}))
            stack.enter_context(patch.object(base, "_runtime", return_value=dict(test="mock backend")))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys),
                                   ("evaluate_world", self.evaluate), ("summary_input", lambda row: {k: v for k, v in row.items() if k != "trace"})):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    def folder(self, name="E129-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E129-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def finals(self):
        for index in range(3):
            self.invoke("--condition-split", "final", "--name", f"E129-A{index}", "--seed", str(2026120801+index),
                        "--eval-worlds", "128", "--start", "file:" + base.SOURCES[index]["path"])

    def audit(self):
        checked, panels = {}, []
        for index in range(3):
            folder = self.folder(f"E129-A{index}")
            panels.append(dict(index=index, run=folder.relative_to(self.root).as_posix(), measurements=self.result(f"E129-A{index}")["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for source in (*base.SOURCES, *base.DIRECT_SOURCES):
            checked[source["path"]] = source["sha256"]
        report = dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return path

    def test_thin_writer_uses_new_sources_assignment_and_timing_metadata(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.calls, [(2026120800, 0, 8), (2026120800, 1, 8), (2026120800, 2, 16), (2026120800, 3, 16)])
        self.assertEqual((config["protocol"], config["assignment_stream_tag"], config["retention_ticks"]), (entry.PROTOCOL, 0xE129, [0, 33]))
        self.assertEqual(config["arms"], list(entry.ARMS))
        self.assertTrue(all(config[name] == value for name, value in entry.CONFIG_EXTRA.items()))
        self.assertEqual(len(result["family_wall_seconds"]), 4)
        self.assertTrue(all(t >= 0 for t in result["family_wall_seconds"]))
        self.assertEqual((result["gate_status"], result["scientific_result"]), ("not_applicable", False))
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), common=False)
        self.assertEqual(before, self.snapshot(folder))

    def test_wrong_seeds_settings_and_retry_names_are_rejected(self):
        for flags in (("--seed", "2026120700"), ("--generations", "1"), ("--eval-worlds", "8"),
                      ("--cfg", "T=300"), ("--name", "E129-benchmark-retry1"), ("--allow-dirty",)):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        self.assertFalse(self.calls)

    def test_failed_prefix_is_preserved_without_measurements(self):
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertNotIn("measurements", result)

    def test_earlier_study_outcomes_are_not_inputs(self):
        folder = self.root / "runs" / "unused_E128-A0"
        folder.mkdir()
        (folder / "config.json").write_text("unread historical metadata")
        (folder / "summary.json").write_text("unread historical outcomes")
        self.invoke()
        self.assertEqual(self.result()["status"], "finished")

    def test_final_aggregate_and_report_wait_for_independent_three_panel_audit(self):
        self.finals()
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = self.root / "results/RESULTS.md"
        self.assertIn("Replicated intervention decisions: **undecided**", table.read_text(encoding="utf-8"))
        self.audit()
        result = entry.build_comparison(self.root)
        self.assertEqual(result["replicated_gates"]["primary"], dict(count=2, seeds=[0, 1], passed=False))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        self.assertIn("| primary | 2/3 | 0, 1 | FAIL |", table.read_text(encoding="utf-8"))
        self.assertTrue((self.root / "results/diagnostics/E129-comparison.json").exists())
        raw = self.folder("E129-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes() + b"changed after audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
