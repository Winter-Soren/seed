"""E132 all-assigned paired endpoints and immutable runner fixtures; no simulation."""
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

from scripts import run_experiment as dispatcher, run_voluntary_offers as entry, summarize as report


def family(index, *, improve=True):
    row = dict(base_index=np.asarray(index), pre_old_S_by_rule=np.full(2, .5))
    for arm in entry.ARMS:
        food = arm.endswith("stationary") or arm == "near_reversal" and improve
        both = arm.startswith("near_") and improve
        fields = dict(local_S=.5 if food else 0., both_colors=int(both), solo_Y=int(food),
            first_food=int(food), first_poison=0, no_contact=int(not food), exposure_survival=1,
            readout_survival=1, solo_survival=1, intake_A=1, intake_B=int(both), realized_energy=-.192)
        row.update({arm + "_" + field + "_by_rule": np.full(2, value) for field, value in fields.items()})
    return row


class VoluntaryOffersStatisticsTests(unittest.TestCase):
    def test_complete_gate_and_all_three_lineages(self):
        value = entry.summarize([family(i) for i in range(4)])
        self.assertTrue(value["gate"]["passed"])
        self.assertEqual(value["components"]["both_colors_gain"], dict(mean=1., se=0., n=4))
        self.assertEqual((value["initial_rule_replays"], value["continuation_lives"], value["agents_per_rule"]), (8, 32, 1))
        self.assertEqual(set(value["by_initial_rule"]["0"]), {"pre_old_local", "arms", "components", "safeguards"})
        panels = [dict(index=i, measurements=entry.summarize([family(j, improve=i != 2) for j in range(4)])) for i in range(3)]
        self.assertEqual(entry._replicated_gates(panels)["voluntary_access"], dict(count=2, seeds=[0, 1], passed=False))

    def test_standard_error_averages_rules_before_independent_families(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            row["near_reversal_first_food_by_rule"] = np.asarray([1, 0])
            row["near_reversal_first_poison_by_rule"] = np.asarray([0, 1])
            row["near_reversal_no_contact_by_rule"][:] = 0
            row["near_reversal_solo_Y_by_rule"] = np.asarray([1, -1])
        result = entry.summarize(rows)
        self.assertEqual(result["arms"]["near_reversal"]["solo_Y"], dict(mean=0., se=0., n=4))
        self.assertEqual(result["by_initial_rule"]["1"]["arms"]["near_reversal"]["solo_Y"]["mean"], -1.)
        rows[0]["near_reversal_local_S_by_rule"][:] = -.5
        local = entry.summarize(rows)["arms"]["near_reversal"]["local_S"]
        self.assertAlmostEqual(local["se"], np.std([-.5, .5, .5, .5], ddof=1)/2)

    def test_death_after_exposure_keeps_exposure_safeguard_and_zeros_choices(self):
        rows = [family(i) for i in range(4)]
        for field in ("local_S", "solo_Y", "first_food", "first_poison", "readout_survival", "solo_survival"):
            rows[-1]["near_reversal_" + field + "_by_rule"][:] = 0
        rows[-1]["near_reversal_no_contact_by_rule"][:] = 1
        result = entry.summarize(rows)
        arm = result["arms"]["near_reversal"]
        self.assertEqual((arm["exposure_survival"]["mean"], arm["readout_survival"]["mean"]), (1., .75))
        self.assertEqual((arm["solo_Y"]["mean"], arm["no_contact"]["mean"]), (.75, .25))
        self.assertEqual(result["safeguards"]["reversal_exposure_survival"]["mean"], 0.)

    def test_fatal_contacts_still_count_both_colors(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            for field in ("local_S", "solo_Y", "first_food", "first_poison", "exposure_survival", "readout_survival", "solo_survival"):
                row["near_reversal_" + field + "_by_rule"][:] = 0
            row["near_reversal_no_contact_by_rule"][:] = 1
        result = entry.summarize(rows)
        self.assertEqual(result["components"]["both_colors_gain"]["mean"], 1.)
        self.assertFalse(result["gate"]["passed"])
        self.assertIn("reversal_survival_noninferiority", result["gate"]["failed_criteria"])

    def test_size_thresholds_inclusive_evidence_and_safeguards_strict(self):
        measured = entry.summarize([family(i) for i in range(4)])
        components, safeguards = copy.deepcopy(measured["components"]), copy.deepcopy(measured["safeguards"])
        components["both_colors_gain"] = dict(mean=.10, se=0., n=4)
        components["solo_gain"] = dict(mean=.05, se=0., n=4)
        self.assertTrue(entry._gate(components, safeguards)["passed"])
        components["both_colors_gain"]["se"] = .05
        components["solo_gain"]["se"] = .025
        safeguards["stationary_solo"]["lower_2se"] = -.05
        gate = entry._gate(components, safeguards)
        for key in ("both_colors_improve", "solo_gain", "stationary_solo_noninferiority"):
            self.assertFalse(gate["criteria"][key])

    def test_direct_and_stationary_adjusted_solo_effects_are_separate(self):
        measured = entry.summarize([family(i) for i in range(4)])
        components = copy.deepcopy(measured["components"])
        components["adjusted_solo_gain"] = dict(mean=0., se=0., n=4)
        gate = entry._gate(components, measured["safeguards"])
        self.assertTrue(gate["criteria"]["solo_gain"])
        self.assertFalse(gate["criteria"]["adjusted_solo_gain"])
        self.assertFalse(gate["passed"])

    def test_rejects_revived_scoring_bad_intake_and_changed_family_denominator(self):
        rows = [family(i) for i in range(4)]
        rows[0]["near_reversal_readout_survival_by_rule"][0] = 0
        with self.assertRaises(ValueError):
            entry.summarize(rows)
        rows = [family(i) for i in range(4)]
        rows[0]["near_reversal_intake_A_by_rule"][0] = 9
        with self.assertRaises(ValueError):
            entry.summarize(rows)
        rows[0]["base_index"] = np.asarray(9)
        with self.assertRaises(ValueError):
            entry.summarize(rows)


class VoluntaryOffersRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = dispatcher.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="offers-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sources = []
        for index, source in enumerate(entry.SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.linspace(-1, 1, 155, dtype=np.float32)+np.float32(index), allow_pickle=False)
            sources.append(dict(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.sources = tuple(sources)
        patched_sources = patch.object(entry, "SOURCES", self.sources)
        patched_sources.start()
        self.addCleanup(patched_sources.stop)
        self.args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E132-benchmark",
            "--seed", "2026121100", "--generations", "0", "--eval-worlds", "4", "--start", "file:"+self.sources[0]["path"]]
        self.calls, self.fail_at, self.now, self.expire, self.bad_source = [], None, 0., False, False

    @staticmethod
    def keys(master, index):
        return np.asarray([[master, 6*index+i] for i in range(6)], np.uint32)

    def assignments(self, master, count):
        return dict(family_keys=np.asarray([self.keys(master, i) for i in range(count)]), initial_rules=np.asarray([0, 1], np.int32))

    def evaluate(self, flat, keys, index, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate offer interruption")
        self.calls.append((int(keys[0, 0]), index))
        if self.expire:
            self.now = 181.
        raw = family(index, improve=int(keys[0, 0]) != 2026121103)
        raw.update(keys=keys.copy(), initial_rules=np.asarray([0, 1]), genome_sha256=entry._array_sha(flat), trace=np.zeros((2, 64, 1)))
        if self.bad_source:
            raw["genome_sha256"][0] ^= 1
        return raw

    def invoke(self, *flags, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(dispatcher, "ROOT", self.root))
            stack.enter_context(patch.object(dispatcher, "git_info", return_value=dict(commit="a"*40, dirty=dirty)))
            stack.enter_context(patch.object(dispatcher.sys, "argv", ["runner", *(self.args if common else []), *flags]))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b"*64}))
            stack.enter_context(patch.object(entry, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys), ("evaluate_world", self.evaluate),
                                   ("summary_input", lambda raw: {name: value for name, value in raw.items() if name != "trace"})):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            dispatcher.main()

    def folder(self, name="E132-benchmark"):
        return next((self.root / "runs").glob("*_"+name))

    def result(self, name="E132-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    def finals(self):
        for index in range(3):
            self.invoke("--condition-split", "final", "--name", f"E132-A{index}", "--seed", str(2026121101+index),
                "--eval-worlds", "64", "--start", "file:"+self.sources[index]["path"])

    def audit(self):
        panels, checked = [], {}
        for index in range(3):
            folder = self.folder(f"E132-A{index}")
            panels.append(dict(index=index, run=folder.relative_to(self.root).as_posix(), measurements=self.result(f"E132-A{index}")["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.update({source["path"]: source["sha256"] for source in self.sources})
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))))

    def test_development_has_fixed_one_creature_apparatus_and_pinned_evidence(self):
        self.invoke()
        config, result = json.loads((self.folder() / "config.json").read_text()), self.result()
        self.assertEqual(self.calls, [(2026121100, index) for index in range(4)])
        self.assertEqual(config["source_genome"], self.sources[0])
        self.assertEqual(config["cfg"]["M"], 1)
        self.assertEqual(config["apparatus"], entry.APPARATUS)
        self.assertEqual(config["max_seconds"], 180)
        self.assertNotIn("policy_order", config)
        self.assertNotIn("family_policy_seconds", result)
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertFalse(result["scientific_result"])
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((self.folder() / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])

    def test_registration_dirty_source_and_duplicate_attempts_are_rejected(self):
        for flags in (("--seed", "2026121000"), ("--eval-worlds", "128"), ("--generations", "1"), ("--cfg", "M=4"),
                      ("--no-plastic",), ("--name", "E132-benchmark-retry1"), ("--allow-dirty",), ("--start", "file:"+self.sources[1]["path"])):
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

    def test_source_tampering_is_rejected_before_any_family_and_raw_identity_before_save(self):
        path = self.root / self.sources[0]["path"]
        content = path.read_bytes()
        path.write_bytes(content+b"changed")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)
        path.write_bytes(content)
        self.bad_source = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_interrupted_prefix_is_preserved_without_a_measurement(self):
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertNotIn("measurements", result)
        self.assertEqual(len(result["raw_files"]), 2)

    def test_deadline_records_only_completed_families(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        self.assertEqual((self.result()["status"], self.result()["completed_families"]), ("interrupted", 1))

    def test_wall_timing_includes_persisted_status_and_fsync(self):
        writer, sync = entry._write_status, entry.os.fsync
        def timed_write(*args, **kwargs):
            writer(*args, **kwargs)
            self.now += .5
        def timed_sync(*args, **kwargs):
            sync(*args, **kwargs)
            self.now += .25
        with patch.object(entry, "_write_status", side_effect=timed_write), patch.object(entry.os, "fsync", side_effect=timed_sync):
            self.invoke()
        result = self.result()
        self.assertEqual(len(result["family_wall_seconds"]), 4)
        self.assertTrue(all(wall >= inner+.75 for wall, inner in zip(result["family_wall_seconds"], result["family_seconds"])))

    def test_incomplete_triplet_does_not_read_a_finished_partial_outcome(self):
        self.invoke("--condition-split", "final", "--name", "E132-A0", "--seed", "2026121101", "--eval-worlds", "64")
        (self.folder("E132-A0") / "summary.json").write_text("outcomes must remain unread")
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")

    def test_complete_triplet_and_audit_required_for_report_and_stale_bytes_rejected(self):
        self.finals()
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        with patch.object(report, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            report.main()
        table = self.root / "results/RESULTS.md"
        self.assertIn("Replicated voluntary access: **undecided**", table.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "results/diagnostics/E132-comparison.json").exists())
        self.audit()
        value = entry.build_comparison(self.root)
        self.assertEqual(value["replicated_gates"]["voluntary_access"], dict(count=2, seeds=[0, 1], passed=False))
        with patch.object(report, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            report.main()
        self.assertIn("Replicated voluntary access: **FAIL** (2/3 lineages)", table.read_text(encoding="utf-8"))
        self.assertIn("n=64", table.read_text(encoding="utf-8"))
        raw = self.folder("E132-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes()+b"changed after audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
