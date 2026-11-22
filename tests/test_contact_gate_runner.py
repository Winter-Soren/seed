"""E133 paired uncertainty, frozen gates and immutable runner fixtures; no simulation."""
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

from scripts import run_experiment as dispatcher, run_contact_gate as entry, summarize as report


def family(index, *, improve=True):
    row = dict(base_index=index, pre_old=np.full(2, .5), sparse_feedback_overlap=.25, arms={})
    for arm in entry.ARMS:
        policy, history = arm.split("_")
        local = {"native": [.4, .15], "contact": [.4, .4],
                 "sparse": [.2, .2], "reopened": [.4, .15]}[policy]
        if history == "stationary":
            local = [.4, .4]
        if not improve:
            local = [.4, .4]
        food = [1, int(policy == "contact" and improve)] if history == "reversal" else [1, 1]
        if policy == "sparse" and history == "reversal":
            food = [0, 0]
        values = dict(local_S=local, solo_Y=food, first_food=food, first_poison=[0, 0],
                      no_contact=[1-x for x in food], exposure_survival=[1, 1], solo_survival=[1, 1])
        row["arms"][arm] = {name: np.tile(np.asarray(value, np.float64), (2, 1)) for name, value in values.items()}
    return row


def raw_family(index, keys, flat, *, improve=True):
    compact = family(index, improve=improve)
    raw = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(), initial_rules=np.asarray([0, 1], np.int32),
               genome_sha256=entry._array_sha(flat), pre_old=compact["pre_old"],
               sparse_offsets=np.asarray([0, 1, 2, 3, 0, 1, 2, 3], np.int32))
    raw.update({arm+"_"+field: value for arm, fields in compact["arms"].items() for field, value in fields.items()})
    return raw


def compact_family(raw):
    return dict(base_index=int(raw["base_index"]), pre_old=raw["pre_old"],
                sparse_feedback_overlap=float(np.mean(raw["sparse_offsets"] == 3)),
                arms={arm: {field: raw[arm+"_"+field] for field in entry.FIELDS} for arm in entry.ARMS})


class ContactGateStatisticsTests(unittest.TestCase):
    def test_complete_gate_and_all_three_lineages(self):
        result = entry.summarize([family(i) for i in range(4)])
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(set(result), {"families", "arms", "components", "safeguards", "gate"})
        self.assertEqual(result["components"]["retention_advantage"], dict(mean=.25, se=0., n=4))
        self.assertEqual(result["components"]["sparse_feedback_overlap"], dict(mean=.25, se=0., n=4))
        self.assertNotIn("sparse_feedback_overlap", result["gate"]["criteria"])
        self.assertEqual(result["arms"]["contact_reversal"]["local_S"], [dict(mean=.4, se=0., n=4)]*2)
        panels = [dict(index=i, measurements=entry.summarize([family(j, improve=i != 2) for j in range(4)])) for i in range(3)]
        self.assertEqual(entry._replicated_gates(panels)["contact_gate"], dict(count=2, seeds=[0, 1], passed=False))

    def test_rules_are_averaged_before_uncertainty_and_differences_are_paired(self):
        rows = [family(i) for i in range(4)]
        for row in rows:
            arm = row["arms"]["contact_reversal"]
            arm["first_food"][:, 1] = [1, 0]
            arm["first_poison"][:, 1] = [0, 1]
            arm["no_contact"][:, 1] = 0
            arm["solo_Y"][:, 1] = [1, -1]
        result = entry.summarize(rows)
        self.assertEqual(result["arms"]["contact_reversal"]["solo_Y"][1], dict(mean=0., se=0., n=4))
        offsets = [-.1, 0., .1, .2]
        for offset, row in zip(offsets, rows):
            for arm in row["arms"].values():
                arm["local_S"] += offset
        result = entry.summarize(rows)
        self.assertAlmostEqual(result["arms"]["contact_reversal"]["local_S"][1]["se"], np.std(offsets, ddof=1)/2)
        self.assertAlmostEqual(result["components"]["gate_local_vs_sparse"]["se"], 0.)

    def test_acquisition_and_delay_ablation_are_required_for_retention_credit(self):
        result = entry.summarize([family(i) for i in range(4)])
        components = copy.deepcopy(result["components"])
        components["native_immediate"] = dict(mean=0., se=0., n=4)
        components["delay_ablation"] = dict(mean=0., se=0., n=4)
        gate = entry._gate(components, result["safeguards"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_criteria"], ["native_immediate", "delay_ablation"])

    def test_threshold_sizes_inclusive_evidence_and_safeguards_strict(self):
        result = entry.summarize([family(i) for i in range(4)])
        components, safeguards = copy.deepcopy(result["components"]), copy.deepcopy(result["safeguards"])
        for name in ("gate_immediate", "native_immediate", "gate_delayed"):
            components[name] = dict(mean=.1, se=0., n=4)
        for name in ("retention_advantage", "delay_ablation"):
            components[name] = dict(mean=.025, se=0., n=4)
        for name in ("gate_local_vs_sparse", "gate_solo_vs_native", "gate_solo_vs_sparse"):
            components[name] = dict(mean=.05, se=0., n=4)
        self.assertTrue(entry._gate(components, safeguards)["passed"])
        components["gate_immediate"]["se"] = .05
        components["retention_advantage"]["se"] = .0125
        safeguards["gate_own_retention"]["lower_2se"] = -.05
        gate = entry._gate(components, safeguards)
        self.assertEqual(gate["failed_criteria"], ["gate_immediate", "retention_advantage", "gate_own_retention"])

    def test_stationary_adjustment_is_descriptive_not_a_gate(self):
        result = entry.summarize([family(i) for i in range(4)])
        components = copy.deepcopy(result["components"])
        for control in ("native", "sparse"):
            components["adjusted_solo_vs_"+control] = dict(mean=-1., se=1., n=4)
        gate = entry._gate(components, result["safeguards"])
        self.assertTrue(gate["passed"])
        self.assertNotIn("adjusted_solo_vs_native", gate["criteria"])

    def test_dead_exposure_retains_denominator_and_cannot_revive(self):
        rows = [family(i) for i in range(4)]
        arm = rows[-1]["arms"]["contact_reversal"]
        for name in ("local_S", "solo_Y", "first_food", "first_poison", "exposure_survival", "solo_survival"):
            arm[name][:, 1] = 0
        arm["no_contact"][:, 1] = 1
        result = entry.summarize(rows)
        self.assertEqual(result["arms"]["contact_reversal"]["no_contact"][1]["mean"], .25)
        self.assertEqual(result["arms"]["contact_reversal"]["solo_Y"][1]["mean"], .75)
        arm["solo_survival"][:, 1] = 1
        with self.assertRaises(ValueError):
            entry.summarize(rows)
        arm["solo_survival"][:, 1] = 0
        arm["local_S"][:, 1] = .2
        with self.assertRaises(ValueError):
            entry.summarize(rows)

    def test_bad_category_nonfinite_wrong_axis_and_missing_assignment_are_rejected(self):
        mutations = [
            lambda rows: rows[0]["arms"]["contact_reversal"]["first_food"].fill(.5),
            lambda rows: rows[0]["pre_old"].fill(np.nan),
            lambda rows: rows[0]["arms"]["native_stationary"].update(local_S=np.zeros(2)),
            lambda rows: rows[0].update(base_index=9),
            lambda rows: rows[0].update(sparse_feedback_overlap=.2),
        ]
        for mutation in mutations:
            rows = [family(i) for i in range(4)]
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                entry.summarize(rows)


class ContactGateRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = dispatcher.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="contact-gate-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sources = []
        for index, source in enumerate(entry.SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.linspace(-1, 1, 155, dtype=np.float32)+np.float32(index), allow_pickle=False)
            sources.append(dict(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.sources = tuple(sources)
        patched = patch.object(entry, "SOURCES", self.sources)
        patched.start()
        self.addCleanup(patched.stop)
        self.args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development", "--name", "E133-benchmark",
                     "--seed", "2026121200", "--generations", "0", "--eval-worlds", "4", "--start", "file:"+self.sources[0]["path"]]
        self.calls, self.fail_at, self.now, self.expire, self.bad_source = [], None, 0., False, False

    @staticmethod
    def keys(master, index):
        return np.asarray([[master, 7*index+i] for i in range(7)], np.uint32)

    def assignments(self, master, count):
        return dict(family_keys=np.asarray([self.keys(master, i) for i in range(count)]), initial_rules=np.asarray([0, 1], np.int32))

    def evaluate(self, flat, keys, index, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate contact-gate interruption")
        self.calls.append((int(keys[0, 0]), index))
        if self.expire:
            self.now = 221.
        raw = raw_family(index, keys, flat, improve=int(keys[0, 0]) != 2026121203)
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
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys),
                                   ("evaluate_world", self.evaluate), ("summary_input", compact_family)):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            dispatcher.main()

    def folder(self, name="E133-benchmark"):
        return next((self.root / "runs").glob("*_"+name))

    def result(self, name="E133-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    def finals(self):
        for index in range(3):
            self.invoke("--condition-split", "final", "--name", f"E133-A{index}", "--seed", str(2026121201+index),
                        "--eval-worlds", "64", "--start", "file:"+self.sources[index]["path"])

    def audit(self):
        panels, checked = [], {}
        for index in range(3):
            folder = self.folder(f"E133-A{index}")
            panels.append(dict(index=index, run=folder.relative_to(self.root).as_posix(), measurements=self.result(f"E133-A{index}")["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.update({source["path"]: source["sha256"] for source in self.sources})
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))))

    def test_development_preserves_fixed_apparatus_source_and_raw_evidence(self):
        self.invoke()
        config, result = json.loads((self.folder() / "config.json").read_text()), self.result()
        self.assertEqual(self.calls, [(2026121200, index) for index in range(4)])
        self.assertEqual(config["source_genome"], self.sources[0])
        self.assertEqual(config["cfg"]["M"], 1)
        self.assertEqual(config["apparatus"], entry.APPARATUS)
        self.assertEqual(config["assignment_tag"], 0xE133)
        self.assertEqual(config["max_seconds"], 220)
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertFalse(result["scientific_result"])
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((self.folder() / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])

    def test_registration_dirty_source_and_duplicate_attempts_are_rejected(self):
        for flags in (("--seed", "2026121100"), ("--eval-worlds", "128"), ("--generations", "1"), ("--cfg", "M=4"),
                      ("--no-plastic",), ("--name", "E133-benchmark-retry1"), ("--allow-dirty",), ("--start", "file:"+self.sources[1]["path"])):
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

    def test_source_tampering_rejected_before_simulation_and_raw_identity_before_save(self):
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

    def test_interrupted_prefix_preserved_without_a_measurement(self):
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
        self.invoke("--condition-split", "final", "--name", "E133-A0", "--seed", "2026121201", "--eval-worlds", "64")
        (self.folder("E133-A0") / "summary.json").write_text("outcomes must remain unread")
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")

    def test_triplet_audit_and_exact_unchanged_bytes_required_for_report(self):
        self.finals()
        self.assertEqual(entry.build_comparison(self.root)["status"], "undecided")
        with patch.object(report, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            report.main()
        table = self.root / "results/RESULTS.md"
        self.assertIn("Replicated contact gate: **undecided**", table.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "results/diagnostics/E133-comparison.json").exists())
        self.audit()
        value = entry.build_comparison(self.root)
        self.assertEqual(value["replicated_gates"]["contact_gate"], dict(count=2, seeds=[0, 1], passed=False))
        with patch.object(report, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            report.main()
        self.assertIn("Replicated contact gate: **FAIL** (2/3 lineages)", table.read_text(encoding="utf-8"))
        self.assertIn("n=64", table.read_text(encoding="utf-8"))
        raw = self.folder("E133-A1") / "families/family_0000.npz"
        raw.write_bytes(raw.read_bytes()+b"changed after audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            entry.build_comparison(self.root)

    def test_audit_must_agree_with_summary_not_only_the_joint_boolean(self):
        self.finals()
        self.audit()
        path = self.root / entry.AUDIT_PATH
        audit = json.loads(path.read_text())
        audit["panels"][0]["measurements"]["components"]["pre_old"]["mean"] += .01
        path.write_text(json.dumps(audit))
        with self.assertRaises(ValueError):
            entry.build_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
