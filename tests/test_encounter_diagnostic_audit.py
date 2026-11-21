"""E127 recovery provenance and locked evidence selection; no simulator or real runs."""
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

from tools import audit_encounter_diagnostic as audit


def evidence(name):
    pin = audit.RECOVERIES[name]
    development = pin["expected"] == 4
    master = 2026120600 if development else 2026120601
    config = dict(protocol=audit.PROTOCOL, condition_split="development" if development else "final",
        generation=0, eval_seed=master, expected_families=pin["expected"], scientific_result=False,
        args=dict(name=pin["name"], seed=master, generations=0, eval_worlds=pin["expected"], allow_dirty=False, resume=None),
        git=dict(commit=pin["commit"], dirty=False))
    status = dict(status=pin["status"], completed_families=pin["completed"], expected_families=pin["expected"],
                  family_seconds=[1.] * pin["completed"])
    manifests = [dict(index=index, path=f"families/family_{index:04d}.npz", sha256=f"{index:064x}")
                 for index in range(pin["completed"])]
    summary = dict(status, protocol=audit.PROTOCOL, condition_split=config["condition_split"],
        evaluation_seed=master, source_label="A0", seed_index=0, scientific_result=False,
        artifact_validity="passed" if development else "incomplete",
        gate_status="not_applicable" if development else "undecided", raw_files=manifests)
    if development:
        summary["measurements"] = {"unused": "historical development outcomes are not reconstructed"}
    else:
        summary["error"] = "deliberate fixture integrity failure"
    records = [dict(family=index, pre_contacts=8, keys=np.arange(index * 12, index * 12 + 12).reshape(6, 2).tolist(),
                    raw_file=copy.deepcopy(manifest)) for index, manifest in enumerate(manifests)]
    return config, status, summary, records, [manifest["path"] for manifest in manifests]


class EncounterDiagnosticAuditTests(unittest.TestCase):
    def test_exact_failed_prefix_and_successful_old_rehearsal_are_preserved(self):
        for name in audit.RECOVERIES:
            with self.subTest(name=name):
                audit.recovery_evidence(*evidence(name), name)

    def test_invented_success_partial_counts_or_outcome_summary_are_rejected(self):
        changes = (
            lambda data: data[1].update(status="finished"),
            lambda data: data[1].update(completed_families=10),
            lambda data: data[2].update(measurements={}),
            lambda data: data[3].pop(),
            lambda data: data[4].pop(),
            lambda data: data[1]["family_seconds"].pop(),
        )
        for change in changes:
            with self.subTest(change=change):
                data = evidence(audit.FINAL_RECOVERY)
                change(data)
                with self.assertRaises(audit.base.AuditError):
                    audit.recovery_evidence(*data, audit.FINAL_RECOVERY)

    def test_source_assignment_and_manifest_corruption_are_rejected(self):
        changes = (
            lambda data: data[0]["git"].update(commit="0" * 40),
            lambda data: data[0]["args"].update(resume="old-checkpoint"),
            lambda data: data[0].update(recovery_of={}),
            lambda data: data[2]["raw_files"][0].update(index=1),
            lambda data: data[3][0]["raw_file"].update(sha256="f" * 64),
            lambda data: data[3][0]["keys"][0].__setitem__(0, -1),
        )
        for change in changes:
            with self.subTest(change=change):
                data = evidence(audit.FINAL_RECOVERY)
                change(data)
                with self.assertRaises(audit.base.AuditError):
                    audit.recovery_evidence(*data, audit.FINAL_RECOVERY)

    def scratch_auditor(self, root):
        checker = audit.Auditor.__new__(audit.Auditor)
        checker.root, checker.hashes, checker.preserved = root.resolve(), {}, {}
        return checker

    def test_preserved_archives_are_opaque_and_byte_corruption_is_rejected(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="encounter-audit-", dir=scratch) as directory:
            checker = self.scratch_auditor(Path(directory))
            raw = checker.root / "runs/old/families/family_0000.npz"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"opaque archive: never interpreted")
            with patch.object(audit.np, "load", side_effect=AssertionError("old family decoded")):
                proof = checker.inventory("runs/old")
                self.assertEqual(proof["files"]["families/family_0000.npz"], hashlib.sha256(raw.read_bytes()).hexdigest())
                checker.preserved["old"] = proof
                checker.verify_preserved()
                raw.write_bytes(b"corrupt archive")
                with self.assertRaises(audit.base.AuditError):
                    checker.verify_preserved()

    @staticmethod
    def create_panel(root, name):
        folder = root / "runs" / ("fixture_" + name)
        (folder / "families").mkdir(parents=True)
        (folder / "status.json").write_text(json.dumps(dict(status="finished", completed_families=128, expected_families=128)))
        (folder / "summary.json").write_bytes(b"must remain unopened until all panels are ready")
        for index in range(128):
            (folder / "families" / f"family_{index:04d}.npz").touch()
        return folder

    def test_original_a0_cannot_substitute_for_recovery_and_selection_is_status_only(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="encounter-audit-", dir=scratch) as directory:
            checker = self.scratch_auditor(Path(directory))
            for name in ("E127-A0", "E127-A1", "E127-A2"):
                self.create_panel(checker.root, name)
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle(False)
            self.create_panel(checker.root, audit.FINAL_RECOVERY)
            with patch.object(audit.np, "load", side_effect=AssertionError("outcome read during selection")):
                selected = checker.completed_bundle(False)
            self.assertEqual([Path(folder).name for folder in selected],
                             ["fixture_" + name for name in (audit.FINAL_RECOVERY, "E127-A1", "E127-A2")])
            self.assertTrue(all(path.endswith("status.json") for path in checker.hashes))

    def test_all_final_sources_are_checked_before_any_final_outcome(self):
        events = []

        class BrokenLastSource:
            def __init__(self, *_):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development=False):
                return ["development"] if development else ["A0-recovery", "A1", "A2"]

            def metadata(self, folder, *_):
                events.append(folder)
                if folder == "A2":
                    raise audit.base.AuditError("incorrect last source")
                return ()

            def panel(self, *_):
                raise AssertionError("a final outcome was opened before checking all sources")

        with patch.object(audit, "Auditor", BrokenLastSource), patch.object(audit.sys, "argv",
                ["audit", "--source-commit", "a" * 40]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main(), 1)
        self.assertEqual(events, ["development", "A0-recovery", "A1", "A2"])
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
