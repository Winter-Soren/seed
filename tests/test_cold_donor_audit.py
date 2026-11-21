"""NumPy-only independent audit checks; no engine imports or saved run data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import audit_cold_donor as audit


def synthetic_family(index=0):
    """Hand-written events for four full-length slots, without simulating physics."""
    shape = (6, 129)
    row = dict(keys=np.arange(8 * index + 1, 8 * index + 9, dtype=np.uint32).reshape(4, 2),
               rule=np.array(index % 2), steps=np.array(300), agents=np.array(4),
               cold_start=np.array(150), weight_bound=np.array(3.),
               archive_before=np.zeros(shape, np.float32), archive_after=np.zeros(shape, np.float32),
               donor_delta=np.zeros((4, *shape), np.float32), donor_mask=np.ones((4, *shape), bool),
               donor_counts=np.full(shape, 4., np.float32), recipient_mask=np.ones((4, *shape), bool),
               recipient_birth=np.zeros((4, *shape), np.float32),
               recipient_initial_delta=np.zeros((4, *shape), np.float32))
    row["donor_delta"][0, 0, 0] = 1.
    row["archive_after"][0, 0] = .25
    row["recipient_initial_delta"][:, 0, 0] = .25
    for arm, arrival in (("donor", 301), ("inherited", 1), ("fresh", 4)):
        prefix = arm + "_"
        for event in audit.EVENTS:
            row[prefix + event] = np.full((300, 4), event.startswith("alive"), bool)
        if arrival <= 300:
            row[prefix + "sheltered"][arrival - 1] = True
        row[prefix + "first_entry"] = np.full(4, arrival, np.int32)
        row[prefix + "initial_sheltered"] = np.zeros(4, bool)
        row[prefix + "initial_position"] = np.full((4, 2), 8, np.int32)
        row[prefix + "shelter"] = np.zeros((16, 16), bool)
        row[prefix + "shelter"][:4, :4] = True
        row[prefix + "energy_lost"] = np.zeros((300, 4), np.float32)
        row[prefix + "fitness"] = np.array(0., np.float32)
        for label in ("birth", "initial", "terminal"):
            row[prefix + label + "_sha256"] = np.zeros((4, 32), np.uint8)
        row[prefix + "changed_connections"] = np.zeros(4, np.int32)
        row[prefix + "mean_abs_weight_change"] = np.zeros(4, np.float32)
    row["inherited_initial_sha256"][:, 0] = 1
    row["inherited_terminal_sha256"][:, 0] = 1
    row["donor_terminal_sha256"][0, 0] = 1
    row["donor_changed_connections"][0] = 1
    row["donor_mean_abs_weight_change"][0] = 1. / 183**2
    return row


def recovery_fixture(root):
    """Tiny byte artifacts exercise provenance; these are deliberately not worlds."""
    parent = root / audit.RECOVERY_PARENT
    folder_name = "runs/20261121-190000_E117-A2-recovery"
    folder = root / folder_name
    (parent / "families").mkdir(parents=True)
    (folder / "families").mkdir(parents=True)
    (folder / "parent").mkdir()
    old, new = "a" * 40, "b" * 40
    scientific = dict(seed="c" * 40, cold_donor="d" * 40)
    parent_config = dict(git=dict(commit=old), runtime=dict(python="synthetic"),
                         args=dict(name="E117-A2", resume=None, seed=2026112103))
    parent_status = dict(status="running", completed_families=181, expected_families=256, elapsed_seconds=181.)
    (parent / "config.json").write_text(json.dumps(parent_config), encoding="utf-8")
    (parent / "status.json").write_text(json.dumps(parent_status), encoding="utf-8")
    (parent / "genome_start.npy").write_bytes(b"synthetic genome")
    artifacts, lines = [], []
    for i in range(181):
        relative = f"families/family_{i:04d}.npz"
        data = f"synthetic family {i}".encode()
        (parent / relative).write_bytes(data)
        (folder / relative).write_bytes(data)
        artifact = dict(index=i, path=relative, sha256=hashlib.sha256(data).hexdigest())
        artifacts.append(artifact)
        lines.append(json.dumps(dict(family=i, rule=i % 2, raw_file=artifact, elapsed_seconds=float(i + 1))))
    prefix = ("\n".join(lines) + "\n").encode()
    (parent / "metrics.jsonl").write_bytes(prefix)
    (folder / "metrics.jsonl").write_bytes(prefix + b"new records are validated separately\n")
    guard = b"\n".join(json.dumps(row).encode() for row in [
        dict(event="invoked", expectedCommit=old, mode="A2", fullLimitSeconds=9900),
        dict(event="preflight", commit=old),
        dict(event="stopped", processStarted=True, wallSeconds=audit.RECOVERY_CHARGE, error="Call cancelled ")]) + b"\n"
    guard_path = root / audit.RECOVERY_GUARD
    guard_path.parent.mkdir(parents=True)
    guard_path.write_bytes(guard)
    (folder / "parent/guard.jsonl").write_bytes(guard)
    for name in ("config.json", "status.json"):
        (folder / "parent" / name).write_bytes((parent / name).read_bytes())
    pinned = {
        "RECOVERY_CONFIG_SHA": hashlib.sha256((parent / "config.json").read_bytes()).hexdigest(),
        "RECOVERY_STATUS_SHA": hashlib.sha256((parent / "status.json").read_bytes()).hexdigest(),
        "RECOVERY_METRICS_SHA": hashlib.sha256(prefix).hexdigest(),
        "RECOVERY_GUARD_SHA": hashlib.sha256(guard).hexdigest()}
    recovery = dict(version="cold-donor-recovery-v1", parent_run=audit.RECOVERY_PARENT,
        parent_source_commit=old, parent_config_sha256=pinned["RECOVERY_CONFIG_SHA"],
        parent_status_sha256=pinned["RECOVERY_STATUS_SHA"], parent_metrics_sha256=pinned["RECOVERY_METRICS_SHA"],
        parent_genome_sha256=audit.ORIGINALS[2][1], guard_path=audit.RECOVERY_GUARD,
        guard_sha256=pinned["RECOVERY_GUARD_SHA"], completed_families=181, first_new_family=181,
        parent_elapsed_seconds=audit.RECOVERY_CHARGE, parent_last_metric_elapsed_seconds=181.,
        scientific_source=scientific, copied_raw_files=artifacts)
    config = dict(git=dict(commit=new), runtime=parent_config["runtime"], cumulative_max_seconds=9900, recovery=recovery,
                  args=dict(parent_config["args"], name=audit.RECOVERY_NAME, resume=audit.RECOVERY_PARENT))
    summary = dict(elapsed_seconds=audit.RECOVERY_CHARGE + 50., invocation_elapsed_seconds=50., recovery=recovery,
        segments=[dict(kind="preserved_parent", source_commit=old, first_family=0, stop_family=181,
                       elapsed_seconds=audit.RECOVERY_CHARGE, timing_basis="supervisor_wall"),
                  dict(kind="recovery", source_commit=new, first_family=181, stop_family=256,
                       elapsed_seconds=50., timing_basis="runner_monotonic")])
    status = dict(status="finished", completed_families=256, expected_families=256,
                  elapsed_seconds=audit.RECOVERY_CHARGE + 50.1, invocation_elapsed_seconds=50.1)
    (folder / "status.json").write_text(json.dumps(status), encoding="utf-8")
    auditor = audit.Auditor.__new__(audit.Auditor)
    auditor.root, auditor.hashes, auditor.a2_recovery = root, {}, folder_name
    auditor.config = mock.Mock(return_value=parent_config)
    auditor.scientific_code = mock.Mock(return_value=scientific)
    return auditor, folder_name, config, summary, pinned


class ColdDonorAuditTests(unittest.TestCase):
    def scratch_directory(self):
        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        scratch.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix="cold-donor-audit-", dir=scratch)
        self.assertTrue(Path(directory.name).resolve().is_relative_to(scratch.resolve()))
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_hand_calculated_metrics_and_all_assigned_survival(self):
        raw = [synthetic_family(i) for i in range(2)]
        families = [audit.family_arrays(row, i) for i, row in enumerate(raw)]
        result = audit.measurements(families, raw)
        self.assertEqual(result["latency_improvement"], dict(mean=3., se=0.))
        self.assertEqual(result["survival_difference"], dict(mean=0., se=0.))
        self.assertEqual(result["outside_born_agents"], 8)
        self.assertEqual(result["donor"]["outdoor_cold_agent_steps"], dict(mean=600., se=0.))
        self.assertEqual(result["archive"]["recipients_changed"], 8)

    def test_fatal_entry_remains_nonarrival_in_denominator(self):
        row = synthetic_family()
        row["inherited_alive_after"][:, 0] = False
        row["inherited_alive_before"][1:, 0] = False
        row["inherited_first_entry"][0] = 301
        family = audit.family_arrays(row, 0)
        result = audit.measurements([family], [row])
        self.assertEqual(result["inherited"]["latency"], dict(mean=76., se=None))
        self.assertEqual(result["inherited"]["survival"], dict(mean=.75, se=None))
        self.assertEqual(audit.competing_outcomes([family])["inherited"]["outside_died_without_arrival"], 1)

    def test_birth_inside_excluded_from_latency_but_kept_in_survival(self):
        row = synthetic_family()
        for arm in ("inherited", "fresh"):
            row[arm + "_initial_sheltered"][0] = True
            row[arm + "_initial_position"][0] = [0, 0]
            row[arm + "_first_entry"][0] = 0
        family = audit.family_arrays(row, 0)
        result = audit.measurements([family], [row])
        self.assertEqual(result["assigned_agents"], 4)
        self.assertEqual(result["outside_born_agents"], 3)
        self.assertEqual(result["latency_improvement"]["mean"], 3.)
        self.assertEqual(audit.competing_outcomes([family])["fresh"]["born_inside"], 1)

    def test_corrupt_archive_pairing_death_and_cost_fail(self):
        original = synthetic_family()
        changes = (("archive_before", (0, 0), .1), ("archive_after", (0, 0), .3),
                   ("donor_counts", (0, 0), 3.), ("recipient_initial_delta", (0, 0, 0), .3),
                   ("inherited_birth_sha256", (0, 0), 1), ("inherited_first_entry", 0, 3),
                   ("fresh_alive_before", (1, 0), False), ("fresh_energy_lost", (0, 0), .03))
        for name, at, value in changes:
            with self.subTest(name=name):
                row = copy.deepcopy(original)
                row[name][at] = value
                with self.assertRaises(audit.AuditError):
                    audit.family_arrays(row, 0)

    def test_identical_initial_weights_require_identical_paired_events(self):
        row = synthetic_family()
        for key in ("archive_after", "donor_delta", "recipient_initial_delta",
                    "donor_changed_connections", "donor_mean_abs_weight_change",
                    "donor_terminal_sha256", "inherited_initial_sha256", "inherited_terminal_sha256"):
            row[key][:] = 0
        with self.assertRaisesRegex(audit.AuditError, "zero-overlay paired trajectories"):
            audit.family_arrays(row, 0)

    def test_all_three_completion_guard_reads_no_measurements(self):
        for recovered in (False, True):
            with self.subTest(recovered=recovered):
                auditor = audit.Auditor.__new__(audit.Auditor)
                auditor.root, auditor.hashes = self.scratch_directory(), {}
                for index in range(3):
                    name = audit.RECOVERY_NAME if recovered and index == 2 else f"E117-A{index}"
                    relative = f"runs/20261121-010000_{name}"
                    folder = auditor.root / relative
                    if recovered and index == 2:
                        auditor.a2_recovery = relative
                    (folder / "families").mkdir(parents=True)
                    status = dict(status="running" if index == 2 else "finished",
                                  completed_families=181 if index == 2 else 256, expected_families=256)
                    (folder / "status.json").write_text(json.dumps(status), encoding="utf-8")
                    (folder / "summary.json").write_text("THIS MUST NEVER BE PARSED", encoding="utf-8")
                    if index < 2:
                        for family in range(256):
                            (folder / "families" / f"family_{family:04d}.npz").touch()
                with self.assertRaisesRegex(audit.Incomplete, "no measurements opened"):
                    auditor.completed_bundle()
                self.assertTrue(all(path.endswith("/status.json") for path in auditor.hashes))

    def test_manual_uncertainty_and_gate_boundaries(self):
        audit.self_check()

    def test_recovery_reuses_exact_prefix_and_separate_compute_segments(self):
        auditor, folder, config, summary, pinned = recovery_fixture(self.scratch_directory())
        with mock.patch.multiple(audit, **pinned):
            report = auditor.recovery_provenance(folder, config, summary)
        self.assertEqual(report["copied_families"], 181)
        self.assertEqual(report["new_families"], 75)
        self.assertEqual(report["invocation_elapsed_seconds"], 50.)
        self.assertEqual(report["preserved_parent_elapsed_seconds"], audit.RECOVERY_CHARGE)

    def test_recovery_rejects_changed_prefix_raw_status_and_ledger(self):
        for relative, replacement in (
                ("metrics.jsonl", b"changed prefix\n"),
                ("families/family_0000.npz", b"changed raw file"),
                ("parent/status.json", b"changed status"),
                ("parent/guard.jsonl", b"changed ledger")):
            with self.subTest(relative=relative):
                auditor, folder, config, summary, pinned = recovery_fixture(self.scratch_directory())
                auditor.path(folder + "/" + relative).write_bytes(replacement)
                with mock.patch.multiple(audit, **pinned), self.assertRaises(audit.AuditError):
                    auditor.recovery_provenance(folder, config, summary)

    def test_recovery_rejects_new_parent_evidence_or_changed_compute_charge(self):
        for change in ("parent_summary", "charge", "scientific_source"):
            with self.subTest(change=change):
                auditor, folder, config, summary, pinned = recovery_fixture(self.scratch_directory())
                if change == "parent_summary":
                    auditor.path(audit.RECOVERY_PARENT + "/summary.json").write_text("{}", encoding="utf-8")
                elif change == "charge":
                    summary["elapsed_seconds"] += 1.
                else:
                    config["recovery"]["scientific_source"] = dict(seed="e" * 40, cold_donor="d" * 40)
                with mock.patch.multiple(audit, **pinned), self.assertRaises(audit.AuditError):
                    auditor.recovery_provenance(folder, config, summary)


if __name__ == "__main__":
    unittest.main()
