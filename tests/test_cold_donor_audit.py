"""NumPy-only independent audit checks; no engine imports or saved run data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

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


class ColdDonorAuditTests(unittest.TestCase):
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
        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cold-donor-audit-", dir=scratch) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(scratch.resolve()))
            auditor = audit.Auditor.__new__(audit.Auditor)
            auditor.root, auditor.hashes = Path(directory), {}
            for index in range(3):
                folder = auditor.root / "runs" / f"20261121-010000_E117-A{index}"
                (folder / "families").mkdir(parents=True)
                status = dict(status="running" if index == 2 else "finished",
                              completed_families=0 if index == 2 else 256, expected_families=256)
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


if __name__ == "__main__":
    unittest.main()
