"""Independent E119 arithmetic and integrity checks using synthetic evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import audit_inherited_evolution as audit


CFG = dict(audit.base.CFG, G=4, M=2, T=4, R=0, H=3)


def arm_fields(turn=4, archive=None, mask=None):
    shape = (2, *audit.block_shape(CFG))
    archive = np.zeros(shape[1:], np.float32) if archive is None else archive
    mask = np.ones(shape, bool) if mask is None else mask
    fresh = np.zeros(shape, np.float32)
    initial = audit.overlay(fresh, mask, archive)
    shelter = np.zeros((4, 4), bool)
    shelter[0, 0] = True
    alive = np.ones((4, 2), bool)
    sheltered = np.zeros((4, 2), bool)
    sheltered[:, 1] = True
    if turn <= 4:
        sheltered[turn - 1, 0] = True
    inside = np.array([False, True])
    flags = np.zeros((4, 2), bool)
    fingerprint = np.ones((2, 32), np.uint8) if archive.any() else np.zeros((2, 32), np.uint8)
    return dict(food=flags.copy(), poison=flags.copy(), alive_before=alive.copy(), alive_after=alive.copy(),
        sheltered=sheltered, shelter=shelter, initial_sheltered=inside,
        first_entry=np.array([turn, 0]), fitness=np.array(0.), energy_lost=np.zeros((4, 2), np.float32),
        birth_connections=mask.sum((1, 2)), birth_sha256=np.zeros((2, 32), np.uint8),
        fresh_sha256=np.zeros((2, 32), np.uint8), initial_sha256=fingerprint,
        terminal_sha256=np.zeros((2, 32), np.uint8), fresh_sensor_weights=fresh,
        initial_sensor_weights=initial, terminal_sensor_weights=initial.copy(), sensor_mask=mask,
        persistence_active=flags.copy(), persistence_remembered=flags.copy(), persistence_tactic_attempted=flags.copy())


def synthetic_family(index=0):
    shape = audit.block_shape(CFG)
    zero, old, newest = (np.full(shape, value, np.float32) for value in (0, .2, .3))
    founder = np.zeros(155, np.float32)
    parent, child = np.ones(155, np.float32), np.full(155, 2, np.float32)
    inputs = dict(founder_empty=(founder, zero), parent_previous=(parent, old),
                  parent_latest=(parent, newest), child_previous=(child, old),
                  child_latest=(child, newest), child_empty=(child, zero))
    raw = dict(keys=np.arange(index * 4 + 1, index * 4 + 5, dtype=np.uint32).reshape(2, 2),
        steps=np.array(4), agents=np.array(2), rule=np.array(index % 2),
        initial_position=np.array([[1, 1], [0, 0]], np.int32))
    for arm, (genome, archive) in inputs.items():
        raw[arm + "_genome"], raw[arm + "_archive"] = genome.copy(), archive.copy()
        fields = arm_fields(1 if arm == "child_latest" else 4, archive)
        raw.update({arm + "_" + name: value for name, value in fields.items()})
    return raw, inputs


def synthetic_donors():
    genome = np.zeros(155, np.float32)
    old = np.full(audit.block_shape(CFG), .2, np.float32)
    raw = dict(genome=genome, archive_before=old, round_index=np.array(0), worlds=np.array(4),
        agents=np.array(2), steps=np.array(4), weight_bound=np.array(3., np.float32),
        keys=np.arange(1, 17, dtype=np.uint32).reshape(4, 2, 2), rules=np.arange(4) % 2,
        initial_positions=np.tile(np.array([[1, 1], [0, 0]], np.int32), (4, 1, 1)))
    worlds = []
    for index in range(4):
        mask = np.ones((2, *old.shape), bool)
        if index == 0:
            mask[1, 0, 0] = False
        row = arm_fields(4, old, mask)
        if index == 0:
            row["terminal_sensor_weights"][0, 0, 0] += np.float32(.7)
        worlds.append(row)
    raw.update({"donor_" + name: np.stack([row[name] for row in worlds]) for name in worlds[0]})
    raw["donor_mask"] = raw["donor_sensor_mask"].copy()
    raw["donor_delta"] = raw["donor_terminal_sensor_weights"] - raw["donor_initial_sensor_weights"]
    raw["donor_counts"] = raw["donor_mask"].sum((0, 1), dtype=np.int32)
    raw["archive_after"] = np.clip(old + raw["donor_delta"].sum((0, 1))
        / np.maximum(raw["donor_counts"], 1).astype(np.float32), -3, 3)
    return raw, genome, old


class InheritedEvolutionAuditTests(unittest.TestCase):
    def test_existing_optimizer_and_gate_boundaries(self):
        audit.self_check()

    def test_all_present_donors_include_zero_updates(self):
        raw, genome, old = synthetic_donors()
        report = audit.donor_archive(raw, genome, old, 0, CFG)
        self.assertEqual(report["donors"], 8)
        self.assertEqual(int(raw["donor_counts"][0, 0]), 7)
        self.assertAlmostEqual(float(raw["archive_after"][0, 0]), .3)
        self.assertEqual(report["changed_archive_entries"], 1)
        corrupt = copy.deepcopy(raw)
        corrupt["donor_counts"][0, 0] = 1
        with self.assertRaises(audit.base.AuditError):
            audit.donor_archive(corrupt, genome, old, 0, CFG)

    def test_rejects_recounted_inheritance_or_wrong_parent(self):
        raw, genome, old = synthetic_donors()
        for field in ("donor_delta", "archive_before", "genome", "donor_initial_sensor_weights"):
            corrupt = copy.deepcopy(raw)
            corrupt[field].flat[0] += .1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.donor_archive(corrupt, genome, old, 0, CFG)

    def test_six_arm_comparison_and_inside_birth_accounting(self):
        first, inputs = synthetic_family()
        second, _ = synthetic_family(1)
        rows = [audit.family_events(raw, index, inputs, CFG) for index, raw in enumerate((first, second))]
        result = audit.measurements(rows, agents=2)
        self.assertEqual(result["assigned_agents"], 4)
        self.assertEqual(result["outside_born_agents"], 2)
        contrast = result["contrasts"]["child_vs_parent"]
        self.assertEqual(contrast["latency_improvement"], dict(mean=3., se=0.))
        self.assertTrue(contrast["gate"]["passed"])
        self.assertEqual(set(result["contrasts"]), set(audit.CONTRASTS))

    def test_rejects_corrupted_weight_transfer_or_arrival(self):
        raw, inputs = synthetic_family()
        for field in ("child_latest_initial_sensor_weights", "child_latest_archive", "child_latest_first_entry",
                      "child_empty_genome", "parent_latest_fresh_sensor_weights"):
            corrupt = copy.deepcopy(raw)
            corrupt[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.family_events(corrupt, 0, inputs, CFG)

    def test_fatal_shelter_entry_counts_as_nonarrival(self):
        raw, inputs = synthetic_family()
        raw["child_latest_alive_after"][:, 0] = False
        raw["child_latest_alive_before"][1:, 0] = False
        raw["child_latest_first_entry"][0] = 5
        result = audit.family_events(raw, 0, inputs, CFG)
        self.assertEqual(result["child_latest"]["arrival"].tolist(), [5, 0])
        self.assertEqual(result["child_latest"]["deaths"], 1)

    def test_incomplete_three_seed_bundle_does_not_open_outcomes(self):
        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            folder = root / "runs" / "000_E119-A0"
            folder.mkdir(parents=True)
            (folder / "status.json").write_text(json.dumps(dict(status="training", generation=1)), encoding="utf-8")
            auditor = audit.Auditor.__new__(audit.Auditor)
            auditor.root, auditor.hashes, auditor.used_keys = root, {}, set()
            with mock.patch.object(auditor, "arrays", side_effect=AssertionError("opened outcomes")):
                with self.assertRaises(audit.base.Incomplete):
                    auditor.completed_bundle()
            self.assertEqual(set(auditor.hashes), {"runs/000_E119-A0/status.json"})


if __name__ == "__main__":
    unittest.main()
