"""Synthetic evidence checks for the independent E120 auditor."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import audit_social_learning as audit


CFG = dict(audit.CFG, G=4, T=4, R=1, H=3, n_items=1)


def synthetic_family(index=0):
    steps, size, radius = CFG["T"], CFG["G"], CFG["R"]
    sensory = 4 * (2 * radius + 1) ** 2 + 3
    neurons = sensory + CFG["H"] + 6
    grid = np.zeros((size, size), np.int32)
    grid[0, 0], grid[2, 2] = 1, 2
    positions = np.tile(np.array([[[1, 1]]], np.int32), (steps, 1, 1))
    alive = np.ones((steps, 1), bool)
    false = np.zeros((steps, 1), bool)
    row = dict(keys=np.arange(index * 8 + 1, index * 8 + 9, dtype=np.uint32).reshape(4, 2),
        family_index=np.array(index), rule=np.array(index % 2), yoked_swap=np.array(index // 2 % 2),
        steps=np.array(steps), watch_start=np.array(steps // 2), agents=np.array(1),
        radius=np.array(radius), grid_size=np.array(size), hidden_units=np.array(CFG["H"]),
        initial_energy=np.array(CFG["e0"], np.float32),
        teacher_initial_grid=grid.copy(), teacher_initial_position=np.array([[1, 1]], np.int32),
        teacher_grid=np.tile(grid, (steps, 1, 1)), teacher_position=positions,
        teacher_signal=np.full((steps, 1), .5, np.float32),
        teacher_energy=(1. - .003 * np.arange(1, steps + 1, dtype=np.float32))[:, None],
        teacher_food=false.copy(), teacher_poison=false.copy(), teacher_alive_before=alive.copy(),
        teacher_alive_after=alive.copy(), solo_initial_grid=grid.copy(),
        solo_initial_position=np.array([[3, 3]], np.int32),
        teacher_birth_sha256=np.zeros(32, np.uint8), teacher_terminal_sha256=np.ones(32, np.uint8))
    frames, cameras, _, _ = audit.reconstruct_movie(row, CFG)
    row["camera_positions"] = cameras
    inputs = audit.control_frames(frames, index // 2 % 2, radius)
    initial = dict(W=np.zeros((neurons, neurons), np.float32), x=np.zeros(neurons, np.float32),
                   Wp=np.zeros((sensory, CFG["H"]), np.float32))
    for arm in audit.ARMS:
        p = arm + "_"
        row.update({p + "frames": inputs[arm], p + "watch_motor": np.tile([1, 0, 0, 0, 0], (2, 1)),
            p + "watch_denergy": np.zeros(2, np.float32), p + "birth_sha256": audit.fingerprint(initial["W"]).copy(),
            p + "food": false.copy(), p + "poison": false.copy(), p + "alive_before": alive.copy(),
            p + "alive_after": alive.copy(), p + "first_food": np.array(False), p + "first_poison": np.array(False),
            p + "no_meal": np.array(True), p + "first_meal_step": np.array(steps + 1),
            p + "fitness": np.array(0.), p + "weight_change_count": np.array(0),
            p + "solo_initial_energy": np.array([CFG["e0"]], np.float32),
            p + "solo_initial_alive": np.array([True]), p + "solo_initial_denergy": np.array([0.], np.float32)})
        for state, value in initial.items():
            fingerprint = audit.fingerprint(value)
            for moment in ("watch_initial", "watch_final", "solo_initial"):
                row[p + moment + "_" + state + "_sha256"] = fingerprint.copy()
    return row


class SocialLearningAuditTests(unittest.TestCase):
    def test_camera_control_inputs_use_only_visible_information(self):
        row = synthetic_family(2)
        result = audit.family_events(row, 2, CFG)
        self.assertEqual(result["visible_frames"], 2)
        pixels = (2 * CFG["R"] + 1) ** 2
        np.testing.assert_array_equal(row["observed_frames"][:, :pixels],
                                      row["yoked_frames"][:, pixels:2 * pixels])
        self.assertFalse(row["no_demo_frames"][:, :4 * pixels].any())
        self.assertFalse(row["scene_only_frames"][:, 2 * pixels:4 * pixels].any())
        for arm in audit.ARMS:
            np.testing.assert_array_equal(row[arm + "_frames"][:, -3:], [[0, 0, 1], [0, 0, 1]])

    def test_changed_demo_channels_reward_and_motor_are_rejected(self):
        for field, location in (("yoked_frames", (0, 0)), ("scene_only_frames", (0, 18)),
                                ("observed_watch_denergy", (0,)), ("no_demo_watch_motor", (0, 1)),
                                ("observed_solo_initial_energy", (0,)), ("no_demo_solo_initial_denergy", (0,))):
            with self.subTest(field=field):
                raw = synthetic_family(2)
                raw[field][location] += 1
                with self.assertRaises(audit.base.AuditError):
                    audit.family_events(raw, 2, CFG)

    def test_solo_continuity_birth_and_key_mismatches_are_rejected(self):
        for field in ("observed_birth_sha256", "no_demo_solo_initial_W_sha256", "scene_only_watch_initial_x_sha256"):
            with self.subTest(field=field):
                row = synthetic_family()
                row[field][0] ^= 1
                with self.assertRaises(audit.base.AuditError):
                    audit.family_events(row, 0, CFG)
        row = synthetic_family()
        row["keys"][2] = row["keys"][0]
        with self.assertRaises(audit.base.AuditError):
            audit.family_events(row, 0, CFG)

    def test_poison_first_and_no_meal_keep_every_newborn(self):
        row = synthetic_family()
        row["observed_poison"][0] = True
        row["observed_food"][2] = True
        row["observed_first_poison"] = np.array(True)
        row["observed_no_meal"] = np.array(False)
        row["observed_first_meal_step"] = np.array(1)
        result = audit.family_events(row, 0, CFG)
        self.assertFalse(result["observed"]["first_food"])
        self.assertTrue(result["observed"]["first_poison"])
        self.assertTrue(result["no_demo"]["no_meal"])
        summary = audit.measurements([result, result])
        self.assertEqual(summary["assigned_newborns"], 2)
        self.assertEqual(summary["arms"]["observed"]["first_food"]["mean"], 0.)
        self.assertEqual(summary["arms"]["observed"]["first_poison_count"], 2)
        self.assertEqual(summary["arms"]["no_demo"]["no_meal_count"], 2)

    def test_teacher_event_energy_and_dead_visibility_checks(self):
        row = synthetic_family()
        row["teacher_energy"][1, 0] += .1
        with self.assertRaises(audit.base.AuditError):
            audit.family_events(row, 0, CFG)
        row = synthetic_family()
        row["teacher_food"][0] = True
        with self.assertRaises(audit.base.AuditError):
            audit.family_events(row, 0, CFG)

    def test_paired_uncertainty_and_strict_gate_boundary(self):
        result = audit.family_events(synthetic_family(), 0, CFG)
        first, second = copy.deepcopy(result), copy.deepcopy(result)
        first["observed"]["first_food"] = True
        second["observed"]["first_food"] = False
        estimates = audit.measurements([first, second])
        for control in audit.CONTROLS:
            self.assertEqual(estimates["contrasts"][control]["first_food_difference"], dict(mean=.5, se=.5))
        self.assertFalse(audit.scientific_gate(dict(mean=2., se=1.), dict(mean=0., se=0.))["passed"])
        self.assertTrue(audit.scientific_gate(dict(mean=2.001, se=1.), dict(mean=-2., se=1.))["passed"])
        self.assertFalse(audit.scientific_gate(dict(mean=1., se=0.), dict(mean=-.1, se=0.))["passed"])

    def test_final_preflight_does_not_open_partial_outcomes(self):
        scratch = Path(__file__).resolve().parents[1] / ".tmp"
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            root = Path(name)
            folder = root / "runs" / "test_E120-A0"
            (folder / "families").mkdir(parents=True)
            (folder / "status.json").write_text(json.dumps(dict(status="running", completed_families=1,
                                                               expected_families=128)))
            (folder / "summary.json").write_text("must remain unopened")
            instance = audit.Auditor.__new__(audit.Auditor)
            instance.root, instance.hashes, instance.used_keys = root, {}, set()
            with self.assertRaises(audit.base.Incomplete):
                instance.completed_bundle(False)
            self.assertNotIn("runs/test_E120-A0/summary.json", instance.hashes)


if __name__ == "__main__":
    unittest.main()
