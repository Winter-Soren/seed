"""Small E133 audit fixtures; no simulator, PRNG, runs or evaluation worlds."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_contact_gate as audit


def visible(grid, position, energy, delta):
    """Literal five-by-five channels, independent of the audit observation helper."""
    observation = np.zeros(103, np.float32)
    for row in range(5):
        for column in range(5):
            item = grid[(position[0]+row-2) % 16, (position[1]+column-2) % 16]
            if item:
                observation[(int(item)-1)*25+row*5+column] = 1
    observation[100:] = (np.float32(energy)-np.float32(1), np.float32(4)*np.float32(delta), 1)
    return observation


def fixture_history(script, initial_rule, current_rule=None, initial=None):
    """Explicit scalar walk; no production transition or audit validator used."""
    post = initial is not None
    energy, delta = (np.float32(1), np.float32(0)) if not post else (initial["energy"][-1], initial["denergy"][-1])
    previous = np.zeros(103, np.float32) if not post else initial["observation"][-1]
    efference = np.zeros(5, np.float32) if not post else np.eye(5, dtype=np.float32)[initial["action"][-1]]
    position, grid = np.asarray([8, 8], np.int32), np.zeros((16, 16), np.int32)
    rows = []
    for tick in range(65 if post else 32):
        if post and (tick >= 32 or tick % 4 == 0):
            grid[:] = 0
        presenting = tick < 32 and tick % 4 == 2 if post else tick % 2 == 0
        if presenting:
            trial = 16+tick//4 if post else tick//2
            grid[:] = 0
            for color in (1, 2):
                cell = (position+audit.MOVES[script[trial, color-1]]) % 16
                grid[tuple(cell)] = color
        obs = visible(grid, position, energy, delta)
        if post:
            action = int(script[16+tick//4, tick % 4+1]) if tick < 32 and tick % 4 >= 2 else 0
        else:
            action = int(script[tick//2, 3+tick % 2])
        position = (position+audit.MOVES[action]) % 16
        item = int(grid[tuple(position)])
        rule = initial_rule if not post else current_rule
        new_energy = np.float32(energy-np.float32(.003))
        if item:
            new_energy = np.float32(new_energy + np.float32(.25 if item == 1+rule else -.25))
            grid[tuple(position)] = 0
        new_energy = np.float32(min(2., max(0., new_energy)))
        new_delta = np.float32(new_energy-energy)
        rows.append(dict(observation=obs, prior_observation=previous.copy(), prior_efference=efference.copy(),
            action=np.asarray(action, np.int32), item=np.asarray(item, np.int32), energy_before=np.asarray(energy),
            energy=np.asarray(new_energy), incoming_denergy=np.asarray(delta), denergy=np.asarray(new_delta),
            alive_before=np.asarray(True), alive_after=np.asarray(new_energy > 0), predictor=np.asarray(True)))
        energy, delta, previous, efference = new_energy, new_delta, obs, np.eye(5, dtype=np.float32)[action]
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def fixture_solo(grid, position, regrowth, rule):
    """Northward first food/poison, then clipping or death in the same grid."""
    grid, position = grid.copy(), position.copy()
    energy, live, rows = np.float32(1), True, []
    for tick in range(50):
        position = (position + audit.MOVES[1] * int(live)) % 16
        item = int(grid[tuple(position)])
        food, poison = bool(live and item == 1+rule), bool(live and item == 2-rule)
        updated = np.float32(energy - np.float32(.003) * np.float32(live))
        if food:
            updated = np.float32(updated+np.float32(.25))
        if poison:
            updated = np.float32(updated-np.float32(.25))
        updated = np.float32(min(2., max(0., updated)))
        after = bool(live and updated > 0)
        rows.append(dict(grid_action=np.asarray(1, np.int32), grid_item=np.asarray(item, np.int32),
            grid_food=np.asarray(food), grid_poison=np.asarray(poison), grid_alive_before=np.asarray(live),
            grid_alive_after=np.asarray(after), grid_energy_before=np.asarray(energy), grid_energy=np.asarray(updated),
            grid_denergy=np.asarray(np.float32(updated-energy)*np.float32(live)), grid_regrowth_cell=regrowth[tick]))
        if food or poison:
            grid[tuple(position)] = 0
            grid[tuple(regrowth[tick])] = item
        energy, live = updated, after
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def family_fixture():
    genome, keys = np.zeros(155, np.float32), np.arange(14, dtype=np.uint32).reshape(7, 2)
    script = np.asarray([[1, 3, 1, 1, 2], [1, 3, 2, 3, 4]] * 12, np.int32)
    offsets = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], np.int32)
    grid, position = np.zeros((16, 16), np.int32), np.asarray([8, 8], np.int32)
    grid[7, 8], grid[0, 0] = 1, 2
    regrowth = np.asarray([[(6-tick) % 16, 8] for tick in range(50)], np.int32)
    random = dict(script=script.copy(), sparse_offsets=offsets.copy(), solo_grid=grid.copy(),
                  solo_position=position.copy(), solo_regrowth_cell=regrowth)
    raw = dict(base_index=np.asarray(0, np.int32), keys=keys.copy(), initial_rules=np.asarray([0, 1], np.int32),
        genome_sha256=audit.fingerprint(genome), script=script, sparse_offsets=offsets, solo_grid=grid, solo_position=position)
    pre = [fixture_history(script, rule) for rule in (0, 1)]
    for field in ("observation", "action", "energy", "denergy", "item", "alive_after"):
        raw["pre_"+field] = np.stack([item[field] for item in pre])
    raw.update(pre_logits=np.zeros((2, 12, 5), np.float32), pre_probabilities=np.full((2, 12, 5), .2, np.float32),
        pre_alive=np.ones(2, bool), pre_neural_hashes=np.zeros((2, 4, 32), np.uint8))
    for arm in audit.ARMS:
        policy, history = arm.split("_")
        rows = [fixture_history(script, rule, rule if history == "stationary" else 1-rule, pre[rule]) for rule in (0, 1)]
        for field in rows[0]:
            raw[arm+"_"+field] = np.stack([row[field] for row in rows])
        enabled = np.zeros(65, bool)
        if policy == "native":
            enabled[:] = True
        elif policy == "sparse":
            enabled[np.arange(8)*4+offsets] = True
        else:
            enabled[3:32:4] = True
            if policy == "reopened":
                enabled[32:] = True
        raw[arm+"_hebbian"] = np.broadcast_to(enabled, (2, 65)).copy()
        raw[arm+"_weight_same"] = ~raw[arm+"_hebbian"]
        hashes = np.zeros((2, 2, 4, 32), np.uint8)
        hashes[:, :, :3] = 1
        if policy in ("native", "reopened"):
            hashes[:, 1, 0] = 2
        raw.update({arm+"_probe_logits": np.zeros((2, 2, 12, 5), np.float32),
            arm+"_probe_probabilities": np.full((2, 2, 12, 5), .2, np.float32),
            arm+"_probe_alive": np.ones((2, 2), bool), arm+"_checkpoint_hashes": hashes,
            arm+"_grid_initial_hashes": hashes.copy()})
        solo = [fixture_solo(grid, position, regrowth, rule if history == "stationary" else 1-rule) for rule in (0, 1)]
        for field in audit.GRID_FIELDS:
            raw[arm+"_"+field] = np.stack([np.stack([row[field]] * 2) for row in solo])
    return raw, genome, keys, random


class ContactGateAuditTests(unittest.TestCase):
    def test_complete_numeric_family_reconstructs_same_histories_and_solo_deaths(self):
        raw, genome, keys, random = family_fixture()
        row = audit.audit_family(raw, 0, genome, keys, random)
        np.testing.assert_array_equal(row["arms"]["contact_reversal"]["solo_Y"], [[-1, -1], [1, 1]])
        np.testing.assert_array_equal(row["arms"]["native_stationary"]["solo_survival"], [[1, 1], [0, 0]])
        self.assertEqual(row["sparse_feedback_overlap"], .25)
        measured = audit.summarize([row] * 4)
        self.assertFalse(measured["gate"]["passed"])
        self.assertEqual(measured["arms"]["contact_reversal"]["solo_Y"][1], dict(mean=0., se=0., n=4))

    def test_current_scene_cannot_replace_previous_observed_contact(self):
        raw, genome, keys, random = family_fixture()
        contact = raw["contact_reversal_observation"][0]
        efference = np.eye(5, dtype=np.float32)[raw["contact_reversal_action"][0]]
        self.assertTrue(audit.contact_mask(contact[2], efference[2], np.asarray(True)))
        self.assertFalse(raw["contact_reversal_hebbian"][0, 2])
        self.assertTrue(raw["contact_reversal_hebbian"][0, 3])
        for field, index, value in (("contact_reversal_hebbian", (0, 2), True),
                                   ("contact_reversal_prior_efference", (0, 3, 1), 0.),
                                   ("contact_reversal_prior_observation", (0, 3, 7), 0.)):
            altered = copy.deepcopy(raw)
            altered[field][index] = value
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(altered, 0, genome, keys, random)

    def test_realized_clipped_feedback_not_unbounded_resource_gain(self):
        raw, genome, keys, random = family_fixture()
        self.assertEqual(raw["native_stationary_grid_energy"][0, 0, 5], 2.)
        self.assertEqual(raw["native_stationary_grid_denergy"][0, 0, 5], 0.)
        for field, value in (("native_stationary_grid_denergy", .247), ("native_stationary_grid_energy", 2.247)):
            changed = copy.deepcopy(raw)
            changed[field][0, 0, 5] = value
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)
        previous, eff = np.zeros(103, np.float32), np.eye(5, dtype=np.float32)[1]
        previous[7] = 1
        self.assertTrue(audit.contact_mask(previous, eff, np.asarray(True)))
        previous[101] = 0
        self.assertTrue(audit.contact_mask(previous, eff, np.asarray(True)))

    def test_any_input_action_feedback_or_energy_mismatch_is_rejected(self):
        raw, genome, keys, random = family_fixture()
        mutations = (("pre_action", (0, 0)), ("pre_item", (0, 0)), ("pre_energy", (0, 0)),
            ("pre_denergy", (0, 0)), ("pre_observation", (0, 1, 101)),
            ("contact_reversal_action", (0, 3)), ("contact_reversal_item", (0, 2)),
            ("contact_reversal_observation", (0, 3, 101)), ("contact_reversal_incoming_denergy", (0, 3)),
            ("contact_reversal_energy_before", (0, 4)), ("contact_reversal_energy", (0, 4)),
            ("contact_reversal_denergy", (0, 4)))
        for field, index in mutations:
            changed = copy.deepcopy(raw)
            changed[field][index] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)

    def test_random_sparse_schedule_has_exact_dose_and_actual_overlap(self):
        raw, genome, keys, random = family_fixture()
        mask = raw["sparse_reversal_hebbian"]
        self.assertTrue((mask.sum(axis=1) == 8).all())
        self.assertFalse(mask[:, 32:].any())
        self.assertEqual(np.count_nonzero(mask[0, 3:32:4]), 2)
        for field, index in (("sparse_offsets", (0,)), ("sparse_reversal_hebbian", (0, 0))):
            changed = copy.deepcopy(raw)
            changed[field][index] = 1 if field == "sparse_offsets" else False
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)

    def test_disabled_weight_changes_and_predictor_freezing_are_rejected(self):
        raw, genome, keys, random = family_fixture()
        for field, index in (("contact_reversal_weight_same", (0, 0)), ("contact_reversal_predictor", (0, 0)),
                             ("sparse_reversal_weight_same", (0, 32))):
            changed = copy.deepcopy(raw)
            changed[field][index] = False
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)
        for field, index in (("contact_reversal_checkpoint_hashes", (0, 1, 0, 0)),
                             ("reopened_reversal_checkpoint_hashes", (0, 0, 1, 0)),
                             ("contact_reversal_grid_initial_hashes", (0, 1, 1, 0))):
            changed = copy.deepcopy(raw)
            changed[field][index] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)

    def test_solo_physics_rejects_wrong_label_regrowth_motion_or_revival(self):
        raw, genome, keys, random = family_fixture()
        for field, index in (("native_stationary_grid_food", (0, 0, 0)),
            ("native_stationary_grid_poison", (0, 0, 0)), ("native_stationary_grid_action", (0, 0, 0)),
            ("native_stationary_grid_item", (0, 0, 0)), ("native_stationary_grid_regrowth_cell", (0, 0, 0, 0)),
            ("native_stationary_grid_alive_before", (1, 0, 5)), ("native_stationary_grid_alive_after", (1, 0, 5))):
            changed = copy.deepcopy(raw)
            changed[field][index] = not changed[field][index] if changed[field].dtype.kind == "b" else changed[field][index]+1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)

    def test_dead_probe_placeholders_cannot_receive_fresh_body_or_choices(self):
        raw, _, _, random = family_fixture()
        dead = {name: np.zeros_like(raw["contact_reversal_"+name][0, 0]) for name in audit.GRID_FIELDS}
        result = audit.solo_trace(dead, raw["solo_grid"], raw["solo_position"], random["solo_regrowth_cell"], 0, False)
        self.assertEqual(result, dict(first_food=0., first_poison=0., no_contact=1., solo_Y=0., solo_survival=0.))
        for field in ("grid_alive_before", "grid_food", "grid_action", "grid_energy_before"):
            changed = copy.deepcopy(dead)
            changed[field][0] = 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.solo_trace(changed, raw["solo_grid"], raw["solo_position"], random["solo_regrowth_cell"], 0, False)
        zeros = np.zeros((12, 5), np.float32)
        self.assertEqual(audit.local_score(zeros, zeros, False, 0), 0.)
        with self.assertRaises(audit.base.AuditError):
            audit.local_score(zeros, np.full((12, 5), .2, np.float32), False, 0)

    def test_live_local_probabilities_keep_every_action_and_target_actual_rule(self):
        logits = np.zeros((12, 5), np.float32)
        logits[np.arange(12), audit.PLACEMENTS[:, 0]] = 1
        values = np.exp(logits.astype(np.float64))
        probabilities = (values / values.sum(axis=-1, keepdims=True)).astype(np.float32)
        self.assertGreater(audit.local_score(logits, probabilities, True, 0), 0)
        self.assertAlmostEqual(audit.local_score(logits, probabilities, True, 0), -audit.local_score(logits, probabilities, True, 1))
        probabilities[:, 0] = 0
        with self.assertRaises(audit.base.AuditError):
            audit.local_score(logits, probabilities, True, 0)

    def test_rules_collapse_inside_families_and_changes_use_paired_se(self):
        rows = audit.synthetic_rows()
        for index, row in enumerate(rows):
            row["pre_old"] = np.asarray([index/3, 1-index/3])
        measured = audit.summarize(rows)
        self.assertEqual(measured["components"]["pre_old"], dict(mean=.5, se=0., n=4))
        self.assertGreater(measured["components"]["gate_immediate"]["se"], 0)
        self.assertLess(measured["components"]["retention_advantage"]["se"], 1e-15)
        self.assertEqual(measured["families"], 4)

    def test_missing_acquisition_cannot_be_renamed_as_successful_retention(self):
        measured = audit.summarize(audit.synthetic_rows())
        self.assertTrue(measured["gate"]["passed"])
        for missing in ("pre_old", "native_immediate", "gate_immediate", "gate_delayed"):
            components = copy.deepcopy(measured["components"])
            components[missing] = dict(mean=0., se=0., n=4)
            with self.subTest(missing=missing):
                self.assertFalse(audit.decisions(components, measured["safeguards"])["passed"])

    def test_all_effect_floors_are_inclusive_but_evidence_and_safeguards_strict(self):
        measured = audit.summarize(audit.synthetic_rows())
        for name, floor in audit.COMPONENT_FLOORS.items():
            components = copy.deepcopy(measured["components"])
            value = floor if floor else .2
            components[name] = dict(mean=value, se=0., n=4)
            self.assertTrue(audit.decisions(components, measured["safeguards"])["criteria"][name])
            components[name]["se"] = value/2
            self.assertFalse(audit.decisions(components, measured["safeguards"])["criteria"][name])
        for name in audit.SAFEGUARD_NAMES:
            safeguards = copy.deepcopy(measured["safeguards"])
            safeguards[name]["lower_2se"] = -.05
            self.assertFalse(audit.decisions(measured["components"], safeguards)["criteria"][name])

    def test_descriptive_interactions_and_overlap_do_not_change_registered_gate(self):
        measured = audit.summarize(audit.synthetic_rows())
        for field in ("adjusted_solo_vs_native", "adjusted_solo_vs_sparse", "sparse_feedback_overlap"):
            measured["components"][field] = dict(mean=-100, se=100, n=4)
        self.assertTrue(audit.decisions(measured["components"], measured["safeguards"])["passed"])

    def test_dead_assignments_remain_zero_scored_in_family_denominator(self):
        rows = audit.synthetic_rows()
        dead = rows[0]["arms"]["contact_reversal"]
        for field in ("local_S", "solo_Y", "first_food", "first_poison", "exposure_survival", "solo_survival"):
            dead[field][:] = 0
        dead["no_contact"][:] = 1
        measured = audit.summarize(rows)
        self.assertEqual(measured["arms"]["contact_reversal"]["solo_Y"][1], dict(mean=.75, se=.25, n=4))
        dead["first_food"][:] = 1
        with self.assertRaises(audit.base.AuditError):
            audit.summarize(rows)

    def test_all_registered_conditions_must_hold_in_each_original_lineage(self):
        panels = [audit.summarize(audit.synthetic_rows()) for _ in range(3)]
        self.assertTrue(audit.replicated_gates(panels)["contact_gate"]["passed"])
        panels[2]["components"]["delay_ablation"]["mean"] = 0.
        panels[2]["gate"] = audit.decisions(panels[2]["components"], panels[2]["safeguards"])
        self.assertEqual(audit.replicated_gates(panels)["contact_gate"], dict(count=2, seeds=[0, 1], passed=False))
        with self.assertRaises(audit.base.AuditError):
            audit.replicated_gates(panels[:2])

    def test_strict_raw_schema_and_numeric_dtypes_reject_missing_or_hidden_inputs(self):
        raw, genome, keys, random = family_fixture()
        for mutation in ("missing", "hidden", "float64", "nonfinite", "genome"):
            changed = copy.deepcopy(raw)
            if mutation == "missing":
                del changed["contact_reversal_weight_same"]
            elif mutation == "hidden":
                changed["hidden_rule"] = np.asarray(1, np.int32)
            elif mutation == "float64":
                changed["contact_reversal_observation"] = changed["contact_reversal_observation"].astype(np.float64)
            elif mutation == "nonfinite":
                changed["pre_energy"][0, 0] = np.nan
            else:
                changed["genome_sha256"][0] += 1
            with self.subTest(mutation=mutation), self.assertRaises(audit.base.AuditError):
                audit.audit_family(changed, 0, genome, keys, random)

    def test_all_three_statuses_are_checked_before_opening_final_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(3):
                folder = root / "runs" / f"opaque_E133-A{index}"
                (folder / "families").mkdir(parents=True)
                (folder / "summary.json").write_text("must not be opened", encoding="utf-8")
                (folder / "status.json").write_text("{}", encoding="utf-8")
                for family in range(64):
                    (folder / "families" / f"family_{family:04d}.npz").touch()
            reads = []
            checker = object.__new__(audit.Auditor)
            checker.root, checker.path = root, lambda name: root / name

            def status(path):
                reads.append(path)
                self.assertTrue(path.endswith("status.json"))
                return dict(status="finished" if "A2/" not in path else "running", completed_families=64, expected_families=64)

            checker.json = status
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle()
            self.assertEqual(len(reads), 3)

    def test_archive_payloads_cannot_hide_objects_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checker = object.__new__(audit.Auditor)
            checker.path, checker.read = lambda name: root / name, lambda name: (root / name).read_bytes()
            for bad in (np.asarray([np.nan]), np.asarray([object()], dtype=object)):
                np.savez(root / "raw.npz", evidence=bad)
                with self.subTest(dtype=bad.dtype), self.assertRaises((audit.base.AuditError, ValueError)):
                    checker.arrays("raw.npz")

    def test_main_fails_closed_without_opening_partial_measurements(self):
        class Pending:
            def __init__(self, *args):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development):
                raise audit.base.Incomplete("a registered panel remains incomplete")

            def metadata(self, *args):
                raise AssertionError("must not open partial measurements")

        output = io.StringIO()
        with patch.object(audit, "Auditor", Pending), patch("sys.argv", ["audit", "--source-commit", "a"*40]), contextlib.redirect_stdout(output):
            self.assertEqual(audit.main(), 2)
        report = json.loads(output.getvalue())
        self.assertEqual((report["status"], report["timing"]["completed_families"]), ("incomplete", 0))
        self.assertIsNone(report["replicated_gates"])

    def test_final_audit_reconstructs_development_and_all_three_final_panels(self):
        calls, measured = [], audit.summarize(audit.synthetic_rows())

        class Complete:
            def __init__(self, *args):
                self.hashes, self.family_seconds, self.expected_trees = {}, [], {}

            def completed_bundle(self, development):
                calls.append(("bundle", development))
                return ["development"] if development else ["A0", "A1", "A2"]

            def metadata(self, folder, index, development):
                calls.append(("metadata", folder))
                return None

            def panel(self, folder, index, development, prepared):
                calls.append(("reconstruct", folder))
                self.family_seconds.extend([0.] * (4 if development else 64))
                return dict(index=index, run=folder, measurements=measured)

        output = io.StringIO()
        with patch.object(audit, "Auditor", Complete), patch("sys.argv", ["audit", "--source-commit", "a"*40]), contextlib.redirect_stdout(output):
            self.assertEqual(audit.main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(calls[0], ("bundle", False))
        self.assertEqual([value for kind, value in calls if kind == "reconstruct"], ["development", "A0", "A1", "A2"])
        self.assertEqual(report["timing"]["completed_families"], 196)
        self.assertEqual(len(report["panels"]), 3)


if __name__ == "__main__":
    unittest.main()
