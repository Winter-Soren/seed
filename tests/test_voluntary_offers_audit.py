"""Small independent E132 fixtures; no simulator, PRNG or historical assay."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_voluntary_offers as audit


def probe_fixture(*, dead=False):
    _, source, _, _, _ = audit.synthetic_trace()
    source["alive"][:] = not dead
    source["energy"][:] = 0 if dead else 1
    placements = np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5) if a != b], np.int32)
    random = dict(probe_keys=np.arange(24, dtype=np.uint32).reshape(12, 2),
                  probe_action_keys=np.arange(100, 124, dtype=np.uint32).reshape(12, 2))
    result = {"source_" + name + "_sha256": source[name + "_sha256"].copy() for name in audit.NEURAL}
    result.update({"initial_" + name: np.repeat(source[name][None], 12, axis=0) for name in audit.BODY})
    result.update(performed=np.asarray(not dead), initial_neural_matches=np.ones((12, 4), bool),
        final_frozen_matches=np.ones((12, 3), bool), hebbian_enabled=np.zeros(12, bool),
        predictor_enabled=np.zeros(12, bool), feedback_applied=np.zeros(12, bool),
        observation=np.zeros((12, 1, 103), np.float32), decision_motor=np.zeros((12, 5), np.float32),
        logits=np.zeros((12, 5), np.float32), probabilities=np.zeros((12, 5), np.float32),
        sampled_action=np.full(12, -1, np.int32), action_key=np.zeros((12, 2), np.uint32))
    if not dead:
        result["initial_grid"] = np.stack([audit.encounter.pair_grid(pair) for pair in placements])
        result["initial_key"] = random["probe_keys"].copy()
        result["action_key"] = random["probe_action_keys"].copy()
        result["observation"] = np.stack([audit.encounter.observation(grid, np.asarray([8, 8]), 1, 0)[None]
                                          for grid in result["initial_grid"]])
        result["sampled_action"][:] = 0
        result["probabilities"][:] = .2
    return result, source, placements, random


class VoluntaryOffersAuditTests(unittest.TestCase):
    def test_torus_geometry_preserves_ordered_distinct_noncentral_colors(self):
        position = np.asarray([[0, 0]], np.int32)
        offsets = np.asarray([[15, 0], [0, 15]], np.int32)
        grid = audit.offered_grid(position, offsets)
        self.assertEqual((grid[15, 0], grid[0, 15], np.count_nonzero(grid)), (1, 2, 2))
        for invalid in ([[0, 0], [0, 15]], [[15, 0], [15, 0]], [[-1, 0], [0, 15]]):
            with self.subTest(offsets=invalid), self.assertRaises(audit.base.AuditError):
                audit.offered_grid(position, np.asarray(invalid, np.int32))

    def test_valid_voluntary_walk_consumes_each_color_once_per_block(self):
        values, initial, final, offsets, random = audit.synthetic_trace()
        audit.trace(values, initial, final, 0, 0, random, offsets=offsets)
        self.assertEqual((values["food"].sum(), values["poison"].sum(), int(final["t"])), (8, 8, 64))
        np.testing.assert_array_equal(final["counts"], [[[8, 8], [0, 0]]])

    def test_regrowth_color_swap_and_teleport_cannot_hide_behind_same_meal_counts(self):
        values, initial, final, offsets, random = audit.synthetic_trace()
        changes = [("grid_before", (1, 7, 8), 1), ("grid_before", (2, 8, 7), 1),
                   ("position", (2, 0, 0), 9)]
        for field, index, replacement in changes:
            altered = copy.deepcopy(values)
            altered[field][index] = replacement
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.trace(altered, initial, final, 0, 0, random, offsets=offsets)

    def test_forced_action_and_false_motor_or_energy_feedback_are_rejected(self):
        values, initial, final, offsets, random = audit.synthetic_trace()
        for field in ("sampled_action", "efference", "incoming_efference", "incoming_denergy", "carried_denergy"):
            altered = copy.deepcopy(values)
            altered[field].flat[5] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.trace(altered, initial, final, 0, 0, random, offsets=offsets)

    def test_current_rule_and_actual_resource_energy_must_agree(self):
        values, initial, final, offsets, random = audit.synthetic_trace()
        for field in ("food", "poison", "item", "unclipped_energy", "denergy", "rule", "key_after"):
            altered = copy.deepcopy(values)
            if altered[field].dtype.kind == "b":
                altered[field].flat[0] = ~altered[field].flat[0]
            else:
                altered[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.trace(altered, initial, final, 0, 0, random, offsets=offsets)

    def test_fatal_contact_counts_but_later_slots_are_fully_frozen(self):
        values, initial, final, offsets, random = audit.synthetic_trace(energy=.05, rule=1)
        audit.trace(values, initial, final, 1, 0, random, offsets=offsets)
        self.assertEqual((int(values["poison"].sum()), int(final["counts"].sum()), int(final["t"])), (1, 1, 1))
        self.assertTrue(values["skipped"][1:].all())
        self.assertTrue((values["denergy"][1:] == 0).all())
        self.assertTrue((values["carried_denergy"][1:] < 0).all())
        for field in ("grid", "key_after", "t", "alive_after", "denergy"):
            altered = copy.deepcopy(values)
            altered[field][2:3].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.trace(altered, initial, final, 1, 0, random, offsets=offsets)

    def test_empty_feedback_tick_preserves_true_pending_last_outcome(self):
        values, initial, final, _, random = audit.synthetic_trace()
        before = copy.deepcopy(final)
        for name, field in (("grid", "grid_before"), ("pos", "position_before"), ("energy", "energy_before"),
                            ("sig", "signal_before"), ("denergy", "incoming_denergy"), ("alive", "alive_before"),
                            ("key", "key_before"), ("t", "t")):
            before[name] = values[field][-1].copy()
        flush = {name: value[-1:].copy() for name, value in values.items()}
        audit.trace(flush, before, final, 0, 63, random, clear=True)
        flush["incoming_denergy"][:] = 0
        with self.assertRaises(audit.base.AuditError):
            audit.trace(flush, before, final, 0, 63, random, clear=True)

    def test_discarded_flush_cannot_replace_unflushed_continuation_clone(self):
        _, _, source, _, _ = audit.synthetic_trace()
        audit.same_state(copy.deepcopy(source), source, "unflushed clone")
        for field in ("t", "key", "denergy", "x_sha256", "W_sha256", "Wp_sha256", "mask_sha256"):
            changed = copy.deepcopy(source)
            changed[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.same_state(changed, source, "unflushed clone")

    def test_living_readout_requires_all_five_probabilities_and_actual_identity_proofs(self):
        probe, source, placements, random = probe_fixture()
        self.assertEqual(audit.local_probe(probe, source, placements, random, 0), 0.)
        for field in ("probabilities", "initial_neural_matches", "final_frozen_matches", "feedback_applied", "source_x_sha256"):
            changed = copy.deepcopy(probe)
            if field in ("initial_neural_matches", "final_frozen_matches"):
                changed[field].flat[0] = False
            elif field == "probabilities":
                changed[field][:, 0] = 0
            else:
                changed[field].flat[0] = 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.local_probe(changed, source, placements, random, 0)

    def test_dead_readout_is_unexecuted_not_uniform_or_resurrected(self):
        probe, source, placements, random = probe_fixture(dead=True)
        self.assertEqual(audit.local_probe(probe, source, placements, random, 1), 0.)
        for field in ("probabilities", "initial_alive", "performed"):
            changed = copy.deepcopy(probe)
            changed[field][...] = .2 if field == "probabilities" else 1
            with self.subTest(field=field), self.assertRaises(audit.base.AuditError):
                audit.local_probe(changed, source, placements, random, 1)

    def test_sample_se_collapses_rules_within_family_and_uses_paired_differences(self):
        rows = audit.synthetic_rows()
        for index, row in enumerate(rows):
            row["pre_old"] = np.asarray([index / 3, 1 - index / 3])
            row["arms"]["near_reversal"]["local_S"] = row["pre_old"].copy()
            good = float(index % 2)
            for arm in ("near_stationary", "sham_stationary"):
                row["arms"][arm]["first_food"][:] = good
                row["arms"][arm]["solo_Y"][:] = good
                row["arms"][arm]["no_contact"][:] = 1-good
        result = audit.summarize(rows)
        self.assertEqual(result["pre_old_local"], dict(mean=.5, se=0., n=4))
        self.assertGreater(result["by_initial_rule"]["0"]["pre_old_local"]["se"], 0)
        self.assertEqual(result["safeguards"]["stationary_solo"], dict(mean=0., se=0., n=4, lower_2se=0.))

    def test_feedback_death_keeps_exposure_survival_but_zeroes_all_readouts(self):
        rows = audit.synthetic_rows()
        arm = rows[0]["arms"]["near_reversal"]
        for field in ("local_S", "solo_Y", "first_food", "first_poison", "readout_survival", "solo_survival"):
            arm[field][:] = 0
        arm["no_contact"][:] = 1
        result = audit.summarize(rows)
        self.assertEqual(result["arms"]["near_reversal"]["exposure_survival"]["mean"], 1.)
        self.assertEqual(result["arms"]["near_reversal"]["readout_survival"]["mean"], .75)
        self.assertEqual(result["arms"]["near_reversal"]["solo_Y"], dict(mean=.75, se=.25, n=4))
        arm["first_food"][:] = 1
        with self.assertRaises(audit.base.AuditError):
            audit.summarize(rows)

    def test_solo_score_is_first_signed_item_not_last_or_total_meals(self):
        food, poison, alive = np.zeros(50, bool), np.zeros(50, bool), np.ones(50, bool)
        poison[0], food[1:] = True, True
        result = audit.encounter.grid_outcomes(food, poison, alive)
        self.assertEqual((result["first_food"], result["first_poison"], result["food_minus_poison"]), (0., 1., -1.))
        result = audit.encounter.grid_outcomes(np.zeros(50, bool), np.zeros(50, bool), np.zeros(50, bool))
        self.assertEqual((result["no_contact"], result["food_minus_poison"]), (1., 0.))

    def test_registered_floors_are_inclusive_and_evidence_and_survival_are_strict(self):
        result = audit.summarize(audit.synthetic_rows())
        components, safeguards = copy.deepcopy(result["components"]), copy.deepcopy(result["safeguards"])
        for name, floor in (("both_colors_gain", .10), ("solo_gain", .05)):
            components[name] = dict(mean=floor, se=0., n=4)
        self.assertTrue(audit.decisions(components, safeguards)["passed"])
        components["solo_gain"]["se"] = .025
        self.assertFalse(audit.decisions(components, safeguards)["criteria"]["solo_gain"])
        components["solo_gain"]["se"] = 0.
        safeguards["reversal_exposure_survival"]["lower_2se"] = -.05
        self.assertFalse(audit.decisions(components, safeguards)["criteria"]["reversal_survival_noninferiority"])

    def test_every_gate_must_hold_in_each_of_three_lineages(self):
        panels = [audit.summarize(audit.synthetic_rows()) for _ in range(3)]
        self.assertTrue(audit.replicated_gates(panels)["voluntary_access"]["passed"])
        panels[2]["components"]["adjusted_solo_gain"]["mean"] = 0.
        panels[2]["gate"] = audit.decisions(panels[2]["components"], panels[2]["safeguards"])
        self.assertEqual(audit.replicated_gates(panels)["voluntary_access"], dict(count=2, seeds=[0, 1], passed=False))
        with self.assertRaises(audit.base.AuditError):
            audit.replicated_gates(panels[:2])

    def test_all_three_statuses_precede_any_held_out_artifact_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(3):
                folder = root / "runs" / f"opaque_E132-A{index}"
                (folder / "families").mkdir(parents=True)
                (folder / "summary.json").write_text("must not be opened", encoding="utf-8")
                (folder / "status.json").write_text("{}", encoding="utf-8")
                for family in range(64):
                    (folder / "families" / f"family_{family:04d}.npz").touch()
            reads = []
            checker = object.__new__(audit.Auditor)
            checker.root = root
            checker.path = lambda name: root / name

            def status(path):
                reads.append(path)
                self.assertTrue(path.endswith("status.json"))
                return dict(status="finished" if "A2/" not in path else "running", completed_families=64, expected_families=64)

            checker.json = status
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle()
            self.assertEqual(len(reads), 3)

    def test_complete_archive_still_rejects_nonfinite_and_object_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checker = object.__new__(audit.Auditor)
            checker.path = lambda name: root / name
            checker.read = lambda name: (root / name).read_bytes()
            for bad in (np.asarray([np.nan]), np.asarray([object()], dtype=object)):
                np.savez(root / "raw.npz", evidence=bad)
                with self.subTest(dtype=bad.dtype), self.assertRaises((audit.base.AuditError, ValueError)):
                    checker.arrays("raw.npz")

    def test_main_has_fail_closed_partial_receipt_without_loading_outcomes(self):
        class Pending:
            def __init__(self, *args):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development):
                raise audit.base.Incomplete("one registered lineage is incomplete")

            def metadata(self, *args):
                raise AssertionError("partial final must not open metadata or outcomes")

        output = io.StringIO()
        with patch.object(audit, "Auditor", Pending), patch("sys.argv", ["audit", "--source-commit", "a" * 40]), contextlib.redirect_stdout(output):
            self.assertEqual(audit.main(), 2)
        receipt = json.loads(output.getvalue())
        self.assertEqual((receipt["status"], receipt["timing"]["completed_families"]), ("incomplete", 0))
        self.assertIsNone(receipt["replicated_gates"])


if __name__ == "__main__":
    unittest.main()
