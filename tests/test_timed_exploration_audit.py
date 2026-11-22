"""E131 independent timing/quota evidence, clustered statistics and admission."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_timed_exploration as audit


class TimedExplorationAuditTests(unittest.TestCase):
    def test_birth_has_no_feedback_and_twelfth_failed_outcome_triggers_next_action(self):
        life, schedule, _ = audit.synthetic_life()
        self.assertEqual(life["counter"][0, 0], 0)
        self.assertEqual(life["counter"][11, 0], 11)
        self.assertFalse(life["override"][11, 0])
        self.assertTrue(life["override"][12, 0])
        observed = life["observed_denergy"].copy()
        observed[12, 0] = .1
        value = audit.reconstruct_policy("adaptive", observed, life["alive_before"], life["ordinary_action"], schedule)
        self.assertEqual(value["counter"][12, 0], 0)
        self.assertFalse(value["override"][12, 0])
        self.assertTrue(value["override"][24, 0])

    def test_phase_delay_and_cross_block_counter_are_not_reset_at_reversal(self):
        life, schedule, keys = audit.synthetic_life()
        audit.audit_life(life, "adaptive", schedule, keys)
        self.assertFalse(life["override"][:24, 3].any())
        self.assertTrue(life["override"][24, 3])
        self.assertTrue(life["override"][25, 0])
        self.assertEqual(life["counter_before"][25, 0], 12)
        self.assertFalse(life["spent_before"][25, 0])
        self.assertEqual(life["block_offset"][150].tolist(), [0, 20, 12, 1])

    def test_forced_last_tick_and_last_partial_block_have_no_horizon_exception(self):
        life, schedule, _ = audit.synthetic_life()
        observed = np.full((300, 4), .1, np.float32)
        value = audit.reconstruct_policy("adaptive", observed, life["alive_before"], life["ordinary_action"], schedule)
        self.assertTrue(value["override"][24, 0])
        self.assertTrue(value["override"][48, 3])
        self.assertFalse(value["override"][299, 3])
        self.assertEqual(int(value["override"][:, 3].sum()), 11)

    def test_random_quota_offsets_ignore_feedback_and_use_the_same_assigned_directions(self):
        life, schedule, _ = audit.synthetic_life("random_quota")
        first = audit.reconstruct_policy("random_quota", np.zeros((300, 4), np.float32), life["alive_before"], life["ordinary_action"], schedule)
        second = audit.reconstruct_policy("random_quota", np.ones((300, 4), np.float32), life["alive_before"], life["ordinary_action"], schedule)
        np.testing.assert_array_equal(first["override"], second["override"])
        self.assertEqual(np.flatnonzero(first["override"][:, 1]).tolist(), list(range(12, 300, 25)))
        np.testing.assert_array_equal(first["direction"], life["direction"])

    def test_death_truncated_blocks_allow_zero_and_never_override_dead_animals(self):
        poison = np.zeros((300, 4), bool)
        poison[:4] = True
        life, schedule, keys = audit.synthetic_life(poison=poison)
        audit.audit_life(life, "adaptive", schedule, keys)
        self.assertFalse(life["alive_before"][4:].any())
        self.assertFalse(life["override"].any())
        self.assertEqual(life["counter"][4, 0], life["counter"][3, 0])

    def test_food_clipped_at_energy_cap_is_not_a_positive_realized_outcome(self):
        food = np.zeros((300, 4), bool)
        food[:30] = True
        life, schedule, keys = audit.synthetic_life(food=food)
        audit.audit_life(life, "adaptive", schedule, keys)
        self.assertTrue(life["food"][6, 0])
        self.assertEqual(life["observed_denergy"][6, 0], 0.)
        self.assertEqual(life["counter"][6, 0], 1)
        altered = copy.deepcopy(life)
        altered["observed_denergy"][6, 0] = .25
        with self.assertRaises(audit.base.AuditError):
            audit.audit_life(altered, "adaptive", schedule, keys)

    def test_quota_counter_actual_action_and_next_tick_efference_corruption_are_rejected(self):
        life, schedule, keys = audit.synthetic_life()
        for field, index in (("counter", (12, 0)), ("spent_before", (25, 0)), ("active", (0, 3)),
                             ("action", (12, 0)), ("efference", (12, 0, 0)), ("incoming_efference", (13, 0, 0)),
                             ("action_key", (12, 0))):
            with self.subTest(field=field):
                altered = copy.deepcopy(life)
                altered[field][index] = not bool(altered[field][index]) if altered[field].dtype == np.bool_ else altered[field][index] + 1
                with self.assertRaises(audit.base.AuditError):
                    audit.audit_life(altered, "adaptive", schedule, keys)

    def test_an_override_can_be_the_same_as_the_ordinary_draw(self):
        life, schedule, keys = audit.synthetic_life()
        audit.audit_life(life, "adaptive", schedule, keys)
        self.assertGreater(int(life["override"][:, 0].sum()), 0)
        self.assertEqual(int(life["override_changed"][:, 0].sum()), 0)
        changed = copy.deepcopy(life)
        changed["override_changed"][:, 0] = changed["override"][:, 0]
        with self.assertRaises(audit.base.AuditError):
            audit.audit_life(changed, "adaptive", schedule, keys)

    def test_disabled_observational_path_requires_native_action_and_inactive_state(self):
        life, schedule, keys = audit.synthetic_life("unchanged")
        audit.audit_life(life, "unchanged", schedule, keys)
        np.testing.assert_array_equal(life["action"], life["ordinary_action"])
        self.assertFalse(life["override"].any())
        life["active"][150, 0] = True
        with self.assertRaises(audit.base.AuditError):
            audit.audit_life(life, "unchanged", schedule, keys)

    def test_food_latency_counts_every_animal_and_uses_zero_based_fifty_turn_window(self):
        food, poison = np.zeros((2, 300, 4), bool), np.zeros((2, 300, 4), bool)
        before, after = np.ones_like(food), np.ones_like(food)
        food[:, 150, 0], food[:, 199, 1], food[:, 200, 2] = True, True, True
        poison[:, 151, 1] = True
        after[:, 10:, 3], before[:, 11:, 3] = False, False
        values = audit.event_outcomes(food, poison, before, after, np.zeros_like(food))
        self.assertEqual(values["first_food_latency"][0].tolist(), [0, 49, 50, 50])
        self.assertEqual(values["first_any_latency"][0].tolist(), [0, 1, 50, 50])
        self.assertEqual(values["Q"][0, 3].tolist(), [1, -1, 0, 0])

    def test_family_paired_statistics_and_override_ratios_do_not_use_pseudoreplicates(self):
        rows = audit.synthetic_rows()
        for row in rows:
            for arm in audit.ARMS:
                row[arm]["live_turns"][:, 0] = 100
        result = audit.summarize(rows)
        value = result["components"]["adaptive_reversal_late"]
        self.assertEqual(value["n"], 4)
        self.assertAlmostEqual(value["se"], np.std([.25, .5, .75, 1.], ddof=1) / 2)
        self.assertEqual(result["comparisons"]["ordinary"]["late_Q_difference"]["se"], 0.)
        self.assertAlmostEqual(result["arms"]["adaptive_reversal"]["override_live_fraction"]["mean"], 96 / 2000)
        self.assertEqual(result["by_initial_rule"]["0"]["comparisons"]["ordinary"]["latency_saved"]["n"], 4)

    def test_stationary_deterioration_cannot_replace_direct_late_reversal_improvement(self):
        result = audit.summarize(audit.synthetic_rows())
        comparisons = copy.deepcopy(result["comparisons"])
        comparisons["ordinary"]["late_Q_difference"] = dict(mean=0., se=0., n=4)
        gate = audit.decisions(result["components"], comparisons)
        self.assertTrue(gate["criteria"]["ordinary_adjusted_Q_improves"])
        self.assertFalse(gate["passed"])

    def test_effect_floor_evidence_and_survival_bounds_keep_their_registered_edges(self):
        result = audit.summarize(audit.synthetic_rows())
        comparisons = copy.deepcopy(result["comparisons"])
        comparisons["ordinary"]["adjusted_late_Q"] = dict(mean=.05, se=0., n=4)
        self.assertTrue(audit.decisions(result["components"], comparisons)["passed"])
        comparisons["ordinary"]["adjusted_late_Q"]["se"] = .025
        self.assertFalse(audit.decisions(result["components"], comparisons)["criteria"]["ordinary_adjusted_Q_improves"])
        comparisons["ordinary"]["adjusted_late_Q"]["se"] = 0.
        comparisons["quota"]["safeguards"]["reversal_end_survival"]["lower_2se"] = -.05
        self.assertFalse(audit.decisions(result["components"], comparisons)["criteria"]["quota_reversal_survival_noninferiority"])

    def test_both_controls_must_qualify_in_every_one_of_three_lineages(self):
        panels = [audit.summarize(audit.synthetic_rows()) for _ in range(3)]
        panels[-1]["comparisons"]["quota"]["latency_saved"] = dict(mean=0., se=0., n=4)
        panels[-1]["gate"] = audit.decisions(panels[-1]["components"], panels[-1]["comparisons"])
        self.assertEqual(audit.replicated_gates(panels)["timed_exploration"], dict(count=2, seeds=[0, 1], passed=False))

    def test_runtime_admission_uses_all_positive_records_and_only_three_warm_families(self):
        times = np.asarray([[100., 1000., 100.], [1., 1.5, 2.], [2., 3., 4.], [3., 4.5, 6.]])
        result = audit.operational_cost(times)
        self.assertEqual(result["ratio"], 1.5)
        self.assertEqual(result["quota_ratio"], 2.)
        self.assertTrue(result["passed"])
        times[2, 1] += .01
        self.assertFalse(audit.operational_cost(times)["passed"])
        times[0, 2] = 0.
        with self.assertRaises(audit.base.AuditError):
            audit.operational_cost(times)

    def test_old_complete_panels_do_not_unlock_new_worlds(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="timed-audit-", dir=scratch) as directory:
            checker = audit.Auditor.__new__(audit.Auditor)
            checker.root, checker.hashes = Path(directory), {}
            for index in range(3):
                folder = checker.root / f"runs/synthetic_E130-A{index}"
                folder.mkdir(parents=True)
                (folder / "status.json").write_text(json.dumps(dict(status="finished", completed_families=64, expected_families=64)))
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle(False)
            self.assertEqual(checker.hashes, {})

    def test_all_source_metadata_precedes_outcomes_and_failed_cost_blocks_final_read(self):
        for failure in ("source", "runtime"):
            seen = []

            class Stub:
                def __init__(self, *_):
                    self.hashes, self.family_seconds = {}, []

                def completed_bundle(self, development=False):
                    return ["development"] if development else ["A0", "A1", "A2"]

                def metadata(self, folder, *_):
                    seen.append("metadata:" + folder)
                    if folder == "A2" and failure == "source":
                        raise audit.base.AuditError("wrong source")
                    return ()

                def panel(self, folder, *_):
                    seen.append("outcomes:" + folder)
                    if folder != "development":
                        raise AssertionError("final outcomes opened without admission")
                    return dict(operational_cost=dict(passed=False))

            with self.subTest(failure=failure), patch.object(audit, "Auditor", Stub), patch.object(audit.sys, "argv",
                    ["audit", "--source-commit", "a" * 40]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), 1)
            self.assertEqual(seen[:4], ["metadata:development", "metadata:A0", "metadata:A1", "metadata:A2"])
            self.assertEqual(seen[4:], [] if failure == "source" else ["outcomes:development"])


if __name__ == "__main__":
    unittest.main()
