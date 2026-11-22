"""E129 independent intervention evidence, rescue hierarchy and E128 compatibility."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_retention_rescue as audit
from test_retention_diagnostic_audit import empty_interval_fixture, rows_fixture


def frozen_interval_fixture():
    acquired, states, trace, key = empty_interval_fixture()
    trace["hebbian_enabled"][:] = False
    trace["W_input_matches_acquired"] = np.ones(33, bool)
    trace["W_output_matches_acquired"] = np.ones(33, bool)
    return acquired, [states[0], states[2]], states[1], trace, key


class RetentionRescueAuditTests(unittest.TestCase):
    def test_e128_default_statistics_and_active_interval_are_unchanged(self):
        rows = rows_fixture()
        self.assertEqual(audit.retention.summarize(rows), audit.retention.summarize(rows, delays=(0, 1, 33)))
        acquired, states, trace, key = empty_interval_fixture()
        before = copy.deepcopy(trace)
        audit.retention.active_interval(acquired, states, trace, key, True, 1)
        audit.retention.active_interval(acquired, states, trace, key, True, 1, hebbian=True)
        for field in trace:
            self.assertEqual(trace[field].tobytes(), before[field].tobytes())
        self.assertEqual(audit.retention.Auditor.RUN_PREFIX, "E128")
        self.assertEqual(audit.retention.Auditor.DELAYS, (0, 1, 33))
        self.assertEqual(audit.retention.Auditor.ARMS, audit.retention.ARMS)

    def test_frozen_interval_keeps_predictor_active_and_allows_predictor_change(self):
        acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
        endpoints[-1]["Wp_sha256"][0] ^= 1
        audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=True)
        self.assertFalse(trace["hebbian_enabled"].any())
        self.assertTrue(trace["predictor_enabled"].all())

    def test_each_turn_input_output_and_actual_update_masks_are_required(self):
        for field, tick in (("W_input_matches_acquired", 0), ("W_output_matches_acquired", 32),
                            ("hebbian_enabled", 0), ("predictor_enabled", 17)):
            with self.subTest(field=field):
                acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
                trace[field][tick] = not bool(trace[field][tick])
                with self.assertRaises(audit.base.AuditError):
                    audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=True)

    def test_true_weight_predicates_cannot_hide_changed_endpoint_or_continuation(self):
        for target in ("endpoint", "continuation"):
            acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
            state = endpoints[-1] if target == "endpoint" else continuation
            state["W_sha256"][0] ^= 1
            with self.assertRaises(audit.base.AuditError):
                audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=True)

    def test_normal_path_refuses_a_frozen_mask_and_fake_weight_predicate_dtype(self):
        acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
        with self.assertRaises(audit.base.AuditError):
            audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=False)
        trace["W_input_matches_acquired"] = np.ones(33, np.int32)
        with self.assertRaises(audit.base.AuditError):
            audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=True)

    def test_two_behavioral_endpoints_and_real_one_turn_continuation_are_distinct(self):
        acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
        continuation["t"] = np.asarray(64, np.int32)
        with self.assertRaises(audit.base.AuditError):
            audit.interval(acquired, endpoints, continuation, trace, key, True, 1, frozen=True)
        acquired, endpoints, continuation, trace, key = frozen_interval_fixture()
        with self.assertRaises(audit.base.AuditError):
            audit.interval(acquired, [endpoints[0], continuation, endpoints[-1]], continuation, trace, key, True, 1, frozen=True)

    def test_primary_floor_is_inclusive_and_paired_uncertainty_is_strict(self):
        frozen = dict(local=dict(passed=True), grid=dict(passed=True))
        at_floor = audit.intervention_gates(dict(mean=.025, se=0), dict(mean=.1, se=0), frozen)
        at_error = audit.intervention_gates(dict(mean=.025, se=.0125), dict(mean=.1, se=0), frozen)
        self.assertTrue(at_floor["primary"]["passed"])
        self.assertFalse(at_error["primary"]["passed"])
        self.assertEqual(at_error["primary"]["failed_criteria"], ["evidence"])

    def test_rescue_requires_frozen_e128_gates_and_positive_grid_advantage(self):
        frozen = dict(local=dict(passed=False), grid=dict(passed=True))
        result = audit.intervention_gates(dict(mean=.1, se=0), dict(mean=.1, se=0), frozen)
        self.assertTrue(result["primary"]["passed"])
        self.assertFalse(result["local_rescue"]["passed"])
        frozen["local"]["passed"] = True
        result = audit.intervention_gates(dict(mean=.1, se=0), dict(mean=0, se=0), frozen)
        self.assertTrue(result["local_rescue"]["passed"])
        self.assertFalse(result["transfer_rescue"]["passed"])

    def test_effects_use_family_pairing_and_normal_summary_reuses_e128(self):
        rows = audit._fixture_rows()
        result = audit.summarize(rows)
        self.assertEqual(result["normal"], audit.retention.summarize(rows, delays=(0, 33)))
        effect = result["intervention"]["local_difference_33"]
        self.assertAlmostEqual(effect["mean"], .12)
        self.assertLess(effect["se"], 1e-15)
        self.assertEqual(effect["n"], 4)
        self.assertEqual(set(result["frozen"]["arms"]), set(audit.OFF_ARMS))
        self.assertEqual(result["retention_ticks"], [0, 33])

    def test_forged_immediate_treatment_difference_is_rejected(self):
        rows = audit._fixture_rows()
        rows[0]["plastic_hebbian_off_reversal_local_by_rule"][0, 0, 4] += .01
        with self.assertRaises(audit.base.AuditError):
            audit.summarize(rows)

    def test_all_three_same_lineages_must_pass_each_rescue_stage(self):
        result = audit.summarize(audit._fixture_rows())
        panels = [copy.deepcopy(result) for _ in range(3)]
        panels[2]["frozen"]["architectures"]["plastic"]["grid"]["passed"] = False
        effect = panels[2]["intervention"]
        effect.update(audit.intervention_gates(effect["local_difference_33"], effect["grid_difference_33"],
                                              panels[2]["frozen"]["architectures"]["plastic"]))
        replicated = audit.replicated_gates(panels)
        self.assertTrue(replicated["primary"]["passed"])
        self.assertTrue(replicated["local_rescue"]["passed"])
        self.assertFalse(replicated["transfer_rescue"]["passed"])
        self.assertEqual(replicated["transfer_rescue"]["seeds"], [0, 1])

    def test_old_study_completion_cannot_unlock_new_heldout_panels(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="retention-rescue-audit-", dir=scratch) as directory:
            checker = audit.Auditor.__new__(audit.Auditor)
            checker.root, checker.hashes = Path(directory), {}
            for index in range(3):
                folder = checker.root / f"runs/old_E128-A{index}"
                folder.mkdir(parents=True)
                (folder / "status.json").write_text(json.dumps(dict(status="finished", completed_families=128, expected_families=128)))
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle(False)
            self.assertEqual(checker.hashes, {})

    def test_all_new_sources_are_checked_before_the_first_outcome(self):
        events = []

        class WrongLastSource:
            def __init__(self, *_):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development=False):
                return ["development"] if development else ["A0", "A1", "A2"]

            def metadata(self, folder, *_):
                events.append(folder)
                if folder == "A2":
                    raise audit.base.AuditError("wrong intervention source")
                return ()

            def panel(self, *_):
                raise AssertionError("outcome opened before all source checks")

        with patch.object(audit, "Auditor", WrongLastSource), patch.object(audit.sys, "argv",
                ["audit", "--source-commit", "a" * 40]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main(), 1)
        self.assertEqual(events, ["development", "A0", "A1", "A2"])
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
