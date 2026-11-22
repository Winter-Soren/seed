"""E130 independent first-item accounting, family uncertainty and strict gates."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_founder_reversal as audit


def fixture():
    raw, genomes, keys, births = audit.synthetic_family()
    return raw, genomes, keys, births, audit.audit_family(raw, 0, genomes, keys, births)


class FounderReversalAuditTests(unittest.TestCase):
    def test_complete_native_events_reconstruct_all_endpoints(self):
        raw, genomes, keys, births, row = fixture()
        result = audit.summarize([row] * 4)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["components"]["descendant_recovery"], dict(mean=2., se=0., n=4))
        self.assertEqual(result["components"]["recovery_interaction"]["mean"], 2.)
        self.assertEqual((result["initial_rule_replays"], result["genotype_history_rollouts"]), (8, 32))
        self.assertEqual(result["by_initial_rule"]["0"]["components"], result["by_initial_rule"]["1"]["components"])
        self.assertEqual(audit.audit_family(raw, 0, genomes, keys, births)["gen200_reversal"]["Q"][0, 5].tolist(), [1] * 4)

    def test_first_item_is_not_last_meal_or_total_food(self):
        food, poison = np.zeros((2, 300, 4), bool), np.zeros((2, 300, 4), bool)
        alive = np.ones_like(food)
        poison[:, 0, 0], food[:, 1:50, 0] = True, True
        food[:, 49, 1] = food[:, 50, 1] = food[:, 299, 1] = True
        value = audit.event_outcomes(food, poison, alive, alive)
        self.assertEqual(value["Q"][0, 0].tolist(), [-1, 1, 0, 0])
        self.assertEqual(value["Q"][0, 1].tolist(), [0, 1, 0, 0])
        self.assertEqual(value["Q"][0, 5].tolist(), [0, 1, 0, 0])
        np.testing.assert_array_equal(value["first_food"] + value["first_poison"] + value["no_contact"], True)

    def test_death_without_contact_is_zero_not_removed_or_replaced(self):
        food, poison = np.zeros((2, 300, 4), bool), np.zeros((2, 300, 4), bool)
        before, after = np.ones_like(food), np.ones_like(food)
        after[:, 20:, 0], before[:, 21:, 0] = False, False
        value = audit.event_outcomes(food, poison, before, after)
        self.assertEqual(value["Q"].shape, (2, 6, 4))
        self.assertTrue(value["no_contact"].all())
        self.assertFalse(value["end_survival"][:, 0].any())
        food[:, 21, 0] = True
        with self.assertRaisesRegex(audit.base.AuditError, "dead"):
            audit.event_outcomes(food, poison, before, after)
        food[:] = False
        after[:, 22, 0] = True
        with self.assertRaises(audit.base.AuditError):
            audit.event_outcomes(food, poison, before, after)

    def test_switch_at_151_is_rejected_even_with_identical_counts_and_window_scores(self):
        raw, genomes, keys, births, _ = fixture()
        raw["gen200_reversal_poison"][:, 150] = False
        raw["gen200_reversal_poison"][:, 151] = True
        with self.assertRaisesRegex(audit.base.AuditError, "turn150"):
            audit.audit_family(raw, 0, genomes, keys, births)

    def test_changed_pre_switch_history_is_rejected_even_with_identical_half_counts(self):
        raw, genomes, keys, births, _ = fixture()
        raw["gen200_reversal_food"][:, 100] = False
        raw["gen200_reversal_food"][:, 149] = True
        with self.assertRaisesRegex(audit.base.AuditError, "before the hidden switch"):
            audit.audit_family(raw, 0, genomes, keys, births)

    def test_native_counts_fitness_survival_and_genome_corruption_are_rejected(self):
        for field in ("gen0_reversal_native_counts", "gen200_stationary_native_fitness",
                      "gen0_stationary_native_survived", "gen0_genome_sha256", "birth_keys"):
            with self.subTest(field=field):
                raw, genomes, keys, births, _ = fixture()
                altered = copy.deepcopy(raw)
                altered[field].flat[0] += 1
                with self.assertRaises(audit.base.AuditError):
                    audit.audit_family(altered, 0, genomes, keys, births)

    def test_original_animals_rules_indices_and_archive_fields_are_required(self):
        raw, genomes, keys, births, _ = fixture()
        for field, bad in (("gen0_stationary_food", raw["gen0_stationary_food"][:, :, :3]),
                           ("initial_rules", np.asarray([1, 0], np.int32)),
                           ("base_index", np.asarray(1, np.int32)),
                           ("flip_step", np.asarray(150, np.int64))):
            with self.subTest(field=field):
                altered = dict(raw, **{field: bad})
                with self.assertRaises(audit.base.AuditError):
                    audit.audit_family(altered, 0, genomes, keys, births)
        with self.assertRaises(audit.base.AuditError):
            audit.audit_family(dict(raw, substitute=np.asarray(0)), 0, genomes, keys, births)

    def test_standard_error_uses_family_after_four_animals_and_both_rules(self):
        _, _, _, _, original = fixture()
        rows = []
        for positives in (0, 2, 6, 8):
            row = copy.deepcopy(original)
            q = np.full(8, -1, np.int8)
            q[:positives] = 1
            for arm in audit.ARMS:
                row[arm]["Q"][:, 2] = q.reshape(2, 4)
                row[arm]["first_food"][:, 2] = q.reshape(2, 4) == 1
                row[arm]["first_poison"][:, 2] = q.reshape(2, 4) == -1
                row[arm]["no_contact"][:, 2] = False
            rows.append(row)
        result = audit.summarize(rows)
        expected = np.std([-1., -.5, .5, 1.], ddof=1) / 2
        value = result["components"]["descendant_pre"]
        self.assertEqual(value["n"], 4)
        self.assertAlmostEqual(value["se"], expected)
        self.assertEqual(result["safeguards"]["pre_Q"]["se"], 0.)
        self.assertEqual(result["by_initial_rule"]["0"]["components"]["descendant_pre"]["n"], 4)

    def test_inclusive_effect_floor_strict_two_se_and_strict_lower_bound(self):
        *_, row = fixture()
        result = audit.summarize([row] * 4)
        c, s = copy.deepcopy(result["components"]), copy.deepcopy(result["safeguards"])
        c["descendant_recovery"] = dict(mean=.05, se=0., n=4)
        self.assertTrue(audit.decisions(c, s)["passed"])
        c["descendant_recovery"]["se"] = .025
        self.assertFalse(audit.decisions(c, s)["criteria"]["descendant_recovers"])
        c["descendant_recovery"]["se"] = 0.
        s["stationary_end_survival"]["lower_2se"] = -.05
        self.assertFalse(audit.decisions(c, s)["criteria"]["stationary_survival_noninferiority"])

    def test_founder_deterioration_cannot_substitute_for_descendant_recovery(self):
        *_, row = fixture()
        result = audit.summarize([row] * 4)
        result["components"]["descendant_recovery"] = dict(mean=0., se=0., n=4)
        result["gate"] = audit.decisions(result["components"], result["safeguards"])
        self.assertTrue(result["gate"]["criteria"]["recovery_interaction"])
        self.assertFalse(result["gate"]["passed"])
        self.assertEqual(result["gate"]["failed_criteria"], ["descendant_recovers"])

    def test_replication_requires_every_criterion_in_each_of_three_lineages(self):
        *_, row = fixture()
        panels = [audit.summarize([row] * 4) for _ in range(3)]
        self.assertTrue(audit.replicated_gates(panels)["evolutionary_recovery"]["passed"])
        panels[1]["safeguards"]["pre_Q"]["lower_2se"] = -.06
        panels[1]["gate"] = audit.decisions(panels[1]["components"], panels[1]["safeguards"])
        result = audit.replicated_gates(panels)["evolutionary_recovery"]
        self.assertEqual(result, dict(count=2, seeds=[0, 2], passed=False))
        panels[1]["gate"]["passed"] = True
        with self.assertRaises(audit.base.AuditError):
            audit.replicated_gates(panels)

    def test_incomplete_triplet_opens_only_status_and_no_outcome_files(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="founder-audit-", dir=scratch) as directory:
            checker = audit.Auditor.__new__(audit.Auditor)
            checker.root, checker.hashes = Path(directory), {}
            for index in range(3):
                folder = checker.root / f"runs/synthetic_E130-A{index}"
                (folder / "families").mkdir(parents=True)
                (folder / "summary.json").write_text("OUTCOMES MUST STAY SEALED")
                (folder / "status.json").write_text(json.dumps(dict(status="finished", completed_families=64, expected_families=64)))
                for family in range(63 if index == 2 else 64):
                    (folder / "families" / f"family_{family:04d}.npz").write_bytes(b"SEALED")
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle(False)
            self.assertEqual(len(checker.hashes), 3)
            self.assertTrue(all(path.endswith("/status.json") for path in checker.hashes))

    def test_all_sources_are_checked_before_first_final_outcome(self):
        seen = []

        class WrongLastSource:
            def __init__(self, *_):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development=False):
                return ["development"] if development else ["A0", "A1", "A2"]

            def metadata(self, folder, *_):
                seen.append(folder)
                if folder == "A2":
                    raise audit.base.AuditError("wrong source")
                return ()

            def panel(self, *_):
                raise AssertionError("outcomes opened too soon")

        with patch.object(audit, "Auditor", WrongLastSource), patch.object(audit.sys, "argv",
                ["audit", "--source-commit", "a" * 40]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main(), 1)
        self.assertEqual(seen, ["development", "A0", "A1", "A2"])
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
