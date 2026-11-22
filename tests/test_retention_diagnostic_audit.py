"""Independent retention gates, assigned denominators and empty-world corruption fixtures."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_retention_diagnostic as audit


def gate_fixture():
    estimate = dict(mean=.2, se=0., n=4)
    local = dict(by_delay={str(delay): dict(signed_preference=estimate.copy(), reversal_minus_stationary=estimate.copy())
                          for delay in audit.DELAYS}, change_33_minus_0=dict(mean=0., se=0., n=4, lower_2se=0.))
    grid = dict(by_delay={str(delay): dict(first_food=estimate.copy(), food_minus_poison=estimate.copy(), reversal_minus_stationary=estimate.copy())
                         for delay in audit.DELAYS}, change_33_minus_0=dict(mean=0., se=0., n=4, lower_2se=0.))
    return local, grid


def rows_fixture():
    rows = []
    for value in (.2, .4, .6, .8):
        row = {}
        for arm in audit.ARMS:
            row[arm + "_pre_local_by_rule"] = np.broadcast_to([.2, .2, .4, .6, 0], (2, 5)).copy()
            local = np.zeros((2, 3, 5))
            if arm.endswith("reversal"):
                local[..., 4] = np.asarray([[value - .05], [value + .05]]) + np.asarray([0, -.1, .01])
            local[..., 0], local[..., 1], local[..., 2] = (1 + local[..., 4]) / 2, (1 - local[..., 4]) / 2, 1
            row[arm + "_local_by_rule"] = local
            row[arm + "_grid_by_rule"] = np.broadcast_to([1, 0, 0, 1, 1, 1, 1], (2, 3, 7)).copy()
        rows.append(row)
    return rows


def empty_interval_fixture():
    """An explicit neutral walker, not a simulated controller or held-out world."""
    x = np.linspace(-.1, .1, 157, dtype=np.float32)[None]
    acquired = dict(grid=np.ones((16, 16), np.int32), pos=np.asarray([[8, 8]], np.int32),
        energy=np.asarray([.8], np.float32), alive=np.ones(1, np.float32), sig=np.zeros(1, np.float32),
        denergy=np.asarray([.247], np.float32), key=np.asarray([4, 5], np.uint32), t=np.asarray(64, np.int32),
        counts=np.zeros((1, 2, 2), np.float32), x_sha256=audit.fingerprint(x))
    acquired.update({field + "_sha256": audit.fingerprint(np.ones(1, np.float32)) for field in ("W", "Wp", "mask")})
    key = np.asarray([7, 0], np.uint32)
    state = dict(acquired, grid=np.zeros((16, 16), np.int32), key=key)
    states, traces = [copy.deepcopy(state)], []
    for tick in range(33):
        observation = audit.encounter.observation(state["grid"], state["pos"][0], state["energy"][0], state["denergy"][0])[None]
        following = x.copy()
        following[:, :103] = observation
        efference = np.eye(5, dtype=np.float32)[[0]]
        following[:, -6:-1] = efference
        energy = state["energy"] - np.float32(.003) * state["alive"]
        delta, signal = energy - state["energy"], following[:, -1] * state["alive"]
        next_key = np.asarray([7, tick + 1], np.uint32)
        trace = dict(grid_before=state["grid"].copy(), position_before=state["pos"].copy(), energy_before=state["energy"].copy(),
            signal_before=state["sig"].copy(), incoming_denergy=state["denergy"].copy(), incoming_activity=x.copy(),
            observation=observation, decision_motor=np.zeros((1, 5), np.float32), logits=np.zeros((1, 5), np.float32),
            probabilities=np.full((1, 5), .2, np.float32), sampled_action=np.zeros(1, np.int32), forced_action=np.asarray(-1, np.int32),
            action=np.zeros(1, np.int32), action_key=np.asarray([8, tick], np.uint32), regrowth_key=np.asarray([9, tick], np.uint32),
            regrowth_cell=np.zeros((1, 2), np.int32), key_before=state["key"].copy(), key_after=next_key,
            grid=state["grid"].copy(), position=state["pos"].copy(), energy=energy, unclipped_energy=energy.copy(), signal=signal,
            alive_before=state["alive"].copy(), alive_after=state["alive"].copy(), denergy=delta, activity=following,
            efference=efference, food=np.zeros(1, bool), poison=np.zeros(1, bool), item=np.zeros(1, np.int32),
            surprise=np.asarray(0, np.float32), rule=np.asarray(1, np.int32), t=np.asarray(64 + tick, np.int32),
            hebbian_enabled=np.asarray(True), predictor_enabled=np.asarray(True))
        traces.append(trace)
        state = dict(state, energy=energy, denergy=delta, sig=signal, key=next_key, t=np.asarray(65 + tick, np.int32),
                     x_sha256=audit.fingerprint(following))
        x = following
        if tick in (0, 32):
            states.append(copy.deepcopy(state))
    return acquired, states, {field: np.stack([trace[field] for trace in traces]) for field in traces[0]}, key


class RetentionDiagnosticAuditTests(unittest.TestCase):
    def test_inclusive_effect_floor_and_strict_two_se_boundary(self):
        local, grid = gate_fixture()
        local["by_delay"]["0"]["signed_preference"] = dict(mean=.1, se=0., n=4)
        self.assertTrue(audit.retention_gate(local, grid))
        local["by_delay"]["0"]["signed_preference"]["se"] = .05
        self.assertFalse(audit.retention_gate(local, grid))
        self.assertEqual(local["failed_criteria"], ["S0"])

    def test_retention_margin_is_strict_and_requires_immediate_acquisition(self):
        for part in ("local", "grid"):
            local, grid = gate_fixture()
            chosen = local if part == "local" else grid
            chosen["change_33_minus_0"]["lower_2se"] = -.05
            self.assertFalse(audit.retention_gate(local, grid))
            self.assertEqual(chosen["failed_criteria"], ["retention"])
        local, grid = gate_fixture()
        local["by_delay"]["0"]["signed_preference"]["mean"] = 0.
        self.assertFalse(audit.retention_gate(local, grid))

    def test_one_turn_is_descriptive(self):
        local, grid = gate_fixture()
        for value in (*local["by_delay"]["1"].values(), *grid["by_delay"]["1"].values()):
            value["mean"] = -1.
        self.assertTrue(audit.retention_gate(local, grid))

    def test_paired_change_and_rule_average_use_family_se(self):
        result = audit.summarize(rows_fixture())["architectures"]["plastic"]["local"]
        self.assertAlmostEqual(result["change_33_minus_0"]["mean"], .01)
        self.assertLess(result["change_33_minus_0"]["se"], 1e-15)
        estimate = result["by_delay"]["0"]["signed_preference"]
        self.assertEqual(estimate["n"], 4)
        self.assertAlmostEqual(estimate["se"], np.std([.2, .4, .6, .8], ddof=1) / 2)

    def test_signed_first_contact_gets_its_own_family_uncertainty(self):
        rows = rows_fixture()
        for row, pair in zip(rows, ((1, -1), (1, 1), (-1, -1), (0, 0))):
            for arm in audit.ARMS:
                row[arm + "_grid_by_rule"] = row[arm + "_grid_by_rule"].astype(float)
                row[arm + "_grid_by_rule"][..., 3] = np.asarray(pair)[:, None]
        signed = audit.summarize(rows)["architectures"]["plastic"]["grid"]["by_delay"]["33"]["food_minus_poison"]
        self.assertEqual(signed["mean"], 0.)
        self.assertEqual(signed["n"], 4)
        self.assertAlmostEqual(signed["se"], np.sqrt(1 / 6))

    def test_replication_requires_all_three_and_same_lineages(self):
        panels = []
        for _ in range(3):
            local, grid = gate_fixture()
            audit.retention_gate(local, grid)
            panels.append(dict(architectures={name: dict(local=copy.deepcopy(local), grid=copy.deepcopy(grid)) for name in ("plastic", "direct")}))
        self.assertTrue(audit.replicated_gates(panels)["plastic"]["combined"]["passed"])
        panels[2]["architectures"]["plastic"]["grid"]["passed"] = False
        self.assertTrue(audit.replicated_gates(panels)["plastic"]["local"]["passed"])
        self.assertFalse(audit.replicated_gates(panels)["plastic"]["combined"]["passed"])
        panels[1]["architectures"]["plastic"]["local"]["passed"] = False
        self.assertFalse(audit.replicated_gates(panels)["plastic"]["local"]["passed"])

    def test_all_33_actual_empty_turns_with_pending_feedback_are_checked(self):
        acquired, states, trace, key = empty_interval_fixture()
        audit.active_interval(acquired, states, trace, key, True, 1)
        self.assertEqual(int(states[-1]["t"]), 97)
        self.assertLess(float(states[-1]["energy"][0]), float(acquired["energy"][0]))

    def test_empty_interval_resets_missing_turns_fake_resources_and_frozen_updates_fail(self):
        corruptions = (
            lambda states, trace: states[0]["denergy"].fill(0),
            lambda states, trace: trace["key_before"].__setitem__(1, trace["key_before"][0]),
            lambda states, trace: trace["incoming_activity"][1].fill(0),
            lambda states, trace: trace["grid"][2].__setitem__((0, 0), 1),
            lambda states, trace: trace["t"].__setitem__(1, 0),
            lambda states, trace: trace["predictor_enabled"].fill(False),
            lambda states, trace: states[1]["pos"].fill(0),
        )
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                acquired, states, trace, key = empty_interval_fixture()
                corrupt(states, trace)
                with self.assertRaises(audit.base.AuditError):
                    audit.active_interval(acquired, states, trace, key, True, 1)
        acquired, states, trace, key = empty_interval_fixture()
        with self.assertRaises(audit.base.AuditError):
            audit.active_interval(acquired, states, {field: value[:32] for field, value in trace.items()}, key, True, 1)

    def test_all_panels_must_finish_before_any_summary_or_raw_read(self):
        scratch = audit.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="retention-audit-", dir=scratch) as directory:
            checker = audit.Auditor.__new__(audit.Auditor)
            checker.root, checker.hashes = Path(directory), {}
            folder = checker.root / "runs/fixture_E128-A0"
            (folder / "families").mkdir(parents=True)
            (folder / "status.json").write_text(json.dumps(dict(status="finished", completed_families=128, expected_families=128)))
            (folder / "summary.json").write_bytes(b"do not parse outcomes")
            for index in range(128):
                (folder / "families" / f"family_{index:04d}.npz").touch()
            with self.assertRaises(audit.base.Incomplete):
                checker.completed_bundle(False)
            self.assertTrue(all(path.endswith("status.json") for path in checker.hashes))

    def test_every_final_source_is_preflighted_before_any_outcome(self):
        events = []

        class WrongLastSource:
            def __init__(self, *_):
                self.hashes, self.family_seconds = {}, []

            def completed_bundle(self, development=False):
                return ["development"] if development else ["A0", "A1", "A2"]

            def metadata(self, folder, *_):
                events.append(folder)
                if folder == "A2":
                    raise audit.base.AuditError("wrong final source")
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
