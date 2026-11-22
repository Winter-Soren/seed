"""Independent E129 audit: interval-only Hebbian intervention and paired rescue gates."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_retention_diagnostic as retention

base = retention.base
require, same, subset, typed = retention.require, retention.same, retention.subset, retention.typed
exact, prefixed, stat = retention.exact, retention.prefixed, retention.stat
PROTOCOL = "retention-intervention-v1"
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_hebbian_off_stationary",
        "plastic_hebbian_off_reversal", "direct_stationary", "direct_reversal")
OFF_ARMS = ARMS[2:4]
DELAYS = (0, 33)
CONFIG_EXTRA = dict(intervention="hebbian-off-during-retention-only", continuation_tick=1,
                    plastic_predictor_active=True, plastic_grid_hebbian_active=True)


def frozen_weight_evidence(acquired, states, trace):
    require(not typed(trace, "hebbian_enabled", (33,), "biu", binary=True).any(), "Hebbian updates are disabled from the first empty turn")
    require(typed(trace, "predictor_enabled", (33,), "biu", binary=True).all(), "the predictor remains active for all 33 turns")
    for field in ("W_input_matches_acquired", "W_output_matches_acquired"):
        require(typed(trace, field, (33,), "b", binary=True).all(), "every actual frozen input/output weight representation is unchanged")
    for state in states:
        exact(state["W_sha256"], acquired["W_sha256"], "frozen checkpoints retain the acquired weight bytes")


def interval(acquired, states, continuation, trace, key, plastic, rule, *, frozen):
    require(len(states) == 2, "only the registered zero/33 behavioral endpoints")
    full_states = [states[0], continuation, states[1]]
    if frozen:
        require(plastic, "the intervention applies only to the plastic controller")
        frozen_weight_evidence(acquired, full_states, trace)
    retention.active_interval(acquired, full_states, trace, key, plastic, rule, hebbian=False if frozen else None)
    exact(continuation["mask_sha256"], acquired["mask_sha256"], "continuation retains its inherited wiring")
    if not plastic:
        for field in ("W", "Wp"):
            exact(continuation[field + "_sha256"], acquired[field + "_sha256"], "direct weights stay fixed at the continuation boundary")


def audit_family(raw, index, pre, plastic, direct, probe_keys):
    exact(raw["continuation_tick"], np.asarray(1, np.int32), "actual one-turn continuation without a behavioral readout")
    checks = {}
    for arm in ARMS:
        def check(acquired, states, trace, key, is_plastic, rule, *, arm=arm):
            initial_rule = rule if arm.endswith("stationary") else 1 - rule
            continuation = {field: value[initial_rule] for field, value in prefixed(raw, arm + "_continuation_").items()}
            interval(acquired, states, continuation, trace, key, is_plastic, rule, frozen=arm in OFF_ARMS)
        checks[arm] = check
    rows = retention.audit_family(raw, index, pre, plastic, direct, probe_keys,
                                   arms=ARMS, delays=DELAYS, interval_checks=checks)
    for arm in OFF_ARMS:
        changed = prefixed(raw, arm + "_")
        normal = prefixed(raw, "plastic_" + arm.rsplit("_", 1)[-1] + "_")
        for field, value in changed.items():
            if field.startswith(("birth_", "pre_", "baseline_probe_", "guided_", "acquired_")):
                exact(value, normal[field], "one intervention starts only after identical acquisition")
            elif field.startswith(("delay_", "probe_", "grid_")):
                exact(value[:, 0], normal[field][:, 0], "complete immediate treatment copies and readouts agree")
        for field in ("energy", "denergy", "alive_before", "alive_after"):
            exact(changed["retention_" + field], normal["retention_" + field], "same ordinary empty-world physical cost")
    return rows


def intervention_gates(local_effect, grid_effect, frozen):
    def decision(criteria):
        return dict(criteria=criteria, passed=all(criteria.values()), failed_criteria=[name for name, passed in criteria.items() if not passed])
    primary = decision(dict(minimum=local_effect["mean"] >= .025,
                            evidence=local_effect["se"] is not None and local_effect["mean"] > 2 * local_effect["se"]))
    local = decision(dict(primary=primary["passed"], frozen_local=frozen["local"]["passed"]))
    transfer = decision(dict(local_rescue=local["passed"], frozen_grid=frozen["grid"]["passed"],
        grid_advantage=grid_effect["se"] is not None and grid_effect["mean"] > 2 * grid_effect["se"]))
    return dict(primary=primary, local_rescue=local, transfer_rescue=transfer)


def summarize(rows):
    normal = retention.summarize(rows, delays=DELAYS)
    frozen_rows = []
    for row in rows:
        projected = dict(row)
        for history in ("stationary", "reversal"):
            for field in ("pre_local_by_rule", "local_by_rule", "grid_by_rule"):
                before, after = (np.asarray(row[prefix + history + "_" + field], np.float64)
                                 for prefix in ("plastic_", "plastic_hebbian_off_"))
                exact(before if field == "pre_local_by_rule" else before[:, 0],
                      after if field == "pre_local_by_rule" else after[:, 0], "immediate outcome evidence is identical before intervention")
                projected["plastic_" + history + "_" + field] = after
        frozen_rows.append(projected)
    complete = retention.summarize(frozen_rows, delays=DELAYS)
    frozen = {field: complete[field] for field in ("families", "initial_rule_replays", "retention_ticks")}
    frozen.update(arms={"plastic_hebbian_off_" + history: complete["arms"]["plastic_" + history] for history in ("stationary", "reversal")},
        architectures=dict(plastic=complete["architectures"]["plastic"]),
        by_initial_rule={rule: dict(plastic=value["plastic"]) for rule, value in complete["by_initial_rule"].items()})
    effects, by_rule = {}, {"0": {}, "1": {}}
    for field, column, name in (("local_by_rule", 4, "local_difference_33"), ("grid_by_rule", 0, "grid_difference_33")):
        normal_values, frozen_values = (np.stack([row[prefix + field] for row in rows]).astype(np.float64)[:, :, -1, column]
                                       for prefix in ("plastic_reversal_", "plastic_hebbian_off_reversal_"))
        effects[name] = stat(frozen_values.mean(1) - normal_values.mean(1))
        for rule in (0, 1):
            by_rule[str(rule)][name] = stat(frozen_values[:, rule] - normal_values[:, rule])
    intervention = dict(**effects, **intervention_gates(effects["local_difference_33"], effects["grid_difference_33"],
                                                       frozen["architectures"]["plastic"]), by_initial_rule=by_rule)
    return dict(families=len(rows), initial_rule_replays=2 * len(rows), retention_ticks=list(DELAYS),
                normal=normal, frozen=frozen, intervention=intervention)


def replicated_gates(panels):
    require(len(panels) == 3, "all three original lineages required")
    qualified = {name: [] for name in ("primary", "local_rescue", "transfer_rescue")}
    for index, panel in enumerate(panels):
        effect = panel["intervention"]
        decisions = intervention_gates(effect["local_difference_33"], effect["grid_difference_33"], panel["frozen"]["architectures"]["plastic"])
        for name, decision in decisions.items():
            same(effect[name], decision, "rescue requires every component in the same lineage")
            if decision["passed"]:
                qualified[name].append(index)
    return {name: dict(count=len(indices), seeds=indices, passed=len(indices) == 3) for name, indices in qualified.items()}


class Auditor(retention.Auditor):
    RUN_PREFIX, MASTER, PROTOCOL = "E129", 2026120800, PROTOCOL
    ARMS, DELAYS, ASSIGNMENT_TAG = ARMS, DELAYS, 0xE129
    SOURCE_PATHS = retention.SOURCE_PATHS | {"seed/retention_rescue.py", "scripts/run_retention_intervention.py"}
    reconstruct_family, summarize_rows = staticmethod(audit_family), staticmethod(summarize)

    def metadata(self, folder, index, development):
        prepared = super().metadata(folder, index, development)
        subset(prepared[0], CONFIG_EXTRA, "one interval-only intervention with ordinary predictor and transfer learning")
        return prepared


def _fixture_rows():
    rows = []
    for value in (.35, .45, .55, .65):
        row = {}
        for arm in ARMS:
            reversed_history, off = arm.endswith("reversal"), arm in OFF_ARMS
            row[arm + "_pre_local_by_rule"] = np.broadcast_to([.2, .2, .4, .6, 0], (2, 5)).copy()
            signed = np.asarray([value, value if off else value - .12]) if reversed_history else np.full(2, -.1)
            local = np.zeros((2, 2, 5), np.float64)
            local[..., 0], local[..., 1], local[..., 2], local[..., 4] = (1 + signed) / 2, (1 - signed) / 2, 1, signed
            row[arm + "_local_by_rule"] = local
            grid = np.zeros((2, 2, 7), np.float64)
            good = np.asarray([1, int(off)]) if reversed_history else np.zeros(2)
            grid[..., 0], grid[..., 1], grid[..., 3] = good, 1 - good, 2 * good - 1
            grid[..., 4], grid[..., 5], grid[..., 6] = 1, 2 * good - 1, 1
            row[arm + "_grid_by_rule"] = grid
        rows.append(row)
    return rows


def self_check():
    previous = retention.self_check()
    measured = summarize(_fixture_rows())
    effect = measured["intervention"]
    require(effect["local_difference_33"]["se"] < 1e-15 and effect["transfer_rescue"]["passed"], "paired causal effect and successful synthetic rescue")
    frozen = measured["frozen"]["architectures"]["plastic"]
    floor = intervention_gates(dict(mean=.025, se=0), dict(mean=.1, se=0), frozen)
    boundary = intervention_gates(dict(mean=.025, se=.0125), dict(mean=.1, se=0), frozen)
    require(floor["primary"]["passed"] and not boundary["primary"]["passed"], "inclusive causal floor and strict paired evidence")
    panels = [copy.deepcopy(measured) for _ in range(3)]
    require(replicated_gates(panels)["transfer_rescue"]["passed"], "same three lineages complete every stage")
    panels[2]["frozen"]["architectures"]["plastic"]["local"]["passed"] = False
    panels[2]["intervention"].update(intervention_gates(effect["local_difference_33"], effect["grid_difference_33"],
        panels[2]["frozen"]["architectures"]["plastic"]))
    require(replicated_gates(panels)["primary"]["passed"] and not replicated_gates(panels)["local_rescue"]["passed"], "primary alone is not rescue")
    acquired = dict(W_sha256=np.zeros(32, np.uint8))
    states = [copy.deepcopy(acquired) for _ in range(3)]
    trace = dict(hebbian_enabled=np.zeros(33, bool), predictor_enabled=np.ones(33, bool),
                 W_input_matches_acquired=np.ones(33, bool), W_output_matches_acquired=np.ones(33, bool))
    frozen_weight_evidence(acquired, states, trace)
    for field in ("W_input_matches_acquired", "W_output_matches_acquired", "predictor_enabled", "hebbian_enabled"):
        bad = copy.deepcopy(trace)
        bad[field][0] = not bool(bad[field][0])
        try:
            frozen_weight_evidence(acquired, states, bad)
        except base.AuditError:
            continue
        raise base.AuditError("forged interval evidence accepted: " + field)
    states[-1]["W_sha256"][0] = 1
    try:
        frozen_weight_evidence(acquired, states, trace)
    except base.AuditError:
        pass
    else:
        raise base.AuditError("true predicates with a corrupt endpoint accepted")
    require("seed.seed" not in sys.modules and "jax" not in sys.modules, "self-check imports no simulator or random runtime")
    return dict(protocol=PROTOCOL, mode="self-check", status="passed", checks=previous["checks"] + 4, simulator_imported=False,
        coverage=previous["coverage"] + ["unchanged E128 defaults and paired E129 effects", "causal floor and strict evidence",
            "same-lineage primary/local/transfer hierarchy", "actual masks, all-turn weight predicates and endpoint corruption"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2, allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required")
    started, auditor = time.monotonic(), None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            auditor.metadata(auditor.completed_bundle(True)[0], 0, True)
        prepared = [auditor.metadata(folder, index, args.development) for index, folder in enumerate(folders)]
        panels = [auditor.panel(folder, index, args.development, prepared[index]) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit, registered_code_trees=auditor.expected_trees,
            panels=panels, replicated_gates=None if args.development else replicated_gates([panel["measurements"] for panel in panels]),
            limitations=["Per-turn weight-identity predicates come from source-verified bitwise comparisons of actual carried matrices; dense weight histories are not independently reconstructed.",
                "Endpoint hashes, actual update masks, empty-world physics, cloned readouts and family-level gates are checked without replaying neural dynamics.",
                "The intervention leaves the predictor and recurrence active. Their downstream consequences are part of its total effect.",
                "This diagnostic tests an imposed update schedule, not an evolved ability to choose that schedule."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), replicated_gates=None)
        code = 2
    except (base.AuditError, OSError, KeyError, ValueError, TypeError, IndexError, zipfile.BadZipFile, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", replicated_gates=None)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        elapsed, reconstruction = time.monotonic() - started, sum(auditor.family_seconds)
        report["timing"] = dict(elapsed_seconds=elapsed, setup_and_summary_seconds=elapsed - reconstruction,
            reconstruction_seconds=reconstruction, family_seconds=auditor.family_seconds, completed_families=len(auditor.family_seconds))
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
