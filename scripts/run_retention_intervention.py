"""E129 thin intervention layer over the immutable retention-study runner."""
import hashlib
import sys

import numpy as np

from seed import retention_rescue as engine
from scripts import run_retention_diagnostic as base


PROTOCOL = "retention-intervention-v1"
STUDY_ID = "E129"
DEVELOPMENT_SEED = 2026120800
ASSIGNMENT_TAG = 0xE129
MAX_SECONDS = {"development": 120, "final": 300}
RETENTION_TICKS = (0, 33)
ARMS = ("plastic_stationary", "plastic_reversal", "plastic_hebbian_off_stationary",
        "plastic_hebbian_off_reversal", "direct_stationary", "direct_reversal")
KEY_ROLES, LOCAL_FIELDS, GRID_FIELDS = base.KEY_ROLES, base.LOCAL_FIELDS, base.GRID_FIELDS
AUDIT_PATH = "results/diagnostics/E129-independent-audit.json"
CONFIG_EXTRA = dict(intervention="hebbian-off-during-retention-only", continuation_tick=1,
                    plastic_predictor_active=True, plastic_grid_hebbian_active=True)


def _source_code(root):
    result = base._source_code(root)
    for name in ("seed/retention_rescue.py", "scripts/run_retention_intervention.py"):
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def _gates(local_difference, grid_difference, frozen):
    def decision(criteria):
        return dict(criteria=criteria, passed=all(criteria.values()),
                    failed_criteria=[name for name, value in criteria.items() if not value])

    primary = decision(dict(minimum=local_difference["mean"] >= .025,
                            evidence=local_difference["mean"] > 2*local_difference["se"]))
    local = decision(dict(primary=primary["passed"], frozen_local=frozen["local"]["passed"]))
    transfer = decision(dict(local_rescue=local["passed"], frozen_grid=frozen["grid"]["passed"],
                             grid_advantage=grid_difference["mean"] > 2*grid_difference["se"]))
    return dict(primary=primary, local_rescue=local, transfer_rescue=transfer)


def summarize(rows):
    normal = base.summarize(rows, retention_ticks=RETENTION_TICKS)
    frozen_rows = []
    suffixes = ("pre_old_by_rule", "pre_local_by_rule", "local_by_rule", "grid_by_rule")
    for row in rows:
        mapped = dict(row)
        for history in ("stationary", "reversal"):
            for suffix in suffixes:
                normal_value = np.asarray(row[f"plastic_{history}_{suffix}"], np.float64)
                frozen_value = np.asarray(row[f"plastic_hebbian_off_{history}_{suffix}"], np.float64)
                if frozen_value.shape != normal_value.shape:
                    raise ValueError("Hebbian-off evidence must retain every paired rule and endpoint")
                before, after = ((normal_value[:, 0], frozen_value[:, 0]) if suffix in ("local_by_rule", "grid_by_rule")
                                 else (normal_value, frozen_value))
                if before.shape != after.shape or before.tobytes() != after.tobytes():
                    raise ValueError("normal and Hebbian-off histories must share the exact immediate snapshot")
                mapped[f"plastic_{history}_{suffix}"] = frozen_value
        frozen_rows.append(mapped)
    frozen_full = base.summarize(frozen_rows, retention_ticks=RETENTION_TICKS)
    frozen = {name: frozen_full[name] for name in ("families", "initial_rule_replays", "retention_ticks")}
    frozen.update(arms={f"plastic_hebbian_off_{history}": frozen_full["arms"][f"plastic_{history}"]
                        for history in ("stationary", "reversal")},
                  architectures=dict(plastic=frozen_full["architectures"]["plastic"]),
                  by_initial_rule={rule: dict(plastic=values["plastic"])
                                   for rule, values in frozen_full["by_initial_rule"].items()})
    effects, rule_effects = {}, {"0": {}, "1": {}}
    for field, column, name in (("local_by_rule", 4, "local_difference_33"), ("grid_by_rule", 0, "grid_difference_33")):
        intact = np.asarray([row[f"plastic_reversal_{field}"] for row in rows], np.float64)[:, :, -1, column]
        off = np.asarray([row[f"plastic_hebbian_off_reversal_{field}"] for row in rows], np.float64)[:, :, -1, column]
        effects[name] = base._estimate(off.mean(axis=1) - intact.mean(axis=1))
        for rule in (0, 1):
            rule_effects[str(rule)][name] = base._estimate(off[:, rule] - intact[:, rule])
    intervention = dict(**effects, **_gates(effects["local_difference_33"], effects["grid_difference_33"],
                                           frozen["architectures"]["plastic"]), by_initial_rule=rule_effects)
    return dict(families=len(rows), initial_rule_replays=2*len(rows), retention_ticks=list(RETENTION_TICKS),
                normal=normal, frozen=frozen, intervention=intervention)


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("retention intervention requires all three registered lineages")
    qualified = {name: [] for name in ("primary", "local_rescue", "transfer_rescue")}
    for panel in panels:
        result = panel["measurements"]
        intervention = result["intervention"]
        expected = _gates(intervention["local_difference_33"], intervention["grid_difference_33"],
                          result["frozen"]["architectures"]["plastic"])
        for name, decision in expected.items():
            if intervention[name] != decision or type(intervention[name]["passed"]) is not bool:
                raise ValueError("intervention rescue gates must use the same lineage and its frozen criteria")
            if decision["passed"]:
                qualified[name].append(panel["index"])
    return {name: dict(count=len(indices), seeds=indices, passed=len(indices) == 3)
            for name, indices in qualified.items()}


def run_tracked(args, git, root, *, explicit=None):
    return base.run_tracked(args, git, root, explicit=explicit, study=sys.modules[__name__])


def build_comparison(root):
    return base.build_comparison(root, study=sys.modules[__name__])
