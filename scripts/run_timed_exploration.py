"""E131 timed exploration: paired evidence using E130's immutable panel writer."""
import hashlib
import json
import sys

import numpy as np

from seed import timed_exploration as engine
from scripts import run_founder_reversal as base


PROTOCOL = "timed-exploration-v1"
STUDY_ID = "E131"
DEVELOPMENT_SEED = 2026121000
MAX_SECONDS = {"development": 200, "final": 600}
POLICIES = ("unchanged", "adaptive", "random_quota")
ARMS = tuple(policy + "_" + history for policy in POLICIES for history in ("stationary", "reversal"))
KEY_ROLES = ("world_action", "development", "exploration")
BIN_EDGES = base.BIN_EDGES
ROLLOUTS_FIELD = "policy_history_rollouts"
SOURCES = base.FINAL_SOURCES
SOURCE_SLOTS = (("genome_final.npy", "source_genome", "genome_sha256", SOURCES),)
CONFIG_EXTRA = dict(source_generations=[200], policy_order=list(POLICIES), assignment_tag=0xE131,
    policy_parameters=dict(block_length=25, quota_per_block=1, counter_cap=12, trigger=12,
        phase_delay_min=0, phase_delay_max=24, quota_offset_min=0, quota_offset_max=24,
        cardinal_min=1, cardinal_max=4, feedback="positive_observed_realized_denergy"))
AUDIT_PATH = "results/diagnostics/E131-independent-audit.json"
CONTROLS = {"ordinary": "unchanged", "quota": "random_quota"}


def _gate(components, comparisons):
    checks = dict(adaptive_late_positive=base._advantage(components["adaptive_reversal_late"]))
    for control in CONTROLS:
        values = comparisons[control]
        checks.update({control + "_latency_improves": base._advantage(values["latency_saved"]),
                       control + "_late_Q_improves": base._advantage(values["late_Q_difference"]),
                       control + "_adjusted_Q_improves": base._advantage(values["adjusted_late_Q"], .05)})
        for criterion, field in (("stationary_late_noninferiority", "stationary_late_Q"),
                                 ("reversal_survival_noninferiority", "reversal_end_survival"),
                                 ("stationary_survival_noninferiority", "stationary_end_survival")):
            checks[control + "_" + criterion] = values["safeguards"][field]["lower_2se"] > -.05
    return dict(criteria=checks, passed=all(checks.values()), failed_criteria=[key for key, passed in checks.items() if not passed])


def summarize(rows):
    n, arrays, extra = len(rows), base._family_arrays(rows, ARMS), {}
    for arm in ARMS:
        fields = {name: np.asarray([row[arm + "_" + name + "_by_rule"] for row in rows], np.float64)
                  for name in ("latency_first_food", "latency_first_any", "override_count", "alive_turns", "changed_action_count")}
        if any(value.shape != (n, 2, 4) or not np.isfinite(value).all() or np.any(value != np.floor(value)) for value in fields.values()):
            raise ValueError("exploration endpoints must retain every assigned animal and both rules")
        food, any_item, count, live, changed = (fields[name] for name in fields)
        if (np.any(any_item < 0) or np.any(food > 50) or np.any(any_item > food)
                or np.any(live < 1) or np.any(live > 300) or np.any(count < 0) or np.any(count > live)
                or np.any(changed < 0) or np.any(changed > count)
                or arm.startswith("unchanged_") and np.any(count)):
            raise ValueError("invalid latency censoring, live-turn denominator or realized override count")
        extra[arm] = fields

    def measure(rule=None):
        family, arms = base._arm_summaries(arrays, rule)
        for arm, fields in extra.items():
            selected = {name: value if rule is None else value[:, rule:rule+1] for name, value in fields.items()}
            for field, output in (("latency_first_food", "first_food_latency"), ("latency_first_any", "first_any_latency"),
                                  ("override_count", "override_count"), ("changed_action_count", "changed_action_count")):
                family[arm][output] = selected[field].mean(axis=(1, 2))
                arms[arm][output] = base._estimate(family[arm][output])
            fraction = selected["override_count"].sum(axis=(1, 2)) / selected["alive_turns"].sum(axis=(1, 2))
            arms[arm]["override_live_fraction"] = base._estimate(fraction)
            changed_fraction = selected["changed_action_count"].sum(axis=(1, 2)) / selected["alive_turns"].sum(axis=(1, 2))
            arms[arm]["changed_action_live_fraction"] = base._estimate(changed_fraction)
        adaptive_rev, adaptive_stat = family["adaptive_reversal"], family["adaptive_stationary"]
        components = dict(adaptive_reversal_late=base._estimate(adaptive_rev["Q"][:, 5]))
        comparisons = {}
        for control, policy in CONTROLS.items():
            rev, stat = family[policy + "_reversal"], family[policy + "_stationary"]
            difference = adaptive_rev["Q"][:, 5] - rev["Q"][:, 5]
            comparisons[control] = dict(latency_saved=base._estimate(rev["first_food_latency"]-adaptive_rev["first_food_latency"]),
                late_Q_difference=base._estimate(difference),
                adjusted_late_Q=base._estimate(difference-(adaptive_stat["Q"][:, 5]-stat["Q"][:, 5])),
                safeguards=dict(stationary_late_Q=base._estimate(adaptive_stat["Q"][:, 5]-stat["Q"][:, 5], bound=True),
                    reversal_end_survival=base._estimate(adaptive_rev["end_survival"]-rev["end_survival"], bound=True),
                    stationary_end_survival=base._estimate(adaptive_stat["end_survival"]-stat["end_survival"], bound=True)))
        return dict(arms=arms, components=components, comparisons=comparisons)

    pooled = measure()
    return dict(families=n, initial_rule_replays=2*n, policy_history_rollouts=12*n, agents_per_rule=4,
                bin_edges=list(BIN_EDGES), flip_step=150, **pooled, gate=_gate(pooled["components"], pooled["comparisons"]),
                by_initial_rule={str(rule): measure(rule) for rule in (0, 1)})


def _replicated_gates(panels):
    if [panel["index"] for panel in panels] != [0, 1, 2]:
        raise ValueError("timed exploration requires three separate registered lineages")
    qualifying = []
    for panel in panels:
        values = panel["measurements"]
        expected = _gate(values["components"], values["comparisons"])
        if values["gate"] != expected or type(values["gate"]["passed"]) is not bool:
            raise ValueError("exploration gate disagrees with its recorded paired components")
        if expected["passed"]:
            qualifying.append(panel["index"])
    return dict(timed_exploration=dict(count=len(qualifying), seeds=qualifying, passed=len(qualifying) == 3))


def operational_cost(status):
    if (status.get("status") != "finished" or status.get("completed_families") != 4
            or status.get("expected_families") != 4 or status.get("policy_order") != list(POLICIES)):
        raise ValueError("operational cost requires the complete four-family development rehearsal")
    times = np.asarray(status.get("family_policy_seconds"), np.float64)
    if times.shape != (4, 3) or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("development timing must retain all cold and warm policy observations")
    ordinary, adaptive, quota = (float(value) for value in np.median(times[1:], axis=0))
    if ordinary <= 0:
        raise ValueError("ordinary warm timing must be positive for an incremental cost comparison")
    ratio = adaptive / ordinary
    return dict(warm_family_indices=[1, 2, 3], ordinary_median_seconds=ordinary,
                adaptive_median_seconds=adaptive, quota_median_seconds=quota,
                ratio=ratio, quota_ratio=quota/ordinary, max_ratio=1.5, passed=ratio <= 1.5)


def _source_code(root):
    names = ("seed/exploration.py", "seed/timed_exploration.py", "scripts/run_timed_exploration.py")
    return {**base._source_code(root), **{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def _comparison_extra(root, audit, verified, decisions):
    candidates = list((root / "runs").glob("*_E131-benchmark"))
    if len(candidates) != 1:
        raise ValueError("exploration adoption needs the single registered development rehearsal")
    folder = candidates[0]
    config = json.loads(verified(folder / "config.json"))
    state = json.loads(verified(folder / "status.json"))
    if (config.get("protocol") != PROTOCOL or config.get("condition_split") != "development"
            or config.get("args", {}).get("name") != "E131-benchmark" or config.get("eval_seed") != DEVELOPMENT_SEED
            or config.get("source_genome") != SOURCES[0] or config.get("expected_families") != 4):
        raise ValueError("development cost must come from the unchanged registered source and assignments")
    verified(folder / "metrics.jsonl")
    verified(folder / "assignments.npz", config["assignments"]["sha256"])
    verified(folder / "genome_final.npy", SOURCES[0]["sha256"])
    summary = json.loads(verified(folder / "summary.json"))
    if summary.get("artifact_validity") != "passed" or len(summary.get("raw_files", [])) != 4:
        raise ValueError("development evidence is incomplete")
    for index, artifact in enumerate(summary["raw_files"]):
        if artifact.get("index") != index or artifact.get("path") != f"families/family_{index:04d}.npz":
            raise ValueError("development timing must retain all four original families")
        verified(folder / artifact["path"], artifact["sha256"])
    cost = operational_cost(state)
    if cost != operational_cost(summary) or cost != audit.get("operational_cost"):
        raise ValueError("runner and independent audit disagree on the separate operational cost")
    passed = decisions["timed_exploration"]["passed"]
    return dict(operational_cost=cost, adoption=dict(behavioral_passed=passed,
                operational_passed=cost["passed"], passed=passed and cost["passed"]))


def run_tracked(args, git, root, *, explicit=None):
    if (tuple(engine.POLICIES) != POLICIES or engine.ASSIGNMENT_TAG != 0xE131
            or engine.POLICY_PARAMETERS != CONFIG_EXTRA["policy_parameters"]):
        raise ValueError("exploration policy parameters differ from registration")
    return base.run_tracked(args, git, root, explicit=explicit, study=sys.modules[__name__])


def build_comparison(root):
    return base.build_comparison(root, study=sys.modules[__name__])
