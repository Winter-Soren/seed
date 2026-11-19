"""EVAL v5: fixed E115 continuations, paired worlds and the preregistered gate."""
import numpy as np

from seed.seed import Cfg
from scripts import eval_v3, eval_v4

PROTOCOL = "v5"
EVAL_SEED = 2026111903
N_WORLDS = 256
SPLIT_SEEDS = {"development": 41, "final": EVAL_SEED}
REFERENCES = eval_v4.REFERENCES
CONTROLLERS = ("rules-v0", *REFERENCES)
CONDITIONS = eval_v4.CONDITIONS
CONTRAST_METRICS = eval_v4.CONTRAST_METRICS
PAIRING_KEYS = eval_v4.PAIRING_KEYS
PANELS = tuple(f"{arm}{seed}" for seed in range(3) for arm in ("start", "mean", "min")) + REFERENCES


def evaluate_controller(mu_or_none, c, n_worlds=N_WORLDS, eval_seed=EVAL_SEED, *,
                        controller="rules-v0", on_condition=None):
    """Reuse v4 physics and measurements on the separately registered v5 worlds."""
    if controller not in CONTROLLERS or type(n_worlds) is not int or n_worlds < 2:
        raise ValueError("EVAL v5 needs a registered controller and at least two base worlds")
    if eval_seed not in SPLIT_SEEDS.values():
        raise ValueError("use the registered EVAL v5 held-out seed or development seed 41")
    if eval_seed == EVAL_SEED:
        if c != Cfg() or n_worlds != N_WORLDS:
            raise ValueError("held-out EVAL v5 fixes default Cfg and 256 base worlds")
    elif (c.T > 30 or c.M > 4 or c.H > 6 or c.R > 1 or c.G > 8 or n_worlds > 4):
        raise ValueError("development uses only tiny fixtures")
    return eval_v4._evaluate_controller(mu_or_none, c, n_worlds, eval_seed,
                                        controller=controller, on_condition=on_condition)


def compare_panels(raw_by_panel):
    """Return 30 descriptive contrasts and the fixed numerical research gate.

    Run/provenance validity is the runner's responsibility. This function requires
    all 14 evaluated panels but cannot certify the six source training runs.
    Independent samples are base worlds; each vector already averages agents and
    the two rules as prescribed in v4. There is no pooling across training seeds.
    """
    if set(raw_by_panel) != set(PANELS):
        raise ValueError("EVAL v5 comparison requires exactly the 14 registered panels")
    anchor = raw_by_panel[PANELS[0]]
    if any(key not in anchor for key in PAIRING_KEYS):
        raise ValueError("paired controller provenance is incomplete")
    wkeys = np.asarray(anchor["wkeys"])
    if wkeys.ndim != 2 or wkeys.shape[0] < 2:
        raise ValueError("paired controller provenance needs at least two base worlds")
    n_worlds = wkeys.shape[0]
    vectors = {}
    for label in PANELS:
        raw = raw_by_panel[label]
        for key in PAIRING_KEYS:
            if key not in raw or not np.array_equal(anchor[key], raw[key]):
                raise ValueError(f"paired controller {key} mismatch or missing for {label}")
        vectors[label] = {}
        for metric in CONTRAST_METRICS:
            key = "paired_" + metric
            if key not in raw:
                raise ValueError(f"missing {metric} vector for {label}")
            vector = np.asarray(raw[key], dtype=np.float64)
            if vector.shape != (n_worlds,) or not np.isfinite(vector).all():
                raise ValueError("contrast vectors must be finite and match the base worlds")
            vectors[label][metric] = vector
    rows, per_seed = [], []
    for seed in range(3):
        components = {}
        for control in ("mean", "start"):
            comparison = "min-minus-" + control
            minuend, subtrahend = f"min{seed}", f"{control}{seed}"
            for metric in CONTRAST_METRICS:
                delta = vectors[minuend][metric] - vectors[subtrahend][metric]
                estimate = eval_v3.mean_se(delta)
                rows.append(dict(seed=seed, comparison=comparison, metric=metric,
                                 minuend=minuend, subtrahend=subtrahend, **estimate,
                                 n_worlds=n_worlds, per_world_difference=delta.tolist()))
                if metric == "individual_min_return":
                    threshold = 2 * estimate["se"]
                    components[comparison + "_minimum_return"] = dict(
                        **estimate, threshold=threshold, operator=">",
                        passed=bool(estimate["mean"] > threshold))
                elif control == "start" and metric == "joint_survival":
                    threshold = -2 * estimate["se"]
                    components["min-minus-start_joint_survival"] = dict(
                        **estimate, threshold=threshold, operator=">=",
                        passed=bool(estimate["mean"] >= threshold))
        per_seed.append(dict(seed=seed, **components,
                             passed=all(item["passed"] for item in components.values())))
    improved = all(item[name]["passed"] for item in per_seed for name in (
        "min-minus-mean_minimum_return", "min-minus-start_minimum_return"))
    survival = all(item["min-minus-start_joint_survival"]["passed"] for item in per_seed)
    gate = dict(passed=improved and survival, all_minimum_improvements=improved,
                no_joint_survival_regression=survival, per_seed=per_seed)
    return dict(contrasts=rows, gate=gate)
