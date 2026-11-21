"""EVAL v6: fixed E116 learning-rate comparison on fresh paired worlds."""
from scripts import eval_v5

PROTOCOL = "v6"
EVAL_SEED = 2026112001
N_WORLDS = eval_v5.N_WORLDS
SPLIT_SEEDS = {"development": 41, "final": EVAL_SEED}
REFERENCES = eval_v5.REFERENCES
CONTROLLERS = eval_v5.CONTROLLERS
CONDITIONS = eval_v5.CONDITIONS
CONTRAST_METRICS = eval_v5.CONTRAST_METRICS
PAIRING_KEYS = eval_v5.PAIRING_KEYS
PANELS = tuple(f"{arm}{seed}" for seed in range(3) for arm in ("start", "control", "candidate")) + REFERENCES


def evaluate_controller(mu_or_none, c, n_worlds=N_WORLDS, eval_seed=EVAL_SEED, *,
                        controller="rules-v0", on_condition=None):
    return eval_v5._evaluate_registered(mu_or_none, c, n_worlds, eval_seed, protocol=PROTOCOL,
                                        final_seed=EVAL_SEED, controller=controller, on_condition=on_condition)


def compare_panels(raw_by_panel):
    """Thirty fixed contrasts; run/provenance validity remains the runner's job."""
    return eval_v5._compare_panels(raw_by_panel, candidate="candidate", control="control", protocol=PROTOCOL)
