"""BROADER v1: saved-genome parameter shifts, separate from EVAL v1/v2."""
import hashlib
import io
import math
import time
from pathlib import Path

import numpy as np

from seed.seed import Cfg, genome_size, test_genome


PROTOCOL = "broader-v1"
SPLIT_SEEDS = {"development": 8108, "final": 8109}
CONDITIONS = {
    "control": {},
    "sparse": {"n_items": 12},
    "dense": {"n_items": 36},
    "low-cost": {"decay": 0.0015},
    "high-cost": {"decay": 0.006},
}


def validate_request(args, c):
    if (args.generations != 0 or not args.start.startswith("file:") or args.resume
            or args.quick or args.no_plastic):
        raise ValueError("broader-v1 requires --generations 0, --start file:<path>, and no resume/quick/no-plastic")
    if args.eval_worlds < 1:
        raise ValueError("broader-v1 needs at least one evaluation world")
    if c.n_items != Cfg().n_items or c.decay != Cfg().decay:
        raise ValueError("broader-v1 owns n_items/decay shifts; do not override their control values")
    if c.G < 9 or c.T < 2 or c.M < 1 or c.H < 1 or c.n_htypes < 1 or c.H % c.n_htypes:
        raise ValueError("broader-v1 needs space for 72 items, T>=2, M>=1 and valid hidden types")
    if args.condition_split == "final" and (c != Cfg() or args.eval_worlds != 128):
        raise ValueError("broader-v1 final is fixed to default Cfg and 128 evaluation worlds")


def load_saved_genome(spec, c):
    """Hash and load the same bytes; never load pickle or change the source file."""
    path = Path(spec.removeprefix("file:"))
    data = path.read_bytes()
    flat = np.load(io.BytesIO(data), allow_pickle=False)
    if not isinstance(flat, np.ndarray) or flat.shape != (genome_size(c),) or flat.dtype != np.float32 or not np.isfinite(flat).all():
        raise ValueError("saved genome must be a finite float32 vector with the configured genome size")
    return flat, data, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def evaluate_conditions(mu, c, n_worlds, split):
    """Each pair and shift uses identical world/development/action PRNG keys."""
    results = {}
    for label, shift in CONDITIONS.items():
        shifted = c._replace(**shift)
        started = time.monotonic()
        learning = test_genome(mu, shifted, n_worlds=n_worlds, plastic=True, seed=SPLIT_SEEDS[split])
        frozen = test_genome(mu, shifted, n_worlds=n_worlds, plastic=False, seed=SPLIT_SEEDS[split])
        if any(not math.isfinite(value) for metrics in (learning, frozen) for value in metrics.values()):
            raise ValueError(f"nonfinite evaluation metric in {label}")
        gap = learning["fitness"] - frozen["fitness"]
        results[label] = dict(cfg_shift=shift, learning=learning, frozen=frozen,
                              learning_minus_frozen_fitness=gap,
                              learning_accuracy_gain=learning["acc_second"] - learning["acc_first"],
                              frozen_accuracy_gain=frozen["acc_second"] - frozen["acc_first"],
                              seconds=time.monotonic() - started)
        control = results["control"]
        results[label].update(
            learning_fitness_shift=learning["fitness"] - control["learning"]["fitness"],
            frozen_fitness_shift=frozen["fitness"] - control["frozen"]["fitness"],
            learning_advantage_shift=gap - control["learning_minus_frozen_fitness"])
    return results
