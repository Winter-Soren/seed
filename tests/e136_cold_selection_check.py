"""Trace two real population shapes in a fresh process without sampling worlds."""
import json
from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seed import background_selection as study
from seed import encounter_diagnostic as encounter
from seed import scene_transfer as scene


def main():
    assert not scene._first_decisions.cache_info().currsize
    assert not encounter._unravel.cache_info().currsize
    checks = ((8, 0), (16, 8))
    args = (jax.ShapeDtypeStruct((16, 155), np.float32),
            jax.ShapeDtypeStruct((32, 5), np.int32),
            jax.ShapeDtypeStruct((16, 2), np.uint32),
            jax.ShapeDtypeStruct((2,), np.uint32),
            jax.ShapeDtypeStruct((16, 16), np.int32),
            jax.ShapeDtypeStruct((2,), np.uint32))
    for pre, gap in checks:
        with jax.checking_leaks():
            found = jax.eval_shape(study._population_cell(pre, gap), *args)
        assert found["probabilities"].shape == (16, 2, 2, 2, 12, 5)
        assert found["view_fitness"].shape == (16, 2)
    print(json.dumps(dict(status="passed", cells=checks, cold_cache=True,
                          abstract_trace_only=True, executed_rollouts=0, eval_keys_created=False)))


if __name__ == "__main__":
    main()
