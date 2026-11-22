"""E134 runner checks using synthetic identities and no registered final seeds."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import run_replay_selection as runner


class ReplaySelectionRunnerTests(unittest.TestCase):
    def test_manifest_has_matched_strata_and_independent_key_roles(self):
        result = runner.evaluation_assignments(41, 128)
        repeated = runner.evaluation_assignments(41, 128)
        for name in result:
            np.testing.assert_array_equal(result[name], repeated[name])
        family_keys = result["family_keys"]
        self.assertEqual(family_keys.shape, (128, 6, 2))
        self.assertEqual(len({key.tobytes() for key in family_keys.reshape(-1, 2)}), 128*6)
        cells = np.stack([result[name] for name in ("pre_contacts", "gap", "sham_rule")], axis=1)
        _, counts = np.unique(cells, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(8, 16))
        np.testing.assert_array_equal(result["initial_rules"], [0, 1])
        dev = runner.evaluation_assignments(42, 4)
        np.testing.assert_array_equal(np.stack([dev[name] for name in ("pre_contacts", "gap", "sham_rule")], axis=1),
                                      [(8, 16, 0), (16, 32, 0), (8, 32, 1), (16, 16, 1)])

    def test_final_endpoint_receipts_require_the_complete_triplet(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runs").mkdir()
            with self.assertRaisesRegex(ValueError, "all three"):
                runner._endpoint_receipts(root, {}, "a"*40)


if __name__ == "__main__":
    unittest.main()
