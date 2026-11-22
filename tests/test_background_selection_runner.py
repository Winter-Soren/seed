"""Synthetic E136 assignment and endpoint-seal invariants."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import run_background_selection as runner


class BackgroundSelectionRunnerTests(unittest.TestCase):
    def test_training_arm_pairs_share_seed_and_workload(self):
        requests = runner._requests()
        seeds = set()
        for index in range(3):
            isolated = requests[f"E136-A{index}-isolated"]
            background = requests[f"E136-A{index}-background"]
            self.assertEqual(isolated[:3]+isolated[4:], background[:3]+background[4:])
            self.assertNotEqual(isolated[3], background[3])
            self.assertEqual(isolated[-2:], (10, 0))
            seeds.add(isolated[4])
            seeds.add(requests[f"E136-A{index}"][4])
        self.assertEqual(len(seeds), 6)

    def test_balanced_evaluation_families_use_independent_roles(self):
        result = runner.evaluation_assignments(49, 128)
        keys = result["family_keys"]
        self.assertEqual(keys.shape, (128, 7, 2))
        self.assertEqual(len({key.tobytes() for key in keys.reshape(-1, 2)}), 128*7)
        np.testing.assert_array_equal(result["initial_rules"], [0, 1])
        np.testing.assert_array_equal(np.stack((result["pre_contacts"],result["gap"]),axis=1),
                                      np.tile([(8,16),(16,32),(8,32),(16,16)], (32,1)))

    def test_final_seal_requires_all_six_endpoints(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "runs").mkdir()
            with self.assertRaisesRegex(ValueError, "paired training endpoints"):
                runner._endpoint_receipts(root, {}, "0"*40)


if __name__ == "__main__":
    unittest.main()
