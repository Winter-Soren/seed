"""E138 release and assignment fixtures without opening registered final keys."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_solo_selection as runner


class SoloSelectionRunnerTests(unittest.TestCase):
    def test_registered_arms_pair_work_and_masters(self):
        requests = runner._requests()
        for lineage in range(3):
            name = f"E138-A{lineage}"
            proxy, direct = (requests[f"{name}-{arm}"] for arm in runner.VIEWS)
            self.assertEqual(proxy[:3] + proxy[4:], direct[:3] + direct[4:])
            self.assertEqual(proxy[3], "background")
            self.assertEqual(direct[3], "solo")
            self.assertEqual(proxy[-2:], (10, 0))
            self.assertEqual(requests[name][-2:], (0, 128))
            self.assertNotEqual(requests[name][-3], proxy[-3])

    def test_assignment_aliases_cannot_open_keys_without_exact_endpoint_pins(self):
        commit = "a" * 40
        receipts = [{"seed_index": lineage, "view": view}
                    for lineage in range(3) for view in runner.VIEWS]
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            scratch = root / ".tmp"
            scratch.mkdir()
            content = json.dumps({"git_commit": commit, "attempt": 10, "endpoints": receipts}).encode()
            (scratch / "E138-endpoints.json").write_bytes(content)
            reference = dict(finalPanelsReleased=False, releasedSource=commit,
                             endpointManifestSha256=hashlib.sha256(content).hexdigest())
            path = scratch / "E138-offline-reference.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed"):
                runner._verify_final_release(root, receipts, commit)
            reference["finalPanelsReleased"] = True
            path.write_text(json.dumps(reference), encoding="utf-8")
            runner._verify_final_release(root, receipts, commit)
            with self.assertRaisesRegex(ValueError, "endpoint"):
                runner._verify_final_release(root, receipts[::-1], commit)


if __name__ == "__main__":
    unittest.main()
