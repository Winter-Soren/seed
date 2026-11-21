"""Pure-Python reporting checks; no engine imports or runtime experiments."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import summarize


class EcologySummaryTests(unittest.TestCase):
    def test_complete_partial_and_pending_fixtures_stay_out_of_scientific_score_table(self):
        scratch = summarize.ROOT/".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ecology-summary-", dir=scratch) as temporary:
            root = Path(temporary)
            for name, status, completed in (("complete", "finished", 3), ("partial", "interrupted", 1),
                                             ("pending", None, None)):
                folder = root/"runs"/name
                folder.mkdir(parents=True)
                config = dict(args=dict(eval_protocol="ecology-lineage-v1"), condition_split="development",
                              schedule=[{}, {}, {}], git=dict(commit="a"*40, dirty=False))
                (folder/"config.json").write_text(json.dumps(config), encoding="utf-8")
                if status is not None:
                    (folder/"summary.json").write_text(json.dumps(dict(status=status, completed_cohorts=completed,
                        expected_cohorts=3, condition_split="development")), encoding="utf-8")
            before = {p: p.read_bytes() for p in (root/"runs").rglob("*.json")}
            with patch.object(summarize, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            text = (root/"results/RESULTS.md").read_text()
            scientific, development = text.split("## Ecology/lineage development fixtures", 1)
            for name in ("complete", "partial", "pending"):
                self.assertNotIn("| "+name+" |", scientific)
            self.assertIn("| complete | finished | 3/3 | development | aaaaaaa |", development)
            self.assertIn("| partial | interrupted | 1/3 | development | aaaaaaa |", development)
            self.assertIn("| pending | no summary | ?/3 | development | aaaaaaa |", development)
            self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == "__main__":
    unittest.main()
