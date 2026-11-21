"""Runner persistence checks use constructed movies, never simulated worlds."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_social_channel as runner
from scripts import social_channel as channel
from test_social_channel import family


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "seed" / "seed.py").is_file())


class ChannelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e121-runner-test-", dir=ROOT / ".tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source_path = self.root / "fixtures" / "a0.npy"
        source_path.parent.mkdir()
        stream = io.BytesIO()
        np.save(stream, np.zeros(155, np.float32), allow_pickle=False)
        content = stream.getvalue()
        source_path.write_bytes(content)
        self.source = dict(label="A0", training_seed=0, path="fixtures/a0.npy",
                           sha256=hashlib.sha256(content).hexdigest())
        self.args = types.SimpleNamespace(eval_protocol=runner.PROTOCOL, condition_split="development",
            name="E121-benchmark", seed=48, eval_worlds=4, start="file:fixtures/a0.npy", generations=0,
            quick=False, allow_dirty=False, resume=None, cfg=None, no_plastic=False, controller=None,
            structural_plasticity=None, objective=None, memory_condition=None, persistence=False,
            max_train_seconds=None, pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10,
            hypothesis="synthetic persistence check")
        self.git = dict(commit="a" * 40, dirty=False)
        self.addCleanup(patch.stopall)
        patch.object(runner, "SOURCES", (self.source,)).start()
        patch.object(runner, "_runtime", return_value={"synthetic_test": True}).start()
        patch.object(channel, "family_keys", side_effect=self.keys).start()
        patch.object(channel, "evaluate_family", side_effect=self.row).start()
        # The evidence here is constructed public video, not a hidden simulated teacher.
        patch.object(channel, "validate_family", return_value=None).start()

    @staticmethod
    def keys(master, index):
        return np.array([[master, 2 * index], [master, 2 * index + 1]], np.uint32)

    @staticmethod
    def row(flat, keys, index, c):
        return dict(family(index), keys=keys.copy(), family_index=np.asarray(index, np.int32),
                    yoked_swap=np.asarray(bool((index // 2) % 2)))

    def test_complete_development_evidence_is_immutable_and_not_scientific(self):
        folder = runner.run_tracked(self.args, self.git, self.root)
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["completed_families"], 4)
        self.assertEqual(summary["artifact_validity"], "passed")
        self.assertFalse(summary["scientific_result"])
        self.assertEqual(summary["gate_status"], "not_applicable")
        self.assertNotIn("gate", summary)
        self.assertEqual(len(summary["raw_files"]), 4)
        for artifact in summary["raw_files"]:
            content = (folder / artifact["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), artifact["sha256"])
            with np.load(io.BytesIO(content), allow_pickle=False) as saved:
                channel.validate_decoding(dict(saved))
        before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "attempt exists"):
            runner.run_tracked(self.args, self.git, self.root)
        after = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_failure_preserves_the_prefix_without_a_scientific_decision(self):
        def stop_after_first(flat, keys, index, c):
            if index == 1:
                raise RuntimeError("synthetic interruption")
            return self.row(flat, keys, index, c)
        with patch.object(channel, "evaluate_family", side_effect=stop_after_first):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                runner.run_tracked(self.args, self.git, self.root)
        folder, = (self.root / "runs").iterdir()
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["completed_families"], 1)
        self.assertEqual(summary["gate_status"], "undecided")
        self.assertEqual(summary["artifact_validity"], "incomplete")
        self.assertFalse(summary["scientific_result"])
        self.assertNotIn("measurements", summary)
        self.assertNotIn("gate", summary)
        self.assertEqual(len(list((folder / "families").glob("*.npz"))), 1)

    def test_duplicate_teacher_keys_fail_before_a_second_family_is_run(self):
        with patch.object(channel, "family_keys", return_value=self.keys(48, 0)):
            with self.assertRaisesRegex(ValueError, "unique across all families"):
                runner.run_tracked(self.args, self.git, self.root)
        self.assertEqual(channel.evaluate_family.call_count, 1)

    def test_unregistered_identity_source_or_override_never_creates_a_run(self):
        for changed in ({"seed": 49}, {"eval_worlds": 128}, {"generations": 1},
                        {"start": "file:fixtures/wrong.npy"}, {"allow_dirty": True}):
            args = types.SimpleNamespace(**(vars(self.args) | changed))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.run_tracked(args, self.git, self.root)
        with self.assertRaises(ValueError):
            runner.run_tracked(self.args, self.git, self.root, explicit={"--sigma"})
        with self.assertRaises(ValueError):
            runner.run_tracked(self.args, dict(self.git, dirty=True), self.root)
        (self.root / self.source["path"]).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            runner.run_tracked(self.args, self.git, self.root)
        self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
