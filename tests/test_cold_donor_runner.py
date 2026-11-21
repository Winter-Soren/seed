"""E117 runner validation, immutable evidence and interrupted-attempt handling."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from jax.flatten_util import ravel_pytree

from scripts import run_cold_donor as entry, run_experiment as runner, summarize
from seed.seed import Cfg


class ColdDonorRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cold-donor-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in entry.SOURCES:
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E117-benchmark", "--seed", "42", "--generations", "0", "--eval-worlds", "4",
                       "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls = 0
        self.fail_before = None
        self.nonfinite = False
        self.now = 0.
        self.expire_after_family = False
        self.called_indices = []

    @staticmethod
    def fake_keys(seed, index):
        return np.arange(index * 8, index * 8 + 8, dtype=np.uint32).reshape(4, 2)

    def fake_family(self, genome, keys, rule, c, ecology):
        self.assertIsInstance(genome, dict)
        flat, _ = ravel_pytree(genome)
        self.assertEqual(flat.shape, (155,))
        self.assertEqual(c, Cfg())
        self.assertEqual(ecology, entry.ECOLOGY)
        if self.calls == self.fail_before:
            raise RuntimeError("deliberate family interruption")
        self.called_indices.append(int(keys[0, 0]) // 8)
        self.calls += 1
        if self.expire_after_family:
            self.now = 601.
        return dict(keys=keys, rule=np.asarray(rule, np.int32),
                    observation=np.asarray(np.nan if self.nonfinite else self.calls, np.float32))

    @staticmethod
    def fake_summary(rows):
        estimate = dict(mean=5., se=.25 if len(rows) > 1 else None)
        arm = {name: dict(estimate) for name in ("latency", "survival", "fitness", "shelter_arrival",
            "cold_sheltered_fraction", "outdoor_cold_agent_steps", "extra_energy_lost")}
        return dict(families=len(rows), assigned_agents=4 * len(rows), outside_born_agents=3 * len(rows),
                    agents_per_family=4, rule_counts={str(rule): sum(int(row["rule"]) == rule for row in rows)
                                                    for rule in (0, 1)},
                    inherited=arm, fresh=arm, donor=arm, latency_improvement=dict(estimate),
                    survival_difference=dict(mean=0., se=.01 if len(rows) > 1 else None),
                    gate=dict(passed=True if len(rows) > 1 else None,
                              latency_passed=True if len(rows) > 1 else None,
                              survival_passed=True if len(rows) > 1 else None))

    def invoke(self, *extra, base=True, actual=False, dirty=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if base else []), *extra]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if not actual:
                stack.enter_context(patch.object(entry.cold_donor, "family_keys", side_effect=self.fake_keys))
                stack.enter_context(patch.object(entry.cold_donor, "evaluate_family", side_effect=self.fake_family))
                stack.enter_context(patch.object(entry.cold_donor, "validate_family"))
                stack.enter_context(patch.object(entry.cold_donor, "summarize", side_effect=self.fake_summary))
            runner.main()

    def folder(self, name="E117-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def attempts(self):
        return list((self.root / "runs").glob("*_E117-*"))

    @staticmethod
    def snapshot(folder):
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def recovery_fixture(self):
        """Synthetic interrupted artifacts, never actual held-out evidence."""
        source = entry.SOURCES[2]
        args = dict(name="E117-A2", start="file:" + source["path"], generations=0, pop=128, worlds=8,
            sigma=.1, lr=.1, seed=2026112103, no_plastic=False, controller=None, objective=None,
            structural_plasticity=None, eval_worlds=256, eval_protocol=entry.PROTOCOL, memory_condition=None,
            condition_split="final", checkpoint_every=10, max_train_seconds=None, cfg=None,
            hypothesis="", quick=False, resume=None, allow_dirty=False)
        cfg = dict(protocol=entry.PROTOCOL, args=args, source=dict(source),
            git=dict(commit=entry.RECOVERY_COMMIT, dirty=False), condition_split="final", cfg=Cfg()._asdict(),
            cfg_overrides={}, ecology=dict(version="ecology-v1", movement_energy_cost=.01, seasonal_energy_cost=.02),
            persistence="off", plastic=True, controller="rules-v0", genome_size=155, seed=2026112103,
            eval_seed=2026112103, eval_worlds=256, expected_families=256, key_order=list(entry.KEY_ORDER),
            cold_start=150, max_seconds=9900, raw_schema=entry.PROTOCOL,
            raw_pattern="families/family_{index:04d}.npz", runtime=entry._runtime())
        parent = self.root / entry.RECOVERY_PARENT
        (parent / "families").mkdir(parents=True)
        (parent / "config.json").write_text(json.dumps(cfg))
        (parent / "status.json").write_text(json.dumps(dict(status="running", completed_families=181,
            expected_families=256, elapsed_seconds=190.)))
        (parent / "genome_start.npy").write_bytes((self.root / source["path"]).read_bytes())
        records = []
        for index in range(181):
            keys = self.fake_keys(2026112103, index)
            row = dict(keys=keys, rule=np.asarray(index % 2, np.int32), observation=np.asarray(index, np.float32))
            path = f"families/family_{index:04d}.npz"
            np.savez_compressed(parent / path, **row)
            artifact = dict(index=index, path=path, sha256=hashlib.sha256((parent / path).read_bytes()).hexdigest())
            record = dict(family=index, rule=index % 2, keys=keys.tolist(), raw_file=artifact,
                          elapsed_seconds=float(index + 1), measurements=dict(families=1))
            records.append((json.dumps(record) + "\r\n").encode())
        (parent / "metrics.jsonl").write_bytes(b"".join(records))
        guard = self.root / entry.RECOVERY_GUARD
        guard.parent.mkdir(parents=True)
        guard.write_text(json.dumps(dict(event="stopped", processStarted=True, error="Call cancelled ",
                                         wallSeconds=entry.RECOVERY_PARENT_SECONDS)) + "\n")
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in dict(
            config=parent / "config.json", status=parent / "status.json", metrics=parent / "metrics.jsonl",
            guard=guard).items()}
        return parent, hashes

    def invoke_recovery(self, hashes, *extra):
        with patch.object(entry, "RECOVERY_HASHES", hashes), patch.object(entry, "_scientific_source",
                return_value=dict(seed="b" * 40, cold_donor="c" * 40)):
            self.invoke("--condition-split", "final", "--name", entry.RECOVERY_NAME, "--seed", "2026112103",
                        "--eval-worlds", "256", "--start", "file:" + entry.SOURCES[2]["path"],
                        "--resume", entry.RECOVERY_PARENT, *extra)

    def test_recovery_copies_exact_prefix_and_evaluates_only_the_remaining_families(self):
        parent, hashes = self.recovery_fixture()
        before = self.snapshot(parent)
        self.invoke_recovery(hashes)
        folder = self.folder(entry.RECOVERY_NAME)
        result = json.loads((folder / "summary.json").read_text())
        cfg = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.called_indices, list(range(181, 256)))
        self.assertEqual(self.calls, 75)
        self.assertEqual(before, self.snapshot(parent))
        self.assertEqual((result["status"], result["completed_families"]), ("finished", 256))
        self.assertEqual((cfg["max_seconds"], cfg["cumulative_max_seconds"]), (9499, 9900))
        self.assertEqual(cfg["recovery"], result["recovery"])
        self.assertEqual(cfg["recovery"]["parent_source_commit"], entry.RECOVERY_COMMIT)
        self.assertEqual(cfg["git"]["commit"], "a" * 40)
        self.assertEqual(cfg["args"]["resume"], entry.RECOVERY_PARENT)
        for relative in ("config.json", "status.json"):
            self.assertEqual((folder / "parent" / relative).read_bytes(), before[relative])
        self.assertEqual((folder / "parent/guard.jsonl").read_bytes(), (self.root / entry.RECOVERY_GUARD).read_bytes())
        copied_metrics = (folder / "metrics.jsonl").read_bytes()
        self.assertTrue(copied_metrics.startswith(before["metrics.jsonl"]))
        for index in range(181):
            relative = f"families/family_{index:04d}.npz"
            self.assertEqual((folder / relative).read_bytes(), before[relative])
        history = [json.loads(line) for line in copied_metrics.splitlines()]
        self.assertEqual(len(history), 256)
        self.assertEqual([record["family"] for record in history], list(range(256)))
        self.assertTrue(all(history[i]["elapsed_seconds"] <= history[i + 1]["elapsed_seconds"] for i in range(255)))
        for record in history[181:]:
            self.assertEqual(record["elapsed_seconds"], entry.RECOVERY_PARENT_SECONDS + record["invocation_elapsed_seconds"])
            self.assertEqual(record["source_commit"], "a" * 40)
        self.assertEqual(result["elapsed_seconds"], entry.RECOVERY_PARENT_SECONDS + result["invocation_elapsed_seconds"])
        self.assertEqual([(s["first_family"], s["stop_family"]) for s in result["segments"]], [(0, 181), (181, 256)])
        saved = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke_recovery(hashes)
        self.assertEqual(saved, self.snapshot(folder))

    def test_recovery_rejects_tampered_or_extra_prefix_artifacts_before_new_folder(self):
        parent, hashes = self.recovery_fixture()
        paths = [parent / "config.json", parent / "status.json", parent / "metrics.jsonl",
                 parent / "genome_start.npy", parent / "families/family_0007.npz", self.root / entry.RECOVERY_GUARD]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(path=path.name), self.assertRaises(SystemExit):
                self.invoke_recovery(hashes)
            path.write_bytes(original)
            self.assertFalse(list((self.root / "runs").glob("*_" + entry.RECOVERY_NAME)))
        (parent / "summary.json").write_text("{}")
        with self.assertRaises(SystemExit):
            self.invoke_recovery(hashes)
        self.assertEqual(self.calls, 0)

    def test_recovery_failure_retains_prefix_and_never_recreates_old_families(self):
        parent, hashes = self.recovery_fixture()
        before = self.snapshot(parent)
        self.fail_before = 1
        with self.assertRaisesRegex(RuntimeError, "family interruption"):
            self.invoke_recovery(hashes)
        folder = self.folder(entry.RECOVERY_NAME)
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(self.called_indices, [181])
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 182))
        self.assertNotIn("gate", result)
        self.assertNotIn("measurements", result)
        self.assertEqual(result["gate_status"], "undecided")
        self.assertEqual(before, self.snapshot(parent))
        self.assertTrue((folder / "metrics.jsonl").read_bytes().startswith(before["metrics.jsonl"]))

    def test_recovery_requires_exact_parent_identity_and_unchanged_scientific_source(self):
        parent, hashes = self.recovery_fixture()
        for extra in (("--resume", "runs/wrong"), ("--name", "E117-A2"), ("--seed", "2026112102"),
                      ("--condition-split", "development"), ("--eval-worlds", "75"), ("--cfg", "T=300")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke_recovery(hashes, *extra)
        with patch.object(entry, "RECOVERY_HASHES", hashes), patch.object(entry, "_scientific_source",
                side_effect=[dict(seed="b" * 40, cold_donor="c" * 40), dict(seed="d" * 40, cold_donor="c" * 40)]):
            with self.assertRaises(SystemExit):
                self.invoke("--condition-split", "final", "--name", entry.RECOVERY_NAME, "--seed", "2026112103",
                            "--eval-worlds", "256", "--start", "file:" + entry.SOURCES[2]["path"],
                            "--resume", entry.RECOVERY_PARENT)
        self.assertFalse(list((self.root / "runs").glob("*_" + entry.RECOVERY_NAME)))
        self.assertEqual(self.calls, 0)

    def test_benchmark_keeps_exact_source_full_ecology_and_exclusive_family_files(self):
        self.invoke()
        folder = self.folder()
        cfg = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(cfg["cfg"], dict(Cfg()._asdict()))
        self.assertEqual(cfg["ecology"], dict(version="ecology-v1", movement_energy_cost=.01, seasonal_energy_cost=.02))
        self.assertEqual(cfg["max_seconds"], 600)
        self.assertEqual(cfg["persistence"], "off")
        self.assertEqual(cfg["source"], entry.SOURCES[0])
        self.assertEqual(hashlib.sha256((folder / "genome_start.npy").read_bytes()).hexdigest(), entry.SOURCES[0]["sha256"])
        self.assertEqual((result["status"], result["completed_families"], result["expected_families"]), ("finished", 4, 4))
        self.assertEqual(result["artifact_validity"], "passed")
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertNotIn("gate", result["measurements"])
        history = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["rule"] for row in history], [0, 1, 0, 1])
        self.assertEqual(len(history), 4)
        for i, artifact in enumerate(result["raw_files"]):
            path = folder / artifact["path"]
            self.assertEqual(artifact["path"], f"families/family_{i:04d}.npz")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            with np.load(path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["keys"], self.fake_keys(42, i))
                self.assertEqual(int(arrays["rule"]), i % 2)
            self.assertEqual(history[i]["raw_file"], artifact)
        self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "finished")
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), base=False)
        self.assertEqual(before, self.snapshot(folder))

    def test_rejects_overrides_and_noncanonical_identity_before_creating_attempt(self):
        invalid = [("--quick",), ("--allow-dirty",), ("--cfg", "T=300"), ("--generations", "1"),
                   ("--seed", "41"), ("--eval-worlds", "8"), ("--name", "other"), ("--condition-split", "final"),
                   ("--start", "biology"), ("--start", "file:" + entry.SOURCES[1]["path"]),
                   ("--no-plastic",), ("--persistence",), ("--pop", "128"), ("--worlds", "8"),
                   ("--checkpoint-every", "10"), ("--max-train-seconds", "10"), ("--sigma", "0.1"),
                   ("--lr", "0.1"), ("--controller", "rules-v0"), ("--structural-plasticity", "off"),
                   ("--objective", "paired-min"), ("--memory-condition", "shared")]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke(*args)
            self.assertFalse(self.attempts())
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        with self.assertRaises(SystemExit):
            self.invoke("--allow-dirty", dirty=True)
        self.assertFalse(self.attempts())
        self.assertEqual(self.calls, 0)

    def test_all_final_seed_source_pairs_are_fixed_and_npy_hash_is_checked(self):
        args = argparse.Namespace(eval_protocol=entry.PROTOCOL, condition_split="final", name="E117-A0",
            generations=0, seed=2026112101, eval_worlds=256, quick=False, allow_dirty=False, resume=None, cfg=None,
            no_plastic=False, controller=None, structural_plasticity=None, objective=None, memory_condition=None,
            max_train_seconds=None, pop=128, worlds=8, sigma=.1, lr=.1, checkpoint_every=10,
            start="file:" + entry.SOURCES[0]["path"])
        git = dict(commit="a" * 40, dirty=False)
        for source in entry.SOURCES:
            args.name = "E117-" + source["label"]
            args.seed = 2026112101 + source["training_seed"]
            args.start = "file:" + source["path"]
            checked, content, flat = entry._validate(args, git, self.root, None)
            self.assertEqual(checked, source)
            self.assertEqual(flat.shape, (155,))
            self.assertEqual(hashlib.sha256(content).hexdigest(), source["sha256"])
            args.seed += 1
            with self.assertRaises(ValueError):
                entry._validate(args, git, self.root, None)
        path = self.root / entry.SOURCES[0]["path"]
        changed = bytearray(path.read_bytes())
        changed[-1] ^= 1
        path.write_bytes(changed)
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.attempts())

    def test_final_gate_requires_all_256_families(self):
        self.invoke("--condition-split", "final", "--name", "E117-A0", "--seed", "2026112101", "--eval-worlds", "256")
        folder = self.folder("E117-A0")
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual(result["completed_families"], 256)
        self.assertEqual(result["measurements"]["rule_counts"], {"0": 128, "1": 128})
        self.assertTrue(result["scientific_result"])
        self.assertEqual(result["gate"], dict(passed=True, latency_passed=True, survival_passed=True))
        self.assertEqual(result["gate_status"], "passed")
        self.assertEqual(len(result["raw_files"]), 256)
        self.assertEqual(json.loads((folder / "config.json").read_text())["max_seconds"], 9900)

    def test_interrupted_final_retains_completed_family_without_gate_or_pooled_scores(self):
        self.fail_before = 1
        with self.assertRaisesRegex(RuntimeError, "family interruption"):
            self.invoke("--condition-split", "final", "--name", "E117-A0", "--seed", "2026112101", "--eval-worlds", "256")
        folder = self.folder("E117-A0")
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 1))
        self.assertEqual(result["gate_status"], "undecided")
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        self.assertNotIn("measurements", result)
        self.assertTrue((folder / "families/family_0000.npz").exists())
        self.assertFalse((folder / "families/family_0001.npz").exists())
        self.assertEqual(len((folder / "metrics.jsonl").read_text().splitlines()), 1)
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke("--condition-split", "final", "--name", "E117-A0", "--seed", "2026112101", "--eval-worlds", "256")
        self.assertEqual(before, self.snapshot(folder))

    def test_nonfinite_raw_is_rejected_before_saving_family(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 0))
        self.assertFalse(list((folder / "families").glob("*.npz")))
        self.assertNotIn("gate", result)

    def test_deadline_preserves_last_completed_family(self):
        self.expire_after_family = True
        with patch.object(entry.time, "monotonic", side_effect=lambda: self.now), self.assertRaises(TimeoutError):
            self.invoke()
        result = json.loads((self.folder() / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("interrupted", 1))
        self.assertEqual(result["gate_status"], "undecided")
        self.assertTrue((self.folder() / "families/family_0000.npz").exists())

    def test_summary_separates_benchmark_and_suppresses_partial_measurements(self):
        self.invoke()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        rendered = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("Cold donor inheritance (cold-donor-v1)", rendered)
        self.assertIn("5.000000 ± 0.250000", rendered)
        self.assertIn("not applicable", rendered)
        path = self.folder() / "summary.json"
        partial = json.loads(path.read_text())
        partial.update(status="interrupted", completed_families=3, artifact_validity="incomplete")
        path.write_text(json.dumps(partial))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        rendered = (self.root / "results/RESULTS.md").read_text()
        self.assertNotIn("5.000000 ± 0.250000", rendered)
        self.assertIn("undecided", rendered)

    def test_actual_runner_decodes_and_checkpoints_small_development_fixture(self):
        # A test-local shape reduction exercises the actual CLI/storage boundary, never a final seed.
        tiny = Cfg(G=6, M=2, T=4, R=1, H=3, n_items=4)
        with patch.object(entry, "Cfg", return_value=tiny):
            self.invoke(actual=True)
        folder = self.folder()
        result = json.loads((folder / "summary.json").read_text())
        self.assertEqual((result["status"], result["completed_families"]), ("finished", 4))
        self.assertEqual(result["measurements"]["assigned_agents"], 8)
        self.assertFalse(result["scientific_result"])
        self.assertNotIn("gate", result)
        for artifact in result["raw_files"]:
            with np.load(folder / artifact["path"], allow_pickle=False) as arrays:
                row = dict(arrays)
            entry.cold_donor.validate_family(row, tiny)
            self.assertEqual(row["inherited_food"].shape, (4, 2))
            np.testing.assert_array_equal(row["inherited_birth_sha256"], row["fresh_birth_sha256"])


if __name__ == "__main__":
    unittest.main()
