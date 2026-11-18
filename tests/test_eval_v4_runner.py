"""E114 dispatch, exact inputs, incremental evidence and development-only integration."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seed import direct
from scripts import eval_v3, eval_v4, run_experiment as runner, summarize


class V4RunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = runner.Cfg(G=2, M=2, T=4, R=0, H=3, n_items=1)
        self.manifest = []
        for label, controller, seed in [(f"A{i}", "rules-v0", i) for i in range(3)] + [
                (f"direct{i}", direct.FORMAT, i) for i in range(3)]:
            folder = self.root / "fixtures" / label
            folder.mkdir(parents=True)
            size = direct.genome_size(self.cfg) if controller == direct.FORMAT else 155
            flat = np.full(size, .01 * seed, dtype=np.float32)
            np.save(folder / "genome_final.npy", flat)
            args = dict(seed=seed, start="blank", generations=200, no_plastic=controller == direct.FORMAT,
                        quick=False, allow_dirty=False, structural_plasticity=None,
                        pop=128, worlds=8, sigma=.1, lr=.1)
            config = dict(args=args, cfg=self.cfg._asdict(), genome_size=size,
                          git=dict(commit="a" * 40, dirty=False))
            if controller == direct.FORMAT:
                args.update(controller=controller, eval_protocol=runner.DIRECT_PROTOCOL,
                            condition_split="final", structural_plasticity="off")
                config.update(controller=controller, encoding=dict(format=controller, weights=size))
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "summary.json").write_text(json.dumps(dict(
                status="finished", generation=200, final_genome_hash=runner.genome_hash(flat))))
            self.manifest.append(dict(label=label, controller=controller, seed=seed,
                                      run=f"fixtures/{label}", sha256=hashlib.sha256(
                                          (folder / "genome_final.npy").read_bytes()).hexdigest()))
        self.argv = ["runner", "--name", "E114-fixture", "--eval-protocol", "v4", "--generations", "0",
                     "--condition-split", "development", "--eval-worlds", "2"]
        for key in ("G", "M", "T", "R", "H", "n_items"):
            self.argv += ["--cfg", f"{key}={getattr(self.cfg, key)}"]

    def snapshot(self, folder):
        return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def folder(self):
        return next((self.root / "runs").glob("*_E114-fixture"))

    def fake_evaluate(self, flat, c, n_worlds, seed, *, controller, on_condition):
        """Deterministic event fixture; uses the real NumPy summaries and contrasts."""
        self.assertEqual((c, n_worlds, seed), (self.cfg, 2, 41))
        common = dict(wkeys=np.arange(2*n_worlds, dtype=np.uint32).reshape(n_worlds, 2),
                      dkeys=np.zeros((n_worlds, 2), np.uint32), initial_grid=np.zeros((n_worlds, c.G, c.G)),
                      initial_pos=np.zeros((n_worlds, c.M, 2)),
                      agent_ids=np.broadcast_to(np.arange(c.M), (n_worlds, c.M)))
        raw, conditions = dict(common), {}
        for label in eval_v4.CONDITIONS:
            events = {key: np.zeros((n_worlds, c.T, c.M), dtype=bool) for key in eval_v3.EVENT_KEYS}
            events["food"][:, 0] = True
            events["alive_before"][:] = True
            events["alive_after"][:] = True
            metric, per_agent = eval_v4.summarize_condition(events, c.T // 2)
            condition = {label + "_" + key: value for key, value in {**events, **per_agent}.items()}
            condition.update({label + "_fitness": per_agent["agent_return"].mean(1),
                              label + "_rule": np.full(n_worlds, int(label[-1]))})
            metric.update(controller=controller, initial_rule=int(label[-1]),
                          flip_step=c.T // 2 if label.startswith("reversal") else None)
            raw.update(condition)
            conditions[label] = metric
            on_condition(label, metric, {**common, **condition})
        paired, paired_arrays = eval_v4.summarize_pairs(raw)
        raw.update(paired_arrays)
        return dict(conditions=conditions, paired=paired), raw

    def invoke(self, extra=(), *, evaluate=None, real=False, argv=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "V4_SOURCES", tuple(self.manifest)))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="b" * 40, dirty=False)))
            stack.enter_context(patch.object(runner.sys, "argv", (self.argv if argv is None else argv) + list(extra)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if not real:
                stack.enter_context(patch.object(eval_v4, "evaluate_controller", side_effect=evaluate or self.fake_evaluate))
            training = stack.enter_context(patch.object(runner, "make_evaluator"))
            direct_training = stack.enter_context(patch.object(direct, "make_evaluator"))
            runner.main()
            training.assert_not_called()
            direct_training.assert_not_called()

    def test_exact_copies_complete_manifest_raw_panels_and_distinct_table(self):
        original = self.snapshot(self.root / "fixtures")
        self.invoke()
        folder = self.folder()
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(self.snapshot(self.root / "fixtures"), original)
        self.assertEqual((summary["status"], summary["generation"], summary["completed_panels"]), ("finished", 0, 11))
        self.assertEqual((summary["eval_seed"], summary["eval_worlds"]), (41, 2))
        expected_labels = [source["label"] for source in self.manifest] + list(runner.V4_REFERENCES)
        self.assertEqual(list(summary["panels"]), expected_labels)
        self.assertEqual(len(summary["contrasts"]), 15)
        self.assertNotIn("final_learning", summary)
        self.assertNotIn("final_frozen", summary)
        self.assertFalse((folder / "genome_final.npy").exists())
        self.assertFalse((folder / "metrics.jsonl").exists())
        for metadata in config["input_genomes"]:
            source = self.root / metadata["run"]
            copied = folder / metadata["copied_genome"]
            self.assertEqual(copied.read_bytes(), (source / "genome_final.npy").read_bytes())
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), metadata["sha256"])
            self.assertEqual(hashlib.sha256((source / "config.json").read_bytes()).hexdigest(), metadata["config_sha256"])
        self.assertEqual(len(list(folder.glob("*/*.npz"))), 55)
        for label, panel in summary["panels"].items():
            self.assertEqual(panel["status"], "finished")
            self.assertEqual(set(panel["conditions"]), set(eval_v4.CONDITIONS))
            with np.load(folder / label / "paired.npz", allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["paired_individual_min_return"], [1., 1.])
            with np.load(folder / label / "stationary_r0.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["stationary_r0_food"].shape, (2, 4, 2))
                self.assertEqual(raw["stationary_r0_food"].dtype, np.dtype(bool))
                self.assertEqual(raw["stationary_r0_agent_return"].shape, (2, 2))
                self.assertIn("initial_grid", raw.files)
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("EVAL v4 paired-world diagnostics", table)
        self.assertIn("1.000000 ± 0.000000", table)
        self.assertIn("direct-rnn-v1", table)
        self.assertIn("development", table)

    def test_real_tiny_aggregate_uses_only_development_worlds(self):
        original = self.snapshot(self.root / "fixtures")
        self.invoke(real=True)
        folder = self.folder()
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_panels"], summary["eval_seed"]), ("finished", 11, 41))
        self.assertEqual(self.snapshot(self.root / "fixtures"), original)
        self.assertEqual(len(summary["contrasts"]), 15)
        for row in summary["contrasts"]:
            self.assertEqual(row["n_worlds"], 2)
            panel = summary["panels"][f"A{row['seed']}"]["paired"][row["metric"]]
            direct_panel = summary["panels"][f"direct{row['seed']}"]["paired"][row["metric"]]
            self.assertAlmostEqual(row["mean"], panel["mean"] - direct_panel["mean"])
            self.assertEqual(eval_v3.mean_se(row["per_world_difference"])["se"], row["se"])
        for label in summary["panels"]:
            with np.load(folder / label / "stationary_r0.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["wkeys"].shape, (2, 2))
                self.assertEqual(raw["stationary_r0_food"].shape, (2, 4, 2))
        with np.load(folder / "mixture/stationary_r0.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["specialist_targets"], [0, 1])

    def test_invalid_requests_create_no_folder(self):
        invalid = (("--generations", "1"), ("--start", "biology"), ("--controller", "rules-v0"),
                   ("--controller", direct.FORMAT), ("--no-plastic",), ("--quick",),
                   ("--structural-plasticity", runner.MODE), ("--resume", "unused"),
                   ("--max-train-seconds", "1"), ("--allow-dirty",), ("--seed", "41"),
                   ("--pop", "2"), ("--worlds", "2"), ("--sigma", ".2"), ("--lr", ".2"),
                   ("--checkpoint-every", "1"), ("--eval-worlds", "1"), ("--eval-worlds", "5"),
                   ("--cfg", "T=32"), ("--cfg", "T=5"), ("--cfg", "M=1"),
                   ("--cfg", "R=2"), ("--cfg", "H=9"), ("--cfg", "G=9"),
                   ("--cfg", "gain=nan"), ("--name", "../unsafe"))
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(extra)
            self.assertFalse((self.root / "runs").exists())
        final = ["runner", "--name", "E114-fixture", "--eval-protocol", "v4", "--generations", "0",
                 "--condition-split", "final", "--eval-worlds", "256"]
        for extra in (("--cfg", "G=16"), ("--eval-worlds", "128"), ("--condition-split", "development")):
            with self.subTest(final_extra=extra), self.assertRaises(SystemExit):
                self.invoke(extra, argv=final)
            self.assertFalse((self.root / "runs").exists())

    def test_invalid_last_source_rejected_before_any_artifact(self):
        folder = self.root / self.manifest[-1]["run"]
        path = folder / "genome_final.npy"
        original = path.read_bytes()
        for replacement in (np.zeros(3, np.float32), np.full(direct.genome_size(self.cfg), np.nan, np.float32),
                            np.zeros(direct.genome_size(self.cfg), np.int32)):
            with self.subTest(shape=replacement.shape, dtype=replacement.dtype):
                np.save(path, replacement)
                self.manifest[-1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                before = self.snapshot(self.root / "fixtures")
                with self.assertRaises(SystemExit):
                    self.invoke()
                self.assertEqual(before, self.snapshot(self.root / "fixtures"))
                self.assertFalse((self.root / "runs").exists())
        path.write_bytes(original)
        self.manifest[-1]["sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse((self.root / "runs").exists())

    def test_invalid_provenance_rejected_and_legacy_null_is_allowed(self):
        config_path = self.root / "fixtures/A2/config.json"
        original = config_path.read_bytes()
        for key, value in (("git", dict(commit="a" * 40, dirty=True)), ("genome_size", 154),
                           ("cfg", runner.Cfg()._asdict())):
            config = json.loads(original)
            config[key] = value
            config_path.write_text(json.dumps(config))
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.invoke()
            self.assertFalse((self.root / "runs").exists())
        config_path.write_bytes(original)
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "V4_SOURCES", self.manifest):
            loaded = runner.load_v4_sources(self.cfg)
        self.assertEqual(tuple(loaded), tuple(source["label"] for source in self.manifest))
        self.assertIsNone(json.loads(original)["args"]["structural_plasticity"])

    def test_registered_six_source_files_validate_read_only(self):
        before = {source["run"]: self.snapshot(runner.ROOT / source["run"]) for source in runner.V4_SOURCES}
        loaded = runner.load_v4_sources(runner.Cfg())
        self.assertEqual(tuple(loaded), ("A0", "A1", "A2", "direct0", "direct1", "direct2"))
        self.assertEqual(before, {source["run"]: self.snapshot(runner.ROOT / source["run"]) for source in runner.V4_SOURCES})
        self.assertEqual([len(loaded[label]["flat"]) for label in loaded], [155] * 3 + [8424] * 3)

    def test_failure_keeps_completed_panel_and_partial_next_panel(self):
        completed_bytes = {}
        calls = 0

        def fail_after_condition(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                result = self.fake_evaluate(*args, **kwargs)
                completed_bytes.update(self.snapshot(self.folder() / "A0"))
                return result
            callback = kwargs["on_condition"]

            def stopped(label, metrics, arrays):
                callback(label, metrics, arrays)
                raise RuntimeError("fixture failure after completed condition")

            return self.fake_evaluate(*args, **dict(kwargs, on_condition=stopped))

        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.invoke(evaluate=fail_after_condition)
        folder = self.folder()
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_panels"]), ("failed", 1))
        self.assertEqual(summary["panels"]["A0"]["status"], "finished")
        self.assertEqual(summary["panels"]["A1"]["status"], "failed")
        self.assertEqual(len(summary["panels"]["A1"]["conditions"]), 1)
        self.assertNotIn("contrasts", summary)
        for path, data in completed_bytes.items():
            self.assertEqual((folder / "A0" / path).read_bytes(), data)
        self.assertTrue((folder / "A0/paired.npz").exists())
        self.assertTrue((folder / "A1/stationary_r0.npz").exists())

    def test_interrupt_or_deadline_preserves_completed_condition(self):
        for failure in (KeyboardInterrupt, TimeoutError):
            with self.subTest(failure=failure):
                name = "E114-" + failure.__name__

                def interrupted(*args, **kwargs):
                    callback = kwargs["on_condition"]

                    def stop(label, metrics, arrays):
                        if failure is TimeoutError:
                            with patch.object(runner, "V4_MAX_SECONDS", 0):
                                callback(label, metrics, arrays)
                        callback(label, metrics, arrays)
                        raise KeyboardInterrupt("fixture stop")

                    return self.fake_evaluate(*args, **dict(kwargs, on_condition=stop))

                with self.assertRaises(failure):
                    self.invoke(("--name", name), evaluate=interrupted)
                folder = next((self.root / "runs").glob("*_" + name))
                summary = json.loads((folder / "summary.json").read_text())
                self.assertEqual(summary["status"], "interrupted")
                self.assertEqual(summary["panels"]["A0"]["status"], "interrupted")
                self.assertEqual(len(summary["panels"]["A0"]["conditions"]), 1)
                self.assertTrue((folder / "A0/stationary_r0.npz").exists())

    def test_nonfinite_metrics_keep_raw_evidence_without_nan_summary(self):
        def invalid(*args, **kwargs):
            callback = kwargs["on_condition"]

            def nonfinite(label, metrics, arrays):
                callback(label, dict(metrics, fitness=float("nan")), arrays)

            return self.fake_evaluate(*args, **dict(kwargs, on_condition=nonfinite))

        with self.assertRaises(SystemExit):
            self.invoke(evaluate=invalid)
        summary_text = (self.folder() / "summary.json").read_text()
        summary = json.loads(summary_text)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["panels"]["A0"]["conditions"], {})
        self.assertTrue((self.folder() / "A0/stationary_r0.npz").exists())
        self.assertNotIn("NaN", summary_text)

    def test_duplicate_condition_resume_and_repeated_attempt_cannot_overwrite(self):
        saved = {}

        def duplicate(*args, **kwargs):
            callback = kwargs["on_condition"]

            def twice(label, metrics, arrays):
                callback(label, metrics, arrays)
                saved.update(self.snapshot(self.folder() / "A0"))
                callback(label, metrics, arrays)

            return self.fake_evaluate(*args, **dict(kwargs, on_condition=twice))

        with self.assertRaises(SystemExit):
            self.invoke(evaluate=duplicate)
        folder = self.folder()
        self.assertEqual(saved, self.snapshot(folder / "A0"))
        original = self.snapshot(folder)
        for argv in (None, ["runner", "--resume", str(folder)], self.argv + ["--resume", str(folder)]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.invoke(argv=argv)
            self.assertEqual(original, self.snapshot(folder))
        self.assertEqual(len(list((self.root / "runs").iterdir())), 1)

    def test_missing_summary_and_partial_results_stay_visible(self):
        folder = self.root / "runs" / "unfinished-v4"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(dict(
            args=dict(eval_protocol="v4", condition_split="final"), eval_seed=eval_v4.EVAL_SEED, eval_worlds=256)))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("unfinished-v4 | no summary | pending", table)
        self.assertIn("undefined", table)
        self.assertIn(str(eval_v4.EVAL_SEED), table)


if __name__ == "__main__":
    unittest.main()
