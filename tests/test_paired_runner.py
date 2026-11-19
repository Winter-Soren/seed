"""Paired training recovery, immutable provenance and deferred v5 evaluation."""
import contextlib
import functools
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seed import paired
from scripts import eval_v3, eval_v4, eval_v5, run_experiment as runner, summarize


class ProcessLoss(BaseException):
    pass


class PairedRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = runner.Cfg(G=3, M=2, T=6, R=0, H=3, n_items=2)
        self.manifest = []
        for seed in range(3):
            folder = self.root / "fixtures" / f"A{seed}"
            folder.mkdir(parents=True)
            flat = np.full(155, seed * .01, np.float32)
            np.save(folder / "genome_final.npy", flat)
            config = dict(cfg=self.cfg._asdict(), genome_size=155, git=dict(commit="a" * 40, dirty=False),
                          args=dict(seed=seed, start="blank", generations=200, no_plastic=False,
                                    quick=False, allow_dirty=False, pop=128, worlds=8, sigma=.1, lr=.1))
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "summary.json").write_text(json.dumps(dict(
                status="finished", generation=200, final_genome_hash=runner.genome_hash(flat))))
            self.manifest.append(dict(label=f"A{seed}", controller="rules-v0", seed=seed,
                                      run=f"fixtures/A{seed}", sha256=hashlib.sha256(
                                          (folder / "genome_final.npy").read_bytes()).hexdigest()))
        self.source = self.root / self.manifest[0]["run"] / "genome_final.npy"
        self.now, self.crash_at, self.nonfinite_at = 0., None, None
        self.real_step = runner.es_step
        self.factory = functools.lru_cache(None)(paired.make_evaluator)
        self.common = ["--eval-protocol", runner.PAIRED_PROTOCOL, "--objective", "paired-mean",
                       "--condition-split", "development", "--start", f"file:{self.source}",
                       "--generations", "4", "--pop", "2", "--worlds", "2", "--eval-worlds", "2",
                       "--seed", "41", "--checkpoint-every", "2"]
        self.cfg_args = [value for key in ("G", "M", "T", "R", "H", "n_items")
                         for value in ("--cfg", f"{key}={getattr(self.cfg, key)}")]
        self.common += self.cfg_args

    def folder(self, name):
        return next((self.root / "runs").glob("*_" + name))

    def snapshot(self, folder):
        return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def step(self, *args):
        if args[0]["t"] == self.crash_at:
            raise ProcessLoss("fixture process loss")
        state, row = self.real_step(*args)
        self.now += 1.
        if args[0]["t"] == self.nonfinite_at:
            state = dict(state, mu=np.full_like(state["mu"], np.nan))
        return state, row

    def invoke(self, *args, diagnostic=False, evaluator=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "V4_SOURCES", tuple(self.manifest)))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="b" * 40, dirty=False)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *args]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if diagnostic:
                stack.enter_context(patch.object(eval_v5, "evaluate_controller", side_effect=evaluator or self.fixture_evaluate))
            else:
                stack.enter_context(patch.object(paired, "make_evaluator", side_effect=self.factory))
                stack.enter_context(patch.object(runner, "es_step", side_effect=self.step))
                stack.enter_context(patch.object(runner.time, "monotonic", side_effect=lambda: self.now))
                for module, function in ((runner, "test_genome"), (eval_v3, "evaluate_conditions"),
                                         (eval_v4, "evaluate_controller"), (eval_v5, "evaluate_controller")):
                    stack.enter_context(patch.object(module, function, side_effect=AssertionError("training evaluated held-out worlds")))
            runner.main()

    def fixture_evaluate(self, flat, c, worlds, seed, *, controller, on_condition):
        self.assertEqual((c, worlds, seed), (self.cfg, 2, 41))
        common = dict(wkeys=np.zeros((worlds, 2), np.uint32), dkeys=np.zeros((worlds, 2), np.uint32),
                      initial_grid=np.zeros((worlds, c.G, c.G), np.int32),
                      initial_pos=np.zeros((worlds, c.M, 2), np.int32),
                      agent_ids=np.broadcast_to(np.arange(c.M), (worlds, c.M)))
        raw, conditions = dict(common), {}
        for label in eval_v5.CONDITIONS:
            events = {key: np.zeros((worlds, c.T, c.M), bool) for key in eval_v3.EVENT_KEYS}
            events["food"][:, 0] = True
            events["alive_before"][:] = events["alive_after"][:] = True
            metric, per_agent = eval_v4.summarize_condition(events, c.T // 2)
            arrays = {label + "_" + key: value for key, value in {**events, **per_agent}.items()}
            arrays.update({label + "_fitness": per_agent["agent_return"].mean(1),
                           label + "_rule": np.full(worlds, int(label[-1]))})
            metric["controller"] = controller
            conditions[label] = metric
            raw.update(arrays)
            on_condition(label, metric, {**common, **arrays})
        estimates, arrays = eval_v4.summarize_pairs(raw)
        raw.update(arrays)
        return dict(conditions=conditions, paired=estimates), raw

    def train_bundle(self):
        for seed in range(3):
            source = self.root / self.manifest[seed]["run"] / "genome_final.npy"
            for arm in ("mean", "min"):
                self.invoke("--name", f"E115-{arm}-s{seed}", *self.common, "--objective", "paired-" + arm,
                            "--start", f"file:{source}", "--generations", "1")

    def diagnostic(self, **kwargs):
        self.invoke("--name", "E115-v5", "--eval-protocol", "v5", "--condition-split", "development",
                    "--generations", "0", "--eval-worlds", "2", *self.cfg_args, diagnostic=True, **kwargs)

    def test_real_both_objectives_split_and_flushed_tail_are_identical(self):
        initial = []
        for arm in ("mean", "min"):
            common = [*self.common, "--objective", "paired-" + arm]
            self.invoke("--name", "whole-" + arm, *common)
            whole = self.folder("whole-" + arm)
            self.invoke("--name", "split-" + arm, *common, "--max-train-seconds", "2")
            split = self.folder("split-" + arm)
            prefix = (split / "metrics.jsonl").read_bytes()
            self.assertEqual(runner.load_checkpoint(split)["t"], 2)
            self.invoke("--resume", str(split))
            self.assertTrue((split / "metrics.jsonl").read_bytes().startswith(prefix))
            self.crash_at = 3
            with self.assertRaises(ProcessLoss):
                self.invoke("--name", "crash-" + arm, *common)
            crash = self.folder("crash-" + arm)
            prefix = (crash / "metrics.jsonl").read_bytes()
            self.assertEqual(len(prefix.splitlines()), 3)
            self.crash_at = None
            self.invoke("--resume", str(crash))
            self.assertTrue((crash / "metrics.jsonl").read_bytes().startswith(prefix))
            self.assertEqual(json.loads((crash / "config.json").read_text())["resumes"][-1]["replayed_generations"], 1)
            for result in (split, crash):
                self.assertEqual((whole / "genome_final.npy").read_bytes(), (result / "genome_final.npy").read_bytes())
                for key, value in runner.load_checkpoint(whole).items():
                    np.testing.assert_array_equal(value, runner.load_checkpoint(result)[key])
            config = json.loads((whole / "config.json").read_text())
            initial.append(config["initial_optimizer"])
            self.assertEqual((whole / "genome_start.npy").read_bytes(), self.source.read_bytes())
            summary = json.loads((whole / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["training_complete"], summary["evaluation_status"]),
                             ("finished", True, "deferred"))
            self.assertNotIn("final_learning", summary)
            self.assertNotIn("evaluations", summary)
            self.assertFalse((whole / "events.npz").exists())
        self.assertEqual(initial[0], initial[1])
        self.assertEqual(initial[0]["generation"], 0)
        self.assertTrue(initial[0]["adam_m_zero"] and initial[0]["adam_v_zero"])

    def test_invalid_training_requests_and_objective_on_legacy_create_no_artifacts(self):
        for extra in (("--controller", runner.direct.FORMAT), ("--no-plastic",), ("--quick",),
                      ("--structural-plasticity", runner.MODE), ("--allow-dirty",), ("--start", "blank"),
                      ("--seed", "42"), ("--generations", "5"), ("--pop", "6"), ("--worlds", "5"),
                      ("--cfg", "H=9"), ("--cfg", "n_htypes=1"), ("--cfg", "gain=nan"),
                      ("--cfg", "gain=-1"), ("--cfg", "micro=0"),
                      ("--sigma", "nan"), ("--lr", "0"), ("--checkpoint-every", "0"),
                      ("--eval-protocol", "v1")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke("--name", "invalid", *self.common, *extra)
            self.assertFalse((self.root / "runs").exists())
        missing_objective = self.common.copy()
        index = missing_objective.index("--objective")
        del missing_objective[index:index+2]
        with self.assertRaises(SystemExit):
            self.invoke("--name", "no-objective", *missing_objective)
        self.assertFalse((self.root / "runs").exists())
        final = ["--name", "E115-mean-s0", "--eval-protocol", runner.PAIRED_PROTOCOL,
                 "--objective", "paired-mean", "--generations", "50", "--pop", "64", "--worlds", "4",
                 "--seed", "1000", "--eval-worlds", "256", "--start", f"file:{self.source}"]
        for extra in (("--generations", "49"), ("--pop", "128"), ("--worlds", "8"), ("--seed", "1003"),
                      ("--checkpoint-every", "1"), ("--name", "alternate"), ("--cfg", "T=300"),
                      ("--sigma", ".05"), ("--eval-worlds", "128")):
            with self.subTest(final_extra=extra), self.assertRaises(SystemExit):
                self.invoke(*final, *extra)
            self.assertFalse((self.root / "runs").exists())
        np.save(self.source, np.full(155, np.nan, np.float32))
        with self.assertRaises(SystemExit):
            self.invoke("--name", "nonfinite-source", *self.common)
        self.assertFalse((self.root / "runs").exists())

    def test_resume_rejects_changed_settings_source_and_optimizer_before_writes(self):
        self.invoke("--name", "resume", *self.common, "--max-train-seconds", "2")
        folder = self.folder("resume")
        before = self.snapshot(folder)
        for extra in (("--objective", "paired-min"), ("--start", "biology"), ("--gen", "1"),
                      ("--pop", "4"), ("--worlds", "1"), ("--sigma", ".2"), ("--lr", ".2"),
                      ("--seed", "42"), ("--no-plastic",), ("--controller", runner.direct.FORMAT),
                      ("--checkpoint-every", "1"), ("--eval-worlds", "4"), ("--name", "changed"),
                      ("--cfg", "G=3"), ("--condition-split", "final"), ("--quick",)):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke("--resume", str(folder), *extra)
            self.assertEqual(before, self.snapshot(folder))
        config_path = folder / "config.json"
        config = json.loads(config_path.read_text())
        config["initial_optimizer"]["adam_m_zero"] = False
        config_path.write_text(json.dumps(config))
        changed = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder))
        self.assertEqual(changed, self.snapshot(folder))
        config_path.write_bytes(before["config.json"])
        with (folder / "genome_start.npy").open("wb") as output:
            np.lib.format.write_array(output, np.load(self.source), version=(2, 0))
        changed = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder))
        self.assertEqual(changed, self.snapshot(folder))

    def test_invalid_checkpoints_and_new_steps_preserve_valid_evidence(self):
        self.invoke("--name", "state", *self.common, "--max-train-seconds", "2")
        folder = self.folder("state")
        with np.load(folder / "checkpoint.npz", allow_pickle=False) as checkpoint:
            good = {key: checkpoint[key].copy() for key in checkpoint.files}
        for field, invalid in dict(mu=good["mu"][:-1], m=np.full_like(good["m"], np.nan),
                                   v=np.full_like(good["v"], -1), key=good["key"].astype(np.int64), t=np.array(.5)).items():
            np.savez(folder / "checkpoint.npz", **dict(good, **{field: invalid}))
            before = self.snapshot(folder)
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.invoke("--resume", str(folder))
            self.assertEqual(before, self.snapshot(folder))
        self.nonfinite_at = 0
        with self.assertRaises(ValueError):
            self.invoke("--name", "invalid-step", *self.common)
        failed = self.folder("invalid-step")
        self.assertEqual(json.loads((failed / "summary.json").read_text())["status"], "failed")
        self.assertEqual(runner.load_checkpoint(failed)["t"], 0)
        self.assertEqual((failed / "metrics.jsonl").read_bytes(), b"")
        self.assertFalse((failed / "genome_final.npy").exists())

    def test_duplicate_name_and_completed_training_cannot_be_rewritten(self):
        self.invoke("--name", "done", *self.common, "--generations", "1")
        folder = self.folder("done")
        before = self.snapshot(folder)
        for args in (("--name", "done", *self.common), ("--resume", str(folder))):
            with self.assertRaises(SystemExit):
                self.invoke(*args)
            self.assertEqual(before, self.snapshot(folder))
        (folder / "genome_final.npy").unlink()
        (folder / "summary.json").unlink()
        before = self.snapshot(folder)
        with self.assertRaisesRegex(SystemExit, "final checkpoint"):
            self.invoke("--resume", str(folder))
        self.assertEqual(before, self.snapshot(folder))

    def test_v5_loads_exact_six_completed_continuations_and_records_gate(self):
        self.train_bundle()
        training = self.snapshot(self.root / "runs")
        self.diagnostic()
        folder = self.folder("E115-v5")
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual((summary["status"], summary["completed_panels"], summary["artifact_validity"]),
                         ("finished", 14, "passed"))
        self.assertEqual(tuple(summary["panels"]), eval_v5.PANELS)
        self.assertEqual(summary["eval_seed"], 41)
        self.assertEqual(len(summary["contrasts"]), 30)
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(len(list(folder.glob("*/*.npz"))), 70)
        self.assertEqual(len(config["input_genomes"]), 9)
        for metadata in config["input_genomes"]:
            self.assertEqual((folder / metadata["copied_genome"]).read_bytes(),
                             (self.root / metadata["run"] / "genome_final.npy").read_bytes())
        for relative, data in training.items():
            self.assertEqual((self.root / "runs" / relative).read_bytes(), data)
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("Paired objective continuation", table)
        self.assertIn("EVAL v5 paired objective comparison", table)
        self.assertIn("finished | FAIL", table)
        self.assertIn("deferred", table)
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder))
        self.assertEqual(before, self.snapshot(folder))

    def test_v5_rejects_missing_duplicate_incomplete_or_modified_sources_before_folder(self):
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertFalse((self.root / "runs").exists())
        self.train_bundle()
        target = self.folder("E115-min-s2")
        summary_path = target / "summary.json"
        original = summary_path.read_bytes()
        summary = json.loads(original)
        summary["status"] = "interrupted"
        summary_path.write_text(json.dumps(summary))
        before = self.snapshot(self.root / "runs")
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertEqual(before, self.snapshot(self.root / "runs"))
        summary_path.write_bytes(original)
        config_path = target / "config.json"
        original_config = config_path.read_bytes()
        config = json.loads(original_config)
        config["initial_optimizer"]["adam_v_zero"] = False
        config_path.write_text(json.dumps(config))
        before = self.snapshot(self.root / "runs")
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertEqual(before, self.snapshot(self.root / "runs"))
        config_path.write_bytes(original_config)
        duplicate = self.root / "runs" / "duplicate_E115-mean-s0"
        duplicate.mkdir()
        before = self.snapshot(self.root / "runs")
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertEqual(before, self.snapshot(self.root / "runs"))
        self.assertFalse(list((self.root / "runs").glob("*_E115-v5")))

    def test_v5_failure_keeps_completed_panel_without_gate_and_rejects_retry(self):
        self.train_bundle()
        calls = 0
        completed = {}

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                result = self.fixture_evaluate(*args, **kwargs)
                completed.update(self.snapshot(self.folder("E115-v5") / "start0"))
                return result
            callback = kwargs["on_condition"]

            def stop(label, metric, arrays):
                callback(label, metric, arrays)
                raise RuntimeError("fixture diagnostic failure")

            return self.fixture_evaluate(*args, **dict(kwargs, on_condition=stop))

        with self.assertRaises(RuntimeError):
            self.diagnostic(evaluator=fail)
        folder = self.folder("E115-v5")
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_panels"]), ("failed", 1))
        self.assertEqual(summary["panels"]["start0"]["status"], "finished")
        self.assertEqual(summary["panels"]["mean0"]["status"], "failed")
        self.assertNotIn("gate", summary)
        self.assertNotIn("artifact_validity", summary)
        for relative, data in completed.items():
            self.assertEqual((folder / "start0" / relative).read_bytes(), data)
        self.assertTrue((folder / "mean0/stationary_r0.npz").exists())
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertEqual(before, self.snapshot(folder))


if __name__ == "__main__":
    unittest.main()
