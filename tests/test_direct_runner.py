"""Tiny development-only direct ES, tracked diagnostics and recovery invariants."""
import contextlib
import functools
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from seed import direct
from scripts import eval_v3, run_experiment as runner, summarize


class HardStop(BaseException):
    """Model process loss, which cannot execute the runner's exception cleanup."""


class DirectRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 0.
        self.crash_at = None
        self.nonfinite_at = None
        self.real_step = runner.es_step
        self.factory = functools.lru_cache(None)(direct.make_evaluator)
        self.common = ["--controller", direct.FORMAT, "--eval-protocol", "direct-v1",
                       "--condition-split", "development", "--no-plastic", "--start", "blank",
                       "--generations", "4", "--pop", "2", "--worlds", "2", "--eval-worlds", "2",
                       "--seed", "41", "--checkpoint-every", "2", "--cfg", "G=3", "--cfg", "M=1",
                       "--cfg", "T=6", "--cfg", "R=0", "--cfg", "H=3", "--cfg", "n_items=2"]

    def step(self, *args):
        if args[0]["t"] == self.crash_at:
            raise HardStop("simulated process loss")
        state, row = self.real_step(*args)
        self.now += 2.
        if args[0]["t"] == self.nonfinite_at:
            state = dict(state, mu=np.full_like(state["mu"], np.nan))
        return state, row

    def invoke(self, *args):
        with (patch.object(runner, "ROOT", self.root),
              patch.object(runner, "git_info", return_value=dict(commit="fixture", dirty=False)),
              patch.object(direct, "make_evaluator", side_effect=self.factory),
              patch.object(runner, "es_step", side_effect=self.step),
              patch.object(runner.time, "monotonic", side_effect=lambda: self.now),
              patch.object(runner.sys, "argv", ["runner", *args]),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            runner.main()

    def folder(self, name):
        return next((self.root / "runs").glob("*_" + name))

    def snapshot(self, directory):
        return {p.name: p.read_bytes() for p in directory.iterdir()}

    def test_real_es_split_and_flushed_tail_replay_are_identical(self):
        self.invoke("--name", "whole", *self.common)
        self.invoke("--name", "split", *self.common, "--max-train-seconds", "4")
        split = self.folder("split")
        self.assertEqual(runner.load_checkpoint(split)["t"], 2)
        interrupted = json.loads((split / "summary.json").read_text())
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertNotIn("evaluations", interrupted)
        self.assertFalse((split / "events.npz").exists())
        self.invoke("--resume", str(split))

        self.crash_at = 3
        with self.assertRaises(HardStop):
            self.invoke("--name", "crash", *self.common)
        crash = self.folder("crash")
        self.assertEqual(runner.load_checkpoint(crash)["t"], 2)
        prefix = (crash / "metrics.jsonl").read_bytes()
        self.assertEqual(len(prefix.splitlines()), 3)
        self.crash_at = None
        self.invoke("--resume", str(crash))
        self.assertTrue((crash / "metrics.jsonl").read_bytes().startswith(prefix))
        self.assertEqual(json.loads((crash / "config.json").read_text())["resumes"][-1]["replayed_generations"], 1)

        whole = self.folder("whole")
        rows = lambda folder: [{k: v for k, v in json.loads(line).items() if k != "seconds"}
                               for line in (folder / "metrics.jsonl").read_text().splitlines()]
        for folder in (split, crash):
            for key, value in runner.load_checkpoint(whole).items():
                np.testing.assert_array_equal(value, runner.load_checkpoint(folder)[key], err_msg=key)
            self.assertEqual(rows(whole), rows(folder))
            self.assertEqual((whole / "genome_final.npy").read_bytes(), (folder / "genome_final.npy").read_bytes())

    def test_real_final_summary_uses_direct_labels_raw_events_and_development_seed(self):
        self.invoke("--name", "diagnostic", *self.common, "--generations", "0")
        folder = self.folder("diagnostic")
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual((config["controller"], summary["controller"]), (direct.FORMAT, direct.FORMAT))
        self.assertEqual((summary["eval_seed"], summary["eval_worlds"]), (41, 2))
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["final_direct"], summary["evaluations"]["stationary"]["genome"])
        self.assertNotIn("final_learning", summary)
        self.assertNotIn("final_frozen", summary)
        self.assertFalse(config["encoding"]["lifetime_weight_updates"])
        self.assertIsNone(config["encoding"]["weight_bounds"])
        self.assertEqual(summary["initial_weights"], summary["final_weights"])
        c = runner.Cfg(**config["cfg"])
        _, key = jax.random.split(jax.random.PRNGKey(41))
        expected = np.asarray(direct.init_genome(c, key))
        np.testing.assert_array_equal(np.load(folder / "genome_start.npy"), expected)
        self.assertEqual(config["start_genome_hash"], runner.genome_hash(expected))
        self.assertEqual(config["genome_size"], direct.genome_size(c))
        with np.load(folder / "events.npz", allow_pickle=False) as raw:
            for condition in ("stationary", "reversal"):
                estimate = eval_v3.mean_se(raw[f"{condition}_genome_fitness"])
                metric = summary["evaluations"][condition]["genome"]
                self.assertEqual((metric["fitness"], metric["fitness_se"]), (estimate["mean"], estimate["se"]))
                self.assertEqual(raw[f"{condition}_genome_food"].shape, (2, 6, 1))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("Direct recurrent baseline", table)
        self.assertIn("direct-rnn-v1 | development | 0/0 | 41 | stationary", table)

    def test_invalid_requests_fail_before_creating_artifacts(self):
        final = ["--controller", direct.FORMAT, "--eval-protocol", "direct-v1", "--no-plastic", "--eval-worlds", "256"]
        for extra in (("--controller", "rules-v0"), ("--eval-protocol", "v1"), ("--start", "biology"),
                      ("--generations", "199"), ("--eval-worlds", "128"), ("--seed", "3"),
                      ("--pop", "64"), ("--worlds", "4"), ("--sigma", ".05"), ("--lr", ".02"),
                      ("--cfg", "T=300"), ("--quick",), ("--structural-plasticity", runner.MODE)):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(*final, *extra)
            self.assertFalse((self.root / "runs").exists())
        with self.assertRaises(SystemExit):
            self.invoke("--controller", direct.FORMAT, "--eval-protocol", "direct-v1", "--eval-worlds", "256")
        for extra in (("--cfg", "T=32"), ("--generations", "5"), ("--pop", "10"),
                      ("--eval-worlds", "1"), ("--cfg", "e0=nan"), ("--sigma", "nan")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(*self.common, *extra)
            self.assertFalse((self.root / "runs").exists())

    def test_saved_v3_cli_does_not_infer_direct_format_from_file_size(self):
        with self.assertRaises(SystemExit):
            self.invoke("--eval-protocol", "v3", "--controller", direct.FORMAT, "--generations", "0",
                        "--start", "file:unused.npy", "--eval-worlds", "256", "--no-plastic")
        self.assertFalse((self.root / "runs").exists())

    def test_resume_cannot_change_controller_protocol_split_or_configuration(self):
        self.invoke("--name", "resume", *self.common, "--max-train-seconds", "2")
        folder = self.folder("resume")
        before = self.snapshot(folder)
        for extra in (("--controller", "rules-v0"), ("--eval-protocol", "v1"),
                      ("--condition-split", "final"), ("--cfg", "G=3"), ("--quick",)):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke("--resume", str(folder), *extra)
            self.assertEqual(self.snapshot(folder), before)

    def test_bad_direct_checkpoints_are_rejected_before_any_write(self):
        self.invoke("--name", "invalid", *self.common, "--max-train-seconds", "2")
        folder = self.folder("invalid")
        with np.load(folder / "checkpoint.npz", allow_pickle=False) as checkpoint:
            good = {key: checkpoint[key].copy() for key in checkpoint.files}
        corruptions = dict(mu=good["mu"][:-1], m=np.full_like(good["m"], np.nan),
                           v=np.full_like(good["v"], -1), key=good["key"].astype(np.int64),
                           t=np.array(.5))
        for field, corrupt in corruptions.items():
            np.savez(folder / "checkpoint.npz", **dict(good, **{field: corrupt}))
            before = self.snapshot(folder)
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.invoke("--resume", str(folder))
            self.assertEqual(self.snapshot(folder), before)

    def test_finished_direct_run_is_immutable_on_resume(self):
        self.invoke("--name", "finished", *self.common, "--generations", "0")
        folder = self.folder("finished")
        before = self.snapshot(folder)
        with self.assertRaisesRegex(SystemExit, "finished"):
            self.invoke("--resume", str(folder))
        self.assertEqual(self.snapshot(folder), before)

    def test_existing_events_are_preserved_with_unfinished_or_missing_summary(self):
        self.invoke("--name", "events", *self.common, "--generations", "0")
        folder = self.folder("events")
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertTrue((folder / "events.npz").exists())
        for status in ("interrupted", "failed", None):
            if status is None:
                summary_path.unlink()
            else:
                summary_path.write_text(json.dumps(dict(summary, status=status)))
            before = self.snapshot(folder)
            with self.subTest(status=status), self.assertRaisesRegex(SystemExit, "evaluated"):
                self.invoke("--resume", str(folder))
            self.assertEqual(self.snapshot(folder), before)

    def test_encoding_metadata_and_start_bytes_are_checked_before_resume(self):
        self.invoke("--name", "metadata", *self.common, "--max-train-seconds", "2")
        folder = self.folder("metadata")
        config = json.loads((folder / "config.json").read_text())
        config["encoding"]["weight_bounds"] = [-3, 3]
        (folder / "config.json").write_text(json.dumps(config))
        before = self.snapshot(folder)
        with self.assertRaisesRegex(SystemExit, "metadata"):
            self.invoke("--resume", str(folder))
        self.assertEqual(self.snapshot(folder), before)
        config["encoding"]["weight_bounds"] = None
        (folder / "config.json").write_text(json.dumps(config))
        start = np.load(folder / "genome_start.npy")
        start[0] += 1
        np.save(folder / "genome_start.npy", start)
        before = self.snapshot(folder)
        with self.assertRaisesRegex(SystemExit, "starting genome"):
            self.invoke("--resume", str(folder))
        self.assertEqual(self.snapshot(folder), before)

    def test_invalid_new_step_preserves_last_valid_state_and_records_failure(self):
        self.nonfinite_at = 0
        with self.assertRaisesRegex(ValueError, "finite"):
            self.invoke("--name", "nonfinite", *self.common)
        folder = self.folder("nonfinite")
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["generation"]), ("failed", 0))
        self.assertTrue(np.isfinite(runner.load_checkpoint(folder)["mu"]).all())
        self.assertEqual((folder / "metrics.jsonl").read_text(), "")
        self.assertFalse((folder / "genome_final.npy").exists())

    def test_config_only_direct_run_remains_visible_in_results(self):
        folder = self.root / "runs" / "unfinished"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(dict(
            args=dict(eval_protocol="direct-v1", controller=direct.FORMAT, condition_split="final",
                      generations=200, seed=0, eval_worlds=256), controller=direct.FORMAT,
            eval_seed=eval_v3.EVAL_SEED, git=dict(commit="fixture", dirty=False))))
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            summarize.main()
        table = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("unfinished | no summary | direct-rnn-v1", table)
        self.assertIn("pending", table)


if __name__ == "__main__":
    unittest.main()
