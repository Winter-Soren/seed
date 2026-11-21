"""Runner-only fixtures: immutable segments, exact tied-state continuation and evaluation lock."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_reversal_evolution as entry, run_experiment as runner, summarize


class ReversalEvolutionRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="reversal-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_calls, self.eval_calls = [], []
        self.fail_at, self.advance_seconds, self.now = None, 0., 0.
        self.corrupt_tie, self.corrupt_keys, self.bad_candidates = False, False, False
        self.aggregate_calls = []

    @staticmethod
    def initial(controller, master, c):
        size = 155 if controller == "rules-v0" else 8
        return dict(mu=np.zeros(size, np.float32), m=np.zeros(size, np.float32), v=np.zeros(size, np.float32),
                    t=0, key=np.asarray([master, 0], np.uint32))

    def sampling(self, state, pop, pairs, sigma):
        master, counter = (int(value) for value in state["key"])
        offset = (counter % 2000000) * 1000
        worlds = np.asarray([[master, offset + i] for i in range(pairs)], np.uint32)
        births = np.asarray([[master, offset + 10 + i] for i in range(pop // 2 * pairs)], np.uint32).reshape(pop // 2, pairs, 2)
        return dict(noise_key=np.asarray([master, offset + 800], np.uint32), half_eps_sha256=np.zeros(32, np.uint8),
            candidate_sha256=np.zeros((pop, 32), np.uint8), world_keys=worlds,
            development_keys=np.concatenate((births, births)), next_key=np.asarray([master, (counter + 1) % 2**32], np.uint32),
            assignment_key=np.asarray([master, offset + 900], np.uint32),
            permutation=np.arange(4, dtype=np.int32), flip_steps=np.asarray([100, 125, 150, 175], np.int32))

    def evaluator(self, controller, c):
        def evaluate(candidates, world_keys, birth_keys):
            scores = np.zeros((128, 4), np.float32)
            if int(world_keys[0, 1]) != 1000:
                scores[:] = np.arange(128, dtype=np.float32)[:, None]
            return dict(fitness=scores, counts=np.zeros((128, 4, 2, 2), np.int32),
                        survived=np.ones((128, 4), np.float32))
        return evaluate

    def step(self, state, evaluate, pop, pairs, sigma, lr, l2):
        attempt = int(state["key"][1]) + 1
        if attempt == self.fail_at:
            raise RuntimeError("deliberate incomplete attempted round")
        sampling = self.sampling(state, pop, pairs, sigma)
        worlds = sampling["world_keys"] + np.uint32(1) if self.corrupt_keys else sampling["world_keys"]
        result = evaluate(np.broadcast_to(state["mu"], (pop, state["mu"].size)), worlds, sampling["development_keys"])
        informative = bool(np.any(result["fitness"] != result["fitness"][0]))
        updated = {key: np.asarray(value).copy() for key, value in state.items()}
        if informative:
            for field, change in (("mu", .01), ("m", .1), ("v", .001)):
                updated[field] += np.float32(change)
            updated["t"] = int(state["t"]) + 1
        elif self.corrupt_tie:
            updated["mu"] += np.float32(.01)
        updated["key"] = sampling["next_key"]
        self.train_calls.append((int(state["key"][0]), attempt, informative))
        self.now += self.advance_seconds
        return updated, dict(fit_mean=float(result["fitness"].mean()), informative=informative)

    @staticmethod
    def schedule(master, count):
        return dict(key=np.asarray([master, 10000], np.uint32), permutation=np.arange(count, dtype=np.int32),
                    flip_steps=np.repeat(np.asarray([100, 125, 150, 175], np.int32), count // 4))

    def evaluate_world(self, plastic, recurrent, keys, index, flip, c):
        self.eval_calls.append((int(keys[0, 0]), index))
        return dict(keys=keys.copy(), base_index=np.asarray(index), flip_step=np.asarray(flip),
            plastic_genome_sha256=entry.engine.array_hash(plastic), direct_genome_sha256=entry.engine.array_hash(recurrent),
            score=np.asarray(.75))

    def aggregate(self, panels):
        self.aggregate_calls.append(panels)
        self.assertEqual([panel["seed_index"] for panel in panels], [0, 1, 2])
        return dict(passed=True, passing_seeds=3, all_seed_survival_passed=True)

    def contexts(self, stack):
        stack.enter_context(patch.object(runner, "ROOT", self.root))
        stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=False)))
        stack.enter_context(patch.object(entry, "TOTAL_ATTEMPTS", 4))
        stack.enter_context(patch.object(entry, "SEGMENT_ATTEMPTS", 2))
        stack.enter_context(patch.object(entry, "FINAL_WORLDS", 4))
        stack.enter_context(patch.object(entry.direct, "genome_size", return_value=8))
        stack.enter_context(patch.object(entry, "_provenance", return_value=dict(runtime={"test": True}, code={}, prng={})))
        stack.enter_context(patch.object(entry.engine, "initialize_state", side_effect=self.initial))
        stack.enter_context(patch.object(entry.engine, "sampling", side_effect=self.sampling))
        stack.enter_context(patch.object(entry.engine, "make_evaluator", side_effect=self.evaluator))
        stack.enter_context(patch.object(entry.engine, "es_step", side_effect=self.step))
        stack.enter_context(patch.object(entry.engine, "pack_evidence", side_effect=lambda result: result))
        stack.enter_context(patch.object(entry.engine, "validate_training_evidence"))
        stack.enter_context(patch.object(entry.engine, "candidate_hashes", side_effect=lambda x:
                                        np.full((128, 32), int(self.bad_candidates), np.uint8)))
        stack.enter_context(patch.object(entry.engine, "average_centered_ranks", side_effect=lambda x: np.zeros(128, np.float32)))
        stack.enter_context(patch.object(entry.engine, "evaluation_schedule", side_effect=self.schedule))
        stack.enter_context(patch.object(entry.engine, "evaluation_keys", side_effect=lambda master, i:
                                        np.asarray([[master, i * 2], [master, i * 2 + 1]], np.uint32)))
        stack.enter_context(patch.object(entry.engine, "evaluate_world", side_effect=self.evaluate_world))
        stack.enter_context(patch.object(entry.engine, "summary_input", side_effect=lambda row: row))
        stack.enter_context(patch.object(entry.engine, "summarize", side_effect=lambda rows:
                                        dict(protocol=entry.PROTOCOL, base_worlds=len(rows))))
        stack.enter_context(patch.object(entry.engine, "aggregate", side_effect=self.aggregate))
        stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def invoke(self, *extra, architecture="plastic", index=0, part=1, resume=None, evaluation=False, development=False):
        if evaluation:
            args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "final", "--name", f"E126-eval-s{index}",
                    "--seed", str(2026120501 + index), "--generations", "0", "--eval-worlds", "4", "--start", "blank"]
        else:
            name = f"E126-benchmark-{architecture}" if development else f"E126-{architecture}-s{index}-part{part:03d}"
            args = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development" if development else "final",
                    "--name", name, "--controller", entry.CONTROLLERS[architecture], "--seed",
                    str(2026120400 if development else 2026120401 + index), "--generations", "3" if development else "4",
                    "--eval-worlds", "4", "--start", "blank", "--pop", "128", "--worlds", "4", "--sigma", "0.1",
                    "--lr", "0.1", "--checkpoint-every", "10", "--max-train-seconds", "780" if development else "3000"]
            if resume is not None:
                args += ["--resume", str(resume)]
        with contextlib.ExitStack() as stack:
            self.contexts(stack)
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *args, *extra]))
            runner.main()

    def folder(self, architecture="plastic", index=0, part=1, *, evaluation=False, development=False):
        name = (f"E126-eval-s{index}" if evaluation else f"E126-benchmark-{architecture}" if development
                else f"E126-{architecture}-s{index}-part{part:03d}")
        return next((self.root / "runs").glob("*_" + name))

    @staticmethod
    def read(folder):
        return json.loads((folder / "summary.json").read_text())

    @staticmethod
    def snapshot(folder):
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def complete_targets(self):
        for index in range(3):
            for architecture in entry.CONTROLLERS:
                self.invoke(architecture=architecture, index=index)
                self.invoke(architecture=architecture, index=index, part=2, resume=self.folder(architecture, index))

    def comparison(self):
        with contextlib.ExitStack() as stack:
            self.contexts(stack)
            return entry.build_comparison(self.root)

    def test_split_tied_round_preserves_moments_and_exact_continuation(self):
        self.invoke()
        first = self.folder()
        result = self.read(first)
        self.assertEqual((result["status"], result["attempted_rounds"], result["optimizer_updates"]), ("segment_finished", 2, 1))
        before = self.snapshot(first)
        with np.load(first / "samples/attempt_0001.npz") as a, np.load(first / "samples/attempt_0002.npz") as b:
            for name in ("next_mu", "next_m", "next_v", "next_t"):
                np.testing.assert_array_equal(a[name], b[name])
            self.assertFalse(np.array_equal(a["next_key"], b["next_key"]))
            self.assertIn("noise_key", b)
            self.assertNotIn("half_eps", b)
        self.invoke(part=2, resume=first)
        final = self.read(self.folder(part=2))
        self.assertEqual((final["status"], final["attempted_rounds"], final["optimizer_updates"]), ("target_finished", 4, 3))
        self.assertEqual(before, self.snapshot(first))
        self.assertFalse(self.eval_calls)
        with np.load(self.folder(part=2) / final["terminal_checkpoint"]["path"]) as state:
            np.testing.assert_array_equal(state["key"], np.asarray([2026120401, 4], np.uint32))
            np.testing.assert_allclose(state["mu"], .03)

    def test_failed_round_resumes_in_new_folder_without_rewriting_prefix(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.invoke()
        first = self.folder()
        self.assertEqual(self.read(first)["attempted_rounds"], 1)
        before = self.snapshot(first)
        self.fail_at = None
        self.invoke(part=2, resume=first)
        self.assertEqual(self.read(self.folder(part=2))["attempted_rounds"], 3)
        self.assertEqual(before, self.snapshot(first))

    def test_partial_terminal_journal_is_retained_and_not_counted_twice(self):
        self.invoke()
        first = self.folder()
        with (first / "metrics.jsonl").open("ab") as stream:
            stream.write(b'{"attempt":3')
        before = self.snapshot(first)
        self.invoke(part=2, resume=first)
        result = self.read(self.folder(part=2))
        self.assertEqual(result["attempted_rounds"], 4)
        self.assertEqual(result["parent"]["ignored_unflushed_journal_bytes"], len(b'{"attempt":3'))
        self.assertEqual(before, self.snapshot(first))

    def test_duplicate_names_generic_resume_and_branching_are_rejected(self):
        self.invoke()
        first = self.folder()
        before = self.snapshot(first)
        with self.assertRaises(SystemExit):
            self.invoke()
        self.invoke(part=2, resume=first)
        with self.assertRaises(SystemExit):
            self.invoke(part=3, resume=first)
        with contextlib.ExitStack() as stack:
            self.contexts(stack)
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", "--resume", str(first)]))
            with self.assertRaises(SystemExit):
                runner.main()
        self.assertEqual(before, self.snapshot(first))

    def test_corrupted_record_hash_blocks_continuation(self):
        self.invoke()
        path = self.folder() / "samples/attempt_0002.npz"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaises(SystemExit):
            self.invoke(part=2, resume=self.folder())
        self.assertFalse(list((self.root / "runs").glob("*_E126-plastic-s0-part002")))

    def test_all_tie_mutation_fails_with_previous_state_preserved(self):
        self.corrupt_tie = True
        with self.assertRaises(SystemExit):
            self.invoke()
        result = self.read(self.folder())
        self.assertEqual((result["status"], result["attempted_rounds"]), ("failed", 1))

    def test_actual_world_draw_mismatch_fails_before_a_completed_round(self):
        self.corrupt_keys = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.read(self.folder())["attempted_rounds"], 0)

    def test_actual_candidate_digest_mismatch_blocks_the_round(self):
        self.bad_candidates = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.read(self.folder())["attempted_rounds"], 0)

    def test_scientific_overrides_are_rejected_before_creation(self):
        for extra in (("--pop", "16"), ("--worlds", "8"), ("--sigma", ".05"), ("--start", "biology"),
                      ("--generations", "8"), ("--allow-dirty",), ("--no-plastic",), ("--cfg", "M=4")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke(*extra)
        self.assertFalse((self.root / "runs").exists())

    def test_soft_stop_saves_exact_nonperiodic_terminal_checkpoint(self):
        self.advance_seconds = 2000.
        self.invoke()
        result = self.read(self.folder())
        self.assertEqual((result["status"], result["attempted_rounds"]), ("segment_finished", 1))
        self.assertEqual(result["terminal_checkpoint"]["path"], "checkpoints/attempt_0001.npz")

    def test_hard_limit_keeps_complete_prefix_and_scientific_gate_undecided(self):
        self.advance_seconds = 3601.
        with self.assertRaises(TimeoutError):
            self.invoke()
        result = self.read(self.folder())
        self.assertEqual((result["status"], result["attempted_rounds"]), ("interrupted", 1))
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "undecided")

    def test_benchmarks_use_separate_development_streams_without_heldout(self):
        for architecture in entry.CONTROLLERS:
            self.invoke(architecture=architecture, development=True)
        files = [self.folder(architecture, development=True) / "checkpoints/start.npz" for architecture in entry.CONTROLLERS]
        with np.load(files[0]) as p, np.load(files[1]) as d:
            self.assertFalse(np.array_equal(p["key"], d["key"]))
        self.assertFalse(self.eval_calls)
        for architecture in entry.CONTROLLERS:
            result = self.read(self.folder(architecture, development=True))
            self.assertEqual(result["attempted_rounds"], 3)
            self.assertFalse(result["scientific_result"])
            self.assertEqual(result["gate_status"], "undecided")

    def test_evaluation_is_locked_until_all_six_complete_targets_exist(self):
        with self.assertRaises(SystemExit):
            self.invoke(evaluation=True)
        self.invoke()
        self.invoke(part=2, resume=self.folder())
        with self.assertRaises(SystemExit):
            self.invoke(evaluation=True)
        self.assertFalse(self.eval_calls)
        self.assertFalse(list((self.root / "runs").glob("*_E126-eval-*")))

    def test_final_pairing_and_aggregate_wait_for_all_three_complete_panels(self):
        self.complete_targets()
        before = {p.name: self.snapshot(p) for p in (self.root / "runs").iterdir()}
        for index in range(3):
            self.invoke(evaluation=True, index=index)
            decision = self.comparison()
            self.assertEqual(decision["status"], "complete" if index == 2 else "undecided")
        self.assertEqual(len(self.aggregate_calls), 1)
        self.assertEqual(len(self.eval_calls), 12)
        for name, contents in before.items():
            self.assertEqual(contents, self.snapshot(self.root / "runs" / name))
        for index in range(3):
            result = self.read(self.folder(index=index, evaluation=True))
            self.assertEqual(len(result["sources"]), 6)
            self.assertFalse(result["scientific_result"])
            self.assertEqual(result["gate_status"], "undecided")

    def test_unsealed_complete_journal_blocks_eval_and_seals_without_new_evolution(self):
        self.complete_targets()
        tip = self.folder("direct", 2, 2)
        (tip / "summary.json").rename(tip / "unsealed-summary.fixture.json")
        before, calls = self.snapshot(tip), len(self.train_calls)
        with self.assertRaises(SystemExit):
            self.invoke(evaluation=True)
        self.assertFalse(self.eval_calls)
        self.invoke(architecture="direct", index=2, part=3, resume=tip)
        self.assertEqual(calls, len(self.train_calls))
        sealed = self.folder("direct", 2, 3)
        self.assertTrue(json.loads((sealed / "config.json").read_text())["sealing_recovery"])
        self.assertEqual((self.read(sealed)["attempted_rounds"], self.read(sealed)["completed_segment_attempts"]), (4, 0))
        self.assertEqual(before, self.snapshot(tip))
        self.invoke(evaluation=True)
        self.assertEqual(len(self.eval_calls), 4)

    def test_corrupt_terminal_seal_blocks_final_evaluation(self):
        self.complete_targets()
        tip = self.folder("direct", 2, 2)
        path = tip / self.read(tip)["terminal_genome"]["path"]
        path.write_bytes(path.read_bytes() + b"corrupt seal")
        with self.assertRaises(SystemExit):
            self.invoke(evaluation=True)
        self.assertFalse(self.eval_calls)

    def test_partial_results_table_cannot_claim_the_rung(self):
        self.invoke()
        with contextlib.ExitStack() as stack:
            self.contexts(stack)
            stack.enter_context(patch.object(summarize, "ROOT", self.root))
            summarize.main()
        text = (self.root / "results/RESULTS.md").read_text()
        self.assertIn("reversal-evolution-v1", text)
        self.assertIn("Complete three-seed rung decision: **undecided**", text)
        self.assertFalse((self.root / "results/diagnostics/E126-comparison.json").exists())

    def test_summary_cli_resolves_e126_import_without_loading_backend_for_empty_report(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "summarize.py").write_bytes(Path(summarize.__file__).read_bytes())
        backend = scripts / "run_reversal_evolution.py"
        backend.write_text("raise AssertionError('empty report must not import the E126 backend')\n")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "scripts/summarize.py"]
        empty = subprocess.run(command, cwd=self.root, env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(empty.returncode, 0, empty.stderr)
        backend.write_text("def build_comparison(root):\n"
                           "    (root / 'e126-builder-called').write_text('yes')\n"
                           "    return {'status': 'undecided'}\n")
        run = self.root / "runs" / "fixture_E126-plastic-s0-part001"
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(dict(stage="training", args=dict(
            eval_protocol=entry.PROTOCOL, condition_split="final"))))
        report = subprocess.run(command, cwd=self.root, env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertEqual((self.root / "e126-builder-called").read_text(), "yes")
        self.assertIn("Complete three-seed rung decision: **undecided**",
                      (self.root / "results/RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
