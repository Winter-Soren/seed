"""E127 source isolation, immutable paired evidence and independently audited replication."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_encounter_diagnostic as entry, run_experiment as runner, summarize


class EncounterDiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="encounter-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            path = self.root / source["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes((runner.ROOT / source["path"]).read_bytes())
        self.common = ["--eval-protocol", entry.PROTOCOL, "--condition-split", "development",
                       "--name", "E127-benchmark", "--seed", "2026120600", "--generations", "0",
                       "--eval-worlds", "4", "--start", "file:" + entry.SOURCES[0]["path"]]
        self.calls, self.projected = [], []
        self.fail_at, self.reuse, self.bad_source, self.nonfinite, self.expire = None, False, False, False, False
        self.now = 0.

    @staticmethod
    def assignments(master, count):
        return dict(key=np.asarray([master, 1000000], np.uint32), permutation=np.arange(count, dtype=np.int32),
                    pre_contacts=np.repeat(np.asarray([8, 16], np.int32), count // 2))

    def keys(self, master, index):
        index = 0 if self.reuse else index
        return np.asarray([[master, index * 6 + role] for role in range(6)], np.uint32)

    def evaluate(self, plastic, direct, keys, index, pre_contacts, *, c):
        if index == self.fail_at:
            raise RuntimeError("deliberate diagnostic interruption")
        if self.expire:
            self.now = 181.
        self.calls.append((int(keys[0, 0]), index, pre_contacts))
        return dict(keys=keys.copy(), base_index=np.asarray(index), pre_contacts=np.asarray(pre_contacts),
            initial_rules=np.asarray([0, 1]), plastic_genome_sha256=entry._array_sha(plastic),
            direct_genome_sha256=np.zeros(32, np.uint8) if self.bad_source else entry._array_sha(direct),
            outcome=np.asarray(np.nan if self.nonfinite else .5), large_trace=np.zeros((2, 32, 50), np.float32))

    def projection(self, row):
        compact = {key: row[key] for key in ("keys", "base_index", "outcome")}
        self.projected.append(compact)
        return compact

    @staticmethod
    def measurements(rows):
        master = int(rows[0]["keys"][0, 0])
        index, n = max(0, master - 2026120601), len(rows)
        metric = dict(mean=.25, se=.02, n=n)
        architecture = lambda local, transfer: dict(local_pass=local, transfer_pass=transfer,
            local={name: metric for name in ("pre_old", "post_target", "reversal_minus_stationary")},
            transfer={name: metric for name in ("reversal_minus_stationary", "food_minus_poison")})
        return dict(families=n, initial_rule_replays=2*n,
            arms={arm: dict(local=[], grid={name: metric for name in
                ("first_food", "first_poison", "no_contact", "food_minus_poison", "latency", "net_return", "survival")})
                  for arm in entry.ARMS},
            architectures=dict(plastic=architecture(index in (0, 2), index in (1, 2)),
                               direct=architecture(index in (0, 1), index in (0, 1))),
            hebbian=dict(hebbian_local_pass=True, local_difference=metric, transfer_difference=metric))

    def invoke(self, *extra, dirty=False, common=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a" * 40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if common else []), *extra]))
            stack.enter_context(patch.object(entry, "_runtime", return_value=dict(test="mock backend")))
            stack.enter_context(patch.object(entry, "_source_code", return_value={"fixture": "b" * 64}))
            stack.enter_context(patch.object(entry.time, "monotonic", side_effect=lambda: self.now))
            for name, function in (("assignments_manifest", self.assignments), ("evaluation_keys", self.keys),
                                   ("evaluate_world", self.evaluate), ("summary_input", self.projection),
                                   ("summarize", self.measurements)):
                stack.enter_context(patch.object(entry.engine, name, side_effect=function))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            runner.main()

    @staticmethod
    def final_args(index):
        return ["--condition-split", "final", "--name", f"E127-A{index}", "--seed", str(2026120601 + index),
                "--eval-worlds", "128", "--start", "file:" + entry.SOURCES[index]["path"]]

    def folder(self, name="E127-benchmark"):
        return next((self.root / "runs").glob("*_" + name))

    def result(self, name="E127-benchmark"):
        return json.loads((self.folder(name) / "summary.json").read_text())

    @staticmethod
    def snapshot(folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def finish_panels(self):
        self.failed_a0()
        for index in range(3):
            self.invoke(*self.final_args(index), "--name", entry.FINAL_NAMES[index])

    def pin_predecessor(self, name, folder):
        config = json.loads((folder / "config.json").read_text())
        pin = dict(run=folder.relative_to(self.root).as_posix(), commit=config["git"]["commit"],
                   inventory_sha256=entry._inventory_sha(entry._file_inventory(folder)))
        update = patch.dict(entry.RECOVERY_PREDECESSORS, {name: pin})
        update.start()
        self.addCleanup(update.stop)
        return pin

    def failed_a0(self):
        self.fail_at = 11
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke(*self.final_args(0))
        self.fail_at = None
        folder = self.folder("E127-A0")
        self.pin_predecessor(entry.A0_RECOVERY, folder)
        return folder

    def comparison(self):
        with patch.object(entry.engine, "assignments_manifest", side_effect=self.assignments), \
                patch.object(entry.engine, "evaluation_keys", side_effect=self.keys):
            return entry.build_comparison(self.root)

    def write_audit(self):
        checked, panels = {}, []
        for index in range(3):
            folder = self.folder(entry.FINAL_NAMES[index])
            result = self.result(entry.FINAL_NAMES[index])
            panels.append(dict(index=index, run=folder.relative_to(self.root).as_posix(), measurements=result["measurements"]))
            for path in folder.rglob("*"):
                if path.is_file():
                    checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in self.folder("E127-A0").rglob("*"):
            if path.is_file():
                checked[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for source in (*entry.SOURCES, *entry.DIRECT_SOURCES):
            checked[source["path"]] = source["sha256"]
        report = dict(status="passed", protocol=entry.PROTOCOL, mode="final", registered_source="a"*40,
            checked_file_sha256=checked, panels=panels, replicated_gates=entry._replicated_gates(panels))
        path = self.root / entry.AUDIT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return path

    def test_development_uses_frozen_sources_and_saves_complete_assignments(self):
        self.invoke()
        folder, result = self.folder(), self.result()
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.calls, [(2026120600, 0, 8), (2026120600, 1, 8), (2026120600, 2, 16), (2026120600, 3, 16)])
        self.assertEqual((config["grid_steps"], config["post_contacts"], config["probe_placements"]), (50, 16, 12))
        self.assertEqual(config["arms"], list(entry.ARMS))
        self.assertEqual(config["key_roles"], list(entry.KEY_ROLES))
        self.assertEqual(result["family_pre_contacts"], [8, 8, 16, 16])
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["gate_status"], "not_applicable")
        self.assertTrue(all("large_trace" not in row for row in self.projected))
        for artifact in result["raw_files"]:
            self.assertEqual(hashlib.sha256((folder / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
        with np.load(folder / "families/family_0000.npz") as data:
            self.assertIn("large_trace", data)

    def test_final_panels_and_independent_audit_are_both_required(self):
        failed = self.failed_a0()
        before = self.snapshot(failed)
        for index in range(3):
            self.invoke(*self.final_args(index), "--name", entry.FINAL_NAMES[index])
            self.assertEqual(self.comparison()["status"], "undecided")
        self.write_audit()
        decision = self.comparison()
        self.assertEqual(decision["status"], "complete")
        self.assertEqual(decision["replicated_gates"]["plastic"]["local"], dict(count=2, seeds=[0, 2], passed=True))
        self.assertEqual(decision["replicated_gates"]["plastic"]["transfer"], dict(count=1, seeds=[2], passed=False))
        self.assertTrue(decision["replicated_gates"]["plastic"]["hebbian_local"]["passed"])
        self.assertTrue(decision["replicated_gates"]["direct"]["transfer"]["passed"])
        self.assertEqual(decision["excluded_partial_runs"], [failed.relative_to(self.root).as_posix()])
        self.assertEqual(decision["panels"][0]["measurements"]["families"], 128)
        self.assertEqual(before, self.snapshot(failed))

    def test_no_e126_data_or_calibration_is_needed_or_read(self):
        for name in ("unused_E126-plastic-s0-part001", "unused_E123-A0"):
            folder = self.root / "runs" / name
            folder.mkdir()
            (folder / "config.json").write_text("forbidden prior experiment")
            (folder / "summary.json").write_text("forbidden prior outcomes")
        self.invoke()
        self.assertEqual(self.result()["status"], "finished")

    def test_overrides_wrong_identity_and_dirty_code_fail_before_creation(self):
        for flags in (("--seed", "2026120501"), ("--generations", "1"), ("--eval-worlds", "8"),
                      ("--start", "biology"), ("--pop", "16"), ("--cfg", "T=300"), ("--allow-dirty",),
                      ("--controller", "direct-rnn-v1"), ("--resume", "unused")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        with self.assertRaises(SystemExit):
            self.invoke(dirty=True)
        self.assertFalse(list((self.root / "runs").glob("*_E127-*")))

    def test_source_digest_tamper_blocks_before_creation(self):
        path = self.root / entry.DIRECT_SOURCES[0]["path"]
        path.write_bytes(path.read_bytes() + b"wrong source")
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.calls)

    def test_existing_attempt_and_generic_resume_preserve_evidence(self):
        self.invoke()
        before = self.snapshot(self.folder())
        with self.assertRaises(SystemExit):
            self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(self.folder()), common=False)
        self.assertEqual(before, self.snapshot(self.folder()))

    def empty_failure(self):
        self.fail_at = 0
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        self.fail_at = None
        return self.folder()

    def test_empty_failed_benchmark_allows_exactly_one_pinned_retry(self):
        failed = self.empty_failure()
        before = self.snapshot(failed)
        self.invoke("--name", entry.BENCHMARK_RETRY)
        retry = self.folder(entry.BENCHMARK_RETRY)
        config = json.loads((retry / "config.json").read_text())
        self.assertEqual(config["retry_of"], dict(run=failed.relative_to(self.root).as_posix(),
            files={name: hashlib.sha256(raw).hexdigest() for name, raw in before.items()}))
        self.assertEqual((config["eval_seed"], config["expected_families"]), (2026120600, 4))
        self.assertEqual(config["source"], entry.SOURCES[0])
        self.assertEqual(config["direct_source"], entry.DIRECT_SOURCES[0])
        self.assertEqual(self.result(entry.BENCHMARK_RETRY)["status"], "finished")
        self.assertEqual(before, self.snapshot(failed))
        retried = self.snapshot(retry)
        with self.assertRaises(SystemExit):
            self.invoke("--name", entry.BENCHMARK_RETRY)
        self.assertEqual(retried, self.snapshot(retry))
        self.assertEqual(before, self.snapshot(failed))

    def test_retry_requires_original_failure_and_does_not_expand_registered_names(self):
        for flags in (("--name", entry.BENCHMARK_RETRY), ("--name", "E127-benchmark-retry2"),
                      ("--condition-split", "final", "--name", entry.BENCHMARK_RETRY),
                      ("--condition-split", "final", "--name", "E127-A0-retry1")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.invoke(*flags)
        self.assertFalse(self.calls)
        self.assertFalse(list((self.root / "runs").glob("*_E127-*")))

    def test_retry_rejects_completed_evidence_or_changed_original_identity(self):
        failed = self.empty_failure()
        changes = (
            ("status.json", {"status": "finished"}),
            ("status.json", {"status": "interrupted"}),
            ("status.json", {"completed_families": 1}),
            ("summary.json", {"status": "finished"}),
            ("summary.json", {"raw_files": [{"index": 0}]}),
            ("summary.json", {"measurements": {}}),
            ("config.json", {"eval_seed": 2026120601}),
            ("config.json", {"source": entry.SOURCES[1]}),
            ("config.json", {"direct_source": entry.DIRECT_SOURCES[1]}),
            ("config.json", {"grid_steps": 300}),
        )
        for name, update in changes:
            path, original = failed / name, (failed / name).read_bytes()
            data = json.loads(original)
            data.update(update)
            path.write_text(json.dumps(data))
            try:
                with self.subTest(name=name, update=update), self.assertRaises(SystemExit):
                    self.invoke("--name", entry.BENCHMARK_RETRY)
            finally:
                path.write_bytes(original)
        for relative in ("metrics.jsonl", "families/family_0000.npz"):
            path = failed / relative
            original = path.read_bytes() if path.exists() else None
            path.write_bytes(b"unexpected evidence")
            try:
                with self.subTest(evidence=relative), self.assertRaises(SystemExit):
                    self.invoke("--name", entry.BENCHMARK_RETRY)
            finally:
                if original is not None:
                    path.write_bytes(original)
                else:
                    path.unlink()
        self.assertFalse(self.calls)
        self.assertFalse(list((self.root / "runs").glob("*_" + entry.BENCHMARK_RETRY)))

    def test_retry_requires_all_retained_files_unique_predecessor_and_same_assignments(self):
        failed = self.empty_failure()
        before = self.snapshot(failed)
        for name in before:
            path = failed / name
            path.unlink()
            try:
                with self.subTest(missing=name), self.assertRaises(SystemExit):
                    self.invoke("--name", entry.BENCHMARK_RETRY)
            finally:
                path.write_bytes(before[name])
        duplicate = self.root / "runs" / "duplicate_E127-benchmark"
        duplicate.mkdir()
        try:
            with self.assertRaises(SystemExit):
                self.invoke("--name", entry.BENCHMARK_RETRY)
        finally:
            duplicate.rmdir()
        with np.load(failed / "assignments.npz") as archive:
            changed = {name: archive[name].copy() for name in archive.files}
        changed["key"][1] += 1
        np.savez_compressed(failed / "assignments.npz", **changed)
        config = json.loads(before["config.json"])
        config["assignments"]["sha256"] = hashlib.sha256((failed / "assignments.npz").read_bytes()).hexdigest()
        (failed / "config.json").write_text(json.dumps(config))
        try:
            with self.assertRaises(SystemExit):
                self.invoke("--name", entry.BENCHMARK_RETRY)
        finally:
            for name in ("config.json", "assignments.npz"):
                (failed / name).write_bytes(before[name])
        self.assertEqual(before, self.snapshot(failed))
        self.assertFalse(self.calls)

    def test_retry_rechecks_preserved_failure_before_claiming_completion(self):
        failed = self.empty_failure()
        original_evaluate = self.evaluate

        def change_failure(*args, **kwargs):
            row = original_evaluate(*args, **kwargs)
            if args[3] == 3:
                path = failed / "summary.json"
                data = json.loads(path.read_text())
                data["error"] = "changed after retry began"
                path.write_text(json.dumps(data))
            return row

        self.evaluate = change_failure
        with self.assertRaises(SystemExit):
            self.invoke("--name", entry.BENCHMARK_RETRY)
        result = self.result(entry.BENCHMARK_RETRY)
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 4))
        self.assertNotIn("measurements", result)
        self.assertEqual(result["gate_status"], "undecided")

    def test_final_recovery_replays_all_assignments_and_never_decodes_old_outcomes(self):
        failed = self.failed_a0()
        before = self.snapshot(failed)
        old_raw = {raw for name, raw in before.items() if name.startswith("families/")}
        real_load = np.load

        def checked_load(source, *args, **kwargs):
            if isinstance(source, io.BytesIO):
                self.assertNotIn(source.getvalue(), old_raw)
            elif isinstance(source, (str, Path)):
                self.assertNotIn((failed / "families").resolve(), Path(source).resolve().parents)
            return real_load(source, *args, **kwargs)

        self.calls.clear()
        with patch.object(entry.np, "load", side_effect=checked_load):
            self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
        self.assertEqual([index for _, index, _ in self.calls], list(range(128)))
        self.assertTrue(all(master == 2026120601 for master, _, _ in self.calls))
        recovered = self.folder(entry.A0_RECOVERY)
        config = json.loads((recovered / "config.json").read_text())
        self.assertEqual(config["recovery_of"], dict(run=failed.relative_to(self.root).as_posix(),
            files={name: hashlib.sha256(raw).hexdigest() for name, raw in before.items()}))
        result = self.result(entry.A0_RECOVERY)
        self.assertEqual((result["completed_families"], len(result["raw_files"])), (128, 128))
        self.assertEqual(before, self.snapshot(failed))
        complete = self.snapshot(recovered)
        with self.assertRaises(SystemExit):
            self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
        self.assertEqual(complete, self.snapshot(recovered))

    def test_recovery_rejects_missing_changed_and_wrong_source_predecessors(self):
        with self.assertRaises(SystemExit):
            self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
        failed = self.failed_a0()
        self.calls.clear()
        for relative in ("status.json", "metrics.jsonl", "families/family_0010.npz"):
            path = failed / relative
            raw = path.read_bytes()
            path.write_bytes(raw + b"changed")
            try:
                with self.subTest(path=relative), self.assertRaises(SystemExit):
                    self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
            finally:
                path.write_bytes(raw)
        pin = dict(entry.RECOVERY_PREDECESSORS[entry.A0_RECOVERY], commit="b"*40)
        with patch.dict(entry.RECOVERY_PREDECESSORS, {entry.A0_RECOVERY: pin}), self.assertRaises(SystemExit):
            self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
        for name in ("E127-A0-recovery2", "E127-A1-recovery1", "E127-benchmark-recovery3"):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.invoke(*self.final_args(0), "--name", name)
        self.assertFalse(self.calls)
        self.assertFalse(list((self.root / "runs").glob("*_" + entry.A0_RECOVERY)))

    def test_recovery_checks_prefix_journal_order_keys_and_raw_manifest(self):
        failed = self.failed_a0()
        path = failed / "metrics.jsonl"
        original = path.read_bytes()
        for change in ("order", "keys", "raw_digest"):
            rows = [json.loads(line) for line in original.splitlines()]
            if change == "order":
                rows[0], rows[1] = rows[1], rows[0]
            elif change == "keys":
                rows[0]["keys"][0][1] += 1
            else:
                rows[0]["raw_file"]["sha256"] = "0"*64
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.pin_predecessor(entry.A0_RECOVERY, failed)
            try:
                with self.subTest(change=change), self.assertRaises(SystemExit):
                    self.invoke(*self.final_args(0), "--name", entry.A0_RECOVERY)
            finally:
                path.write_bytes(original)
        self.assertFalse(list((self.root / "runs").glob("*_" + entry.A0_RECOVERY)))

    def test_corrected_rehearsal_keeps_the_entire_previous_development_chain(self):
        original = self.empty_failure()
        self.invoke("--name", entry.BENCHMARK_RETRY)
        previous = self.folder(entry.BENCHMARK_RETRY)
        self.pin_predecessor(entry.BENCHMARK_RECOVERY, previous)
        original_before, before = self.snapshot(original), self.snapshot(previous)
        self.calls.clear()
        self.invoke("--name", entry.BENCHMARK_RECOVERY)
        self.assertEqual(self.calls, [(2026120600, 0, 8), (2026120600, 1, 8),
                                     (2026120600, 2, 16), (2026120600, 3, 16)])
        config = json.loads((self.folder(entry.BENCHMARK_RECOVERY) / "config.json").read_text())
        self.assertEqual(config["recovery_of"], dict(run=previous.relative_to(self.root).as_posix(),
            files={name: hashlib.sha256(raw).hexdigest() for name, raw in before.items()}))
        self.assertEqual(self.result(entry.BENCHMARK_RECOVERY)["gate_status"], "not_applicable")
        self.assertEqual(original_before, self.snapshot(original))
        self.assertEqual(before, self.snapshot(previous))
        with self.assertRaises(SystemExit):
            self.invoke("--name", entry.BENCHMARK_RECOVERY)

    def test_status_timing_uses_actual_permuted_boundary_schedule(self):
        schedule = self.assignments(2026120600, 4)
        schedule["permutation"] = np.asarray([2, 0, 3, 1], np.int32)
        schedule["pre_contacts"] = np.asarray([16, 8, 16, 8], np.int32)
        self.assignments = lambda master, count: schedule
        self.invoke()
        status = json.loads((self.folder() / "status.json").read_text())
        self.assertEqual(status["family_pre_contacts"], [16, 8, 16, 8])
        self.assertEqual(len(status["family_seconds"]), len(status["family_pre_contacts"]))
        self.assertEqual(status["family_pre_contacts"], self.result()["family_pre_contacts"])

    def test_family_wall_timing_includes_journal_sync_and_status_write(self):
        write_status, fsync = entry._write_status, entry.os.fsync

        def timed_status(*args, **kwargs):
            write_status(*args, **kwargs)
            self.now += .5

        def timed_sync(*args, **kwargs):
            fsync(*args, **kwargs)
            self.now += .25

        with patch.object(entry, "_write_status", side_effect=timed_status), \
                patch.object(entry.os, "fsync", side_effect=timed_sync):
            self.invoke()
        status = json.loads((self.folder() / "status.json").read_text())
        result = self.result()
        self.assertEqual(len(status["family_wall_seconds"]), status["expected_families"])
        self.assertEqual(status["family_wall_seconds"], result["family_wall_seconds"])
        self.assertTrue(all(wall >= 0 and wall >= inner + .75
                            for wall, inner in zip(status["family_wall_seconds"], status["family_seconds"])))

    def test_failure_keeps_prefix_without_result_or_decision(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("failed", 2))
        self.assertEqual(len(result["family_wall_seconds"]), 2)
        self.assertNotIn("measurements", result)
        self.assertEqual(result["gate_status"], "undecided")

    def test_repeated_assignment_keys_and_nonfinite_events_are_rejected(self):
        self.reuse = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 1)

    def test_nonfinite_evidence_is_rejected_before_save(self):
        self.nonfinite = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_raw_controller_digest_must_match_selected_source(self):
        self.bad_source = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.result()["completed_families"], 0)

    def test_deadline_preserves_complete_family_without_partial_scientific_result(self):
        self.expire = True
        with self.assertRaises(TimeoutError):
            self.invoke()
        result = self.result()
        self.assertEqual((result["status"], result["completed_families"]), ("interrupted", 1))
        self.assertNotIn("measurements", result)

    def test_stale_audit_and_disagreeing_replication_are_rejected(self):
        self.finish_panels()
        audit_path = self.write_audit()
        report = json.loads(audit_path.read_text())
        report["replicated_gates"]["plastic"]["transfer"]["passed"] = True
        audit_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.comparison()
        self.write_audit()
        path = self.folder("E127-A1") / "families/family_0000.npz"
        path.write_bytes(path.read_bytes() + b"changed since audit")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.comparison()

    def test_audit_must_select_recovery_and_cover_every_retained_partial_file(self):
        self.finish_panels()
        path = self.write_audit()
        original = path.read_bytes()
        report = json.loads(original)
        report["panels"][0]["run"] = self.folder("E127-A0").relative_to(self.root).as_posix()
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "different final panels"):
            self.comparison()
        report = json.loads(original)
        old_raw = (self.folder("E127-A0") / "families/family_0010.npz").relative_to(self.root).as_posix()
        del report["checked_file_sha256"][old_raw]
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.comparison()

    def test_generated_report_keeps_decisions_locked_until_verified(self):
        self.finish_panels()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(entry.engine, "assignments_manifest", side_effect=self.assignments), \
                patch.object(entry.engine, "evaluation_keys", side_effect=self.keys):
            summarize.main()
        path = self.root / "results/RESULTS.md"
        self.assertIn("Replicated diagnostic decisions: **undecided**", path.read_text(encoding="utf-8"))
        self.assertFalse((self.root / "results/diagnostics/E127-comparison.json").exists())
        self.write_audit()
        with patch.object(summarize, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(entry.engine, "assignments_manifest", side_effect=self.assignments), \
                patch.object(entry.engine, "evaluation_keys", side_effect=self.keys):
            summarize.main()
        text = path.read_text(encoding="utf-8")
        self.assertIn("| plastic | transfer | 1/3 | 2 | FAIL |", text)
        self.assertIn("n=128", text)
        self.assertIn("preserved partial; excluded", text)
        self.assertIn("eleven families are not added", text)
        self.assertTrue((self.root / "results/diagnostics/E127-comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
