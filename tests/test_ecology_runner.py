"""Development-only ecology entry, immutable attempts, and streamed cohort evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import ecology_lineage as entry, run_experiment as runner
from seed.seed import dims


class EcologyRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT/".tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ecology-runner-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.common = ["--eval-protocol", entry.PROTOCOL, "--quick", "--condition-split", "development",
                       "--seed", "41", "--generations", "0"]
        self.fail_before = None

    def fake_lineage(self, g, wkeys, dkeys, rules, c, *, ecologies, plastic, on_life, persistence=None):
        self.assertEqual(c, entry.DEVELOPMENT_CFG)
        self.assertEqual(len(wkeys), 3)
        self.assertEqual(len({np.asarray(key).tobytes() for key in dkeys}), 3)
        self.assertTrue(plastic)
        self.assertEqual([(e.movement_energy_cost, e.seasonal_energy_cost) for e in ecologies],
                         [(0., 0.), (.01, 0.), (.01, .02)])
        _, sensory, motors, _, _ = dims(c, ecology=ecologies[0])
        archive = np.zeros((motors, sensory), np.float32)
        trait = {} if persistence is None else dict(persistence=dict(
            version=persistence.version, memory_steps=persistence.memory_steps, stall_steps=persistence.stall_steps))
        for i in range(3):
            if i == self.fail_before:
                raise RuntimeError("fixture cohort failure")
            raw = {}
            for condition in ("inherited", "fresh"):
                events = dict(food=np.zeros((c.T, c.M), bool), poison=np.zeros((c.T, c.M), bool),
                              alive_before=np.ones((c.T, c.M), np.float32), alive_after=np.ones((c.T, c.M), np.float32))
                if condition == "inherited" and i:
                    events["food"][0, 0] = True
                occupied = np.zeros((c.T, c.M), bool)
                occupied[0, 0] = occupied[1, 1] = True
                raw[condition] = dict(events=events, fitness=float(events["food"].sum()/c.M),
                    ecology=dict(shelter=np.eye(c.G, dtype=bool), sheltered=occupied,
                                 energy_lost=np.zeros((c.T, c.M), np.float32), initial_sheltered=np.array([True, False])))
                if persistence is not None:
                    raw[condition]["persistence"] = {name: np.zeros((c.T, c.M), bool) for name in
                        ("active", "remembered", "progress_observed", "stalled", "tactic_attempted", "action_switched")}
            row = dict(life=i, agent_ids=np.column_stack((np.full(c.M, i), np.arange(c.M))),
                world_key=np.asarray(wkeys[i]), development_key=np.asarray(dkeys[i]), rule=int(rules[i]),
                ecology=dict(version="ecology-v1", movement_energy_cost=ecologies[i].movement_energy_cost,
                             seasonal_energy_cost=ecologies[i].seasonal_energy_cost), **raw,
                archive_before=archive.copy(), archive_after=archive.copy(),
                donor_delta=np.zeros((c.M, motors, sensory), np.float32),
                donor_mask=np.ones((c.M, motors, sensory), np.float32), donor_counts=np.full_like(archive, c.M), **trait)
            state = dict(protocol="lineage-v1", state_version=1, method="sensor-motor-mean-delta-v1",
                genome_sha256="a"*64, layout=dict(sensors=sensory, motors=motors, hidden=c.H, hidden_types=c.n_htypes,
                view_radius=c.R, weight_bound=c.wmax, ecology_version="ecology-v1"), next_life=i+1, archive=archive.copy(),
                used_world_keys=np.asarray(wkeys[:i+1]), used_development_keys=np.asarray(dkeys[:i+1]), **trait)
            row["archive_state"] = state
            on_life(row)
        return dict(protocol="lineage-v1", mode="collective", genome=g, config=c._asdict(),
                    lives=[], completed_lives=3, final_archive=archive, archive_state=state, **trait)

    def invoke(self, *extra, base=True, actual=False, dirty=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "ROOT", self.root))
            stack.enter_context(patch.object(runner, "git_info", return_value=dict(commit="a"*40, dirty=dirty)))
            stack.enter_context(patch.object(runner.sys, "argv", ["runner", *(self.common if base else []), *extra]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if not actual:
                stack.enter_context(patch.object(entry.lineage, "collective_lives", side_effect=self.fake_lineage))
            runner.main()

    def folder(self, name):
        return next((self.root/"runs").glob("*_"+name))

    def snapshot(self, folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    def read_cohort(self, folder, index):
        meta = json.loads((folder/f"cohort_{index:03d}.json").read_text())
        with np.load(folder/f"cohort_{index:03d}.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"array", "shape", "dtype"}:
                    array = arrays[value["array"]]
                    self.assertEqual(list(array.shape), value["shape"])
                    self.assertEqual(str(array.dtype), value["dtype"])
                    return array
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value
        return unpack(meta)

    def test_saved_source_selected_memory_and_both_traces_are_preserved(self):
        source = self.root/"source.npy"
        np.save(source, np.zeros(155, np.float32))
        data = source.read_bytes()
        for condition in ("shared", "empty"):
            self.invoke("--name", condition, "--start", "file:"+str(source), "--memory-condition", condition)
            folder = self.folder(condition)
            config = json.loads((folder/"config.json").read_text())
            summary = json.loads((folder/"summary.json").read_text())
            history = [json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
            self.assertEqual((folder/"genome_start.npy").read_bytes(), data)
            self.assertEqual(config["source"]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(config["args"]["start"], "file:genome_start.npy")
            self.assertEqual([row["phase"] for row in history], list(entry.PHASES))
            self.assertTrue(config["schedule_independent_of_competence"])
            self.assertTrue(config["paired_traces_retained"])
            self.assertEqual(summary["completed_cohorts"], 3)
            self.assertEqual(summary["status"], "finished")
            self.assertFalse(summary["scientific_result"])
            self.assertNotIn("gate", summary)
            self.assertNotIn("persistence", config["args"])
            self.assertNotIn("persistence", config)
            self.assertNotIn("persistence", summary)
            self.assertEqual(history[1]["selected"]["fitness"], .5 if condition == "shared" else 0.)
            self.assertEqual(history[1]["selected"]["agents_initially_sheltered"], 1)
            self.assertEqual(history[1]["selected"]["agents_reaching_shelter"], 1)
            row = self.read_cohort(folder, 1)
            self.assertIn("inherited", row)
            self.assertIn("fresh", row)
            self.assertEqual(row["inherited"]["events"]["alive_before"].dtype, np.float32)
            self.assertEqual(json.loads((folder/"status.json").read_text())["status"], "finished")

    def test_invalid_science_or_ignored_options_create_no_run(self):
        for extra in (("--condition-split", "final"), ("--seed", "42"), ("--generations", "1"),
                      ("--cfg", "T=300"), ("--no-plastic",), ("--controller", "direct-rnn-v1"),
                      ("--objective", "paired-min"), ("--max-train-seconds", "100"), ("--worlds", "2"),
                      ("--start", "biology"), ("--name", "../escape")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke("--name", "invalid", *extra)
            self.assertFalse((self.root/"runs").exists())
        without_quick = [arg for arg in self.common if arg != "--quick"]
        with self.assertRaises(SystemExit):
            self.invoke(*without_quick, base=False)
        with self.assertRaises(SystemExit):
            self.invoke("--memory-condition", "shared", base=False)
        protocols = (None, "v1", "v2", "structural-v1", runner.broader_eval.PROTOCOL, runner.reversal_eval.PROTOCOL,
                     runner.eval_v3.PROTOCOL, runner.eval_v4.PROTOCOL, runner.eval_v5.PROTOCOL, runner.eval_v6.PROTOCOL,
                     runner.DIRECT_PROTOCOL, *runner.PAIRED_PROTOCOLS)
        for protocol in protocols:
            options = [] if protocol is None else ["--eval-protocol", protocol]
            with self.subTest(protocol=protocol), self.assertRaises(SystemExit):
                self.invoke(*options, "--persistence", base=False)
        with self.assertRaises(SystemExit):
            self.invoke("--persistence", "--resume", str(self.root/"old-run"), base=False)
        self.assertFalse((self.root/"runs").exists())

    def test_completed_attempt_cannot_be_replaced_or_resumed(self):
        self.invoke("--name", "once", dirty=True)
        folder = self.folder("once")
        self.assertTrue(json.loads((folder/"config.json").read_text())["git"]["dirty"])
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke("--name", "once")
        with self.assertRaises(SystemExit):
            self.invoke("--resume", str(folder), base=False)
        self.assertEqual(before, self.snapshot(folder))

    def test_failure_or_deadline_retains_completed_cohort_and_terminal_status(self):
        self.fail_before = 1
        with self.assertRaisesRegex(RuntimeError, "fixture cohort"):
            self.invoke("--name", "failed")
        folder = self.folder("failed")
        self.assertTrue((folder/"cohort_000.npz").exists())
        self.assertFalse((folder/"cohort_001.npz").exists())
        summary = json.loads((folder/"summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_cohorts"]), ("failed", 1))
        self.fail_before = None
        with patch.object(entry, "MAX_SECONDS", 0), self.assertRaises(TimeoutError):
            self.invoke("--name", "deadline")
        summary = json.loads((self.folder("deadline")/"summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_cohorts"]), ("interrupted", 1))

    def test_actual_cli_fixture_has_three_fresh_cohorts_and_real_ecology_events(self):
        self.invoke("--name", "actual", "--memory-condition", "empty", actual=True)
        folder = self.folder("actual")
        summary = json.loads((folder/"summary.json").read_text())
        self.assertEqual((summary["status"], summary["completed_cohorts"]), ("finished", 3))
        keys = []
        for i in range(3):
            row = self.read_cohort(folder, i)
            keys.append(row["development_key"].tobytes())
            np.testing.assert_array_equal(row["agent_ids"], [[i, 0], [i, 1]])
            for arm in ("inherited", "fresh"):
                self.assertNotIn("persistence", row[arm])
                self.assertTrue(row[arm]["events"]["alive_before"][0].all())
                self.assertEqual(row[arm]["ecology"]["shelter"].shape, (6, 6))
                self.assertEqual(row[arm]["ecology"]["energy_lost"].shape, (12, 2))
            if i == 0:
                np.testing.assert_array_equal(row["archive_before"], 0.)
                np.testing.assert_array_equal(row["inherited"]["events"]["food"], row["fresh"]["events"]["food"])
                np.testing.assert_array_equal(row["inherited"]["ecology"]["energy_lost"], 0.)
        self.assertEqual(len(set(keys)), 3)
        state = entry.load_development_archive(folder)
        self.assertNotIn("persistence", state)
        self.assertEqual(state["next_life"], 3)
        np.testing.assert_array_equal(state["archive"], row["archive_after"])
        for key in ("archive", "used_world_keys", "used_development_keys"):
            np.testing.assert_array_equal(state[key], row["archive_state"][key])
        with self.assertRaisesRegex(ValueError, "seed 41"):
            entry.run_development_curriculum(seed=123)
        self.invoke("--name", "actual-persistent", "--memory-condition", "empty", "--persistence", actual=True)
        enabled = self.folder("actual-persistent")
        setting = dict(version="persistence-v1", memory_steps=4, stall_steps=2)
        config = json.loads((enabled/"config.json").read_text())
        summary = json.loads((enabled/"summary.json").read_text())
        self.assertEqual((enabled/"genome_start.npy").read_bytes(), (folder/"genome_start.npy").read_bytes())
        self.assertIs(config["args"]["persistence"], True)
        self.assertEqual(config["persistence"], setting)
        self.assertEqual(summary["persistence"], setting)
        loaded = entry.load_development_archive(enabled)
        self.assertEqual(loaded["persistence"], setting)
        for i in range(3):
            off, on = self.read_cohort(folder, i), self.read_cohort(enabled, i)
            self.assertEqual(on["persistence"], setting)
            self.assertEqual(on["archive_state"]["persistence"], setting)
            for key in ("world_key", "development_key", "agent_ids"):
                np.testing.assert_array_equal(off[key], on[key])
            for condition, arm in entry.MEMORY_CONDITIONS.items():
                activity = on[arm]["persistence"]
                for name in ("active", "remembered", "progress_observed", "stalled", "tactic_attempted", "action_switched"):
                    self.assertEqual(activity[name].shape, (12, 2))
                    self.assertEqual(summary["observations"][i]["paired"][condition]["persistence_"+name+"_agent_steps"],
                                     int(activity[name].sum()))
                for name in ("previous_action", "selected_action"):
                    self.assertEqual(activity[name].shape, (12, 2))
        np.testing.assert_array_equal(loaded["archive"], on["archive_after"])

    def test_optional_persistence_archive_binding_is_checked(self):
        self.invoke("--name", "persistent", "--persistence")
        folder = self.folder("persistent")
        expected = dict(version="persistence-v1", memory_steps=4, stall_steps=2)
        self.assertEqual(entry.load_development_archive(folder)["persistence"], expected)
        path = folder/"lineage_final.json"
        original = path.read_bytes()
        for setting in (dict(expected, version="other"), dict(expected, memory_steps=0),
                        dict(expected, memory_steps=3), dict(expected, extra=1)):
            metadata = json.loads(original)
            metadata["archive_state"]["persistence"] = setting
            path.write_text(json.dumps(metadata))
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                entry.load_development_archive(folder)
        metadata = json.loads(original)
        metadata["archive_state"].pop("persistence")
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "persistence setting disagrees"):
            entry.load_development_archive(folder)
        path.write_bytes(original)
        config_path = folder/"config.json"
        config = json.loads(config_path.read_text())
        config.pop("persistence")
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "persistence setting disagrees"):
            entry.load_development_archive(folder)

    def test_archive_loader_rejects_partial_legacy_and_unsafe_array_references(self):
        self.invoke("--name", "portable")
        folder = self.folder("portable")
        state = entry.load_development_archive(folder)
        self.assertEqual(state["archive"].shape, (6, 49))
        meta_path = folder/"lineage_final.json"
        original = meta_path.read_bytes()
        for field, invalid in (("array", "../outside"), ("shape", [49, 6]), ("dtype", "object")):
            metadata = json.loads(original)
            metadata["archive_state"]["archive"][field] = invalid
            meta_path.write_text(json.dumps(metadata))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "array reference"):
                entry.load_development_archive(folder)
        metadata = json.loads(original)
        metadata.pop("archive_state")
        meta_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "no portable archive_state"):
            entry.load_development_archive(folder)
        meta_path.write_bytes(original)
        summary_path = folder/"summary.json"
        summary = json.loads(summary_path.read_text())
        summary["final_memory"]["metadata"] = "../outside.json"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "fixed local"):
            entry.load_development_archive(folder)
        summary["status"] = "interrupted"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "finished three-cohort"):
            entry.load_development_archive(folder)


if __name__ == "__main__":
    unittest.main()
