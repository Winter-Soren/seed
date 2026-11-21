"""E116 runner boundaries and tiny seed-41 recovery; no held-out simulation."""
import contextlib
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from seed import paired
from scripts import eval_v3, eval_v4, eval_v5, eval_v6, run_experiment as runner
import test_paired_runner as fixtures


class E116RunnerTests(unittest.TestCase):
    folder = fixtures.PairedRunnerTests.folder
    snapshot = fixtures.PairedRunnerTests.snapshot
    step = fixtures.PairedRunnerTests.step
    fixture_evaluate = fixtures.PairedRunnerTests.fixture_evaluate

    def setUp(self):
        fixtures.PairedRunnerTests.setUp(self)
        self.common[self.common.index(runner.PAIRED_PROTOCOL)] = runner.PAIRED_V2_PROTOCOL
        self.common[self.common.index('paired-mean')] = 'paired-min'

    def invoke(self, *args, diagnostic=False, evaluator=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'ROOT', self.root))
            stack.enter_context(patch.object(runner, 'V4_SOURCES', tuple(self.manifest)))
            stack.enter_context(patch.object(runner, 'git_info', return_value=dict(commit='b'*40, dirty=False)))
            stack.enter_context(patch.object(runner.sys, 'argv', ['runner', *args]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if diagnostic:
                stack.enter_context(patch.object(eval_v6, 'evaluate_controller', side_effect=evaluator or self.fixture_evaluate))
            else:
                stack.enter_context(patch.object(paired, 'make_evaluator', side_effect=self.factory))
                stack.enter_context(patch.object(runner, 'es_step', side_effect=self.step))
                stack.enter_context(patch.object(runner.time, 'monotonic', side_effect=lambda: self.now))
                for module, function in ((runner, 'test_genome'), (eval_v3, 'evaluate_conditions'),
                                         (eval_v4, 'evaluate_controller'), (eval_v5, 'evaluate_controller'),
                                         (eval_v6, 'evaluate_controller')):
                    stack.enter_context(patch.object(module, function,
                        side_effect=AssertionError('training evaluated held-out worlds')))
            runner.main()

    def final_args(self, seed=0, arm='control'):
        return dict(name=f'E116-{arm}-s{seed}', eval_protocol=runner.PAIRED_V2_PROTOCOL,
                    objective='paired-min', condition_split='final', controller='rules-v0',
                    start='file:'+runner.V4_SOURCES[seed]['run']+'/genome_final.npy',
                    generations=20, seed=2000+seed, pop=64, worlds=4, sigma=.1,
                    lr=.1 if arm == 'control' else .02, checkpoint_every=10, eval_worlds=256,
                    no_plastic=False, quick=False, allow_dirty=False, structural_plasticity='off')

    def train_bundle(self):
        for seed in range(3):
            source = self.root / self.manifest[seed]['run'] / 'genome_final.npy'
            for arm, lr in (('control', '.1'), ('candidate', '.02')):
                self.invoke('--name', f'E116-{arm}-s{seed}', *self.common, '--lr', lr,
                            '--start', f'file:{source}', '--generations', '1')

    def diagnostic(self, **kwargs):
        self.invoke('--name', 'E116-v6', '--eval-protocol', 'v6', '--condition-split', 'development',
                    '--generations', '0', '--eval-worlds', '2', *self.cfg_args, diagnostic=True, **kwargs)

    def test_final_settings_and_original_sources_are_pinned_for_every_pair(self):
        for seed in range(3):
            sources = []
            for arm in ('control', 'candidate'):
                args = self.final_args(seed, arm)
                self.assertEqual(runner.validate_eval_protocol(args, {}), 'paired-train-v2')
                flat, data, metadata = runner.paired_source(args, runner.Cfg())
                self.assertEqual(hashlib.sha256(data).hexdigest(), runner.V4_SOURCES[seed]['sha256'])
                self.assertEqual(metadata['source_run'], runner.V4_SOURCES[seed]['run'])
                sources.append((data, metadata))
            self.assertEqual(sources[0], sources[1])
        wrong_source = self.final_args()
        wrong_source['start'] = self.final_args(1)['start']
        with self.assertRaisesRegex(SystemExit, 'pinned original'):
            runner.paired_source(wrong_source, runner.Cfg())
        for change in (dict(generations=19), dict(generations=50), dict(seed=1000), dict(pop=128),
                       dict(worlds=8), dict(sigma=.05), dict(lr=.03), dict(lr=.02),
                       dict(name='alternate'), dict(name='E116-candidate-s0'), dict(objective='paired-mean'),
                       dict(checkpoint_every=1), dict(eval_worlds=128)):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                runner.validate_eval_protocol(dict(self.final_args(), **change), {})
        with self.assertRaises(SystemExit):
            runner.validate_eval_protocol(self.final_args(), {'T': 300})
        legacy = dict(self.final_args(), eval_protocol=runner.PAIRED_PROTOCOL, generations=50,
                      seed=1000, name='E115-mean-s0', objective='paired-mean')
        self.assertEqual(runner.validate_eval_protocol(legacy, {}), 'paired-train-v1')
        with self.assertRaises(SystemExit):
            runner.validate_eval_protocol(dict(legacy, lr=.02), {})

    def test_invalid_training_requests_create_no_artifacts(self):
        for extra in (('--objective', 'paired-mean'), ('--lr', '.03'), ('--seed', '42'),
                      ('--generations', '5'), ('--pop', '6'), ('--worlds', '5'), ('--quick',),
                      ('--allow-dirty',), ('--no-plastic',), ('--start', 'blank'),
                      ('--structural-plasticity', runner.MODE), ('--eval-protocol', 'v1')):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke('--name', 'invalid', *self.common, *extra)
            self.assertFalse((self.root/'runs').exists())

    def test_both_learning_rates_resume_exactly_and_defer_all_evaluation(self):
        finals, initial = [], []
        for arm, lr in (('control', '.1'), ('candidate', '.02')):
            self.invoke('--name', 'whole-'+arm, *self.common, '--lr', lr)
            whole = self.folder('whole-'+arm)
            self.invoke('--name', 'split-'+arm, *self.common, '--lr', lr, '--max-train-seconds', '2')
            split = self.folder('split-'+arm)
            prefix = (split/'metrics.jsonl').read_bytes()
            self.assertEqual(runner.load_checkpoint(split)['t'], 2)
            self.invoke('--resume', str(split))
            self.assertTrue((split/'metrics.jsonl').read_bytes().startswith(prefix))
            self.assertEqual((whole/'genome_final.npy').read_bytes(), (split/'genome_final.npy').read_bytes())
            for key, value in runner.load_checkpoint(whole).items():
                np.testing.assert_array_equal(value, runner.load_checkpoint(split)[key])
            config = json.loads((whole/'config.json').read_text())
            initial.append(config['initial_optimizer'])
            self.assertEqual(config['eval_protocol'], 'paired-train-v2')
            self.assertEqual(config['args']['lr'], float(lr))
            self.assertEqual((whole/'genome_start.npy').read_bytes(), self.source.read_bytes())
            summary = json.loads((whole/'summary.json').read_text())
            self.assertEqual((summary['status'], summary['training_complete'], summary['evaluation_status']),
                             ('finished', True, 'deferred'))
            self.assertEqual(summary['eval_protocol'], 'paired-train-v2')
            self.assertFalse(any(key in summary for key in ('evaluations', 'final_learning', 'final_frozen')))
            self.assertFalse((whole/'events.npz').exists())
            finals.append(runner.load_checkpoint(whole))
        self.assertEqual(initial[0], initial[1])
        self.assertEqual(initial[0]['generation'], 0)
        self.assertTrue(initial[0]['adam_m_zero'] and initial[0]['adam_v_zero'])
        np.testing.assert_array_equal(finals[0]['key'], finals[1]['key'])
        self.assertFalse(np.array_equal(finals[0]['mu'], finals[1]['mu']))
        self.crash_at = 3
        with self.assertRaises(fixtures.ProcessLoss):
            self.invoke('--name', 'crash', *self.common, '--lr', '.02')
        crash = self.folder('crash')
        prefix = (crash/'metrics.jsonl').read_bytes()
        self.crash_at = None
        self.invoke('--resume', str(crash))
        self.assertTrue((crash/'metrics.jsonl').read_bytes().startswith(prefix))
        self.assertEqual(json.loads((crash/'config.json').read_text())['resumes'][-1]['replayed_generations'], 1)
        for key, value in finals[1].items():
            np.testing.assert_array_equal(value, runner.load_checkpoint(crash)[key])

    def test_resume_rejects_changed_protocol_arm_and_provenance_before_writes(self):
        self.invoke('--name', 'resume', *self.common, '--max-train-seconds', '2')
        folder = self.folder('resume')
        before = self.snapshot(folder)
        for extra in (('--lr', '.02'), ('--objective', 'paired-mean'), ('--eval-protocol', runner.PAIRED_PROTOCOL),
                      ('--generations', '3'), ('--seed', '42'), ('--name', 'candidate'), ('--start', 'biology'),
                      ('--condition-split', 'final'), ('--cfg', 'G=3'), ('--allow-dirty',)):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke('--resume', str(folder), *extra)
            self.assertEqual(before, self.snapshot(folder))
        for field, value in (('git', dict(commit='b'*40, dirty=True)),
                             ('initial_optimizer', dict(method='es_init', adam_m_zero=False)),
                             ('eval_protocol', runner.PAIRED_PROTOCOL)):
            config = json.loads(before['config.json'])
            config[field] = value
            (folder/'config.json').write_text(json.dumps(config))
            changed = self.snapshot(folder)
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.invoke('--resume', str(folder))
            self.assertEqual(changed, self.snapshot(folder))
        (folder/'config.json').write_bytes(before['config.json'])
        self.invoke('--resume', str(folder))
        completed = self.snapshot(folder)
        for args in (('--resume', str(folder)), ('--name', 'resume', *self.common)):
            with self.assertRaises(SystemExit):
                self.invoke(*args)
            self.assertEqual(completed, self.snapshot(folder))

    def test_v6_records_nine_exact_inputs_fourteen_panels_and_thirty_contrasts(self):
        self.train_bundle()
        training = self.snapshot(self.root/'runs')
        self.diagnostic()
        folder = self.folder('E116-v6')
        summary, config = [json.loads((folder/name).read_text()) for name in ('summary.json', 'config.json')]
        self.assertEqual((summary['status'], summary['completed_panels'], summary['artifact_validity']),
                         ('finished', 14, 'passed'))
        self.assertEqual(tuple(summary['panels']), eval_v6.PANELS)
        self.assertEqual((summary['eval_protocol'], summary['eval_seed']), ('v6', 41))
        self.assertEqual(config['internal_seconds_limit'], 1140)
        self.assertEqual(len(summary['contrasts']), 30)
        self.assertEqual({row['comparison'] for row in summary['contrasts']},
                         {'candidate-minus-control', 'candidate-minus-start'})
        self.assertFalse(summary['gate']['passed'])
        self.assertEqual(len(list(folder.glob('*/*.npz'))), 70)
        self.assertEqual([item['label'] for item in config['input_genomes']], list(eval_v6.PANELS[:9]))
        for metadata in config['input_genomes']:
            self.assertEqual((folder/metadata['copied_genome']).read_bytes(),
                             (self.root/metadata['run']/'genome_final.npy').read_bytes())
        for relative, data in training.items():
            self.assertEqual((self.root/'runs'/relative).read_bytes(), data)
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.invoke('--resume', str(folder))
        self.assertEqual(before, self.snapshot(folder))

    def test_v6_rejects_incomplete_or_mismatched_sources_before_artifacts(self):
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertFalse((self.root/'runs').exists())
        self.train_bundle()
        target = self.folder('E116-candidate-s2')
        saved = self.snapshot(target)
        mutations = [('summary.json', 'status', 'interrupted'), ('summary.json', 'eval_protocol', runner.PAIRED_PROTOCOL),
                     ('summary.json', 'evaluation_status', 'finished'), ('config.json', 'git', dict(commit='b'*40, dirty=True))]
        for filename, key, value in mutations:
            data = json.loads(saved[filename])
            data[key] = value
            (target/filename).write_text(json.dumps(data))
            before = self.snapshot(self.root/'runs')
            with self.subTest(key=key), self.assertRaises(SystemExit):
                self.diagnostic()
            self.assertEqual(before, self.snapshot(self.root/'runs'))
            (target/filename).write_bytes(saved[filename])
        config = json.loads(saved['config.json'])
        config['args']['lr'] = .1
        (target/'config.json').write_text(json.dumps(config))
        with self.assertRaises(SystemExit):
            self.diagnostic()
        (target/'config.json').write_bytes(saved['config.json'])
        with np.load(target/'checkpoint.npz', allow_pickle=False) as checkpoint:
            changed = {name: checkpoint[name].copy() for name in checkpoint.files}
        changed['key'][0] ^= np.uint32(1)
        np.savez(target/'checkpoint.npz', **changed)
        before = self.snapshot(self.root/'runs')
        with (patch.object(runner, 'ROOT', self.root),
              patch.object(runner, 'V4_SOURCES', tuple(self.manifest)),
              self.assertRaisesRegex(ValueError, 'RNG')):
            runner.load_v6_sources(self.cfg, 'development')
        self.assertEqual(before, self.snapshot(self.root/'runs'))
        (target/'checkpoint.npz').write_bytes(saved['checkpoint.npz'])
        (self.root/'runs'/'duplicate_E116-control-s0').mkdir()
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertFalse(list((self.root/'runs').glob('*_E116-v6')))

    def test_v6_interruption_preserves_completed_and_partial_outputs_without_gate(self):
        self.train_bundle()
        calls, completed = 0, {}

        def stop_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self.fixture_evaluate(*args, **kwargs)
            completed.update(self.snapshot(self.folder('E116-v6')/'start0'))
            callback = kwargs['on_condition']

            def stop(label, metrics, arrays):
                callback(label, metrics, arrays)
                raise TimeoutError('fixture deadline')

            return self.fixture_evaluate(*args, **dict(kwargs, on_condition=stop))

        with self.assertRaises(TimeoutError):
            self.diagnostic(evaluator=stop_after_first)
        folder = self.folder('E116-v6')
        summary = json.loads((folder/'summary.json').read_text())
        self.assertEqual((summary['status'], summary['completed_panels']), ('interrupted', 1))
        self.assertEqual(summary['panels']['control0']['status'], 'interrupted')
        self.assertFalse(any(key in summary for key in ('gate', 'contrasts', 'artifact_validity')))
        self.assertTrue((folder/'control0/stationary_r0.npz').exists())
        self.assertEqual(completed, self.snapshot(folder/'start0'))
        before = self.snapshot(folder)
        with self.assertRaises(SystemExit):
            self.diagnostic()
        self.assertEqual(before, self.snapshot(folder))

    def test_v6_final_rejects_unregistered_name_before_source_reads(self):
        with patch.object(runner, 'load_v6_sources', side_effect=AssertionError('unexpected source read')):
            with self.assertRaises(SystemExit):
                self.invoke('--name', 'E116-v6-retry', '--eval-protocol', 'v6', '--generations', '0',
                            '--eval-worlds', '256', diagnostic=True)


if __name__ == '__main__':
    unittest.main()
