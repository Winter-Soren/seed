"""Real tiny structural ES: split/resume and flushed-tail crash/replay."""
import contextlib
import functools
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_experiment as runner


class StructuralReplayTests(unittest.TestCase):
    def test_real_structural_es_resumes_and_replays_exactly(self):
        scratch = runner.ROOT / '.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            factory = functools.lru_cache(None)(runner.make_evaluator)
            real_step = runner.es_step
            real_test = runner.test_genome
            now = 0.
            crash_at = None

            def step(*args):
                nonlocal now
                if args[0]['t'] == crash_at:
                    raise RuntimeError('simulated hard stop')
                state, row = real_step(*args)
                self.assertIn('structure', row)
                self.assertGreater(row['structure']['added'], 0.)
                now += 2.
                return state, row

            def invoke(*args):
                with (patch.object(runner, 'ROOT', root),
                      patch.object(runner, 'git_info', return_value=dict(commit='fixture', dirty=False)),
                      patch.object(runner, 'make_evaluator', side_effect=factory),
                      patch.object(runner, 'es_step', side_effect=step),
                      patch.object(runner, 'test_genome', side_effect=lambda *a, **kw: real_test(*a, **dict(kw, n_worlds=2))),
                      patch.object(runner.time, 'monotonic', side_effect=lambda: now),
                      patch.object(runner.sys, 'argv', ['runner', *args]),
                      contextlib.redirect_stdout(io.StringIO())):
                    runner.main()

            common = ['--generations', '4', '--pop', '2', '--worlds', '2', '--eval-worlds', '2',
                      '--checkpoint-every', '2', '--structural-plasticity', 'activity-grow-prune-v1',
                      '--eval-protocol', 'structural-v1', '--cfg', 'G=4', '--cfg', 'M=1', '--cfg', 'R=0',
                      '--cfg', 'H=3', '--cfg', 'T=22', '--cfg', 'n_items=2']
            directory = lambda name: next((root / 'runs').glob('*_' + name))
            invoke('--name', 'whole', *common)
            invoke('--name', 'split', *common, '--max-train-seconds', '4')
            split = directory('split')
            self.assertEqual(runner.load_checkpoint(split)['t'], 2)
            invoke('--resume', str(split))

            crash_at = 3
            with self.assertRaisesRegex(RuntimeError, 'simulated hard stop'):
                invoke('--name', 'crash', *common)
            crash = directory('crash')
            self.assertEqual(runner.load_checkpoint(crash)['t'], 2)
            original = (crash / 'metrics.jsonl').read_bytes()
            self.assertEqual(len(original.splitlines()), 3)
            crash_at = None
            invoke('--resume', str(crash))
            self.assertTrue((crash / 'metrics.jsonl').read_bytes().startswith(original))
            self.assertEqual(json.loads((crash / 'config.json').read_text())['resumes'][-1]['replayed_generations'], 1)

            rows = lambda d: [{k: v for k, v in json.loads(line).items() if k != 'seconds'}
                              for line in (d / 'metrics.jsonl').read_text().splitlines()]
            whole = directory('whole')
            for candidate in (split, crash):
                for k, value in runner.load_checkpoint(whole).items():
                    np.testing.assert_array_equal(value, runner.load_checkpoint(candidate)[k], err_msg=k)
                self.assertEqual(rows(whole), rows(candidate))
                np.testing.assert_array_equal(np.load(whole / 'genome_final.npy'), np.load(candidate / 'genome_final.npy'))


if __name__ == '__main__':
    unittest.main()
