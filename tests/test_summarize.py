"""Keep protocol labels and failures visible in generated results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import summarize


class SummaryTests(unittest.TestCase):
    def test_separate_structural_conditions_and_failed_runs_preserve_v1_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = dict(start='blank', no_plastic=False, generations=0, pop=2, worlds=2, sigma=0.1, lr=0.1)
            result = dict(fitness=1., acc_first=0.5, acc_second=0.6, survived=0.5, eaten=10)
            for name, structural, summary in (
                ('old', False, dict(status='finished', final_learning=result, final_frozen=result)),
                ('structural', True, dict(status='finished', conditions={'Weight-only': result, 'Weight+structure': result, 'Frozen': result})),
                ('failed', True, dict(status='failed', error='validation failure')),
            ):
                directory = root / 'runs' / name
                directory.mkdir(parents=True)
                config = dict(args=dict(args, **({'eval_protocol': 'structural-v1'} if structural else {})),
                              git=dict(commit='abcdef0123456789', dirty=structural))
                (directory / 'config.json').write_text(json.dumps(config))
                (directory / 'summary.json').write_text(json.dumps(summary))
            with patch.object(summarize, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            text = (root / 'results' / 'RESULTS.md').read_text()
            v1, structural = text.split('## Structural v1')
            self.assertIn('| old | finished | blank | yes |', v1)
            self.assertNotIn('| structural |', v1)
            for label in ('Weight-only', 'Weight+structure', 'Frozen'):
                self.assertIn('| structural | finished | ' + label + ' |', structural)
            self.assertIn('| failed | failed |', structural)
            self.assertEqual(structural.count('| abcdef0* |'), 4)
            self.assertIn('| abcdef0 |', v1)

    def test_four_protocols_remain_separate_with_string_and_dict_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = dict(start='blank', no_plastic=False, generations=0, pop=2, worlds=2, sigma=0.1, lr=0.1)
            result = dict(fitness=1., acc_first=0.5, acc_second=0.6, survived=0.5, eaten=10)
            panels = (
                ('legacy', None, None, dict(status='finished', generation=0, final_learning=result, final_frozen=result)),
                ('diagnostic', 'v2', 'v2', dict(status='finished', eval_seed=123, eval_worlds=128,
                                               evaluations={k: result for k in ('both', 'hebbian_only', 'frozen')})),
                ('structure', 'structural-v1', None, dict(status='finished', conditions={'Weight+structure': result})),
                ('broader', 'broader-v1', dict(name='broader-v1', split='final', seed=8109, n_worlds=128),
                 dict(status='finished', condition_evaluations={'control': dict(learning=result, frozen=result,
                                                                              learning_minus_frozen_fitness=0.)})),
                ('broader-failed', 'broader-v1', dict(name='broader-v1', split='development', seed=8108, n_worlds=1),
                 dict(status='failed', error='nonfinite')),
            )
            for name, protocol, metadata, summary in panels:
                directory = root / 'runs' / name
                directory.mkdir(parents=True)
                config = dict(args=dict(args), git=dict(commit='abcdef0123456789', dirty=protocol is not None))
                if protocol is not None:
                    config['args']['eval_protocol'] = protocol
                if metadata is not None:
                    config['eval_protocol'] = metadata
                (directory / 'config.json').write_text(json.dumps(config))
                (directory / 'summary.json').write_text(json.dumps(summary))
            with patch.object(summarize, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()) as output:
                summarize.main()
            text = (root / 'results' / 'RESULTS.md').read_text()
            v1, remaining = text.split('## EVAL v2 diagnostics')
            v2, remaining = remaining.split('## Structural v1 (separate diagnostic)')
            structural, broader = remaining.split('## BROADER v1 (separate parameter-shift protocol)')
            for section, names in ((v1, ('legacy',)), (v2, ('diagnostic',)),
                                   (structural, ('structure',)), (broader, ('broader', 'broader-failed'))):
                for name, *_ in panels:
                    self.assertEqual(f'| {name} |' in section, name in names, name)
            self.assertIn('| legacy | finished | blank | yes | 0/0 | 2×2 | 0.1/0.1 |  |  | 1.00 | 1.00 | 0.50 → 0.60 | abcdef0 |', v1)
            self.assertEqual(broader.count('| abcdef0* |'), 2)
            self.assertIn('| broader-failed | failed | development | 8108 | 1 | pending |', broader)
            self.assertIn('with 5 runs', output.getvalue())


if __name__ == '__main__':
    unittest.main()
