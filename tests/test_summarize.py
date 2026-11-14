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
                              git=dict(commit='abcdef0123456789', dirty=False))
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


if __name__ == '__main__':
    unittest.main()
