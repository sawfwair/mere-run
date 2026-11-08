import json
from pathlib import Path
import tempfile
import unittest
from compare_glimmer_quantization import compare


class ComparisonTests(unittest.TestCase):
    def test_matched_scores_pass_but_degradation_and_misalignment_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = [dict(kind='likelihood', modelRoot='BF16', id=str(i),
                             tokens=[i] * 2049, nll=[2.] * 2048, top1=[i] * 2048) for i in range(16)]
            candidate = [dict(r, modelRoot='Q8', bits=8) for r in baseline]
            def write(name, rows):
                path = root / name
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                return path
            ref = write('reference', baseline)
            result = compare(ref, write('candidate', candidate))
            self.assertTrue(result['passes_likelihood_diagnostic'])
            self.assertEqual(result['perplexity_ratio'], 1)
            self.assertEqual(result['ratio_95_percent_window_bootstrap'], [1, 1])
            self.assertEqual(result['scored_tokens'], 32768)
            degraded = [dict(r, nll=[2.1] * 2048) for r in candidate]
            self.assertFalse(compare(ref, write('degraded', degraded))['passes_likelihood_diagnostic'])
            candidate[0]['tokens'] = [999] * 2049
            with self.assertRaisesRegex(ValueError, 'exactly the same'):
                compare(ref, write('mismatched', candidate))
            with self.assertRaisesRegex(ValueError, '16 complete'):
                compare(ref, write('partial', baseline[:-1]))


if __name__ == '__main__':
    unittest.main()
