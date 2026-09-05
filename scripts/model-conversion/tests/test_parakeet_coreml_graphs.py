"""Run with the converter's pinned Torch environment; no checkpoint required."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from parakeet_coreml_graphs import first_index_digits


class ParakeetANESelectionTests(unittest.TestCase):
    def test_matches_argmax_for_real_vocabulary_geometry(self):
        generator = torch.Generator().manual_seed(436)
        scores = torch.randn((16, 8, 8193), generator=generator).half()
        digits = first_index_digits(scores)
        actual = digits[..., 0].int() * 128 + digits[..., 1].int()
        self.assertTrue(torch.equal(actual, scores.argmax(dim=-1)))

    def test_preserves_first_index_ties_across_groups_and_blank(self):
        scores = torch.zeros((1, 5, 8193), dtype=torch.float16)
        scores[0, 1, [127, 128, 8192]] = 2
        scores[0, 2, [2049, 4095]] = 2
        scores[0, 3, 8191] = 2
        scores[0, 4, 8192] = 2
        digits = first_index_digits(scores)
        actual = digits[..., 0].int() * 128 + digits[..., 1].int()
        self.assertEqual(actual.tolist(), [[0, 127, 2049, 8191, 8192]])

    def test_duration_ties_and_negative_logits(self):
        scores = torch.tensor([[[-7, -3, -3, -4, -8], [-8, -7, -6, -5, -1]]]).half()
        digits = first_index_digits(scores)
        self.assertEqual(digits.tolist(), [[[0, 1], [0, 4]]])


if __name__ == "__main__":
    unittest.main()
