#!/usr/bin/env python3
"""Regression checks for signed affine Q2 levels and weighted-error acceptance."""
import unittest
import mlx.core as mx
from convert_kolibri_mlx import refit_q2
from refit_kolibri_mixed2 import refit


class RefitTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(19)
        self.weight = mx.random.normal((3, 8, 256)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(self.weight, group_size=128, bits=2)
        shifts = mx.arange(16, dtype=mx.uint32) * 2
        codes = ((packed[..., None] >> shifts) & 3).reshape(3, 8, 2, 128)
        flip = scales[..., None] > 0
        codes = mx.where(flip, 3 - codes, codes).astype(mx.uint32).reshape(3, 8, 16, 16)
        self.packed = (codes << shifts).sum(-1).astype(mx.uint32)
        self.biases = mx.where(scales > 0, biases + 3 * scales, biases).astype(mx.bfloat16)
        self.scales = -mx.abs(scales)
        self.moment = mx.exp(mx.random.normal((256,)))
        mx.eval(self.weight, self.packed, self.scales, self.biases, self.moment)
        self.assertTrue(mx.all(self.scales < 0).item())

    def error(self, packed, scales, biases, weighted):
        decoded = mx.dequantize(packed, scales, biases, group_size=128, bits=2).astype(mx.float32)
        squared = (self.weight.astype(mx.float32) - decoded) ** 2
        if weighted:
            squared *= mx.maximum(self.moment, self.moment.mean() * .001)
        return squared.reshape(3, 8, 2, 128).sum(-1)

    def test_unweighted_fit_preserves_or_improves_every_signed_group(self):
        before = self.error(self.packed, self.scales, self.biases, False)
        fitted = refit_q2(self.weight, self.packed, self.scales, self.biases, mx)
        self.assertTrue(mx.all(self.error(*fitted, False) <= before + 1e-5).item())

    def test_weighted_fit_preserves_or_improves_every_signed_group(self):
        before = self.error(self.packed, self.scales, self.biases, True)
        packed, scales, biases, fraction, ratio = refit(
            self.weight, self.packed, self.scales, self.biases, self.moment, mx)
        self.assertTrue(mx.all(self.error(packed, scales, biases, True) <= before + 1e-5).item())
        self.assertGreaterEqual(fraction, 0)
        self.assertLessEqual(ratio, 1 + 1e-6)

    def test_constant_groups_remain_finite(self):
        weight = mx.full((2, 128), 0.25, dtype=mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=2)
        packed, scales, biases, _, _ = refit(weight, packed, scales, biases, mx.ones((128,)), mx)
        decoded = mx.dequantize(packed, scales, biases, group_size=128, bits=2)
        self.assertTrue(mx.all(mx.isfinite(decoded)).item())
        self.assertLess(mx.max(mx.abs(decoded - weight)).item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
