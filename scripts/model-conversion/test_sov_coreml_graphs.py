#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.7.0", "numpy==2.3.5"]
# ///
"""Independent canonical-operator tests for SOV's conversion layout rewrites."""
import unittest
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from convert_sov_coreml import attention_kernel, channel_layer_norm, feedforward_kernel
from prepare_sov_coreml_calibration import audio_features
from profile_sov_ane_kernels import partition_feedforward_weights, rescale_norm_projection


class SOVCoreMLLayoutTests(unittest.TestCase):
    def test_query_tiles_preserve_all_keys_and_head_order(self):
        torch.manual_seed(17)
        q, k, v = [torch.randn(2, 3, 4, 7) for _ in range(3)]
        expected = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        expected = expected.transpose(-1, -2).reshape(2, 12, 1, 7)
        for layout in ["qk", "channel", "sdpa", "einsum", "packed", "packed-streaming", "direct", "direct-streaming", "batch", "batch-streaming"]:
            for group in [1, 2, 3]:
                for chunk in [1, 3, 7, 16]:
                    with self.subTest(layout=layout, group=group, chunk=chunk):
                        actual = attention_kernel(q, k, v, group, chunk, layout)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_late_key_affects_early_query_across_tile_boundaries(self):
        q = torch.ones(1, 2, 4, 7)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[..., -1] = 7
        actual = attention_kernel(q, k, v, 1, 3, "channel")
        torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_channel_normalization_matches_affine_layer_norm(self):
        torch.manual_seed(31)
        x = torch.randn(2, 8, 3, 5) * 3 + 2
        weight, bias = torch.randn(8), torch.randn(8)
        expected = F.layer_norm(x.permute(0, 2, 3, 1), (8,), weight, bias, 1e-5).permute(0, 3, 1, 2)
        for scale in [1.0, 8.0]:
            for layout in ["centered", "instance", "moments"]:
                actual = channel_layer_norm(x, weight, bias, scale=scale, layout=layout)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_online_softmax_retains_full_context_with_key_and_query_tails(self):
        torch.manual_seed(53)
        q, k, v = [torch.randn(2, 3, 4, 7) * 3 for _ in range(3)]
        expected = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        expected = expected.transpose(-1, -2).reshape(2, 12, 1, 7)
        for query_chunk in [1, 3, 8]:
            for key_chunk in [1, 3, 7, 9]:
                actual = attention_kernel(q, k, v, 1, query_chunk, "streaming", key_chunk)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_channel_tiled_ffn_retains_bias_and_tail_channels(self):
        torch.manual_seed(71)
        x = torch.randn(2, 4, 1, 7)
        first, first_bias = torch.randn(9, 4), torch.randn(9)
        second, second_bias = torch.randn(4, 9), torch.randn(4)
        expected = F.linear(F.gelu(F.linear(x.permute(0, 2, 3, 1), first, first_bias)), second, second_bias).permute(0, 3, 1, 2)
        for token_chunk in [1, 3, 8]:
            for channel_chunk in [1, 4, 9, 12]:
                for schedule in ["tokens", "channels"]:
                    actual = feedforward_kernel(x, first, first_bias, second, second_bias, token_chunk, channel_chunk, schedule)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_aligned_padding_masks_arbitrary_extra_keys(self):
        torch.manual_seed(79)
        q, k, v = [torch.randn(2, 3, 4, 7) for _ in range(3)]
        expected = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        expected = expected.transpose(-1, -2).reshape(2, 12, 1, 7)
        q, k, v = [torch.cat([x, torch.randn(2, 3, 4, 1)*1000], dim=-1) for x in [q, k, v]]
        for layout in ["qk", "channel", "sdpa", "einsum", "streaming", "packed", "packed-streaming", "direct-streaming", "batch-streaming"]:
            for group in [1, 2, 3]:
                for chunk in [3, 4, 8]:
                    actual = attention_kernel(q, k, v, group, chunk, layout, key_chunk_size=3, valid_frames=7)
                    torch.testing.assert_close(actual[..., :7], expected, rtol=1e-4, atol=1e-5)

    def test_balancing_norm_and_projection_preserves_nonzero_affine_bias(self):
        torch.manual_seed(89)
        x = torch.randn(2, 7, 8)
        gamma, beta, weight, bias = torch.randn(8), torch.randn(8), torch.randn(12, 8), torch.randn(12)
        scale = torch.exp(torch.randn(8))
        expected = F.linear(F.layer_norm(x, (8,), gamma, beta), weight, bias)
        balanced_gamma, balanced_beta, balanced_weight = rescale_norm_projection(gamma, beta, weight, scale)
        actual = F.linear(F.layer_norm(x, (8,), balanced_gamma, balanced_beta), balanced_weight, bias)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_outlier_neuron_partition_preserves_ffn_and_adds_bias_once(self):
        torch.manual_seed(97)
        x = torch.randn(2, 7, 4)
        first, first_bias = torch.randn(9, 4), torch.randn(9)
        second, second_bias = torch.randn(4, 9), torch.randn(4)
        expected = F.linear(F.gelu(F.linear(x, first, first_bias)), second, second_bias)
        branches, _ = partition_feedforward_weights(first, first_bias, second, second_bias, torch.tensor([8, 0, 4]))
        values = []
        for name in ["regular", "outlier"]:
            hidden = F.gelu(F.linear(x, branches[f"fc1_{name}.weight"], branches[f"fc1_{name}.bias"]))
            values.append(F.linear(hidden, branches[f"fc2_{name}.weight"], branches[f"fc2_{name}.bias"]))
        torch.testing.assert_close(values[0]+values[1], expected, rtol=1e-5, atol=1e-5)

    def test_calibration_frontend_matches_independent_numpy_global_window_fixture(self):
        samples = torch.zeros(960000)
        t = torch.arange(1600) / 16000
        samples[:1600] = 0.1 * torch.sin(2 * math.pi * 440 * t) + 0.03 * torch.cos(2 * math.pi * 1234 * t)
        t = torch.arange(160) / 16000
        samples[480000:480160] = 0.8 * torch.sin(2 * math.pi * 700 * t)
        actual = audio_features(samples)
        fixture = Path(__file__).resolve().parents[2] / "Tests/MereRunCoreTests/Fixtures/SOV/frontend.json"
        for point in json.loads(fixture.read_text()):
            self.assertAlmostEqual(float(actual[point["window"], point["channel"], point["frame"]]), point["value"], delta=0.0003)


if __name__ == "__main__":
    unittest.main()
