"""Mathematical checks for the expanded export; run with its pinned environment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("graph", ROOT / "embeddinggemma2_export_text.py")
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)
GRAPH.PRECISION_MODE = "fp32"


class ExpandedAttentionTests(unittest.TestCase):
    def test_local_mask_boundary_and_padding_are_both_applied(self):
        model = SimpleNamespace(one=torch.tensor(1.0), zero=torch.tensor(0.0),
                                mask_value=torch.tensor(-10000.0), window=torch.tensor(512))
        mask = torch.ones(1, 1024)
        mask[:, 1000:] = 0
        full, local = GRAPH.EmbeddingGemma2Text._biases(model, mask, 1024)
        self.assertEqual(local[0, 0, 0, 512].item(), 0)
        self.assertLess(local[0, 0, 0, 513].item(), -9000)
        self.assertEqual(local[0, 0, 512, 0].item(), 0)  # bidirectional, not causal
        self.assertLess(local[0, 0, 513, 0].item(), -9000)
        self.assertEqual(full[0, 0, 0, 999].item(), 0)
        self.assertLess(full[0, 0, 0, 1000].item(), -9000)
        self.assertLess(local[0, 0, 999, 1000].item(), -9000)

    def test_fused_attention_preserves_unit_scale_for_gqa_and_mqa(self):
        torch.manual_seed(42)
        for kv_heads in (1, 2):
            cfg = SimpleNamespace(hidden=8, n_heads=4, head_dim=[4], kv_heads=[kv_heads], eps=1e-6)
            attention = GRAPH.Attention(cfg, 0).float().eval()
            x = torch.randn(1, 17, 8)
            cos = torch.ones(1, 1, 17, 4)
            sin = torch.zeros_like(cos)
            bias = torch.zeros(1, 1, 17, 17)
            bias[:, :, :, -3:] = -10000
            with torch.inference_mode():
                actual = attention(x, cos, sin, bias)
                q = attention.q_norm(attention.q_proj(x).view(1, 17, 4, 4).transpose(1, 2))
                k = attention.k_norm(attention.k_proj(x).view(1, 17, kv_heads, 4).transpose(1, 2))
                v = attention.v_norm(attention.v_proj(x).view(1, 17, kv_heads, 4).transpose(1, 2))
                k = k.repeat_interleave(4 // kv_heads, 1)
                v = v.repeat_interleave(4 // kv_heads, 1)
                weights = torch.softmax(q @ k.transpose(-1, -2) + bias, -1)
                expected = attention.o_proj((weights @ v).transpose(1, 2).reshape(1, 17, 16))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
