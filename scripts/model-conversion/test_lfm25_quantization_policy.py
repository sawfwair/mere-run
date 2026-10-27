#!/usr/bin/env python3
"""Protect the high-quality mixed policy from quantizing sensitive components."""
import unittest
from convert_lfm25_24b_mixed_mlx import PROFILES,policy
class QuantizationPolicyTests(unittest.TestCase):
 def test_sensitive_components_remain_in_original_precision(self):
  paths=['model.embed_tokens','model.embedding_norm','model.layers.0.feed_forward.gate_proj',
         'model.layers.2.feed_forward.gate','model.layers.2.self_attn.q_proj','model.layers.3.conv.in_proj']
  for profile in PROFILES:
   for path in paths:
    with self.subTest(profile=profile,path=path): self.assertIsNone(policy(profile,path))
 def test_only_down_projection_uses_eight_bits_in_down8_profile(self):
  prefix='model.layers.2.feed_forward.switch_mlp.'
  self.assertEqual(policy('mixed6-down8',prefix+'gate_proj')['bits'],6)
  self.assertEqual(policy('mixed6-down8',prefix+'up_proj')['bits'],6)
  self.assertEqual(policy('mixed6-down8',prefix+'down_proj')['bits'],8)
 def test_reference_has_no_quantization(self):
  self.assertIsNone(policy('reference','model.layers.2.feed_forward.switch_mlp.gate_proj'))
 def test_group_precision_is_explicit(self):
  path='model.layers.2.feed_forward.switch_mlp.up_proj'
  self.assertEqual(policy('mixed6-group32',path),dict(bits=6,group_size=32,mode='affine'))
  self.assertEqual(policy('mixed6-group64',path),dict(bits=6,group_size=64,mode='affine'))
 def test_quality_policy_has_twelve_eight_bit_banks_and_twenty_six_six_bit_banks(self):
  selected=[policy('mixed6-8-quality',f'model.layers.{layer}.feed_forward.switch_mlp.gate_proj') for layer in range(2,40)]
  self.assertEqual(sum(q['bits']==8 for q in selected),12)
  self.assertEqual(sum(q['bits']==6 for q in selected),26)
  self.assertTrue(all(q['group_size']==32 for q in selected))
 def test_unknown_profile_is_rejected(self):
  with self.assertRaises(ValueError): policy('unknown','model.embed_tokens')
if __name__=='__main__': unittest.main()
