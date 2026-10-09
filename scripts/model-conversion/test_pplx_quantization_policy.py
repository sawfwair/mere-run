#!/usr/bin/env python3
"""Policy checks that require no MLX, GPU, model files or network."""
import unittest
import tempfile
from pathlib import Path
from convert_pplx_embed_v2_mlx import checksums
from publish_pplx_embed_v2_mlx import verify_local_manifest
from convert_pplx_embed_v2_mlx import bits_for, MODELS


class PPLXQuantizationPolicyTests(unittest.TestCase):
    def test_all_32_transformer_layers_have_the_intended_profile(self):
        count=1
        self.assertEqual(bits_for('language_model.embed_tokens.weight',[248320,4096]),8)
        for layer in range(32):
            paths=['mlp.gate_proj','mlp.up_proj','mlp.down_proj']
            paths += (['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj']
                      if layer%4==3 else ['linear_attn.in_proj_qkv','linear_attn.in_proj_z','linear_attn.out_proj'])
            for path in paths:
                self.assertEqual(bits_for(f'language_model.layers.{layer}.{path}.weight',[4096,4096]),4)
                self.assertEqual(bits_for(f'language_model.layers.{layer}.{path}.weight',[4096,4096],'q8'),8)
                count+=1
        self.assertEqual(count,201)
        self.assertTrue(all(len(pin)==40 for _,pin in MODELS.values()))

    def test_sensitive_layers_and_incompatible_shapes_retain_fp32(self):
        for key in ['language_model.layers.0.linear_attn.in_proj_a.weight',
                    'language_model.layers.0.linear_attn.in_proj_b.weight',
                    'language_model.layers.0.linear_attn.conv1d.weight','language_model.norm.weight',
                    'visual.blocks.0.attn.qkv.weight','contextual_projection.weight','linear.weight']:
            self.assertIsNone(bits_for(key,[4096,4096]))
        self.assertIsNone(bits_for('language_model.embed_tokens.weight',[64,63]))
        self.assertIsNone(bits_for('language_model.layers.0.mlp.up_proj.weight',[32]))

    def test_publication_rejects_modified_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'model.safetensors').write_bytes(b'fixture')
            checksums(root)
            verify_local_manifest(root)
            (root/'model.safetensors').write_bytes(b'modified')
            with self.assertRaises(ValueError): verify_local_manifest(root)


if __name__=='__main__': unittest.main()
