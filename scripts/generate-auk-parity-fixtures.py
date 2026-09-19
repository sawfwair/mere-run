#!/usr/bin/env python3
"""Regenerate bounded AuK fixtures from a pinned upstream MLX source checkout.

Requires mlx==0.32.2 and numpy; no model downloads. Pass the upstream checkout
at revision 6943a1e967409e8c73139a7a345f2a611cfb3dd6 as the only argument.
"""
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

upstream = Path(sys.argv[1]).resolve()
revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
assert revision == "6943a1e967409e8c73139a7a345f2a611cfb3dd6"
sys.path.insert(0, str(upstream / "src"))
from auk_mlx.dit import DiTConfig, Flux2Edit
from auk_mlx.layers import Activation1d
from auk_mlx.vae import BigVGANFlowVAE, VAEConfig
from auk_mlx.qwen_thinker import AudioConfig, TextConfig, ThinkerEncoder

out = Path(__file__).resolve().parents[1] / "Tests/H3RuntimeTests/Fixtures/AuK"
out.mkdir(parents=True, exist_ok=True)
mx.random.seed(123)
# Deliberately non-analytic rotary frequencies catch accidental recomputation.
freq = np.array([1.0, 0.125], dtype=np.float32)
dit = Flux2Edit(DiTConfig(dim=16, heads=4, dim_head=4, text_hidden_dim=12,
                        latent_dim=8, num_layers=1, num_single_layers=1), inv_freq=freq)
# Upstream custom convolutions initialize to zero; use deterministic nonzero
# tensors so positional convolution mistakes cannot pass unnoticed.
for conv in dit.audio_embed.conv_pos_embed.conv1d:
    conv.weight = mx.random.normal(conv.weight.shape) * 0.03
mx.save_safetensors(str(out / "dit.safetensors"), dict(tree_flatten(dit.parameters())))
x = mx.random.normal((1, 5, 8))
text = mx.random.normal((1, 3, 12))
ref = mx.random.normal((1, 2, 8))
fixtures = {"latent": x, "text": text, "reference_input": ref, "inv_freq": mx.array(freq)}
for name, reference, cfg in [("plain", None, False), ("reference", ref, False), ("guided", ref, True)]:
    y = dit(x, text, mx.array([0.3]), ref=reference, cfg_infer=cfg, cache=False)
    if cfg:
        y = y[0:1] + 2 * (y[0:1] - y[1:2])
    fixtures[name] = y
act = Activation1d(3, causal=True)
act.act.alpha = mx.array([0.1, -0.2, 0.3])
act.act.beta = mx.array([-0.1, 0.2, 0.4])
wave = mx.random.normal((1, 17, 3))
fixtures.update({"act_input": wave, "act_output": act(wave), "act_alpha": act.act.alpha, "act_beta": act.act.beta})
tc = TextConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=32, vocab_size=48)
ac = AudioConfig(d_model=16, encoder_layers=2, encoder_attention_heads=4,
                 encoder_ffn_dim=32, output_dim=16, n_window=4, num_mel_bins=4)
thinker = ThinkerEncoder(tc, ac)
mx.save_safetensors(str(out / "thinker.safetensors"), dict(tree_flatten(thinker.parameters())))
mel = mx.random.normal((1, 13, 4))
ids = mx.array([[1, 2, 40, 40, 40, 3]])
fusion = mx.array([0.25, -0.4])
for name, audio in [("thinker_text", None), ("thinker_audio", mel)]:
    hs = thinker(ids, audio_features=audio, audio_token_mask=ids == 40 if audio is not None else None)
    stacked = mx.stack([mx.fast.layer_norm(h, None, None, 1e-5) for h in hs[1:]])
    fixtures[name] = (stacked * mx.softmax(fusion)[:, None, None, None]).sum(axis=0) * 1.7
fixtures.update({"mel": mel, "ids": ids, "layer_weights": fusion})
vae = BigVGANFlowVAE(VAEConfig(upsample_initial_channel=64,
    downsample_channels=[2, 2, 4, 4, 8, 8, 16]))
params = {}
for name, value in tree_flatten(vae.parameters()):
    if name.endswith(".weight"):
        params[name] = mx.random.normal(value.shape) * 0.03
    elif name.endswith(".bias"):
        params[name] = mx.random.normal(value.shape) * 0.01
    else:
        params[name] = value
vae.load_weights(list(params.items()))
mx.save_safetensors(str(out / "vae.safetensors"), params)
wave = mx.random.normal((1, 1920, 1)) * 0.1
latent = mx.random.normal((1, 2, 64))
fixtures.update({"vae_input": wave, "vae_encoded": vae.encode(wave),
                 "vae_latent": latent, "vae_decoded": vae.decode(latent)})
mx.save_safetensors(str(out / "expected.safetensors"), fixtures)
(out / "provenance.json").write_text(json.dumps({"upstreamRevision": revision, "mlx": "0.32.2", "seed": 123,
    "scope": "Random small-model component parity; not trained-checkpoint or speech-quality evidence."}, indent=2) + "\n")
print(out)
