#!/usr/bin/env python3
"""Export small random-weight oracles from the pinned YuE2 source (no model download).

Requires torch==2.10.0, transformers==4.57.6, safetensors==0.7.0,
tiktoken==0.12.0, and numpy==2.2.6. Run on CPU.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REVISION = "0edaf2f4053ef4731334b8329834b107977f9637"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, help="Optional pinned qwen.tiktoken, used only to export reference IDs")
args = parser.parse_args()
revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
if revision != REVISION:
    raise SystemExit(f"Expected YuE source {REVISION}, got {revision}")
sys.path.insert(0, str(args.upstream / "src"))
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM, StaticKVCache
from yue2.modeling_vae import YuE2VAEConfig, YuE2VAE
from yue2.nar import CachedNAR, Chunk
from yue2.sampling import distribution
from yue2.protocol import Sampling, CODEC_OFFSET, CODEC_SIZE, MUSIC_END

torch.set_num_threads(1)
torch.manual_seed(831001)
args.output.mkdir(parents=True, exist_ok=True)
config = YuE2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                   num_key_value_heads=1, head_dim=8, intermediate_size=32,
                   vocab_size=32, max_position_embeddings=256, max_latent_frames=256)
model = YuE2ForCausalLM(config).eval()
(args.output / "model-config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
tokens = [1, 5, 9, 3, 7]
noise = torch.randn(6, 64)
expected = {"tokens": torch.tensor(tokens, dtype=torch.int32), "noise": noise}
if args.tokenizer:
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    text = "hello e\u0301 <abc>\n海の歌 🎵\n1234\tWe're here."
    ids = YuE2TextTokenizer(args.tokenizer).encode(text)
    (args.output / "tokenizer-reference.json").write_text(json.dumps(ids) + "\n")
with torch.inference_mode():
    for label, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        model.to(dtype)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(args.output / f"model-{label}.safetensors"))
        cache = StaticKVCache(num_layers=2, batch_size=1, num_kv_heads=1,
                              max_seq_len=32, head_dim=8, dtype=dtype, device="cpu")
        ids = torch.tensor([tokens])
        expected[f"{label}.prefill"] = model(ids, past_key_values=cache, use_cache=True).logits[0, -1].float()
        expected[f"{label}.cached"] = model(torch.tensor([[11]]), past_key_values=cache, use_cache=True).logits[0, -1].float()
        expected[f"{label}.uncached"] = model(torch.tensor([tokens + [11]]), use_cache=False).logits[0, -1].float()
        engine = CachedNAR(model, Chunk(tokens, noise))
        expected[f"{label}.velocity"] = engine.velocity(noise.to(dtype), 0.4).float()
        expected[f"{label}.midpoint"] = engine.solve(steps=3)
        engine.close()
        logits = torch.zeros(184704, dtype=dtype)
        logits[42] = 1000
        logits[CODEC_OFFSET:CODEC_OFFSET + 5] = torch.tensor([2.2345, 2.2234, 1.1, -2.3456, 0], dtype=dtype)
        logits[MUSIC_END] = 1.5
        scores = distribution(logits[None], Sampling(top_p=.6, top_k=4, min_tokens=0, max_tokens=32),
                              [CODEC_OFFSET, CODEC_OFFSET, CODEC_OFFSET + 1], 5, "semantic", label == "bf16")[0]
        expected[f"{label}.sampling"] = torch.cat([scores[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE], scores[MUSIC_END:MUSIC_END + 1]]).float()
    decoder_config = dict(channels=2, c_mults=[1, 2, 2, 2, 2, 2], strides=[2, 2, 4, 4, 5, 6],
                          latent_dim=64, out_channels=2, use_snake=True, snake_type="vanilla",
                          final_tanh=False, use_filter=False)
    vae_config = YuE2VAEConfig(decoder_config=decoder_config, decode_core_frames=8, decode_halo_frames=16)
    vae = YuE2VAE(vae_config, decoder_only=True)
    save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(args.output / "decoder.safetensors"))
    (args.output / "decoder-config.json").write_text(json.dumps(vae_config.to_dict(), indent=2) + "\n")
    latents = torch.randn(1, 64, 35)
    expected["decoder.latents"] = latents[0].T.contiguous()
    expected["decoder.full"] = vae.decode(latents)[0].T.contiguous()
    expected["decoder.tiled"] = vae.decode_tiled(latents)[0].T.contiguous()
    save_file({k: v.contiguous() for k, v in expected.items()}, str(args.output / "expected.safetensors"))
    (args.output / "provenance.json").write_text(json.dumps(dict(
        source="https://github.com/multimodal-art-projection/YuE", revision=REVISION,
        torch=torch.__version__, seed=831001, trained_weights=False,
        decoder_halo=vae.required_halo(), decoder_length=vae.natural_output_length(35),
        max_tiled_difference=(expected["decoder.full"] - expected["decoder.tiled"]).abs().max().item(),
    ), indent=2) + "\n")
print(args.output)
