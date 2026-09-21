#!/usr/bin/env python3
"""Regenerate small CPU reference fixtures; never runs in the native runtime.

Requires torch==2.14.0, transformers==5.17.0, torchvision==0.29.0,
safetensors, Pillow, numpy, and Diffusers pinned to
8d3c30bfda9b511c00992f40cff4170a5502814d. No model checkpoints are downloaded.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Transformer2DModel, QwenImage21KVCache,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parents[2] / 'Tests/ImageRuntimeTests/Fixtures/QwenImage21'
ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(21)
config = dict(patch_size=1, in_channels=4, out_channels=4, num_layers=2,
              attention_head_dim=6, num_attention_heads=2, context_in_dim=8,
              mlp_ratio=2, axes_dims_rope=[2, 2, 2], eps=1e-6, causal_condition=True)
model = QwenImage21Transformer2DModel(**config).eval()
# Nontrivial shared modulation, normalizations, and projections; use model initialization.
text = torch.randn(1, 5, 8)
latents = torch.randn(1, 12, 4)
# Two adjacent references, one intervening text token, and a target image.
mask = torch.tensor([[False, True, True, False, False, True]])
shapes = [[(1, 2, 2), (1, 2, 2), (1, 2, 2)]]
with torch.no_grad():
    full = model(hidden_states=latents, encoder_hidden_states=text, timestep=torch.tensor([0.75]),
                 img_mask=mask, img_shapes=shapes).sample[:, -4:]
    cache = QwenImage21KVCache(2)
    model(hidden_states=latents, encoder_hidden_states=text, timestep=torch.tensor([0.75]),
          img_mask=mask, img_shapes=shapes, kv_cache=cache, kv_cache_mode='extract')
    changed = latents.clone(); changed[:, -4:] += 0.125
    cached = model(hidden_states=changed, encoder_hidden_states=text, timestep=torch.tensor([0.4]),
                   img_mask=mask, img_shapes=shapes, kv_cache=cache, kv_cache_mode='cached').sample[:, -4:]
    uncached = model(hidden_states=changed, encoder_hidden_states=text, timestep=torch.tensor([0.4]),
                     img_mask=mask, img_shapes=shapes).sample[:, -4:]
    torch.testing.assert_close(cached, uncached, atol=1e-6, rtol=1e-5)
save_file(dict(model.state_dict()), str(ROOT / 'transformer.safetensors'))
save_file({'text': text, 'latents': latents, 'full': full.contiguous(), 'cached': cached.contiguous()}, str(ROOT / 'transformer-results.safetensors'))
(ROOT / 'transformer-config.json').write_text(json.dumps(config, indent=2) + '\n')
vae_config = dict(base_dim=2, decoder_base_dim=2, z_dim=2, dim_mult=[1,2,4,8,8],
                  num_res_blocks=1, attn_scales=[], temperal_downsample=[False,True,True,True],
                  in_channels=4, out_channels=4, is_residual=True, patch_size=None,
                  scale_factor_spatial=16, latents_mean=[0.25,-0.5], latents_std=[1.5,2.0])
vae = AutoencoderKLQwenImage21(**vae_config).eval()
pixels = torch.randn(1,4,1,32,32).clamp(-1,1)
z = torch.randn(1,2,1,2,2)
with torch.no_grad():
    mean = torch.tensor(vae_config['latents_mean']).view(1,2,1,1,1)
    std = torch.tensor(vae_config['latents_std']).view(1,2,1,1,1)
    encoded = (vae.encode(pixels).latent_dist.mode() - mean) / std
    decoded = (vae.decode(z * std + mean).sample + 1) / 2
save_file(dict(vae.state_dict()), str(ROOT / 'vae.safetensors'))
save_file({'pixels': pixels, 'latents': z, 'encoded': encoded, 'decoded': decoded}, str(ROOT / 'vae-results.safetensors'))
(ROOT / 'vae-config.json').write_text(json.dumps(vae_config, indent=2) + '\n')
scheduler_config = dict(base_image_seq_len=256,max_image_seq_len=8192,base_shift=.5,max_shift=.9,
                        shift_terminal=.02,use_dynamic_shifting=True,time_shift_type='exponential',
                        invert_sigmas=False,stochastic_sampling=False,use_beta_sigmas=False,
                        use_exponential_sigmas=False,use_karras_sigmas=False)
scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
mu = .5 + (4096-256)*(.9-.5)/(8192-256)
scheduler.set_timesteps(sigmas=np.linspace(1,1/40,40),mu=mu)
(ROOT / 'scheduler.json').write_text(json.dumps({'config':scheduler_config,'sigmas':scheduler.sigmas.tolist()},indent=2)+'\n')
rgba = np.arange(7*5*4,dtype=np.uint8).reshape(5,7,4)
resized = np.asarray(Image.fromarray(rgba).resize((4,8),Image.Resampling.LANCZOS))
(ROOT / 'resize.json').write_text(json.dumps({'input':rgba.flatten().tolist(),'output':resized.flatten().tolist()},indent=2)+'\n')
(ROOT / 'provenance.json').write_text(json.dumps({'diffusers_revision':'8d3c30bfda9b511c00992f40cff4170a5502814d',
    'seed':21,'torch':torch.__version__,'transformers':'5.17.0',
    'checkpoint_header_revision':'b3179ad355be050328e483a9dfdd9e60cd62adfa','scope':'tiny random CPU float32 models; no real checkpoint or quality claims'},indent=2)+'\n')
print('Wrote reference fixtures to',ROOT)

# Independent Qwen3-VL vision/deepstack/final-activation fixture (transformers==5.17.0).
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from safetensors.torch import save_file
from pathlib import Path
root = ROOT
torch.manual_seed(21)
torch.set_num_threads(1)
config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,rope_theta=5000000,rope_scaling={'rope_type':'default','mrope_section':[2,1,1],'mrope_interleaved':True}),vision_config=dict(depth=2,hidden_size=16,intermediate_size=32,num_heads=2,patch_size=16,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=16,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=101,vision_start_token_id=100,vision_end_token_id=102,video_token_id=103)
model=Qwen3VLForConditionalGeneration(config).eval()
pixels=torch.randn(1,3,32,32)
# Qwen image processor patch order: merged block, local patch row/column, channel, temporal, patch row/column.
patches=pixels.unsqueeze(2).repeat(1,1,2,1,1).reshape(1,3,2,1,2,16,1,2,16).permute(0,3,6,4,7,1,2,5,8).reshape(4,-1)
ids=torch.tensor([[1,100,101,102,7,9]])
handle=model.model.language_model.norm.register_forward_hook(lambda module,args,output:args[0])
with torch.no_grad():
 result=model(input_ids=ids,mm_token_type_ids=(ids==101).long(),attention_mask=torch.ones_like(ids),pixel_values=patches,image_grid_thw=torch.tensor([[1,2,2]]),output_hidden_states=True).hidden_states[-1]
handle.remove()
save_file(dict(model.state_dict()),str(root/'text-encoder.safetensors'))
visual=model.model.visual(patches,grid_thw=torch.tensor([[1,2,2]]))
save_file({'pixels':pixels,'ids':ids.to(torch.int32),'expected':result,'patches':patches,'visual':visual.pooler_output,'deep0':visual.deepstack_features[0],'deep1':visual.deepstack_features[1]},str(root/'text-results.safetensors'))
print('text fixture',result.shape)

# BF16 time embedding must round the input timestep before the sinusoidal projection.
from safetensors.torch import load_file
bf_config = json.loads((ROOT / 'transformer-config.json').read_text())
bf_model = QwenImage21Transformer2DModel(**bf_config).eval()
bf_model.load_state_dict(load_file(str(ROOT / 'transformer.safetensors')))
bf_model = bf_model.to(torch.bfloat16)
bf_inputs = load_file(str(ROOT / 'transformer-results.safetensors'))
with torch.no_grad():
    bf_output = bf_model(hidden_states=bf_inputs['latents'].to(torch.bfloat16),
        encoder_hidden_states=bf_inputs['text'].to(torch.bfloat16), timestep=torch.tensor([0.413725]),
        img_mask=torch.tensor([[False,True,True,False,False,True]]),
        img_shapes=[[(1,2,2),(1,2,2),(1,2,2)]]).sample[:, -4:]
save_file({'expected':bf_output.contiguous()},str(ROOT / 'transformer-bf16-results.safetensors'))
