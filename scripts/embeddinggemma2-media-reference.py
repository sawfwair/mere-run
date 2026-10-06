#!/usr/bin/env python3
"""Regenerate synthetic Gemma 4 media tower fixtures using Transformers 5.19.0.

No released checkpoint weights are used. Run from the repository root in a
Python environment with torch and transformers; CPU float32 and SDPA are used.
"""
import torch,json,math
from pathlib import Path
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel,Gemma4AudioModel
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig,Gemma4AudioConfig
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder
from transformers import EmbeddingGemma2TextConfig
torch.set_num_threads(1)
out=Path('Tests/MereRunCoreTests/Fixtures/EmbeddingGemma2/media.json')
def values(name,t):
 phase=sum(name.encode())%11
 if name.endswith(('input_min','output_min')): return torch.full_like(t,-.75)
 if name.endswith(('input_max','output_max')): return torch.full_like(t,.75)
 v=torch.arange(t.numel(),dtype=torch.float32).reshape(t.shape)
 if 'norm' in name and name.endswith('weight'): return 1+(v%5-2)*.03125
 return torch.sin(v*.43+phase*.27)*(.01 if name.endswith('bias') else .04)
def prepare(model,embedder,prefix,eprefix):
 weights={prefix+'.'+k:values(prefix+'.'+k,v) for k,v in model.state_dict().items()}
 ew={eprefix+'.'+k:values(eprefix+'.'+k,v) for k,v in embedder.state_dict().items()}
 model.load_state_dict({k.removeprefix(prefix+'.'):v for k,v in weights.items()})
 embedder.load_state_dict({k.removeprefix(eprefix+'.'):v for k,v in ew.items()})
 return {**weights,**ew}
def packed(ws):return {k:dict(shape=list(v.shape),values=v.flatten().tolist()) for k,v in ws.items()}
text=EmbeddingGemma2TextConfig(hidden_size=8)
v=Gemma4VisionConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,global_head_dim=4,patch_size=2,pooling_kernel_size=2,position_embedding_size=32,use_clipped_linears=False,standardize=False)
v._attn_implementation='sdpa'
vm=Gemma4VisionModel(v).eval();ve=EmbeddingGemma2MultimodalEmbedder(v,text).eval();vw=prepare(vm,ve,'vision_tower','embed_vision')
pos=torch.tensor([[[x,y] for y in range(2) for x in range(4)]+[[-1,-1]]*8]);pix=(torch.arange(16*12).reshape(1,16,12)%31).float()*.025
with torch.inference_mode(): vision=ve(vm(pixel_values=pix,pixel_position_ids=pos).last_hidden_state)
a=Gemma4AudioConfig(hidden_size=8,num_hidden_layers=2,num_attention_heads=2,output_proj_dims=12,subsampling_conv_channels=[128,2],conv_kernel_size=3,attention_chunk_size=2,attention_context_left=3,attention_context_right=0,gradient_clipping=2.,use_clipped_linears=True)
a._attn_implementation='sdpa'
am=Gemma4AudioModel(a).eval();ae=EmbeddingGemma2MultimodalEmbedder(a,text).eval();aw=prepare(am,ae,'audio_tower','embed_audio')
feat=torch.sin(torch.arange(13*128).reshape(1,13,128).float()*.1);mask=torch.tensor([[True]*11+[False]*2])
with torch.inference_mode():
 y=am(input_features=feat,attention_mask=mask);audio=ae(y.last_hidden_state)[y.attention_mask]
f={'transformers':__import__('transformers').__version__,'text_hidden_size':8,'vision_config':v.to_dict(),'audio_config':a.to_dict(),
   'vision':{'weights':packed(vw),'pixels':pix.tolist(),'positions':pos[0].tolist(),'expected':vision.tolist()},
   'audio':{'weights':packed(aw),'features':feat.tolist(),'mask':mask[0].tolist(),'expected':audio.tolist()}}
out.write_text(json.dumps(f,indent=2)+'\n');print(out,vision.shape,audio.shape)

# Independent frontend receipt; the last analysis frame is right-padded and masked.
import numpy as np
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
frontend = Gemma4AudioFeatureExtractor()
samples = (np.sin(np.arange(1599, dtype=np.float64) * .037) * .2).astype(np.float32)
features = frontend([samples], return_tensors='np')
frontend_out = out.with_name('audio-frontend.json')
frontend_keys = ('sampling_rate', 'feature_size', 'frame_length', 'hop_length', 'fft_length', 'mel_floor',
                 'min_frequency', 'max_frequency', 'dither', 'preemphasis', 'input_scale_factor', 'per_bin_mean', 'per_bin_stddev')
frontend_config = {key: (getattr(frontend, key).tolist() if isinstance(getattr(frontend, key), np.ndarray)
                         else getattr(frontend, key)) for key in frontend_keys}
frontend_out.write_text(json.dumps(dict(config=frontend_config, features=features['input_features'].tolist(),
                                       mask=features['input_features_mask'][0].tolist()), indent=2) + '\n')
