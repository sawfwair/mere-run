"""Run the pinned upstream IDOL graph with real local weights and save a numerical trace.

Reference-only Python. The product inference path is Swift/MLX.
Requires the IDOL SDK dependencies. Background removal uses CPU because torchvision
has no MPS deform_conv2d kernel; the transformer and encoders use MPS.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from tensordict import TensorDict

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--dino',type=Path,required=True)
p.add_argument('--flux',type=Path,required=True)
p.add_argument('--image',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=50)
p.add_argument('--background-only',action='store_true')
p.add_argument('--decoder-only',action='store_true')
p.add_argument('--mode',default='synthetic')
p.add_argument('--gaussians',type=int,default=262144)
a=p.parse_args()
assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()=='4702a7f00850e34b6e83a2170cc123c3e9570744'
sys.path.insert(0,str(a.source))
from idol.models import BiRefNet,GaussianFixedlenDecoder,LatentSeqMMFlowModel,OctreeProbabilityFixedlenDecoder
from idol.models.layers import RePo3DRotaryEmbedding
from idol.pipeline import IdolPipeline
from idol.conditioning import ImageEncoder
from idol.sampler import sample_flow
from idol.camera import decode_camera
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor

a.output.mkdir(parents=True,exist_ok=True)
start=time.monotonic();timings={};trace={}
def mark(name):
    timings[name]=time.monotonic()-start
    print(json.dumps(dict(stage=name,elapsed=timings[name])),flush=True)
config=json.loads((a.checkpoint/'config.json').read_text())
pipe=IdolPipeline.__new__(IdolPipeline)
pipe.config=config;pipe.device=torch.device('mps');pipe.optimize=False

def load(model,name,device):
    model.load_state_dict(load_file(a.checkpoint/(name+'.safetensors')),strict=True)
    return model.eval().requires_grad_(False).to(device)
if a.decoder_only:
    trace=load_file(a.output/'trace.safetensors')
    decoder=load(OctreeProbabilityFixedlenDecoder(**config['decoder']['args']),'decoder',pipe.device)
    queries=trace['anchors'][:,:32].to(pipe.device)
    latent=trace['flow_latent'].to(pipe.device)
    trace['octree_queries']=queries.cpu().contiguous()
    with torch.no_grad():
        for resolution in [2,16,256]:
            trace[f'octree_{resolution}']=decoder(queries,torch.tensor([resolution],device=pipe.device),latent).cpu().contiguous()
    save_file(trace,a.output/'trace.safetensors')
    mark('octree-parity-exported')
    sys.exit(0)
pipe.background_remover=load(BiRefNet(),'birefnet','cpu')
mark('birefnet-loaded')
if a.background_only:
    original_forward=pipe.background_remover.forward
    def mask_forward(x):
        result=original_forward(x)
        trace['biref_input']=x.cpu().contiguous()
        trace['biref_alpha']=result.cpu().contiguous()
        return result
    pipe.background_remover.forward=mask_forward
    with torch.no_grad(): prepared=pipe.preprocess(str(a.image),mode=a.mode)
    prepared.save(a.output/'input.png')
    save_file(trace,a.output/'trace.safetensors')
    mark('birefnet-exported')
    sys.exit(0)
with torch.no_grad():
    prepared=pipe.preprocess(str(a.image),mode=a.mode)
prepared.save(a.output/'input.png')
mark('preprocessed')
del pipe.background_remover
pipe.image_encoder=ImageEncoder(str(a.dino),str(a.flux),dinov3_dtype='bf16',device=pipe.device)
mark('encoders-loaded')
def traced_posterior(self,generator=None):
    noise=randn_tensor(self.mean.shape,generator=generator,device=self.parameters.device,dtype=self.parameters.dtype)
    trace['vae_noise']=noise.detach().cpu().contiguous()
    trace['vae_mean']=self.mean.detach().float().cpu().contiguous()
    trace['vae_logvar']=self.logvar.detach().float().cpu().contiguous()
    return self.mean+self.std*noise
DiagonalGaussianDistribution.sample=traced_posterior
import numpy as np
pixels=torch.from_numpy(np.array(prepared)).permute(2,0,1).float().div(255.)[None].to(pipe.device)
torch.manual_seed(42)
with torch.no_grad():
    condition=pipe.image_encoder(pixels)
trace['feature1']=condition['feature1'].cpu().contiguous();trace['feature2']=condition['feature2'].cpu().contiguous()
mark('conditioned')
del pipe.image_encoder
torch.mps.empty_cache()
pipe.denoiser=load(LatentSeqMMFlowModel(**config['denoiser']['args']),'denoiser',pipe.device)
for module in pipe.denoiser.modules():
    if isinstance(module,RePo3DRotaryEmbedding):module.real_valued=True
noise=TensorDict(dict(latent=torch.randn(1,8192,16,device=pipe.device),camera=torch.randn(1,1,5,device=pipe.device)),batch_size=1,device=pipe.device)
trace['noise_latent']=noise['latent'].cpu().contiguous();trace['noise_camera']=noise['camera'].cpu().contiguous()
# Save conditioning before the full sampler so a failed backend still leaves useful evidence.
save_file(trace,a.output/'trace-conditioning.safetensors')
mark('denoiser-loaded')
calls=0
def denoiser(x,t,cond):
    global calls
    value=pipe.denoiser(x,t,cond)
    if calls<2:
        prefix='positive' if calls==0 else 'negative'
        trace[prefix+'_latent']=value['latent'].cpu().contiguous();trace[prefix+'_camera']=value['camera'].cpu().contiguous()
        mark(prefix+'-velocity')
    calls+=1
    if calls%10==0:mark(f'flow-step-{calls//2}')
    return value
with torch.no_grad():
    state=sample_flow(denoiser,noise,cond=condition,neg_cond=condition.apply(torch.zeros_like),steps=a.steps,
                      rescale_t=3.,cfg_strength=3.,verbose=False,batch_cfg=False)
trace['flow_latent']=state['latent'].cpu().contiguous();trace['flow_camera']=state['camera'].cpu().contiguous()
mark('sampled')
del pipe.denoiser
torch.mps.empty_cache()
pipe.decoder=load(OctreeProbabilityFixedlenDecoder(**config['decoder']['args']),'decoder',pipe.device)
with torch.no_grad():
    anchors=OctreeProbabilityFixedlenDecoder.sample(pipe.decoder,state['latent'],num_points=a.gaussians//32,level=config['max_voxel_level'])
trace['anchors']=anchors.cpu().contiguous()
mark('anchors')
del pipe.decoder
torch.mps.empty_cache()
pipe.decoder_gs=load(GaussianFixedlenDecoder(**config['decoder_gs']['args']),'decoder_gs',pipe.device)
with torch.no_grad():
    features=pipe.decoder_gs(anchors,state['latent'])
    gaussian=pipe.decoder_gs.to_representation(anchors,features)[0]
trace['gaussian_features']=features.cpu().contiguous()
gaussian.save_ply(str(a.output/'object.ply'))
camera=decode_camera(state['camera'][0,0]);(a.output/'camera.json').write_text(json.dumps(camera,indent=2))
save_file(trace,a.output/'trace.safetensors')
mark('exported')
(a.output/'timings.json').write_text(json.dumps(dict(timings=timings,torch=torch.__version__,seed=42,steps=a.steps,
    gaussians=a.gaussians,image=str(a.image),mode=a.mode,source_revision='4702a7f00850e34b6e83a2170cc123c3e9570744'),indent=2))
