"""Tiny unmodified-forward BiRefNet and torchvision deformable-convolution oracles.

python export_birefnet_fixtures.py --source /path/to/idol-sdk --output Tests/ImageRuntimeTests/Fixtures/Idol
Requires torch, torchvision, numpy, pillow, safetensors. Production weights are not bundled.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torchvision.ops import deform_conv2d

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip() == '4702a7f00850e34b6e83a2170cc123c3e9570744'
spec = importlib.util.spec_from_file_location('birefnet_reference', args.source/'idol/models/birefnet.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
torch.manual_seed(123)
results = {}
for kernel, stride in [(1,1),(3,1),(7,1),(3,2)]:
    label = f'k{kernel}s{stride}'
    height, width = 6, 8
    out_h, out_w = (height-1)//stride+1, (width-1)//stride+1
    image = torch.randn(1,3,height,width)
    offsets = torch.randn(1,2*kernel*kernel,out_h,out_w)*1.7
    masks = torch.rand(1,kernel*kernel,out_h,out_w)*2
    weights = torch.randn(4,3,kernel,kernel)*.1
    results.update({label+'_image':image,label+'_offsets':offsets,label+'_mask':masks,label+'_weight':weights,
        label+'_output':deform_conv2d(image,offsets,weights,padding=kernel//2,stride=stride,mask=masks)})
    results[label+'_half'] = deform_conv2d(image.half(),offsets.half(),weights.half(),padding=kernel//2,
                                          stride=stride,mask=masks.half()).float()
# Use original operators and forwards with reduced architecture dimensions.
def swin_init(self):
    torch.nn.Module.__init__(self)
    self.num_layers=4; self.embed_dim=8; self.num_features=[8,16,32,64]
    self.patch_embed=ref._SwinPatchEmbed(patch_size=4,in_channels=3,embed_dim=8)
    self.layers=torch.nn.ModuleList([ref._SwinBasicLayer(8<<i,2,2<<i,4,downsample=i<3) for i in range(4)])
    for i in range(4): self.add_module(f'norm{i}',torch.nn.LayerNorm(8<<i))
ref._SwinLarge.__init__=swin_init
original_basic=ref._BasicDecBlk.__init__
def basic_init(self,in_channels,out_channels,inter_channels=8):
    original_basic(self,in_channels,out_channels,inter_channels=8)
ref._BasicDecBlk.__init__=basic_init
original_simple=ref._SimpleConvs.__init__
def simple_init(self,in_channels,out_channels,inter_channels=8):
    original_simple(self,in_channels,out_channels,inter_channels=8)
ref._SimpleConvs.__init__=simple_init
def aspp_init(self,in_channels,out_channels=None,parallel_block_sizes=(1,3,7)):
    torch.nn.Module.__init__(self)
    out_channels=out_channels or in_channels; inter=4
    self.aspp1=ref._ASPPModuleDeformable(in_channels,inter,1,padding=0)
    self.aspp_deforms=torch.nn.ModuleList([ref._ASPPModuleDeformable(in_channels,inter,k,padding=k//2) for k in parallel_block_sizes])
    self.global_avg_pool=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),torch.nn.Conv2d(in_channels,inter,1,bias=False),torch.nn.BatchNorm2d(inter),torch.nn.ReLU())
    self.conv1=torch.nn.Conv2d(inter*5,out_channels,1,bias=False)
    self.bn1=torch.nn.BatchNorm2d(out_channels);self.relu=torch.nn.ReLU()
ref._ASPPDeformable.__init__=aspp_init
ref.BiRefNet._CHANNELS=(128,64,32,16)
model=ref.BiRefNet().eval()
with torch.no_grad():
    # Exercise nontrivial evaluation BatchNorm as well as affine norms.
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.running_mean.copy_(torch.randn_like(module.running_mean)*.1)
            module.running_var.copy_(torch.rand_like(module.running_var)+.5)
    image=torch.randn(1,3,64,128)
    results['network_image']=image
    results['network_fp32']=model(image)
    for i,value in enumerate(model.bb(image)):results[f'swin_{i}']=value
    results['resized']=F.interpolate(image,size=(39,77),mode='bilinear',align_corners=True)
    results['patches']=ref._image2patches(image,torch.zeros(1,1,8,16))
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},args.output/'birefnet-tiny.safetensors')
    results['network_fp16']=model.half()(image.half()).float()
save_file({k:v.contiguous() for k,v in results.items()},args.output/'birefnet-results.safetensors')
