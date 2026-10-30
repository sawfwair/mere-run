"""Pillow byte-level oracles for native IDOL Lanczos and foreground framing."""
import argparse
import importlib.util
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.numpy import save_file

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip() == '4702a7f00850e34b6e83a2170cc123c3e9570744'
spec=importlib.util.spec_from_file_location('idol_preprocess',a.source/'idol/preprocess.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(91)
arrays={}
for label,mode in [('rgba','RGBA'),('rgb','RGB')]:
    data=rng.integers(0,256,(9,13,4 if mode=='RGBA' else 3),dtype=np.uint8)
    image=Image.fromarray(data,mode=mode)
    arrays[label+'_input']=np.array(image.convert('RGBA'))
    for w,h in [(17,11),(5,7),(5,19)]:
        arrays[f'{label}_{w}_{h}']=np.array(image.resize((w,h),Image.Resampling.LANCZOS).convert('RGBA'))
image=Image.fromarray(rng.integers(0,256,(19,27,3),dtype=np.uint8),mode='RGB')
arrays['framing_input']=np.array(image.convert('RGBA'))
# The source first resizes the short side; this supplied mask isolates preprocessing from model numerics.
work=image.resize((round(27*64/19),64),Image.Resampling.LANCZOS)
y,x=np.indices((64,work.width));d=((x-work.width*.48)/(work.width*.4))**2+((y-31)/27)**2
mask=np.clip((1-d)*480,0,255).astype(np.uint8);arrays['framing_mask']=mask
class Remover:
    def remove_background(self,image):
        result=image.convert('RGBA');result.putalpha(Image.fromarray(mask,mode='L'));return result
for mode,margin in ref.CROP_MARGIN.items():
    arrays['framing_'+mode]=np.array(ref.preprocess_image(image,Remover(),image_size=64,crop_margin=margin).convert('RGBA'))
a.output.parent.mkdir(parents=True,exist_ok=True);save_file(arrays,a.output)
