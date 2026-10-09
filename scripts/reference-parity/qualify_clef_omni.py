#!/usr/bin/env python3
"""Eight bounded Clef Omni BF16 or reconstructed-Q4 reference probes.

Requires the pinned joint_schema_model.py, Transformers 5.10.2, PyTorch, NumPy,
Pillow, librosa, soundfile and PyAV. Synthetic media probes measure agreement,
not broad model accuracy. In particular, temporal-pair color order is not a
validated video-understanding benchmark.
"""
import argparse, hashlib, importlib.util, json, sys, time, wave
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from PIL import Image

if version('transformers') != '5.10.2':
 raise RuntimeError('Reference qualification requires Transformers 5.10.2.')
torch.set_num_threads(1)

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--quantized',type=Path)
args=parser.parse_args()
source=args.source;root=args.output;root.mkdir(parents=True,exist_ok=True)
reference=args.reference
label='q4-reconstructed' if args.quantized else 'bf16' 
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='21d05ebb8cbea26a26af65eca3d16cf4b69bf80a320e2d5b203e86f0d632a670'
spec=importlib.util.spec_from_file_location('clef_omni_reference',reference);ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
for color in ['red','blue']:
 Image.new('RGB',(64,64),color).save(root/(color+'.png'))
samples=(np.sin(np.arange(16000)*2*np.pi*440/16000)*10000).astype('<i2')
with wave.open(str(root/'tone.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(samples.tobytes())
route={'type':'choice','instructions':'Choose the matching category.','criteria':{'animals':'Animals and pets','billing':'Invoices and payments','weather':'Weather and forecasts'}}
cases=[
 dict(id='route',state='The invoice was paid yesterday.',questions={'category':route}),
 dict(id='json',state={'paid':True,'amount':48,'currency':'CAD'},questions={'paid':{'type':'noul','instructions':'Has the invoice been paid?'},'currency':{'type':'choice','instructions':'What currency is used?','criteria':{'CAD':'Canadian dollars','USD':'US dollars'}}}),
 dict(id='multilingual',state='Mañana lloverá y hará frío.',questions={'category':route}),
 dict(id='score',state='The service is down for every customer and payments are failing.',questions={'urgency':{'type':'score','instructions':'How urgent is the incident?','criteria':['Low','Moderate','Critical']}}),
 dict(id='image',state='Inspect the image.',images=[str(root/'red.png')],questions={'color':{'type':'choice','instructions':'What is the dominant color?','criteria':{'blue':'Blue','red':'Red'}}}),
 dict(id='audio',state='Listen to the clip.',audio=[str(root/'tone.wav')],questions={'speech':{'type':'noul','instructions':'Is spoken language audible?'}}),
 dict(id='video',state='Inspect the frames in order.',videos=[[str(root/'red.png'),str(root/'blue.png')]],questions={'last':{'type':'choice','instructions':'What is the last frame color?','criteria':{'blue':'Blue','red':'Red'}}}),
 dict(id='mixed',state='Inspect the image and listen to the sound.',images=[str(root/'blue.png')],audio=[str(root/'tone.wav')],questions={'color':{'type':'choice','instructions':'What is the dominant image color?','criteria':{'blue':'Blue','red':'Red'}},'speech':{'type':'noul','instructions':'Is spoken language audible?'}}),
]
for case in cases:
 request={k:v for k,v in case.items() if k!='id'};request.update(model='clef-omni',max_tokens=8192)
 (root/(case['id']+'.request.json')).write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
print('Loading pinned BF16 thinker',flush=True)
started=time.monotonic();model,processor=ref.load_release_model(source,attn_implementation='sdpa')
if args.quantized:
 from safetensors import safe_open
 qindex=json.loads((args.quantized/'model.safetensors.index.json').read_text())['weight_map']
 readers={name:safe_open(args.quantized/name,framework='pt',device='cpu') for name in set(qindex.values())}
 def tensor(key):return readers[qindex[key]].get_tensor(key)
 started_reconstruction=time.monotonic()
 with torch.no_grad():
  for layer,block in enumerate(model.thinker.model.layers):
   experts=block.mlp.experts
   intermediate=model.thinker.config.text_config.moe_intermediate_size
   for expert in range(model.thinker.config.text_config.num_experts):
    for projection in ['gate_proj','up_proj','down_proj']:
     base=f'thinker.model.layers.{layer}.mlp.experts.{expert}.{projection}'
     packed=tensor(base+'.weight').numpy()
     scales=tensor(base+'.scales').float().numpy()
     biases=tensor(base+'.biases').float().numpy()
     codes=((packed[...,None] >> (np.arange(8,dtype=np.uint32)*4)) & 15).reshape(packed.shape[0],-1).astype(np.float32)
     values=(codes.reshape(*scales.shape,64)*scales[...,None]+biases[...,None]).reshape(codes.shape)
     target=experts.down_proj[expert] if projection=='down_proj' else experts.gate_up_proj[expert, :intermediate] if projection=='gate_proj' else experts.gate_up_proj[expert, intermediate:]
     target.copy_(torch.from_numpy(values).to(device=target.device,dtype=target.dtype))
   print('Reconstructed Q4 layer',layer,flush=True)
 print('Q4 reconstruction seconds',time.monotonic()-started_reconstruction,flush=True)
outputs={}
for case in cases:
 request={k:v for k,v in case.items() if k!='id'};request['model']='clef-omni'
 if 'videos' in request:request['videos']=[np.stack([np.asarray(Image.open(p).convert('RGB')) for p in frames]) for frames in request['videos']]
 before=time.monotonic()
 with torch.inference_mode():out=ref.systemone(model,processor,request,max_length=8192)
 encoded=ref.encode_record(processor.tokenizer,request,max_length=8192,processor=processor)
 outputs[case['id']]=dict(result=out,token_ids=list(encoded.input_ids),elapsed_seconds=time.monotonic()-before)
 (root/('reference-'+label+'.json')).write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
 print(case['id'],out,flush=True)
(root/('reference-'+label+'-receipt.json')).write_text(json.dumps(dict(source_revision='0db1cd2607d76a7bdb2a382f659e7b313079f84b',torch=torch.__version__,transformers=version('transformers'),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,cases=len(outputs)),indent=2)+'\n')
