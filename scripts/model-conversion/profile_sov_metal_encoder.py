#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Compare live custom int8 FFNs in the complete, untrimmed SOV encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import mlx.core as mx
from profile_sov_metal_quant import DynamicInt8, KERNEL_PATH, metrics

GELU = mx.compile(lambda x: x*(1+mx.erf(x/2**0.5))/2)

class Encoder:
    def __init__(self, root, first_custom=0, custom_fc2=True, exponent=2, columns=32):
        all_weights = mx.load(str(root/'model-00016-of-00016.safetensors'))
        self.w = {k[len('model.audio_tower.'):]:v for k,v in all_weights.items() if k.startswith('model.audio_tower.')}
        self.p = {k[len('model.embed_audio.'):]:v for k,v in all_weights.items() if k.startswith('model.embed_audio.')}
        self.operators = [DynamicInt8(256,exponent,True,True,False,True,columns), DynamicInt8(256,exponent,True,True,True,True,columns)]
        self.packed = {}
        self.first_custom = first_custom
        for layer in range(first_custom,32):
            for stage in (['fc1','fc2'] if custom_fc2 else ['fc1']):
                key=f'layers.{layer}.{stage}'
                self.packed[key]=self.operators[stage=='fc2'].pack(self.w[key+'.weight'])
        mx.eval(self.w,self.p,self.packed)
        self.functions = [[self.layer(i, custom) for i in range(32)] for custom in (False,True)]

    def linear(self,x,key,custom=False):
        if custom and key in self.packed:
            shape=x.shape
            y=self.operators[key.endswith('fc2')].matmul(x.reshape(-1,shape[-1]), self.packed[key], self.w[key+'.bias'])
            return y.reshape(*shape[:-1],y.shape[-1])
        return mx.addmm(self.w[key+'.bias'],x,self.w[key+'.weight'].T) if key+'.bias' in self.w else x@self.w[key+'.weight'].T

    def norm(self,x,key):
        return mx.fast.layer_norm(x,self.w[key+'.weight'],self.w[key+'.bias'],eps=1e-5)

    def layer(self,index,custom):
        prefix=f'layers.{index}.'
        def fn(x):
            z=self.norm(x,prefix+'self_attn_layer_norm')
            def split(a):return a.reshape(a.shape[0],a.shape[1],20,64).transpose(0,2,1,3)
            q,k,v=[split(self.linear(z,prefix+'self_attn.'+p+'_proj')) for p in ['q','k','v']]
            a=mx.fast.scaled_dot_product_attention(q,k,v,scale=0.125).transpose(0,2,1,3).reshape(x.shape)
            residual=x+self.linear(a,prefix+'self_attn.out_proj')
            return residual+self.linear(GELU(self.linear(self.norm(residual,prefix+'final_layer_norm'),prefix+'fc1',custom)),prefix+'fc2',custom)
        # Swift dispatches individual module operators. Keep the same boundaries.
        return fn

    def __call__(self,mel,custom):
        x=mel.transpose(0,2,1).astype(mx.bfloat16)
        for name,stride in [('conv1',1),('conv2',2)]:
            x=GELU(mx.conv1d(x,self.w[name+'.weight'].transpose(0,2,1),stride=stride,padding=1)+self.w[name+'.bias'])
        x=x+self.w['embed_positions.weight']
        for fn in self.functions[int(custom)]:
            x=fn(x);mx.eval(x)
        x=self.norm(x,'layer_norm')
        x=x.reshape(x.shape[0],375,5120)
        x=GELU(x@self.p['linear_1.weight'].T)@self.p['linear_2.weight'].T
        mx.eval(x)
        return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--first-custom',type=int,default=0)
    p.add_argument('--columns',type=int,choices=[32,64],default=32)
    p.add_argument('--exponent',type=int,choices=[0,1,2],default=2)
    p.add_argument('--fc1-only',action='store_true')
    p.add_argument('--repetitions',type=int,default=5)
    args=p.parse_args()
    arrays=mx.load(str(args.reference));mel=arrays['mel']
    enc=Encoder(args.checkpoint,args.first_custom,not args.fc1_only,args.exponent,args.columns)
    times=[[],[]];results=[None,None]
    for index in range(2+args.repetitions):
        for candidate in ([0,1] if index%2==0 else [1,0]):
            started=time.perf_counter();results[candidate]=enc(mel,bool(candidate));elapsed=time.perf_counter()-started
            if index>=2:times[candidate].append(elapsed)
        print(index,[x[-1] if x else None for x in times],flush=True)
    expected=arrays['audio_embeddings'].reshape(results[0].shape)
    receipt=dict(columns=args.columns,kernelSHA256=hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest(),mlx=mx.__version__,exponent=args.exponent,firstCustom=args.first_custom,fc1Only=args.fc1_only,reference=str(args.reference),
        baselineSeconds=times[0],customSeconds=times[1],baselineMedianSeconds=statistics.median(times[0]),customMedianSeconds=statistics.median(times[1]),
        speedup=statistics.median(times[0])/statistics.median(times[1]),baselineNativeErrors=metrics(results[0],expected),customNativeErrors=metrics(results[1],expected))
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    mx.save_safetensors(str(args.output.with_suffix('.safetensors')),{'baseline':results[0],'custom':results[1]})
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
