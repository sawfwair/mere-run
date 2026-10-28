"""Regenerate tower-reference.json with NumPy; run from any directory."""
from pathlib import Path
import json
import numpy as np
f = Path(__file__).resolve().parent
D=16;N=3;V=4096
# Values are Float32 sine waves. Swift creates the same parameter values; all
# attention, normalization, rotary, residual and gating arithmetic here is NumPy.
def tensor(name,shape):
 seed=sum(name.encode())%97;count=np.prod(shape);return (np.sin((np.arange(count,dtype=np.float64)+seed)*0.17)*0.03).astype(np.float32).reshape(shape)
def norm(x,path):return (x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6))*(1+tensor(path+'.weight',(x.shape[-1],)))
def linear(x,path,out):return x@tensor(path+'.weight',(out,x.shape[-1])).T
def rope(x):
 y=x.copy();angles=np.arange(N,dtype=np.float32)[:,None]*(1/np.power(1e7,np.arange(0,4,2,dtype=np.float32)/4))[None,:];c=np.cos(angles)[:,None,:];s=np.sin(angles)[:,None,:];even=x[:,:,:4:2];odd=x[:,:,1:4:2];y[:,:,:4:2]=even*c-odd*s;y[:,:,1:4:2]=odd*c+even*s;return y
hidden=tensor('hidden',(1,N,D))[0];code=np.array([0,17,29]);semantic=np.array([0,1,1],dtype=np.float32)[:,None];extra=tensor('extra',(N,D))*semantic
p='layer';emb=tensor(p+'.token_embedding.weight',(V,D))[code]*semantic+extra
mixed=linear(np.concatenate([norm(emb,p+'.enorm'),norm(hidden,p+'.hnorm')],-1),p+'.eh_proj',D);x=norm(mixed,p+'.input_norm');packed=linear(x,p+'.qkv',48);q=rope(norm(packed[:,:16].reshape(N,2,8),p+'.q_norm'));k=rope(norm(packed[:,32:40].reshape(N,1,8),p+'.k_norm'));v=packed[:,40:].reshape(N,1,8);k=np.repeat(k,2,axis=1);v=np.repeat(v,2,axis=1);score=np.einsum('thd,shd->hts',q,k)/np.sqrt(np.float32(8));score=np.where(np.tril(np.ones((N,N),dtype=bool))[None,:,:],score,-np.inf);prob=np.exp(score-np.max(score,axis=-1,keepdims=True));prob/=prob.sum(axis=-1,keepdims=True);att=np.einsum('hts,shd->thd',prob,v).reshape(N,16);gates=1/(1+np.exp(-packed[:,16:32]));res=mixed+linear(att*gates,p+'.out',D);n=norm(res,p+'.post_norm');gate=linear(n,p+'.gate',32);ff=(gate/(1+np.exp(-gate)))*linear(n,p+'.up',32);out=norm(res+linear(ff,p+'.down',D),p+'.final_layernorm');(f/'tower-reference.json').write_text(json.dumps(dict(hidden=hidden.flatten().tolist(),extra=extra.flatten().tolist(),codes=code.tolist(),semantic=semantic.flatten().tolist(),output=out.flatten().tolist()),indent=2)+'\n')
