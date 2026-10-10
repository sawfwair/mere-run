#!/usr/bin/env python3
"""Regenerate asset-free CQ2/CQ4 matrix and gather goldens with NumPy."""
import numpy as np,json
out=[]
for bits in [2,4]:
 rows,cols=4,256; g=128
 packed=np.random.RandomState(bits).randint(0,256,size=rows*cols*bits//8,dtype=np.uint8)
 cb=np.linspace(-.2,.2,1<<bits,dtype=np.float32)
 norms=np.array([.5,.9,0,2,.01,1,3,.125],np.float16).astype(np.float32)
 shifts=np.arange(8//bits,dtype=np.uint8)*bits
 idx=((packed[:,None]>>shifts)&((1<<bits)-1)).reshape(rows,cols)
 w=(cb[idx].reshape(rows,-1,g)*norms.reshape(rows,-1,1)).astype(np.float64)
 h=1
 while h<g:
  pairs=w.reshape(-1,2,h);a,b=pairs[:,0].copy(),pairs[:,1].copy();pairs[:,0],pairs[:,1]=a+b,a-b;h*=2
 w=w.reshape(rows,cols)/np.sqrt(g)
 x=np.sin(np.arange(3*cols)*.1).astype(np.float32).reshape(3,cols)
 out.append(dict(bits=bits,packed=packed.tolist(),norms=norms.tolist(),codebook=cb.tolist(),input=x.ravel().tolist(),rows=w.ravel().tolist(),output=(x@w.T).ravel().tolist()))
open('Tests/SpeechRuntimeTests/Fixtures/Whistle/cq-kernels.json','w').write(json.dumps(out,separators=(',',':'))+'\n')
