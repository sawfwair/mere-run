"""Independent scalar reference for Flint; standard library only, no native model imports."""
import json, math, random, struct
from pathlib import Path
out=Path(__file__).resolve().parent
c=json.loads((out/'config.json').read_text());c.update(hidden_size=8,intermediate_size=12,moe_intermediate_size=6,shared_expert_intermediate_size=10,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=4,num_experts=4,num_experts_per_tok=2,vocab_size=17,max_position_embeddings=128,sliding_window=5,layer_types=['full_attention','sliding_attention','full_attention'],no_rope_layer_indices=[0,2],mlp_only_layers=[0],eos_token_id=[16],bos_token_id=0,pad_token_id=16)
(out/'tiny-config.json').write_text(json.dumps(c,indent=2)+'\n')
quantized_config = dict(c, hidden_size=64, head_dim=16, intermediate_size=64,
                        moe_intermediate_size=64, shared_expert_intermediate_size=64,
                        quantization={"bits": 4, "group_size": 64},
                        mlx_export={"bits": 4, "embedding_bits": 8, "routed_expert_bits": 4})
(out/'tiny-int4-config.json').write_text(json.dumps(quantized_config, indent=2)+'\n')
rng=random.Random(20261021);weights={}
def weight(name,shape,norm=False):
 n=math.prod(shape);v=[struct.unpack('<f',struct.pack('<f',(1 if norm else 0)+rng.uniform(-.25,.25)))[0] for _ in range(n)];weights[name]=(shape,v)
weight('model.embed_tokens.weight',[17,8]);weight('model.norm.weight',[8],True);weight('lm_head.weight',[17,8])
for l in range(3):
 p=f'model.layers.{l}.'
 for k in ['input_layernorm','post_attention_layernorm']:weight(p+k+'.weight',[8],True)
 for k,o,i in [('q_proj',16,8),('k_proj',8,8),('v_proj',8,8),('o_proj',8,16)]:weight(p+'self_attn.'+k+'.weight',[o,i])
 for k in ['q_norm','k_norm']:weight(p+'self_attn.'+k+'.weight',[4],True)
 weight(p+'self_attn.softmax_offset',[4])
 def mlp(path,n):
  for k,o,i in [('gate_proj',n,8),('up_proj',n,8),('down_proj',8,n)]:weight(path+k+'.weight',[o,i])
 if l==0:mlp(p+'mlp.',12)
 else:
  weight(p+'mlp.gate.weight',[4,8]);weight(p+'mlp.gate.expert_bias',[4]);mlp(p+'mlp.shared_expert.',10)
  for e in range(4):mlp(p+f'mlp.experts.{e}.',6)
def values(n):return weights[n][1]
def linear(x,n):
 shape,w=weights[n+'.weight'];return [sum(v*t for v,t in zip(x,w[j*len(x):(j+1)*len(x)])) for j in range(shape[0])]
def norm(x,n):
 scale=1/math.sqrt(sum(t*t for t in x)/len(x)+1e-6);return [v*scale*w for v,w in zip(x,values(n+'.weight'))]
def mlp(x,n):
 a=linear(x,n+'gate_proj');b=linear(x,n+'up_proj');return linear([u/(1+math.exp(-u))*v for u,v in zip(a,b)],n+'down_proj')
def rope(x,pos):
 return [x[0]*math.cos(pos)-x[2]*math.sin(pos),x[1]*math.cos(pos/100)-x[3]*math.sin(pos/100),x[2]*math.cos(pos)+x[0]*math.sin(pos),x[3]*math.cos(pos/100)+x[1]*math.sin(pos/100)]
tokens=[1,3,4,2,8,9,0,5,6,1,10,7,2]
hidden=[values('model.embed_tokens.weight')[t*8:(t+1)*8] for t in tokens]
for l in range(3):
 p=f'model.layers.{l}.';q=[];k=[];v=[]
 for pos,h in enumerate(hidden):
  x=norm(h,p+'input_layernorm');qs=linear(x,p+'self_attn.q_proj');ks=linear(x,p+'self_attn.k_proj');vs=linear(x,p+'self_attn.v_proj')
  qh=[norm(qs[j:j+4],p+'self_attn.q_norm') for j in range(0,16,4)];kh=[norm(ks[j:j+4],p+'self_attn.k_norm') for j in range(0,8,4)]
  if l==1:qh=[rope(t,pos) for t in qh];kh=[rope(t,pos) for t in kh]
  q.append(qh);k.append(kh);v.append([vs[j:j+4] for j in range(0,8,4)])
 result=[]
 for pos,h in enumerate(hidden):
  att=[]
  for head in range(4):
   ids=list(range(max(0,pos-4) if l==1 else 0,pos+1));scores=[sum(a*b for a,b in zip(q[pos][head],k[t][head//2]))/2 for t in ids];sink=values(p+'self_attn.softmax_offset')[head];mx=max(scores+[sink]);ex=[math.exp(s-mx) for s in scores];den=sum(ex)+math.exp(sink-mx)
   att += [sum(e*v[t][head//2][d] for e,t in zip(ex,ids))/den for d in range(4)]
  h=[a+b for a,b in zip(h,linear(att,p+'self_attn.o_proj'))];x=norm(h,p+'post_attention_layernorm')
  if l==0:y=mlp(x,p+'mlp.')
  else:
   scores=[1/(1+math.exp(-t)) for t in linear(x,p+'mlp.gate')];bias=values(p+'mlp.gate.expert_bias');ids=sorted(range(4),key=lambda e:scores[e]+bias[e],reverse=True)[:2];den=sum(scores[e] for e in ids)+1e-20;y=mlp(x,p+'mlp.shared_expert.')
   for e in ids:y=[a+b*scores[e]/den for a,b in zip(y,mlp(x,p+f'mlp.experts.{e}.'))]
  result.append([a+b for a,b in zip(h,y)])
 hidden=result
logits=[t for h in hidden for t in linear(norm(h,'model.norm'),'lm_head')]
(out/'reference.json').write_text(json.dumps({'tokens':tokens,'logits':logits})+'\n')
def save(path,w):
 header={};data=bytearray()
 for n,(shape,v) in sorted(w.items()):
  start=len(data);data.extend(struct.pack('<'+'f'*len(v),*v));header[n]={'dtype':'F32','shape':shape,'data_offsets':[start,len(data)]}
 h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8);path.write_bytes(struct.pack('<Q',len(h))+h+data)
save(out/'source.safetensors',weights)
(out/'source-index.json').write_text(json.dumps({'weight_map':{k:'source.safetensors' for k in weights}})+'\n')
stacked={k:v for k,v in weights.items() if '.experts.' not in k}
for l in [1,2]:
 for proj in ['gate_proj','up_proj','down_proj']:
  vs=[weights[f'model.layers.{l}.mlp.experts.{e}.{proj}.weight'] for e in range(4)];stacked[f'model.layers.{l}.mlp.switch_mlp.{proj}.weight']=([4]+vs[0][0],[t for v in vs for t in v[1]])
save(out/'weights.safetensors',stacked)
