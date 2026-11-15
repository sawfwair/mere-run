"""Generate independent NumPy FP32 references for SOV audio and mixed-mask DFlash."""
import json, math
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1]
def tensor(x):
 a=np.asarray(x,dtype=np.float32);return {'shape':list(a.shape),'values':a.reshape(-1).tolist()}
def values(shape,phase,scale=.13):
 return (np.sin(np.arange(np.prod(shape),dtype=np.float32)*.173+phase)*scale).reshape(shape)
def weights(shapes):
 return {k: values(shape,i*.31, .07) + (1 if 'norm' in k and k.endswith('weight') else 0) for i,(k,shape) in enumerate(shapes.items())}
def linear(x,w,p):return np.einsum("...i,oi->...o",x,w[p+'.weight'])+(w.get(p+'.bias',0))
def norm(x,w,p,rms=False):
 x=x.astype(np.float32)
 if rms:y=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
 else:y=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
 return y*w[p+'.weight']+w.get(p+'.bias',0)
def gelu(x):return x*.5*(1+np.vectorize(math.erf)(x/math.sqrt(2))).astype(np.float32)
def attention(q,k,v,mask=None):
 scores=np.einsum("bhqd,bhkd->bhqk",q,k)/math.sqrt(q.shape[-1])
 if mask is not None:scores=np.where(mask,scores,-np.inf)
 e=np.exp(scores-scores.max(-1,keepdims=True));return np.einsum("bhqk,bhkd->bhqd",e/e.sum(-1,keepdims=True),v)
def split(x,h):return x.reshape(x.shape[0],x.shape[1],h,-1).transpose(0,2,1,3)
def merge(x):return x.transpose(0,2,1,3).reshape(x.shape[0],x.shape[2],-1)
audio_config={'d_model':4,'encoder_layers':1,'encoder_attention_heads':2,'encoder_ffn_dim':6,'num_mel_bins':128,'max_source_positions':4}
audio_shapes={'conv1.weight':[4,3,128],'conv1.bias':[4],'conv2.weight':[4,3,4],'conv2.bias':[4],'embed_positions.weight':[4,4]}
p='layers.0.'
for key in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.out_proj']:
 audio_shapes[p+key+'.weight']=[4,4]
 if 'k_proj' not in key: audio_shapes[p+key+'.bias']=[4]
for key in ['self_attn_layer_norm','final_layer_norm','layer_norm']:
 for suffix in ['weight','bias']:audio_shapes[(p if key!='layer_norm' else '')+key+'.'+suffix]=[4]
for key,shape in [('fc1',[6,4]),('fc2',[4,6])]:
 audio_shapes[p+key+'.weight']=shape; audio_shapes[p+key+'.bias']=[shape[0]]
aw=weights(audio_shapes)
mel=values([1,128,8],.4)
def conv(x,w,b,stride):
 padded=np.pad(x,((0,0),(1,1),(0,0)))
 return np.stack([np.einsum('bki,oki->bo',padded[:,i:i+3,:],w)+b for i in range(0,x.shape[1],stride)],axis=1)
x=gelu(conv(mel.transpose(0,2,1),aw['conv1.weight'],aw['conv1.bias'],1))
x=gelu(conv(x,aw['conv2.weight'],aw['conv2.bias'],2))+aw['embed_positions.weight']
y=norm(x,aw,p+'self_attn_layer_norm')
y=linear(merge(attention(split(linear(y,aw,p+'self_attn.q_proj'),2),split(linear(y,aw,p+'self_attn.k_proj'),2),split(linear(y,aw,p+'self_attn.v_proj'),2))),aw,p+'self_attn.out_proj')
x=x+y
x=x+linear(gelu(linear(norm(x,aw,p+'final_layer_norm'),aw,p+'fc1')),aw,p+'fc2')
encoded=norm(x,aw,'layer_norm')
pw=weights({'linear_1.weight':[8,8],'linear_2.weight':[8,8]})
projected=linear(gelu(linear(encoded.reshape(1,2,8),pw,'linear_1')),pw,'linear_2')
text_config={'model_type':'gemma4_text','hidden_size':8,'num_hidden_layers':3,'intermediate_size':12,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':4,'max_position_embeddings':128,'rms_norm_eps':1e-6,'vocab_size':32,'hidden_size_per_layer_input':0,'sliding_window':3,'layer_types':['sliding_attention','full_attention','sliding_attention'],'attention_k_eq_v':False,'tie_word_embeddings':True}
config={'model_type':'sovereignai_transcribe','router_dtype':'float32','audio_config':audio_config,'text_config':text_config,'audio_token_id':25,'downsample_factor':2}
dc={'hidden_size':8,'intermediate_size':12,'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':4,'rms_norm_eps':1e-6,'vocab_size':32,'sliding_window':3,'layer_types':['sliding_attention','full_attention'],'block_size':4,'dflash_config':{'mask_token_id':4,'target_layer_ids':[0,1]},'rope_parameters':{'rope_theta':10000,'rope_type':'default'},'hidden_act':'silu'}
ds={'fc.weight':[8,16],'hidden_norm.weight':[8],'norm.weight':[8]}
for i in range(2):
 p=f'layers.{i}.'
 for name,shape in [('q_proj',[8,8]),('k_proj',[4,8]),('v_proj',[4,8]),('o_proj',[8,8])]:ds[p+'self_attn.'+name+'.weight']=shape
 for name in ['q_norm','k_norm']:ds[p+'self_attn.'+name+'.weight']=[4]
 for name,shape in [('gate_proj',[12,8]),('up_proj',[12,8]),('down_proj',[8,12])]:ds[p+'mlp.'+name+'.weight']=shape
 ds[p+'input_layernorm.weight']=[8]; ds[p+'post_attention_layernorm.weight']=[8]
dw=weights(ds)
context=values([1,5,16],.8)
noise=values([1,4,8],1.7)
projected_context=norm(linear(context,dw,'fc'),dw,'hidden_norm',True)
def rope(x,offset):
 positions=np.arange(offset,offset+x.shape[2],dtype=np.float32)
 freq=1/(10000**(np.arange(0,4,2,dtype=np.float32)/4))
 phase=np.outer(positions,freq)
 cos=np.concatenate([np.cos(phase)]*2,-1)[None,None]
 sin=np.concatenate([np.sin(phase)]*2,-1)[None,None]
 rotated=np.concatenate([-x[...,2:],x[...,:2]],-1)
 return x*cos+rotated*sin
x=noise
for i in range(2):
 p=f'layers.{i}.';a=p+'self_attn.'
 y=norm(x,dw,p+'input_layernorm',True)
 q=rope(norm(split(linear(y,dw,a+'q_proj'),2),dw,a+'q_norm',True),5)
 kv=np.concatenate([projected_context,y],axis=1)
 k=rope(norm(split(linear(kv,dw,a+'k_proj'),1),dw,a+'k_norm',True),0)
 v=split(linear(kv,dw,a+'v_proj'),1)
 mask=None
 if i==0:
  query=np.arange(5,9)[:,None];key=np.arange(9)[None,:]
  mask=(key<=query)&(key>query-3)
 y=linear(merge(attention(q,k,v,mask)),dw,a+'o_proj')
 x=x+y
 y=norm(x,dw,p+'post_attention_layernorm',True)
 gate=linear(y,dw,p+'mlp.gate_proj'); up=linear(y,dw,p+'mlp.up_proj')
 x=x+linear(gate/(1+np.exp(-gate))*up,dw,p+'mlp.down_proj')
denoised=norm(x,dw,'norm',True)
fixture={'config':config,'draftConfig':dc,'audioWeights':{k:tensor(v) for k,v in aw.items()},'projectorWeights':{k:tensor(v) for k,v in pw.items()},'draftWeights':{k:tensor(v) for k,v in dw.items()},'mel':tensor(mel),'encoded':tensor(encoded),'projected':tensor(projected),'context':tensor(context),'noise':tensor(noise),'denoised':tensor(denoised)}
out=r/'Tests/SOVTranscribeRuntimeTests/Fixtures/SOV';out.mkdir(parents=True,exist_ok=True)
(out/'reference.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')

# Whisper frontend reference: two windows, one global peak and STFT over their boundary.
samples=np.zeros(960000,dtype=np.float32)
for i in range(1600):samples[i]=.1*math.sin(2*math.pi*440*i/16000)+.03*math.cos(2*math.pi*1234*i/16000)
for i in range(160):samples[480000+i]=.8*math.sin(2*math.pi*700*i/16000)
frames=np.lib.stride_tricks.sliding_window_view(np.pad(samples,(200,200),mode='reflect'),400)[::160][:-1]
window=(.5-.5*np.cos(2*np.pi*np.arange(400)/400)).astype(np.float32)
power=np.abs(np.fft.rfft(frames*window,axis=-1))**2
mel_max=15+np.log(8000/1000)/(np.log(6.4)/27)
mel_points=np.linspace(0,mel_max,130)
hz=np.where(mel_points<15,mel_points*200/3,1000*np.exp((mel_points-15)*np.log(6.4)/27))
frequencies=np.arange(201)*40
bank=np.maximum(0,np.minimum((frequencies[None,:]-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None]),
                              (hz[2:,None]-frequencies[None,:])/(hz[2:,None]-hz[1:-1,None])))
bank*=2/(hz[2:]-hz[:-2])[:,None]
logmel=np.log10(np.maximum(power@bank.T,1e-10))
features=(np.maximum(logmel,logmel.max()-8)+4)/4
points=[]
for frame in [0,1,8,10,12,2999,3000,3001,3002,3010,5999]:
 for channel in [0,5,25,60,100,127]:points.append({'window':frame//3000,'channel':channel,'frame':frame%3000,'value':float(features[frame,channel])})
(out/'frontend.json').write_text(json.dumps(points,separators=(',',':'))+'\n')
(r/'Tests/MereRunCoreTests/Fixtures/SOV/frontend.json').write_text(json.dumps(points,separators=(',',':'))+'\n')
