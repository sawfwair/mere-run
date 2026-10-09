#!/usr/bin/env python3
"""Small paired FP32/packed-weight diagnostic, not a retrieval benchmark.

Requires torch==2.8.0, transformers==5.4.0, accelerate==1.10.1 and Pillow. Uses independent PyTorch affine unpacking.
The reference uses Transformers' Qwen3.5 model and upstream input/output math.
"""
import argparse
import json
from pathlib import Path
from convert_pplx_embed_v2_mlx import MODELS, digest, write_json, checksums

DOCUMENTS = [
    'Photosynthesis uses sunlight, water and carbon dioxide to produce sugars and oxygen in plants.',
    'A TCP connection uses a three-way handshake: SYN, SYN-ACK, then ACK.',
    'Tides are driven primarily by the gravitational attraction of the Moon and Sun.',
    'Bread dough rises because yeast fermentation produces carbon dioxide.',
    'Marie Curie researched radioactivity and won Nobel Prizes in physics and chemistry.',
    'Les baleines sont des mammifères marins qui respirent de l’air.',
    '太陽光発電は太陽の光を電気に変換します。',
    'El agua hierve a cien grados Celsius al nivel del mar.',
]
QUERIES = ['How do plants turn sunlight into food?', 'What establishes a TCP connection?',
           'Why does sea level rise and fall?', 'Why does yeast make bread rise?',
           'Who researched radioactivity and won two Nobel prizes?', 'Comment respirent les baleines ?',
           '太陽光発電の仕組みは？', '¿A qué temperatura hierve el agua?']


def prepare(tokenizer, chunks, task, contextual, skip):
    if not contextual:
        marker = '[Q] ' if task == 'query' else '[D] '
        ids = [tokenizer.convert_tokens_to_ids(marker)] + tokenizer.encode(chunks[0], add_special_tokens=False)
        ids = ids[:1024 if task == 'query' else 4096]
        return ids, [], [i for i, token in enumerate(ids) if task == 'query' or token not in skip]
    if task == 'query':
        ids = [tokenizer.convert_tokens_to_ids('[Q] ')] + tokenizer.encode(chunks[0], add_special_tokens=False, split_special_tokens=True)
        return ids, [(0, len(ids))], []
    text = '[D] ' + '<|chunk_sep|>'.join(chunks)
    encoded = tokenizer(text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
    spans, cursor = [], 4
    for i, chunk in enumerate(chunks):
        if i: cursor += len('<|chunk_sep|>')
        end = cursor + len(chunk)
        indices = [j for j, (a,b) in enumerate(encoded['offset_mapping']) if b > a and a < end and b > cursor and end > cursor]
        spans.append((indices[0], indices[-1]+1) if indices else (0,0))
        cursor = end
    return encoded['input_ids'], spans, []


def run(args):
    import torch
    import torch.nn.functional as F
    import transformers
    import numpy as np
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5Model
    assert transformers.__version__ == '5.4.0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    contextual = args.kind == 'context'
    config = json.loads((args.source / 'config.json').read_text())
    config['model_type'] = 'qwen3_5'
    config.pop('auto_map', None)
    model = Qwen3_5Model.from_pretrained(args.source, config=Qwen3_5Config(**config), dtype=torch.float32,
                device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.source, config=Qwen3_5Config(**config), trust_remote_code=False)
    if contextual:
        projection = load_file(args.source / 'contextual_head.safetensors')['contextual_projection.weight'].cuda()
    else:
        projection = load_file(args.source / '1_Dense/model.safetensors')['linear.weight'].cuda()
    skip = set()
    if not contextual:
        skip = {tokenizer.convert_tokens_to_ids(t) for t in json.loads((args.source/'2_MultiVectorMask/config.json').read_text())['skiplist_words']}
    cases = [{'task':'query', 'chunks':[s]} for s in QUERIES] + [{'task':'document','chunks':[s]} for s in DOCUMENTS]
    cases += [{'task':'query','chunks':['café 東京 👩🏽‍🔬 [Q] <|chunk_sep|>']},
              {'task':'document','chunks':['Punctuation: commas, brackets [D], and symbols!']}]
    if contextual:
        cases += [{'task':'document','chunks':['café', '', '東京 👩🏽‍🔬', '[Q] injected <|chunk_sep|>']},
                  {'task':'document','chunks':[DOCUMENTS[0], DOCUMENTS[1], DOCUMENTS[2]]}]
    for case in cases:
        ids, spans, keep = prepare(tokenizer, case['chunks'], case['task'], contextual, skip)
        case.update(input_ids=ids, spans=spans, retained_indices=keep)
    image_inputs = None
    if not contextual:
        from PIL import Image
        image_array = np.zeros((256,256,3), dtype=np.uint8)
        image_array[:,:,:] = [245,245,230]
        image_array[32:112,32:224] = [30,80,190]
        image_array[144:224,32:128] = [190,60,35]
        image_array[144:224,160:224] = [40,150,90]
        Image.fromarray(image_array).save(args.artifact/'diagnostic-image.png')
        frame=torch.from_numpy(image_array.copy()).permute(2,0,1).float().unsqueeze(0)/255*2-1
        patches=frame.reshape(1,3,8,2,16,8,2,16).permute(0,2,5,3,6,1,4,7).reshape(256,3,256)
        pixels=patches.unsqueeze(2).expand(256,3,2,256).reshape(256,1536).cuda()
        ids=[tokenizer.convert_tokens_to_ids('[D] '),config['vision_start_token_id']]+[config['image_token_id']]*64+[config['vision_end_token_id']]
        cases.append({'task':'document','chunks':[], 'image':'diagnostic-image.png',
                      'input_ids':ids,'spans':[],'retained_indices':list(range(len(ids)))})
        image_inputs={'pixel_values':pixels,'image_grid_thw':torch.tensor([[1,16,16]],device='cuda'),
                      'mm_token_type_ids':torch.tensor([[0,0]+[1]*64+[0]],device='cuda')}
    @torch.inference_mode()
    def evaluate():
        outputs=[]
        for i, case in enumerate(cases):
            image_kwargs=image_inputs if 'image' in case else {}
            hidden = model(input_ids=torch.tensor([case['input_ids']],device='cuda'), use_cache=False,**image_kwargs).last_hidden_state[0].float()
            if contextual:
                if case['task']=='query': pooled = hidden.mean(dim=0,keepdim=True)
                else:
                    prefix = torch.cat([hidden.new_zeros(1,hidden.shape[-1]), hidden.cumsum(dim=0)])
                    pooled = torch.stack([(prefix[b]-prefix[a])/max(b-a,1) for a,b in case['spans']])
                vectors = torch.round(torch.tanh(F.linear(pooled,projection))*127).clamp(-128,127)
            else:
                vectors = F.normalize(F.linear(hidden,projection)[case['retained_indices']],dim=-1)
            outputs.append(vectors.cpu().numpy())
            print(json.dumps({'evaluated':i,'phase':phase,'tokens':len(case['input_ids'])}),flush=True)
        return outputs
    phase='source'; original=evaluate()
    contract=json.loads((args.artifact/'config.json').read_text())['quantization']['modules']
    index=json.loads((args.artifact/'model.safetensors.index.json').read_text())['weight_map']
    params=dict(model.named_parameters())
    conversion=json.loads((args.artifact/'PPLX_CONVERSION.json').read_text())
    for shard in sorted(set(index['language_model.'+path+'.weight'] for path in contract)):
        arrays=load_file(args.artifact/shard)
        for path, policy in contract.items():
            key='language_model.'+path+'.weight'
            if index[key]!=shard: continue
            prefix=key.removesuffix('.weight')
            bits, group = policy['bits'], policy['group_size']
            packed = arrays[key].to(device='cuda', dtype=torch.int64)
            shifts = torch.arange(0,32,bits,device='cuda',dtype=torch.int64)
            q = ((packed.unsqueeze(-1) >> shifts) & ((1 << bits)-1)).float()
            q = q.reshape(packed.shape[0],-1,group)
            scales = arrays[prefix+'.scales'].cuda().unsqueeze(-1)
            biases = arrays[prefix+'.biases'].cuda().unsqueeze(-1)
            value = (q*scales+biases).reshape(params[key].shape)
            sample=torch.tensor(conversion['weight_reconstruction'][key]['dequantized_sample'],device='cuda')
            torch.testing.assert_close(value[0,:sample.numel()],sample,rtol=0,atol=0.000001)
            with torch.no_grad(): params[key].copy_(value)
            del value, q, packed, scales, biases
            torch.cuda.empty_cache()
        del arrays
    phase='quantized'; quantized=evaluate()
    cosines=[]
    for a,b in zip(original,quantized):
        for x,y in zip(a,b):
            if np.linalg.norm(x)==0 and np.linalg.norm(y)==0: continue
            cosines.append(float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))))
    def scores(vectors):
        queries, docs=vectors[:8],vectors[8:16]
        if contextual:
            return np.array([[np.dot(q[0],d[0])/(np.linalg.norm(q[0])*np.linalg.norm(d[0])) for d in docs] for q in queries])
        return np.array([[(q@d.T).max(axis=1).sum() for d in docs] for q in queries])
    a,b=scores(original),scores(quantized)
    top_agreement=float(np.mean(a.argmax(axis=1)==b.argmax(axis=1)))
    gates={'minimum_vector_cosine':min(cosines)>=0.95,'mean_vector_cosine':float(np.mean(cosines))>=0.98,
           'retrieval_top1_agreement':top_agreement==1.0,'finite':all(np.isfinite(v).all() for v in quantized)}
    for case, source, quant in zip(cases,original,quantized):
        case.update(source_vectors=source.tolist(),quantized_vectors=quant.tolist())
    write_json(args.artifact/'PPLX_REFERENCE_VECTORS.json',{'schema_version':1,'kind':args.kind,'cases':cases})
    report={'schema_version':1,'kind':args.kind,'source_revision':MODELS[args.kind][1],
        'conversion_sha256':digest(args.artifact/'PPLX_CONVERSION.json'),
        'config_sha256':digest(args.artifact/'config.json'),'index_sha256':digest(args.artifact/'model.safetensors.index.json'),
        'scope':'Small paired text/chunk retrieval diagnostic'+(' and one synthetic image parity case' if not contextual else '')+'. Broad benchmark, image retrieval quality and maximum context unqualified.',
        'transformers_version':transformers.__version__,'torch_version':torch.__version__,
        'minimum_vector_cosine':min(cosines),'mean_vector_cosine':float(np.mean(cosines)),
        'retrieval_top1_agreement':top_agreement,'source_top1':a.argmax(axis=1).tolist(),
        'quantized_top1':b.argmax(axis=1).tolist(),'max_score_absolute_error':float(np.max(abs(a-b))),
        'gates':{k:bool(v) for k,v in gates.items()},'passes_diagnostic_gates':bool(all(gates.values())),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    write_json(args.artifact/'PPLX_QUALIFICATION.json',report)
    import shutil
    shutil.copyfile(__file__, args.artifact / Path(__file__).name)
    checksums(args.artifact)
    print(json.dumps(report),flush=True)
    if not all(gates.values()): raise SystemExit('Diagnostic gates failed; do not publish this profile.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=MODELS,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    run(parser.parse_args())
