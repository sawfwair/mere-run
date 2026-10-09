#!/usr/bin/env python3
"""Compare native full-checkpoint CLI output with remote dequantized references.

Standard library only. Requires a locally downloaded MLX artifact containing
PPLX_REFERENCE_VECTORS.json. Captures process timing and macOS memory statistics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time


def cosine(a,b):
    aa=sum(x*x for x in a); bb=sum(x*x for x in b)
    if aa==0 and bb==0: return 1.0
    return sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else 0.0


def run(args):
    reference_path=args.artifact/'PPLX_REFERENCE_VECTORS.json'
    reference=json.loads(reference_path.read_text())
    contextual=reference['kind']=='context'
    results=[]
    with tempfile.TemporaryDirectory(prefix='pplx-native-') as folder:
        for task in ['query','document','image']:
            cases=[c for c in reference['cases'] if ('image' in c if task=='image' else c['task']==task and 'image' not in c)]
            if not cases: continue
            command=['/usr/bin/time','-l',str(args.cli.resolve()),'text','embed','--model',str(args.artifact.resolve()),'--task','document' if task=='image' else task]
            if task=='image':
                command+=['--image']+[str(args.artifact/c['image']) for c in cases]
            elif contextual:
                chunks=Path(folder)/'chunks.json'
                chunks.write_text(json.dumps({'documents':[c['chunks'] for c in cases]}))
                command+=['--chunks-json',str(chunks)]
            else: command += [c['chunks'][0] for c in cases]
            started=time.monotonic()
            completed=subprocess.run(command,capture_output=True,text=True,timeout=1200)
            elapsed=time.monotonic()-started
            (args.output.parent/f'{reference["kind"]}-{task}-native.stderr').write_text(completed.stderr)
            if completed.returncode: raise RuntimeError(f'Native {task} failed ({completed.returncode}): '+completed.stderr[-1200:])
            payload=json.loads(completed.stdout)
            (args.output.parent/f'{reference["kind"]}-{task}-native.json').write_text(completed.stdout)
            assert len(payload['data'])==len(cases)
            similarities=[]; max_error=0.0
            for actual,case in zip(payload['data'],cases):
                assert actual['tokenCount']==len(case['input_ids']), (actual['tokenCount'],len(case['input_ids']))
                target=case['quantized_vectors']
                assert len(actual['embeddings'])==len(target)
                for a,b in zip(actual['embeddings'],target):
                    assert len(a)==len(b) and all(math.isfinite(v) for v in a)
                    similarities.append(cosine(a,b))
                    max_error=max(max_error,max(abs(x-y) for x,y in zip(a,b)))
            passes=min(similarities)>=0.999 and max_error<=(2.0 if contextual else 0.002)
            results.append({'task':task,'cases':len(cases),'elapsed_seconds':elapsed,
                'minimum_vector_cosine':min(similarities),'mean_vector_cosine':sum(similarities)/len(similarities),
                'max_absolute_error':max_error,'passes':passes,'timing_and_memory':completed.stderr})
    binary_sha=hashlib.sha256()
    with args.cli.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): binary_sha.update(block)
    memory=subprocess.check_output(['sysctl','-n','hw.memsize'],text=True).strip()
    report={'schema_version':1,'machine_memory_bytes':int(memory),'cli_sha256':binary_sha.hexdigest(),'kind':reference['kind'],'artifact_path':str(args.artifact.resolve()),
        'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'scope':('Real full-checkpoint CLI query/chunk parity on this Mac.' if contextual else
                 'Real full-checkpoint CLI text parity and synthetic image smoke on this Mac.')+
                ' Maximum context and general retrieval quality unqualified.',
        'results':results,'passes':all(r['passes'] for r in results)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results']}))
    for r in results: print(json.dumps({k:v for k,v in r.items() if k!='timing_and_memory'}))
    if not report['passes']: raise SystemExit('Native checkpoint parity failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--cli',type=Path,default=Path('.build/debug/mere.run'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
