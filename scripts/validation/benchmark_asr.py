#!/usr/bin/env python3
"""Compare the native ASR families on identical, checksum-verified recordings.

Inference runs exclusively in the optimized Swift executable. Models stay resident
across both recordings. Python orchestrates serial processes and scores text.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time

from compare_nemotron_transcribe import digest, errors, words


def variants(root, args):
    local = root/'.build/qualification/asr-comparison'
    models = local/'models'
    sov = root/'.build/qualification/sov'
    return [
        dict(id='parakeet-mlx', label='Parakeet 0.6B v3 · MLX BF16', backend='parakeet', model=models/'parakeet', flags=[]),
        dict(id='parakeet-coreml', label='Parakeet 0.6B v3 · Core ML', backend='parakeet', model=local/'parakeet-coreml', flags=['--artifact', str(local/'parakeet-coreml')]),
        dict(id='qwen-8bit', label='Qwen3-ASR 1.7B · MLX 8-bit', backend='qwen', model=models/'qwen', flags=[]),
        dict(id='whistle-beam5', label='Whistle · packed · beam 5', backend='whistle', model=models/'whistle', flags=['--beam-size','5']),
        dict(id='whistle-greedy', label='Whistle · packed · greedy', backend='whistle', model=models/'whistle', flags=['--beam-size','1']),
        dict(id='nemotron-int4', label='Nemotron 3.5 Transcribe 30B-A4B · expert INT4', backend='nemotron', model=args.nemotron_root, flags=[]),
        dict(id='sov-target-only', label='SOV · BF16 · target only', backend='sov', model=sov/'target', flags=[]),
        dict(id='sov-dflash-bf16', label='SOV · BF16 + DFlash', backend='sov', model=sov/'target', flags=['--draft-model',str(sov/'draft')]),
        dict(id='sov-dflash-int8', label='SOV · custom int8 FFN + DFlash', backend='sov', model=sov/'target', flags=['--draft-model',str(sov/'draft'),'--experimental-dynamic-int8']),
        dict(id='sov-dflash-coreml', label='SOV · palettized Core ML + DFlash', backend='sov', model=sov/'target', flags=['--draft-model',str(sov/'draft'),'--artifact',str(sov/'research/coreml-einsum-p8')]),
    ]


def summarize(spec, sessions, cases):
    by_case = {}
    for case in cases:
        samples = [m for session in sessions for r in session['recordings'] if Path(r['audio']).name == Path(case['audio']).name for m in r['measurements']]
        if not samples:
            continue
        latencies = [m['seconds'] for m in samples]
        counts = [errors(case['reference'],m['text']) for m in samples]
        median = statistics.median(latencies)
        by_case[case['case']] = dict(
            audioSeconds=case['audio_seconds'],reference=case['reference'],referenceWords=len(words(case['reference'])),
            measuredRuns=len(samples),medianSeconds=median,minSeconds=min(latencies),maxSeconds=max(latencies),
            realtimeSpeedup=case['audio_seconds']/median,realTimeFactor=median/case['audio_seconds'],
            wordErrors=counts,wer=counts[0]/len(words(case['reference'])),transcript=samples[0]['text'],
            transcriptConsistent=all(m['text']==samples[0]['text'] for m in samples),
            maxMLXActiveBytes=max(m['mlxPeakActiveBytes'] for m in samples),
        )
    return dict(id=spec['id'],label=spec['label'],backend=spec['backend'],
        modelLoadSeconds=[s['modelLoadSeconds'] for s in sessions],
        peakProcessResidentBytes=max(s.get('peakProcessResidentBytes',0) or 0 for s in sessions),
        firstModelInferenceSeconds=[s['recordings'][0]['firstInference']['seconds'] for s in sessions],
        cases=by_case)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--nemotron-root',type=Path,default=Path.home()/'Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe')
    parser.add_argument('--variants',help='Comma-separated variant IDs; default all.')
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--keep-awake',action='store_true',help='Hold macOS display/system idle-sleep assertions for each benchmark process.')
    parser.add_argument('--resume',action='store_true',help='Reuse completed receipts with matching commands and binary hashes.')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.rounds<1 or args.repetitions<1 or args.warmups<0:parser.error('Invalid measurement controls.')
    root=args.root.resolve();binary=(args.binary or root/'.build/release/mere.run').resolve()
    output=args.output or root/'.build/qualification/asr-comparison/results'
    output.mkdir(parents=True,exist_ok=True)
    reference=json.loads((root/'docs/qualification/nemotron-transcribe-native-20261028/report.json').read_text())
    cases=reference['cases']
    for case in cases:
        if digest(root/case['audio']) != case['audio_sha256']:raise ValueError('Fixture checksum changed: '+case['audio'])
    specs=variants(root,args)
    if args.variants:
        chosen=args.variants.split(',');available={s['id'] for s in specs}
        if set(chosen)-available:parser.error('Unknown variants: '+str(set(chosen)-available))
        specs=[s for name in chosen for s in specs if s['id']==name]
    binary_hash=digest(binary)
    state=dict(schemaVersion=1,platform=platform.platform(),binarySHA256=binary_hash,rounds=args.rounds,
        warmups=args.warmups,repetitions=args.repetitions,maxTokens=448,
        scoring='Lowercase, punctuation ignored, curly apostrophes normalized; internal apostrophes and spelling differences preserved.',
        keepAwake=args.keep_awake,
        cases=[{k:c[k] for k in ['case','audio','audio_sha256','audio_seconds','reference']} for c in cases],
        timings='Resident pipeline over prepared 16kHz mono arrays, including features, decode, cleanup, and stream synchronization. Loading and file I/O separate. One process per variant/session; serial execution.',
        variants=[dict(s,model=str(s['model'])) for s in specs],sessions={},failures=[],skipped=[])
    for iteration in range(args.rounds):
        order=specs if iteration%2==0 else list(reversed(specs))
        audio_order=cases if iteration%2==0 else list(reversed(cases))
        for spec in order:
            key=f"{iteration+1:02d}-{spec['id']}"
            required=[spec['model']]+[Path(spec['flags'][i+1]) for i,v in enumerate(spec['flags']) if v in ['--artifact','--draft-model']]
            if any(not p.is_dir() for p in required):
                state['skipped'].append(dict(session=key,reason='Missing local checkpoint/artifact',paths=[str(p) for p in required if not p.is_dir()]))
                continue
            cmd=[str(binary),'model','benchmark','asr']+[str(root/c['audio']) for c in audio_order]+[
                '--backend',spec['backend'],'--model-root',str(spec['model']),'--warmups',str(args.warmups),
                '--repetitions',str(args.repetitions),'--max-tokens','448','--json']+spec['flags']
            if args.keep_awake:
                if platform.system()!='Darwin':parser.error('--keep-awake requires macOS.')
                cmd=['/usr/bin/caffeinate','-di']+cmd
            print(key,flush=True)
            if args.dry_run:
                print(json.dumps(cmd),flush=True);continue
            receipt_path=output/(key+'.json');meta_path=output/(key+'.meta.json')
            metadata=dict(command=cmd,binarySHA256=binary_hash)
            reuse=args.resume and receipt_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get('command')==cmd and json.loads(meta_path.read_text()).get('binarySHA256')==binary_hash and json.loads(meta_path.read_text()).get('exitCode')==0
            if not reuse:
                started=time.perf_counter()
                with receipt_path.open('w') as stdout,(output/(key+'.log')).open('w') as stderr:
                    completed=subprocess.run(cmd,stdout=stdout,stderr=stderr,cwd=root,timeout=1800)
                metadata.update(processWallSeconds=time.perf_counter()-started,exitCode=completed.returncode)
                meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
                if completed.returncode:
                    state['failures'].append(dict(session=key,exitCode=completed.returncode,log=str(output/(key+'.log'))))
                    print('FAILED',key,completed.returncode,flush=True);continue
            receipt=json.loads(receipt_path.read_text())
            if receipt['backend']!=spec['backend'] or receipt['warmups']!=args.warmups or receipt['repetitions']!=args.repetitions:raise ValueError('Receipt does not match the requested measurement.')
            state['sessions'][key]=receipt
            summary=summarize(spec,[receipt],cases)
            print(json.dumps({name:dict(milliseconds=round(c['medianSeconds']*1000,3),wer=c['wer'],consistent=c['transcriptConsistent']) for name,c in summary['cases'].items()}),flush=True)
            (output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    state['summary']=[summarize(s,[r for key,r in state['sessions'].items() if key.endswith('-'+s['id'])],cases) for s in specs if any(key.endswith('-'+s['id']) for key in state['sessions'])]
    (output/'comparison.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state['summary'],indent=2),flush=True)


if __name__=='__main__':main()
