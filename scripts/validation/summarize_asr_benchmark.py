#!/usr/bin/env python3
"""Publish native ASR timing receipts, reference scores, and compiler evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import xml.etree.ElementTree as ET


def sha(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=p.parse_args();root=args.root.resolve();work=root/'.build/qualification/asr-comparison'
    primary=json.loads((work/'awake-recheck/comparison.json').read_text())
    background=json.loads((work/'results/comparison.json').read_text())
    if len(primary['summary'])!=10 or primary['failures'] or primary['skipped']:raise ValueError('Controlled comparison is incomplete.')
    label='Parakeet 0.6B v3 · MLX BF16'
    for study in [primary,background]:
        for value in study['summary']+study['variants']:
            if value['id']=='parakeet-mlx':value['label']=label
    trace_path=work/'parakeet-27.2-ane-events.xml';tree=ET.parse(trace_path)
    ids={node.get('id'):node for node in tree.iter() if node.get('id')};predictions={}
    for row in tree.findall('.//row'):
        node=row.find('formatted-label')
        if node is None:continue
        node=ids[node.get('ref')] if node.get('ref') else node
        label_=node.get('fmt','')
        if 'Prediction' in label_:
            kind=label_.split(' Prediction')[0];predictions[kind]=predictions.get(kind,0)+1
    if not predictions.get('encoder_main__Op0_AneInference') or not predictions.get('decoder_main__Op0_AneInference'):
        raise ValueError('ANE encoder/decoder prediction evidence is missing.')
    manifest=json.loads((work/'parakeet-coreml/parakeet-coreml.json').read_text())
    placement=json.loads((work/'parakeet-coreml-placement.json').read_text())
    frozen={}
    for case,name in [('short','short'),('human','long')]:
        expected=json.loads((root/f'.build/qualification/sov/push/pipeline-mlx-{name}.json').read_text())['measurements'][0]['tokens']
        values=[m['sov']['tokens']==expected for study in [primary,background] for s in study['sessions'].values() if s['backend']=='sov' for r in s['recordings'] if Path(r['audio']).name==Path(next(c['audio'] for c in study['cases'] if c['case']==case)).name for m in r['measurements']]
        frozen[case]=dict(tokenCount=len(expected),measuredRuns=len(values),allMatch=all(values))
    files=['Sources/MereRunCLI/Commands/ModelBenchmarkASRCommand.swift','Sources/AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift','Sources/AudioSTT/Parakeet/ParakeetCoreMLEncoder.swift','Sources/AudioSOVTranscribeModel/Kernels/SOVDynamicInt8.metal','scripts/model-conversion/convert_parakeet_coreml.py','scripts/validation/benchmark_asr.py','scripts/validation/summarize_asr_benchmark.py','Package.resolved']
    report=dict(schemaVersion=1,date='2026-11-13',primary=primary,initialUncontrolledStudy=background,
        checkpointRevisions={
            'parakeet-mlx':json.loads((work/'parakeet-hub.json').read_text())['sha'],
            'parakeet-coreml':manifest['source']['revision'],
            'qwen-8bit':json.loads((work/'qwen-hub.json').read_text())['sha'],
            'whistle':json.loads((work/'whistle-hub.json').read_text())['sha'],
            'sov':'505d8fae243924c0d8acdc622f7f0abab344eba3',
            'sov-dflash':'8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05'},
        processReceipts={phase:{f.stem.removesuffix('.meta'):json.loads(f.read_text()) for f in (work/folder).glob('*.meta.json')} for phase,folder in [('controlled','awake-recheck'),('initial','results')]},
        sourceSHA256={f:sha(root/f) for f in files},
        parakeetCompiler=dict(conversion=manifest['conversion'],verification=manifest['verification'],manifestSHA256=sha(work/'parakeet-coreml/parakeet-coreml.json'),
            allNonconstantOperationsPreferANE=placement['allNonconstantOperationsPreferANE'],actualANEPredictionCounts=predictions),
        sovFrozenTokenParity=frozen,
        validation=dict(releaseBuild='passed',swiftFocusedTestsPassed=15,pythonConverterTestsPassed=8,strictLint='passed',fullGate='deferred at user request'),
        executionOrder=[q.stem.removesuffix('.meta') for q in sorted((work/'awake-recheck').glob('*.meta.json'),key=lambda v:v.stat().st_mtime_ns)],
        precisionNotes={'parakeet-mlx':'Source checkpoint is FP32; native ModelWeightsLoader casts floating weights to BF16.','parakeet-coreml':'Encoder and decoder graphs are FP16, with exact base128 token decisions.','qwen-8bit':'Default 1.7B 8-bit checkpoint.','whistle':'Published packed CQ2/CQ4 weights, depth 8; beam width as labelled.','nemotron-int4':'Default routed-expert INT4 checkpoint; other weights keep native precision.','sov':'Original BF16 target and DFlash; experimental int8 affects audio FFNs only.'})
    dest=root/'docs/qualification';dest.mkdir(exist_ok=True)
    prefix='asr-comparison-2026-11-13'
    (dest/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    shutil.copyfile(trace_path,dest/(prefix+'-ane-events.xml'))
    rows=[];memory=[];ranges=[]
    def duration(v):return f'{v*1000:.1f} ms' if v<1 else f'{v:.3f} s'
    for s in sorted(primary['summary'],key=lambda v:v['cases']['human']['medianSeconds']):
        a,b=s['cases']['short'],s['cases']['human']
        rows.append(f"| {s['label']} | {duration(a['medianSeconds'])} | {duration(b['medianSeconds'])} | {b['realtimeSpeedup']:.1f}× | {b['wordErrors'][0]}/129 ({b['wer']*100:.2f}%) |")
        memory.append(f"| {s['label']} | {s['modelLoadSeconds'][0]:.3f} s | {s['firstModelInferenceSeconds'][0]:.3f} s | {s['peakProcessResidentBytes']/2**30:.2f} GiB |")
        ranges.append(f"| {s['label']} | {duration(a['minSeconds'])}–{duration(a['maxSeconds'])} | {duration(b['minSeconds'])}–{duration(b['maxSeconds'])} |")
    text='''# Native ASR comparison — 2026-11-13

All five supported ASR families ran through the same release Swift executable
on an Apple M5 Max with 128 GiB unified memory. The table compares current native
pipelines, their default checkpoints, and selected decoder/provider controls.
All models transcribed the short reference without word errors. Long-recording
scores are from one 129-word human-speech sample; they do not establish general
ASR accuracy or language coverage.

## Resident latency

| Model / execution | 3.52 s recording | 46.57 s recording | Long audio / inference time | Long word errors |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

Each entry is a median of five measured runs after two warmups on each recording.
One fresh process loads each variant once and keeps it resident across both
recordings. The controlled session holds temporary display and system idle-sleep
assertions for each process with `caffeinate -di`. All inference processes run
serially, outside Instruments and without concurrent builds. File decoding and
checkpoint loading are excluded from resident latency; feature extraction,
prefill, generation, alignment when applicable, cleanup, and stream completion
are included. The output budget is 448; Whistle retains its native per-window
318-token ceiling. Whistle uses full decoder depth and text-only output.
Parakeet's intrinsic alignment remains included.

These are complete backend paths with their own windowing policies. Parakeet
MLX processes the long recording as one window, whereas its static Core ML
encoder uses five overlapping/padded windows. SOV uses two 30-second windows.
Core ML decoder speed and encoder speed must therefore be interpreted alongside
window counts, rather than attributed entirely to hardware.

## Loading and memory

| Model / execution | Model preparation | First inference after preparation | Peak process RSS |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(memory)+'''

These are observed fresh-process timings without clearing filesystem or shader
caches. The first input is the short clip. Preparation and first inference are
separate: their sum plus audio decoding approximates the first request.
Process RSS includes loading and warmup and comes from `getrusage`; it does not
measure every driver/ANE-service allocation. Per-run MLX active-memory peaks,
which exclude Core ML allocations, are preserved separately in the JSON.
Core ML process RSS was about 1.3 GiB during the initial compilation/load and
0.17 GiB with the warmed service cache; RSS alone cannot establish total ANE
model memory.

## Variability and checks

| Model / execution | Short measured range | Long measured range |
| --- | ---: | ---: |
'''+ '\n'.join(ranges)+'''

The raw receipt also retains the initial two-session comparison, which reversed
model and recording order and recorded ten measured runs per variant/clip.
That batch showed substantial SOV timing swings, including slower custom-int8
runs. A separate held-awake session returned timings closer to the earlier
standalone SOV measurements. The primary table uses the same explicit awake
condition for every model. This association does not isolate idle power as the
sole cause of the earlier variance, and small SOV end-to-end differences should
not be treated as a guaranteed speedup. No observations from the initial batch
were discarded.

Scoring lowercases text, ignores punctuation, normalizes curly apostrophes, and
preserves contractions and spelling differences. Both recordings, references,
checksums, every measured transcript, and all timing samples are in the JSON.
All SOV measured token sequences were checked against the earlier frozen
13-/138-token references; see `sovFrozenTokenParity` in the receipt.

The Parakeet providers both scored 0/11 short and 4/129 long word errors, but
their long transcripts differ in proper-name spelling. Different windowing and
FP16/BF16 arithmetic are part of these provider paths; bit-exact cross-provider
transcript parity is not established by equal WER.

## Xcode compatibility and ANE execution

The converter uses the selected installed Xcode and records its actual version.
The loader validates the artifact format, source pin, conversion dependencies,
and complete file-hash closure, with compiler version retained as provenance.
It accepts compatible artifacts without matching a hard-coded Xcode release.
Legacy, 27.2, and future compiler metadata have regression coverage; blank
provenance remains invalid.

The new Parakeet artifact was built with Xcode 27.2 / build 27B52. Encoder trace
and decoder-state checks had zero maximum absolute error before FP16 conversion.
Every nonconstant encoder/decoder operation prefers ANE. A separate Instruments
trace recorded two encoder and forty decoder ANE prediction intervals, confirming
hardware execution. Profiling was excluded from benchmark timing.

## Reproduce

```bash
swift build -c release --product mere.run
python3 scripts/validation/benchmark_asr.py --dry-run
python3 scripts/validation/benchmark_asr.py --rounds 1 --keep-awake \\
  --output .build/qualification/asr-comparison/awake-recheck
```

The script requires local checkpoints at the paths shown by its plan and never
downloads models while scoring. `model benchmark asr` accepts arbitrary installed
roots for individual models. See [benchmarking](../benchmarking.md#native-asr-timing)
for options. The source, binary, checkpoint revisions, command receipts, compiler
verification, and full samples are recorded in the [JSON report](./'''+prefix+'''.json).
The independent [ANE hardware event export](./'''+prefix+'''-ane-events.xml) records
execution. The release build, 15 focused Swift tests, eight converter tests,
and strict lint passed. The full repository gate remains deferred as requested.
'''
    (dest/(prefix+'.md')).write_text(text)
    print('\n'.join(rows),flush=True)
    print('SOV frozen parity:',frozen,flush=True)


if __name__=='__main__':main()
