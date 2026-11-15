#!/usr/bin/env python3
"""Pair projected-tail trimming with full padding on one resident SOV model."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess


def summarize(report):
    measured = report['measurements']
    assert measured and len(measured) % 2 == 0
    pairs = []
    baseline, candidate = [], []
    for index in range(0, len(measured), 2):
        pair = [index, index + 1]
        b = next(i for i in pair if measured[i].get('audioTokenMargin') is None)
        c = next(i for i in pair if measured[i].get('audioTokenMargin') is not None)
        assert measured[b]['fullAudioTokens'] == measured[c]['fullAudioTokens']
        pairs.append(dict(resident=1-report['residentSeconds'][c]/report['residentSeconds'][b],
                          prefill=1-measured[c]['prefillSeconds']/measured[b]['prefillSeconds'],
                          prefillSavedSeconds=measured[b]['prefillSeconds']-measured[c]['prefillSeconds'],
                          generation=1-measured[c]['generationSeconds']/measured[b]['generationSeconds'],
                          tokensAgree=measured[b]['tokens']==measured[c]['tokens']))
        baseline.append(measured[b]); candidate.append(measured[c])
    rng = random.Random(0)
    boot = sorted(statistics.median(rng.choices([p['resident'] for p in pairs], k=len(pairs))) for _ in range(2000))
    return dict(pairs=pairs, medianResidentReduction=statistics.median(p['resident'] for p in pairs),
                medianPrefillReduction=statistics.median(p['prefill'] for p in pairs),
                medianPrefillSavedSeconds=statistics.median(p['prefillSavedSeconds'] for p in pairs),
                bootstrapResident95Percent=[boot[50],boot[1949]], wins=sum(p['resident']>0 for p in pairs),
                tokensAgree=sum(p['tokensAgree'] for p in pairs),
                baselinePromptTokens=baseline[0]['promptTokens'], candidatePromptTokens=candidate[0]['promptTokens'],
                baselineRepeatable=all(m['tokens']==baseline[0]['tokens'] for m in baseline),
                candidateRepeatable=all(m['tokens']==candidate[0]['tokens'] for m in candidate),
                baselineMedianResidentSeconds=statistics.median(m['totalSeconds'] for m in baseline),
                candidateMedianResidentSeconds=statistics.median(m['totalSeconds'] for m in candidate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--packed-weights', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('.build/release/mere.run'))
    parser.add_argument('--margins', type=int, nargs='+', default=[25,12,0])
    parser.add_argument('--clips', nargs='+', choices=['short','long','wrapped'], default=['short','long','wrapped'])
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    binarySHA256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    clips = dict(short='Tests/MereRunCoreTests/Fixtures/D1/released/speech.wav',
                 long='Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav',
                 wrapped='.build/qualification/sov/remaining/wrapped.wav')
    summaries = []
    for margin in (args.margins[::-1] if args.reverse else args.margins):
        for clip in (args.clips[::-1] if args.reverse else args.clips):
            stem = f'margin-{margin}-{clip}'
            cmd = [str(args.binary), 'model', 'benchmark', 'sov', clips[clip], '--model-root', args.model_root,
                   '--draft-model', args.draft_model, '--packed-weights', args.packed_weights,
                   '--experimental-decoder', 'q8-metal', '--experimental-attention', '--generation-experiment', 'rollback',
                   '--audio-token-margin', str(margin), '--interleave-audio-tokens', '--warmups', '1',
                   '--repetitions', str(args.repetitions), '--json']
            if args.reverse:
                cmd.append('--audio-tokens-first')
            print('running', stem, flush=True)
            with (args.output/(stem+'.json')).open('w') as stdout, (args.output/(stem+'.log')).open('w') as stderr:
                subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)
            result = dict(clip=clip,margin=margin,**summarize(json.loads((args.output/(stem+'.json')).read_text())))
            summaries.append(result)
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
