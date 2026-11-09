#!/usr/bin/env python3
"""Compare matched native Glimmer likelihood windows against BF16.

This fixed text diagnostic does not replace coding or vision task qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

WINDOWS = 16
SCORED_TOKENS = 2048
MAX_RATIO = 1.05


def read(path):
    rows = [r for r in map(json.loads, Path(path).read_text().splitlines()) if r['kind'] == 'likelihood']
    if len(rows) != WINDOWS or len({r['id'] for r in rows}) != WINDOWS:
        raise ValueError('Expected 16 complete, distinct held-out windows')
    if len({(r['modelRoot'], r.get('bits')) for r in rows}) != 1:
        raise ValueError('A report must describe one checkpoint precision')
    rows.sort(key=lambda r: r['id'])
    for row in rows:
        if len(row['tokens']) != SCORED_TOKENS + 1 or len(row['nll']) != SCORED_TOKENS or len(row['top1']) != SCORED_TOKENS:
            raise ValueError('Incomplete or misaligned scored token window')
        if not all(math.isfinite(x) for x in row['nll']):
            raise ValueError('Nonfinite likelihood score')
    return rows


def compare(reference, candidate):
    ref, cand = read(reference), read(candidate)
    windows = []
    for a, b in zip(ref, cand):
        if a['id'] != b['id'] or a['tokens'] != b['tokens']:
            raise ValueError('Candidate and baseline must score exactly the same token sequences')
        delta = statistics.fmean(y - x for x, y in zip(a['nll'], b['nll']))
        windows.append(dict(id=a['id'], delta_nll=delta, perplexity_ratio=math.exp(delta),
                            top1_agreement=sum(x == y for x, y in zip(a['top1'], b['top1'])) / SCORED_TOKENS))
    deltas = [row['delta_nll'] for row in windows]
    rng = random.Random(0)
    boot = sorted(math.exp(statistics.fmean(rng.choices(deltas, k=WINDOWS))) for _ in range(10000))
    interval = [boot[int((len(boot) - 1) * p)] for p in [.025, .975]]
    ratio = math.exp(statistics.fmean(deltas))
    token_suite = json.dumps([(r['id'], r['tokens']) for r in ref], separators=(',', ':')).encode()
    return dict(schema_version=1, reference=str(reference), candidate=str(candidate),
                reference_bits=ref[0].get('bits'), candidate_bits=cand[0].get('bits'),
                suite_sha256=hashlib.sha256(token_suite).hexdigest(), windows=WINDOWS,
                scored_tokens=WINDOWS * SCORED_TOKENS,
                reference_perplexity=math.exp(statistics.fmean(x for r in ref for x in r['nll'])),
                candidate_perplexity=math.exp(statistics.fmean(x for r in cand for x in r['nll'])),
                perplexity_ratio=ratio, ratio_95_percent_window_bootstrap=interval,
                top1_agreement=statistics.fmean(row['top1_agreement'] for row in windows),
                maximum_perplexity_ratio=MAX_RATIO,
                passes_likelihood_diagnostic=ratio <= MAX_RATIO and interval[1] <= MAX_RATIO,
                by_window=windows,
                limitation='Bounded text regression diagnostic; coding, vision, and general quality require separate evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'by_window'}))
