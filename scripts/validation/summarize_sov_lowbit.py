#!/usr/bin/env python3
"""Compare held-out SOV expert policies with a paired Q8 recognition baseline.

WER normalization: Unicode NFKC and case folding, punctuation replaced by spaces,
then whitespace tokenization. This is an exploratory comparison, not the model
publisher's benchmark normalizer. Bootstrap resamples whole clips, seed zero.
"""
import argparse
import json
from pathlib import Path
import random
import unicodedata


def words(text):
    return ''.join(c if unicodedata.category(c)[0] in 'LMN' or c.isspace() else ' '
                   for c in unicodedata.normalize('NFKC', text).casefold()).split()


def distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, actual in enumerate(hypothesis, 1):
            next_row.append(min(row[j] + 1, next_row[j - 1] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def summarize(baseline, candidate):
    b = {entry['sample']['id']: entry for entry in baseline['entries']}
    c = {entry['sample']['id']: entry for entry in candidate['entries']}
    assert b.keys() == c.keys(), 'Evaluation sets differ'
    rows = []
    for key in b:
        assert b[key]['sample'] == c[key]['sample'], 'Evaluation inputs differ'
        sample = b[key]['sample']
        reference = words(sample['reference'])
        rows.append(dict(id=key, language=sample['language'], words=len(reference),
                         baselineErrors=distance(reference, words(b[key]['measurement']['text'])),
                         candidateErrors=distance(reference, words(c[key]['measurement']['text'])),
                         tokensAgree=b[key]['measurement']['tokens'] == c[key]['measurement']['tokens'],
                         hitTokenLimit=len(c[key]['measurement']['tokens']) == 448))
    def group(rows):
        count = sum(row['words'] for row in rows)
        before = sum(row['baselineErrors'] for row in rows) / count
        after = sum(row['candidateErrors'] for row in rows) / count
        return dict(clips=len(rows), referenceWords=count, baselineWER=before, candidateWER=after,
                    differencePercentagePoints=100 * (after - before), tokensAgree=sum(row['tokensAgree'] for row in rows),
                    tokenLimitCount=sum(row['hitTokenLimit'] for row in rows))
    result = group(rows)
    result['languages'] = {language: group([r for r in rows if r['language'] == language]) for language in sorted({r['language'] for r in rows})}
    rng = random.Random(0)
    differences = sorted(group(rng.choices(rows, k=len(rows)))['differencePercentagePoints'] for _ in range(2000))
    result['pairedClipBootstrap95Percent'] = [differences[50], differences[1949]]
    result['rows'] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    results = {path.stem: summarize(baseline, json.loads(path.read_text())) for path in args.candidates}
    args.output.write_text(json.dumps(dict(normalizer=__doc__, comparisons=results), indent=2) + '\n')
    for name, result in results.items():
        print(name, f"WER {result['baselineWER']:.2%} -> {result['candidateWER']:.2%}",
              f"delta {result['differencePercentagePoints']:+.3f} pp", 'CI', result['pairedClipBootstrap95Percent'],
              'tokens', result['tokensAgree'], '/', result['clips'])


if __name__ == '__main__':
    main()
