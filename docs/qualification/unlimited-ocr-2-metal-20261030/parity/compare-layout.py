"""Compare raw block bodies, coordinates, and price strings against pinned upstream."""
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import re


def blocks(text):
    result = []
    for page, content in enumerate(text.split('<PAGE>')[1:]):
        for header, body in re.findall(r'<BLOCK>(.*?)<FORK>(.*?)(?=<BLOCK>|$)', content, flags=re.S):
            match = re.fullmatch(r'(.*?)\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]', header)
            if match:
                result.append(dict(page=page, type=match[1], box=list(map(int, match.groups()[1:])),
                                   body=' '.join(body.split())))
    return result


def compare(native, reference):
    actual, expected = blocks(native), blocks(reference)
    candidates, native_groups = defaultdict(list), defaultdict(list)
    for block in expected:
        candidates[(block['page'], block['type'], block['body'])].append(block)
    for block in actual:
        native_groups[(block['page'], block['type'], block['body'])].append(block)
    matched, unmatched, upstream_unmatched, duplicate_groups = [], [], [], []
    text_matches = 0
    for key in sorted(native_groups.keys() | candidates.keys()):
        left, right = native_groups[key], candidates[key]
        common = min(len(left), len(right))
        text_matches += common
        unmatched.extend(left[common:])
        upstream_unmatched.extend(right[common:])
        if len(left) == len(right) == 1:
            matched.append([a - b for a, b in zip(left[0]['box'], right[0]['box'])])
        elif common:
            duplicate_groups.append(dict(type=key[1], body=key[2], nativeCount=len(left), upstreamCount=len(right)))
    differences = [abs(value) for row in matched for value in row]
    return dict(nativeBlocks=len(actual), upstreamBlocks=len(expected), exactTextAndTypeMatches=text_matches,
                unambiguousCoordinatePairs=len(matched), duplicateBodyGroupsExcluded=duplicate_groups,
                exactBoxesAmongUnambiguousPairs=sum(all(value == 0 for value in row) for row in matched),
                meanAbsoluteCoordinateDelta=sum(differences) / len(differences) if differences else None,
                maximumAbsoluteCoordinateDelta=max(differences, default=None),
                nativeBlockTypes=dict(Counter(block['type'] for block in actual)),
                upstreamBlockTypes=dict(Counter(block['type'] for block in expected)),
                nativeOnly=[dict(type=block['type'], body=block['body'][:400]) for block in unmatched],
                upstreamOnly=[dict(type=block['type'], body=block['body'][:400]) for block in upstream_unmatched],
                nativeExactPrice9000=sum(block['body'] == '9.000' for block in actual),
                upstreamExactPrice9000=sum(block['body'] == '9.000' for block in expected))


def banned_selected_tokens(tokens, size=35, window=128):
    # If the unconstrained greedy winner is never banned, enabling this constraint
    # preserves the whole reference path. This checks the saved receipt trace.
    banned = []
    for step, token in enumerate(tokens):
        if step < size:
            continue
        prefix = tokens[step-size+1:step]
        for start in range(max(0, step-window), step-size+1):
            if tokens[start:start+size-1] == prefix and tokens[start+size-1] == token:
                banned.append(step)
                break
    return banned


def main():
    output = Path(__file__).resolve().parent
    root = output.parent
    cases = [('receipt-base', 'receipt-base-bfloat16-ngram0.json'),
             ('w9-form-base', 'w9-form-base-bfloat16-ngram35.json')]
    rows = []
    for case, reference_name in cases:
        reference = json.loads((root / 'upstream' / reference_name).read_text())
        before = json.loads((root / 'results' / (case + '.json')).read_text())
        after = json.loads((output / 'results' / (case + '.json')).read_text())
        row = dict(caseID=case, before=compare(before['text'], reference['text']),
                   after=compare(after['text'], reference['text']),
                   referenceNoRepeatSize=reference['noRepeatNgramSize'], nativeNoRepeatSize=after['noRepeatNgramSize'])
        if case == 'receipt-base':
            row['referenceChosenTokensBannedBy35TokenConstraint'] = banned_selected_tokens(reference['token_ids'])
        rows.append(row)
        for name, record in [('before', before), ('after', after)]:
            diff = difflib.unified_diff(record['text'].splitlines(keepends=True), reference['text'].splitlines(keepends=True),
                                        fromfile=f'{name}-native/{case}', tofile=f'upstream/{reference_name}')
            (output / f'{case}-{name}.diff').write_text(''.join(diff))
    (output / 'layout-comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['caseID'], json.dumps({phase: {key: row[phase][key] for key in
              ['exactTextAndTypeMatches', 'meanAbsoluteCoordinateDelta', 'maximumAbsoluteCoordinateDelta',
               'nativeExactPrice9000', 'nativeBlocks']} for phase in ['before', 'after']}))


if __name__ == '__main__':
    main()
