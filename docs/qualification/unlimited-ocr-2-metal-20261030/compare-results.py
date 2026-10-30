"""Compare preserved native and upstream transcriptions without changing either output."""
import difflib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if c.isalnum())


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'inputs-manifest.json').read_text())
    items = {item['id']: item for item in manifest['cases']}
    rows = []
    for case_id, size, native_folder in [
        ('receipt-base', 0, 'baseline-without-ngram'),
        ('w9-form-base', 0, 'baseline-without-ngram'),
        ('w9-form-base', 35, 'results'),
    ]:
        native_path = root / native_folder / (case_id + '.json')
        reference_path = root / 'upstream' / f'{case_id}-bfloat16-ngram{size}.json'
        native = json.loads(native_path.read_text())
        reference = json.loads(reference_path.read_text())
        anchors = items[case_id]['pages'][0]['anchors']
        row = dict(caseID=case_id, noRepeatNgramSize=size,
                   nativeRecord=str(native_path.relative_to(root)),
                   upstreamRecord=str(reference_path.relative_to(root)),
                   exactTextMatch=native['text'] == reference['text'])
        for label, text in [('native', native['text']), ('upstream', reference['text'])]:
            row[label + 'MissingAnchors'] = [anchor for anchor in anchors if normalized(anchor) not in normalized(text)]
            row[label + 'LongestDottedLine'] = max((part.count('.') for part in re.findall(r'(?:\.\s*){2,}', text)), default=0)
        row.update(nativeTokens=native['tokensGenerated'], upstreamTokens=len(reference['token_ids']),
                   nativeReachedCap=native['reachedTokenLimit'], upstreamFinishReason=reference['finish_reason'],
                   upstreamMaxTokens=reference['maxTokens'], nativeMaxTokens=items[case_id]['maxTokens'])
        rows.append(row)
        diff = difflib.unified_diff(native['text'].splitlines(keepends=True),
                                    reference['text'].splitlines(keepends=True),
                                    fromfile=row['nativeRecord'], tofile=row['upstreamRecord'])
        (root / 'upstream' / f'{case_id}-ngram{size}.diff').write_text(''.join(diff))
    (root / 'upstream' / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['caseID'], 'ngram', row['noRepeatNgramSize'],
              'native missing', row['nativeMissingAnchors'], 'upstream missing', row['upstreamMissingAnchors'])


if __name__ == '__main__':
    main()
