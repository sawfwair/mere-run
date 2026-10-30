"""Independently check page content/order and summarize the saved Metal records."""
import argparse
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if c.isalnum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = []
    for item in manifest['cases']:
        record = json.loads((args.results / (item['id'] + '.json')).read_text())
        text = record['text']
        pieces = text.split('<PAGE>')
        pages = pieces[1:]
        missing, unexpected = [], []
        for index, page in enumerate(item['pages']):
            content = normalized(pages[index]) if index < len(pages) else ''
            missing.append([anchor for anchor in page['anchors'] if normalized(anchor) not in content])
            unexpected.append([anchor for anchor in page.get('forbidden', []) if normalized(anchor) in content])
        assert missing == record['missingAnchorsByPage']
        assert unexpected == record['unexpectedAnchorsByPage']
        assert record['checkpointRevision'] == 'c14071b549f4fcab10909f3c4acc6a2edb366947'
        assert record['mlxDevice'] == 'gpu'
        assert record['noRepeatNgramSize'] == 35
        assert record['ngramWindow'] == (1024 if len(item['pages']) > 1 else 128)
        boxes = [tuple(map(int, match)) for match in re.findall(r'<BLOCK>[^\[\n]+\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]', text)]
        invalid = [box for box in boxes if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)]
        passed = (not record['reachedTokenLimit'] and record['pagesGenerated'] == len(item['pages'])
                  and len(pages) == len(item['pages']) and not normalized(pieces[0])
                  and not any(missing) and not any(unexpected))
        cases.append(dict(
            id=item['id'], passed=passed, crop=item['crop'], expectedPages=len(item['pages']),
            actualPages=len(pages), anchorCount=sum(len(page['anchors']) for page in item['pages']),
            missingAnchorsByPage=missing, unexpectedAnchorsByPage=unexpected,
            tokens=record['tokensGenerated'], reachedTokenLimit=record['reachedTokenLimit'],
            wallSeconds=record['seconds'], loadSeconds=record['loadSeconds'],
            visionSeconds=record['visionSeconds'], prefillSeconds=record['prefillSeconds'],
            decodeSeconds=record['decodeSeconds'],
            decodeTokensPerSecond=record['tokensGenerated'] / record['decodeSeconds'],
            peakMLXGiB=record['peakMemoryBytes'] / 1024 ** 3,
            emittedBlockBoxes=len(boxes), invalidBlockBoxes=invalid,
        ))
    summary = dict(
        schemaVersion=1, checkpoint='baidu/Unlimited-OCR-2',
        revision='c14071b549f4fcab10909f3c4acc6a2edb366947',
        device=record['device'], physicalMemoryGiB=record['physicalMemoryBytes'] / 1024 ** 3,
        caseCount=len(cases), passedCount=sum(case['passed'] for case in cases),
        totalPages=sum(case['expectedPages'] for case in cases),
        totalAnchors=sum(case['anchorCount'] for case in cases),
        totalTokens=sum(case['tokens'] for case in cases),
        maxPeakMLXGiB=max(case['peakMLXGiB'] for case in cases),
        invalidBlockBoxCount=sum(len(case['invalidBlockBoxes']) for case in cases),
        cases=cases,
    )
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    for case in cases:
        print(f"{case['id']}: {'PASS' if case['passed'] else 'FAIL'}, {case['tokens']} tokens, "
              f"{case['decodeTokensPerSecond']:.1f} decode tok/s, {case['peakMLXGiB']:.2f} GiB MLX peak")
    raise SystemExit(0 if all(case['passed'] for case in cases) else 1)


if __name__ == '__main__':
    main()
