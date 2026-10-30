"""Check installed-model routing and single/multi-page CLI output, keeping raw streams."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if c.isalnum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library-results', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    environment = dict(os.environ, MLX_ENABLE_TF32='0')
    preflight = subprocess.run([binary, 'model', 'pull', 'vision-ocr-unlimited-2', '--preflight', '--json'],
                               capture_output=True, text=True, check=True)
    model = json.loads(preflight.stdout)['result']['models'][0]
    assert model['installed'] and model['runtime_ready']
    rows = []
    cases = [
        ('receipt', ['receipt.jpeg'], [['CASH BILL', '25/12/2018', 'MANIS', '9.00']]),
        ('receipt-and-document', ['receipt.jpeg', 'document.png'],
         [['CASH BILL', '25/12/2018', '9.00'], ['MERE RUN DOCUMENT CHECK', 'OCR-3-TEST-008', '60.50']]),
    ]
    for name, images, expected in cases:
        command = [binary, 'vision', 'ocr'] + [str((args.inputs / image).resolve()) for image in images]
        command += ['--backend', 'unlimited', '--temperature', '0', '--unlimited-crop', 'base',
                    '--max-tokens', '2048', '--quiet']
        start = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True, env=environment)
        seconds = time.monotonic() - start
        (args.output / (name + '.stdout.txt')).write_text(run.stdout)
        (args.output / (name + '.stderr.txt')).write_text(run.stderr)
        pieces = run.stdout.split('<PAGE>')
        pages = pieces[1:]
        missing = [[anchor for anchor in anchors if index >= len(pages)
                    or normalized(anchor) not in normalized(pages[index])]
                   for index, anchors in enumerate(expected)]
        passed = (run.returncode == 0 and len(pages) == len(expected) and not normalized(pieces[0])
                  and not any(missing) and 'reached --max-tokens' not in run.stderr)
        library = json.loads((args.library_results / 'receipt-base.json').read_text())
        matches_library = (run.stdout.rstrip('\n') == library['text'].rstrip('\n')) if name == 'receipt' else None
        passed = passed and matches_library is not False
        rows.append(dict(caseID=name, images=images, seconds=seconds, exitCode=run.returncode,
                         pageMarkers=len(pages), missingAnchorsByPage=missing, passed=passed,
                         stdoutSHA256=hashlib.sha256(run.stdout.encode()).hexdigest(),
                         matchesLibraryReceipt=matches_library))
        print(name, 'PASS' if passed else 'FAIL', f'{seconds:.1f}s', flush=True)
    summary = dict(installed=True, runtimeReady=True, model=model['id'], mlxEnableTF32=False, cases=rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if all(row['passed'] for row in rows) else 1)


if __name__ == '__main__':
    main()
