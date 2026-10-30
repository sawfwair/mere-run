"""Compare receipt OCR across CLI/XCTest with verified Metal artifacts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unicodedata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if c.isalnum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--test-bundle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary, bundle, checkpoint = args.binary.resolve(), args.test_bundle.resolve(), args.checkpoint.resolve()
    report = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('layout', report.parent / 'parity/compare-layout.py')
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    reference = json.loads((report.parent / 'upstream/receipt-base-bfloat16-ngram0.json').read_text())
    frozen = json.loads((report.parent / 'inputs-manifest.json').read_text())
    anchors = next(case['pages'][0]['anchors'] for case in frozen['cases'] if case['id'] == 'receipt-base')
    env = dict(os.environ, MLX_ENABLE_TF32='0')
    cases = [('jpeg-base', 'receipt.jpeg', 'base'), ('jpeg-auto', 'receipt.jpeg', 'auto'),
             ('png-base', 'receipt-canonical.png', 'base')]
    rows = []
    for name, image, crop in cases:
        source = (args.inputs / image).resolve()
        command = [str(binary), 'vision', 'ocr', str(source), '--backend', 'unlimited',
                   '--unlimited-model', str(checkpoint), '--unlimited-crop', crop,
                   '--temperature', '0', '--max-tokens', '1024', '--quiet',
                   '--unlimited-no-repeat-ngram-size', '35', '--unlimited-ngram-window', '128']
        start = time.monotonic()
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        (output / (name + '.stdout.txt')).write_text(run.stdout)
        (output / (name + '.stderr.txt')).write_text(run.stderr)
        rows.append(dict(caseID=name, command=command, seconds=time.monotonic() - start,
                         exitCode=run.returncode, inputSHA256=digest(source),
                         stdoutSHA256=hashlib.sha256(run.stdout.encode()).hexdigest(),
                         metalLibrarySHA256=digest(binary.parent / 'mlx.metallib'),
                         upstream=layout.compare(run.stdout, reference['text']),
                         missingFrozenAnchors=[anchor for anchor in anchors if normalized(anchor) not in normalized(run.stdout)]))
        print(name, run.returncode, f"{rows[-1]['seconds']:.2f}s", flush=True)
    with tempfile.TemporaryDirectory(prefix='unlimited-ocr-pair-') as temporary:
        inputs = Path(temporary)
        for _, image, _ in cases:
            link = inputs / image
            if not link.exists():
                link.symlink_to((args.inputs / image).resolve())
        manifest = dict(cases=[dict(id=name, crop=crop, maxTokens=1024,
                                   pages=[dict(image=image, anchors=['CASH BILL'])])
                               for name, image, crop in cases])
        (inputs / 'manifest.json').write_text(json.dumps(manifest))
        (output / 'pair-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        test_env = dict(env, MERERUN_TEST_MLX_DEVICE='gpu', MERERUN_TEST_DEBUG_MLX='1',
                        MERERUN_TEST_UNLIMITED_OCR='1', MERERUN_TEST_UNLIMITED_OCR_ROOT=str(checkpoint),
                        MERERUN_TEST_UNLIMITED_OCR_INPUTS=str(inputs),
                        MERERUN_TEST_UNLIMITED_OCR_OUTPUT=str(output / 'harness'))
        test_env.pop('MERERUN_TEST_UNLIMITED_OCR_CASES', None)
        command = ['/Applications/Xcode.app/Contents/Developer/usr/bin/xctest', '-XCTest',
                   'MereRunCoreTests.UnlimitedOCRCheckpointTests/testFullCheckpointDocumentsOnMetal', str(bundle)]
        run = subprocess.run(command, env=test_env, capture_output=True, text=True)
        (output / 'harness.stdout.txt').write_text(run.stdout)
        (output / 'harness.stderr.txt').write_text(run.stderr)
    texts = []
    for row in rows:
        name = row['caseID']
        native = json.loads((output / 'harness' / (name + '.json')).read_text())
        text = (output / (name + '.stdout.txt')).read_text()
        row['matchesXCTest'] = text == native['text'] + '\n'
        row['harnessUpstream'] = layout.compare(native['text'], reference['text'])
        texts.append(text)
    summary = dict(binarySHA256=digest(binary), mlxEnableTF32=False, cases=rows,
                   harnessExitCode=run.returncode, allOutputsIdentical=len(set(texts)) == 1,
                   harnessMetalLibrarySHA256=digest(bundle / 'Contents/MacOS/mlx.metallib'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert run.returncode == 0 and summary['allOutputsIdentical']
    assert all(row['exitCode'] == 0 and row['matchesXCTest'] for row in rows)
    assert all(row['metalLibrarySHA256'] == summary['harnessMetalLibrarySHA256'] for row in rows)


if __name__ == '__main__':
    main()
