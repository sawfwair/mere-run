"""Compare current native JPEG and canonical PNG through the public CLI."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / 'cli'
    output.mkdir(exist_ok=True)
    module = importlib.util.spec_from_file_location('layout', root.parent / 'parity/compare-layout.py')
    layout = importlib.util.module_from_spec(module)
    module.loader.exec_module(layout)
    reference = json.loads((root.parent / 'upstream/receipt-base-bfloat16-ngram0.json').read_text())
    rows, texts = [], []
    binary = args.binary.resolve()
    for name, image, crop in [('jpeg-base', 'receipt.jpeg', 'base'),
                              ('jpeg-auto', 'receipt.jpeg', 'auto'),
                              ('png-base', 'receipt-canonical.png', 'base')]:
        source = (args.inputs / image).resolve()
        command = [str(binary), 'vision', 'ocr', str(source), '--backend', 'unlimited',
                   '--unlimited-crop', crop, '--temperature', '0', '--max-tokens', '1024',
                   '--unlimited-no-repeat-ngram-size', '35', '--unlimited-ngram-window', '128', '--quiet']
        start = time.monotonic()
        run = subprocess.run(command, env=dict(os.environ, MLX_ENABLE_TF32='0'), capture_output=True, text=True)
        seconds = time.monotonic() - start
        (output / (name + '.stdout.txt')).write_text(run.stdout)
        (output / (name + '.stderr.txt')).write_text(run.stderr)
        comparison = layout.compare(run.stdout, reference['text'])
        passed = run.returncode == 0 and run.stdout.count('<PAGE>') == 1 and not run.stderr
        rows.append(dict(caseID=name, command=command, seconds=seconds, exitCode=run.returncode,
                         passed=passed, inputSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         stdoutSHA256=hashlib.sha256(run.stdout.encode()).hexdigest(), upstream=comparison))
        texts.append(run.stdout)
        print(name, run.returncode, f'{seconds:.3f}s', flush=True)
    summary = dict(binarySHA256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   binaryBytes=binary.stat().st_size, mlxEnableTF32=False, cases=rows,
                   allOutputsIdentical=len(set(texts)) == 1,
                   matchesPreviousCanonicalPNG=texts[0] == (root.parent / 'parity/canonical.stdout.txt').read_text())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert summary['allOutputsIdentical'] and all(row['passed'] for row in rows)


if __name__ == '__main__':
    main()
