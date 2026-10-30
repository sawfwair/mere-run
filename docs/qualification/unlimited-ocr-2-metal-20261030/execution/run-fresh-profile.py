"""Qualify a rebuilt Metal profile, restoring product resources afterward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    for name in ['products', 'kernels', 'inputs', 'checkpoint', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    products, kernels, output = args.products.resolve(), args.kernels.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    subprocess.run([str(root / 'scripts/build_mlx_metallib.sh'), '--verify-only', str(kernels)], check=True)
    resources = products / 'mlx-swift_Cmlx.bundle/Contents/Resources'
    assert resources.resolve().is_relative_to(products), 'Only build-product resources can be replaced.'
    backup = {name: (resources / name).read_bytes() for name in ['default.metallib', 'default.metallib.version']}
    env = dict(os.environ, MLX_ENABLE_TF32='0', MERERUN_TEST_MLX_DEVICE='gpu', MERERUN_TEST_DEBUG_MLX='1',
               MERERUN_TEST_UNLIMITED_OCR='1', MERERUN_TEST_UNLIMITED_OCR_ROOT=str(args.checkpoint.resolve()),
               MERERUN_TEST_UNLIMITED_OCR_INPUTS=str(args.inputs.resolve()),
               MERERUN_TEST_UNLIMITED_OCR_OUTPUT=str(output), MERERUN_TEST_UNLIMITED_OCR_CLI=str(products / 'mere.run'))
    cases = [('cross-process', 'MereRunCoreTests',
              'MereRunCoreTests.UnlimitedOCRCheckpointTests/testReceiptMatchesPublicCLIOnMetal', 1),
             ('numerical', 'UnlimitedOCRRuntimeTests', 'UnlimitedOCRRuntimeTests.UnlimitedOCRRuntimeTests', 18)]
    results = []
    try:
        for name in backup:
            (resources / name).write_bytes((kernels / name).read_bytes())
        for name, bundle, test, expected in cases:
            command = ['/Applications/Xcode.app/Contents/Developer/usr/bin/xctest', '-XCTest', test,
                       str(products / (bundle + '.xctest'))]
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            (output / (name + '.stdout.txt')).write_text(run.stdout)
            (output / (name + '.stderr.txt')).write_text(run.stderr)
            counts = re.findall(r'Executed (\d+) tests?[, ]', run.stdout + run.stderr)
            count = max(map(int, counts), default=0)
            results.append(dict(case=name, exitCode=run.returncode, tests=count))
            assert run.returncode == 0 and count == expected, name
        summary = dict(cases=results, metalLibrarySHA256=hashlib.sha256((kernels / 'default.metallib').read_bytes()).hexdigest(),
                       nativeEqualsCLI=(output / 'cross-process.native.txt').read_bytes()
                       == (output / 'cross-process.cli.stdout.txt').read_bytes())
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        assert summary['nativeEqualsCLI']
    finally:
        for name, contents in backup.items():
            (resources / name).write_bytes(contents)


if __name__ == '__main__':
    main()
