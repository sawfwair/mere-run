#!/usr/bin/env python3
"""Run one planned native CLI qualification case and retain its evidence.

Requires Pillow and numpy for artifact inspection. Inference uses only the
selected mere.run executable. Refuses to overwrite an existing case receipt.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--case', required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
case = next(case for case in plan['cases'] if case['id'] == args.case)
args.output.mkdir(parents=True, exist_ok=True)
output = args.output.resolve()
image_path = output / (args.case + '.png')
record_path = output / (args.case + '.json')
if record_path.exists() or image_path.exists():
    raise SystemExit('Case already has evidence; choose a fresh output directory.')
command = [str(args.binary.resolve()), 'image', 'generate', '--model', plan['model'],
           '--prompt', case['prompt'], '--width', str(case['width']), '--height', str(case['height']),
           '--steps', str(case['steps']), '--seed', str(case['seed']), '--output', str(image_path)]
if 'cfg' in case:
    command += ['--cfg', str(case['cfg']), '--negative-prompt', case['negative_prompt']]
for index, name in enumerate(case.get('inputs', [])):
    path = output / name
    if not path.is_file():
        raise SystemExit(f'Missing reference: {path}')
    command += ['--input' if index == 0 else '--ref-image', str(path)]
preflight = subprocess.run(command + ['--preflight', '--json'], capture_output=True, text=True)
(output / (args.case + '.preflight.json')).write_text(preflight.stdout)
(output / (args.case + '.preflight.stderr')).write_text(preflight.stderr)
report = json.loads(preflight.stdout)
if preflight.returncode != 0 or report['status'] == 'blocked':
    raise SystemExit(f'Preflight blocked: {report.get("diagnostics")}')
record = dict(case=case, command=command, preflight_status=report['status'],
              binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              checkpoint_revision=plan['checkpoint_revision'],
              input_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                            for name in case.get('inputs', [])})
stdout_path = output / (args.case + '.stdout.log')
stderr_path = output / (args.case + '.stderr.log')
start = time.monotonic()
events = []
with stdout_path.open('w') as stdout, stderr_path.open('w', buffering=1) as stderr:
    process = subprocess.Popen(['/usr/bin/time', '-l'] + command + ['--receipt', '--progress-json'],
                               stdout=stdout, stderr=subprocess.PIPE, text=True)
    for line in process.stderr:
        stderr.write(line)
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(dict(elapsed_seconds=time.monotonic() - start, payload=event))
    exit_code = process.wait()
record.update(exit_code=exit_code, elapsed_seconds=time.monotonic() - start, progress_events=events)
timing = re.search(r'(\d+)\s+maximum resident set size', stderr_path.read_text())
record['maximum_resident_bytes'] = int(timing[1]) if timing else None
footprint = re.search(r'(\d+)\s+peak memory footprint', stderr_path.read_text())
record['peak_memory_footprint_bytes'] = int(footprint[1]) if footprint else None
if exit_code == 0:
    receipt = json.loads(stdout_path.read_text().splitlines()[-1])
    if receipt['event'] != 'result' or receipt['exit'] != 0:
        raise ValueError(f'Invalid success receipt: {receipt}')
    record['receipt'] = receipt
    with Image.open(image_path) as image:
        image.load()
        rgba = np.asarray(image.convert('RGBA'))
        record.update(format=image.format, mode=image.mode, size=list(image.size),
                      sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                      pixel_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
                      alpha_min=int(rgba[..., 3].min()), alpha_max=int(rgba[..., 3].max()),
                      transparent_fraction=float(np.mean(rgba[..., 3] < 16)),
                      opaque_fraction=float(np.mean(rgba[..., 3] > 239)),
                      rgb_standard_deviation=float(rgba[..., :3].std()))
        if image.size != (case['width'], case['height']) or image.format != 'PNG':
            raise ValueError('Output dimensions or format differ from request.')
record['visual_inspection'] = 'pending'
record_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(exit_code)
