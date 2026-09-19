#!/usr/bin/env python3
"""Run a bounded AuK cookbook qualification through the public native CLI.

Writes commands, WAV hashes, numerical audio statistics, timing, memory and
local-ASR transcripts. These receipts are evidence, not a subjective quality score.
Requires numpy, soundfile, soxr; no inference runs in Python.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import soundfile as sf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--thinker', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--variant', choices=['base', 'flash'], required=True)
p.add_argument('--only', help='Comma-separated case ID prefixes')
a = p.parse_args()
revision = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != '6943a1e967409e8c73139a7a345f2a611cfb3dd6':
    raise SystemExit('Unexpected upstream revision')
spec = importlib.util.spec_from_file_location('cookbook', a.upstream / 'scripts/run_cookbook_mlx.py')
cookbook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cookbook)
prefixes = (a.only or '1.1,1.2,2.1,3.1,3.2,3.3,5.1,zh-tts').split(',')
cases = cookbook.CASES + [('zh-tts', 'Chinese instruct TTS', '用自然、清晰的普通话说：“欢迎回家，今天工作怎么样？”', None, 3.0)]
a.out.mkdir(parents=True, exist_ok=True)
receipts = []
for case_id, label, instruction, audio, seconds in cases:
    if not any(case_id.startswith(prefix) for prefix in prefixes):
        continue
    destination = a.out / (case_id + '.wav')
    source = a.upstream / audio if audio else None
    command = [str(a.binary), 'audio', 'edit', instruction, '--model', 'audio-auk-' + a.variant,
               '--model-path', str(a.model), '--thinker-path', str(a.thinker), '--seed', '42', '--output', str(destination)]
    if source:
        command += ['--audio', str(source)]
    if seconds is not None:
        command += ['--duration', str(seconds)]
    print('Running', a.variant, case_id, flush=True)
    started = time.monotonic()
    with (a.out / (case_id + '.json')).open('w') as stdout, (a.out / (case_id + '.log')).open('w') as stderr:
        result = subprocess.run(['/usr/bin/time', '-l', *command], stdout=stdout, stderr=stderr, timeout=1800)
    receipt = {'case': case_id, 'label': label, 'argv': command, 'exit_code': result.returncode,
               'elapsed_seconds': time.monotonic() - started, 'upstream_revision': revision}
    log = (a.out / (case_id + '.log')).read_text()
    for key in ['maximum resident set size', 'peak memory footprint']:
        match = re.search(r'(\d+)\s+' + key, log)
        if match:
            receipt[key.replace(' ', '_') + '_bytes'] = int(match[1])
    if result.returncode == 0:
        samples, rate = sf.read(destination, dtype='float32', always_2d=True)
        receipt.update(sample_rate=rate, frames=len(samples), channels=samples.shape[1],
                       rms=float(np.sqrt(np.mean(samples**2))), peak=float(np.abs(samples).max()),
                       finite=bool(np.isfinite(samples).all()),
                       clipped_fraction=float((np.abs(samples) >= 0.999).mean()),
                       sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        language = 'zh' if case_id in ['zh-tts', '5.1-enhance'] else 'en'
        asr_command = [str(a.binary), 'speech', 'transcribe', str(destination), '--backend',
                       'qwen' if language == 'zh' else 'parakeet', '--language', language, '--quiet', '--no-timestamps']
        asr = subprocess.run(asr_command, text=True, capture_output=True, timeout=600)
        (a.out / (case_id + '.asr.log')).write_text(asr.stderr)
        receipt.update(asr_argv=asr_command, asr_exit_code=asr.returncode, transcript=asr.stdout.strip())
        if source:
            source_samples, source_rate = sf.read(source, dtype='float32', always_2d=True)
            source_command = list(asr_command)
            source_command[3] = str(source)
            source_asr = subprocess.run(source_command, text=True, capture_output=True, timeout=600)
            (a.out / (case_id + '.source-asr.log')).write_text(source_asr.stderr)
            receipt['source'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'seconds': len(source_samples) / source_rate,
                                 'rms': float(np.sqrt(np.mean(source_samples**2))),
                                 'asr_exit_code': source_asr.returncode, 'transcript': source_asr.stdout.strip()}
    receipts.append(receipt)
    (a.out / 'receipts.json').write_text(json.dumps(receipts, indent=2, ensure_ascii=False))
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if result.returncode != 0:
        raise SystemExit('Native execution failed; inspect ' + str(a.out / (case_id + '.log')))
