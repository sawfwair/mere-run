#!/usr/bin/env python3
"""Score bounded AuK audio receipts using content and signal checks.

These checks are objective proxies, not listening or speaker-similarity scores.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--receipts', type=Path, required=True)
a = p.parse_args()
revision = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != '6943a1e967409e8c73139a7a345f2a611cfb3dd6':
    raise SystemExit('Unexpected upstream revision')
spec = importlib.util.spec_from_file_location('verify', a.upstream / 'scripts/verify_cookbook_mlx.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

def units(text, chinese=False):
    return list(re.sub(r'[^\u4e00-\u9fff]', '', text)) if chinese else re.sub(r"[^a-z' ]", '', text.lower()).split()

def error_rate(expected, actual):
    row = list(range(len(actual) + 1))
    for i, source in enumerate(expected, 1):
        next_row = [i]
        for j, target in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j-1] + (source != target)))
        row = next_row
    return row[-1] / max(len(expected), 1)

results = []
for receipt in json.loads(a.receipts.read_text()):
    case = receipt['case']
    metrics = {'case': case}
    valid = receipt['exit_code'] == 0 and receipt.get('finite', False) and receipt.get('asr_exit_code') == 0
    if not valid:
        results.append(dict(metrics, passed=False, reason='Generation or ASR failed'))
        continue
    text = receipt['transcript']
    source = receipt.get('source')
    wave = verify.load(str(a.receipts.parent / (case + '.wav')))
    if source:
        original = verify.load(source['path'])
        source_units = units(source['transcript'], case == '5.1-enhance')
        content_error = error_rate(source_units, units(text, case == '5.1-enhance')) if source_units else None
        metrics['source_content_error_rate'] = content_error
    if case in ['1.1-zeroshot-tts', '1.2-instruct-tts', 'zh-tts']:
        expected = {'1.1-zeroshot-tts': "Ladies and gentlemen it's an honor to have the opportunity to address such a distinguished audience",
                    '1.2-instruct-tts': 'Welcome home how was work today', 'zh-tts': '欢迎回家今天工作怎么样'}[case]
        value = error_rate(units(expected, case == 'zh-tts'), units(text, case == 'zh-tts'))
        metrics.update(content_error_rate=value, target=expected)
        passed = value <= 0.2
    elif case == '2.1-content-edit':
        normalized = ' '.join(units(text))
        new_present = 'living well with dreams unmet' in normalized
        old_absent = 'accepting what we cannot have' not in normalized
        metrics.update(replacement_present=new_present, old_phrase_absent=old_absent)
        passed = new_present and old_absent
    elif case == '3.1-pitch':
        before, after = verify.median_f0(original), verify.median_f0(wave)
        cents = 1200 * np.log2(after / before) if before > 0 and after > 0 else 0
        metrics.update(source_f0_hz=before, output_f0_hz=after, cents=float(cents))
        passed = 80 < cents < 330 and (content_error is None or content_error <= 0.2)
    elif case == '3.2-speed':
        ratio = len(original) / len(wave)
        metrics.update(duration_ratio=ratio, source_syllable_proxy=verify.speech_rate(original), output_syllable_proxy=verify.speech_rate(wave))
        passed = 1.35 < ratio < 1.7 and content_error is not None and content_error <= 0.2
    elif case == '3.3-volume':
        delta = float(verify.db(wave) - verify.db(original))
        metrics['gain_db'] = delta
        passed = 7 <= delta <= 13 and (content_error is None or content_error <= 0.2)
    elif case == '5.1-enhance':
        def floor(x):
            frames = [np.sqrt(np.mean(x[i:i+600]**2)) for i in range(0, len(x)-600, 240)]
            return float(np.percentile(frames, 10))
        before, after = floor(original), floor(wave)
        delta = float(20 * np.log10((after + 1e-12) / (before + 1e-12)))
        metrics['quiet_frame_rms_change_db'] = delta
        passed = delta < -3 and content_error is not None and content_error <= 0.3
    else:
        passed = False
        metrics['reason'] = 'No predefined quality criterion for this case'
    metrics['passed'] = bool(passed)
    results.append(metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
(a.receipts.parent / 'assessments.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
