#!/usr/bin/env python3
"""Measure native YuE2 stage reuse and checkpoint execution using local assets.

This script never downloads models or accepts licenses. Supply complete local
YuE2 and SheetSage2 model directories. Output directories must be new. Results
measure this fixed corpus and debug/release binary, not general listening quality.
"""
import argparse
import array
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def latent_values(path):
    with path.open('rb') as stream:
        if stream.read(8) != b'\x93NUMPY\x01\x00':
            raise ValueError('Expected a version-1 NumPy latent artifact')
        length = struct.unpack('<H', stream.read(2))[0]
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        if header['descr'] != '<f4' or header['fortran_order']:
            raise ValueError('Expected little-endian C-order float32 latent data')
        values = array.array('f', stream.read())
        if sys.byteorder != 'little':
            values.byteswap()
        if len(values) != math.prod(header['shape']):
            raise ValueError('Latent array size differs from its header')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Latent array contains non-finite values')
        return header['shape'], values


def difference(left, right):
    left_shape, left_values = latent_values(left)
    right_shape, right_values = latent_values(right)
    if left_shape != right_shape:
        raise ValueError(f'Latent shapes differ: {left_shape} versus {right_shape}')
    return {'shape': left_shape, 'byte_identical': digest(left) == digest(right),
            'maximum_absolute_error': max(abs(a - b) for a, b in zip(left_values, right_values))}


def audio_metrics(path):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
    stream = next(value for value in probe['streams'] if value['codec_type'] == 'audio')
    samples = array.array('f', subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-']))
    if sys.byteorder != 'little':
        samples.byteswap()
    finite = all(math.isfinite(value) for value in samples)
    peak = (max(abs(value) for value in samples) if samples else 0) if finite else None
    rms = math.sqrt(math.fsum(value * value for value in samples) / max(1, len(samples))) if finite else None
    return {'channels': stream['channels'], 'sample_rate': int(stream['sample_rate']),
            'samples_per_channel': len(samples) // stream['channels'], 'finite': finite,
            'peak': peak, 'rms': rms, 'sha256': digest(path),
            'decoded_float32_sha256': hashlib.sha256(samples.tobytes()).hexdigest()}


class Qualification:
    def __init__(self, binary, output):
        self.binary = binary.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.results = {'binary_sha256': digest(self.binary), 'cases': [], 'comparisons': {}}

    def run(self, name, arguments):
        log = self.output / (name + '.stderr.log')
        stdout = self.output / (name + '.stdout')
        print('Running ' + name, flush=True)
        started = time.monotonic()
        with log.open('w') as diagnostic, stdout.open('w') as result:
            process = subprocess.Popen(['/usr/bin/time', '-l', str(self.binary), *map(str, arguments)],
                                       stdout=result, stderr=diagnostic, start_new_session=True)
            try:
                exit_code = process.wait(timeout=1800)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        elapsed = time.monotonic() - started
        text = log.read_text()
        measurements = {}
        for label in ['maximum resident set size', 'peak memory footprint']:
            match = re.search(r'(\d+)\s+' + label, text)
            if match:
                measurements[label.replace(' ', '_') + '_bytes'] = int(match[1])
        record = {'name': name, 'arguments': list(map(str, arguments)), 'exit_code': exit_code,
                  'wall_seconds': elapsed, **measurements}
        self.results['cases'].append(record)
        self.save()
        if exit_code:
            raise RuntimeError(f'{name} failed; see {log}')
        return record

    def save(self):
        (self.output / 'measurements.json').write_text(json.dumps(self.results, indent=2) + '\n')


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--request', type=Path, required=True, help='Fixed short native JSON request with an external score.')
parser.add_argument('--long-request', type=Path, help='Optional fixed long native JSON request.')
parser.add_argument('--sheetsage', type=Path)
parser.add_argument('--recording', type=Path)
parser.add_argument('--legacy-vae', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
for model, files in [(args.model, ['model.safetensors', 'qwen.tiktoken', 'vae/model.safetensors']),
                     (args.sheetsage, ['model.safetensors', 'mert2/model.safetensors'])]:
    if model is not None:
        for name in files:
            if not (model / name).is_file():
                parser.error(f'Missing local checkpoint: {model / name}')
suite = Qualification(args.binary, args.output)
suite.results['request'] = json.loads(args.request.read_text())
model = args.model.resolve()
out = suite.output
try:
    suite.run('preflight', ['music', 'yue2', 'validate', args.request.resolve(), '--tokenizer', model / 'qwen.tiktoken'])
    suite.run('plan', ['music', 'yue2', 'plan', '--model', model, '--request', args.request.resolve(), '--output', out / 'score'])
    suite.run('semantic', ['music', 'yue2', 'semantic', out / 'score', '--model', model, '--output', out / 'semantic'])
    suite.run('acoustic', ['music', 'yue2', 'synthesize', out / 'semantic', '--model', model, '--output', out / 'acoustic'])
    suite.run('decode', ['music', 'yue2', 'decode', out / 'acoustic', '--vae', model / 'vae', '--output', out / 'staged.wav'])
    suite.run('decode-repeat', ['music', 'yue2', 'decode', out / 'acoustic', '--vae', model / 'vae', '--output', out / 'replayed.wav'])
    suite.results['comparisons']['decode_replay_byte_identical'] = digest(out / 'staged.wav') == digest(out / 'replayed.wav')
    suite.run('acoustic-low-memory', ['music', 'yue2', 'synthesize', out / 'semantic', '--model', model,
                                     '--memory-mode', 'low-memory', '--output', out / 'acoustic-low-memory'])
    comparison = difference(out / 'acoustic/latent.npy', out / 'acoustic-low-memory/latent.npy')
    suite.results['comparisons']['low_memory_latents'] = comparison
    suite.results['audio'] = audio_metrics(out / 'staged.wav')
    request = suite.results['request']
    sampling = request['semanticSampling']
    source = out / 'direct-source.abc'
    source.write_text(request['abc'])
    arguments = ['music', 'generate', request['style'], '--model', model, '--abc-file', source,
                 '--score-mode', request.get('planning', 'full'), '--lyrics', request.get('lyrics', ''),
                 '--seed', request.get('seed', 831001), '--steps', request.get('steps', 32),
                 '--guidance-scale', request.get('guidanceScale', 1), '--min-frames', sampling['minimumTokens'],
                 '--max-frames', sampling['maximumTokens'], '--semantic-temperature', sampling['temperature'],
                 '--semantic-top-p', sampling['topP'], '--semantic-top-k', sampling['topK'],
                 '--semantic-repetition-penalty', sampling['repetitionPenalty'], '--export-format', 'float32',
                 '--normalize', 'none', '--fade-in-ms', 0, '--fade-out-ms', 0, '--output', out / 'direct.wav']
    if request.get('instrumental'):
        arguments.append('--instrumental')
    suite.run('direct-generation', arguments)
    suite.results['direct_audio'] = audio_metrics(out / 'direct.wav')
    suite.results['comparisons']['saved_stages_match_direct_pcm'] = (
        suite.results['audio']['decoded_float32_sha256'] == suite.results['direct_audio']['decoded_float32_sha256'])
    if args.legacy_vae:
        suite.run('legacy-decode', ['music', 'yue2', 'decode', out / 'acoustic', '--vae', args.legacy_vae.resolve(),
                                   '--output', out / 'legacy.wav'])
        suite.results['legacy_audio'] = audio_metrics(out / 'legacy.wav')
    if args.sheetsage and args.recording:
        suite.run('transcription', ['music', 'yue2', 'transcribe', args.recording.resolve(), '--model', args.sheetsage.resolve(),
                                   '--output', out / 'transcription', '--allow-notation-failure'])
        transcription = json.loads((out / 'transcription/transcription.json').read_text())
        suite.results['transcription'] = {key: transcription.get(key) for key in
                                          ['duration', 'windows', 'truncatedWindows', 'notationError', 'notationDiagnostics']}
        suite.results['transcription']['events'] = len(transcription['events'])
    if args.long_request:
        request = json.loads(args.long_request.read_text())
        caption = request['style']
        abc = out / 'long-source.abc'
        abc.write_text(request['abc'])
        frames = request['semanticSampling']['maximumTokens']
        suite.run('long-generation', ['music', 'generate', caption, '--model', model, '--abc-file', abc,
                                     '--instrumental', '--min-frames', frames, '--max-frames', frames,
                                     '--seed', request['seed'], '--steps', request['steps'], '--stages-output', out / 'long-stages',
                                     '--export-format', 'float32', '--normalize', 'none', '--fade-in-ms', 0,
                                     '--fade-out-ms', 0, '--output', out / 'long.wav'])
        suite.results['long_audio'] = audio_metrics(out / 'long.wav')
    checks = {
        'decode_replay': suite.results['comparisons']['decode_replay_byte_identical'],
        'phase_loading_parity': comparison['maximum_absolute_error'] <= 2e-5,
        'saved_stages_match_direct_pcm': suite.results['comparisons']['saved_stages_match_direct_pcm'],
        'finite_non_silent_audio': suite.results['audio']['finite'] and suite.results['audio']['rms'] > 1e-5,
        'stereo_48khz': suite.results['audio']['channels'] == 2 and suite.results['audio']['sample_rate'] == 48000,
    }
    if 'legacy_audio' in suite.results:
        checks['legacy_finite_non_silent'] = suite.results['legacy_audio']['finite'] and suite.results['legacy_audio']['rms'] > 1e-5
    if 'transcription' in suite.results:
        checks['transcription_events'] = suite.results['transcription']['events'] > 0
        checks['transcription_not_truncated'] = not suite.results['transcription']['truncatedWindows']
        checks['notation_export'] = suite.results['transcription']['notationError'] is None
    if 'long_audio' in suite.results:
        checks['long_finite_non_silent'] = suite.results['long_audio']['finite'] and suite.results['long_audio']['rms'] > 1e-5
        checks['long_sample_count'] = suite.results['long_audio']['samples_per_channel'] == frames * 1920 - 64
    suite.results['checks'] = checks
    suite.results['passed'] = all(checks.values())
except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
    suite.results['passed'] = False
    suite.results['error'] = str(error)
finally:
    suite.save()
print(out / 'measurements.json')
raise SystemExit(0 if suite.results['passed'] else 1)
