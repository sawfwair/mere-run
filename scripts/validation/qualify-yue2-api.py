#!/usr/bin/env python3
"""Qualify the local YuE2 queue with separately supplied checkpoint assets.

Measures cold/warm jobs, resident replay, bounded admission, queued and running
cancellation, and reuse of completed stages. Binds only 127.0.0.1. No downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--request', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
base = f'http://127.0.0.1:{port}'
record = {'jobs': [], 'checks': {}}
plan = json.loads(args.request.read_text())
jobs = args.output / 'jobs'


def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    message = urllib.request.Request(base + path, data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(message, timeout=30) as response:
        return json.load(response)


def wait(job, terminal=True, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = request('/v1/jobs/' + job['id'])
        ready = current['status'] in ['succeeded', 'truncated', 'failed', 'cancelled'] if terminal else current['status'] == 'running'
        if ready:
            return current
        time.sleep(0.2)
    raise TimeoutError('Job did not reach its expected state')


def completed(job):
    result = wait(job)
    if result['status'] not in ['succeeded', 'truncated']:
        raise RuntimeError(f'Job did not generate audio: {result}')
    return result


def latent_hash(job):
    return hashlib.sha256((jobs / job['id'] / 'stages/acoustic/latent.npy').read_bytes()).hexdigest()


diagnostic = (args.output / 'server.stderr.log').open('w')
output = (args.output / 'server.stdout').open('w')
server = subprocess.Popen(['/usr/bin/time', '-l', str(args.binary.resolve()), 'music', 'yue2', 'serve',
                           '--model', str(args.model.resolve()), '--output', str(jobs), '--port', str(port),
                           '--max-pending', '3', '--memory-mode', 'resident'],
                          stdout=output, stderr=diagnostic, start_new_session=True)
try:
    deadline = time.monotonic() + 60
    while True:
        try:
            record['health'] = request('/health')
            break
        except urllib.error.URLError:
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Server did not become healthy')
            time.sleep(0.2)
    started = time.monotonic()
    cold = request('/v1/jobs', plan)
    completed(cold)
    record['jobs'].append({'name': 'cold', 'id': cold['id'], 'wall_seconds': time.monotonic() - started})
    started = time.monotonic()
    warm = request('/v1/jobs', plan)
    queued = [request('/v1/jobs', plan), request('/v1/jobs', plan)]
    try:
        request('/v1/jobs', plan)
        record['checks']['bounded_queue'] = False
    except urllib.error.HTTPError as error:
        record['checks']['bounded_queue'] = error.code == 429
    for job in queued:
        request('/v1/jobs/' + job['id'] + '/cancel', {})
    completed(warm)
    record['jobs'].append({'name': 'warm', 'id': warm['id'], 'wall_seconds': time.monotonic() - started})
    record['checks']['resident_latents_identical'] = latent_hash(cold) == latent_hash(warm)
    record['checks']['queued_cancellation'] = all(request('/v1/jobs/' + job['id'])['status'] == 'cancelled'
                                                 and not (jobs / job['id']).exists() for job in queued)
    for name in ['audio', 'score', 'latents']:
        with urllib.request.urlopen(base + '/v1/jobs/' + warm['id'] + '/' + name, timeout=30) as response:
            record['checks']['artifact_' + name] = bool(response.read())
    long_plan = json.loads(json.dumps(plan))
    long_plan['semanticSampling']['minimumTokens'] = 2000
    long_plan['semanticSampling']['maximumTokens'] = 2000
    running = request('/v1/jobs', long_plan)
    following = request('/v1/jobs', plan)
    wait(running, terminal=False)
    deadline = time.monotonic() + 60
    score = jobs / running['id'] / 'stages/score/stage.json'
    while not score.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Cancelled job did not finish its score')
        time.sleep(0.2)
    # Allow request checksumming to finish and enter AR decoding.
    time.sleep(12)
    started = time.monotonic()
    cancelled = request('/v1/jobs/' + running['id'] + '/cancel', {})
    record['cancel_request_seconds'] = time.monotonic() - started
    wait(following, terminal=False)
    record['cancel_to_next_running_seconds'] = time.monotonic() - started
    record['checks']['cancel_releases_queue_within_ten_seconds'] = record['cancel_to_next_running_seconds'] < 10
    completed(following)
    record['checks']['running_cancellation'] = cancelled['status'] == 'cancelled'
    record['checks']['queue_recovers_after_cancel'] = latent_hash(following) == latent_hash(warm)
    retained = jobs / running['id'] / 'stages/score'
    result = subprocess.run([str(args.binary.resolve()), 'music', 'yue2', 'inspect', str(retained)],
                            capture_output=True, timeout=30)
    record['checks']['cancelled_score_inspectable'] = result.returncode == 0
    record['passed'] = all(record['checks'].values())
except (OSError, ValueError, TimeoutError, RuntimeError, subprocess.SubprocessError) as error:
    record['passed'] = False
    record['error'] = str(error)
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    diagnostic.close()
    output.close()
    (args.output / 'measurements.json').write_text(json.dumps(record, indent=2) + '\n')
print(args.output / 'measurements.json')
raise SystemExit(0 if record['passed'] else 1)
