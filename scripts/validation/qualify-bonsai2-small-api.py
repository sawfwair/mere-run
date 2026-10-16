#!/usr/bin/env python3
"""Collect native resident Bonsai quality and timing snapshots over loopback HTTP."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import ast
import os
import re
import subprocess
import time
import urllib.error
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def machine():
    result = {"time": time.time(), "load_average": list(os.getloadavg())}
    for key in ["vm.swapusage", "kern.memorystatus_vm_pressure_level"]:
        process = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        result[key] = process.stdout.strip()
    rows = subprocess.check_output(["ps", "-axo", "pid,rss,comm"], text=True).splitlines()[1:]
    result["competing_processes"] = []
    for row in rows:
        fields = row.strip().split(None, 2)
        if len(fields) == 3 and Path(fields[2]).name in {"mere.run", "xctest", "swift-frontend", "clang"}:
            result["competing_processes"].append({"pid": int(fields[0]), "rss_kib": int(fields[1]),
                                                   "executable": Path(fields[2]).name})
    return result


def clean(text):
    return text.rsplit("</think>", 1)[-1].strip()


def object_score(text, expected):
    try:
        actual = json.loads(clean(text))
    except (ValueError, TypeError):
        return {"passed": False, "checks": {"strict_json": False}, "expected": expected}
    checks = {key: actual.get(key) == value for key, value in expected.items()} if isinstance(actual, dict) else {"object": False}
    return {"passed": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def grounded_score(text, expected):
    answer = clean(text)
    checks = {}
    for phrase in expected.get("requiredPhrases", []):
        checks["required:" + phrase] = phrase.lower() in answer.lower()
    for phrase in expected.get("forbiddenPhrases", []):
        checks["forbidden:" + phrase] = phrase.lower() not in answer.lower()
    for pattern in expected.get("requiredRegexes", []):
        checks["regex:" + pattern] = re.search(pattern, answer) is not None
    try:
        obj = json.loads(answer)
        if not isinstance(obj, dict):
            obj = {}
    except ValueError:
        obj = {}
    if expected.get("mustBeJSON"):
        checks["json_object"] = bool(obj)
    for key in expected.get("requiredJSONKeys", []):
        checks["key:" + key] = key in obj
    for key, value in expected.get("expectedJSONValues", {}).items():
        checks["value:" + key] = str(obj.get(key)).lower() == value.lower()
    if "minBulletCount" in expected:
        checks["bullet_count"] = len(re.findall(r"(?m)^\s*(?:[-*]|[0-9]+[.)])\s+", answer)) >= expected["minBulletCount"]
    return {"passed": all(checks.values()), "passed_checks": sum(checks.values()),
            "total_checks": len(checks), "checks": checks}


def code_score(text, function):
    source = clean(text)
    if source.startswith("```"):
        source = source.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)):
                raise ValueError("Generated code contains unsupported statements")
            if isinstance(node, ast.Attribute) and node.attr not in {"append", "lower", "get", "sort", "copy", "extend", "pop", "count", "join", "split", "strip", "upper"}:
                raise ValueError("Generated code contains unsupported attribute access")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("Generated code contains a reserved name")
        tests = {
            "count_vowels": [(("Hello",), 2), (("AEIOU",), 5), (("rhythm",), 0), (("",), 0), (("Bonsai 2",), 3)],
            "merge_intervals": [(([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]),
                                (([],), []), (([[1, 4], [4, 5]],), [[1, 5]]),
                                (([[5, 7], [1, 2]],), [[1, 2], [5, 7]]),
                                (([[1, 9], [2, 3]],), [[1, 9]])],
        }[function]
        # The child has closed builtins, no imports, a CPU limit, and a short wall timeout.
        runner = """import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
p=json.load(sys.stdin)
allowed={k:__builtins__.__dict__[k] for k in ['len','range','enumerate','sum','min','max','abs','sorted','set','tuple','list','str','dict','int','float','all','any','zip']}
ns={'__builtins__':allowed}
exec(compile(p['source'],'generated','exec'),ns)
print(json.dumps([ns[p['function']](*args)==expected for args,expected in p['tests']]))
"""
        process = subprocess.run(["python3", "-I", "-c", runner],
                                 input=json.dumps({"source": source, "function": function, "tests": tests}),
                                 capture_output=True, text=True, timeout=5)
        if process.returncode:
            raise ValueError(process.stderr[-1000:])
        checks = json.loads(process.stdout)
        return {"passed": all(checks), "passed_checks": sum(checks), "total_checks": len(checks)}
    except (ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        return {"passed": False, "passed_checks": 0, "total_checks": 5, "error": str(error)}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18753')
    parser.add_argument('--phase', choices=['quality', 'throughput'], required=True)
    parser.add_argument('--condition', choices=['sidecar-default', 'card-presence-probe'], default='sidecar-default')
    parser.add_argument('--sizes', nargs='+', choices=['2', '4', '9'], default=['2', '4', '9'])
    args = parser.parse_args()
    if args.condition != 'sidecar-default' and args.phase != 'quality':
        parser.error('The presence probe is a separate quality condition')
    root = args.evidence_root.resolve(strict=True)
    helpers = globals()
    binary_hash = helpers['digest'](args.binary)
    case_file = root / ('quality-cases.json' if args.condition == 'sidecar-default' else 'recommended-diagnostic-plan.json')
    case_hash = helpers['digest'](case_file)
    suffix = '' if args.condition == 'sidecar-default' else '-' + args.condition
    destination = root / ('api-' + args.phase + suffix)
    destination.mkdir(exist_ok=True)
    saved = destination / 'results.json'
    records = []
    if args.phase == 'quality' and saved.exists():
        prior = json.loads(saved.read_text())
        if prior['binary_sha256'] != binary_hash:
            raise RuntimeError('Saved run uses another binary; use a fresh evidence root')
        prior_case_hash = prior.get('case_manifest_sha256')
        if prior_case_hash is None and (root / 'plan.json').exists():
            plan = json.loads((root / 'plan.json').read_text())
            key = 'quality_case_manifest_sha256' if args.condition == 'sidecar-default' else 'secondary_case_manifest_sha256'
            prior_case_hash = plan.get(key)
        if prior_case_hash != case_hash:
            raise RuntimeError('Saved corpus differs; use a fresh evidence root')
        records = prior['records']

    def http(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(args.endpoint + path, data=body,
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode()
            (destination / ('http-error-' + str(time.time_ns()) + '.json')).write_text(body)
            raise RuntimeError(f'HTTP {error.code}: {body}') from error

    server_pid = int(http('/runtime/status')['process']['processID'])

    def save():
        (destination / 'results.json').write_text(json.dumps({'binary_sha256': binary_hash,
            'collector_sha256': helpers['digest'](__file__), 'condition': args.condition, 'case_manifest_sha256': case_hash, 'records': records}, indent=2) + '\n')

    def request(size, case):
        label = size + 'b-' + case['label']
        model = f'text-chat-bonsai-2-{size}b-2bit'
        messages = []
        if 'system' in case:
            messages.append({'role': 'system', 'content': case['system']})
        content = case['prompt']
        if 'image' in case:
            image = root / 'fixtures' / case['image']
            uri = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()
            content = [{'type': 'text', 'text': content}, {'type': 'image_url', 'image_url': {'url': uri}}]
        messages.append({'role': 'user', 'content': content})
        payload = {'model': model, 'messages': messages, 'max_tokens': (8192 if args.condition == 'card-presence-probe' else 4096) if args.phase == 'quality' else case['tokens'],
                   'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0, 'stream': False}
        if args.condition == 'card-presence-probe':
            payload.update(presence_penalty=1.0, repetition_penalty=1.0)
        request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if args.phase == 'quality':
            matching = [entry for entry in records if entry['label'] == label]
            if matching:
                prior = matching[-1]
                prior_hash = prior.get('request_sha256')
                if prior_hash is None:
                    # Older archived runs store the exact body beside their raw receipt.
                    response_id = prior['api_response']['id']
                    for path in destination.glob(label + '-*.receipt.json'):
                        if json.loads(path.read_text())['api_response']['id'] == response_id:
                            body_path = path.with_name(path.name.replace('.receipt.json', '.request.json'))
                            body = json.loads(body_path.read_text())
                            prior_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                            break
                if prior_hash != request_hash:
                    raise RuntimeError('Saved request differs; use a fresh evidence root for a new trial')
                print('REUSE_SAVED', label, flush=True)
                return prior
        before = helpers['machine']()
        before['competing_processes'] = [p for p in before['competing_processes'] if p['pid'] != server_pid]
        tag = label + '-' + str(time.time_ns())
        print('START', label, flush=True)
        (destination / (tag + '.request.json')).write_text(json.dumps(payload, indent=2) + '\n')
        started = time.time()
        monotonic_started = time.monotonic()
        response = http('/v1/chat/completions', payload)
        monotonic_elapsed = time.monotonic() - monotonic_started
        elapsed = time.time() - started
        status = http('/runtime/status')
        if status.get('process', {}).get('processID') != server_pid:
            raise RuntimeError('The API process changed during collection')
        entry = next(item for item in status['models'] if item['id'] == model)
        record = {'label': label, 'transport': 'native resident loopback API',
                  'response': response['choices'][0]['message'].get('content') or '',
                  'api_response': response, 'usage': response.get('usage'), 'wall_seconds': elapsed,
                  'monotonic_wall_seconds': monotonic_elapsed, 'condition': args.condition,
                  'request_sha256': request_hash,
                  'timing': entry.get('benchmarkStats', {}).get('lastRequest'), 'model_status': entry,
                  'memory': status['memory'], 'process': status.get('process'),
                  'before': before, 'after': helpers['machine'](), 'binary_sha256': binary_hash,
                  'collector_sha256': helpers['digest'](__file__)}
        (destination / (tag + '.receipt.json')).write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print('END', label, round(elapsed, 2), flush=True)
        return record

    cases = json.loads((root / 'quality-cases.json').read_text())
    if args.condition == 'card-presence-probe':
        cases = json.loads((root / 'recommended-diagnostic-plan.json').read_text())['cases']
    for size in args.sizes:
        if args.phase == 'quality':
            for case in cases:
                record = request(size, case)
                text = record['response']
                if case['type'] == 'grounded':
                    record['score'] = helpers['grounded_score'](text, case['expected'])
                elif case['type'] == 'arithmetic':
                    record['score'] = {'passed': helpers['clean'](text).strip(' .\n') == '126', 'expected': '126'}
                elif case['type'] == 'code':
                    record['score'] = helpers['code_score'](text, case['function'])
                else:
                    record['score'] = helpers['object_score'](text, case['expected'])
                save()
        else:
            for modality, prompt, image, tokens in [
                ('text', 'Write a detailed guide of at least 1000 words explaining how a community can turn an abandoned railway station into a public library. Cover accessibility, building repairs, staffing, book selection, funding, and an opening-day scene.', None, 512),
                ('vision', 'Describe this image in detail in about 250 words. State all text, object counts, colors, and spatial relationships. Explain how its layout helps a reader compare the shapes.', 'shapes.png', 256)]:
                for repetition in range(4):
                    case = {'label': modality + ('-warmup' if repetition == 0 else f'-measured-{repetition}'),
                            'prompt': prompt, 'tokens': min(128, tokens) if repetition == 0 else tokens}
                    if image:
                        case['image'] = image
                    record = request(size, case)
                    record['measurement'] = 'discarded warm-up' if repetition == 0 else 'resident request after warm-up'
                    save()
        # Release the previous model through the server's normal residency control plane.
        result = http(f'/runtime/models/text-chat-bonsai-2-{size}b-2bit/unload', {})
        (destination / (size + 'b-unload.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', args.phase, flush=True)


if __name__ == '__main__':
    main()
