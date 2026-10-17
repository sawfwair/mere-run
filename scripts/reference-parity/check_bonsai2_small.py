#!/usr/bin/env python3
"""Compare frozen Bonsai reference/native exports on CPU; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metrics(left, right):
    if left.shape != right.shape:
        return {'reference_shape': list(left.shape), 'native_shape': list(right.shape), 'shape_equal': False}
    left, right = left.astype(np.float64), right.astype(np.float64)
    difference = right - left
    return {'shape_equal': True, 'max_absolute_delta': float(np.max(np.abs(difference))),
            'rmse': float(np.sqrt(np.mean(difference ** 2))),
            'relative_l2': float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(left.ravel()), 1e-30)),
            'finite': bool(np.isfinite(left).all() and np.isfinite(right).all())}


def softmax(values):
    values = values.astype(np.float64)
    values -= np.max(values, axis=-1, keepdims=True)
    values = np.exp(values)
    return values / np.sum(values, axis=-1, keepdims=True)


def compare(root, size, native_prefix):
    reference_dir, native_dir = root / ('reference-' + size), root / (native_prefix + '-' + size)
    reference = json.loads((reference_dir / 'results.json').read_text())
    native = json.loads((native_dir / 'results.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    limits = protocol['gates']
    results = []
    for expected in reference['cases']:
        actual = next(case for case in native if case['id'] == expected['id'])
        reference_path = reference_dir / expected['arrays']
        if digest(reference_path) != expected['arrays_sha256']:
            raise ValueError('Reference tensor hash changed: ' + str(reference_path))
        native_path = native_dir / (expected['id'] + '.safetensors')
        left, right = load_file(reference_path), load_file(native_path)
        gates = {'prompt_ids': expected['prompt_ids'] == actual['prompt_ids'],
                 'image_grid': expected['grid'] == actual['grid'],
                 'continuation': expected['continuation'] == actual['continuation'],
                 'rope_delta': expected['rope_delta'] == actual['rope_delta']}
        tensors = {}
        for key in ['patches', 'image_embeddings', 'input_embeddings', 'positions', 'logits']:
            if key not in left and key not in right:
                continue
            if key not in left or key not in right:
                tensors[key] = {'missing': 'reference' if key not in left else 'native'}
                continue
            tensors[key] = metrics(left[key], right[key])
        if expected['grid']:
            gates['mrope_positions'] = ('positions' in left and 'positions' in right
                                         and np.array_equal(left['positions'], right['positions']))
            patch = tensors['patches']
            gates['normalized_patches'] = bool(patch.get('finite') and patch['max_absolute_delta'] <= limits['normalized_patches_max_absolute_delta'])
            image = tensors['image_embeddings']
            gates['image_embeddings'] = bool(image.get('finite') and image['relative_l2'] <= limits['image_embedding_relative_l2'])
        if left['logits'].shape == right['logits'].shape:
            expected_top, actual_top = np.argmax(left['logits'], axis=-1), np.argmax(right['logits'], axis=-1)
            probability_delta = np.max(np.abs(softmax(left['logits']) - softmax(right['logits'])), axis=-1)
            mismatch = np.flatnonzero(expected_top != actual_top)
            tensors['logits'].update(max_probability_delta=float(np.max(probability_delta)),
                top1_agreement_count=int(np.sum(expected_top == actual_top)), positions=int(len(expected_top)),
                first_top1_mismatch=int(mismatch[0]) if len(mismatch) else None,
                per_position_max_probability_delta=probability_delta.tolist())
            gates['probabilities'] = bool(tensors['logits']['finite'] and np.max(probability_delta) <= limits['maximum_probability_delta'])
            gates['teacher_forced_greedy_argmax'] = len(mismatch) == 0
            gates['exported_argmax_consistent'] = actual_top.tolist() == actual['native_argmax'] and expected_top.tolist() == expected['continuation']
        else:
            gates['probabilities'] = gates['teacher_forced_greedy_argmax'] = False
        results.append({'id': expected['id'], 'passed': all(gates.values()), 'gates': gates, 'tensors': tensors,
                        'reference_sha256': digest(reference_path), 'native_sha256': digest(native_path)})
    return {'size': size, 'passed': all(case['passed'] for case in results), 'protocol': protocol,
            'reference_versions': reference['versions'], 'native_prefix': native_prefix, 'cases': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', choices=['2', '4', '9'], required=True)
    parser.add_argument('--native-prefix', default='native')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.root, args.size, args.native_prefix)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'size': args.size, 'passed': report['passed'],
                      'cases': [{'id': c['id'], 'failed_gates': [k for k, v in c['gates'].items() if not v]} for c in report['cases']]}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
