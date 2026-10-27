#!/usr/bin/env python3
"""Compare native Metal logits with hash-verified CUDA source and quant references."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from lfm25_quality18 import sha, save


def load_logits(path):
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 1024 * 1024:
            raise ValueError('Oversized safetensors header')
        tensor = json.loads(f.read(length))['logits']
        assert tensor['dtype'] == 'F32'
        start, end = tensor['data_offsets']
        f.seek(8 + length + start)
        result = np.fromfile(f, dtype='<f4', count=(end - start) // 4).reshape(tensor['shape'])
    assert np.isfinite(result).all()
    return result


def metrics(reference, candidate, target):
    def lsm(x):
        x = x.astype(np.float64)
        x -= x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    lp, lq = lsm(reference), lsm(candidate)
    delta = lq[np.arange(len(target)), target] - lp[np.arange(len(target)), target]
    return dict(mean_kl=float((np.exp(lp) * (lp - lq)).sum(-1).mean()),
                top1_agreement=float((reference.argmax(-1) == candidate.argmax(-1)).mean()),
                delta_nll=float(-delta.mean()), perplexity_ratio=float(np.exp(-delta.mean())),
                maximum_absolute_logit_difference=float(np.abs(reference - candidate).max()))


def aggregate(rows, gates):
    n = sum(r['tokens'] for r in rows)
    out = dict(cases=len(rows), tokens=n)
    for key in ['mean_kl', 'top1_agreement', 'delta_nll']:
        out[key] = sum(r['tokens'] * r[key] for r in rows) / n
    out['perplexity_ratio'] = float(np.exp(out['delta_nll']))
    out['worst_case_mean_kl'] = max(r['mean_kl'] for r in rows)
    out['worst_case_perplexity_ratio'] = max(r['perplexity_ratio'] for r in rows)
    out['maximum_absolute_logit_difference'] = max(r['maximum_absolute_logit_difference'] for r in rows)
    out['passes_absolute_gates'] = (
        out['mean_kl'] <= gates['mean_kl_max']
        and out['top1_agreement'] >= gates['top1_agreement_min']
        and out['perplexity_ratio'] <= gates['perplexity_ratio_max']
        and out['worst_case_mean_kl'] <= gates['case_mean_kl_max']
        and out['worst_case_perplexity_ratio'] <= gates['case_perplexity_ratio_max'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['suite', 'native', 'model', 'output', 'run-receipt']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--selected-reference', default='selected')
    p.add_argument('--reference', action='append', required=True,
                   help='NAME=DIRECTORY; repeat a name for calibration and heldout directories.')
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    run = json.loads(a.run_receipt.read_text())
    expected_inputs = dict(frozen_suite=sha(a.suite), config=sha(a.model/'config.json'), conversion=sha(a.model/'MERERUN_CONVERSION.json'), weight_index=sha(a.model/'model.safetensors.index.json'))
    assert run['exit_code'] == 0 and run['job'] == 'logits'
    assert run['runtime_environment'] == dict(MERERUN_TEST_MLX_DEVICE='gpu', MLX_ENABLE_TF32='0')
    assert run['input_hashes_before'] == run['input_hashes_after'] == expected_inputs
    cases = suite['cases']
    expected = {r['id'] for r in cases}
    assert len(expected) == len(cases)
    native_receipt = json.loads((a.native / 'qualification-receipt.json').read_text())
    assert {r['id'] for r in native_receipt['cases']} == expected
    assert Path(native_receipt['modelPath']).resolve() == a.model.resolve()
    references = {}
    reference_configs = {}
    reference_conversions = {}
    for value in a.reference:
        name, folder = value.split('=', 1)
        root = Path(folder)
        receipt = json.loads((root / 'receipt.json').read_text())
        assert receipt['frozen_suite_sha256'] == sha(a.suite)
        assert receipt['compute_dtype'] == 'float32'
        reference_configs.setdefault(name, set()).add(receipt['config_sha256'])
        if 'conversion_sha256' in receipt: reference_conversions.setdefault(name, set()).add(receipt['conversion_sha256'])
        mapping = references.setdefault(name, {})
        for r in receipt['cases']:
            assert r['id'] not in mapping
            mapping[r['id']] = (root, r)
    assert a.selected_reference in references
    assert reference_configs[a.selected_reference] == {sha(a.model / 'config.json')}
    assert reference_conversions[a.selected_reference] == {sha(a.model / 'MERERUN_CONVERSION.json')}
    for mapping in references.values():
        assert set(mapping) == expected
    reports = {name: [] for name in references}
    native_hashes = []
    for case in cases:
        file = a.native / (case['id'] + '.safetensors')
        candidate = load_logits(file)
        target = np.array(case['tokens'][case['score_start']:])
        assert candidate.shape == (len(target), 128000)
        native_hashes.append(dict(id=case['id'], sha256=sha(file)))
        for name, mapping in references.items():
            root, row = mapping[case['id']]
            ref_file = root / row['logits_file']
            assert sha(ref_file) == row['sha256']
            reference = load_logits(ref_file)
            assert reference.shape == candidate.shape
            reports[name].append(dict(id=case['id'], split=case['split'], tokens=len(target),
                                      **metrics(reference, candidate, target)))
        print(case['id'], flush=True)
    result = dict(schema_version=1, backend='native Swift MLX/Metal',
                  frozen_suite_sha256=sha(a.suite),
                  conversion_sha256=sha(a.model / 'MERERUN_CONVERSION.json'),
                  native_receipt_sha256=sha(a.native / 'qualification-receipt.json'),
                  run_receipt_sha256=sha(a.run_receipt),
                  scope='teacher-forced full-vocabulary parity and source distortion; separate generation/context evidence required',
                  native_files=native_hashes, reference_conversion_hashes={k: sorted(v) for k, v in reference_conversions.items()}, reference_config_hashes={k: sorted(v) for k, v in reference_configs.items()}, comparisons={})
    for name, rows in reports.items():
        subsets = {'all': rows}
        for split in ['calibration', 'heldout']:
            subset = [r for r in rows if r['split'] == split]
            if subset: subsets[split] = subset
        result['comparisons'][name] = dict(aggregates={k: aggregate(v, suite['gates']) for k, v in subsets.items()}, cases=rows)
    parity = result['comparisons'][a.selected_reference]['aggregates']['all']
    result['native_backend_parity'] = dict(reference=a.selected_reference,
        gates=dict(mean_kl_max=1e-6, case_mean_kl_max=1e-5, top1_agreement_min=.9999),
        passes=parity['mean_kl']<=1e-6 and parity['worst_case_mean_kl']<=1e-5 and parity['top1_agreement']>=.9999)
    save(a.output, result)
    print(json.dumps({k: v['aggregates'] for k, v in result['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
