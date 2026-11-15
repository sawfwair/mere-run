#!/usr/bin/env python3
"""Compare fresh-process SOV packing/reuse on one storage device."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('.build/release/mere.run'))
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--packed-weights', type=Path, required=True)
    parser.add_argument('--draft-model', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--packed-first', action='store_true', help='Reverse the initial pair order.')
    parser.add_argument('--convert', action='store_true', help='Require an absent pack and measure first conversion.')
    parser.add_argument('--prewarm', action='store_true', help='Read each mode’s weight files before timing; no system-wide cache flush.')
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    if args.convert and args.packed_weights.exists():
        parser.error('--convert requires an absent packed directory')
    if not args.convert and not args.packed_weights.exists():
        parser.error('Create the packed directory first or use --convert')
    args.output.mkdir(parents=True, exist_ok=True)
    base = [str(args.binary.resolve()), 'model', 'benchmark', 'sov', str(args.audio),
            '--model-root', str(args.model_root), '--draft-model', str(args.draft_model),
            '--experimental-decoder', 'q8-metal', '--experimental-attention',
            '--generation-experiment', 'rollback', '--warmups', '0', '--repetitions', '1', '--json']
    index = json.loads((args.model_root / 'model.safetensors.index.json').read_text())['weight_map']
    source_files = [args.model_root / name for name in sorted(set(index.values()))]
    encoder_files = [args.model_root / name for name in sorted({v for k, v in index.items() if k.startswith(('model.audio_tower.', 'model.embed_audio.'))})]
    draft_files = sorted(args.draft_model.glob('*.safetensors'))
    metadata = dict(binarySHA256=digest(args.binary), audioSHA256=digest(args.audio),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    protocol='Fresh CLI process per case; alternating AB/BA order; no co-resident comparison model; no cache flush.',
                    cachePolicy='Explicit weight-file pre-read before each reuse/baseline case.' if args.prewarm else 'OS file cache uncontrolled; these are not certified cold-disk timings.')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    reports = []
    libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int


    def run(label, packed, conversion=False):
        prewarm_seconds = None
        prewarm_bytes = None
        if args.prewarm and not conversion:
            files = (sorted(args.packed_weights.glob('*.safetensors')) + encoder_files if packed else source_files) + draft_files
            prewarm_started = time.monotonic()
            for file in files:
                with file.open('rb') as handle:
                    for _ in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                        pass
            prewarm_seconds = time.monotonic() - prewarm_started
            prewarm_bytes = sum(f.stat().st_size for f in files)
            print(f'{label}: pre-read {sum(f.stat().st_size for f in files) / 2**30:.2f} GiB in {time.monotonic()-prewarm_started:.2f}s', flush=True)
        command = base + (['--packed-weights', str(args.packed_weights)] if packed else [])
        before = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
        start = time.monotonic()
        print('running', label, flush=True)
        with (args.output / f'{label}.json').open('w') as stdout, (args.output / f'{label}.log').open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            usage = dict(pageins=0, diskIOBytesRead=0, diskIOBytesWritten=0)
            # RUSAGE_INFO_V4 begins with a 16-byte UUID, followed by uint64 fields.
            # Allocate more than the struct size; offsets follow sys/resource.h.
            storage = (ctypes.c_uint64 * 512)()
            try:
                while process.poll() is None:
                    if libproc.proc_pid_rusage(process.pid, 4, ctypes.byref(storage)) == 0:
                        for key, index in [('pageins', 6), ('diskIOBytesRead', 18), ('diskIOBytesWritten', 19)]:
                            usage[key] = max(usage[key], storage[index])
                    time.sleep(0.05)
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
        elapsed = time.monotonic() - start
        report = json.loads((args.output / f'{label}.json').read_text())
        assert report['decoderExperts'] == 'q8-metal'
        if packed:
            assert report['packedWeights']['cacheHit'] != conversion
        sample = dict(label=label, processWallSeconds=elapsed, preReadSeconds=prewarm_seconds, preReadBytes=prewarm_bytes, processIO=usage, modelLoadSeconds=report['modelLoadSeconds'],
                      loadAndFirstTranscriptionSeconds=report['audioReadSeconds'] + report['modelLoadSeconds'] + report['firstInference']['totalSeconds'],
                      loadMLXPeakBytes=report['modelLoadPeakMemoryBytes'], loadProcessMemory=report['modelLoadProcessMemory'],
                      packedWeights=report.get('packedWeights'), tokens=report['firstInference']['tokens'],
                      vmStatBefore=before, vmStatAfter=subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout)
        reports.append(sample)
        (args.output / 'samples.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(json.dumps({k: v for k, v in sample.items() if k not in ['tokens', 'vmStatBefore', 'vmStatAfter']}), flush=True)

    if args.convert:
        run('conversion', True, conversion=True)
    for pair in range(args.pairs):
        for packed in ([False, True] if (pair % 2 == 0) != args.packed_first else [True, False]):
            run(f'pair-{pair + 1}-' + ('packed' if packed else 'legacy'), packed)
    expected = reports[0]['tokens']
    assert all(r['tokens'] == expected for r in reports), 'Transcript token parity failed'
    modes = {}
    for mode in ['legacy', 'packed']:
        group = [r for r in reports if r['label'].endswith('-' + mode)]
        modes[mode] = dict(runs=len(group), modelLoadSeconds=statistics.median(r['modelLoadSeconds'] for r in group),
                           loadAndFirstTranscriptionSeconds=statistics.median(r['loadAndFirstTranscriptionSeconds'] for r in group),
                           loadMLXPeakGiB=statistics.median(r['loadMLXPeakBytes']/2**30 for r in group),
                           loadPhysicalPeakGiB=statistics.median(r['loadProcessMemory']['peakPhysicalFootprintBytes']/2**30 for r in group))
    summary = dict(tokenParity=True, tokens=len(expected), modes=modes,
                   loadSpeedup=modes['legacy']['modelLoadSeconds']/modes['packed']['modelLoadSeconds'],
                   loadPhysicalPeakReductionPercent=100*(1-modes['packed']['loadPhysicalPeakGiB']/modes['legacy']['loadPhysicalPeakGiB']),
                   pairedLoadReductionPercent=[100*(1-next(r['modelLoadSeconds'] for r in reports if r['label']==f'pair-{p}-packed')/next(r['modelLoadSeconds'] for r in reports if r['label']==f'pair-{p}-legacy')) for p in range(1,args.pairs+1)])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
