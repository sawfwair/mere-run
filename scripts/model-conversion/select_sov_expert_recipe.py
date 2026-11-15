#!/usr/bin/env python3
"""Select per-layer precision from native routed-output calibration scores.

Use --allowed-bits 4 8 to screen a hardware-aware Q4/Q8 schedule. Thresholds
measure local normalized expert output, not recognition error. Q8 is the fallback.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--maximum-relative-rmse', type=float, required=True)
    parser.add_argument('--allowed-bits', nargs='+', type=int, choices=[4, 6, 8], default=[4, 6, 8])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text())
    layers = sorted({score['layer'] for score in scores})
    if layers != list(range(len(layers))) or 8 not in args.allowed_bits:
        parser.error('Scores must cover consecutive decoder layers, and allowed precisions must include Q8.')
    bits = [min((score['bits'] for score in scores if score['layer'] == layer and score['bits'] in args.allowed_bits
                 and score['relativeRMSE'] <= args.maximum_relative_rmse), default=8) for layer in layers]
    args.output.write_text(json.dumps(dict(schemaVersion=1, groupSize=64, layerBits=bits), indent=2) + '\n')
    print({bit: bits.count(bit) for bit in sorted(set(bits))})


if __name__ == '__main__':
    main()
