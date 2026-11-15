#!/usr/bin/env python3
"""Prepare disjoint FLEURS train/test audio for SOV expert calibration.

Requires duckdb and ffmpeg. FLEURS is CC BY 4.0:
https://huggingface.co/datasets/google/fleurs
Only selected audio rows are persisted; checkpoints are not copied.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

import duckdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-language', type=int, default=2)
    parser.add_argument('--test-per-language', type=int, default=10)
    parser.add_argument('--test-offset', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extension_dir = str(args.output.resolve() / 'duckdb-extensions')
    connection = duckdb.connect(config={'extension_directory': extension_dir})
    connection.execute('INSTALL httpfs')
    connection.close()
    languages = ['en_us', 'fr_fr', 'de_de', 'es_419', 'it_it', 'pt_br', 'nl_nl']

    def subset(job):
        language, split, limit = job
        offset = args.test_offset if split == 'test' else 0
        url = f'https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/{language}/{split}/0000.parquet'
        # Resolve once so each Parquet range request goes straight to the CDN.
        # Repeating Hub redirects for every page can exhaust its resolver quota.
        with urlopen(Request(url, method='HEAD')) as response:
            parquet_url = response.geturl()
        connection = duckdb.connect(config={'extension_directory': extension_dir, 'threads': '2', 'allow_asterisks_in_http_paths': 'true'})
        connection.execute('LOAD httpfs')
        rows = connection.execute('SELECT id, audio, raw_transcription FROM read_parquet(?) LIMIT ? OFFSET ?', [parquet_url, limit, offset]).fetchall()
        connection.close()
        result = []
        for ordinal, (identifier, audio, reference) in enumerate(rows):
            stem = args.output / f'{language}-{split}-{ordinal + offset:03}'
            raw, wav = stem.with_suffix('.source'), stem.with_suffix('.wav')
            raw.write_bytes(audio['bytes'])
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(wav)], check=True)
            raw.unlink()
            result.append(dict(id=f'{language}-{split}-{identifier}-{ordinal + offset}', language=language, split=split,
                               audio=str(wav.resolve()), reference=reference, sourceURL=url,
                               sourceAudioSHA256=hashlib.sha256(audio['bytes']).hexdigest(),
                               audioSHA256=hashlib.sha256(wav.read_bytes()).hexdigest()))
        print(f'{language} {split}: {len(result)} clips', flush=True)
        return split, result

    manifests = {'train': [], 'test': []}
    jobs = [(language, split, limit) for language in languages
            for split, limit in [('train', args.train_per_language), ('test', args.test_per_language)] if limit > 0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for split, rows in executor.map(subset, jobs):
            manifests[split].extend(rows)
    for split, rows in manifests.items():
        (args.output / f'{split}.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
