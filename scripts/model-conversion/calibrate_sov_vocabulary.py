#!/usr/bin/env python3
"""Select draft token rows from FLEURS training text in SOV's seven languages.

Requires duckdb and tokenizers. FLEURS is CC BY 4.0:
https://huggingface.co/datasets/google/fleurs
Only the transcript column is selected; benchmark transcripts are not inputs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import duckdb
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows-per-language', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extensions = args.output.resolve() / 'duckdb-extensions'
    setup = duckdb.connect(config={'extension_directory': str(extensions)})
    setup.execute('INSTALL httpfs'); setup.close()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    def language(config):
        url = 'https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/' + config + '/train/0000.parquet'
        connection = duckdb.connect(config={'extension_directory': str(extensions), 'threads': '2'})
        connection.execute('LOAD httpfs')
        rows = connection.execute('SELECT raw_transcription FROM read_parquet(?) LIMIT ?',
            [url, args.rows_per_language]).fetchall()
        connection.close()
        texts = [row[0] for row in rows]
        data = json.dumps(texts, ensure_ascii=False).encode()
        (args.output / (config + '.json')).write_bytes(data)
        counts = Counter(token for text in texts for token in tokenizer.encode(text, add_special_tokens=False).ids)
        return config, counts, dict(url=url, rows=len(texts), tokens=sum(counts.values()), uniqueTokens=len(counts),
                                   transcriptSHA256=hashlib.sha256(data).hexdigest())
    counts = Counter(); languages = {}
    with ThreadPoolExecutor(max_workers=7) as executor:
        for name, values, receipt in executor.map(language, ['en_us', 'fr_fr', 'de_de', 'es_419', 'it_it', 'pt_br', 'nl_nl']):
            counts.update(values); languages[name] = receipt
    selected = {1, 50, 106}
    for token, _ in sorted(counts.items(), key=lambda value: (-value[1], value[0])):
        if len(selected) == 32768: break
        selected.add(token)
    for token in range(tokenizer.get_vocab_size()):
        if len(selected) == 32768: break
        selected.add(token)
    data = json.dumps(sorted(selected)).encode()
    (args.output / 'vocabulary.json').write_bytes(data)
    receipt = dict(dataset='google/fleurs', license='CC BY 4.0', split='train', languages=languages,
        observedTokens=len(counts), selectedTokens=len(selected), vocabularySHA256=hashlib.sha256(data).hexdigest(),
        tokenizerSHA256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        policy='Frequency rank, force SOV EOS IDs, fill unused capacity with ascending token IDs, then sort global IDs.')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
