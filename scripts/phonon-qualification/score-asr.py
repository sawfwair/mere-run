#!/usr/bin/env python3
"""Score local ASR transcripts against supported corpus text. Raw diagnostics stay unscored."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("transcripts", type=Path, help="JSON rows: runtime, id, transcript")
parser.add_argument("output", type=Path)
args = parser.parse_args()
corpus = {p["id"]: p for p in json.loads(Path(__file__).with_name("corpus.json").read_text())}

def words(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE)

def distance(reference, actual):
    previous = list(range(len(actual) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for j, candidate in enumerate(actual, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1]

rows = json.loads(args.transcripts.read_text())
summary = {}
for row in rows:
    prompt = corpus[row["id"]]
    row["scope"] = prompt["scope"]
    if prompt["scope"] != "supported":
        continue
    reference, actual = words(prompt["text"]), words(row["transcript"])
    row.update(reference_words=len(reference), edits=distance(reference, actual))
    total = summary.setdefault(row["runtime"], dict(cases=0, reference_words=0, edits=0))
    total["cases"] += 1
    total["reference_words"] += len(reference)
    total["edits"] += row["edits"]
for total in summary.values():
    total["wer"] = total["edits"] / total["reference_words"]
args.output.write_text(json.dumps(dict(normalization="NFKC, lowercase, punctuation removed; contractions retained", summary=summary, rows=rows), indent=2) + "\n")
print(json.dumps(summary, indent=2))
