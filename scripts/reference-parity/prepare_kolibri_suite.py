#!/usr/bin/env python3
"""Tokenize a small, explicit calibration/holdout suite once for paired logits.

This diagnostic suite is not a standardized benchmark or broad quality claim.
Requires tokenizers and jinja2; reads only the pinned source tokenizer files.
"""
import argparse
import hashlib
import json
from pathlib import Path
from jinja2 import Environment
from tokenizers import Tokenizer

CASES = [
    ("cal-en", "en", "reasoning", "calibration", "Three boxes each contain 8 marbles. Five marbles are removed. How many remain? Answer briefly.", "19 marbles remain: 3 × 8 − 5 = 19."),
    ("cal-de", "de", "reasoning", "calibration", "Ein Zug fährt 90 Kilometer in 1,5 Stunden. Wie hoch ist seine Durchschnittsgeschwindigkeit? Antworte kurz.", "Die Durchschnittsgeschwindigkeit beträgt 60 km/h: 90 ÷ 1,5 = 60."),
    ("cal-code", "en", "code", "calibration", "Write a Python function square(x) that returns x squared. Return only code.", "def square(x):\n    return x * x"),
    ("held-en-fact", "en", "knowledge", "heldout", "What is the chemical symbol for gold? Answer in one sentence.", "The chemical symbol for gold is Au."),
    ("held-de-fact", "de", "knowledge", "heldout", "Welches chemische Symbol hat Eisen? Antworte in einem Satz.", "Das chemische Symbol für Eisen ist Fe."),
    ("held-en-arithmetic", "en", "reasoning", "heldout", "A shop sells 17 notebooks at $6 each and refunds $12. What is its net revenue? Answer briefly.", "The net revenue is $90: 17 × $6 − $12 = $90."),
    ("held-de-arithmetic", "de", "reasoning", "heldout", "Eine Bäckerei verkauft 24 Brote für je 4 Euro und gibt 8 Euro zurück. Wie hoch ist der Nettoerlös? Antworte kurz.", "Der Nettoerlös beträgt 88 Euro: 24 × 4 − 8 = 88."),
    ("held-en-logic", "en", "reasoning", "heldout", "All whales are mammals. Some mammals can fly. Does it follow that some whales can fly? Explain briefly.", "No. The mammals that can fly need not include any whales, so the conclusion does not follow."),
    ("held-de-logic", "de", "reasoning", "heldout", "Alle Rosen sind Pflanzen. Manche Pflanzen sind giftig. Folgt daraus, dass manche Rosen giftig sind? Begründe kurz.", "Nein. Die giftigen Pflanzen müssen keine Rosen sein; die Schlussfolgerung folgt daher nicht."),
    ("held-en-code", "en", "code", "heldout", "Write a Python function is_even(n) that returns whether n is even. Return only code.", "def is_even(n):\n    return n % 2 == 0"),
    ("held-de-code", "de", "code", "heldout", "Schreibe eine Python-Funktion first_or_none(items), die das erste Element oder bei einer leeren Liste None zurückgibt. Gib nur Code aus.", "def first_or_none(items):\n    return items[0] if items else None"),
    ("held-en-translate", "en", "translation", "heldout", "Translate to German: The library opens at nine and closes at six.", "Die Bibliothek öffnet um neun Uhr und schließt um sechs Uhr."),
    ("held-de-translate", "de", "translation", "heldout", "Übersetze ins Englische: Die Brücke wurde vor hundert Jahren gebaut.", "The bridge was built a hundred years ago."),
]


def prepare(source: Path, output: Path):
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    metadata = json.loads((source / "tokenizer_config.json").read_text())
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    def fail(message): raise ValueError(message)
    env.globals["raise_exception"] = fail
    template = env.from_string(metadata["chat_template"])
    cases = list(CASES)
    for language in ["en", "de"]:
        lines = [f"Record {i}: reference code R{i:04d}, status checked." for i in range(80)]
        lines[7] = "Record 7: reference code AZURE-731, status verified."
        if language == "en":
            prompt = "Read these records.\n" + "\n".join(lines) + "\nWhat is the reference code in Record 7? Return only the code."
        else:
            prompt = "Lies diese Datensätze.\n" + "\n".join(lines) + "\nWelcher Referenzcode steht in Record 7? Gib nur den Code aus."
        cases.append((f"held-{language}-window", language, "window-crossing", "heldout", prompt, "AZURE-731"))
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    cases.append(("held-en-tool", "en", "tools", "heldout", "Use the weather tool to look up the weather in Halifax.", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Halifax"}}\n</tool_call>'))
    rows = []
    texts = []
    for case_id, language, task, split, prompt, continuation in cases:
        rendered = template.render(messages=[{"role": "user", "content": prompt}], tools=tools if task == "tools" else None,
                                   add_generation_prompt=True, enable_thinking=False, preserve_thinking=True)
        prefix = tokenizer.encode(rendered, add_special_tokens=False).ids
        targets = tokenizer.encode(continuation, add_special_tokens=False).ids + [127906]
        rows.append(dict(id=case_id, language=language, task=task, split=split,
                         tokens=prefix + targets, scoreStart=len(prefix)))
        texts.append(dict(id=case_id, prompt=rendered, continuation=continuation))
    output.mkdir(parents=True, exist_ok=True)
    (output / "suite.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "texts.json").write_text(json.dumps(texts, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(rows), "tokens": sum(len(r["tokens"]) for r in rows),
                      "scored_tokens": sum(len(r["tokens"]) - r["scoreStart"] for r in rows),
                      "suite_sha256": hashlib.sha256((output / "suite.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
