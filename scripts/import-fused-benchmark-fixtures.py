#!/usr/bin/env python3
"""Import the pinned external cases selected by mere-fused-v1.

The importer downloads only the version-locked source artifacts in
mere-fused-sources-v1.json, verifies their bytes (or selected raw JSONL row),
and writes normalized unstamped JSONL. Use `mere.run model benchmark
fused-fixture` to stamp the canonical content hashes.
"""

import argparse
import ast
import base64
import copy
import gzip
import hashlib
import io
import json
import pathlib
import pickle
import re
import sys
import urllib.request
import zipfile
import zlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    REPO_ROOT
    / "Sources"
    / "MereRunCLI"
    / "BenchmarkSuites"
    / "mere-fused-sources-v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / ".tmp" / "fused-benchmark-fixtures" / "unstamped.jsonl"
DEFAULT_CACHE = REPO_ROOT / ".tmp" / "fused-benchmark-sources"

HUMANEVAL_IDS = ["HumanEval/0", "HumanEval/3", "HumanEval/8", "HumanEval/32", "HumanEval/53", "HumanEval/81"]
MBPP_IDS = ["Mbpp/2", "Mbpp/3", "Mbpp/4", "Mbpp/6", "Mbpp/7", "Mbpp/56"]

BFCL_SINGLE = [
    ("bfcl.v3.simple-python-0", "v3:simple_python:0", "BFCL_v3_simple.json", True),
    ("bfcl.v3.simple-java-0", "v3:simple_java:0", "BFCL_v3_java.json", True),
    ("bfcl.v3.parallel-0", "v3:parallel:0", "BFCL_v3_parallel.json", True),
    ("bfcl.v3.multiple-0", "v3:multiple:0", "BFCL_v3_multiple.json", True),
    ("bfcl.v3.parallel-multiple-0", "v3:parallel_multiple:0", "BFCL_v3_parallel_multiple.json", True),
    ("bfcl.v3.irrelevance-0", "v3:irrelevance:0", "BFCL_v3_irrelevance.json", False),
]

BFCL_MULTI = [
    ("bfcl.v3.multi-turn-base-0", "v3:multi_turn_base:0", "BFCL_v3_multi_turn_base.json", 3),
    ("bfcl.v3.multi-turn-miss-func-0", "v3:multi_turn_miss_func:0", "BFCL_v3_multi_turn_miss_func.json", 3),
    ("bfcl.v3.multi-turn-miss-param-0", "v3:multi_turn_miss_param:0", "BFCL_v3_multi_turn_miss_param.json", 3),
    ("bfcl.v3.multi-turn-long-context-0", "v3:multi_turn_long_context:0", "BFCL_v3_multi_turn_long_context.json", 3),
]

LONG_BENCH = [
    ("longbench.hotpotqa-0", "hotpotqa", "hotpotqa:test:0", "qa-f1", 0.5, 128),
    ("longbench.gov-report-0", "gov_report", "gov_report:test:0", "rouge-l", 0.2, 512),
    ("longbench.qasper-0", "qasper", "qasper:test:0", "qa-f1", 0.5, 128),
    ("longbench.2wikimqa-0", "2wikimqa", "2wikimqa:test:0", "qa-f1", 0.5, 128),
    ("longbench.musique-0", "musique", "musique:test:0", "qa-f1", 0.5, 128),
    ("longbench.multi-news-0", "multi_news", "multi_news:test:0", "rouge-l", 0.2, 512),
    ("longbench.passage-retrieval-0", "passage_retrieval_en", "passage_retrieval_en:test:0", "retrieval", 1.0, 64),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_verified(url, destination, expected_sha, offline):
    if destination.exists() and sha256(destination.read_bytes()) == expected_sha:
        return destination.read_bytes()
    if offline:
        raise RuntimeError("missing or invalid offline source: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mere.run-fused-import/1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    actual = sha256(data)
    if actual != expected_sha:
        raise RuntimeError("SHA-256 mismatch for %s: got %s" % (url, actual))
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return data


def read_jsonl(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def fixture_base(fixture_id, kind, source_version, source_revision, original_id, messages):
    return {
        "id": fixture_id,
        "kind": kind,
        "sourceVersion": source_version,
        "sourceRevision": source_revision,
        "originalID": original_id,
        "messages": messages,
        "contentSHA256": "",
    }


def code_messages(prompt):
    return [
        {
            "role": "system",
            "content": "Complete the Python task. Return only valid Python implementation code without Markdown or prose.",
        },
        {"role": "user", "content": prompt},
    ]


def reference_function(task):
    entry = task["entry_point"]
    if task["task_id"].startswith("HumanEval/"):
        source = task["prompt"] + task["canonical_solution"]
    else:
        source = task["canonical_solution"]
    return re.sub(r"\b%s\b" % re.escape(entry), "_mere_reference", source)


def evalplus_test_code(task):
    inputs = task["base_input"] + task["plus_input"]
    deduplicated = []
    seen = set()
    for item in inputs:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    entry = task["entry_point"]
    special_set = entry in {"similar_elements", "find_char_long"}
    find_zero = entry == "find_zero"
    return """\
import copy as _mere_copy
import math as _mere_math

{reference}

def _mere_poly(coefficients, value):
    return sum(coefficient * _mere_math.pow(value, index) for index, coefficient in enumerate(coefficients))

def _mere_equal(actual, expected):
    if {special_set!r}:
        return set(actual) == set(expected)
    if isinstance(expected, float):
        return abs(actual - expected) <= max({atol!r}, 1e-6)
    return actual == expected

def check(candidate):
    inputs = {inputs!r}
    for item in inputs:
        candidate_input = _mere_copy.deepcopy(item)
        if {find_zero!r}:
            root = candidate(*candidate_input)
            assert abs(_mere_poly(candidate_input[0], root)) <= {atol!r}
        else:
            expected = _mere_reference(*_mere_copy.deepcopy(item))
            actual = candidate(*candidate_input)
            assert _mere_equal(actual, expected), (item, actual, expected)
""".format(
        reference=reference_function(task),
        special_set=special_set,
        atol=task["atol"],
        inputs=deduplicated,
        find_zero=find_zero,
    )


def import_evalplus(lock, cache, offline):
    source = lock["evalplus"]
    revision = source["repositoryRevision"]
    families = [
        (source["humanEval"], HUMANEVAL_IDS, "humaneval-plus.", "HumanEval/"),
        (source["mbpp"], MBPP_IDS, "mbpp-plus.", "Mbpp/"),
    ]
    fixtures = []
    for resource, selected_ids, fixture_prefix, task_prefix in families:
        compressed = fetch_verified(
            resource["url"],
            cache / pathlib.Path(resource["url"]).name,
            resource["sha256"],
            offline,
        )
        rows = {row["task_id"]: row for row in read_jsonl(gzip.decompress(compressed))}
        source_revision = "evalplus@%s;%s@%s;sha256:%s" % (
            revision,
            "HumanEvalPlus" if task_prefix == "HumanEval/" else "MbppPlus",
            resource["version"],
            resource["sha256"],
        )
        for task_id in selected_ids:
            task = rows[task_id]
            fixture = fixture_base(
                fixture_prefix + task_id.split("/")[-1],
                "code",
                resource["version"],
                source_revision,
                task_id,
                code_messages(task["prompt"]),
            )
            fixture.update(
                {
                    "entryPoint": task["entry_point"],
                    "tests": evalplus_test_code(task),
                    "codeEvaluation": "function",
                    "generationBudget": 1024,
                }
            )
            fixtures.append(fixture)
    return fixtures


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("global objects are forbidden in benchmark data")


def decode_lcb_cases(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        unpacked = zlib.decompress(base64.b64decode(value))
        encoded_json = SafeUnpickler(io.BytesIO(unpacked)).load()
        return json.loads(encoded_json)


def selected_lcb_rows(source, cache, offline):
    selected = {item["index"]: item for item in source["selectedRows"]}
    cache_file = cache / "livecodebench-selected.jsonl"
    if cache_file.exists():
        rows = read_jsonl(cache_file.read_bytes())
        if len(rows) == len(selected):
            by_index = {row["_mereRowIndex"]: row for row in rows}
            if all(
                index in by_index
                and by_index[index]["_mereRawSHA256"] == pin["rawSHA256"]
                for index, pin in selected.items()
            ):
                return by_index
    if offline:
        raise RuntimeError("LiveCodeBench selected-row cache is missing or invalid")
    request = urllib.request.Request(source["url"], headers={"User-Agent": "mere.run-fused-import/1"})
    captured = {}
    with urllib.request.urlopen(request, timeout=180) as response:
        for index in range(max(selected) + 1):
            raw = response.readline()
            if not raw:
                raise RuntimeError("LiveCodeBench source ended before selected row %d" % index)
            if index not in selected:
                continue
            raw_without_newline = raw.rstrip(b"\r\n")
            actual = sha256(raw_without_newline)
            if actual != selected[index]["rawSHA256"]:
                raise RuntimeError("LiveCodeBench selected row %d changed: %s" % (index, actual))
            row = json.loads(raw)
            if row["question_id"] != selected[index]["questionID"]:
                raise RuntimeError("LiveCodeBench question id mismatch at row %d" % index)
            row["_mereRowIndex"] = index
            row["_mereRawSHA256"] = actual
            captured[index] = row
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        "".join(json.dumps(captured[index], sort_keys=True) + "\n" for index in sorted(captured)),
        encoding="utf-8",
    )
    return captured


def import_livecodebench(lock, cache, offline):
    source = lock["liveCodeBench"]
    rows = selected_lcb_rows(source, cache, offline)
    fixtures = []
    revision = "dataset@%s;runner@%s" % (
        source["datasetRevision"],
        source["repositoryRevision"],
    )
    for pin in source["selectedRows"]:
        row = rows[pin["index"]]
        metadata = json.loads(row["metadata"])
        entry_point = metadata.get("func_name")
        evaluation = "functional" if entry_point else "stdin"
        prompt = row["question_content"]
        if row["starter_code"]:
            prompt += "\n\nStarter code:\n" + row["starter_code"]
        cases = decode_lcb_cases(row["public_test_cases"]) + decode_lcb_cases(row["private_test_cases"])
        fixture = fixture_base(
            pin["fixtureID"],
            "code",
            source["version"],
            revision,
            "%s:test:%s" % (source["version"], row["question_id"]),
            code_messages(prompt),
        )
        fixture.update(
            {
                "entryPoint": entry_point,
                "codeEvaluation": evaluation,
                "codeTests": [{"input": case["input"], "output": case["output"]} for case in cases],
                "generationBudget": 2048,
            }
        )
        fixtures.append(fixture)
    return fixtures


def bfcl_resource(source, relative_path, cache, offline):
    return fetch_verified(
        source["rawBaseURL"] + relative_path,
        cache / "bfcl" / relative_path,
        source["resources"][relative_path],
        offline,
    )


def first_jsonl(data):
    return json.loads(data.decode("utf-8").splitlines()[0])


def tool_definition(item):
    properties = {}
    for name, definition in item["parameters"]["properties"].items():
        properties[name] = {
            "type": definition.get("type", "string"),
            "description": definition.get("description", ""),
        }
    return {
        "name": item["name"],
        "description": item.get("description", ""),
        "parameters": properties,
        "required": item["parameters"].get("required", []),
    }


def canonical_argument(value):
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bfcl_expectations(ground_truth):
    expectations = []
    for call in ground_truth:
        if not isinstance(call, dict) or len(call) != 1:
            raise RuntimeError("unsupported BFCL ground truth: %r" % call)
        name, arguments = next(iter(call.items()))
        expectations.append(
            {
                "name": name,
                "arguments": {
                    key: [canonical_argument(value) for value in accepted]
                    for key, accepted in arguments.items()
                },
            }
        )
    return expectations


def parse_call_expression(expression):
    parsed = ast.parse(expression, mode="eval").body
    if not isinstance(parsed, ast.Call):
        raise RuntimeError("unsupported BFCL call expression: %s" % expression)

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return dotted_name(node.value) + "." + node.attr
        raise RuntimeError("unsupported BFCL call name")

    return {
        "name": dotted_name(parsed.func),
        "arguments": {keyword.arg: ast.literal_eval(keyword.value) for keyword in parsed.keywords},
    }


def history_tool_messages(turn_index, expressions):
    if not expressions:
        return [{"role": "assistant", "content": "No available tool can complete this request."}]
    calls = [parse_call_expression(expression) for expression in expressions]
    assistant_calls = []
    messages = []
    for index, call in enumerate(calls):
        call_id = "bfcl-turn-%d-call-%d" % (turn_index, index)
        assistant_calls.append(
            {
                "id": call_id,
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
    messages.append({"role": "assistant", "content": "", "toolCalls": assistant_calls})
    for call in assistant_calls:
        messages.append(
            {
                "role": "tool",
                "content": "Tool call completed; scenario state updated.",
                "name": call["name"],
                "toolCallID": call["id"],
            }
        )
    return messages


def import_bfcl(lock, cache, offline):
    source = lock["bfcl"]
    revision = source["repositoryRevision"]
    fixtures = []
    for fixture_id, original_id, filename, has_answer in BFCL_SINGLE:
        row = first_jsonl(bfcl_resource(source, filename, cache, offline))
        fixture = fixture_base(
            fixture_id,
            "tool",
            source["version"],
            revision,
            original_id,
            [
                {
                    "role": "system",
                    "content": "Use the supplied tools exactly. Make every required call, make no extra calls, and do not invent missing functions or parameters.",
                },
                row["question"][0][0],
            ],
        )
        fixture["tools"] = [tool_definition(item) for item in row["function"]]
        if has_answer:
            answer = first_jsonl(
                bfcl_resource(source, "possible_answer/" + filename, cache, offline)
            )
            fixture["expectedToolCalls"] = bfcl_expectations(answer["ground_truth"])
        else:
            fixture["expectsNoToolCalls"] = True
        fixture["generationBudget"] = 256
        fixtures.append(fixture)

    function_docs = []
    for filename in ["multi_turn_func_doc/gorilla_file_system.json", "multi_turn_func_doc/posting_api.json"]:
        function_docs.extend(read_jsonl(bfcl_resource(source, filename, cache, offline)))

    for fixture_id, original_id, filename, target_turn in BFCL_MULTI:
        row = first_jsonl(bfcl_resource(source, filename, cache, offline))
        answer = first_jsonl(
            bfcl_resource(source, "possible_answer/" + filename, cache, offline)
        )
        tools = copy.deepcopy(function_docs)
        if "miss_func" in filename:
            held_names = {
                name for names in row.get("missed_function", {}).values() for name in names
            }
            if target_turn < min(map(int, row["missed_function"].keys())):
                tools = [item for item in tools if item["name"] not in held_names]
        messages = [
            {
                "role": "system",
                "content": (
                    "Continue this pinned BFCL scenario using the supplied tools. Initial state:\n"
                    + json.dumps(row["initial_config"], sort_keys=True, separators=(",", ":"))
                ),
            }
        ]
        for turn in range(target_turn):
            current = row["question"][turn]
            if current:
                messages.extend(current)
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "I have updated some more functions you can choose from. What about now?",
                    }
                )
            messages.extend(history_tool_messages(turn, answer["ground_truth"][turn]))

        current = row["question"][target_turn]
        if current:
            messages.extend(current)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "I have updated some more functions you can choose from. What about now?",
                }
            )
        fixture = fixture_base(
            fixture_id,
            "tool",
            source["version"],
            revision,
            original_id,
            messages,
        )
        fixture["tools"] = [tool_definition(item) for item in tools]
        expected = answer["ground_truth"][target_turn]
        if expected:
            fixture["expectedToolCalls"] = [
                {
                    "name": call["name"],
                    "arguments": {
                        key: [canonical_argument(value)] for key, value in call["arguments"].items()
                    },
                }
                for call in map(parse_call_expression, expected)
            ]
        else:
            fixture["expectsNoToolCalls"] = True
            if "miss_param" in filename:
                fixture["requiredPhrases"] = ["file"]
        fixture["generationBudget"] = 512
        fixtures.append(fixture)
    return fixtures


def import_longbench(lock, cache, offline):
    source = lock["longBench"]
    archive_data = fetch_verified(
        source["dataURL"], cache / "longbench-data.zip", source["dataSHA256"], offline
    )
    prompt_data = fetch_verified(
        source["promptURL"],
        cache / "longbench-dataset2prompt.json",
        source["promptSHA256"],
        offline,
    )
    prompts = json.loads(prompt_data)
    archive = zipfile.ZipFile(io.BytesIO(archive_data))
    revision = "dataset@%s;prompts@%s" % (
        source["datasetRevision"],
        source["repositoryRevision"],
    )
    fixtures = []
    for fixture_id, dataset, original_id, metric, threshold, budget in LONG_BENCH:
        with archive.open("data/%s.jsonl" % dataset) as stream:
            row = json.loads(stream.readline())
        prompt = prompts[dataset].format(context=row["context"], input=row["input"])
        fixture = fixture_base(
            fixture_id,
            "chat",
            source["version"],
            revision,
            original_id,
            [{"role": "user", "content": prompt}],
        )
        fixture.update(
            {
                "textMetric": metric,
                "referenceAnswers": row["answers"],
                "passThreshold": threshold,
                "generationBudget": budget,
            }
        )
        fixtures.append(fixture)
    return fixtures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=pathlib.Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schemaVersion") != 1:
        raise RuntimeError("unsupported fused source lock schema")
    fixtures = []
    fixtures.extend(import_evalplus(lock, args.cache_dir, args.offline))
    fixtures.extend(import_livecodebench(lock, args.cache_dir, args.offline))
    fixtures.extend(import_bfcl(lock, args.cache_dir, args.offline))
    fixtures.extend(import_longbench(lock, args.cache_dir, args.offline))
    if len(fixtures) != 35 or len({item["id"] for item in fixtures}) != 35:
        raise RuntimeError("expected 35 unique imported fixtures, got %d" % len(fixtures))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for fixture in fixtures:
            stream.write(json.dumps(fixture, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    print("Imported 35 pinned fused fixtures to %s" % args.output, file=sys.stderr)


if __name__ == "__main__":
    main()
