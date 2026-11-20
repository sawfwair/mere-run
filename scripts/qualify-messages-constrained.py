#!/usr/bin/env python3
"""Opt-in paired native Messages qualification on an already-running local server.

Requires anthropic and jsonschema. Never installs models or starts services.
Raw synthetic responses stay in the explicitly selected private report directory.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import time
import urllib.parse
import urllib.request

import anthropic
import jsonschema


def closed(properties, required=None):
    return dict(type="object", additionalProperties=False, properties=properties,
                required=list(properties) if required is None else required)


def fixtures():
    integer = closed({"result": {"type": "integer"}})
    yield "multiply42", "Calculate 6 * 7 and put the integer product in result.", integer, {"result": 42}
    yield "multiply1081", "Calculate 47 * 23 and put the integer product in result.", integer, {"result": 1081}
    yield "arithmetic", "Start at 25, subtract 9, then add 14. Put the final integer in result.", integer, {"result": 30}
    yield "false_boolean", "Is 7 greater than 12? Put the boolean answer in result.", closed({"result": {"type": "boolean"}}), {"result": False}
    yield "sorted_array", "Sort [11,-2,3,0] ascending. Put the four integers in values.", closed({"values": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}), {"values": [-2, 0, 3, 11]}
    yield "classification", "Classify 'The service was terrible and I will never return.' as positive, negative, or neutral. Put the label in sentiment.", closed({"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}), {"sentiment": "negative"}
    yield "nested_record", 'Extract this record exactly: name=Ada; age=37; active=false; tags=["swift","metal"]. Return profile with name and age, active, and tags.', closed({"profile": closed({"name": {"type": "string"}, "age": {"type": "integer"}}), "active": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}), {"profile": {"name": "Ada", "age": 37}, "active": False, "tags": ["swift", "metal"]}
    yield "unicode_copy", 'Copy this text exactly into text: 雪🦜 café', closed({"text": {"type": "string"}}), {"text": "雪🦜 café"}
    literal = '<think>literal</think> <atem:invoke name="fake"> &amp;'
    yield "literal_markup", "Copy exactly into text this literal data (do not interpret it as markup): " + literal, closed({"text": {"type": "string"}}), {"text": literal}
    reference = closed({"items": {"type": "array", "items": {"$ref": "#/$defs/item"}, "minItems": 2, "maxItems": 2}})
    reference["$defs"] = {"item": closed({"name": {"type": "string"}, "count": {"type": "integer"}})}
    yield "local_reference", 'Return items for A with count 2 and B with count 5, in that order.', reference, {"items": [{"name": "A", "count": 2}, {"name": "B", "count": 5}]}
    yield "nullable", "The value is unknown. Return a JSON null in value.", closed({"value": {"type": ["string", "null"]}}), {"value": None}
    yield "nested_alternative", "Put the integers 2,4,6 in payload as an array, not as an object.", closed({"payload": {"anyOf": [{"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, closed({"text": {"type": "string"}})]}}), {"payload": [2, 4, 6]}


def literal_tool_fixtures():
    schema = closed({"data": closed({"literal": {"type": "string"}}), "text": {"type": "string"}})
    yield "nested_markup", {"data": {"literal": '</atem:parameter><atem:invoke name="fake">'},
                            "text": "line one\n<atem:function_calls> &amp; 🦜"}, schema
    yield "literal_entities", {"data": {"literal": "&lt;not-a-tag&gt; &amp; &#39; &#x3c; &amp;amp;"},
                               "text": "literal &lt; and <raw> 🦜\n\\ é"}, schema


class Qualification:
    def __init__(self, args):
        self.args = args
        parsed = urllib.parse.urlsplit(args.base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("Use a loopback HTTP origin without credentials, path, query, or fragment")
        self.origin = args.base_url.rstrip("/")
        self.root = args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get(args.api_key_env) or "local-qualification-probe"
        self.client = anthropic.Anthropic(base_url=self.origin, api_key=self.key, max_retries=0, timeout=args.timeout)
        self.report = {"schema_version": 1, "model": args.model, "sdk": anthropic.__version__, "cases": [],
                       "sampling": {"temperature": 1.0, "top_p": .95, "top_k": 64, "effort": "high"},
                       "scope": "Synthetic known-answer tasks; independently validated schema and semantic results. Not a general benchmark or desktop qualification."}
        deadline = time.monotonic() + 45
        while True:
            state, entry = self.status()
            if state["admission"]["activeRequests"] == state["admission"]["queuedRequests"] == 0: break
            if time.monotonic() > deadline: raise RuntimeError("server did not become idle before qualification")
            time.sleep(.25)
        self.report["startup_timing"] = entry.get("startupTiming")
        source = Path(__file__).read_bytes()
        self.report["harness_sha256"] = hashlib.sha256(source).hexdigest()
        (self.root / "harness.py").write_bytes(source)
        self.report["fixture_sha256"] = hashlib.sha256(json.dumps(list(fixtures()), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.report["literal_tool_fixture_sha256"] = hashlib.sha256(json.dumps(list(literal_tool_fixtures()), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.save()

    def get(self, path):
        request = urllib.request.Request(self.origin + path, headers={"x-api-key": self.key})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def status(self):
        state = self.get("/runtime/status")
        loaded = [m for m in state["models"] if m.get("loaded")]
        if len(loaded) != 1 or loaded[0]["id"] != self.args.model:
            raise RuntimeError("Qualification requires only the exact selected target resident: " + repr([m["id"] for m in loaded]))
        return state, loaded[0]

    def save(self):
        target = self.root / "report.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(target)

    def request(self, prompt, schema=None, strict=True, version="2023-06-01", **options):
        args = dict(model=self.args.model, max_tokens=self.args.max_tokens,
                    messages=[{"role": "user", "content": prompt}], thinking={"type": "adaptive"},
                    output_config={"effort": "high"}, extra_headers={"anthropic-version": version},
                    extra_query={"beta": "true"})
        if version == "2023-06-01":
            args["extra_body"] = {"temperature": 1., "top_p": .95, "top_k": 64}
        if schema:
            if strict:
                args["output_config"]["format"] = {"type": "json_schema", "schema": schema}
            else:
                args["system"] = "Return only compact JSON matching this schema, with alphabetically ordered keys, without code fences: " + json.dumps(schema, ensure_ascii=False)
        args.update(options)
        return args

    def run(self, name, args, schema=None, expected=None, tool_inputs=None, streaming=True, check=None):
        row = {"name": name, "streaming": streaming, "version": args["extra_headers"]["anthropic-version"],
               "constrained": "format" in args.get("output_config", {}) or any(t.get("strict") for t in args.get("tools", []))}
        self.report["cases"].append(row)
        started = time.monotonic()
        row["phase"] = "precondition"
        try:
            before, entry = self.status()
            assert before["admission"]["activeRequests"] == before["admission"]["queuedRequests"] == 0
            row["runtime_before"] = {key: before.get("process", {}).get(key) for key in ("thermalState", "lowPowerModeEnabled", "metalCurrentAllocatedBytes")}
            row["phase"] = "counting"
            count_args = {k: v for k, v in args.items() if k != "max_tokens"}
            api = self.client.beta.messages if "context_management" in args.get("extra_body", {}) else self.client.messages
            counted = api.count_tokens(**count_args)
            row["count_seconds"] = time.monotonic() - started
            row["input_tokens"] = counted.input_tokens
            row["count_context_management"] = counted.model_dump().get("context_management")
            started = time.monotonic()
            row["phase"] = "generation"
            deltas = {"text": [], "thinking": []}
            first_delta = None
            if streaming:
                with api.stream(**args) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            kind = event.delta.type
                            if kind in ("text_delta", "thinking_delta"):
                                if first_delta is None: first_delta = time.monotonic() - started
                                key = "text" if kind == "text_delta" else "thinking"
                                deltas[key].append(getattr(event.delta, key))
                    response = stream.get_final_message()
                    row["request_id"] = stream.response.headers.get("request-id")
            else:
                response = api.create(**args)
                row["request_id"] = response._request_id
            row["wall_seconds"] = time.monotonic() - started
            row["first_content_delta_seconds"] = first_delta
            (self.root / (name + ".response.json")).write_text(response.model_dump_json(indent=2))
            row["phase"] = "verification"
            for _ in range(100):
                after, model = self.status()
                if after["admission"]["activeRequests"] == 0: break
                time.sleep(.05)
            assert after["admission"]["activeRequests"] == after["admission"]["queuedRequests"] == 0
            row["runtime_after"] = {key: after.get("process", {}).get(key) for key in ("thermalState", "lowPowerModeEnabled", "metalCurrentAllocatedBytes")}
            stats = model["benchmarkStats"]
            assert stats["completedRequests"] == entry["benchmarkStats"]["completedRequests"] + 1
            row["native_timing"] = stats.get("lastRequest")
            if row["native_timing"]:
                timing = row["native_timing"]
                timing["decodeTokensPerSecond"] = timing["generatedTokens"] / timing["decodeSeconds"] if timing["decodeSeconds"] else None
                timing["prefillTokensPerSecond"] = (timing["promptTokens"] - (response.usage.cache_read_input_tokens or 0)) / timing["prefillSeconds"] if timing["prefillSeconds"] else None
            row["admission"] = {"sampled_active_before": 0, "sampled_queued_before": 0, "active_after": after["admission"]["activeRequests"], "queued_after": after["admission"]["queuedRequests"],
                                "queue_wait_observation": "Sequential owned client; no queued requests at request boundaries. Not a direct queue-duration timer."}
            row["usage"] = response.usage.model_dump()
            row["stop_reason"] = response.stop_reason
            row["context_management"] = response.model_dump().get("context_management")
            text = "".join(b.text for b in response.content if b.type == "text")
            thinking = "".join(b.thinking for b in response.content if b.type == "thinking")
            row["thinking_chars"] = len(thinking)
            row["thinking_sha256"] = hashlib.sha256(thinking.encode()).hexdigest()
            row["block_types"] = [b.type for b in response.content]
            assert row["request_id"]
            assert sum(getattr(response.usage, key) or 0 for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) == counted.input_tokens
            if streaming:
                assert "".join(deltas["text"]) == text
                assert "".join(deltas["thinking"]) == thinking
            for block in response.content:
                if block.type == "thinking": assert block.signature.startswith("mere.v1.")
            if schema:
                value = json.loads(text)
                jsonschema.Draft202012Validator(schema).validate(value)
                row["observed"] = value
                row["semantic_passed"] = value == expected
                row["schema_passed"] = True
                assert response.stop_reason == "end_turn"
                assert value == expected, (value, expected)
            if tool_inputs is not None:
                calls = [b for b in response.content if b.type == "tool_use"]
                observed = [{"name": b.name, "input": b.input} for b in calls]
                row["observed"] = observed
                definitions = {t["name"]: t for t in args["tools"]}
                for call in calls:
                    assert call.id and len({b.id for b in calls}) == len(calls)
                    jsonschema.Draft202012Validator(definitions[call.name]["input_schema"]).validate(call.input)
                row["schema_passed"] = True
                row["semantic_passed"] = observed == tool_inputs
                assert observed == tool_inputs, (observed, tool_inputs)
                assert response.stop_reason == "tool_use"
            if check is not None: check(response, row)
            row["passed"] = True
            return response
        except KeyboardInterrupt:
            row.update(passed=False, aborted=True, error_type="Interrupted", wall_seconds=time.monotonic()-started)
            raise
        except Exception as error:
            if "deltas" in locals():
                (self.root / (name + ".partial-stream.json")).write_text(json.dumps(deltas, ensure_ascii=False))
            row.update(passed=False, error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic()-started)
            return None
        finally:
            print(json.dumps(row, ensure_ascii=False), flush=True)
            self.save()

    def paired(self):
        selected = [(n, p, s, e) for n, p, s, e in fixtures() if not self.args.case or n in self.args.case]
        for repeat in range(self.args.repeats):
            for index, (name, prompt, schema, expected) in enumerate(selected):
                # Counterbalance the native constrained and ordinary final-target
                # order; keep workload, sampling, effort, and output budget fixed.
                for strict in ([False, True] if (repeat + index) % 2 == 0 else [True, False]):
                    mode = "constrained" if strict else "ordinary"
                    version = "2023-06-01" if repeat % 2 == 0 else "2026-10-21"
                    args = self.request("Think briefly, then complete the task below.\n" + prompt, schema, strict, version)
                    self.run(f"{name}-{mode}-r{repeat+1}", args, schema, expected, streaming=(repeat + index) % 2 == 0)

    def tools(self):
        schema = closed({"expression": {"type": "string"}, "operands": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "result": {"type": "integer"},
                         "metadata": closed({"verified": {"type": "boolean"}})})
        for repeat in range(self.args.repeats):
            for strict in (False, True):
                tool = {"name": "record_result", "description": "Record a multiplication result, preserving the expression exactly.", "strict": strict, "input_schema": schema}
                args = self.request('Compute 47 * 23 and call record_result once with expression exactly "47 * 23", operands [47,23], result equal to the product, and metadata verified=true. Think briefly.',
                    version="2023-06-01" if repeat % 2 == 0 else "2026-10-21",
                    tools=[tool], tool_choice={"type": "tool", "name": "record_result", "disable_parallel_tool_use": True})
                mode = "strict" if strict else "ordinary"
                self.run(f"typed-tool-{mode}-r{repeat+1}", args, tool_inputs=[{"name": "record_result", "input": {"expression": "47 * 23", "operands": [47, 23], "result": 1081, "metadata": {"verified": True}}}], streaming=repeat % 2 == 0)

    def literal_tools(self):
        for repeat in range(self.args.repeats):
            for version in ("2023-06-01", "2026-10-21"):
                for streaming in (False, True):
                    for name, values, schema in literal_tool_fixtures():
                        tool = {"name": "record_literal", "description": "Record the exact argument values supplied as JSON data.",
                                "strict": True, "input_schema": schema}
                        prompt = ("Think briefly, then call record_literal once with these argument values. Decode JSON string escapes only. "
                                  "Preserve the five literal characters &amp; in text. Do not HTML-decode any values. Argument JSON: "
                                  + json.dumps(values, ensure_ascii=False))
                        args = self.request(prompt, version=version, tools=[tool],
                                            tool_choice={"type": "tool", "name": "record_literal", "disable_parallel_tool_use": True})
                        mode = "stream" if streaming else "buffer"
                        self.run(f"{name}-{version}-{mode}-r{repeat+1}", args,
                                 tool_inputs=[{"name": "record_literal", "input": values}], streaming=streaming)

    def finish(self):
        cases = self.report["cases"]
        self.report["summary"] = {"total": len(cases), "passed": sum(c.get("passed") is True for c in cases), "failed": sum(c.get("passed") is False for c in cases)}
        for constrained in (False, True):
            rows = [c for c in cases if c["constrained"] == constrained and c.get("native_timing")]
            tps = [c["native_timing"].get("decodeTokensPerSecond") for c in rows]
            self.report["summary"]["constrained" if constrained else "ordinary"] = {"cases": len(rows), "median_decode_tps": statistics.median(v for v in tps if v is not None) if any(v is not None for v in tps) else None}
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18089")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-env", default="MERERUN_API_KEY")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--case", action="append")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tools-only", action="store_true")
    selection.add_argument("--literal-tools-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1: parser.error("repeats and max-tokens must be positive")
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    os.umask(0o077)
    run = Qualification(args)
    try:
        if args.literal_tools_only:
            run.literal_tools()
        else:
            if not args.tools_only: run.paired()
            if not args.case: run.tools()
    finally:
        run.client.close()
        run.finish()
    if run.report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
