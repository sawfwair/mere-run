#!/usr/bin/env python3
"""Opt-in native qualification against an already-running mere.run server.

Requires the official anthropic SDK. Never starts services, installs models,
or executes client tools. Writes metrics without credentials or thinking text.
"""
import argparse
import base64
import copy
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import anthropic


def pdf_fixture():
    """Two source pages using the built-in Helvetica font, with no external assets."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>"]
    for content in (6, 7):
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
                        "/Resources << /Font << /F1 5 0 R >> >> /Contents %d 0 R >>" % content).encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for text in ("Alpha source page: code 42", "Beta source page: code 1081"):
        stream = ("BT /F1 12 Tf 20 200 Td (" + text + ") Tj ET\n").encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream")
    result = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(result)
    result += b"xref\n0 8\n0000000000 65535 f \n"
    result += b"".join(("%010d 00000 n \n" % offset).encode() for offset in offsets[1:])
    return result + ("trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref).encode()


def qualify(args, report):
    origin = args.base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Use an HTTP(S) server origin without credentials, query, or fragment")
    key = os.environ.get(args.api_key_env) or "local-qualification-probe"
    request = urllib.request.Request(origin + "/v1/models/" + urllib.parse.quote(args.model, safe=""), headers={"x-api-key": key})
    with urllib.request.urlopen(request, timeout=10) as response:
        info = json.load(response)
    caps = info["local_capabilities"]
    required = ("messages", "streaming", "token_counting") if args.vision_only else ("messages", "streaming", "tools", "token_counting", "tool_search", "documents")
    for name in required:
        if not caps.get(name):
            raise ValueError("Selected adapter does not support " + name)
    if "adaptive" not in caps["thinking"]:
        raise ValueError("Selected adapter does not support adaptive thinking")
    if not args.vision_only and not {"clear_thinking_20251015", "clear_tool_uses_20250919"}.issubset(caps["context_editing"]):
        raise ValueError("Selected adapter does not support both context-edit strategies")
    report.update(model=info["id"], sdk=anthropic.__version__, capabilities=caps,
                  limits={"context_tokens": info.get("max_input_tokens"), "output_tokens": info.get("max_tokens")})
    # An explicit key disables the SDK's credential auto-discovery.
    client = anthropic.Anthropic(base_url=origin, api_key=key, max_retries=0, timeout=360)
    api = client.beta.messages
    headers = {"anthropic-version": args.api_version}

    def measured(name, operation):
        start = time.monotonic()
        response = operation()
        row = {"name": name, "seconds": time.monotonic() - start, "request_id": getattr(response, "_request_id", None)}
        if hasattr(response, "usage"):
            row.update(usage=response.usage.model_dump(), stop_reason=response.stop_reason, blocks=[b.type for b in response.content])
        else:
            row.update(input_tokens=response.input_tokens, context_management=response.model_dump().get("context_management"))
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        return response

    def vision():
        if not caps["ordered_images"]:
            raise ValueError("Selected adapter does not support ordered images")
        images = []
        for path in args.vision_images:
            media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower())
            if not media:
                raise ValueError("Use PNG or JPEG vision fixtures")
            images.append({"type": "image", "source": {"type": "base64", "media_type": media,
                "data": base64.b64encode(path.read_bytes()).decode()}})
        for first, second in ((0, 1), (1, 0)):
            options = dict(model=args.model, thinking={"type": "adaptive"}, extra_headers=headers,
                extra_body={"cache_control": {"type": "ephemeral"}}, messages=[{"role": "user", "content": [
                    {"type": "text", "text": "First image:"}, images[first],
                    {"type": "text", "text": "Second image:"}, images[second],
                    {"type": "text", "text": "Return only the word printed in the FIRST image."}]}])
            if "low" in caps["effort"]:
                options["output_config"] = {"effort": "low"}
            count = measured("vision_count", lambda: api.count_tokens(**options))

            def generate_image():
                with api.stream(**options, max_tokens=min(args.max_tokens, 256)) as stream:
                    stream.until_done()
                    return stream.get_final_message()

            result = measured("vision_order_stream", generate_image)
            text = "".join(block.text for block in result.content if block.type == "text").strip()
            report["cases"][-1].update(expected_first=args.vision_labels[first], observed_text=text)
            words = re.findall(r"[A-Z]+", text.upper())
            assert args.vision_labels[first].upper() in words and args.vision_labels[second].upper() not in words
            usage = result.usage
            assert usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens == count.input_tokens
            if caps["prompt_cache"]:
                assert 0 < usage.cache_read_input_tokens + usage.cache_creation_input_tokens < count.input_tokens
                warm = measured("vision_covering_marker_warm", lambda: api.create(**options, max_tokens=0))
                usage = warm.usage
                assert usage.output_tokens == 0 and usage.input_tokens > 0
                assert usage.cache_read_input_tokens + usage.cache_creation_input_tokens > 0
                assert usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens == count.input_tokens
            else:
                assert usage.cache_read_input_tokens == usage.cache_creation_input_tokens == 0
                try:
                    api.create(**options, max_tokens=0)
                    raise AssertionError("Unsupported image cache warming was accepted")
                except anthropic.BadRequestError:
                    report["cases"].append({"name": "unsupported_vision_warming", "status": 400})

        if caps["prompt_cache"]:
            prefix = {"type": "text", "text": "Read the labeled images carefully. " * 64,
                      "cache_control": {"type": "ephemeral"}}
            for first, second in ((0, 1), (1, 0)):
                options = dict(model=args.model, system=[prefix], thinking={"type": "adaptive"},
                    extra_headers=headers, messages=[{"role": "user", "content": [
                        images[first], images[second],
                        {"type": "text", "text": "Return only the word printed in the FIRST image."}]}])
                if "low" in caps["effort"]:
                    options["output_config"] = {"effort": "low"}
                count = measured("vision_text_prefix_count", lambda: api.count_tokens(**options))
                if first == 0:
                    warm = measured("vision_text_prefix_warm", lambda: api.create(**options, max_tokens=0))
                    assert warm.usage.cache_read_input_tokens + warm.usage.cache_creation_input_tokens > 0

                def generate_cached_image():
                    with api.stream(**options, max_tokens=min(args.max_tokens, 256)) as stream:
                        stream.until_done()
                        return stream.get_final_message()

                result = measured("vision_text_prefix_reuse", generate_cached_image)
                usage = result.usage
                assert usage.cache_read_input_tokens > 0
                assert usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens == count.input_tokens
                text = "".join(block.text for block in result.content if block.type == "text").strip()
                report["cases"][-1].update(expected_first=args.vision_labels[first], observed_text=text)
                words = re.findall(r"[A-Z]+", text.upper())
                assert args.vision_labels[first].upper() in words and args.vision_labels[second].upper() not in words

    if args.vision_only:
        vision()
        report["passed"] = True
        return

    seed = measured("reasoning_seed", lambda: api.create(model=args.model, max_tokens=128, thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": "Think briefly about the next step, then reply READY."}], extra_headers=headers))
    thinking = [b.model_dump() for b in seed.content if b.type == "thinking"]
    assert thinking, "This seed did not produce thinking; reasoning replay is unqualified"
    search = "tool_search_tool_" + args.search
    tools = [{"type": search + "_20251119", "name": search},
        {"name": "read", "description": "Read a local client fixture.", "input_schema": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "artifact_lookup", "description": "Find an artifact record by key.", "defer_loading": True,
            "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"], "additionalProperties": False}}]
    pdf = {"type": "document", "title": "Local source", "context": "Two synthetic source pages.",
        "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf_fixture()).decode()}}
    query = "pattern" if args.search == "regex" else "query"
    history = [{"role": "user", "content": "Earlier fixture request."},
        {"role": "assistant", "content": thinking + [{"type": "tool_use", "id": "fixture_read", "name": "read", "input": {"path": "fixture.pdf"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "fixture_read", "is_error": True,
            "content": [pdf, {"type": "text", "text": "Old client result."}]}, {"type": "text", "text":
            f"Use {search} with {query} artifact_lookup to discover artifact_lookup, then call artifact_lookup with key alpha. Request the tool; do not invent its result."}]}]
    extra = {"cache_control": {"type": "ephemeral"}, "context_management": {"edits": [
        {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}},
        {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 0},
            "keep": {"type": "tool_uses", "value": 0}, "exclude_tools": [search], "clear_tool_inputs": True}]}}
    options = dict(model=args.model, tools=tools, messages=history, thinking={"type": "adaptive"}, extra_headers=headers,
        extra_body=extra, tool_choice={"type": "tool", "name": search, "disable_parallel_tool_use": True})
    count = measured("prepared_count", lambda: api.count_tokens(**options))
    assert count.context_management.original_input_tokens > count.input_tokens
    if caps["prompt_cache"]:
        warm = measured("cache_warm", lambda: api.create(**options, max_tokens=0))
        cached = warm.usage.cache_read_input_tokens + warm.usage.cache_creation_input_tokens
        reused = measured("cache_reuse", lambda: api.create(**options, max_tokens=0))
        assert cached > 0 and reused.usage.cache_read_input_tokens == cached
        assert warm.usage.output_tokens == reused.usage.output_tokens == 0
        narrower = copy.deepcopy(options)
        narrower["extra_body"].pop("cache_control")
        narrower["messages"][0]["content"] = [{"type": "text", "text": "Earlier fixture request.",
            "cache_control": {"type": "ephemeral"}}]
        narrow_count = measured("earlier_marker_count", lambda: api.count_tokens(**narrower))
        assert narrow_count.input_tokens == count.input_tokens
        short = measured("earlier_marker_warm", lambda: api.create(**narrower, max_tokens=0))
        short_prefix = short.usage.cache_read_input_tokens + short.usage.cache_creation_input_tokens
        assert 0 < short_prefix < cached, "A moved cache marker reused tokens beyond its boundary"
        short_reuse = measured("earlier_marker_reuse", lambda: api.create(**narrower, max_tokens=0))
        assert short_reuse.usage.cache_read_input_tokens == short_prefix and short_reuse.usage.cache_creation_input_tokens == 0
    else:
        try:
            api.create(**options, max_tokens=0)
            raise AssertionError("Unsupported cache warming was accepted")
        except anthropic.BadRequestError:
            report["cases"].append({"name": "unsupported_warming", "status": 400})
    if args.cache_only:
        report["passed"] = True
        return
    starts = []

    def generate():
        with api.stream(**options, max_tokens=args.max_tokens) as stream:
            for event in stream:
                if event.type == "message_start":
                    starts.append(event.message)
            return stream.get_final_message()

    result = measured("deferred_search_stream", generate)
    assert len(starts) == 1
    usage = starts[0].usage
    assert usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens == count.input_tokens
    assert len(result.context_management.applied_edits) == 2
    assert result.stop_reason == "tool_use" and result.usage.output_tokens <= args.max_tokens
    call = result.content[-1]
    assert call.type == "tool_use" and call.name == "artifact_lookup" and call.input == {"key": "alpha"}
    if not caps["prompt_cache"]:
        assert result.usage.cache_read_input_tokens == result.usage.cache_creation_input_tokens == 0
    expected_input = count.input_tokens
    found = False
    for index, block in enumerate(result.content):
        if block.type != "tool_search_tool_result":
            continue
        found |= any(ref.tool_name == "artifact_lookup" for ref in (getattr(block.content, "tool_references", None) or []))
        prefix = history + [{"role": "assistant", "content": [b.model_dump() for b in result.content[:index + 1]]}]
        continued = measured("continuation_count", lambda: api.count_tokens(**dict(options, messages=prefix,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True})))
        expected_input += continued.input_tokens
    assert found, "No matching deferred tool was discovered"
    assert result.usage.input_tokens + result.usage.cache_read_input_tokens + result.usage.cache_creation_input_tokens == expected_input
    replay = history + [{"role": "assistant", "content": [b.model_dump() for b in result.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "is_error": True,
            "content": [pdf, {"type": "text", "text": "Client fixture error."}]}]}]
    measured("reasoning_pdf_error_replay_count", lambda: api.count_tokens(**dict(options, messages=replay)))
    if args.vision_images:
        vision()
    report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MERERUN_API_KEY")
    parser.add_argument("--search", choices=["regex", "bm25"], default="regex")
    parser.add_argument("--api-version", choices=["2023-06-01", "2026-10-21"], default="2026-10-21")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--vision-images", type=Path, nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--vision-labels", nargs=2, metavar=("FIRST_WORD", "SECOND_WORD"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--vision-only", action="store_true")
    mode.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if bool(args.vision_images) != bool(args.vision_labels) or args.vision_only and not args.vision_images:
        parser.error("Supply both --vision-images and --vision-labels for vision checks")
    if args.vision_labels and (not all(re.fullmatch(r"[A-Za-z]+", label) for label in args.vision_labels)
                              or args.vision_labels[0].upper() == args.vision_labels[1].upper()):
        parser.error("Vision labels must be distinct ASCII words")
    checks = "vision" if args.vision_only else "cache" if args.cache_only else "core+vision" if args.vision_images else "core"
    report = {"passed": False, "checks": checks, "cases": []}
    try:
        qualify(args, report)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
