#!/usr/bin/env python3
"""Official SDK protocol checks against MessagesHTTPTests' controlled runtime.

Run through MERERUN_ANTHROPIC_SDK_PYTHON, never against a model qualification
server: fixture prompts deliberately select deterministic test outcomes.
"""
import sys
import copy
import anthropic

base = sys.argv[1]
client = anthropic.Anthropic(base_url=base, api_key="test-key", max_retries=0)
for version in ("2023-06-01", "2026-10-21"):
    headers = {"anthropic-version": version}
    kwargs = dict(model="fixture", max_tokens=16, messages=[{"role": "user", "content": "hello"}], extra_headers=headers)
    message = client.messages.create(**kwargs)
    assert message.type == "message" and message.content[0].text == "42"
    assert message.usage.input_tokens == 7 and message.stop_reason == "end_turn"
    with client.messages.stream(**kwargs) as stream:
        assert "".join(stream.text_stream) == "42"
        final = stream.get_final_message()
    assert final.content[0].text == "42" and final.id.startswith("msg_")
    assert final.usage.output_tokens == 2
    with client.messages.stream(model="fixture", max_tokens=16,
            messages=[{"role": "user", "content": "prefill_ping"}], extra_headers=headers) as stream:
        assert "".join(stream.text_stream) == "42"
        pinged = stream.get_final_message()
    assert pinged.usage.input_tokens == 7 and pinged.usage.output_tokens == 2
    count = client.messages.count_tokens(model="fixture", messages=kwargs["messages"], extra_headers=headers)
    assert count.input_tokens == 7

    with client.messages.stream(**kwargs, thinking={"type": "adaptive"}) as stream:
        stream.until_done()
        reasoned = stream.get_final_message()
    thinking = reasoned.content[0]
    assert thinking.type == "thinking" and thinking.thinking == "work 🧮"
    assert thinking.signature.startswith("mere.v1.")
    client.messages.create(model="fixture", max_tokens=16, thinking={"type": "adaptive"}, extra_headers=headers,
        messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": [b.model_dump() for b in reasoned.content]}, {"role": "user", "content": "continue"}])

    tool = {"name": "calculate", "input_schema": {"type": "object", "properties": {
        "value": {"type": "number"}, "nested": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}, "required": ["value", "nested"]}}
    args = dict(model="fixture", max_tokens=16, tools=[tool], messages=[{"role": "user", "content": "tool"}], extra_headers=headers)
    for streaming in (False, True):
        if streaming:
            with client.messages.stream(**args) as stream:
                stream.until_done()
                called = stream.get_final_message()
        else:
            called = client.messages.create(**args)
        call = called.content[0]
        assert call.type == "tool_use" and call.input == {"value": 42, "nested": {"ok": True}}
        assert called.stop_reason == "tool_use"
        client.messages.create(model="fixture", max_tokens=16, tools=[tool], extra_headers=headers, messages=[
            {"role": "user", "content": "tool"}, {"role": "assistant", "content": [b.model_dump() for b in called.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": "tool failed", "is_error": True}]}])
    catalog = [{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}, dict(tool, defer_loading=True)]
    for streaming in (False, True):
        search_args = dict(model="fixture", max_tokens=16, tools=catalog, messages=[{"role": "user", "content": "search"}], extra_headers=headers)
        if streaming:
            with client.messages.stream(**search_args) as stream:
                stream.until_done()
                discovered = stream.get_final_message()
        else:
            discovered = client.messages.create(**search_args)
        assert discovered.content[0].type == "server_tool_use"
        assert discovered.content[1].type == "tool_search_tool_result"
        assert discovered.content[1].content.tool_references[0].tool_name == "calculate"
        assert discovered.content[-1].text == "42" and discovered.usage.output_tokens == 4
        client.messages.create(model="fixture", max_tokens=16, tools=catalog, extra_headers=headers, messages=[
            {"role": "user", "content": "search"}, {"role": "assistant", "content": [b.model_dump() for b in discovered.content]},
            {"role": "user", "content": "continue"}])
    loaded = client.messages.create(model="fixture", max_tokens=16, tools=catalog,
        messages=[{"role": "user", "content": "search_call"}], extra_headers=headers)
    assert loaded.content[-1].type == "tool_use" and loaded.content[-1].name == "calculate"
    assert loaded.content[-1].input == {"value": 42, "nested": {"ok": True}}
    assert loaded.stop_reason == "tool_use" and loaded.usage.output_tokens == 5
    mixed = client.messages.create(model="fixture", max_tokens=16, tools=[catalog[0], tool],
        messages=[{"role": "user", "content": "mixed"}], extra_headers=headers)
    assert [block.type for block in mixed.content] == ["server_tool_use", "tool_search_tool_result", "tool_use"]
    assert mixed.content[-1].input == {"value": 42, "nested": {"ok": True}}

    # Combined preparation, deferred discovery, cache accounting, signed thinking,
    # client tool replay, and exact count/generation parity through real HTTP.
    advanced_catalog = [dict(catalog[0], cache_control={"type": "ephemeral", "ttl": "1h"}),
        {"name": "read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}, catalog[1]]
    history = [{"role": "user", "content": "advanced"},
        {"role": "assistant", "content": [thinking.model_dump(),
            {"type": "tool_use", "id": "old_read", "name": "read", "input": {"path": "old.txt"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old_read", "content": [
            {"type": "text", "text": "old result " * 50},
            {"type": "document", "title": "Old note", "source": {"type": "text", "media_type": "text/plain", "data": "old document " * 50}}],
            "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "continue"}]}]
    advanced_api = client.beta.messages
    extra = {"cache_control": {"type": "ephemeral"}, "context_management": {"edits": [
        {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}},
        {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 0},
         "keep": {"type": "tool_uses", "value": 0}, "exclude_tools": [catalog[0]["name"]], "clear_tool_inputs": True}]}}
    advanced_args = dict(model="fixture", max_tokens=16, tools=advanced_catalog, messages=history,
        thinking={"type": "adaptive"}, extra_headers=headers, extra_body=extra, betas=["context-management-2025-06-27"])
    count = advanced_api.count_tokens(model="fixture", tools=advanced_catalog, messages=history,
        thinking={"type": "adaptive"}, extra_headers=headers, extra_body=extra)
    assert count.model_dump()["context_management"]["original_input_tokens"] > count.input_tokens
    expected_types = ["thinking", "text", "server_tool_use", "tool_search_tool_result", "thinking", "text", "tool_use"]
    for streaming in (False, True):
        if streaming:
            starts = []
            with advanced_api.stream(**advanced_args) as stream:
                for event in stream:
                    if event.type == "message_start":
                        starts.append(event.message)
                combined = stream.get_final_message()
            assert len(starts) == 1 and starts[0].usage.input_tokens + starts[0].usage.cache_read_input_tokens == count.input_tokens
        else:
            combined = advanced_api.create(**advanced_args)
        assert [b.type for b in combined.content] == expected_types
        assert combined.content[2].id == combined.content[3].tool_use_id
        assert combined.usage.output_tokens == 5 and combined.usage.cache_read_input_tokens == 6
        assert combined.usage.input_tokens + combined.usage.cache_read_input_tokens > count.input_tokens
        edits = combined.model_dump()["context_management"]["applied_edits"]
        assert len(edits) == 2 and edits[0]["cleared_thinking_turns"] == 1 and edits[1]["cleared_tool_uses"] == 1
        assert all(b.signature.startswith("mere.v1.") for b in combined.content if b.type == "thinking")
        replay = history + [{"role": "assistant", "content": [b.model_dump() for b in combined.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": combined.content[-1].id,
                "is_error": True, "content": [{"type": "text", "text": "try again"},
                    {"type": "document", "source": {"type": "text", "data": "client tool failed"}}]}]}]
        replayed = advanced_api.count_tokens(model="fixture", tools=advanced_catalog, messages=replay,
            thinking={"type": "adaptive"}, extra_headers=headers, extra_body=extra)
        assert replayed.input_tokens > 0
        tampered = [dict(b.model_dump()) for b in combined.content]
        tampered[0]["thinking"] += "modified"
        try:
            advanced_api.count_tokens(model="fixture", tools=advanced_catalog, extra_headers=headers,
                messages=history + [{"role": "assistant", "content": tampered}, replay[-1]], extra_body=extra)
            raise AssertionError("Modified reasoning was accepted")
        except anthropic.BadRequestError:
            pass
    exhausted = advanced_api.create(**dict(advanced_args, max_tokens=2))
    assert exhausted.stop_reason == "max_tokens" and exhausted.usage.output_tokens == 2
    assert [b.type for b in exhausted.content] == expected_types[:4]
    invalid_history = copy.deepcopy(history)
    invalid_history[-1]["content"][-1]["text"] = "invalid search"
    for streaming in (False, True):
        invalid_args = dict(advanced_args, messages=invalid_history)
        if streaming:
            with advanced_api.stream(**invalid_args) as stream:
                stream.until_done()
                recovered = stream.get_final_message()
        else:
            recovered = advanced_api.create(**invalid_args)
        search_results = [b for b in recovered.content if b.type == "tool_search_tool_result"]
        assert len(search_results) == 2
        assert search_results[0].content.type == "tool_search_tool_result_error"
        assert search_results[0].content.error_code == "invalid_tool_input"
        assert search_results[1].content.tool_references[0].tool_name == "calculate"
        assert recovered.content[-1].type == "tool_use" and recovered.usage.output_tokens == 7
    invalid_old_thinking = copy.deepcopy(history)
    invalid_old_thinking[1]["content"][0]["thinking"] += "modified before clearing"
    try:
        advanced_api.count_tokens(model="fixture", tools=advanced_catalog, messages=invalid_old_thinking,
            extra_headers=headers, extra_body=extra)
        raise AssertionError("Invalid prior reasoning was ignored during clearing")
    except anthropic.BadRequestError:
        pass
print("Official Anthropic SDK", anthropic.__version__, "buffered, streaming, tools, reasoning continuity and counting passed for both API versions")
