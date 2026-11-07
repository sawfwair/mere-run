#!/usr/bin/env python3
"""Official SDK protocol checks against MessagesHTTPTests' controlled runtime.

Run through MERERUN_ANTHROPIC_SDK_PYTHON, never against a model qualification
server: fixture prompts deliberately select deterministic test outcomes.
"""
import sys
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
    catalog = [{"type": "tool_search_tool_regex_20251119", "name": "catalog_search"}, dict(tool, defer_loading=True)]
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
print("Official Anthropic SDK", anthropic.__version__, "buffered, streaming, tools, reasoning continuity and counting passed for both API versions")
