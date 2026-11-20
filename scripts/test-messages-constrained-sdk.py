#!/usr/bin/env python3
"""Official SDK checks against the controlled MessagesHTTPTests runtime.

These are wire tests; they do not qualify checkpoint quality or throughput.
"""
import json
import sys

import anthropic

client = anthropic.Anthropic(base_url=sys.argv[1], api_key="test-key", max_retries=0)
for version in ("2023-06-01", "2026-10-21"):
    common = dict(model="fixture", max_tokens=16, messages=[{"role": "user", "content": "answer"}],
                  extra_headers={"anthropic-version": version})
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    output = dict(output_config={"format": {"type": "json_schema", "schema": schema}})
    for streaming in (False, True):
        if streaming:
            with client.messages.stream(**common, **output) as stream:
                streamed = "".join(stream.text_stream)
                result = stream.get_final_message()
                assert streamed == result.content[0].text
        else:
            result = client.messages.create(**common, **output)
        assert json.loads(result.content[0].text) == {"ok": True}
        assert result.stop_reason == "end_turn"
    tool = {"name": "edit", "strict": True, "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {"replace": {"type": "boolean"}}, "required": ["replace"]}}
    for streaming in (False, True):
        args = dict(common, tools=[tool], tool_choice={"type": "any", "disable_parallel_tool_use": True})
        if streaming:
            with client.messages.stream(**args) as stream:
                stream.until_done()
                result = stream.get_final_message()
        else:
            result = client.messages.create(**args)
        assert result.stop_reason == "tool_use"
        assert result.content[0].name == "edit"
        assert result.content[0].input == {"replace": True}
        assert result.content[0].id
    count = client.messages.count_tokens(model="fixture", messages=common["messages"],
                                        tools=[tool], extra_headers=common["extra_headers"])
    assert count.input_tokens == 7
print("Constrained Messages SDK fixtures passed for both API versions.")
