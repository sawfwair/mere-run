import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import {
  discoverModels,
  mapModel,
  mapThinkingLevels,
  selectPiModels,
  type MereRunModel,
} from "../extensions/mere-run.ts";

const deepSeek: MereRunModel = {
  id: "text-agent-deepseek-v4-flash",
  name: "DeepSeek V4 Flash",
  task: "chat.completions",
  reasoning: true,
  thinking_levels: ["off", "minimal", "low", "medium", "high", "xhigh"],
  tool_call: true,
  modalities: { input: ["text", "image", "audio"], output: ["text"] },
  limit: { context: 32_768, output: 32_768 },
  openai_compat: {
    supports_store: false,
    supports_developer_role: false,
    supports_reasoning_effort: true,
    supports_usage_in_streaming: true,
    supports_finish_reason: true,
    max_tokens_field: "max_tokens",
    supports_strict_mode: false,
    thinking_format: "deepseek",
    thinking_level_map: { minimal: "low" },
    requires_reasoning_content_on_assistant_messages: true,
  },
};

test("maps self-described mere.run capabilities into Pi model metadata", () => {
  const mapped = mapModel(deepSeek, "http://127.0.0.1:8080/v1");

  assert.equal(mapped.provider, "mere-run");
  assert.equal(mapped.contextWindow, 32_768);
  assert.equal(mapped.maxTokens, 32_768);
  assert.deepEqual(mapped.input, ["text", "image"]);
  assert.equal(mapped.compat?.thinkingFormat, "deepseek");
  assert.equal(mapped.compat?.maxTokensField, "max_tokens");
  assert.equal(mapped.thinkingLevelMap?.minimal, "low");
  assert.equal(mapped.thinkingLevelMap?.max, null);
  assert.equal(mapped.thinkingLevelMap?.off, undefined);
});

test("keeps only chat models that can execute Pi tools", () => {
  const selected = selectPiModels(
    [
      deepSeek,
      { id: "plain-chat", task: "chat.completions", tool_call: false },
      { id: "embedding", task: "embeddings", tool_call: false },
    ],
    "http://127.0.0.1:8080/v1",
  );

  assert.deepEqual(selected.map((model) => model.id), [deepSeek.id]);
});

test("discovers models from the live endpoint with bearer authentication", async () => {
  const server = createServer((request, response) => {
    assert.equal(request.url, "/v1/models");
    assert.equal(request.headers.authorization, "Bearer local-secret");
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ data: [deepSeek] }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));

  try {
    const address = server.address();
    assert(address && typeof address !== "string");
    const models = await discoverModels(
      `http://127.0.0.1:${address.port}/v1`,
      "local-secret",
      AbortSignal.timeout(1_000),
    );
    assert.deepEqual(models.map((model) => model.id), [deepSeek.id]);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
});

test("represents a fixed reasoning lane without inventing adjustable levels", () => {
  const mapped = mapThinkingLevels({
    id: "text-agent-ornith-9b",
    reasoning: true,
    thinking_levels: ["high"],
  });

  assert.equal(mapped?.high, undefined);
  assert.equal(mapped?.off, null);
  assert.equal(mapped?.low, null);
  assert.equal(mapped?.xhigh, null);
});
