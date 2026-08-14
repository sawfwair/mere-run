import { createProvider, openAICompletionsApi, type Model } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export interface MereRunModel {
  id: string;
  name?: string;
  task?: string;
  reasoning?: boolean;
  thinking_levels?: ThinkingLevel[];
  tool_call?: boolean;
  modalities?: { input: string[]; output: string[] };
  limit?: { context: number; output: number };
  openai_compat?: {
    supports_store: boolean;
    supports_developer_role: boolean;
    supports_reasoning_effort: boolean;
    supports_usage_in_streaming: boolean;
    supports_finish_reason: boolean;
    max_tokens_field: "max_tokens" | "max_completion_tokens";
    supports_strict_mode: boolean;
    thinking_format?: "deepseek";
    thinking_level_map?: Partial<Record<ThinkingLevel, string>>;
    requires_reasoning_content_on_assistant_messages: boolean;
  };
}

const thinkingLevels: ThinkingLevel[] = [
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

export function mapThinkingLevels(model: MereRunModel) {
  if (!model.reasoning || !model.thinking_levels?.length) return undefined;
  const supported = new Set(model.thinking_levels);
  const overrides = model.openai_compat?.thinking_level_map ?? {};
  const result: Partial<Record<ThinkingLevel, string | null>> = {};
  for (const level of thinkingLevels) {
    if (!supported.has(level)) result[level] = null;
    else if (overrides[level]) result[level] = overrides[level];
  }
  return result;
}

export function mapModel(model: MereRunModel, baseUrl: string): Model<"openai-completions"> {
  const compat = model.openai_compat;
  return {
    id: model.id,
    name: model.name ?? model.id,
    api: "openai-completions",
    provider: "mere-run",
    baseUrl,
    reasoning: model.reasoning ?? false,
    thinkingLevelMap: mapThinkingLevels(model),
    input: (model.modalities?.input ?? ["text"]).filter(
      (value): value is "text" | "image" => value === "text" || value === "image",
    ),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: model.limit?.context ?? 32_768,
    maxTokens: model.limit?.output ?? 4_096,
    compat: compat
      ? {
          supportsStore: compat.supports_store,
          supportsDeveloperRole: compat.supports_developer_role,
          supportsReasoningEffort: compat.supports_reasoning_effort,
          supportsUsageInStreaming: compat.supports_usage_in_streaming,
          supportsFinishReason: compat.supports_finish_reason,
          maxTokensField: compat.max_tokens_field,
          supportsStrictMode: compat.supports_strict_mode,
          thinkingFormat: compat.thinking_format,
          requiresReasoningContentOnAssistantMessages:
            compat.requires_reasoning_content_on_assistant_messages,
        }
      : undefined,
  };
}

export function selectPiModels(models: MereRunModel[], baseUrl: string) {
  return models
    .filter((model) => model.task === "chat.completions" && model.tool_call === true)
    .map((model) => mapModel(model, baseUrl));
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  signal: AbortSignal,
): Promise<Model<"openai-completions">[]> {
  const response = await fetch(`${baseUrl}/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    signal,
  });
  if (!response.ok) {
    throw new Error(`mere.run model discovery failed: HTTP ${response.status}`);
  }
  const payload = (await response.json()) as { data?: MereRunModel[] };
  return selectPiModels(payload.data ?? [], baseUrl);
}

export default async function mereRunExtension(pi: ExtensionAPI) {
  const baseUrl = process.env.MERERUN_BASE_URL ?? "http://127.0.0.1:8080/v1";
  const fallbackAPIKey = process.env.MERERUN_API_KEY ?? "mere-run";
  const initialModels = await discoverModels(
    baseUrl,
    fallbackAPIKey,
    AbortSignal.timeout(2_000),
  ).catch(() => []);

  pi.registerProvider(
    createProvider({
      id: "mere-run",
      name: "mere.run Local",
      baseUrl,
      auth: {
        apiKey: {
          name: "mere.run local API key",
          async resolve({ ctx, credential }) {
            const key = credential?.key ?? (await ctx.env("MERERUN_API_KEY")) ?? fallbackAPIKey;
            return { auth: { apiKey: key }, source: "mere.run local" };
          },
        },
      },
      models: initialModels,
      async fetchModels(context) {
        const key =
          context.credential?.type === "api_key"
            ? context.credential.key ?? fallbackAPIKey
            : fallbackAPIKey;
        return discoverModels(baseUrl, key, context.signal);
      },
      api: openAICompletionsApi(),
    }),
  );
}
