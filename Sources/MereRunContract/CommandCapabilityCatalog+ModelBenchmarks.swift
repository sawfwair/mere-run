import Foundation

extension MereRunCapabilityCatalog {
    public static let modelBenchmarkQ36MTP = MereRunCommandCapability(
        id: "model.benchmark.q36-mtp",
        command: ["model", "benchmark", "q36-mtp"],
        title: "Qwen-family MTP benchmark",
        summary: "Run prompt, decode-length, and temperature matrices for Qwen-family MTP.",
        options: [
            .init(
                flag: "--repetitions", label: "Repetitions", kind: .integer, defaultValue: "1", group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--warmups", label: "Warmup requests", kind: .integer, defaultValue: "1", group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--warmup-tokens", label: "Warmup tokens", kind: .integer, defaultValue: "16", group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--variants", label: "Variants", kind: .string, defaultValue: "baseline,adaptive,forced",
                group: Group.run, tier: .expert
            ),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--temperature-values", label: "Temperature matrix", kind: .string),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--mtp-block-size", label: "MTP block", kind: .integer),
            .init(flag: "--forced-mtp-min-prompt-tokens", label: "Forced MTP threshold", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkLagunaDFlash = MereRunCommandCapability(
        id: "model.benchmark.laguna-dflash",
        command: ["model", "benchmark", "laguna-dflash"],
        title: "Laguna DFlash benchmark",
        summary: "Compare target-only, fixed DFlash, and adaptive Laguna decode in one resident process.",
        options: [
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory, required: true),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory, required: true),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--repetitions", label: "Repetitions", kind: .integer),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(
                flag: "--fixture",
                label: "Fixture",
                kind: .choice,
                choices: ["deterministic-prose", "grounded-email", "code-completion"]
            ),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--concurrency-values", label: "Concurrency matrix", kind: .string),
            .init(flag: "--warmup-repetitions", label: "Warmups", kind: .integer),
            .init(flag: "--mixed-fixtures", label: "Mixed fixtures", kind: .boolean),
            .init(flag: "--include-automatic", label: "Adaptive routing", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkParakeetCoreML = MereRunCommandCapability(
        id: "model.benchmark.parakeet-coreml",
        command: ["model", "benchmark", "parakeet-coreml"],
        title: "Parakeet Core ML benchmark",
        summary: "Measure the prepared Parakeet Core ML pipeline in one resident release process.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--artifact", label: "Core ML artifact", kind: .directory, required: true),
            .init(flag: "--warmups", label: "Warmups", kind: .integer),
            .init(flag: "--repetitions", label: "Repetitions", kind: .integer),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkChat = MereRunCommandCapability(
        id: "model.benchmark.chat",
        command: ["model", "benchmark", "chat"],
        title: "Chat benchmark",
        summary: "Run a small grounded-chat evaluation slice against local assistant models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["mere-chat-slice"]),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--concurrency", label: "Concurrency", kind: .integer),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--laguna-dflash-min-tokens", label: "Minimum speculative tokens", kind: .integer),
            .init(
                flag: "--laguna-dflash-routing",
                label: "DFlash routing",
                kind: .choice,
                choices: ["automatic", "target-only", "dflash"]
            ),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkCode = MereRunCommandCapability(
        id: "model.benchmark.code",
        command: ["model", "benchmark", "code"],
        title: "Code benchmark",
        summary: "Run a real coding-evaluation slice against local coding models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "Laguna DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Laguna DFlash tokens", kind: .integer),
            .init(
                flag: "--laguna-dflash-min-tokens",
                label: "Laguna DFlash minimum tokens",
                kind: .integer
            ),
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["humaneval-slice"]),
            .init(flag: "--tasks", label: "Tasks", kind: .string),
            .init(flag: "--humaneval-file", label: "HumanEval file", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--thinking", label: "Thinking", kind: .boolean),
            .init(flag: "--execution-timeout", label: "Execution timeout", kind: .number),
            .init(flag: "--python", label: "Python", kind: .string),
            .init(
                flag: "--sandbox",
                label: "Sandbox",
                kind: .choice,
                choices: ["auto", "macos-sandbox-exec", "bubblewrap", "none"]
            ),
            .init(flag: "--allow-code-execution", label: "Allow code execution", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkFused = MereRunCommandCapability(
        id: "model.benchmark.fused",
        command: ["model", "benchmark", "fused"],
        title: "Fused quality suite",
        summary: "Run the versioned Mere Lite or Mere Comprehensive fused quality suite.",
        options: [
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["lite", "comprehensive"]),
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--manifest", label: "Manifest", kind: .file),
            .init(flag: "--external-cases", label: "External cases", kind: .file),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--capabilities", label: "Capabilities", kind: .string),
            .init(flag: "--trials", label: "Trials", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(
                flag: "--logprobs",
                label: "Logprobs",
                kind: .choice,
                choices: ["summary", "tokens", "top"]
            ),
            .init(flag: "--top-logprobs", label: "Top logprobs", kind: .integer),
            .init(
                flag: "--performance-lane",
                label: "Performance lane",
                kind: .choice,
                choices: ["none", "native"]
            ),
            .init(flag: "--execution-timeout", label: "Execution timeout", kind: .number),
            .init(flag: "--python", label: "Python", kind: .string),
            .init(
                flag: "--sandbox",
                label: "Sandbox",
                kind: .choice,
                choices: ["auto", "macos-sandbox-exec", "bubblewrap", "none"]
            ),
            .init(flag: "--allow-code-execution", label: "Allow code execution", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--checkpoint", label: "Checkpoint", kind: .file),
            .init(flag: "--resume", label: "Resume", kind: .boolean),
            .init(flag: "--case-trial-limit", label: "Case trial limit", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkFusedFixture = MereRunCommandCapability(
        id: "model.benchmark.fused-fixture",
        command: ["model", "benchmark", "fused-fixture"],
        title: "Fused fixture hashes",
        summary: "Stamp or verify normalized fused-benchmark JSONL fixture hashes.",
        arguments: [
            .init(name: "input", label: "Fixture JSONL", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--check", label: "Verify only", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkVLM = MereRunCommandCapability(
        id: "model.benchmark.vlm",
        command: ["model", "benchmark", "vlm"],
        title: "Vision-language benchmark",
        summary: "Compare vision-language chat models on synthetic or lmms-eval datasets.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(
                flag: "--dataset",
                label: "Dataset",
                kind: .choice,
                choices: [
                    "synthetic-vqa-v1", "mathvista-testmini", "mmmu-val",
                    "chartqa", "docvqa-val", "mme"
                ]
            ),
            .init(flag: "--lmms-tasks", label: "lmms-eval tasks", kind: .string),
            .init(flag: "--fixture-dir", label: "Fixture directory", kind: .directory),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory),
            .init(flag: "--lmms-eval-root", label: "lmms-eval root", kind: .directory),
            .init(flag: "--lmms-eval-python", label: "lmms-eval Python", kind: .string),
            .init(flag: "--external-endpoint", label: "External endpoint", kind: .boolean),
            .init(flag: "--base-url", label: "Base URL", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--limit", label: "Limit", kind: .string),
            .init(flag: "--log-samples", label: "Log samples", kind: .boolean),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output-dir", optional: true)
    )

    public static let modelBenchmarkToolCalls = MereRunCommandCapability(
        id: "model.benchmark.tool-calls",
        command: ["model", "benchmark", "tool-calls"],
        title: "Tool-call benchmark",
        summary: "Run a small tool-call selection evaluation against local chat models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--laguna-dflash-min-tokens", label: "Minimum speculative tokens", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkToolContinuations = MereRunCommandCapability(
        id: "model.benchmark.tool-continuations",
        command: ["model", "benchmark", "tool-continuations"],
        title: "Tool continuation benchmark",
        summary: "Evaluate Gemma 4 continuation after completed tool calls.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkGemma4KV = MereRunCommandCapability(
        id: "model.benchmark.gemma4-kv",
        command: ["model", "benchmark", "gemma4-kv"],
        title: "Gemma4 KV benchmark",
        summary: "Compare Gemma4 default KV cache decode against packed PolarKV.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkGemma4MTP = MereRunCommandCapability(
        id: "model.benchmark.gemma4-mtp",
        command: ["model", "benchmark", "gemma4-mtp"],
        title: "Gemma4 MTP benchmark",
        summary: "Compare Gemma4 serial decode against verified MTP speculative decode.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--mtp-block-size", label: "MTP block size", kind: .integer),
            .init(flag: "--mtp-min-prompt-tokens", label: "MTP minimum prompt tokens", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkAPIWorkload = MereRunCommandCapability(
        id: "model.benchmark.api-workload",
        command: ["model", "benchmark", "api-workload"],
        title: "API workload benchmark",
        summary: "Replay a chat workload against a running API server and measure cache counters.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--workload-file", label: "Workload file", kind: .file),
            .init(flag: "--turns", label: "Turns", kind: .integer),
            .init(flag: "--shared-prefix-repeat", label: "Shared prefix repeat", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--concurrency", label: "Concurrency", kind: .integer),
            .init(flag: "--timeout-seconds", label: "Timeout", kind: .number),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    // MARK: - Plugins, configuration, and resident services
}
