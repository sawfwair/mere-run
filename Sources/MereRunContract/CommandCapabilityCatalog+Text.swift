import Foundation

extension MereRunCapabilityCatalog {
    public static let textChat = MereRunCommandCapability(
        id: "text.chat",
        command: ["text", "chat"],
        title: "Chat",
        summary: "Run local chat, vision, JSON, LoRA, reasoning, and tool workflows.",
        options: [
            .init(
                flag: "--audio", label: "Audio", kind: .file, group: Group.inputs, tier: .expert,
                blankReadsAsOmitted: true
            ).scoped(Chat.only(.nemotronOmni, ignoredBy: Chat.allCases.filter { ![.nemotronOmni, .diffusionGemma].contains($0) })),
            .init(
                flag: "--video", label: "Video", kind: .file, group: Group.inputs, tier: .expert,
                blankReadsAsOmitted: true
            ).scoped(Chat.only(.nemotronOmni, ignoredBy: Chat.allCases.filter { ![.nemotronOmni, .diffusionGemma].contains($0) })),
            .init(
                flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .expert
            ).scoped(Chat.only(.diffusionGemma, .q35, .q35VL, .q38)),
            .init(
                flag: "--show-unmasking", label: "Show canvas drafts", kind: .boolean, group: Group.run, tier: .expert
            ).scoped(Chat.only(.diffusionGemma)),
            .init(flag: "--prompt", aliases: ["-p"], label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--image", label: "Image", kind: .file, group: Group.inputs, tier: .standard, blankReadsAsOmitted: true)
                .scoped(Chat.only(
                    .gemma4Unified, .museGlimmer, .nemotronOmni, .lfm2VL, .q35VL, .q38,
                    ignoredBy: [.laguna, .nemotronH, .gguf, .psi]
                )),
            .init(flag: "--system", aliases: ["-s"], label: "System prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard,
                range: .init(min: 1, max: 131_072, step: 1)
            ),
            // LFM2.5 caps the context at 32768 tokens whatever is asked for, and takes any smaller one.
            .init(
                flag: "--context-size", label: "Context size", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 512, max: 1_048_576, step: 1)
            ).scoped(
                Chat.rule(.lfm2, range: .init(min: 1, max: 32_768, step: 1), severity: .warning),
                .rule(.lfm2A1B, range: .init(min: 1, max: 32_768, step: 1), severity: .warning),
                .rule(.lfm2VL, range: .init(min: 1, max: 32_768, step: 1), severity: .warning)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ).scoped(Chat.except(ignoredBy: [.diffusionGemma])),
            .init(
                flag: "--top-k", label: "Top-k", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ).scoped(Chat.except(ignoredBy: [.gemma4, .gemma4Unified, .diffusionGemma, .gguf, .psi])),
            .init(
                flag: "--min-p", label: "Min-p", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ).scoped(Chat.except(ignoredBy: [.diffusionGemma])),
            // Gemma 4 takes fractional TurboQuant widths; the affine runtimes take 4 or 8. The Qwen
            // family refuses other widths, while Inkling and LFM2.5 run with a full-precision cache.
            .init(
                flag: "--kv-bits", label: "KV bits", kind: .number,
                group: Group.run, tier: .expert, range: .init(min: 2, max: 8, step: 0.5)
            ).scoped(
                Chat.only(.gemma4, .gemma4Unified, .inkling, .lfm2, .lfm2A1B, .lfm2VL, .q35, .q35VL, .q38, ignoredBy: kvCacheIgnoring),
                .rule(.inkling, values: ["4", "8"], severity: .warning),
                .rule(.lfm2, values: ["4", "8"], severity: .warning),
                .rule(.lfm2A1B, values: ["4", "8"], severity: .warning),
                .rule(.lfm2VL, values: ["4", "8"], severity: .warning),
                .rule(.q35, values: ["4", "8"]), .rule(.q35VL, values: ["4", "8"]), .rule(.q38, values: ["4", "8"])
            ),
            // Gemma 4 Turbo quantizes its KV cache by default, so the scheme, group size, and start
            // apply there without --kv-bits. Inkling and LFM2.5 only ever use the affine scheme.
            .init(
                flag: "--kv-quant-scheme",
                label: "KV quantization",
                kind: .choice,
                choices: ["uniform", "polar", "turboquant"],
                group: Group.run, tier: .expert, choiceSpellings: .caseInsensitive
            ).scoped(
                Chat.only(.gemma4, .gemma4Unified, .q35, .q35VL, .q38, ignoredBy: kvCacheIgnoring + [.inkling, .lfm2, .lfm2A1B, .lfm2VL]),
                .rule(.q35, values: ["uniform"]), .rule(.q35VL, values: ["uniform"]), .rule(.q38, values: ["uniform"])
            ),
            // The affine runtimes choose their own group size and start; Inkling and LFM2.5 never
            // read them, and the Qwen family refuses them.
            .init(
                flag: "--kv-group-size", label: "KV group size", kind: .integer,
                group: Group.run, tier: .expert
            ).scoped(Chat.only(.gemma4, .gemma4Unified, ignoredBy: kvCacheIgnoring + [.inkling, .lfm2, .lfm2A1B, .lfm2VL])),
            .init(
                flag: "--quantized-kv-start", label: "Quantized KV start", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 0, step: 1)
            ).scoped(Chat.only(.gemma4, .gemma4Unified, ignoredBy: kvCacheIgnoring + [.inkling, .lfm2, .lfm2A1B, .lfm2VL])),
            .init(flag: "--model-root", aliases: ["-m"], label: "Model root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            // Constrained JSON decoding runs on Gemma 4 and the Qwen-family runtimes only.
            .init(
                flag: "--response-format",
                label: "Response format",
                kind: .choice,
                choices: TextResponseFormat.allCases.map(\.rawValue),
                defaultValue: TextResponseFormat.text.rawValue, group: Group.output, tier: .standard
            ).scoped(
                Chat.rule(.diffusionGemma, values: ["text"]), .rule(.laguna, values: ["text"]),
                .rule(.inkling, values: ["text"]), .rule(.museGlimmer, values: ["text"]),
                .rule(.nemotronH, values: ["text"]), .rule(.nemotronOmni, values: ["text"]),
                .rule(.lfm2, values: ["text"]), .rule(.lfm2A1B, values: ["text"]), .rule(.lfm2VL, values: ["text"]),
                .rule(.gguf, values: ["text"]), .rule(.psi, values: ["text"])
            ),
            .init(
                flag: "--lora", label: "LoRA", kind: .file, group: Group.modelAndAdapters, tier: .standard,
                blankReadsAsOmitted: true
            ).scoped(Chat.only(
                    // LFM2.5 loads a text adapter only on the 8-bit A1B runtime and fails after
                    // loading any other checkpoint.
                    .gemma4, .gemma4Unified, .laguna, .inkling, .lfm2A1B,
                    ignoredBy: [.q35, .q35VL, .q38, .gguf, .psi]
                )),
            // Without --lora the scale does nothing anywhere, so it never fails on its own.
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ).scoped(Chat.only(
                .gemma4, .gemma4Unified, .laguna, .inkling, .lfm2A1B,
                ignoredBy: [
                    .diffusionGemma, .museGlimmer, .nemotronH, .nemotronOmni, .lfm2, .lfm2VL, .q35, .q35VL, .q38, .gguf, .psi
                ]
            )),
            .init(flag: "--thinking", aliases: ["--show-thinking"], label: "Show thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(flag: "--no-thinking", aliases: ["--no-show-thinking"], label: "Disable thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(
                flag: "--reasoning-effort", label: "Reasoning effort", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ).scoped(
                Chat.only(.inkling, .museGlimmer, .q38),
                .rule(.inkling, range: .init(min: 0, max: 0.99, step: 0.01)),
                .rule(.museGlimmer, range: .init(min: 0, max: 1, step: 0.01)),
                .rule(.q38, range: .init(min: 0, max: 1, step: 0.01))
            ),
            .init(flag: "--stats", label: "Stats", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .standard),
            .init(
                flag: "--markdown",
                label: "Terminal Markdown",
                kind: .choice,
                choices: ["auto", "always", "never"],
                defaultValue: "auto",
                group: Group.output,
                tier: .standard,
                dependsOn: "--stream"
            ),
            // The GGUF and Psi runtimes never see tool definitions, so no tool call comes back.
            .init(flag: "--tools", label: "Tools", kind: .string, group: Group.run, tier: .expert)
                .scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(flag: "--tool-loop", label: "Tool loop", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--tools")
                .scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(
                flag: "--sandbox-dir", label: "Sandbox directory", kind: .directory,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ).scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(
                flag: "--allow-shell-exec", label: "Allow shell", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ).scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(
                flag: "--allow-absolute-tool-paths", label: "Allow absolute paths", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ).scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(
                flag: "--auto-approve-tools", label: "Auto-approve tools", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ).scoped(Chat.except(ignoredBy: [.gguf, .psi])),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--require-installed", label: "Require installed", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text),
        routing: textChatRouting
    )

    private typealias Chat = TextChatFamily
    private typealias Training = TextTrainLoRAFamily

    /// Families whose runtime keeps its own KV cache whatever the KV options say.
    private static let kvCacheIgnoring: [Chat] = [
        .diffusionGemma, .laguna, .museGlimmer, .nemotronH, .nemotronOmni, .gguf, .psi
    ]

    public static let textCode = MereRunCommandCapability(
        id: "text.code",
        command: ["text", "code"],
        title: "Code",
        summary: "Run local code generation with GGUF models through llama.cpp.",
        options: [
            .init(flag: "--prompt", aliases: ["-p"], label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(
                flag: "--system", aliases: ["-s"], label: "System prompt", kind: .string,
                defaultValue: "You are a helpful coding assistant.", group: Group.prompt, tier: .standard
            ),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard,
                range: .init(min: 1, max: 131_072, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.95", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--min-p", label: "Min-p", kind: .number,
                defaultValue: "0.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .file, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--stats", label: "Stats", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .standard)
        ],
        output: .init(kind: .text),
        routing: textCodeRouting
    )

    public static let textEmbed = MereRunCommandCapability(
        id: "text.embed",
        command: ["text", "embed"],
        title: "Embeddings",
        summary: "Generate native Qwen3 text embeddings.",
        arguments: [
            .init(name: "texts", label: "Texts", kind: .string, required: true, repeatable: true)
        ],
        options: [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true),
        routing: textEmbedRouting
    )

    public static let textAnonymize = MereRunCommandCapability(
        id: "text.anonymize",
        command: ["text", "anonymize"],
        title: "Anonymize",
        summary: "Detect and redact PII with the native OpenAI Privacy Filter.",
        arguments: [
            .init(name: "texts", label: "Texts", kind: .string, required: false, repeatable: true)
        ],
        options: [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--replacement", label: "Replacement template", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file)
        ],
        output: .init(kind: .text, flag: "--output", optional: true),
        routing: textAnonymizeRouting
    )

    public static let textDecide = MereRunCommandCapability(
        id: "text.decide", command: ["text", "decide"], title: "Decisions",
        summary: "Evaluate choice, score, and boolean questions with native Laya.",
        options: [
            .init(flag: "--input", aliases: ["-i"], label: "JSON request", kind: .file),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, defaultValue: "text-decide-laya"),
            .init(flag: "--output", aliases: ["-o"], label: "JSON output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--preflight", label: "Inspect token budgets", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true),
        routing: textDecideRouting
    )

    public static let textClassify = MereRunCommandCapability(
        id: "text.classify", command: ["text", "classify"], title: "Classify",
        summary: "Classify text with caller supplied labels using native GLiNER2.5 Decide.",
        options: [
            .init(flag: "--input", label: "JSON request", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string,
                  defaultValue: "text-classify-gliner25-decide"),
            .init(flag: "--output", label: "JSON output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--preflight", label: "Inspect token usage", kind: .boolean),
            .init(flag: "--long", label: "Process long documents", kind: .boolean),
            .init(flag: "--batch", label: "Process a request array", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let textExtract = MereRunCommandCapability(
        id: "text.extract", command: ["text", "extract"], title: "Extract",
        summary: "Extract entities, relations, and structures with native GLiNER2.5 Decide.",
        options: [
            .init(flag: "--input", label: "JSON request", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string,
                  defaultValue: "text-classify-gliner25-decide"),
            .init(flag: "--output", label: "JSON output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--preflight", label: "Inspect token usage", kind: .boolean),
            .init(flag: "--long", label: "Process long documents", kind: .boolean),
            .init(flag: "--batch", label: "Process a request array", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let textTrainLoRA = MereRunCommandCapability(
        id: "text.train-lora",
        command: ["text", "train-lora"],
        title: "Train text LoRA",
        summary: "Train a native text LoRA from OpenAI-style chat SFT JSONL.",
        options: [
            .init(
                flag: "--resume-from", label: "Resume checkpoint", kind: .file, group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--resume-step", label: "Resume step", kind: .integer, group: Group.run, tier: .expert
            ),
            .init(flag: "--data", aliases: ["-d"], label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, required: true),
            .init(flag: "--model", aliases: ["-m"], label: "Base model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--eval", label: "Eval prompts", kind: .file),
            .init(flag: "--adapter-name", label: "Adapter name", kind: .string),
            .init(flag: "--training-steps", aliases: ["--steps"], label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer)
                .scoped(Training.rule(.gemma4VLM, values: ["1"])),
            .init(flag: "--learning-rate", aliases: ["--lr"], label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-sequence-length", label: "Sequence length", kind: .integer),
            // Only Inkling's chat renderer takes an effort; the other trainers render without one.
            .init(
                flag: "--reasoning-effort", label: "Inkling reasoning effort", kind: .number,
                range: .init(min: 0, max: 0.99, step: 0.01)
            ).scoped(Training.only(.inkling, ignoredBy: [.gemma4, .gemma4VLM, .lagunaXS, .lfm2A1B])),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--target-modules", label: "Target modules", kind: .string)
                .scoped(
                    Training.rule(.gemma4, defaultValue: attentionTargets),
                    .rule(.gemma4VLM, defaultValue: attentionTargets),
                    .rule(.lagunaXS, defaultValue: attentionTargets),
                    .rule(.inkling, defaultValue: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"),
                    .rule(.lfm2A1B, defaultValue: "q_proj,k_proj,v_proj,out_proj")
                ),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output"),
        routing: textTrainLoRARouting
    )

    /// The attention projections Gemma 4 and Laguna train by default.
    private static let attentionTargets = "q_proj,k_proj,v_proj,o_proj"
}
