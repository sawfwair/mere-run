import Foundation

extension MereRunCapabilityCatalog {
    public static let textChat = MereRunCommandCapability(
        id: "text.chat",
        command: ["text", "chat"],
        title: "Chat",
        summary: "Run local chat, vision, JSON, LoRA, reasoning, and tool workflows.",
        options: [
            .init(
                flag: "--audio", label: "Audio", kind: .file, group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--video", label: "Video", kind: .file, group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--show-unmasking", label: "Show canvas drafts", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--image", label: "Image", kind: .file, group: Group.inputs, tier: .standard),
            .init(flag: "--system", label: "System prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard,
                range: .init(min: 1, max: 131_072, step: 1)
            ),
            .init(
                flag: "--context-size", label: "Context size", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 512, max: 1_048_576, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--top-k", label: "Top-k", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--min-p", label: "Min-p", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--kv-bits", label: "KV bits", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 2, max: 8, step: 1)
            ),
            .init(
                flag: "--kv-quant-scheme",
                label: "KV quantization",
                kind: .choice,
                choices: ["uniform", "polar", "turboquant"],
                group: Group.run, tier: .expert, dependsOn: "--kv-bits"
            ),
            .init(
                flag: "--kv-group-size", label: "KV group size", kind: .integer,
                group: Group.run, tier: .expert, dependsOn: "--kv-bits"
            ),
            .init(
                flag: "--quantized-kv-start", label: "Quantized KV start", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 0, step: 1), dependsOn: "--kv-bits"
            ),
            .init(flag: "--model-root", label: "Model root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--response-format",
                label: "Response format",
                kind: .choice,
                choices: TextResponseFormat.allCases.map(\.rawValue),
                defaultValue: TextResponseFormat.text.rawValue, group: Group.output, tier: .standard
            ),
            .init(flag: "--lora", label: "LoRA", kind: .file, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ),
            .init(flag: "--thinking", label: "Show thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(flag: "--no-thinking", label: "Disable thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(
                flag: "--reasoning-effort", label: "Inkling reasoning effort", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
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
            .init(flag: "--tools", label: "Tools", kind: .string, group: Group.run, tier: .expert),
            .init(flag: "--tool-loop", label: "Tool loop", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--tools"),
            .init(
                flag: "--sandbox-dir", label: "Sandbox directory", kind: .directory,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--allow-shell-exec", label: "Allow shell", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--allow-absolute-tool-paths", label: "Allow absolute paths", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--auto-approve-tools", label: "Auto-approve tools", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--require-installed", label: "Require installed", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text)
    )

    public static let textCode = MereRunCommandCapability(
        id: "text.code",
        command: ["text", "code"],
        title: "Code",
        summary: "Run local code generation with GGUF models through llama.cpp.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(
                flag: "--system", label: "System prompt", kind: .string,
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
            .init(flag: "--model", label: "Model", kind: .file, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--stats", label: "Stats", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .standard)
        ],
        output: .init(kind: .text)
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
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
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
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--replacement", label: "Replacement template", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--output", label: "Output", kind: .file)
        ],
        output: .init(kind: .text, flag: "--output", optional: true)
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
            .init(flag: "--data", label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--model", label: "Base model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--eval", label: "Eval prompts", kind: .file),
            .init(flag: "--adapter-name", label: "Adapter name", kind: .string),
            .init(flag: "--training-steps", label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-sequence-length", label: "Sequence length", kind: .integer),
            .init(flag: "--reasoning-effort", label: "Inkling reasoning effort", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--target-modules", label: "Target modules", kind: .string),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )
}
