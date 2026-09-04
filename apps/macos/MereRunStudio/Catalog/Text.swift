import Foundation

// MARK: - Text templates

extension CommandCatalog {
    static let textTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .textChat,
            category: .text,
            title: "Chat",
            subtitle: "Local chat models",
            systemImage: "bubble.left.and.bubble.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Summarize diffusion models in one paragraph.",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textCode,
            category: .text,
            title: "Code",
            subtitle: "Local code generation",
            systemImage: "chevron.left.forwardslash.chevron.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Write a tiny Swift function that formats byte counts.",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textEmbed,
            category: .text,
            title: "Embeddings",
            subtitle: "Generate JSON embedding vectors",
            systemImage: "point.3.connected.trianglepath.dotted",
            promptLabel: "Text",
            outputKind: .file("json"),
            defaultPrompt: "semantic search query",
            defaultModel: "text-embed-qwen3-0.6b"
        ),
        CommandTemplate(
            id: .textAnonymize,
            category: .text,
            title: "Anonymize",
            subtitle: "Detect and redact PII",
            systemImage: "eye.slash",
            promptLabel: "Text",
            outputKind: .file("txt"),
            defaultPrompt: "My name is Alice Smith and my email is alice@example.com",
            defaultModel: "text-anonymize-privacy-filter"
        ),
        CommandTemplate(
            id: .textTrainLoRA,
            category: .text,
            title: "Train text LoRA",
            subtitle: "Fine-tune from chat SFT JSONL",
            systemImage: "text.badge.plus",
            inputKind: .file([.json, .plainText]),
            outputKind: .file("safetensors"),
            defaultModel: "text-chat-gemma4-12b-4bit"
        )
    ]
}

// MARK: - Text arguments

extension CommandArguments {
    static func textChat(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.TextChat
        var args = ArgumentBuilder(F.self)
        args.option(F.prompt, draft.prompt)
        if !draft.secondaryText.isBlank { args.option(F.system, draft.secondaryText) }
        if !draft.imagePath.isBlank { args.option(F.image, draft.imagePath) }
        args.option(F.maxTokens, String(draft.maxTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if draft.topK > 0 { args.option(F.topK, String(draft.topK)) }
        if draft.minP > 0 { args.option(F.minP, format(draft.minP)) }
        if draft.kvBits > 0 { args.option(F.kvBits, String(draft.kvBits)) }
        if !draft.kvQuantScheme.isBlank { args.option(F.kvQuantScheme, draft.kvQuantScheme) }
        if draft.kvGroupSize > 0 { args.option(F.kvGroupSize, String(draft.kvGroupSize)) }
        if draft.quantizedKVStart > 0 {
            args.option(F.quantizedKVStart, String(draft.quantizedKVStart))
        }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.optionUnlessDefault(F.responseFormat, draft.responseFormat.rawValue)
        if !draft.loraPath.isBlank {
            args.option(F.lora, draft.loraPath)
            args.option(F.loraScale, format(draft.loraScale))
        }
        // Automatic passes neither half of the pair and leaves the choice to the model.
        let showsThinking: Bool? = draft.thinkingMode == .automatic ? nil : draft.thinkingMode == .show
        args.pair(F.thinking, F.noThinking, showsThinking)
        if let reasoningEffort = draft.reasoningEffort {
            args.option(F.reasoningEffort, format(reasoningEffort))
        }
        if !draft.tools.isBlank { args.option(F.tools, draft.tools) }
        if draft.toolLoop { args.flag(F.toolLoop) }
        if draft.allowShellExec { args.flag(F.allowShellExec) }
        if draft.allowAbsoluteToolPaths { args.flag(F.allowAbsoluteToolPaths) }
        if draft.autoApproveTools { args.flag(F.autoApproveTools) }
        if !draft.sandboxDir.isBlank { args.option(F.sandboxDir, draft.sandboxDir) }
        if draft.stream { args.flag(F.stream) }
        if draft.force { args.flag(F.stats) }
        if draft.quiet { args.flag(F.quiet) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        if draft.requireInstalled { args.flag(F.requireInstalled) }
        return args.arguments
    }

    static func textCode(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.TextCode
        var args = ArgumentBuilder(F.self)
        args.option(F.prompt, draft.prompt)
        if !draft.secondaryText.isBlank { args.option(F.system, draft.secondaryText) }
        args.option(F.maxTokens, String(draft.maxTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        if draft.minP > 0 { args.option(F.minP, format(draft.minP)) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.stream { args.flag(F.stream) }
        if draft.force { args.flag(F.stats) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func textEmbed(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.TextEmbed
        let embeddingTexts = draft.prompt
            .components(separatedBy: .newlines)
            .filter { !$0.isBlank }
        var args = ArgumentBuilder(F.self)
        args.values((embeddingTexts.isEmpty ? [draft.prompt] : embeddingTexts))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.force { args.flag(F.pretty) }
        return args.arguments
    }

    static func textAnonymize(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.TextAnonymize
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.replacement != "[{label}]" {
            args.option(F.replacement, draft.replacement)
        }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.all { args.flag(F.json) }
        if draft.force { args.flag(F.pretty) }
        return args.arguments
    }

    static func textTrainLoRA(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.TextTrainLoRA
        var args = ArgumentBuilder(F.self)
        args.option(F.data, draft.inputPath)
        args.option(F.output, draft.outputPath)
        args.option(F.model, draft.model)
        args.option(F.adapterName, draft.adapterName)
        args.option(F.trainingSteps, String(draft.steps))
        args.option(F.batchSize, String(draft.batchSize))
        args.option(F.learningRate, format(draft.learningRate))
        args.option(F.rank, String(draft.rank))
        args.option(F.maxSequenceLength, String(draft.maxSequenceLength))
        args.option(F.seed, draft.seed)
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.evalPath.isBlank { args.option(F.eval, draft.evalPath) }
        if draft.alpha > 0 { args.option(F.alpha, format(draft.alpha)) }
        if let reasoningEffort = draft.reasoningEffort {
            args.option(F.reasoningEffort, format(reasoningEffort))
        }
        if !draft.targetModules.isBlank {
            args.option(F.targetModules, draft.targetModules)
        }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.visualize {
            args.flag(F.visualize)
            args.option(F.visualizePort, String(draft.visualizePort))
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
