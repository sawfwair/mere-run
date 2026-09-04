import Foundation

// MARK: - Models templates

extension CommandCatalog {
    static let modelsTemplates: [CommandTemplate] = [
        CommandTemplate(id: .modelList, category: .models, title: "List models", subtitle: "Installed and missing managed models", systemImage: "list.bullet.rectangle"),
        CommandTemplate(id: .modelCapabilities, category: .models, title: "Capabilities", subtitle: "Hardware support and recommended pulls", systemImage: "memorychip"),
        CommandTemplate(id: .modelPull, category: .models, title: "Pull model", subtitle: "Download a managed model", systemImage: "arrow.down.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelInfo, category: .models, title: "Model info", subtitle: "Manifest and validation report", systemImage: "info.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRemove, category: .models, title: "Remove model", subtitle: "Delete a local model install", systemImage: "trash", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRepairManifests, category: .models, title: "Repair manifests", subtitle: "Write missing known model manifests", systemImage: "wrench.and.screwdriver"),
        CommandTemplate(
            id: .modelOptimize,
            category: .models,
            title: "Optimize MiniMax-H3",
            subtitle: "Build or replace the inference-only AdaLN cache",
            systemImage: "bolt.badge.clock",
            defaultModel: "video-minimax-h3-fl2va-mlx"
        ),
        CommandTemplate(
            id: .modelBenchmark,
            category: .models,
            title: "Qwen3.6 benchmark",
            subtitle: "Focused Qwen3.6 MTP benchmark",
            systemImage: "speedometer",
            defaultModel: "text-chat-q36-nano"
        ),
        CommandTemplate(
            id: .modelBenchmarkLagunaDFlash,
            category: .models,
            title: "Laguna benchmark",
            subtitle: "Target-only, fixed DFlash, and adaptive routing",
            systemImage: "bolt.horizontal.circle"
        ),
        CommandTemplate(
            id: .modelLocationList,
            category: .models,
            title: "Model locations",
            subtitle: "Writable store, search roots, and explicit bindings",
            systemImage: "externaldrive.badge.checkmark"
        ),
        CommandTemplate(
            id: .modelLocationAdd,
            category: .models,
            title: "Add search root",
            subtitle: "Register a read-only root of canonical model directories",
            systemImage: "externaldrive.badge.plus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationRemove,
            category: .models,
            title: "Remove search root",
            subtitle: "Unregister a search root without deleting files",
            systemImage: "externaldrive.badge.minus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationBind,
            category: .models,
            title: "Bind model directory",
            subtitle: "Point a canonical model id at a read-only directory",
            systemImage: "link.badge.plus",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .modelLocationUnbind,
            category: .models,
            title: "Unbind model directory",
            subtitle: "Remove explicit bindings without deleting files",
            systemImage: "link.circle"
        ),
        CommandTemplate(
            id: .modelBenchmarkFused,
            category: .models,
            title: "Fused quality suite",
            subtitle: "Mere Lite or Mere Comprehensive versioned suite",
            systemImage: "chart.bar.doc.horizontal"
        ),
        CommandTemplate(
            id: .modelBenchmarkChat,
            category: .models,
            title: "Chat benchmark",
            subtitle: "Grounded-chat evaluation slice",
            systemImage: "bubble.left.and.text.bubble.right"
        ),
        CommandTemplate(
            id: .modelBenchmarkCode,
            category: .models,
            title: "Code benchmark",
            subtitle: "Real coding-evaluation slice with sandboxed execution",
            systemImage: "chevron.left.forwardslash.chevron.right"
        ),
        CommandTemplate(
            id: .modelBenchmarkVLM,
            category: .models,
            title: "Vision-language benchmark",
            subtitle: "Synthetic or lmms-eval multimodal datasets",
            systemImage: "photo.badge.checkmark",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .modelBenchmarkToolCalls,
            category: .models,
            title: "Tool-call benchmark",
            subtitle: "Tool selection accuracy across chat models",
            systemImage: "wrench.and.screwdriver"
        ),
        CommandTemplate(
            id: .modelBenchmarkToolContinuations,
            category: .models,
            title: "Tool continuation benchmark",
            subtitle: "Gemma 4 continuation after completed tool calls",
            systemImage: "arrow.turn.down.right"
        ),
        CommandTemplate(
            id: .modelBenchmarkGemma4KV,
            category: .models,
            title: "Gemma4 KV benchmark",
            subtitle: "Default KV cache decode against packed PolarKV",
            systemImage: "memorychip"
        ),
        CommandTemplate(
            id: .modelBenchmarkGemma4MTP,
            category: .models,
            title: "Gemma4 MTP benchmark",
            subtitle: "Serial decode against verified MTP speculative decode",
            systemImage: "bolt.badge.clock"
        ),
        CommandTemplate(
            id: .modelBenchmarkAPIWorkload,
            category: .models,
            title: "API workload benchmark",
            subtitle: "Replay a chat workload against a running API server",
            systemImage: "server.rack"
        ),
        CommandTemplate(
            id: .modelBenchmarkFusedFixture,
            category: .models,
            title: "Fused fixture hashes",
            subtitle: "Stamp or verify normalized fixture JSONL hashes",
            systemImage: "number.square",
            inputKind: .file([.json, .plainText])
        )
    ]
}

// MARK: - Models arguments

extension CommandArguments {
    static func modelList(_ draft: CommandDraft) -> [String] {
        CommandFlags.ModelList.command
    }

    static func modelCapabilities(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelCapabilities
        var args = ArgumentBuilder(F.self)
        if draft.all { args.flag(F.all) }
        if draft.force { args.flag(F.recommended) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelPull(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelPull
        var args = ArgumentBuilder(F.self)
        if draft.all {
            args.flag(F.all)
        } else {
            args.value(draft.model)
        }
        if draft.force { args.flag(F.force) }
        if draft.stream { args.flag(F.allowUnsupported) }
        if draft.quiet { args.flag(F.quiet) }
        if draft.acceptModelLicense { args.flag(F.acceptModelLicense) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelInfo(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelInfo
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if draft.all { args.flag(F.json) }
        if draft.force { args.flag(F.components) }
        return args.arguments
    }

    static func modelRemove(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelRemove
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if draft.force { args.flag(F.force) }
        if draft.modelKeepCache { args.flag(F.keepCache) }
        if draft.modelRemovalJSON { args.flag(F.json) }
        return args.arguments
    }

    static func modelRepairManifests(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelRepairManifests
        var args = ArgumentBuilder(F.self)
        if draft.force { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelOptimize(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelOptimize
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if draft.force { args.flag(F.force) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmark(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkQ36MTP
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if !draft.prompt.isBlank { args.option(F.prompt, draft.prompt) }
        if !draft.benchmarkPromptFile.isBlank {
            args.option(F.promptFile, draft.benchmarkPromptFile)
        }
        args.option(F.promptRepeat, String(draft.benchmarkPromptRepeat))
        args.option(F.decodeTokens, String(draft.benchmarkDecodeTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        args.option(F.contextSize, String(draft.contextSize))
        args.option(F.forcedMTPMinPromptTokens, String(draft.benchmarkForcedMTPMinPromptTokens))
        if !draft.benchmarkPromptRepeatValues.isBlank {
            args.option(F.promptRepeatValues, draft.benchmarkPromptRepeatValues)
        }
        if !draft.benchmarkDecodeTokenValues.isBlank {
            args.option(F.decodeTokenValues, draft.benchmarkDecodeTokenValues)
        }
        if !draft.benchmarkTemperatureValues.isBlank {
            args.option(F.temperatureValues, draft.benchmarkTemperatureValues)
        }
        if !draft.benchmarkMTPBlockSize.isBlank {
            args.option(F.mtpBlockSize, draft.benchmarkMTPBlockSize)
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkLagunaDFlash(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkLagunaDFlash
        var args = ArgumentBuilder(F.self)
        args.option(F.lagunaPath, draft.modelRoot)
        args.option(F.lagunaDFlashPath, draft.secondaryText)
        args.option(F.decodeTokenValues, draft.benchmarkDecodeTokenValues)
        args.option(F.repetitions, String(draft.benchmarkRepetitions))
        args.option(F.lagunaDFlashTokens, String(draft.benchmarkLagunaDFlashTokens))
        args.option(F.temperature, format(draft.temperature))
        args.option(F.topP, format(draft.topP))
        args.option(F.topK, String(draft.topK))
        args.option(F.minP, format(draft.minP))
        args.option(F.fixture, draft.benchmarkFixture)
        args.option(F.contextSize, String(draft.contextSize))
        args.option(F.warmupRepetitions, String(draft.benchmarkWarmupRepetitions))
        if !draft.prompt.isBlank { args.option(F.prompt, draft.prompt) }
        if !draft.benchmarkPromptFile.isBlank {
            args.option(F.promptFile, draft.benchmarkPromptFile)
        }
        if !draft.benchmarkConcurrencyValues.isBlank {
            args.option(F.concurrencyValues, draft.benchmarkConcurrencyValues)
        }
        if draft.benchmarkMixedFixtures { args.flag(F.mixedFixtures) }
        if draft.benchmarkIncludeAutomatic { args.flag(F.includeAutomatic) }
        if draft.benchmarkLogResponses { args.flag(F.logResponses) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelLocationList(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelLocationList
        var args = ArgumentBuilder(F.self)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelLocationAdd(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelLocationAdd
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        return args.arguments
    }

    static func modelLocationRemove(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelLocationRemove
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        return args.arguments
    }

    static func modelLocationBind(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelLocationBind
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        args.value(draft.inputPath)
        if draft.acceptModelLicense { args.flag(F.acceptModelLicense) }
        return args.arguments
    }

    static func modelLocationUnbind(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelLocationUnbind
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        return args.arguments
    }

    static func modelBenchmarkCode(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkCode
        var args = ArgumentBuilder(F.self)
        if !draft.benchmarkModels.isBlank { args.option(F.models, draft.benchmarkModels) }
        if !draft.benchmarkSuite.isBlank { args.option(F.suite, draft.benchmarkSuite) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if !draft.benchmarkSandbox.isBlank { args.option(F.sandbox, draft.benchmarkSandbox) }
        if draft.benchmarkAllowCodeExecution { args.flag(F.allowCodeExecution) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkFused(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkFused
        var args = ArgumentBuilder(F.self)
        if !draft.benchmarkSuite.isBlank { args.option(F.suite, draft.benchmarkSuite) }
        if !draft.benchmarkModels.isBlank { args.option(F.models, draft.benchmarkModels) }
        if !draft.benchmarkCases.isBlank { args.option(F.cases, draft.benchmarkCases) }
        if !draft.benchmarkTrials.isBlank { args.option(F.trials, draft.benchmarkTrials) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if !draft.benchmarkSandbox.isBlank { args.option(F.sandbox, draft.benchmarkSandbox) }
        if draft.benchmarkAllowCodeExecution { args.flag(F.allowCodeExecution) }
        if draft.benchmarkLogResponses { args.flag(F.logResponses) }
        if draft.benchmarkResume { args.flag(F.resume) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkFusedFixture(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkFusedFixture
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if draft.benchmarkFixtureCheck { args.flag(F.check) }
        return args.arguments
    }

    static func modelBenchmarkVLM(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkVLM
        var args = ArgumentBuilder(F.self)
        if !draft.benchmarkModels.isBlank { args.option(F.models, draft.benchmarkModels) }
        if !draft.benchmarkDataset.isBlank { args.option(F.dataset, draft.benchmarkDataset) }
        if !draft.outputPath.isBlank { args.option(F.outputDir, draft.outputPath) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkToolContinuations(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkToolContinuations
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.benchmarkLogResponses { args.flag(F.logResponses) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkAPIWorkload(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkAPIWorkload
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkChat(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkChat
        var args = ArgumentBuilder(F.self)
        if !draft.benchmarkModels.isBlank { args.option(F.models, draft.benchmarkModels) }
        if !draft.benchmarkSuite.isBlank { args.option(F.suite, draft.benchmarkSuite) }
        if !draft.benchmarkCases.isBlank { args.option(F.cases, draft.benchmarkCases) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.benchmarkLogResponses { args.flag(F.logResponses) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkToolCalls(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkToolCalls
        var args = ArgumentBuilder(F.self)
        if !draft.benchmarkModels.isBlank { args.option(F.models, draft.benchmarkModels) }
        if !draft.benchmarkCases.isBlank { args.option(F.cases, draft.benchmarkCases) }
        if draft.maxTokens > 0 { args.option(F.maxTokens, String(draft.maxTokens)) }
        if draft.contextSize > 0 { args.option(F.contextSize, String(draft.contextSize)) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.benchmarkLogResponses { args.flag(F.logResponses) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkGemma4KV(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkGemma4KV
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.benchmarkPromptFile.isBlank {
            args.option(F.promptFile, draft.benchmarkPromptFile)
        }
        if draft.benchmarkPromptRepeat > 0 {
            args.option(F.promptRepeat, String(draft.benchmarkPromptRepeat))
        }
        if draft.benchmarkDecodeTokens > 0 {
            args.option(F.decodeTokens, String(draft.benchmarkDecodeTokens))
        }
        if !draft.benchmarkDecodeTokenValues.isBlank {
            args.option(F.decodeTokenValues, draft.benchmarkDecodeTokenValues)
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func modelBenchmarkGemma4MTP(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelBenchmarkGemma4MTP
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.benchmarkPromptFile.isBlank {
            args.option(F.promptFile, draft.benchmarkPromptFile)
        }
        if draft.benchmarkPromptRepeat > 0 {
            args.option(F.promptRepeat, String(draft.benchmarkPromptRepeat))
        }
        if draft.benchmarkDecodeTokens > 0 {
            args.option(F.decodeTokens, String(draft.benchmarkDecodeTokens))
        }
        if !draft.benchmarkDecodeTokenValues.isBlank {
            args.option(F.decodeTokenValues, draft.benchmarkDecodeTokenValues)
        }
        if !draft.benchmarkMTPBlockSize.isBlank {
            args.option(F.mtpBlockSize, draft.benchmarkMTPBlockSize)
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
