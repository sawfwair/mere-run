import Foundation
import MereRunContract

// MARK: - Operations templates

extension CommandCatalog {
    package static let operationsTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .adapterList,
            category: .operations,
            title: "Browse adapters",
            subtitle: "Verified LoRA catalog and install state",
            systemImage: "square.stack.3d.up"
        ),
        CommandTemplate(
            id: .adapterPull,
            category: .operations,
            title: "Pull adapter",
            subtitle: "Download and verify a cataloged LoRA",
            systemImage: "arrow.down.circle",
            promptLabel: "Adapter ID",
            defaultPrompt: "mere-platform-assistant"
        ),
        CommandTemplate(
            id: .runList,
            category: .operations,
            title: "Browse runs",
            subtitle: "Find local reports or remote jobs",
            systemImage: "clock.arrow.circlepath"
        ),
        CommandTemplate(
            id: .runInspect,
            category: .operations,
            title: "Inspect run",
            subtitle: "Read a durable run, report, plan, or remote job",
            systemImage: "doc.text.magnifyingglass"
        ),
        CommandTemplate(
            id: .runWatch,
            category: .operations,
            title: "Watch remote run",
            subtitle: "Stream SSH or Relay worker events",
            systemImage: "dot.radiowaves.left.and.right"
        ),
        CommandTemplate(
            id: .runFetch,
            category: .operations,
            title: "Fetch remote run",
            subtitle: "Verify and materialize remote artifacts locally",
            systemImage: "square.and.arrow.down",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .runCancel,
            category: .operations,
            title: "Cancel run",
            subtitle: "Request local or remote cancellation",
            systemImage: "stop.circle"
        ),
        CommandTemplate(
            id: .runRetry,
            category: .operations,
            title: "Retry Relay run",
            subtitle: "Retry the same immutable job bundle",
            systemImage: "arrow.clockwise.circle"
        ),
        CommandTemplate(
            id: .evaluationPackValidate,
            category: .operations,
            title: "Validate evaluation pack",
            subtitle: "Verify and content-hash an external pack",
            systemImage: "checkmark.seal",
            inputKind: .directory
        ),
        CommandTemplate(
            id: .evaluationRun,
            category: .operations,
            title: "Run evaluation pack",
            subtitle: "Plan or run matched model, prompt, and adapter arms",
            systemImage: "chart.bar.doc.horizontal",
            promptLabel: "Model bindings (one slot=id per line)",
            secondaryLabel: "Adapter bindings (one slot=reference per line)",
            inputKind: .directory,
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .evaluationPromote,
            category: .operations,
            title: "Promote evaluation report",
            subtitle: "Issue a receipt for a complete gate-passing report",
            systemImage: "checkmark.shield",
            inputKind: .file([.json]),
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .worldServe,
            category: .operations,
            title: "World session",
            subtitle: "Serve a warm DreamX or Cosmos3 world",
            systemImage: "globe.americas.fill",
            defaultModel: "video-dreamx-world-5b-ar-mlx"
        ),
        CommandTemplate(
            id: .statusSnapshot,
            category: .operations,
            title: "Status snapshot",
            subtitle: "Server, loaded models, and local inventory",
            systemImage: "waveform.path.ecg"
        ),
        CommandTemplate(
            id: .qualityGate,
            category: .operations,
            title: "Quality gate",
            subtitle: "Installed-model correctness and performance",
            systemImage: "checkmark.shield",
            outputKind: .file("json")
        ),
        CommandTemplate(
            id: .modelStorage,
            category: .operations,
            title: "Model storage",
            subtitle: "Physical storage, sharing, and reclaimable bytes",
            systemImage: "internaldrive"
        ),
        CommandTemplate(
            id: .modelGarbageCollect,
            category: .operations,
            title: "Storage cleanup",
            subtitle: "Dry-run or execute safe garbage collection",
            systemImage: "trash.slash"
        ),
        CommandTemplate(
            id: .modelRuntimeGet,
            category: .operations,
            title: "Read runtime policy",
            subtitle: "Inspect API residency and generation defaults",
            systemImage: "gearshape.2",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .modelRuntimeSet,
            category: .operations,
            title: "Set runtime policy",
            subtitle: "Pin, expire, alias, and tune resident models",
            systemImage: "slider.horizontal.3",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .graphStudio,
            category: .operations,
            title: "Open Graph Studio",
            subtitle: "Author and execute portable Graph v2 workflows",
            systemImage: "point.3.connected.trianglepath.dotted",
            externalURL: URL(string: "https://studio.mere.run/app")
        ),
        CommandTemplate(
            id: .nodeConsole,
            category: .operations,
            title: "Manage Nodes & Relay",
            subtitle: "Pair GPUs, schedule fleet work, and inspect nodes",
            systemImage: "server.rack",
            externalURL: URL(string: "https://relay.mere.run")
        )
    ]
}

// MARK: - Operations arguments

extension CommandArguments {
    package static func adapterList(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AdapterList
        var args = ArgumentBuilder(F.self)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func adapterPull(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AdapterPull
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if draft.force { args.flag(F.force) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func runList(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunList
        var args = ArgumentBuilder(F.self)
        if !draft.operationsRoot.isBlank { args.option(F.root, draft.operationsRoot) }
        if !draft.operationsExecutor.isBlank {
            args.option(F.executor, draft.operationsExecutor)
            args.option(F.limit, String(draft.operationsLimit))
        } else {
            args.option(F.maxDepth, String(draft.maxDepth))
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func runInspect(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunInspect
        var args = ArgumentBuilder(F.self)
        args.value(draft.operationsReference)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func runWatch(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunWatch
        var args = ArgumentBuilder(F.self)
        args.value(draft.operationsReference)
        args.option(F.pollInterval, format(draft.operationsPollInterval))
        if draft.operationsJSONStream { args.flag(F.jsonStream) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func runFetch(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunFetch
        var args = ArgumentBuilder(F.self)
        args.value(draft.operationsReference)
        args.option(F.into, draft.outputPath)
        if draft.operationsAllArtifacts { args.flag(F.allArtifacts) }
        args.repeated(F.artifact, lineList(draft.operationsArtifacts))
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func runCancel(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunCancel
        var args = ArgumentBuilder(F.self)
        args.value(draft.operationsReference)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func runRetry(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.RunRetry
        var args = ArgumentBuilder(F.self)
        args.value(draft.operationsReference)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func evaluationPackValidate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.EvalPackValidate
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func evaluationRun(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.EvalRun
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        args.repeated(F.model, lineList(draft.prompt))
        args.repeated(F.adapter, lineList(draft.secondaryText))
        if draft.dryRun { args.flag(F.dryRun) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func evaluationPromote(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.EvalPromote
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func worldServe(_ draft: CommandDraft, source: StudioScopeSource) -> [String] {
        typealias F = CommandFlags.WorldServe
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        args.option(F.backend, draft.operationsWorldBackend)
        let scope = source.scope(
            capability: MereRunCapabilityCatalog.worldServe,
            commandLine: [F.backend, draft.operationsWorldBackend, F.model, draft.model]
        )
        if scope.uses(F.baseModel) { args.option(F.baseModel, draft.operationsBaseModel) }
        args.option(F.model, draft.model)
        if !draft.operationsStateDirectory.isBlank {
            args.option(F.stateDirectory, draft.operationsStateDirectory)
        }
        if draft.operationsPrepare { args.flag(F.prepare) }
        return args.arguments
    }

    package static func statusSnapshot(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.Status
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        args.option(F.timeoutSeconds, format(draft.operationsTimeoutSeconds))
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func qualityGate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.Gate
        var args = ArgumentBuilder(F.self)
        args.option(F.suite, draft.operationsGateSuite)
        if draft.operationsUpdateBaselines { args.flag(F.updateBaselines) }
        if draft.operationsStrictPerformance { args.flag(F.strictPerf) }
        if !draft.outputPath.isBlank { args.option(F.jsonOutput, draft.outputPath) }
        if draft.operationsListOnly { args.flag(F.list) }
        return args.arguments
    }

    package static func modelStorage(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelStorage
        var args = ArgumentBuilder(F.self)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func modelGarbageCollect(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelGc
        var args = ArgumentBuilder(F.self)
        if draft.force { args.flag(F.force) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func modelRuntimeGet(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelRuntimeGet
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func modelRuntimeSet(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.ModelRuntimeSet
        var args = ArgumentBuilder(F.self)
        args.value(draft.model)
        if !draft.operationsRuntimeAlias.isBlank {
            args.option(F.alias, draft.operationsRuntimeAlias)
        }
        if draft.operationsClearAlias { args.flag(F.clearAlias) }
        if draft.operationsPinned { args.flag(F.pinned) }
        if draft.operationsUnpinned { args.flag(F.unpinned) }
        if !draft.operationsRuntimeTTL.isBlank {
            args.option(F.ttlSeconds, draft.operationsRuntimeTTL)
        }
        if draft.operationsClearTTL { args.flag(F.clearTTL) }
        if !draft.operationsRuntimeContext.isBlank {
            args.option(F.maxContextTokens, draft.operationsRuntimeContext)
        }
        if draft.operationsClearContext { args.flag(F.clearMaxContextTokens) }
        if !draft.operationsRuntimeMaxTokens.isBlank {
            args.option(F.maxTokens, draft.operationsRuntimeMaxTokens)
        }
        if draft.operationsClearMaxTokens { args.flag(F.clearMaxTokens) }
        if !draft.operationsRuntimeTemperature.isBlank {
            args.option(F.temperature, draft.operationsRuntimeTemperature)
        }
        if draft.operationsClearTemperature { args.flag(F.clearTemperature) }
        if !draft.operationsRuntimeTopP.isBlank {
            args.option(F.topP, draft.operationsRuntimeTopP)
        }
        if draft.operationsClearTopP { args.flag(F.clearTopP) }
        if !draft.operationsRuntimeMinP.isBlank {
            args.option(F.minP, draft.operationsRuntimeMinP)
        }
        if draft.operationsClearMinP { args.flag(F.clearMinP) }
        if !draft.operationsRuntimeEngine.isBlank {
            args.option(F.engine, draft.operationsRuntimeEngine)
        }
        if draft.operationsClearEngine { args.flag(F.clearEngine) }
        if !draft.operationsRuntimeKVCacheMode.isBlank {
            args.option(F.kvCacheMode, draft.operationsRuntimeKVCacheMode)
        }
        if draft.operationsClearKVCacheMode { args.flag(F.clearKVCacheMode) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    /// A row that opens another product in the browser. Studio launches its `externalURL`
    /// rather than running the CLI, so there is no command line to build.
    package static func externalLauncher() -> [String] {
        []
    }
}

// MARK: - Operations validation

extension CommandCatalog {
    /// The reason an operations template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func operationsValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .adapterPull:
            if draft.prompt.isBlank {
                return "Adapter id is required."
            }
        case .runList:
            if draft.operationsRoot.isBlank == draft.operationsExecutor.isBlank {
                return "Choose exactly one local root or remote executor."
            }
            if !draft.operationsExecutor.isBlank && !(1...500).contains(draft.operationsLimit) {
                return "Remote result limit must be between 1 and 500."
            }
            if draft.operationsExecutor.isBlank && draft.maxDepth < 0 {
                return "Scan depth must be zero or greater."
            }
        case .runInspect, .runWatch, .runCancel, .runRetry:
            if draft.operationsReference.isBlank {
                return "Run path or remote reference is required."
            }
            if id == .runWatch && draft.operationsPollInterval < 0.25 {
                return "Polling interval must be at least 0.25 seconds."
            }
            if id == .runWatch && draft.operationsJSONStream && draft.json {
                return "Choose streaming events or one final JSON object, not both."
            }
        case .runFetch:
            if draft.operationsReference.isBlank {
                return "Remote run reference is required."
            }
            if draft.outputPath.isBlank {
                return "Destination run directory is required."
            }
            if draft.operationsAllArtifacts && !CommandArguments.lineList(draft.operationsArtifacts).isEmpty {
                return "Choose all artifacts or named artifacts, not both."
            }
        case .evaluationPackValidate:
            if draft.inputPath.isBlank {
                return "Evaluation pack directory is required."
            }
        case .evaluationRun:
            if draft.inputPath.isBlank {
                return "Evaluation pack directory is required."
            }
            if CommandArguments.lineList(draft.prompt).isEmpty {
                return "At least one model binding is required."
            }
        case .evaluationPromote:
            if draft.inputPath.isBlank {
                return "Evaluation report is required."
            }
        case .worldServe:
            if draft.model.isBlank || draft.operationsBaseModel.isBlank {
                return "Base and world model ids are required."
            }
            if !(1...65_535).contains(draft.port) {
                return "Port must be between 1 and 65535."
            }
            if draft.host != "127.0.0.1" && draft.host != "localhost"
                && draft.host != "::1" && draft.apiKey.isBlank {
                return "An API key is required for non-loopback world servers."
            }
        case .statusSnapshot:
            if !(1...65_535).contains(draft.port) {
                return "Port must be between 1 and 65535."
            }
            if draft.operationsTimeoutSeconds <= 0 {
                return "Probe timeout must be greater than zero."
            }
        case .qualityGate:
            let validSuites = Set(["text", "speech", "vision", "image", "embed"])
            let selectedSuites = Set(
                draft.operationsGateSuite
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            )
            if draft.operationsGateSuite.lowercased() != "all"
                && (selectedSuites.isEmpty || !selectedSuites.isSubset(of: validSuites)) {
                return "Suites must be all or a comma-separated set of text, speech, vision, image, and embed."
            }
        case .modelRuntimeGet, .modelRuntimeSet:
            if draft.model.isBlank {
                return "Managed model id or alias is required."
            }
            if draft.operationsPinned && draft.operationsUnpinned {
                return "Choose pinned or unpinned, not both."
            }
            let conflictingRuntimeValues = [
                (!draft.operationsRuntimeAlias.isBlank, draft.operationsClearAlias, "alias"),
                (!draft.operationsRuntimeTTL.isBlank, draft.operationsClearTTL, "TTL"),
                (!draft.operationsRuntimeContext.isBlank, draft.operationsClearContext, "context"),
                (!draft.operationsRuntimeMaxTokens.isBlank, draft.operationsClearMaxTokens, "max tokens"),
                (!draft.operationsRuntimeTemperature.isBlank, draft.operationsClearTemperature, "temperature"),
                (!draft.operationsRuntimeTopP.isBlank, draft.operationsClearTopP, "top-p"),
                (!draft.operationsRuntimeMinP.isBlank, draft.operationsClearMinP, "min-p"),
                (!draft.operationsRuntimeEngine.isBlank, draft.operationsClearEngine, "engine"),
                (!draft.operationsRuntimeKVCacheMode.isBlank, draft.operationsClearKVCacheMode, "KV cache")
            ]
            if let conflict = conflictingRuntimeValues.first(where: { $0.0 && $0.1 }) {
                return "Set or clear \(conflict.2), not both."
            }
        default:
            break
        }
        return nil
    }
}
