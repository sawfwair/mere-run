import Foundation

// MARK: - Server templates

extension CommandCatalog {
    static let serverTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .pluginList,
            category: .server,
            title: "Plugins",
            subtitle: "List official companion plugins",
            systemImage: "puzzlepiece.extension"
        ),
        CommandTemplate(
            id: .pluginInstall,
            category: .server,
            title: "Install plugin",
            subtitle: "Install an official plugin by id",
            systemImage: "square.and.arrow.down",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .pluginDoctor,
            category: .server,
            title: "Plugin doctor",
            subtitle: "Run an installed plugin's doctor",
            systemImage: "stethoscope",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .openWebui,
            category: .server,
            title: "Open WebUI",
            subtitle: "Start the Open WebUI companion",
            systemImage: "globe",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .apiServe,
            category: .server,
            title: "API server",
            subtitle: "OpenAI-compatible local server",
            systemImage: "network",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .pluginInfo,
            category: .server,
            title: "Plugin details",
            subtitle: "Catalog entry and install command for one plugin",
            systemImage: "info.circle",
            promptLabel: "Plugin id"
        ),
        CommandTemplate(
            id: .pluginRun,
            category: .server,
            title: "Run plugin",
            subtitle: "Run an installed plugin without changing PATH",
            systemImage: "play.rectangle",
            promptLabel: "Plugin entrypoint"
        ),
        CommandTemplate(
            id: .pluginRollback,
            category: .server,
            title: "Roll back plugin",
            subtitle: "Restore a retained signed plugin bundle",
            systemImage: "arrow.uturn.backward.circle",
            promptLabel: "Plugin id"
        ),
        CommandTemplate(
            id: .visionServe,
            category: .server,
            title: "Vision grounding server",
            subtitle: "Resident binary-frame grounding over HTTP",
            systemImage: "viewfinder.circle"
        )
    ]
}

// MARK: - Server arguments

extension CommandArguments {
    static func pluginList(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginList
        var args = ArgumentBuilder(F.self)
        if !draft.pluginCatalogURL.isBlank {
            args.option(F.catalogURL, draft.pluginCatalogURL)
        }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func pluginInstall(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginInstall
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.pluginCatalogURL.isBlank {
            args.option(F.catalogURL, draft.pluginCatalogURL)
        }
        if !draft.pluginChannel.isBlank { args.option(F.channel, draft.pluginChannel) }
        if draft.all { args.flag(F.yes) }
        if draft.force { args.flag(F.force) }
        return args.arguments
    }

    static func pluginDoctor(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginDoctor
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.pluginCatalogURL.isBlank {
            args.option(F.catalogURL, draft.pluginCatalogURL)
        }
        return args.arguments
    }

    static func openWebui(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.OpenWebUIQuickstart
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.engine.isBlank { args.option(F.engine, draft.engine) }
        args.option(F.webuiHost, draft.openWebUIHost)
        args.option(F.webuiPort, String(draft.openWebUIPort))
        args.option(F.containerName, draft.openWebUIContainerName)
        args.option(F.volumeName, draft.openWebUIVolumeName)
        args.option(F.image, draft.openWebUIImage)
        if !draft.model.isBlank { args.option(F.textModel, draft.model) }
        args.option(F.visionModel, draft.openWebUIVisionModel)
        args.option(F.embeddingModel, draft.openWebUIEmbeddingModel)
        args.option(F.imageModel, draft.openWebUIImageModel)
        args.option(F.ttsModel, draft.openWebUITTSModel)
        args.option(F.sttModel, draft.openWebUISTTModel)
        args.option(F.ttsFormat, draft.openWebUITTSFormat)
        args.option(F.adminEmail, draft.openWebUIAdminEmail)
        args.option(F.waitSeconds, String(draft.openWebUIWaitSeconds))
        if draft.openWebUIPull { args.flag(F.pull) }
        if draft.acceptModelLicense { args.flag(F.acceptModelLicense) }
        if draft.openWebUISkipServer { args.flag(F.skipServer) }
        if draft.openWebUISkipDocker { args.flag(F.skipDocker) }
        if draft.openWebUISkipConfigure { args.flag(F.skipConfigure) }
        if draft.openWebUIReset { args.flag(F.reset) }
        if draft.dryRun { args.flag(F.dryRun) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func apiServe(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.APIServe
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        args.option(F.engine, draft.engine)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.apiLoRA.isBlank { args.option(F.lora, draft.apiLoRA) }
        args.option(F.rateLimitPerMinute, String(draft.apiRateLimitPerMinute))
        args.option(F.maxActiveRequests, String(draft.apiMaxActiveRequests))
        args.option(F.memoryGuard, draft.apiMemoryGuard)
        args.option(F.contextSize, String(draft.contextSize))
        if !draft.apiMemoryGuardCustomCeilingGB.isBlank {
            args.option(F.memoryGuardCustomCeilingGb, draft.apiMemoryGuardCustomCeilingGB)
        }
        if draft.kvBits > 0 { args.option(F.kvBits, String(draft.kvBits)) }
        if !draft.kvQuantScheme.isBlank {
            args.option(F.kvQuantScheme, draft.kvQuantScheme)
        }
        if draft.kvGroupSize > 0 {
            args.option(F.kvGroupSize, String(draft.kvGroupSize))
        }
        if draft.quantizedKVStart > 0 {
            args.option(F.quantizedKVStart, String(draft.quantizedKVStart))
        }
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func pluginInfo(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginInfo
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.pluginCatalogURL.isBlank {
            args.option(F.catalogURL, draft.pluginCatalogURL)
        }
        if !draft.pluginChannel.isBlank { args.option(F.channel, draft.pluginChannel) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    static func pluginRun(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginRun
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        return args.arguments
    }

    static func pluginRollback(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.PluginRollback
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if draft.all { args.flag(F.yes) }
        return args.arguments
    }

    static func visionServe(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VisionServe
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.visionServeMaxFrameBytes > 0 {
            args.option(F.maxFrameBytes, String(draft.visionServeMaxFrameBytes))
        }
        if draft.visionServeMaxBatchSize > 0 {
            args.option(F.maxBatchSize, String(draft.visionServeMaxBatchSize))
        }
        if draft.visionServeMaxBatchBytes > 0 {
            args.option(F.maxBatchBytes, String(draft.visionServeMaxBatchBytes))
        }
        if draft.preflight { args.flag(F.preflight) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
