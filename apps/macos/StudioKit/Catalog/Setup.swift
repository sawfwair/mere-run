import Foundation

// MARK: - Setup templates

extension CommandCatalog {
    package static let setupTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .setup,
            category: .setup,
            title: "Setup path",
            subtitle: "Plan or run guided, BYOA, or manual setup",
            systemImage: "wand.and.stars"
        ),
        CommandTemplate(
            id: .agentOnboard,
            category: .setup,
            title: "Agent onboarding",
            subtitle: "Check local readiness and prepare Pi integration",
            systemImage: "person.crop.circle.badge.gearshape",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .agentStatus,
            category: .setup,
            title: "Agent status",
            subtitle: "Inspect Pi, provider, and local agent readiness",
            systemImage: "person.crop.circle.badge.checkmark"
        ),
        CommandTemplate(
            id: .agentInstallPi,
            category: .setup,
            title: "Install Pi",
            subtitle: "Install or replace the optional setup agent",
            systemImage: "square.and.arrow.down"
        ),
        CommandTemplate(
            id: .agentStart,
            category: .setup,
            title: "Start agent",
            subtitle: "Launch a guided setup session",
            systemImage: "terminal.fill",
            promptLabel: "Prompt",
            defaultPrompt: "Guide me through setting up mere.run on this machine. Start by summarizing what this Mac can run, then help me install only supported models."
        )
    ]
}

// MARK: - Setup arguments

extension CommandArguments {
    package static func setup(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.Setup
        var args = ArgumentBuilder(F.self)
        args.option(F.mode, draft.setupMode)
        args.option(F.agentModel, draft.agentModel)
        if draft.force { args.flag(F.install) }
        if draft.stream { args.flag(F.start) }
        if draft.dryRun { args.flag(F.dryRun) }
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.piPath.isBlank { args.option(F.piPath, draft.piPath) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func agentOnboard(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AgentOnboard
        var args = ArgumentBuilder(F.self)
        if draft.force { args.flag(F.pullRecommended) }
        if draft.acceptModelLicense { args.flag(F.acceptModelLicense) }
        if draft.all { args.flag(F.installPi) }
        if draft.stream { args.flag(F.configurePi) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func agentStatus(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AgentStatus
        var args = ArgumentBuilder(F.self)
        if !draft.piPath.isBlank { args.option(F.piPath, draft.piPath) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func agentInstallPi(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AgentInstallPi
        var args = ArgumentBuilder(F.self)
        if draft.force { args.flag(F.force) }
        return args.arguments
    }

    package static func agentStart(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AgentStart
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.piPath.isBlank { args.option(F.piPath, draft.piPath) }
        if !draft.prompt.isBlank { args.option(F.prompt, draft.prompt) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.stream { args.flag(F.skipServer) }
        if draft.force { args.flag(F.allowUnsupported) }
        if draft.noBootstrap { args.flag(F.noBootstrap) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }
}
