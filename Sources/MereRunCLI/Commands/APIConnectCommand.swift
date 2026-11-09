import ArgumentParser
import Foundation
import MereRunCore

struct APIConnect: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "connect", abstract: "Preview or apply client connection settings.",
                                                   subcommands: [APIConnectClaudeCode.self])
}

struct APIConnectClaudeCode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "claude-code", abstract: "Connect Claude Code to a local Messages server.")
    @Option(help: "Server origin, without /v1.") var baseURL = "http://127.0.0.1:8080"
    @Option(help: "Local model ID, preserved in all client routes.") var model: String
    @Option(help: "Settings scope: project or user.") var scope: String = "project"
    @Option(help: "Environment variable containing the server credential.") var apiKeyEnv = "MERERUN_API_KEY"
    @Flag(help: "Atomically merge the preview into settings.") var write = false
    @Flag(help: "Emit machine-readable output.") var json = false

    mutating func run() async throws {
        let origin = try ClaudeCodeConnection.origin(baseURL)
        guard ["project", "user"].contains(scope) else { throw ValidationError("--scope must be project or user") }
        try ClaudeCodeConnection.validateEnvironmentName(apiKeyEnv)
        try ClaudeCodeConnection.validateRouting(environment: ProcessInfo.processInfo.environment, credentialVariable: apiKeyEnv)
        let root = scope == "user" ? FileManager.default.homeDirectoryForCurrentUser : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let settings = root.appendingPathComponent(scope == "user" ? ".claude/settings.json" : ".claude/settings.local.json")
        let credential = ProcessInfo.processInfo.environment[apiKeyEnv]
        let serverModel = try await ClaudeCodeConnection.inspect(origin: origin, model: model, credential: credential, credentialVariable: apiKeyEnv)
        let version = try ClaudeCodeConnection.installedVersion()
        let plan = try ClaudeCodeConnection.plan(origin: origin, model: serverModel, version: version, settings: settings,
                                               credentialVariable: apiKeyEnv, credentialPresent: credential?.isEmpty == false)
        if write { try ClaudeCodeConnection.apply(plan, settings: settings) }
        if json {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(plan), as: UTF8.self))
        } else {
            print("\(write ? "Applied" : "Preview for") \(settings.path)")
            print("Model: \(serverModel.id); context: \(plan.contextTokens); output: \(plan.outputTokens); Claude Code: \(version)")
            print("Messages: streaming=\(plan.capabilities.streaming), tools=\(plan.capabilities.tools), exact counting=\(plan.capabilities.token_counting).")
            print("Credential helper reads \(apiKeyEnv); no credential value is stored.")
            for issue in plan.qualification { print(issue) }
            if !write { print("Run again with --write to apply this project configuration.") }
        }
    }
}
