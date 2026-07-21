import ArgumentParser
import Foundation
import MereRunCore

extension RuntimeServingEngine: ExpressibleByArgument {}
extension RuntimeKVCacheMode: ExpressibleByArgument {}

struct ModelRuntime: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "runtime",
        abstract: "Read and update per-model API runtime settings.",
        subcommands: [
            ModelRuntimeGet.self,
            ModelRuntimeSet.self,
        ]
    )
}

struct ModelRuntimeGet: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "get",
        abstract: "Print typed API runtime settings for a managed model."
    )

    @Argument(help: "Managed model id or configured alias.")
    var model: String

    @Flag(name: [.long], help: "Emit settings as JSON.")
    var json: Bool = false

    func run() throws {
        let store = RuntimeModelSettingsStore()
        let modelID = try RuntimeModelCLI.resolveModelID(model, store: store)
        let settings = try store.settings(for: modelID)

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(settings)
            guard let output = String(data: data, encoding: .utf8) else {
                throw ValidationError("Could not encode runtime settings as UTF-8.")
            }
            print(output)
            return
        }

        print("Runtime settings")
        print("  model: \(modelID)")
        print("  path: \(store.url.path)")
        print("  alias: \(settings.alias ?? "none")")
        print("  pinned: \(settings.pinned)")
        print("  ttlSeconds: \(settings.ttlSeconds.map(String.init) ?? "none")")
        print("  maxContextTokens: \(settings.maxContextTokens.map(String.init) ?? "none")")
        print("  maxTokens: \(settings.maxTokens.map(String.init) ?? "none")")
        print("  temperature: \(settings.temperature.map { String($0) } ?? "none")")
        print("  topP: \(settings.topP.map { String($0) } ?? "none")")
        print("  engineOverride: \(settings.engineOverride?.rawValue ?? "none")")
        print("  kvCacheMode: \(settings.kvCacheMode?.rawValue ?? "none")")
    }
}

struct ModelRuntimeSet: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "set",
        abstract: "Update typed API runtime settings for a managed model."
    )

    @Argument(help: "Managed model id or configured alias.")
    var model: String

    @Option(name: [.long], help: "Set a request-facing alias for this model.")
    var alias: String?

    @Flag(name: [.long], help: "Clear this model's alias.")
    var clearAlias: Bool = false

    @Flag(name: [.long], help: "Keep this model loaded across automatic TTL/LRU eviction.")
    var pinned: Bool = false

    @Flag(name: [.long], help: "Clear the pinned flag.")
    var unpinned: Bool = false

    @Option(name: [.long], help: "Set unload TTL in seconds.")
    var ttlSeconds: Int?

    @Flag(name: [.long], help: "Clear unload TTL.")
    var clearTTL: Bool = false

    @Option(name: [.long], help: "Set the default context token limit.")
    var maxContextTokens: Int?

    @Flag(name: [.long], help: "Clear the default context token limit.")
    var clearMaxContextTokens: Bool = false

    @Option(name: [.long], help: "Set the default max output tokens.")
    var maxTokens: Int?

    @Flag(name: [.long], help: "Clear the default max output tokens.")
    var clearMaxTokens: Bool = false

    @Option(name: [.long], help: "Set the default temperature.")
    var temperature: Double?

    @Flag(name: [.long], help: "Clear the default temperature.")
    var clearTemperature: Bool = false

    @Option(name: [.long], help: "Set the default top_p.")
    var topP: Double?

    @Flag(name: [.long], help: "Clear the default top_p.")
    var clearTopP: Bool = false

    @Option(name: [.long], help: "Set an engine override after catalog compatibility checks.")
    var engine: RuntimeServingEngine?

    @Flag(name: [.long], help: "Clear the engine override.")
    var clearEngine: Bool = false

    @Option(
        name: [.long],
        help: "Set runtime KV cache mode: default; affine4/affine8 (affine8 for Gemma4/Qwen/LFM2); or Gemma4-only polar2/auto."
    )
    var kvCacheMode: RuntimeKVCacheMode?

    @Flag(name: [.long], help: "Clear the runtime KV cache mode.")
    var clearKVCacheMode: Bool = false

    @Flag(name: [.long], help: "Emit updated settings as JSON.")
    var json: Bool = false

    func run() throws {
        try validateFlagPairs()
        let store = RuntimeModelSettingsStore()
        let modelID = try RuntimeModelCLI.resolveModelID(model, store: store)
        var settings = try store.settings(for: modelID)

        if clearAlias {
            settings.alias = nil
        } else if let alias {
            settings.alias = alias
        }
        if pinned { settings.pinned = true }
        if unpinned { settings.pinned = false }
        if clearTTL { settings.ttlSeconds = nil } else if let ttlSeconds { settings.ttlSeconds = ttlSeconds }
        if clearMaxContextTokens {
            settings.maxContextTokens = nil
        } else if let maxContextTokens {
            settings.maxContextTokens = maxContextTokens
        }
        if clearMaxTokens { settings.maxTokens = nil } else if let maxTokens { settings.maxTokens = maxTokens }
        if clearTemperature {
            settings.temperature = nil
        } else if let temperature {
            settings.temperature = temperature
        }
        if clearTopP { settings.topP = nil } else if let topP { settings.topP = topP }
        if clearEngine { settings.engineOverride = nil } else if let engine { settings.engineOverride = engine }
        if clearKVCacheMode {
            settings.kvCacheMode = nil
        } else if let kvCacheMode {
            settings.kvCacheMode = kvCacheMode
        }

        try store.writeSettings(settings, for: modelID)
        let updated = try store.settings(for: modelID)
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(updated)
            guard let output = String(data: data, encoding: .utf8) else {
                throw ValidationError("Could not encode runtime settings as UTF-8.")
            }
            print(output)
        } else {
            print("Updated runtime settings for \(modelID)")
            print("  \(store.url.path)")
        }
    }

    private func validateFlagPairs() throws {
        if pinned && unpinned {
            throw ValidationError("Use either --pinned or --unpinned, not both.")
        }
        if alias != nil && clearAlias {
            throw ValidationError("Use either --alias or --clear-alias, not both.")
        }
        if ttlSeconds != nil && clearTTL {
            throw ValidationError("Use either --ttl-seconds or --clear-ttl, not both.")
        }
        if maxContextTokens != nil && clearMaxContextTokens {
            throw ValidationError("Use either --max-context-tokens or --clear-max-context-tokens, not both.")
        }
        if maxTokens != nil && clearMaxTokens {
            throw ValidationError("Use either --max-tokens or --clear-max-tokens, not both.")
        }
        if temperature != nil && clearTemperature {
            throw ValidationError("Use either --temperature or --clear-temperature, not both.")
        }
        if topP != nil && clearTopP {
            throw ValidationError("Use either --top-p or --clear-top-p, not both.")
        }
        if engine != nil && clearEngine {
            throw ValidationError("Use either --engine or --clear-engine, not both.")
        }
        if kvCacheMode != nil && clearKVCacheMode {
            throw ValidationError("Use either --kv-cache-mode or --clear-kv-cache-mode, not both.")
        }
    }
}

private enum RuntimeModelCLI {
    static func resolveModelID(_ value: String, store: RuntimeModelSettingsStore) throws -> String {
        if let spec = ManagedModelCatalog.spec(for: value) {
            guard spec.supportsRuntimeResidencySettings else {
                throw ValidationError("Model '\(spec.id)' does not have configurable API residency.")
            }
            return spec.id
        }
        let document = try store.load()
        if let match = document.models.first(where: { _, settings in
            settings.alias?.caseInsensitiveCompare(value) == .orderedSame
        }) {
            return match.key
        }
        throw ValidationError("Unknown runtime model or alias: \(value)")
    }
}
