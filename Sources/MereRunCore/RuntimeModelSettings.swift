import Foundation

public enum RuntimeServingEngine: String, Codable, CaseIterable, Hashable, Sendable {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatGemma4 = "text-chat-gemma4"
    case textChatQ35 = "text-chat-q35"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"
}

public enum RuntimeKVCacheMode: String, Codable, CaseIterable, Hashable, Sendable {
    case `default`
    case polar2
    case auto

    public static let gemma4AutoPolarPromptTokenThreshold = 1_024

    public func gemma4Quantization(
        fallback: Gemma4KVCacheQuantization,
        promptTokenCount: Int
    ) -> Gemma4KVCacheQuantization {
        switch self {
        case .default:
            return fallback
        case .polar2:
            return Self.gemma4Polar2Quantization(fallback: fallback)
        case .auto:
            guard promptTokenCount >= Self.gemma4AutoPolarPromptTokenThreshold else {
                return fallback
            }
            return Self.gemma4Polar2Quantization(fallback: fallback)
        }
    }

    private static func gemma4Polar2Quantization(
        fallback: Gemma4KVCacheQuantization
    ) -> Gemma4KVCacheQuantization {
        Gemma4KVCacheQuantization(
            bits: 2,
            scheme: .polar,
            groupSize: fallback.groupSize,
            quantizedStart: 0
        )
    }
}

public struct RuntimeModelSettings: Codable, Hashable, Sendable {
    public var alias: String?
    public var pinned: Bool
    public var ttlSeconds: Int?
    public var maxContextTokens: Int?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var engineOverride: RuntimeServingEngine?
    public var kvCacheMode: RuntimeKVCacheMode?

    public init(
        alias: String? = nil,
        pinned: Bool = false,
        ttlSeconds: Int? = nil,
        maxContextTokens: Int? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        engineOverride: RuntimeServingEngine? = nil,
        kvCacheMode: RuntimeKVCacheMode? = nil
    ) {
        self.alias = alias
        self.pinned = pinned
        self.ttlSeconds = ttlSeconds
        self.maxContextTokens = maxContextTokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.engineOverride = engineOverride
        self.kvCacheMode = kvCacheMode
    }

    public var normalized: RuntimeModelSettings {
        var copy = self
        copy.alias = normalizedOptionalString(alias)
        return copy
    }

    public func validated(for spec: ManagedModelSpec) throws -> RuntimeModelSettings {
        let settings = normalized
        if let alias = settings.alias {
            guard alias.rangeOfCharacter(from: .whitespacesAndNewlines) == nil else {
                throw RuntimeModelSettingsError.invalidAlias("aliases cannot contain whitespace")
            }
            guard !alias.contains("/") else {
                throw RuntimeModelSettingsError.invalidAlias("aliases cannot contain '/'")
            }
        }
        if let ttlSeconds = settings.ttlSeconds, ttlSeconds <= 0 {
            throw RuntimeModelSettingsError.invalidValue("ttlSeconds must be greater than zero")
        }
        if let maxContextTokens = settings.maxContextTokens, maxContextTokens <= 0 {
            throw RuntimeModelSettingsError.invalidValue("maxContextTokens must be greater than zero")
        }
        if let maxTokens = settings.maxTokens, maxTokens <= 0 {
            throw RuntimeModelSettingsError.invalidValue("maxTokens must be greater than zero")
        }
        if let temperature = settings.temperature, !(0...2).contains(temperature) || !temperature.isFinite {
            throw RuntimeModelSettingsError.invalidValue("temperature must be between 0 and 2")
        }
        if let topP = settings.topP, !(0...1).contains(topP) || !topP.isFinite {
            throw RuntimeModelSettingsError.invalidValue("topP must be between 0 and 1")
        }
        if let override = settings.engineOverride,
           override != spec.defaultRuntimeServingEngine {
            throw RuntimeModelSettingsError.incompatibleEngine(
                modelID: spec.id,
                requested: override,
                expected: spec.defaultRuntimeServingEngine
            )
        }
        if let kvCacheMode = settings.kvCacheMode,
           kvCacheMode != .default,
           spec.defaultRuntimeServingEngine != .textChatGemma4 {
            throw RuntimeModelSettingsError.incompatibleKVCacheMode(
                modelID: spec.id,
                requested: kvCacheMode,
                expectedEngine: .textChatGemma4
            )
        }
        guard spec.isAPIServableRuntimeModel else {
            throw RuntimeModelSettingsError.unsupportedModel(spec.id)
        }
        return settings
    }

    private func normalizedOptionalString(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}

public struct RuntimeModelSettingsDocument: Codable, Equatable, Sendable {
    public var version: Int
    public var models: [String: RuntimeModelSettings]

    public init(version: Int = 1, models: [String: RuntimeModelSettings] = [:]) {
        self.version = version
        self.models = models
    }
}

public enum RuntimeModelSettingsError: LocalizedError, Equatable {
    case unknownModel(String)
    case unsupportedModel(String)
    case invalidAlias(String)
    case invalidValue(String)
    case incompatibleEngine(modelID: String, requested: RuntimeServingEngine, expected: RuntimeServingEngine?)
    case incompatibleKVCacheMode(modelID: String, requested: RuntimeKVCacheMode, expectedEngine: RuntimeServingEngine)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let id):
            return "Unknown managed model '\(id)'."
        case .unsupportedModel(let id):
            return "Model '\(id)' is not supported by `mere.run api serve`."
        case .invalidAlias(let reason):
            return "Invalid alias: \(reason)."
        case .invalidValue(let reason):
            return "Invalid runtime setting: \(reason)."
        case .incompatibleEngine(let modelID, let requested, let expected):
            let expectedText = expected?.rawValue ?? "none"
            return "Engine override '\(requested.rawValue)' is not compatible with model '\(modelID)' (expected \(expectedText))."
        case .incompatibleKVCacheMode(let modelID, let requested, let expectedEngine):
            return "KV cache mode '\(requested.rawValue)' is not compatible with model '\(modelID)' (expected \(expectedEngine.rawValue))."
        }
    }
}

public struct RuntimeModelSettingsStore {
    public let url: URL
    private let fileManager: FileManager

    public init(modelsDir: URL = MereRunModelPaths.modelsDir, fileManager: FileManager = .default) {
        self.url = Self.defaultURL(modelsDir: modelsDir)
        self.fileManager = fileManager
    }

    public init(url: URL, fileManager: FileManager = .default) {
        self.url = url
        self.fileManager = fileManager
    }

    public static func defaultURL(modelsDir: URL = MereRunModelPaths.modelsDir) -> URL {
        modelsDir
            .appendingPathComponent(".mere-run", isDirectory: true)
            .appendingPathComponent("runtime-model-settings.json", isDirectory: false)
    }

    public func load() throws -> RuntimeModelSettingsDocument {
        guard fileManager.fileExists(atPath: url.path) else {
            return RuntimeModelSettingsDocument()
        }
        let data = try Data(contentsOf: url)
        let document = try JSONDecoder().decode(RuntimeModelSettingsDocument.self, from: data)
        return RuntimeModelSettingsDocument(
            version: document.version,
            models: document.models.mapValues(\.normalized)
        )
    }

    public func save(_ document: RuntimeModelSettingsDocument) throws {
        let parent = url.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let normalized = RuntimeModelSettingsDocument(
            version: document.version,
            models: document.models.mapValues(\.normalized)
        )
        let data = try encoder.encode(normalized)
        try data.write(to: url, options: .atomic)
    }

    public func settings(for modelID: String) throws -> RuntimeModelSettings {
        try load().models[modelID] ?? RuntimeModelSettings()
    }

    public func writeSettings(_ settings: RuntimeModelSettings, for modelID: String) throws {
        guard let spec = ManagedModelCatalog.spec(for: modelID) else {
            throw RuntimeModelSettingsError.unknownModel(modelID)
        }
        let validated = try settings.validated(for: spec)
        var document = try load()
        if validated == RuntimeModelSettings() {
            document.models.removeValue(forKey: spec.id)
        } else {
            document.models[spec.id] = validated
        }
        try save(document)
    }

    public func resolveModelID(aliasOrID: String, defaultModelID: String) throws -> String {
        let requested = aliasOrID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requested.isEmpty else {
            return defaultModelID
        }

        let document = try load()
        if let match = document.models.first(where: { _, settings in
            settings.alias?.caseInsensitiveCompare(requested) == .orderedSame
        }) {
            return match.key
        }
        if let spec = ManagedModelCatalog.spec(for: requested) {
            return spec.id
        }
        if requested == defaultModelID {
            return defaultModelID
        }
        throw RuntimeModelSettingsError.unknownModel(requested)
    }
}

public extension ManagedModelSpec {
    var defaultRuntimeServingEngine: RuntimeServingEngine? {
        switch validationKind {
        case .codegenGGUF:
            return .textCode
        case .gemma4:
            return .textChatGemma4
        case .q35:
            return .textChatQ35
        case .deepseekV4FlashIMatrixGGUF:
            return .textChatDeepseekV4Flash
        case .hfTextChat where id == ModelResolver.ModelID.mebot.rawValue:
            return .textChatKlein
        default:
            return nil
        }
    }

    var isAPIServableRuntimeModel: Bool {
        defaultRuntimeServingEngine != nil
    }
}
