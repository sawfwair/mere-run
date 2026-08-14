import Foundation

public enum RuntimeServingEngine: String, Codable, CaseIterable, Hashable, Sendable {
    case textCode = "text-code"
    case textChatKlein = "text-chat-klein"
    case textChatGemma4 = "text-chat-gemma4"
    case textChatLaguna = "text-chat-laguna"
    case textChatQ36 = "text-chat-q36"
    case textChatQ35 = "text-chat-q35"
    case textChatLFM2 = "text-chat-lfm2"
    case textChatDeepseekV4Flash = "text-chat-deepseek-v4-flash"
    case textChatMuseGlimmer = "text-chat-muse-glimmer"
    case textChatNemotronH = "text-chat-nemotron-h"

    public var canonical: RuntimeServingEngine {
        switch self {
        case .textChatQ35:
            return .textChatQ36
        default:
            return self
        }
    }

    public func isCompatible(with expected: RuntimeServingEngine) -> Bool {
        canonical == expected.canonical
    }
}

public enum RuntimeKVCacheMode: String, Codable, CaseIterable, Hashable, Sendable {
    case `default`
    /// Affine 4-bit resident K/V for long-context native attention models.
    case affine4
    /// Affine 8-bit resident K/V. This is an explicit quality/memory tradeoff;
    /// it reduces BF16 caches but can be larger than a model-specific 4-bit
    /// default such as Gemma4 TurboQuant.
    case affine8
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
        case .affine4:
            return Gemma4KVCacheQuantization(
                bits: 4,
                scheme: .uniform,
                groupSize: fallback.groupSize,
                quantizedStart: 0
            )
        case .affine8:
            return Gemma4KVCacheQuantization(
                bits: 8,
                scheme: .uniform,
                groupSize: fallback.groupSize,
                quantizedStart: 0
            )
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

    var genericCacheLabel: String {
        switch self {
        case .affine4:
            return "resident-affine-4bit"
        case .affine8:
            return "resident-affine-8bit"
        case .default, .polar2, .auto:
            return "native"
        }
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
    public var minP: Double?
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
        minP: Double? = nil,
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
        self.minP = minP
        self.engineOverride = engineOverride
        self.kvCacheMode = kvCacheMode
    }

    private enum CodingKeys: String, CodingKey {
        case alias
        case pinned
        case ttlSeconds
        case maxContextTokens
        case maxTokens
        case temperature
        case topP
        case minP
        case engineOverride
        case kvCacheMode
        /// New cache modes are written under an additive key so older mere.run
        /// binaries ignore them instead of failing to decode the whole shared
        /// settings document. Existing modes retain their version-1 wire key.
        case extendedKVCacheMode = "kvCacheModeV2"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        alias = try container.decodeIfPresent(String.self, forKey: .alias)
        pinned = try container.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        ttlSeconds = try container.decodeIfPresent(Int.self, forKey: .ttlSeconds)
        maxContextTokens = try container.decodeIfPresent(Int.self, forKey: .maxContextTokens)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        minP = try container.decodeIfPresent(Double.self, forKey: .minP)
        engineOverride = try container.decodeIfPresent(RuntimeServingEngine.self, forKey: .engineOverride)
        if let rawMode = try container.decodeIfPresent(String.self, forKey: .extendedKVCacheMode) {
            kvCacheMode = RuntimeKVCacheMode(rawValue: rawMode)
        } else {
            kvCacheMode = try container.decodeIfPresent(RuntimeKVCacheMode.self, forKey: .kvCacheMode)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(alias, forKey: .alias)
        try container.encode(pinned, forKey: .pinned)
        try container.encodeIfPresent(ttlSeconds, forKey: .ttlSeconds)
        try container.encodeIfPresent(maxContextTokens, forKey: .maxContextTokens)
        try container.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(minP, forKey: .minP)
        try container.encodeIfPresent(engineOverride, forKey: .engineOverride)
        if let kvCacheMode, kvCacheMode == .affine4 || kvCacheMode == .affine8 {
            try container.encode(kvCacheMode.rawValue, forKey: .extendedKVCacheMode)
        } else {
            try container.encodeIfPresent(kvCacheMode, forKey: .kvCacheMode)
        }
    }

    public var normalized: RuntimeModelSettings {
        var copy = self
        copy.alias = normalizedOptionalString(alias)
        return copy
    }

    public func validated(for spec: ManagedModelSpec) throws -> RuntimeModelSettings {
        let settings = normalized
        guard spec.supportsRuntimeResidencySettings else {
            throw RuntimeModelSettingsError.unsupportedModel(spec.id)
        }
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
        if let minP = settings.minP, !(0...1).contains(minP) || !minP.isFinite {
            throw RuntimeModelSettingsError.invalidValue("minP must be between 0 and 1")
        }
        if let override = settings.engineOverride,
           let expected = spec.defaultRuntimeServingEngine,
           !override.isCompatible(with: expected) {
            throw RuntimeModelSettingsError.incompatibleEngine(
                modelID: spec.id,
                requested: override,
                expected: expected
            )
        }
        if spec.isAPISidecarRuntimeModel {
            guard settings.alias == nil,
                  settings.maxContextTokens == nil,
                  settings.maxTokens == nil,
                  settings.temperature == nil,
                  settings.topP == nil,
                  settings.minP == nil,
                  settings.engineOverride == nil,
                  settings.kvCacheMode == nil else {
                throw RuntimeModelSettingsError.invalidValue(
                    "sidecar models support only pinned and ttlSeconds"
                )
            }
        }
        if let kvCacheMode = settings.kvCacheMode, kvCacheMode != .default {
            let engine = spec.defaultRuntimeServingEngine?.canonical
            switch kvCacheMode {
            case .affine4, .affine8:
                let supportedEngines: [RuntimeServingEngine] = [
                    .textChatGemma4,
                    .textChatQ36,
                    .textChatLFM2,
                ]
                guard engine.map(supportedEngines.contains) == true else {
                    let supported = supportedEngines.map(\.rawValue).joined(separator: ", ")
                    throw RuntimeModelSettingsError.invalidValue(
                        "KV cache mode '\(kvCacheMode.rawValue)' is not compatible with model "
                            + "'\(spec.id)' (supported engines: \(supported))"
                    )
                }
            case .polar2, .auto:
                guard engine == .textChatGemma4 else {
                    throw RuntimeModelSettingsError.incompatibleKVCacheMode(
                        modelID: spec.id,
                        requested: kvCacheMode,
                        expectedEngine: .textChatGemma4
                    )
                }
            case .default:
                break
            }
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
    case incompatibleKVCacheMode(
        modelID: String,
        requested: RuntimeKVCacheMode,
        expectedEngine: RuntimeServingEngine
    )

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
        apiProfile?.servingEngine
    }

    var isAPIServableRuntimeModel: Bool {
        defaultRuntimeServingEngine != nil
    }

    var isAPISidecarRuntimeModel: Bool {
        switch validationKind {
        case .flux2Klein, .zimageTurbo, .hidreamO1, .krea2, .ideogram4SDNQ,
             .qwen3TTS, .qwen3ASR, .parakeet, .qwen3Embedding:
            return true
        default:
            return false
        }
    }

    var supportsRuntimeResidencySettings: Bool {
        isAPIServableRuntimeModel || isAPISidecarRuntimeModel
    }
}
