import Foundation
import MLX
import MLXNN

public struct Gemma4MTPStats: Codable, Equatable, Sendable {
    public var available: Bool
    public var enabled: Bool
    public var active: Bool
    public var assistantModelPath: String?
    public var reason: String?
    public var blockSize: Int
    public var threshold: Int
    public var rounds: Int
    public var draftedTokens: Int
    public var acceptedTokens: Int
    public var rejectedTokens: Int

    public init(
        available: Bool = false,
        enabled: Bool = false,
        active: Bool = false,
        assistantModelPath: String? = nil,
        reason: String? = nil,
        blockSize: Int = 0,
        threshold: Int = 0,
        rounds: Int = 0,
        draftedTokens: Int = 0,
        acceptedTokens: Int = 0,
        rejectedTokens: Int = 0
    ) {
        self.available = available
        self.enabled = enabled
        self.active = active
        self.assistantModelPath = assistantModelPath
        self.reason = reason
        self.blockSize = blockSize
        self.threshold = threshold
        self.rounds = rounds
        self.draftedTokens = draftedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
    }
}

struct Gemma4MTPRuntimeState {
    var stats = Gemma4MTPStats()
}

struct Gemma4MTPResources: Sendable, Hashable {
    static let modelId = "text-chat-gemma4-12b-mtp"
    static let upstreamModelId = "google/gemma-4-12B-it-assistant"
    static let defaultBlockSize = Gemma4AssistantConfig.defaultBlockSize
    static let defaultPromptThreshold = 2_048
    static let snapshotPatterns = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    var rootURL: URL

    var configURL: URL { rootURL.appending(path: "config.json") }
    var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    var modelWeightsURL: URL { rootURL.appending(path: "model.safetensors") }

    func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }
        return missing
    }
}

enum Gemma4MTPPolicy {
    static func enabled(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        let raw = environment["MERERUN_GEMMA4_MTP"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }

    static func promptThreshold(environment: [String: String] = ProcessInfo.processInfo.environment) -> Int {
        if let raw = environment["MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS"],
           let value = Int(raw), value >= 0 {
            return value
        }
        return Gemma4MTPResources.defaultPromptThreshold
    }

    static func sampledSpeculationEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let raw = environment["MERERUN_GEMMA4_MTP_SAMPLED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "1" || raw == "true" || raw == "on"
    }

    static func blockSize(
        configured: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let raw = environment["MERERUN_GEMMA4_MTP_BLOCK_SIZE"],
           let value = Int(raw), value >= 2 {
            return min(16, value)
        }
        return min(16, max(2, configured))
    }

    static func activationReason(
        assistant: Gemma4AssistantDraftModel?,
        promptTokenCount: Int,
        generationConfig: GenerationConfig,
        prefixSeedWasUsed: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        activationReason(
            assistantAvailable: assistant != nil,
            promptTokenCount: promptTokenCount,
            generationConfig: generationConfig,
            prefixSeedWasUsed: prefixSeedWasUsed,
            environment: environment
        )
    }

    static func activationReason(
        assistantAvailable: Bool,
        promptTokenCount: Int,
        generationConfig: GenerationConfig,
        prefixSeedWasUsed: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        guard enabled(environment: environment) else {
            return "disabled by MERERUN_GEMMA4_MTP"
        }
        guard assistantAvailable else {
            return "assistant not installed"
        }
        // Sampled decode is speculative-safe here (the verify loop samples the
        // target at every position, so emitted tokens are true target samples
        // regardless of draft policy), but it is opt-in: with sampled drafts
        // the match probability collapses and 7.4k-context decode measured
        // 14.7 tok/s vs 30.2 for the pipelined sampled path. When opted in via
        // MERERUN_GEMMA4_MTP_SAMPLED=1 the drafts are generated greedily to
        // maximize the match rate.
        if generationConfig.temperature != 0, !sampledSpeculationEnabled(environment: environment) {
            return "non-greedy sampling (MERERUN_GEMMA4_MTP_SAMPLED unset)"
        }
        guard !prefixSeedWasUsed else {
            return "prefix KV reuse"
        }
        let threshold = promptThreshold(environment: environment)
        guard promptTokenCount >= threshold else {
            return "prompt below MTP threshold"
        }
        return nil
    }
}
