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
    static let defaultBlockSize = 4
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

struct Gemma4AssistantConfig: Decodable, Sendable, Hashable {
    let modelType: String
    let backboneHiddenSize: Int
    let useOrderedEmbeddings: Bool
    let tieWordEmbeddings: Bool
    let blockSize: Int
    let textConfig: Gemma4TextConfig

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case targetHiddenSize = "target_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case blockSize = "block_size"
        case numAssistantTokens = "num_assistant_tokens"
        case textConfig = "text_config"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let textConfig = try container.decode(Gemma4TextConfig.self, forKey: .textConfig)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_assistant"
        self.backboneHiddenSize = try container.decodeIfPresent(Int.self, forKey: .backboneHiddenSize)
            ?? container.decodeIfPresent(Int.self, forKey: .targetHiddenSize)
            ?? textConfig.hiddenSize
        self.useOrderedEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
            ?? container.decodeIfPresent(Int.self, forKey: .numAssistantTokens)
            ?? Gemma4MTPResources.defaultBlockSize
        self.textConfig = textConfig
    }
}

private final class Gemma4SharedKVAttentionCache: Gemma4AttentionCache {
    private let state: Gemma4SharedKVState
    let offset: Int

    init(state: Gemma4SharedKVState, positionOffset: Int) {
        self.state = state
        self.offset = positionOffset
    }

    func currentState() -> (MLXArray, MLXArray)? {
        (state.keys, state.values)
    }

    func append(keys: MLXArray, values: MLXArray) {
        _ = keys
        _ = values
    }

    func fork() -> Gemma4AttentionCache {
        Gemma4SharedKVAttentionCache(state: state, positionOffset: offset)
    }
}

final class Gemma4AssistantInnerModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: Gemma4TextConfig) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            Gemma4DecoderLayer(config: config, layerIndex: $0, forceKVShared: true)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

struct Gemma4MTPDraft {
    let tokens: [Int]
}

final class Gemma4AssistantDraftModel: Module {
    @ModuleInfo(key: "model") var model: Gemma4AssistantInnerModel
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: Gemma4AssistantConfig

    init(config: Gemma4AssistantConfig) throws {
        guard !config.useOrderedEmbeddings else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 ordered-embedding MTP assistants are not supported by this build.")
        }
        self.config = config
        self._model.wrappedValue = Gemma4AssistantInnerModel(config: config.textConfig)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize,
            config.textConfig.hiddenSize,
            bias: false
        )
        self._postProjection.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.backboneHiddenSize,
            bias: false
        )
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        super.init()
    }

    func draftBlock(
        lastToken: Int,
        hidden: MLXArray,
        sharedKVStates: [String: Gemma4SharedKVState],
        positionOffset: Int,
        blockSize: Int,
        baseModel: any Gemma4CausalModel,
        generationConfig: GenerationConfig,
        repetitionHistory: [Int]
    ) throws -> Gemma4MTPDraft {
        let total = max(1, blockSize) - 1
        guard total > 0 else {
            return Gemma4MTPDraft(tokens: [])
        }

        guard generationConfig.temperature == 0 else {
            return try draftBlockWithHostSampling(
                lastToken: lastToken,
                hidden: hidden,
                sharedKVStates: sharedKVStates,
                positionOffset: positionOffset,
                blockSize: blockSize,
                baseModel: baseModel,
                generationConfig: generationConfig,
                repetitionHistory: repetitionHistory
            )
        }

        var tokenArray = MLXArray([Int32(lastToken)]).reshaped(1, 1)
        var previousHidden = hidden
        var tokenArrays: [MLXArray] = []
        tokenArrays.reserveCapacity(total)
        for _ in 0..<total {
            let tokenEmbedding = baseModel.inputEmbeddings(for: tokenArray)
            let inputs = MLX.concatenated([tokenEmbedding, previousHidden], axis: -1)
            let output = try forward(
                inputsEmbeds: inputs,
                sharedKVStates: sharedKVStates,
                positionOffset: positionOffset
            )
            let draftLogits = output.logits[0, -1, 0...]
            tokenArray = argMax(draftLogits, axis: -1).asType(.int32).reshaped(1, 1)
            tokenArrays.append(tokenArray)
            previousHidden = output.hidden
        }
        let draftTokens = MLX.concatenated(tokenArrays, axis: 1)
        MLX.eval(draftTokens)
        let tokens = draftTokens.asArray(Int32.self).map(Int.init)
        return Gemma4MTPDraft(tokens: tokens)
    }

    private func draftBlockWithHostSampling(
        lastToken: Int,
        hidden: MLXArray,
        sharedKVStates: [String: Gemma4SharedKVState],
        positionOffset: Int,
        blockSize: Int,
        baseModel: any Gemma4CausalModel,
        generationConfig: GenerationConfig,
        repetitionHistory: [Int]
    ) throws -> Gemma4MTPDraft {
        let total = max(1, blockSize) - 1
        var tokenArray = MLXArray([Int32(lastToken)]).reshaped(1, 1)
        var previousHidden = hidden
        var history = repetitionHistoryArray(
            promptTokens: repetitionHistory,
            config: generationConfig
        )
        var banMask: MLXArray?
        var banMaskResolved = false
        var draftTokenArrays: [MLXArray] = []
        draftTokenArrays.reserveCapacity(total)

        // The sampled token feeds the next step as an array, so the whole
        // draft chain schedules with a single readback at the end instead of
        // one blocking sample per drafted token.
        for _ in 0..<total {
            let tokenEmbedding = baseModel.inputEmbeddings(for: tokenArray)
            let inputs = MLX.concatenated([tokenEmbedding, previousHidden], axis: -1)
            let output = try forward(
                inputsEmbeds: inputs,
                sharedKVStates: sharedKVStates,
                positionOffset: positionOffset
            )
            let draftLogits = output.logits[0, -1, 0...]
            if !banMaskResolved {
                banMaskResolved = true
                banMask = tokenBanMask(
                    vocabularySize: draftLogits.dim(-1),
                    dtype: draftLogits.dtype,
                    tokens: generationConfig.bannedTokens
                )
            }
            let next = sampledTokenArray(
                logits: draftLogits,
                config: generationConfig,
                previousTokenIndices: history,
                banMask: banMask
            )
            draftTokenArrays.append(next)
            history = appendingRepetitionHistory(history, token: next, config: generationConfig)
            tokenArray = next.reshaped(1, 1)
            previousHidden = output.hidden
        }

        guard !draftTokenArrays.isEmpty else {
            return Gemma4MTPDraft(tokens: [])
        }
        let stacked = MLX.stacked(draftTokenArrays)
        MLX.eval(stacked)
        return Gemma4MTPDraft(tokens: stacked.asArray(Int32.self).map(Int.init))
    }

    private func forward(
        inputsEmbeds: MLXArray,
        sharedKVStates: [String: Gemma4SharedKVState],
        positionOffset: Int
    ) throws -> (hidden: MLXArray, logits: MLXArray) {
        var hidden = preProjection(inputsEmbeds)
        for layer in model.layers {
            guard let state = sharedKVStates[layer.selfAttention.layerType] else {
                throw Gemma4Error.unsupportedConfiguration(
                    "Gemma4 MTP assistant missing shared KV for \(layer.selfAttention.layerType)."
                )
            }
            hidden = layer(
                hidden,
                cache: Gemma4SharedKVAttentionCache(state: state, positionOffset: positionOffset),
                perLayerInput: nil
            )
        }
        let assistantHidden = model.norm(hidden)
        let backboneHidden = postProjection(assistantHidden)
        let logits = lmHead?(assistantHidden) ?? model.embedTokens.asLinear(assistantHidden)
        return (backboneHidden, logits)
    }

    func loadWeights(from resources: Gemma4MTPResources) throws {
        let include: (String) -> Bool = { key in
            !key.hasPrefix("masked_embedding.")
                && !(key == "lm_head.weight" && self.config.tieWordEmbeddings)
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard include(key) else { return [] }
            return [(key, value)]
        }
        let quantizedModuleResolver: HFSafetensorsWeightsLoader.QuantizedModuleResolver = { _, _, _, _, biases, fallbackGroupSize, fallbackBits in
            if biases != nil || fallbackBits > 4 {
                return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.affine)
            }
            return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.nvfp4)
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: self,
                    groupSize: 64,
                    bits: 4,
                    quantizedModuleResolver: quantizedModuleResolver,
                    mapper: mapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: self,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: self,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([
                resources.modelIndexURL.lastPathComponent,
                resources.modelWeightsURL.lastPathComponent,
            ])
        }
    }

    private static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
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
