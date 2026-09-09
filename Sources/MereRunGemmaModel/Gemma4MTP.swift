import Foundation
import MLX
import MLXNN
import MereRunDecode

package struct Gemma4AssistantConfig: Decodable, Sendable, Hashable {
    package static let defaultBlockSize = 4

    package let modelType: String
    package let backboneHiddenSize: Int
    package let useOrderedEmbeddings: Bool
    package let tieWordEmbeddings: Bool
    package let blockSize: Int
    package let textConfig: Gemma4TextConfig

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

    package init(from decoder: Decoder) throws {
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
            ?? Gemma4AssistantConfig.defaultBlockSize
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

package struct Gemma4MTPDraft {
    package let tokens: [Int]
}

package final class Gemma4AssistantDraftModel: Module {
    @ModuleInfo(key: "model") var model: Gemma4AssistantInnerModel
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    package let config: Gemma4AssistantConfig

    package init(config: Gemma4AssistantConfig) throws {
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

    package func draftBlock(
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

}
