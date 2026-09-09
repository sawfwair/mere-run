import Foundation
import MLX
import MLXFast
import MLXNN

public final class Gemma4TextCausalLM: Module, Gemma4CausalModel, @unchecked Sendable {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel

    public let config: Gemma4TextConfig
    private let finalLogitSoftcapping: Float?

    public init(config: Gemma4TextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._languageModel.wrappedValue = Gemma4LanguageModel(config: config)
        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [AnyObject]? = nil
    ) -> MLXArray {
        forward(inputIds: inputIds, cache: cache as? [Gemma4AttentionCache])
    }

    /// Logits for an explicit set of flattened (`[batch * seq]` row-major)
    /// positions only. SFT training reads logits solely at loss-masked target
    /// positions, and prompt/pad rows are the majority of a chat batch —
    /// projecting them through the 262k-vocab lm_head (plus the float32 loss
    /// chain) is pure waste. Hidden states still flow through every position,
    /// so gradients match the full-logits path exactly.
    package func trainingLogits(inputIds: MLXArray, flatTargetPositions: MLXArray) -> MLXArray {
        let hidden = languageModel(inputIds)
        let flattened = hidden.reshaped([-1, hidden.dim(-1)])
        let selected = take(flattened, flatTargetPositions.asType(.int32), axis: 0)
        return applyFinalSoftcap(languageModel.embedTokens.asLinear(selected))
    }

    package func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var logits = languageModel.logits(inputIds, cache: cache)
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    package func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        applyFinalSoftcap(languageModel.lastPositionLogits(inputIds, cache: cache))
    }

    package func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let output = languageModel.detailedLogits(
            embeddings: languageModel.embeddings(inputIds: tokenIds),
            inputIds: tokenIds,
            cache: cache
        )
        let logits = applyFinalSoftcap(output.logits)
        return Gemma4ForwardOutput(
            logits: logits,
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    package func inputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return languageModel.embeddings(inputIds: tokenIds)
    }

    package func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray {
        applyFinalSoftcap(languageModel.embedTokens.asLinear(languageModel.norm(hidden)))
    }

    package func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.norm(hidden)
    }

    package func makeAttentionCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        languageModel.makeCache(quantization: quantization)
    }

    public func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [AnyObject] {
        makeAttentionCache(quantization: quantization)
    }

    private func applyFinalSoftcap(_ logits: MLXArray) -> MLXArray {
        guard let finalLogitSoftcapping else { return logits }
        let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
        return tanh(logits / softcap) * softcap
    }
}
