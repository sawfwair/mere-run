import MLX
import MLXNN

extension Gemma4TextCausalLM {
    /// Explicit prefix-vocabulary screen; token IDs retain their global row numbers.
    package func makeRestrictedDraftHead(count: Int) -> Linear {
        Linear(weight: languageModel.embedTokens.weight[..<min(count, config.vocabSize)], bias: nil)
    }
    package func makeRestrictedDraftHead(tokenIDs: MLXArray) -> Linear {
        Linear(weight: take(languageModel.embedTokens.weight, tokenIDs, axis: 0), bias: nil)
    }
    /// A separate draft-only head. Target embedding and output parameters remain unchanged.
    package func makeQuantizedDraftHead(groupSize: Int = 64) -> QuantizedLinear {
        QuantizedLinear(weight: languageModel.embedTokens.weight, bias: nil, groupSize: groupSize, bits: 8)
    }

    package func softcapOutputLogits(_ logits: MLXArray) -> MLXArray { applyFinalSoftcap(logits) }

    /// Unscaled embedding rows for external diffusion drafters.
    package func rawInputEmbeddings(_ ids: MLXArray) -> MLXArray {
        languageModel.embedTokens(ids.asType(.int32))
    }

    /// Project already-normalized external hidden states, without applying target RMSNorm.
    package func outputLogits(_ hidden: MLXArray) -> MLXArray {
        applyFinalSoftcap(languageModel.embedTokens.asLinear(hidden))
    }

    package func forwardEmbeddings(
        _ embeddings: MLXArray, inputIDs: MLXArray,
        cache: [Gemma4AttentionCache], captureLayerIndices: Set<Int> = [], lastLogitsOnly: Bool = false
    ) -> (logits: MLXArray, states: [Int: MLXArray]) {
        let result = languageModel.forwardDetailed(
            embeddings: embeddings, inputIds: inputIDs, cache: cache,
            captureSharedKV: false, captureLayerIndices: captureLayerIndices
        )
        let hidden = lastLogitsOnly ? result.hidden[0..., (-1)..., 0...] : result.hidden
        return (outputLogits(hidden), result.capturedLayerStates)
    }

}
