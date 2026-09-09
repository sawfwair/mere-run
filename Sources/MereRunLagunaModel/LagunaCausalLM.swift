import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package struct LagunaForwardOutput {
    package let logits: MLXArray
    package let capturedHiddenStates: [Int: MLXArray]
}

package final class LagunaCausalLM: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") package var model: LagunaLanguageModel
    @ModuleInfo(key: "lm_head") package var lmHead: Linear?

    package let config: LagunaConfig

    package init(config: LagunaConfig, quantizedSharedExperts: Bool = false) {
        self.config = config
        self._model.wrappedValue = LagunaLanguageModel(config: config)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()

        if quantizedSharedExperts, let quantization = config.quantization {
            let mode = QuantizationMode(rawValue: quantization.mode) ?? .affine
            for layer in model.layers where layer.mlp is LagunaSparseMoE {
                MLXNN.quantize(model: layer) { path, _ in
                    guard path.contains("shared_expert") else {
                        return nil
                    }
                    return (
                        groupSize: quantization.groupSize,
                        bits: quantization.bits,
                        mode: mode
                    )
                }
            }
        }
    }

    package func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        forward(inputIDs, cache: cache).logits
    }

    /// Project only loss-bearing flattened token positions through the large
    /// vocabulary head while retaining the full hidden-state training graph.
    package func trainingLogits(
        inputIDs: MLXArray,
        flatTargetPositions: MLXArray
    ) -> MLXArray {
        let hidden = model.forward(
            inputIDs,
            prefillAsyncLadderEnabled: false,
            useCustomKernels: false
        ).hidden
        let flattened = hidden.reshaped([-1, hidden.dim(-1)])
        let selected = take(
            flattened,
            flatTargetPositions.asType(.int32),
            axis: 0
        )
        return logits(from: selected)
    }

    /// Full-vocabulary fallback for training configurations that disable the
    /// gathered loss. Inference-only custom kernels do not define VJPs.
    package func trainingForward(_ inputIDs: MLXArray) -> MLXArray {
        let hidden = model.forward(
            inputIDs,
            prefillAsyncLadderEnabled: false,
            useCustomKernels: false
        ).hidden
        return logits(from: hidden)
    }

    package func forward(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil,
        captureLayerIndices: Set<Int> = [],
        lastPositionOnly: Bool = false,
        terminalPrefillRowEnabled: Bool? = nil
    ) -> LagunaForwardOutput {
        let output = model.forward(
            inputIDs,
            cache: cache,
            captureLayerIndices: captureLayerIndices,
            lastPositionOnly: lastPositionOnly,
            terminalPrefillRowEnabled: terminalPrefillRowEnabled
        )
        return LagunaForwardOutput(
            logits: lmHead?(output.hidden) ?? model.embedTokens.asLinear(output.hidden),
            capturedHiddenStates: output.capturedHiddenStates
        )
    }

    package func lastPositionLogits(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) -> MLXArray {
        forward(inputIDs, cache: cache, lastPositionOnly: true).logits
    }

    package func inputEmbeddings(for inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    package func logits(from hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    package func makeCache() -> [Gemma4AttentionCache] {
        model.makeCache()
    }

    package func preparePrefillAcceleration() -> [MLXArray] {
        model.preparePrefillAcceleration()
    }

    package func prepareRuntimeAcceleration() -> [MLXArray] {
        model.prepareRuntimeAcceleration()
    }

    package func invalidateTextLoRAUnsafeAcceleration() {
        model.invalidateTextLoRAUnsafeAcceleration()
    }
}
