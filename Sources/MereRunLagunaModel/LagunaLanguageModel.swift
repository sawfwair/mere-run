import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package struct LagunaLanguageModelOutput {
    package let hidden: MLXArray
    package let capturedHiddenStates: [Int: MLXArray]
}

package final class LagunaLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "layers") package var layers: [LagunaDecoderLayer]
    @ModuleInfo(key: "norm") package var norm: RMSNorm

    let config: LagunaConfig
    var fullRoPEAngleAtlas: MLXArray?
    var slidingRoPEAngleAtlas: MLXArray?

    package init(config: LagunaConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    package func callAsFunction(_ inputIDs: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        forward(inputIDs, cache: cache).hidden
    }

    package func forward(
        _ inputIDs: MLXArray,
        cache: [Gemma4AttentionCache]? = nil,
        captureLayerIndices: Set<Int> = [],
        lastPositionOnly: Bool = false,
        terminalPrefillRowEnabled: Bool? = nil,
        prefillAsyncLadderEnabled: Bool = true,
        useCustomKernels: Bool = true
    ) -> LagunaLanguageModelOutput {
        var hidden = embedTokens(inputIDs)
        var capturedHiddenStates: [Int: MLXArray] = [:]
        let sequenceLength = hidden.dim(1)
        let usesSharedMasks = LagunaGraphAccelerationPolicy.sharedAttentionMasksEnabled
            && hidden.dim(0) == 1
            && sequenceLength > 1
        let fullLayerIndex = config.layerTypes.firstIndex(of: "full_attention")
        let slidingLayerIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        let fullMask = usesSharedMasks ? fullLayerIndex.map { index in
            layers[index].selfAttention.prefillMask(
                queryLength: sequenceLength,
                cache: cache?[index],
                dtype: hidden.dtype
            )
        } : nil
        let slidingMask = usesSharedMasks ? slidingLayerIndex.map { index in
            layers[index].selfAttention.prefillMask(
                queryLength: sequenceLength,
                cache: cache?[index],
                dtype: hidden.dtype
            )
        } : nil
        let usesPrefillRoPEAtlas = useCustomKernels
            && LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled
            && hidden.dim(0) == 1
            && sequenceLength > 1
        let fullAtlas: MLXArray? = usesPrefillRoPEAtlas ? fullLayerIndex.flatMap { index in
            let offset = cache?[index].offset ?? 0
            guard offset >= 0,
                  offset + sequenceLength <= LagunaFusedPrefill.ropeAngleAtlasLength else {
                return nil
            }
            return fullRoPEAngleAtlas
        } : nil
        let slidingAtlas: MLXArray? = usesPrefillRoPEAtlas ? slidingLayerIndex.flatMap { index in
            let offset = cache?[index].offset ?? 0
            guard offset >= 0,
                  offset + sequenceLength <= LagunaFusedPrefill.ropeAngleAtlasLength else {
                return nil
            }
            return slidingRoPEAngleAtlas
        } : nil
        for (index, layer) in layers.enumerated() {
            let mask = config.layerTypes[index] == "full_attention"
                ? fullMask
                : slidingMask
            let ropeAtlas = config.layerTypes[index] == "full_attention"
                ? fullAtlas
                : slidingAtlas
            let useTerminalPrefillRow =
                (
                    terminalPrefillRowEnabled
                        ?? LagunaGraphAccelerationPolicy.terminalPrefillRowEnabled
                )
                    && lastPositionOnly
                    && hidden.dim(0) == 1
                    && hidden.dim(1) > 1
                    && index == layers.count - 1
                    && !captureLayerIndices.contains(index)
            if useTerminalPrefillRow {
                hidden = layer.callLastPrefillRow(hidden, cache: cache?[index])
            } else {
                hidden = layer(
                    hidden,
                    cache: cache?[index],
                    precomputedMask: mask,
                    precomputedRoPEAtlas: ropeAtlas,
                    useCustomKernels: useCustomKernels
                )
            }
            if captureLayerIndices.contains(index) {
                capturedHiddenStates[index] = hidden
            }
            let ladderStride = prefillAsyncLadderEnabled
                ? LagunaGraphAccelerationPolicy.prefillAsyncLadderStride
                : 0
            if ladderStride > 0,
               sequenceLength > 1,
               (index + 1).isMultiple(of: ladderStride) {
                asyncEval(hidden)
            }
            if useCustomKernels,
               LagunaGraphAccelerationPolicy.decodeAsyncStageEnabled,
               hidden.dim(0) == 1,
               sequenceLength == 1,
               LagunaGraphAccelerationPolicy.decodeAsyncLayerIndices.contains(index) {
                asyncEval(hidden)
            }
        }
        if lastPositionOnly, hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return LagunaLanguageModelOutput(
            hidden: norm(hidden),
            capturedHiddenStates: capturedHiddenStates
        )
    }

    package func makeCache() -> [Gemma4AttentionCache] {
        config.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return Gemma4SlidingKVCache(maxSize: config.slidingWindow)
            }
            return Gemma4FullKVCache()
        }
    }

    package func preparePrefillAcceleration() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if LagunaGraphAccelerationPolicy.prefillQKNormRoPEEnabled,
           let fullLayerIndex = config.layerTypes.firstIndex(of: "full_attention"),
           let slidingLayerIndex = config.layerTypes.firstIndex(of: "sliding_attention"),
           let fullAtlas = layers[fullLayerIndex].selfAttention.prefillRoPEAngleAtlas(
               length: LagunaFusedPrefill.ropeAngleAtlasLength
           ),
           let slidingAtlas = layers[slidingLayerIndex].selfAttention.prefillRoPEAngleAtlas(
               length: LagunaFusedPrefill.ropeAngleAtlasLength
           ) {
            self.fullRoPEAngleAtlas = fullAtlas
            self.slidingRoPEAngleAtlas = slidingAtlas
            arrays.append(contentsOf: [fullAtlas, slidingAtlas])
        }
        for layer in layers {
            guard let sparse = layer.mlp as? LagunaSparseMoE,
                  let warmUp = sparse.preparePrefillAcceleration() else {
                continue
            }
            arrays.append(warmUp)
            break
        }
        return arrays
    }

    package func prepareRuntimeAcceleration() -> [MLXArray] {
        for layer in layers {
            if let sparse = layer.mlp as? LagunaSparseMoE {
                sparse.preparePrefillPairwiseScaleReuse()
            }
        }
        var arrays = preparePrefillAcceleration()
        for layer in layers {
            arrays.append(contentsOf: layer.selfAttention.prepareNativeAffineQKV())
            arrays.append(contentsOf: layer.selfAttention.prepareNativeAffineGProj())
            arrays.append(contentsOf: layer.selfAttention.prepareNativeAffineOProj())
            arrays.append(
                contentsOf: layer.selfAttention.prepareTerminalPrefillProjectionWeights()
            )
        }
        var warmedHeadCounts: Set<Int> = []
        for (layerIndex, layer) in layers.enumerated() {
            let headCount = config.attentionHeads(layerIndex: layerIndex)
            guard warmedHeadCounts.insert(headCount).inserted,
                  let warmUp = layer.selfAttention.prepareFusedGatedAffineOProjWarmUp() else {
                continue
            }
            arrays.append(warmUp)
        }
        var warmedQKVRows: Set<Int> = []
        for layer in layers {
            guard let warmUp = layer.prepareFusedNormAffineQKVWarmUp(),
                  warmedQKVRows.insert(warmUp.rows).inserted else {
                continue
            }
            arrays.append(warmUp.output)
        }
        return arrays
    }

    package func invalidateTextLoRAUnsafeAcceleration() {
        for layer in layers {
            layer.selfAttention.invalidateTextLoRAUnsafeAcceleration()
        }
    }
}
