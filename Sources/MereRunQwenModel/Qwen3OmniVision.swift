#if !os(iOS)
import MLX
import MLXNN
import MereRunTextEncoder

package struct Qwen3OmniVisualFeatures {
    package let embeddings: MLXArray
    package let deepstack: [MLXArray]
}

package final class Qwen3OmniVision: Module {
    @ModuleInfo var tower: QwenVisionTower

    package init(config: Qwen3OmniConfiguration.Vision) {
        self._tower.wrappedValue = QwenVisionTower(configuration: QwenVisionConfiguration(
            depth: config.depth, embedDim: config.hiddenSize, mlpHiddenDim: config.intermediateSize,
            hiddenAct: .geluApproximate, numHeads: config.numHeads, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize, spatialMergeSize: config.spatialMergeSize,
            outHiddenDim: config.outHiddenSize, fullAttentionBlockIndices: Array(0..<config.depth),
            patchEmbedBias: true, numPositionEmbeddings: (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize),
            useLearnedPosEmbed: true, deepstackVisualIndexes: config.deepstackVisualIndexes))
    }

    package func callAsFunction(patches: MLXArray, temporal: Int, height: Int, width: Int) throws -> Qwen3OmniVisualFeatures {
        let output = try tower(patchInputs: patches, grid: [.init(temporal: temporal, height: height, width: width)])
        return Qwen3OmniVisualFeatures(embeddings: output.hiddenStates, deepstack: output.deepstackFeatures)
    }
}
#endif
