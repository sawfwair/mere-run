import Foundation
import MLX
import MLXNN

public final class Q35VisionTower: Module {
    @ModuleInfo(key: "visionTower") private var visionTower: QwenVisionTower
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear?

    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public private(set) var isLoaded: Bool = false

    public init(config: Q35Config) {
        let vision = config.visionConfig
        let activation: QwenVisionConfiguration.Activation
        switch vision.hiddenAct?.lowercased() {
        case "gelu_pytorch_tanh", "gelu_tanh", "gelu":
            activation = .geluApproximate
        default:
            activation = .silu
        }

        let spatialMerge = max(1, vision.spatialMergeSize ?? 2)
        let useLearnedPosEmbed = (vision.numPositionEmbeddings ?? 0) > 0
        let qwenVisionConfig = QwenVisionConfiguration(
            depth: vision.depth,
            embedDim: vision.hiddenSize,
            mlpHiddenDim: vision.intermediateSize,
            hiddenAct: activation,
            numHeads: vision.numHeads,
            eps: 1e-6,
            patchSize: vision.patchSize,
            temporalPatchSize: vision.temporalPatchSize,
            spatialMergeSize: spatialMerge,
            inChannels: vision.inChannels,
            outHiddenDim: vision.outHiddenSize,
            windowSize: vision.windowSize ?? 112,
            fullAttentionBlockIndices: vision.fullAttentionBlockIndexes ?? [],
            patchEmbedBias: false,
            numPositionEmbeddings: vision.numPositionEmbeddings,
            useLearnedPosEmbed: useLearnedPosEmbed,
            deepstackVisualIndexes: vision.deepstackVisualIndexes ?? []
        )

        self.patchSize = qwenVisionConfig.patchSize
        self.temporalPatchSize = qwenVisionConfig.temporalPatchSize
        self.spatialMergeSize = qwenVisionConfig.spatialMergeSize
        self._visionTower.wrappedValue = QwenVisionTower(configuration: qwenVisionConfig)

        if qwenVisionConfig.outHiddenDim != config.textConfig.hiddenSize {
            self._visionProjection.wrappedValue = Linear(
                qwenVisionConfig.outHiddenDim,
                config.textConfig.hiddenSize,
                bias: false
            )
        }
        super.init()
    }

    public func loadWeights(from resources: Q35Resources) throws {
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: resources.modelIndexURL,
                to: self,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: Self.mapVisionWeight
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.modelWeightsURL,
                to: self,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: Self.mapVisionWeight
            )
        }

        isLoaded = true
    }

    public func encodeImage(
        pixelValues: MLXArray,
        gridTHW: (Int, Int, Int)
    ) throws -> MLXArray {
        let patchInputs = Self.preparePatchInputs(
            pixelValues: pixelValues,
            patchSize: patchSize,
            temporalPatchSize: temporalPatchSize,
            mergeSize: spatialMergeSize
        )
        let grid = [QwenVisionGrid(
            temporal: max(1, gridTHW.0),
            height: max(1, gridTHW.1),
            width: max(1, gridTHW.2)
        )]
        let output = try visionTower(patchInputs: patchInputs, grid: grid)

        var embeds = output.hiddenStates
        if let projection = visionProjection {
            embeds = projection(embeds)
        }
        return embeds
    }

    private static func mapVisionWeight(_ rawKey: String, _ value: MLXArray) -> [(String, MLXArray)] {
        var key = rawKey
        if key.hasPrefix("vision_tower.") {
            key = "visionTower." + String(key.dropFirst("vision_tower.".count))
        } else if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        } else {
            return []
        }

        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
        key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".deepstack_merger_list.", with: ".deepstack_merger_list.")
        key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")

        return [(key, value)]
    }

    private static func preparePatchInputs(
        pixelValues: MLXArray,
        patchSize: Int,
        temporalPatchSize: Int,
        mergeSize: Int
    ) -> MLXArray {
        let batch = pixelValues.dim(0)
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)

        let patchH = height / patchSize
        let patchW = width / patchSize
        let numPatches = patchH * patchW

        let blockH = patchH / max(1, mergeSize)
        let blockW = patchW / max(1, mergeSize)
        precondition(blockH > 0 && blockW > 0, "Invalid image grid for Q35 vision tower")

        var x = pixelValues.reshaped(
            batch,
            channels,
            blockH,
            mergeSize,
            patchSize,
            blockW,
            mergeSize,
            patchSize
        )
        x = x.transposed(0, 2, 5, 3, 6, 1, 4, 7)
        x = x.reshaped(batch, numPatches, channels, patchSize * patchSize)

        let repeats = max(1, temporalPatchSize)
        let temporalSlices = (0..<repeats).map { _ in
            x.expandedDimensions(axis: 3)
        }
        let temporal = temporalSlices.count == 1
            ? temporalSlices[0]
            : MLX.concatenated(temporalSlices, axis: 3)

        return temporal.reshaped(batch, numPatches, channels * repeats * patchSize * patchSize)
    }
}
