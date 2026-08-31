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
        guard let vision = config.visionConfig else {
            preconditionFailure("Q35VisionTower requires a Q35 vision config.")
        }
        let activation: QwenVisionConfiguration.Activation
        switch vision.hiddenAct?.lowercased() {
        case "gelu_pytorch_tanh", "gelu_tanh", "gelu":
            activation = .geluApproximate
        default:
            activation = .silu
        }

        let spatialMerge = max(1, vision.spatialMergeSize ?? 2)
        let useLearnedPosEmbed = (vision.numPositionEmbeddings ?? 0) > 0
        let inferredPatchEmbedBias = config.modelType.hasPrefix("qwen3_5") || config.textConfig.isQwen4Exp
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
            patchEmbedBias: vision.patchEmbedBias ?? inferredPatchEmbedBias,
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
        let arrays: [String: MLXArray]
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            let data = try Data(contentsOf: resources.modelIndexURL)
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
            let filenames = Set(index.weightMap.compactMap { key, filename in
                Self.mapVisionWeightKey(key) == nil ? nil : filename
            })
            var selected: [String: MLXArray] = [:]
            for filename in filenames.sorted() {
                let shard = try SafetensorsStreamingLoader.loadArrays(
                    url: resources.rootURL.appendingPathComponent(filename),
                    where: { index.weightMap[$0] == filename && Self.mapVisionWeightKey($0) != nil },
                    dtype: .bfloat16
                )
                selected.merge(shard) { _, replacement in replacement }
            }
            arrays = selected
        } else {
            arrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.modelWeightsURL,
                where: { Self.mapVisionWeightKey($0) != nil },
                dtype: .bfloat16
            )
        }
        let mapped = Dictionary(uniqueKeysWithValues: arrays.flatMap { Self.mapVisionWeight($0.key, $0.value) })
        if HFSafetensorsWeightsLoader.isQuantized(mapped) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(mapped, to: self)
        } else {
            try update(parameters: ModuleParameters.unflattened(mapped), verify: [.shapeMismatch])
        }

        // A managed model may be backed by an external-volume snapshot. Page
        // weights in through bounded command buffers before the first vision
        // forward so disk I/O cannot turn that forward into a Metal timeout.
        let values = parameters().flattened().map(\.1)
        for start in stride(from: 0, to: values.count, by: 8) {
            MLX.eval(Array(values[start..<min(start + 8, values.count)]))
            Stream.gpu.synchronize()
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
        MLX.eval(embeds)
        Stream.gpu.synchronize()
        return embeds
    }

    static func mapVisionWeight(_ rawKey: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard let key = mapVisionWeightKey(rawKey) else { return [] }
        if key == "visionTower.patch_embed.proj.weight", value.ndim == 5 {
            // Converted MLX checkpoints such as Bonsai already store Conv3d
            // kernels as [out, temporal, height, width, channels]. PyTorch
            // checkpoints use [out, channels, temporal, height, width].
            if value.dim(4) == 3 {
                return [(key, value)]
            }
            return [(key, value.transposed(0, 2, 3, 4, 1))]
        }

        return [(key, value)]
    }

    private static func mapVisionWeightKey(_ rawKey: String) -> String? {
        var key = rawKey
        if key.hasPrefix("model.vision_tower.") {
            key = "visionTower." + String(key.dropFirst("model.vision_tower.".count))
        } else if key.hasPrefix("model.visual.") {
            key = "visionTower." + String(key.dropFirst("model.visual.".count))
        } else if key.hasPrefix("vision_tower.") {
            key = "visionTower." + String(key.dropFirst("vision_tower.".count))
        } else if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        } else {
            return nil
        }

        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
        key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".deepstack_merger_list.", with: ".deepstack_merger_list.")
        key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")

        return key
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
