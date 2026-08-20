import Foundation
import MLX
import MLXNN

public struct QwenVLTextEncoderConfig: Decodable, Sendable, Hashable {
    public struct VisionConfig: Decodable, Sendable, Hashable {
        public let depth: Int?
        public let embedDim: Int?
        public let hiddenSize: Int?
        public let numHeads: Int?
        public let mlpRatio: Float?
        public let intermediateSize: Int?
        public let inChans: Int?
        public let inChannels: Int?
        public let patchSize: Int?
        public let spatialPatchSize: Int?
        public let temporalPatchSize: Int?
        public let spatialMergeSize: Int?
        public let outHiddenSize: Int?
        public let windowSize: Int?
        public let fullattBlockIndexes: [Int]?
        public let numPositionEmbeddings: Int?
        public let deepstackVisualIndexes: [Int]?

        enum CodingKeys: String, CodingKey {
            case depth
            case embedDim = "embed_dim"
            case hiddenSize = "hidden_size"
            case numHeads = "num_heads"
            case mlpRatio = "mlp_ratio"
            case intermediateSize = "intermediate_size"
            case inChans = "in_chans"
            case inChannels = "in_channels"
            case patchSize = "patch_size"
            case spatialPatchSize = "spatial_patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case outHiddenSize = "out_hidden_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
            case numPositionEmbeddings = "num_position_embeddings"
            case deepstackVisualIndexes = "deepstack_visual_indexes"
        }
    }

    public struct Quantization: Decodable, Sendable, Hashable {
        public let groupSize: Int?
        public let bits: Int?

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    public struct RopeScaling: Decodable, Sendable, Hashable {
        public let type: String?
        public let mropeSection: [Int]?
        public let mropeInterleaved: Bool?

        enum CodingKeys: String, CodingKey {
            case type
            case ropeType = "rope_type"
            case mropeSection = "mrope_section"
            case mropeInterleaved = "mrope_interleaved"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let explicitType = try container.decodeIfPresent(String.self, forKey: .type)
            let ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType)
            self.type = explicitType ?? ropeType
            self.mropeSection = try container.decodeIfPresent([Int].self, forKey: .mropeSection)
            self.mropeInterleaved = try container.decodeIfPresent(Bool.self, forKey: .mropeInterleaved)
        }
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int?
    public let intermediateSize: Int
    public let ropeTheta: Float?
    public let maxPositionEmbeddings: Int?
    public let rmsNormEps: Float?
    public let headDim: Int?
    public let visionConfig: VisionConfig?
    public let quantization: Quantization?
    public let ropeScaling: RopeScaling?
    public let modelType: String?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case headDim = "head_dim"
        case visionConfig = "vision_config"
        case quantization = "quantization"
        case ropeScaling = "rope_scaling"
        case modelType = "model_type"
    }

    public var isQwen3VL: Bool {
        modelType?.contains("qwen3") == true
    }
}

public final class QwenVLEncoder: Module {
    public struct ConditioningImage: @unchecked Sendable {
        public let pixelValues: MLXArray
        public let tokenRange: Range<Int>
        public let temporalPatchCount: Int
        public let heightPatchCount: Int
        public let widthPatchCount: Int

        public init(
            pixelValues: MLXArray,
            tokenRange: Range<Int>,
            temporalPatchCount: Int = 1,
            heightPatchCount: Int,
            widthPatchCount: Int
        ) {
            self.pixelValues = pixelValues
            self.tokenRange = tokenRange
            self.temporalPatchCount = temporalPatchCount
            self.heightPatchCount = heightPatchCount
            self.widthPatchCount = widthPatchCount
        }
    }

    private static let debugVisionStats: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_VLM_DEBUG_STATS"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()
    private static let disableDeepstack: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_VLM_DISABLE_DEEPSTACK"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    @ModuleInfo(key: "textEncoder") public var textEncoder: QwenTextEncoder
    @ModuleInfo(key: "visionTower") var visionTower: QwenVisionTower

    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear?

    public var visionPatchSize: Int { visionTower.configuration.patchSize }
    public var visionSpatialMergeSize: Int { visionTower.configuration.spatialMergeSize }

    public init(textEncoderConfig: QwenTextEncoderConfiguration, visionConfig: QwenVisionConfiguration) {
        self._textEncoder.wrappedValue = QwenTextEncoder(configuration: textEncoderConfig)
        self._visionTower.wrappedValue = QwenVisionTower(configuration: visionConfig)

        if visionConfig.outHiddenDim != textEncoderConfig.hiddenSize {
            self._visionProjection.wrappedValue = Linear(visionConfig.outHiddenDim, textEncoderConfig.hiddenSize)
        }
        super.init()
        textEncoder.setVisionTower(visionTower)
    }

    public static func imageTokenCount(
        imageHeight: Int,
        imageWidth: Int,
        patchSize: Int = 14,
        spatialMergeSize: Int = 2
    ) -> Int {
        let patchH = imageHeight / patchSize
        let patchW = imageWidth / patchSize
        let mergedH = patchH / spatialMergeSize
        let mergedW = patchW / spatialMergeSize
        return max(1, mergedH * mergedW)
    }

    /// Encodes H3-style multimodal presentations without a chat template or
    /// language-model head. Each image pad run is replaced in place with the
    /// Qwen3-VL vision tower output; the unnormalized activation immediately
    /// after `activationLayer` is returned.
    public func forwardMultimodalActivationHiddenState(
        inputIds: MLXArray,
        attentionMask: MLXArray,
        images: [ConditioningImage],
        activationLayer: Int
    ) throws -> MLXArray? {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 { tokenIds = tokenIds.asType(.int32) }
        let embeddings = textEncoder.encoder.embed(inputIds: tokenIds)
        var deepstackByImage: [[MLXArray]] = []

        for image in images {
            let expectedTokens = max(
                1,
                image.temporalPatchCount
                    * max(1, image.heightPatchCount / visionSpatialMergeSize)
                    * max(1, image.widthPatchCount / visionSpatialMergeSize)
            )
            precondition(
                image.tokenRange.count == expectedTokens,
                "Qwen3-VL image placeholder count does not match its patch grid"
            )
            let patchInputs = Self.preparePatchInputs(
                pixelValues: image.pixelValues,
                patchSize: visionPatchSize,
                mergeSize: visionSpatialMergeSize
            )
            let output = try visionTower(
                patchInputs: patchInputs,
                grid: [QwenVisionGrid(
                    temporal: image.temporalPatchCount,
                    height: image.heightPatchCount,
                    width: image.widthPatchCount
                )]
            )
            var visual = output.hiddenStates
            if let projection = visionProjection { visual = projection(visual) }
            precondition(
                visual.dim(0) == image.tokenRange.count,
                "Qwen3-VL vision tower output does not match its presentation span"
            )
            if visual.dtype != embeddings.dtype { visual = visual.asType(embeddings.dtype) }
            embeddings[0, image.tokenRange, 0...] = visual
            deepstackByImage.append(output.deepstackFeatures)
        }

        let deepstackLayerCount = deepstackByImage.map(\.count).max() ?? 0
        let deepstackByLayer: [[MLXArray]] = (0..<deepstackLayerCount).map { layer in
            deepstackByImage.map { imageFeatures in
                precondition(
                    layer < imageFeatures.count,
                    "Qwen3-VL images must expose the same number of deepstack features"
                )
                return imageFeatures[layer]
            }
        }
        let positionIds = Self.multimodalPositionIDs(
            sequenceLength: embeddings.dim(1),
            images: images,
            spatialMergeSize: visionSpatialMergeSize
        )
        return textEncoder.encoder.forwardMultimodalActivationHiddenState(
            embeddings: embeddings,
            attentionMask: attentionMask,
            positionIds: positionIds,
            visualTokenRanges: images.map(\.tokenRange),
            deepstackFeatures: deepstackByLayer,
            activationLayer: activationLayer
        )
    }

    /// Encodes a complete multimodal presentation and returns the final,
    /// normalized language-model hidden state. Retrieval models pool the last
    /// valid token from this sequence instead of applying the language-model
    /// head used by caption generation.
    public func forwardMultimodalHiddenState(
        inputIds: MLXArray,
        attentionMask: MLXArray,
        images: [ConditioningImage]
    ) throws -> MLXArray {
        let finalLayer = textEncoder.configuration.numHiddenLayers - 1
        guard let hiddenState = try forwardMultimodalActivationHiddenState(
            inputIds: inputIds,
            attentionMask: attentionMask,
            images: images,
            activationLayer: finalLayer
        ) else {
            preconditionFailure("Qwen3-VL text encoder has no final transformer layer")
        }
        return textEncoder.encoder.norm(hiddenState)
    }

    private static func multimodalPositionIDs(
        sequenceLength: Int,
        images: [ConditioningImage],
        spatialMergeSize: Int
    ) -> MLXArray {
        var axes = Array(repeating: [Int32](), count: 3)
        for axis in axes.indices { axes[axis].reserveCapacity(sequenceLength) }
        var cursor = 0
        var nextPosition = 0

        for image in images.sorted(by: { $0.tokenRange.lowerBound < $1.tokenRange.lowerBound }) {
            precondition(image.tokenRange.lowerBound >= cursor && image.tokenRange.upperBound <= sequenceLength)
            while cursor < image.tokenRange.lowerBound {
                for axis in axes.indices { axes[axis].append(Int32(nextPosition)) }
                cursor += 1
                nextPosition += 1
            }

            let temporal = max(1, image.temporalPatchCount)
            let height = max(1, image.heightPatchCount / spatialMergeSize)
            let width = max(1, image.widthPatchCount / spatialMergeSize)
            for temporalIndex in 0..<temporal {
                for heightIndex in 0..<height {
                    for widthIndex in 0..<width {
                        axes[0].append(Int32(nextPosition + temporalIndex))
                        axes[1].append(Int32(nextPosition + heightIndex))
                        axes[2].append(Int32(nextPosition + widthIndex))
                        cursor += 1
                    }
                }
            }
            nextPosition += max(temporal, max(height, width))
        }
        while cursor < sequenceLength {
            for axis in axes.indices { axes[axis].append(Int32(nextPosition)) }
            cursor += 1
            nextPosition += 1
        }
        return MLXArray(axes.flatMap { $0 }, [3, 1, sequenceLength])
    }

    /// Prefill step for generation with vision embeddings injected into the prompt.
    /// Returns (logits, cache, finalSeqLength, ropeDelta) after processing the full prompt.
    public func forwardPrefillForGeneration(
        inputIds: MLXArray,
        imageTokenId: Int,
        visionStartTokenId: Int,
        pixelValues: MLXArray,
        gridThw: [(Int, Int, Int)],
        imageTokenRange: Range<Int>? = nil
    ) throws -> (MLXArray, [KVCache], Int, Int) {
        let cache: [KVCache] = (0..<textEncoder.configuration.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }

        // Vision embeds
        let patchInputs = Self.preparePatchInputs(pixelValues: pixelValues, patchSize: visionPatchSize)
        if Self.debugVisionStats {
            print("[QwenVLEncoder] gridThw=\(gridThw) inputSeq=\(inputIds.dim(1))")
            Self.logTensorStats("patchInputs", tensor: patchInputs)
        }
        let grids = gridThw.map { QwenVisionGrid(temporal: $0.0, height: $0.1, width: $0.2) }
        let output = try visionTower(patchInputs: patchInputs, grid: grids)
        var visionEmbeds = output.hiddenStates
        if Self.debugVisionStats {
            Self.logTensorStats("visionEmbeds.preProjection", tensor: visionEmbeds)
            for (index, feature) in output.deepstackFeatures.enumerated() {
                Self.logTensorStats("deepstack[\(index)]", tensor: feature)
            }
        }
        if let projection = visionProjection {
            visionEmbeds = projection(visionEmbeds)
            if Self.debugVisionStats {
                Self.logTensorStats("visionEmbeds.postProjection", tensor: visionEmbeds)
            }
        }
        let numVisionTokens = visionEmbeds.dim(0)
        let deepstackFeatures = Self.disableDeepstack ? [] : output.deepstackFeatures

        // Build embeddings
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let embeddings = textEncoder.encoder.embed(inputIds: tokenIds)

        // Replace the expanded <|image_pad|> run with vision embeddings in place.
        let (mergedEmbeddings, visualTokenRange, finalSeqLen) = Self.replaceVisionEmbeddings(
            hiddenStates: embeddings,
            inputIds: tokenIds,
            imageTokenId: imageTokenId,
            visionEmbeds: visionEmbeds,
            imageTokenRange: imageTokenRange
        )
        let placeholderPos = visualTokenRange.lowerBound

        // Compute M-RoPE position IDs for the expanded prompt sequence.
        let grid = gridThw.first ?? (1, 1, 1)
        let ropePositions = Self.computePromptRopePositions(
            seqLen: finalSeqLen,
            placeholderPos: placeholderPos,
            numVisionTokens: numVisionTokens,
            gridThw: grid,
            spatialMergeSize: visionSpatialMergeSize
        )
        let positionIds = ropePositions.ids
        if Self.debugVisionStats {
            print(
                "[QwenVLEncoder] placeholderPos=\(placeholderPos) numVisionTokens=\(numVisionTokens) " +
                "finalSeqLen=\(finalSeqLen) maxPos=\(ropePositions.maxPosition)"
            )
        }

        // Run causal forward on the expanded embeddings with deepstack
        let logits = textEncoder.encoder.forwardCausal(
            embeddings: mergedEmbeddings,
            cache: cache,
            positionIds: positionIds,
            visualTokenRange: visualTokenRange,
            deepstackFeatures: deepstackFeatures,
            lastPositionOnly: true
        )
        MLX.eval(logits)

        // Compute rope delta: how much the max position exceeds the sequence length
        let ropeDelta = ropePositions.maxPosition + 1 - finalSeqLen

        return (logits, cache, finalSeqLen, ropeDelta)
    }

    private static func logTensorStats(_ name: String, tensor: MLXArray, prefixCount: Int = 16) {
        let floatTensor = tensor.asType(.float32)
        MLX.eval(floatTensor)
        let minValue = MLX.min(floatTensor).item(Float.self)
        let maxValue = MLX.max(floatTensor).item(Float.self)
        let meanValue = MLX.mean(floatTensor).item(Float.self)

        let flat = floatTensor.reshaped(floatTensor.size)
        let prefixLength = min(prefixCount, flat.size)
        let prefixSlice = flat[0..<prefixLength]
        MLX.eval(prefixSlice)
        let prefixValues = prefixSlice.asArray(Float.self)

        print(
            "[QwenVLEncoder] \(name) shape=\(tensor.shape) dtype=\(tensor.dtype) " +
            "min=\(minValue) max=\(maxValue) mean=\(meanValue) first\(prefixLength)=\(prefixValues)"
        )
    }

    /// Compute M-RoPE position IDs for a prompt that already contains the expanded image-token run.
    private static func computePromptRopePositions(
        seqLen: Int,
        placeholderPos: Int,
        numVisionTokens: Int,
        gridThw: (Int, Int, Int),
        spatialMergeSize: Int
    ) -> (ids: MLXArray, maxPosition: Int) {
        var positions = Array(repeating: [Int](), count: 3)
        for d in 0..<3 { positions[d].reserveCapacity(seqLen) }

        // Text before placeholder: positions 0..<placeholderPos
        for i in 0..<placeholderPos {
            for d in 0..<3 { positions[d].append(i) }
        }

        // Vision tokens with 3D grid positions
        let t = max(1, gridThw.0)
        let h = max(1, gridThw.1 / spatialMergeSize)
        let w = max(1, gridThw.2 / spatialMergeSize)
        let imageBase = placeholderPos

        for ti in 0..<t {
            for hi in 0..<h {
                for wi in 0..<w {
                    positions[0].append(imageBase + ti)
                    positions[1].append(imageBase + hi)
                    positions[2].append(imageBase + wi)
                }
            }
        }

        // Text after placeholder: continues from max vision position + 1
        let maxVisionPos = max(
            positions[0].last ?? 0,
            positions[1].last ?? 0,
            positions[2].last ?? 0
        )
        let textContinueBase = maxVisionPos + 1
        let tokensAfterPlaceholder = max(0, seqLen - placeholderPos - numVisionTokens)
        for i in 0..<tokensAfterPlaceholder {
            for d in 0..<3 { positions[d].append(textContinueBase + i) }
        }

        let flat = positions.flatMap { $0.map(Int32.init) }
        let maxPosition = positions.compactMap { $0.max() }.max() ?? 0
        return (MLXArray(flat, [3, 1, seqLen]), maxPosition)
    }

    // MARK: - Patch inputs + replacement helpers

    private static func preparePatchInputs(pixelValues: MLXArray, patchSize: Int, mergeSize: Int = 2) -> MLXArray {
        let frameCount = pixelValues.dim(0)
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)

        let patchH = height / patchSize
        let patchW = width / patchSize
        let numPatches = patchH * patchW
        let blockH = patchH / mergeSize
        let blockW = patchW / mergeSize
        let spatialSize = patchSize * patchSize  // 256 for patchSize=16
        let temporalPatchSize = 2  // Qwen uses temporal_patch_size=2

        // Input: [batch, C, H, W]
        // Reshape to separate merge blocks:
        // [batch, C, blockH, mergeSize, patchSize, blockW, mergeSize, patchSize]
        var x = pixelValues.reshaped(
            frameCount, channels, blockH, mergeSize, patchSize, blockW, mergeSize, patchSize
        )

        // Transpose to merge-permuted order:
        // [batch, blockH, blockW, mergeSize, mergeSize, C, patchSize, patchSize]
        // This groups 2x2 patches together, matching Qwen's position embedding expectations
        x = x.transposed(0, 2, 5, 3, 6, 1, 4, 7)

        // Flatten to: [batch, numPatches, C, spatialSize]
        x = x.reshaped(frameCount, numPatches, channels, spatialSize)

        // For temporal duplication (temporal_patch_size=2):
        // Duplicate along temporal dimension to match [C, T, H, W] layout
        if !frameCount.isMultiple(of: temporalPatchSize) {
            x = MLX.concatenated([x, x[(frameCount - 1)..., 0..., 0..., 0...]], axis: 0)
        }
        let temporalPatchCount = x.dim(0) / temporalPatchSize
        let temporal = x
            .reshaped(temporalPatchCount, temporalPatchSize, numPatches, channels, spatialSize)
            .transposed(0, 2, 3, 1, 4)

        // One media item per call: temporal patches extend the token axis, not
        // the batch axis described by the single Qwen grid entry.
        return temporal.reshaped(1, temporalPatchCount * numPatches, channels * temporalPatchSize * spatialSize)
    }

    /// Replace the expanded <|image_pad|> token span with vision embeddings.
    /// Returns (mergedEmbeddings, placeholderPosition, sequenceLength).
    private static func replaceVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        visionEmbeds: MLXArray,
        imageTokenRange: Range<Int>?
    ) -> (MLXArray, Range<Int>, Int) {
        let seqLen = hiddenStates.dim(1)

        var visionTensor = visionEmbeds
        if visionTensor.dtype != hiddenStates.dtype {
            visionTensor = visionTensor.asType(hiddenStates.dtype)
        }

        let numVisionTokens = visionTensor.dim(0)
        let resolvedRange: Range<Int>
        if let imageTokenRange {
            resolvedRange = imageTokenRange
        } else {
            // Compatibility fallback for direct API callers. The production
            // caption path passes the already-host-resident token range and
            // avoids this synchronization entirely.
            let tokenArray = inputIds.asType(.int32)
            MLX.eval(tokenArray)
            let tokenValues = tokenArray.asArray(Int32.self)
            let positions = tokenValues.enumerated().compactMap { index, value in
                value == Int32(imageTokenId) ? index : nil
            }
            guard let first = positions.first else {
                return (hiddenStates, 0..<0, seqLen)
            }
            resolvedRange = first..<(first + positions.count)
            precondition(
                positions == Array(resolvedRange),
                "[QwenVLEncoder] expected a contiguous image-token span in the prompt"
            )
        }

        precondition(
            resolvedRange.lowerBound >= 0 && resolvedRange.upperBound <= seqLen,
            "[QwenVLEncoder] image token span falls outside the prompt"
        )
        precondition(
            resolvedRange.count == numVisionTokens,
            "[QwenVLEncoder] image token span mismatch: prompt has \(resolvedRange.count) placeholders, vision tower produced \(numVisionTokens) tokens"
        )

        let merged = hiddenStates
        merged[0, resolvedRange, 0...] = visionTensor
        return (merged, resolvedRange, seqLen)
    }
}

extension QwenEncoder {
    public func forwardCausal(
        embeddings: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil,
        visualTokenRange: Range<Int>? = nil,
        deepstackFeatures: [MLXArray] = [],
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        var h = embeddings
        let n = h.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let cache = cache?.first {
            mask = cache.makeMask(n: n)
        } else if n == 1 {
            mask = .none
        } else {
            mask = .causal
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)

            // Apply deepstack visual features at early layers (0, 1, 2, ...)
            if i < deepstackFeatures.count, let visualTokenRange {
                h = Self.applyingDeepstackFeatures(
                    hiddenStates: h,
                    visualTokenRange: visualTokenRange,
                    visualEmbeds: deepstackFeatures[i]
                )
            }
        }

        h = norm(h)
        if lastPositionOnly && h.dim(1) > 1 {
            h = h[0..., (h.dim(1) - 1)..., 0...]
        }
        return embedTokens.asLinear(h)
    }

    /// Add deepstack visual features to hidden states at visual token positions
    static func applyingDeepstackFeatures(
        hiddenStates: MLXArray,
        visualTokenRange: Range<Int>,
        visualEmbeds: MLXArray
    ) -> MLXArray {
        // visualEmbeds: [numVisionTokens, hiddenDim]
        // hiddenStates: [1, seqLen, hiddenDim]

        let seqLen = hiddenStates.dim(1)
        let numVision = visualEmbeds.dim(0)
        guard visualTokenRange.lowerBound >= 0,
              visualTokenRange.upperBound <= seqLen,
              visualTokenRange.count == numVision else {
            return hiddenStates
        }

        var visualTensor = visualEmbeds
        if visualTensor.dtype != hiddenStates.dtype {
            visualTensor = visualTensor.asType(hiddenStates.dtype)
        }

        // Keep the entire operation on-device. The old implementation read
        // both the mask and every visual feature to Swift, built a nested
        // Float32 buffer, then uploaded it again at each deepstack layer.
        let padded = MLXArray.zeros(hiddenStates.shape, dtype: hiddenStates.dtype)
        padded[0, visualTokenRange, 0...] = visualTensor
        return hiddenStates + padded
    }
}
