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

    /// Prefill step for generation with vision embeddings injected into the prompt.
    /// Returns (logits, cache, finalSeqLength, ropeDelta) after processing the full prompt.
    public func forwardPrefillForGeneration(
        inputIds: MLXArray,
        imageTokenId: Int,
        visionStartTokenId: Int,
        pixelValues: MLXArray,
        gridThw: [(Int, Int, Int)]
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
        let (mergedEmbeddings, placeholderPos, finalSeqLen) = Self.replaceVisionEmbeddings(
            hiddenStates: embeddings,
            inputIds: tokenIds,
            imageTokenId: imageTokenId,
            visionEmbeds: visionEmbeds
        )

        // Build visual position mask for deepstack (marks positions where vision tokens are)
        let visualPosMask = Self.buildVisualPositionMask(
            inputIds: tokenIds,
            imageTokenId: imageTokenId
        )

        // Compute M-RoPE position IDs for the expanded prompt sequence.
        let grid = gridThw.first ?? (1, 1, 1)
        let positionIds = Self.computePromptRopePositions(
            seqLen: finalSeqLen,
            placeholderPos: placeholderPos,
            numVisionTokens: numVisionTokens,
            gridThw: grid,
            spatialMergeSize: visionSpatialMergeSize
        )
        if Self.debugVisionStats {
            let maxPos = positionIds.max().item(Int32.self)
            print(
                "[QwenVLEncoder] placeholderPos=\(placeholderPos) numVisionTokens=\(numVisionTokens) " +
                "finalSeqLen=\(finalSeqLen) maxPos=\(maxPos)"
            )
        }

        // Run causal forward on the expanded embeddings with deepstack
        let logits = textEncoder.encoder.forwardCausal(
            embeddings: mergedEmbeddings,
            cache: cache,
            positionIds: positionIds,
            visualPosMask: visualPosMask,
            deepstackFeatures: deepstackFeatures
        )
        MLX.eval(logits)

        // Compute rope delta: how much the max position exceeds the sequence length
        let maxPos = positionIds.max().item(Int32.self)
        let ropeDelta = Int(maxPos) + 1 - finalSeqLen

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

    /// Build a mask marking image-token positions in the prompt sequence.
    private static func buildVisualPositionMask(inputIds: MLXArray, imageTokenId: Int) -> MLXArray {
        let seqLen = inputIds.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var mask = [Float32](repeating: 0.0, count: seqLen)
        for i in 0..<seqLen where tokenValues[i] == Int32(imageTokenId) {
            mask[i] = 1.0
        }
        return MLXArray(mask, [1, seqLen])
    }

    /// Compute M-RoPE position IDs for a prompt that already contains the expanded image-token run.
    private static func computePromptRopePositions(
        seqLen: Int,
        placeholderPos: Int,
        numVisionTokens: Int,
        gridThw: (Int, Int, Int),
        spatialMergeSize: Int
    ) -> MLXArray {
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
        return MLXArray(flat, [3, 1, seqLen])
    }

    // MARK: - Patch inputs + replacement helpers

    private static func preparePatchInputs(pixelValues: MLXArray, patchSize: Int, mergeSize: Int = 2) -> MLXArray {
        let batch = pixelValues.dim(0)
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
        var x = pixelValues.reshaped(batch, channels, blockH, mergeSize, patchSize, blockW, mergeSize, patchSize)

        // Transpose to merge-permuted order:
        // [batch, blockH, blockW, mergeSize, mergeSize, C, patchSize, patchSize]
        // This groups 2x2 patches together, matching Qwen's position embedding expectations
        x = x.transposed(0, 2, 5, 3, 6, 1, 4, 7)

        // Flatten to: [batch, numPatches, C, spatialSize]
        x = x.reshaped(batch, numPatches, channels, spatialSize)

        // For temporal duplication (temporal_patch_size=2):
        // Duplicate along temporal dimension to match [C, T, H, W] layout
        let t0 = x.expandedDimensions(axis: 3)  // [batch, numPatches, C, 1, spatial]
        let t1 = x.expandedDimensions(axis: 3)  // [batch, numPatches, C, 1, spatial]
        let temporal = MLX.concatenated([t0, t1], axis: 3)  // [batch, numPatches, C, 2, spatial]

        // Flatten to [batch, numPatches, C * T * spatial]
        return temporal.reshaped(batch, numPatches, channels * temporalPatchSize * spatialSize)
    }

    /// Replace the expanded <|image_pad|> token span with vision embeddings.
    /// Returns (mergedEmbeddings, placeholderPosition, sequenceLength).
    private static func replaceVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        visionEmbeds: MLXArray
    ) -> (MLXArray, Int, Int) {
        let seqLen = hiddenStates.dim(1)

        var visionTensor = visionEmbeds
        if visionTensor.dtype != hiddenStates.dtype {
            visionTensor = visionTensor.asType(hiddenStates.dtype)
        }

        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        // Find the contiguous <|image_pad|> span.
        var placeholderPos: Int? = nil
        var placeholderCount = 0
        for position in 0..<seqLen where tokenValues[position] == Int32(imageTokenId) {
            if placeholderPos == nil {
                placeholderPos = position
            }
            placeholderCount += 1
        }

        guard let pos = placeholderPos else {
            return (hiddenStates, 0, seqLen)
        }

        let numVisionTokens = visionTensor.dim(0)
        precondition(
            placeholderCount == numVisionTokens,
            "[QwenVLEncoder] image token span mismatch: prompt has \(placeholderCount) placeholders, vision tower produced \(numVisionTokens) tokens"
        )

        let expectedPositions = Array(pos..<(pos + numVisionTokens))
        let actualPositions = tokenValues.enumerated().compactMap { index, value in
            value == Int32(imageTokenId) ? index : nil
        }
        precondition(
            actualPositions == expectedPositions,
            "[QwenVLEncoder] expected a contiguous image-token span in the prompt"
        )

        let merged = hiddenStates
        merged[0, pos..<(pos + numVisionTokens), 0...] = visionTensor
        return (merged, pos, seqLen)
    }
}

extension QwenEncoder {
    public func forwardCausal(
        embeddings: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil,
        visualPosMask: MLXArray? = nil,
        deepstackFeatures: [MLXArray] = []
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
            if i < deepstackFeatures.count, let visualMask = visualPosMask {
                h = applyDeepstackFeatures(
                    hiddenStates: h,
                    visualPosMask: visualMask,
                    visualEmbeds: deepstackFeatures[i]
                )
            }
        }

        h = norm(h)
        return embedTokens.asLinear(h)
    }

    /// Add deepstack visual features to hidden states at visual token positions
    private func applyDeepstackFeatures(
        hiddenStates: MLXArray,
        visualPosMask: MLXArray,
        visualEmbeds: MLXArray
    ) -> MLXArray {
        // visualPosMask: [1, seqLen] with 1s at vision positions
        // visualEmbeds: [numVisionTokens, hiddenDim]
        // hiddenStates: [1, seqLen, hiddenDim]

        let seqLen = hiddenStates.dim(1)
        let hiddenDim = hiddenStates.dim(2)
        let numVision = visualEmbeds.dim(0)

        // Find indices where mask is 1
        MLX.eval(visualPosMask)
        let maskFlat = visualPosMask.reshaped(seqLen)
        let maskArray = maskFlat.asArray(Float32.self)

        var indices: [Int] = []
        for i in 0..<seqLen where maskArray[i] > 0.5 {
            indices.append(i)
        }

        if indices.isEmpty || indices.count != numVision {
            return hiddenStates
        }

        // Build full-size embeddings with visual embeds at right positions
        var paddedEmbeds = [[Float32]](repeating: [Float32](repeating: 0, count: hiddenDim), count: seqLen)

        // Convert visual embeds to array
        let visualEmbedsTyped = visualEmbeds.asType(.float32)
        MLX.eval(visualEmbedsTyped)
        let visualData = visualEmbedsTyped.asArray(Float32.self)

        // Place visual embeds at their positions
        for (i, pos) in indices.enumerated() {
            let offset = i * hiddenDim
            for j in 0..<hiddenDim {
                paddedEmbeds[pos][j] = visualData[offset + j]
            }
        }

        // Flatten and create MLXArray
        let flatPadded = paddedEmbeds.flatMap { $0 }
        let paddedArray = MLXArray(flatPadded, [seqLen, hiddenDim]).asType(hiddenStates.dtype)

        // Add to hidden states: h + padded_visual_embeds
        let h = hiddenStates.reshaped(seqLen, hiddenDim)
        let result = h + paddedArray

        return result.reshaped(1, seqLen, hiddenDim)
    }
}
