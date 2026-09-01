import Foundation
import MLX
import MLXNN

/// Qwen2.5-VL Encoder for Qwen-Image-Edit.
/// Combines vision tower (for image understanding) with text encoder (for prompt understanding).
/// Produces semantic embeddings that guide the diffusion process.
public final class Qwen25VLEncoder: Module {
    @ModuleInfo(key: "textEncoder") public var textEncoder: QwenTextEncoder
    @ModuleInfo(key: "visionTower") var visionTower: QwenVisionTower
    public let config: QwenImageEditTextEncoderConfig

    /// Projection from vision hidden size to text hidden size (if different)
    @ModuleInfo(key: "vision_projection") private var visionProjection: Linear?

    public init(
        config: QwenImageEditTextEncoderConfig,
        textEncoderConfig: QwenTextEncoderConfiguration,
        visionConfig: QwenVisionConfiguration
    ) {
        self.config = config
        self._textEncoder.wrappedValue = QwenTextEncoder(configuration: textEncoderConfig)
        self._visionTower.wrappedValue = QwenVisionTower(configuration: visionConfig)

        // Add projection if vision and text dimensions differ
        if visionConfig.outHiddenDim != textEncoderConfig.hiddenSize {
            self._visionProjection.wrappedValue = Linear(
                visionConfig.outHiddenDim,
                textEncoderConfig.hiddenSize
            )
        }
        super.init()
    }

    /// Access underlying text encoder for weight loading
    public var underlyingTextEncoder: QwenTextEncoder { textEncoder }

    /// Access underlying vision tower for weight loading (internal)
    var underlyingVisionTower: QwenVisionTower { visionTower }

    // MARK: - Text-Only Encoding

    /// Encode text prompt to embeddings (no image input)
    /// - Parameters:
    ///   - inputIds: Token IDs [batch, seq_len]
    ///   - attentionMask: Attention mask [batch, seq_len]
    /// - Returns: (embeddings, mask)
    public func encodeText(
        inputIds: MLXArray,
        attentionMask: MLXArray?
    ) -> (embeddings: MLXArray, mask: MLXArray) {
        textEncoder.encode(inputIds: inputIds, attentionMask: attentionMask)
    }

    // MARK: - Image-Only Encoding

    /// Encode image through vision tower
    /// - Parameters:
    ///   - pixelValues: Image tensor [batch, channels, height, width] normalized to [0, 1]
    ///   - gridThw: Grid specifications (temporal, height, width) for each image
    /// - Returns: Vision embeddings
    public func encodeImage(
        pixelValues: MLXArray,
        gridThw: [(Int, Int, Int)]
    ) throws -> MLXArray {
        // Prepare patches for vision tower
        let visionConfig = visionTower.configuration
        let patchInputs = Self.preparePatchInputs(
            pixelValues: pixelValues,
            patchSize: visionConfig.patchSize,
            temporalPatchSize: visionConfig.temporalPatchSize,
            mergeSize: visionConfig.spatialMergeSize
        )

        // Create grid metadata
        let grids = gridThw.map { thw in
            QwenVisionGrid(temporal: thw.0, height: thw.1, width: thw.2)
        }

        // Process through vision tower
        let output = try visionTower(patchInputs: patchInputs, grid: grids)

        // Project to text embedding dimension if needed
        var visionEmbeds = output.hiddenStates
        if let projection = visionProjection {
            visionEmbeds = projection(visionEmbeds)
        }

        return visionEmbeds
    }

    // MARK: - Joint Encoding (Image + Text)

    /// Encode image and text together for editing tasks
    /// - Parameters:
    ///   - inputIds: Token IDs with image placeholders [batch, seq_len]
    ///   - attentionMask: Attention mask [batch, seq_len]
    ///   - pixelValues: Image tensor [batch, channels, height, width]
    ///   - gridThw: Grid specifications for images
    ///   - imageTokenId: Token ID for image placeholders
    ///   - visionStartTokenId: Token ID marking start of vision tokens
    /// - Returns: (embeddings, mask) with vision features injected
    public func encodeJoint(
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        pixelValues: MLXArray,
        gridThw: [(Int, Int, Int)],
        imageTokenId: Int,
        visionStartTokenId: Int
    ) throws -> (embeddings: MLXArray, mask: MLXArray) {
        // Encode image
        let visionEmbeds = try encodeImage(pixelValues: pixelValues, gridThw: gridThw)

        // Use the text encoder's joint encoding which handles vision token replacement
        let spatialMergeSize = 2 // Standard for Qwen-VL
        return textEncoder.encodeJoint(
            inputIds: inputIds,
            attentionMask: attentionMask,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId,
            placeholderGridTHW: gridThw,
            spatialMergeSize: spatialMergeSize,
            replacements: [visionEmbeds]
        )
    }

    // MARK: - Encoding for Image Editing

    /// Encode for image editing: produces semantic embeddings from image + edit instruction
    /// - Parameters:
    ///   - inputImage: Input image tensor [1, 3, H, W] in [0, 1] range
    ///   - tokenBatch: Tokenized edit instruction
    ///   - imageTokenId: Token ID for image placeholders
    ///   - visionStartTokenId: Token ID marking start of vision
    /// - Returns: Semantic embeddings for diffusion conditioning
    public func encodeForEditing(
        inputImage: MLXArray,
        tokenBatch: QwenTokenBatch,
        imageTokenId: Int,
        visionStartTokenId: Int
    ) throws -> MLXArray {
        try encodeForEditing(
            inputImages: [inputImage],
            tokenBatch: tokenBatch,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId
        ).embeddings
    }

    public func encodeForEditing(
        inputImages: [MLXArray],
        tokenBatch: QwenTokenBatch,
        imageTokenId: Int,
        visionStartTokenId: Int
    ) throws -> (embeddings: MLXArray, mask: MLXArray) {
        let patchSize = visionTower.configuration.patchSize
        let grids = inputImages.map { image in
            (1, image.dim(2) / patchSize, image.dim(3) / patchSize)
        }
        let replacements = try zip(inputImages, grids).map { image, grid in
            try encodeImage(pixelValues: image, gridThw: [grid])
        }

        return textEncoder.encodeJoint(
            inputIds: tokenBatch.inputIds,
            attentionMask: tokenBatch.attentionMask,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId,
            placeholderGridTHW: grids,
            spatialMergeSize: 2,
            replacements: replacements,
            dropIndex: Qwen25VLTokenizer.promptDropIndex
        )
    }

    // MARK: - Helper Methods

    /// Prepare patch inputs from pixel values
    static func preparePatchInputs(
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
        let blockH = patchH / mergeSize
        let blockW = patchW / mergeSize
        precondition(blockH > 0 && blockW > 0)

        // Qwen's processor groups each mergeSize x mergeSize patch cell before
        // flattening. The vision tower's rotary positions and patch merger use
        // this exact ordering.
        var patches = pixelValues.reshaped(
            batch,
            channels,
            blockH,
            mergeSize,
            patchSize,
            blockW,
            mergeSize,
            patchSize
        )
        patches = patches.transposed(0, 2, 5, 3, 6, 1, 4, 7)
        patches = patches.reshaped(batch, numPatches, channels, patchSize * patchSize)

        let repeats = max(1, temporalPatchSize)
        let temporalSlices = (0..<repeats).map { _ in
            patches.expandedDimensions(axis: 3)
        }
        let temporal = temporalSlices.count == 1
            ? temporalSlices[0]
            : MLX.concatenated(temporalSlices, axis: 3)
        return temporal.reshaped(
            batch,
            numPatches,
            channels * repeats * patchSize * patchSize
        )
    }
}

// MARK: - Convenience Factory

extension Qwen25VLEncoder {
    /// Create encoder from Qwen-Image-Edit configs
    public static func fromConfig(
        textEncoderConfig: QwenImageEditTextEncoderConfig
    ) -> Qwen25VLEncoder {
        // Create QwenTextEncoderConfiguration from QwenImageEditTextEncoderConfig
        let textConfig = QwenTextEncoderConfiguration(
            vocabSize: textEncoderConfig.vocabSize,
            hiddenSize: textEncoderConfig.hiddenSize,
            numHiddenLayers: textEncoderConfig.numHiddenLayers,
            numAttentionHeads: textEncoderConfig.numAttentionHeads,
            numKeyValueHeads: textEncoderConfig.numKeyValueHeads ?? textEncoderConfig.numAttentionHeads,
            intermediateSize: textEncoderConfig.intermediateSize,
            ropeTheta: textEncoderConfig.ropeTheta ?? 1_000_000.0,
            maxPositionEmbeddings: textEncoderConfig.maxPositionEmbeddings ?? 32768,
            rmsNormEps: textEncoderConfig.rmsNormEps ?? 1e-6,
            promptDropIndex: 0,
            headDim: textEncoderConfig.headDim ?? (textEncoderConfig.hiddenSize / textEncoderConfig.numAttentionHeads),
            mropeSection: textEncoderConfig.ropeScaling?.mropeSection,
            mropeInterleaved: false,
            attentionBias: textEncoderConfig.attentionBias ?? true,
            useQKNorm: false
        )

        // Create vision config from nested config or defaults
        let visionConfig: QwenVisionConfiguration
        if let vc = textEncoderConfig.visionConfig {
            visionConfig = QwenVisionConfiguration(
                depth: vc.depth ?? 32,
                embedDim: vc.hiddenSize ?? 1280,
                mlpHiddenDim: vc.intermediateSize ?? 3420,
                hiddenAct: vc.hiddenAct == "silu" ? .silu : .geluApproximate,
                mlpStyle: .gated,
                numHeads: vc.numHeads ?? 16,
                patchSize: vc.spatialPatchSize ?? vc.patchSize ?? 14,
                temporalPatchSize: vc.temporalPatchSize ?? 2,
                spatialMergeSize: vc.spatialMergeSize ?? 2,
                inChannels: 3,
                outHiddenDim: vc.outHiddenSize ?? textEncoderConfig.hiddenSize,
                windowSize: vc.windowSize ?? 112,
                fullAttentionBlockIndices: vc.fullattBlockIndexes ?? [7, 15, 23, 31],
                normBias: false
            )
        } else {
            // Default Qwen2.5-VL vision config
            visionConfig = QwenVisionConfiguration(
                depth: 32,
                embedDim: 1280,
                mlpHiddenDim: 3420,
                hiddenAct: .silu,
                mlpStyle: .gated,
                numHeads: 16,
                patchSize: 14,
                temporalPatchSize: 2,
                spatialMergeSize: 2,
                inChannels: 3,
                outHiddenDim: textEncoderConfig.hiddenSize,
                windowSize: 112,
                fullAttentionBlockIndices: [7, 15, 23, 31],
                normBias: false
            )
        }

        return Qwen25VLEncoder(
            config: textEncoderConfig,
            textEncoderConfig: textConfig,
            visionConfig: visionConfig
        )
    }
}
