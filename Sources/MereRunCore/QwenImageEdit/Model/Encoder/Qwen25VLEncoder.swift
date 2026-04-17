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

        // Connect vision tower to text encoder
        textEncoder.setVisionTower(visionTower)
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
        let patchInputs = preparePatchInputs(pixelValues: pixelValues, gridThw: gridThw)

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
        // Calculate grid from image dimensions
        let height = inputImage.dim(2)
        let width = inputImage.dim(3)
        let gridThw = [(1, height / 14, width / 14)]  // patch_size = 14

        let (embeddings, _) = try encodeJoint(
            inputIds: tokenBatch.inputIds,
            attentionMask: tokenBatch.attentionMask,
            pixelValues: inputImage,
            gridThw: gridThw,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId
        )

        return embeddings
    }

    // MARK: - Helper Methods

    /// Prepare patch inputs from pixel values
    private func preparePatchInputs(
        pixelValues: MLXArray,
        gridThw: [(Int, Int, Int)]
    ) -> MLXArray {
        // For single image editing, we typically have batch=1
        // pixel_values shape: [batch, channels, height, width]
        // Need to convert to patch format: [batch, num_patches, patch_volume]

        let batch = pixelValues.dim(0)
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)

        let patchSize = 14  // Standard for Qwen-VL

        let patchH = height / patchSize
        let patchW = width / patchSize
        let numPatches = patchH * patchW

        // Reshape to patches
        // [B, C, H, W] -> [B, C, pH, patchSize, pW, patchSize]
        var x = pixelValues.reshaped(batch, channels, patchH, patchSize, patchW, patchSize)

        // Permute to [B, pH, pW, C, patchSize, patchSize]
        x = x.transposed(0, 2, 4, 1, 3, 5)

        // Flatten patches: [B, pH*pW, C*patchSize*patchSize]
        x = x.reshaped(batch, numPatches, channels * patchSize * patchSize)

        // For temporal dimension, repeat for single frame (temporalPatchSize = 2)
        // This duplicates the spatial features along the channel dimension
        let repeated = MLX.concatenated([x, x], axis: 2)
        return repeated
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
            headDim: textEncoderConfig.headDim ?? (textEncoderConfig.hiddenSize / textEncoderConfig.numAttentionHeads)
        )

        // Create vision config from nested config or defaults
        let visionConfig: QwenVisionConfiguration
        if let vc = textEncoderConfig.visionConfig {
            visionConfig = QwenVisionConfiguration(
                depth: vc.depth ?? 32,
                embedDim: vc.hiddenSize ?? 1280,
                numHeads: vc.numHeads ?? 16,
                patchSize: vc.spatialPatchSize ?? vc.patchSize ?? 14,
                temporalPatchSize: vc.temporalPatchSize ?? 2,
                spatialMergeSize: vc.spatialMergeSize ?? 2,
                inChannels: 3,
                outHiddenDim: vc.outHiddenSize ?? textEncoderConfig.hiddenSize,
                windowSize: vc.windowSize ?? 112,
                fullAttentionBlockIndices: vc.fullattBlockIndexes ?? [7, 15, 23, 31]
            )
        } else {
            // Default Qwen2.5-VL vision config
            visionConfig = QwenVisionConfiguration(
                depth: 32,
                embedDim: 1280,
                numHeads: 16,
                patchSize: 14,
                temporalPatchSize: 2,
                spatialMergeSize: 2,
                inChannels: 3,
                outHiddenDim: textEncoderConfig.hiddenSize,
                windowSize: 112,
                fullAttentionBlockIndices: [7, 15, 23, 31]
            )
        }

        return Qwen25VLEncoder(
            config: textEncoderConfig,
            textEncoderConfig: textConfig,
            visionConfig: visionConfig
        )
    }
}
