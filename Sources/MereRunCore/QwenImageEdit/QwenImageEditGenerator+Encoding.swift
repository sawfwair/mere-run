import Foundation
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Owns image preprocessing, semantic conditioning, and latent decoding.
/// These helpers form the middle of the edit pipeline between model loading and
/// denoising.
extension QwenImageEditGenerator {
    func encodeInputImage(
        url: URL,
        vae: QwenImageEditVAE,
        targetWidth: Int,
        targetHeight: Int
    ) async throws -> (latents: MLXArray, tensor: MLXArray) {
        #if !canImport(CoreGraphics)
        throw GeneratorError.unsupportedPlatform
        #else
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GeneratorError.inputImageNotFound(url)
        }
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            throw GeneratorError.inputImageDecodeFailed(url)
        }

        let vaeWidth = (targetWidth / 8) * 8
        let vaeHeight = (targetHeight / 8) * 8
        let vlWidth = (targetWidth / 14) * 14
        let vlHeight = (targetHeight / 14) * 14

        let vaeArray = try QwenImageIO.resizedPixelArray(
            from: cgImage,
            width: vaeWidth,
            height: vaeHeight,
            addBatchDimension: true,
            dtype: .float32
        )
        let normalized = QwenImageIO.normalizeForEncoder(vaeArray)
        let latents = vae.encode(normalized).asType(.bfloat16)

        let vlArray = try QwenImageIO.resizedPixelArray(
            from: cgImage,
            width: vlWidth,
            height: vlHeight,
            addBatchDimension: true,
            dtype: .float16
        )
        return (latents, vlArray)
        #endif
    }

    func encodeSemanticEmbeddingsForRequest(
        request: GenerationRequest,
        inputImageTensor: MLXArray,
        tokenizer: Qwen25VLTokenizer,
        encoder: Qwen25VLEncoder?,
        guidanceScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?,
        totalSteps: Int
    ) throws -> MLXArray {
        guard let encoder else {
            throw GeneratorError.modelLoadFailed("Encoder was not loaded.")
        }

        let numImageTokens = Qwen25VLTokenizer.imageTokenCount(
            imageHeight: inputImageTensor.dim(2),
            imageWidth: inputImageTensor.dim(3)
        )

        if guidanceScale > 1.0 {
            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 2, totalSteps: totalSteps))
            var unconditional = try encodeSingleSemanticEmbeddings(
                inputImage: inputImageTensor,
                prompt: request.negativePrompt ?? "",
                numImageTokens: numImageTokens,
                tokenizer: tokenizer,
                encoder: encoder
            ).asType(.bfloat16)
            MLX.eval(unconditional)
            Memory.clearCache()

            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 3, totalSteps: totalSteps))
            var conditional = try encodeSingleSemanticEmbeddings(
                inputImage: inputImageTensor,
                prompt: request.prompt,
                numImageTokens: numImageTokens,
                tokenizer: tokenizer,
                encoder: encoder
            ).asType(.bfloat16)
            MLX.eval(conditional)
            Memory.clearCache()

            let maxSeqLen = max(unconditional.dim(1), conditional.dim(1))
            unconditional = padSemanticEmbeddings(unconditional, to: maxSeqLen)
            conditional = padSemanticEmbeddings(conditional, to: maxSeqLen)

            let semanticEmbeds = MLX.concatenated([unconditional, conditional], axis: 0)
            MLX.eval(semanticEmbeds)
            Memory.clearCache()
            return semanticEmbeds
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 2, totalSteps: totalSteps))
        let semanticEmbeds = try encodeSemanticEmbeddings(
            inputImage: inputImageTensor,
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            guidanceScale: guidanceScale,
            tokenizer: tokenizer,
            encoder: encoder
        ).asType(.bfloat16)
        MLX.eval(semanticEmbeds)
        Memory.clearCache()
        return semanticEmbeds
    }

    private func padSemanticEmbeddings(_ embeds: MLXArray, to seqLen: Int) -> MLXArray {
        if embeds.dim(1) >= seqLen {
            return embeds
        }
        let padShape = [embeds.dim(0), seqLen - embeds.dim(1), embeds.dim(2)]
        let padding = MLXArray.zeros(padShape).asType(embeds.dtype)
        return MLX.concatenated([embeds, padding], axis: 1)
    }

    func encodeSemanticEmbeddings(
        inputImage: MLXArray,
        prompt: String,
        negativePrompt: String?,
        guidanceScale: Float,
        tokenizer: Qwen25VLTokenizer,
        encoder: Qwen25VLEncoder
    ) throws -> MLXArray {
        let numImageTokens = Qwen25VLTokenizer.imageTokenCount(
            imageHeight: inputImage.dim(2),
            imageWidth: inputImage.dim(3)
        )

        if guidanceScale > 1.0 {
            var unconditional = try encodeSingleSemanticEmbeddings(
                inputImage: inputImage,
                prompt: negativePrompt ?? "",
                numImageTokens: numImageTokens,
                tokenizer: tokenizer,
                encoder: encoder
            )
            var conditional = try encodeSingleSemanticEmbeddings(
                inputImage: inputImage,
                prompt: prompt,
                numImageTokens: numImageTokens,
                tokenizer: tokenizer,
                encoder: encoder
            )

            let maxSeqLen = max(unconditional.dim(1), conditional.dim(1))
            unconditional = padSemanticEmbeddings(unconditional, to: maxSeqLen)
            conditional = padSemanticEmbeddings(conditional, to: maxSeqLen)
            return MLX.concatenated([unconditional, conditional], axis: 0)
        }

        return try encodeSingleSemanticEmbeddings(
            inputImage: inputImage,
            prompt: prompt,
            numImageTokens: numImageTokens,
            tokenizer: tokenizer,
            encoder: encoder
        )
    }

    func encodeSingleSemanticEmbeddings(
        inputImage: MLXArray,
        prompt: String,
        numImageTokens: Int,
        tokenizer: Qwen25VLTokenizer,
        encoder: Qwen25VLEncoder
    ) throws -> MLXArray {
        let tokenBatch = tokenizer.encodeForEditing(prompt: prompt, numImageTokens: numImageTokens)

        guard let imageTokenId = tokenizer.imageTokenId,
              let visionStartTokenId = tokenizer.visionStartTokenId else {
            let (embeddings, _) = encoder.encodeText(
                inputIds: tokenBatch.inputIds,
                attentionMask: tokenBatch.attentionMask
            )
            return embeddings
        }

        return try encoder.encodeForEditing(
            inputImage: inputImage,
            tokenBatch: tokenBatch,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId
        )
    }

    func decodeLatents(
        _ latents: MLXArray,
        vae: QwenImageEditVAE,
        height: Int,
        width: Int
    ) -> MLXArray {
        let decoded = vae.decode(latents)
        var image = decoded

        if height != decoded.dim(2) || width != decoded.dim(3) {
            var nhwc = image.transposed(0, 2, 3, 1)
            let hScale = Float(height) / Float(decoded.dim(2))
            let wScale = Float(width) / Float(decoded.dim(3))
            nhwc = MLXNN.Upsample(scaleFactor: .array([hScale, wScale]), mode: .nearest)(nhwc)
            image = nhwc.transposed(0, 3, 1, 2)
        }

        image = QwenImageIO.denormalizeFromDecoder(image)
        return MLX.clip(image, min: 0, max: 1)
    }
}
