import Foundation
import MediaIO
import MLX
import MLXNN

struct QwenImageEditEncodedReferences {
    let plan: QwenImageEditConditioningPlan
    let appearanceLatents: [MLXArray]
    let semanticImages: [MLXArray]
}

struct QwenImageEditSemanticConditioning {
    let embeddings: MLXArray
    let attentionMask: MLXArray
}

/// Owns image preprocessing, semantic conditioning, and latent decoding.
extension QwenImageEditGenerator {
    func encodeReferenceImages(
        urls: [URL],
        vae: QwenImageEditVAE,
        outputWidth: Int,
        outputHeight: Int
    ) async throws -> QwenImageEditEncodedReferences {
        var decodedImages: [MediaImage] = []
        var referencePlans: [QwenImageEditReferencePlan] = []
        decodedImages.reserveCapacity(urls.count)
        referencePlans.reserveCapacity(urls.count)

        for url in urls {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw GeneratorError.inputImageNotFound(url)
            }
            let image: MediaImage
            do {
                image = try MediaImageIO.decode(url)
            } catch {
                throw GeneratorError.inputImageDecodeFailed(url)
            }
            decodedImages.append(image)
            referencePlans.append(QwenImageEditReferencePlan(
                source: url,
                sourceWidth: image.width,
                sourceHeight: image.height
            ))
        }

        let plan = QwenImageEditConditioningPlan(
            outputWidth: outputWidth,
            outputHeight: outputHeight,
            references: referencePlans
        )
        var appearanceLatents: [MLXArray] = []
        var semanticImages: [MLXArray] = []
        appearanceLatents.reserveCapacity(urls.count)
        semanticImages.reserveCapacity(urls.count)

        for (image, reference) in zip(decodedImages, referencePlans) {
            let sourcePixels = try QwenImageIO.array(
                from: image,
                addBatchDimension: false,
                dtype: .float32
            )
            let vaeArray = try QwenImageIO.resize(
                rgbArray: sourcePixels,
                targetHeight: reference.vaeSize.height,
                targetWidth: reference.vaeSize.width
            ).expandedDimensions(axis: 0)
            let normalizedVAEImage = QwenImageIO.normalizeForEncoder(vaeArray)
            appearanceLatents.append(vae.encodeConditioning(normalizedVAEImage).asType(.bfloat16))

            let semanticInput = try QwenImageIO.resize(
                rgbArray: sourcePixels,
                targetHeight: reference.semanticInputSize.height,
                targetWidth: reference.semanticInputSize.width
            )
            let semanticImage = try QwenImageIO.resize(
                rgbArray: semanticInput,
                targetHeight: reference.semanticSize.height,
                targetWidth: reference.semanticSize.width
            ).expandedDimensions(axis: 0).asType(.float16)
            semanticImages.append(Self.normalizeSemanticImage(semanticImage))
        }

        return QwenImageEditEncodedReferences(
            plan: plan,
            appearanceLatents: appearanceLatents,
            semanticImages: semanticImages
        )
    }

    func encodeSemanticEmbeddingsForRequest(
        request: GenerationRequest,
        inputImageTensors: [MLXArray],
        tokenizer: Qwen25VLTokenizer,
        encoder: Qwen25VLEncoder?,
        guidanceScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?,
        totalSteps: Int
    ) throws -> QwenImageEditSemanticConditioning {
        guard let encoder else {
            throw GeneratorError.modelLoadFailed("Encoder was not loaded.")
        }

        let imageTokenCounts = inputImageTensors.map { image in
            Qwen25VLTokenizer.imageTokenCount(
                imageHeight: image.dim(2),
                imageWidth: image.dim(3)
            )
        }

        if guidanceScale > 1.0 {
            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 2, totalSteps: totalSteps))
            var unconditional = try encodeSingleSemanticConditioning(
                inputImages: inputImageTensors,
                prompt: request.negativePrompt ?? " ",
                imageTokenCounts: imageTokenCounts,
                tokenizer: tokenizer,
                encoder: encoder
            )
            MLX.eval(unconditional.embeddings, unconditional.attentionMask)
            Memory.clearCache()

            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 3, totalSteps: totalSteps))
            var conditional = try encodeSingleSemanticConditioning(
                inputImages: inputImageTensors,
                prompt: request.prompt,
                imageTokenCounts: imageTokenCounts,
                tokenizer: tokenizer,
                encoder: encoder
            )
            MLX.eval(conditional.embeddings, conditional.attentionMask)
            Memory.clearCache()

            let maxSeqLen = max(unconditional.embeddings.dim(1), conditional.embeddings.dim(1))
            unconditional = padSemanticConditioning(unconditional, to: maxSeqLen)
            conditional = padSemanticConditioning(conditional, to: maxSeqLen)

            let result = QwenImageEditSemanticConditioning(
                embeddings: MLX.concatenated([unconditional.embeddings, conditional.embeddings], axis: 0),
                attentionMask: MLX.concatenated(
                    [unconditional.attentionMask, conditional.attentionMask],
                    axis: 0
                )
            )
            MLX.eval(result.embeddings, result.attentionMask)
            Memory.clearCache()
            return result
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 2, totalSteps: totalSteps))
        let result = try encodeSingleSemanticConditioning(
            inputImages: inputImageTensors,
            prompt: request.prompt,
            imageTokenCounts: imageTokenCounts,
            tokenizer: tokenizer,
            encoder: encoder
        )
        MLX.eval(result.embeddings, result.attentionMask)
        Memory.clearCache()
        return result
    }

    func encodeSingleSemanticConditioning(
        inputImages: [MLXArray],
        prompt: String,
        imageTokenCounts: [Int],
        tokenizer: Qwen25VLTokenizer,
        encoder: Qwen25VLEncoder
    ) throws -> QwenImageEditSemanticConditioning {
        let tokenBatch = tokenizer.encodeForEditing(
            prompt: prompt,
            numImageTokens: imageTokenCounts,
            maxLength: 1_024
        )

        guard let imageTokenId = tokenizer.imageTokenId,
              let visionStartTokenId = tokenizer.visionStartTokenId else {
            throw GeneratorError.modelLoadFailed("Tokenizer is missing Qwen vision special tokens.")
        }

        let encoded = try encoder.encodeForEditing(
            inputImages: inputImages,
            tokenBatch: tokenBatch,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId
        )
        return QwenImageEditSemanticConditioning(
            embeddings: encoded.embeddings.asType(.bfloat16),
            attentionMask: encoded.mask
        )
    }

    func decodeLatents(
        _ latents: MLXArray,
        vae: QwenImageEditVAE,
        height: Int,
        width: Int
    ) -> MLXArray {
        let decoded = vae.decodeGenerated(latents)
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

    private func padSemanticConditioning(
        _ conditioning: QwenImageEditSemanticConditioning,
        to seqLen: Int
    ) -> QwenImageEditSemanticConditioning {
        guard conditioning.embeddings.dim(1) < seqLen else {
            return conditioning
        }
        let missing = seqLen - conditioning.embeddings.dim(1)
        let embeddingPadding = MLXArray.zeros([
            conditioning.embeddings.dim(0),
            missing,
            conditioning.embeddings.dim(2),
        ]).asType(conditioning.embeddings.dtype)
        let maskPadding = MLXArray.zeros([
            conditioning.attentionMask.dim(0),
            missing,
        ]).asType(conditioning.attentionMask.dtype)
        return QwenImageEditSemanticConditioning(
            embeddings: MLX.concatenated([conditioning.embeddings, embeddingPadding], axis: 1),
            attentionMask: MLX.concatenated([conditioning.attentionMask, maskPadding], axis: 1)
        )
    }

    private static func normalizeSemanticImage(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float32(0.48145466), 0.4578275, 0.40821073]).reshaped(1, 3, 1, 1)
        let std = MLXArray([Float32(0.26862954), 0.26130258, 0.27577711]).reshaped(1, 3, 1, 1)
        return ((image.asType(.float32) - mean) / std).asType(.float16)
    }
}
