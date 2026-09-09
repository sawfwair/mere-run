import Foundation
import MLX
import MLXFast
import MLXNN

public final class Gemma4UnifiedCausalLM: Module, Gemma4CausalModel, @unchecked Sendable {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "vision_embedder") var visionEmbedder: Gemma4UnifiedVisionEmbedder
    @ModuleInfo(key: "embed_vision") var embedVision: Gemma4UnifiedMultimodalEmbedder

    public let config: Gemma4Config
    private let finalLogitSoftcapping: Float?
    private let imageTokenId: Int

    public init(config: Gemma4Config) throws {
        guard let visionConfig = config.visionConfig else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified runtime requires vision_config.")
        }
        guard let imageTokenId = config.imageTokenId else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified runtime requires image_token_id.")
        }
        self.config = config
        self.finalLogitSoftcapping = config.textConfig.finalLogitSoftcapping
        self.imageTokenId = imageTokenId
        self._languageModel.wrappedValue = Gemma4LanguageModel(config: config.textConfig)
        self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(config: visionConfig)
        self._embedVision.wrappedValue = Gemma4UnifiedMultimodalEmbedder(
            embeddingDim: visionConfig.outputProjDims,
            textHiddenSize: config.textConfig.hiddenSize,
            eps: visionConfig.rmsNormEps
        )
        super.init()
    }

    package func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> MLXArray {
        var logits = languageModel.logits(inputIds, cache: cache)
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    package func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        applyFinalSoftcap(languageModel.lastPositionLogits(inputIds, cache: cache))
    }

    package func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]? = nil) -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let output = languageModel.detailedLogits(
            embeddings: languageModel.embeddings(inputIds: tokenIds),
            inputIds: tokenIds,
            cache: cache
        )
        let logits = applyFinalSoftcap(output.logits)
        return Gemma4ForwardOutput(
            logits: logits,
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    package func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        mmTokenTypeIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) throws -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        var embeddings = languageModel.embeddings(inputIds: tokenIds)
        let imageFeatures = try compactImageFeatures(
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds,
            dtype: embeddings.dtype
        )
        embeddings = try replaceImageEmbeddings(
            embeddings,
            inputIds: tokenIds,
            imageFeatures: imageFeatures
        )
        var logits = languageModel.logits(
            embeddings: embeddings,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        if let finalLogitSoftcapping {
            let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    /// Projects logits only at assistant-target positions for image-conditioned
    /// SFT. The VLM training pipeline validates a single-image, batch-one
    /// contract before this differentiable path is constructed.
    package func trainingLogits(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        softTokenCounts: MLXArray,
        mmTokenTypeIds: MLXArray,
        flatTargetPositions: MLXArray
    ) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        let embeddings = languageModel.embeddings(inputIds: tokenIds)
        let embeddedImages = visionEmbedder(
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds
        )
        let projectedImages = embedVision(embeddedImages).asType(embeddings.dtype)

        let typedSoftTokenCounts = softTokenCounts.asType(.int32)
        let typedTokenIds = tokenIds.asType(.int32)
        MLX.eval(typedSoftTokenCounts, typedTokenIds)
        let softTokenCount = Int(typedSoftTokenCounts.asArray(Int32.self)[0])
        let imageFeatures = projectedImages[0, 0..<softTokenCount, 0...]
        let imagePositions = typedTokenIds.asArray(Int32.self).enumerated().compactMap { index, value in
            value == Int32(imageTokenId) ? index : nil
        }
        for (featureIndex, position) in imagePositions.enumerated() {
            embeddings[0, position, 0...] = imageFeatures[featureIndex, 0...]
        }

        let hidden = languageModel.forward(
            embeddings: embeddings,
            inputIds: tokenIds,
            mmTokenTypeIds: mmTokenTypeIds
        )
        let flattened = hidden.reshaped([-1, hidden.dim(-1)])
        let selected = take(
            flattened,
            flatTargetPositions.asType(.int32),
            axis: 0
        )
        return applyFinalSoftcap(languageModel.embedTokens.asLinear(selected))
    }

    package func forwardForSpeculation(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        mmTokenTypeIds: MLXArray,
        cache: [Gemma4AttentionCache]? = nil
    ) throws -> Gemma4ForwardOutput {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        var embeddings = languageModel.embeddings(inputIds: tokenIds)
        let imageFeatures = try compactImageFeatures(
            pixelValues: pixelValues,
            imagePositionIds: imagePositionIds,
            dtype: embeddings.dtype
        )
        embeddings = try replaceImageEmbeddings(
            embeddings,
            inputIds: tokenIds,
            imageFeatures: imageFeatures
        )
        let output = languageModel.detailedLogits(
            embeddings: embeddings,
            inputIds: tokenIds,
            cache: cache,
            mmTokenTypeIds: mmTokenTypeIds
        )
        return Gemma4ForwardOutput(
            logits: applyFinalSoftcap(output.logits),
            hidden: output.hidden,
            sharedKVStates: output.sharedKVStates
        )
    }

    package func inputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        var tokenIds = inputIds
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }
        return languageModel.embeddings(inputIds: tokenIds)
    }

    package func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray {
        applyFinalSoftcap(languageModel.embedTokens.asLinear(languageModel.norm(hidden)))
    }

    package func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.norm(hidden)
    }

    package func makeAttentionCache(quantization: Gemma4KVCacheQuantization? = nil) -> [Gemma4AttentionCache] {
        languageModel.makeCache(quantization: quantization)
    }

    public func makeCache(quantization: Gemma4KVCacheQuantization? = nil) -> [AnyObject] {
        makeAttentionCache(quantization: quantization)
    }

    private func applyFinalSoftcap(_ logits: MLXArray) -> MLXArray {
        guard let finalLogitSoftcapping else { return logits }
        let softcap = MLXArray(finalLogitSoftcapping).asType(logits.dtype)
        return tanh(logits / softcap) * softcap
    }

    private func compactImageFeatures(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray,
        dtype: DType
    ) throws -> MLXArray {
        let embedded = visionEmbedder(pixelValues: pixelValues, imagePositionIds: imagePositionIds)
        let projected = embedVision(embedded).asType(dtype)
        let typedPositions = imagePositionIds.asType(.int32)
        MLX.eval(typedPositions)
        let positions = typedPositions.asArray(Int32.self)
        let batch = projected.dim(0)
        let tokenCount = projected.dim(1)
        let hiddenSize = projected.dim(2)
        guard positions.count >= batch * tokenCount * 2 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified image positions have an invalid shape.")
        }

        var rows: [MLXArray] = []
        rows.reserveCapacity(batch * tokenCount)
        for batchIndex in 0..<batch {
            for tokenIndex in 0..<tokenCount {
                let offset = ((batchIndex * tokenCount) + tokenIndex) * 2
                guard positions[offset] >= 0, positions[offset + 1] >= 0 else {
                    continue
                }
                rows.append(projected[batchIndex, tokenIndex, 0...].reshaped(1, hiddenSize))
            }
        }
        guard !rows.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified image preprocessing produced no visual tokens.")
        }
        return concatenated(rows, axis: 0)
    }

    private func replaceImageEmbeddings(
        _ embeddings: MLXArray,
        inputIds: MLXArray,
        imageFeatures: MLXArray
    ) throws -> MLXArray {
        let tokenIds = inputIds.asType(.int32)
        MLX.eval(tokenIds)
        let values = tokenIds.asArray(Int32.self)
        let sequenceLength = embeddings.dim(1)
        let imagePositions = values.prefix(sequenceLength).enumerated().compactMap { index, value in
            value == Int32(imageTokenId) ? index : nil
        }
        guard imagePositions.count == imageFeatures.dim(0) else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 unified prompt has \(imagePositions.count) image tokens but image preprocessing produced \(imageFeatures.dim(0)) visual features."
            )
        }

        let merged = embeddings
        for (featureIndex, position) in imagePositions.enumerated() {
            merged[0, position, 0...] = imageFeatures[featureIndex, 0...]
        }
        return merged
    }
}
