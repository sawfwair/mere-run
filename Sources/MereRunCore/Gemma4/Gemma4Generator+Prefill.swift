import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func chunkedPrefill(
        model: any Gemma4CausalModel,
        promptTokens: [Int],
        cache: [Gemma4AttentionCache],
        startIndex: Int,
        existingLogits: MLXArray?,
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        checkpointTokenCounts: Set<Int> = [],
        captureSpeculation: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4PrefillResult {
        guard !promptTokens.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
        var hidden: MLXArray?
        var sharedKVStates: [String: Gemma4SharedKVState] = [:]
        if processed > 0, processed < promptTokens.count {
            progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(processed) prompt KV tokens"))
        }
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let end = RuntimePrefillCheckpointPlanner.nextEnd(
                processed: processed,
                total: promptTokens.count,
                chunkSize: Self.prefillChunkSize,
                checkpoints: checkpointTokenCounts
            )
            if promptTokens.count > Self.prefillChunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(promptTokens.count) tokens"))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            let chunkLogits: MLXArray
            if captureSpeculation, end == promptTokens.count {
                let output = model.forwardForSpeculation(inputIds: chunk, cache: cache)
                chunkLogits = output.logits
                hidden = output.hidden
                sharedKVStates = output.sharedKVStates
                MLX.eval(chunkLogits, output.hidden)
            } else {
                chunkLogits = model.prefillStep(inputIds: chunk, cache: cache)
                MLX.eval(chunkLogits)
            }
            logits = chunkLogits
            processed = end
            storePrefixKVCache(
                modelPath: modelPath,
                quantization: quantization,
                promptTokens: promptTokens,
                tokenCount: processed,
                cache: cache,
                logits: chunkLogits,
                priority: checkpointTokenCounts.contains(processed) ? .semantic : .chunk
            )
            await Task.yield()
        }

        guard let logits else {
            throw Gemma4Error.unsupportedConfiguration("Prefill did not produce logits.")
        }
        return Gemma4PrefillResult(logits: logits, hidden: hidden, sharedKVStates: sharedKVStates)
    }

    func unifiedPrefill(
        model: Gemma4UnifiedCausalLM,
        promptTokens: [Int],
        imageBatch: Gemma4UnifiedImageBatch,
        imageTokenId: Int,
        cache: [Gemma4AttentionCache],
        captureSpeculation: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4PrefillResult {
        guard !promptTokens.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Prompt is empty after tokenization.")
        }

        progressHandler?(ChatProgress(
            stage: .encoding,
            message: "Prefilling \(promptTokens.count) multimodal tokens"
        ))
        let inputIds = MLXArray(promptTokens.map(Int32.init))
            .reshaped(1, promptTokens.count)
        let mmTokenTypeIds = Gemma4UnifiedImageProcessor.mmTokenTypeIds(
            tokens: promptTokens,
            imageTokenId: imageTokenId
        )
        let result: Gemma4PrefillResult
        if captureSpeculation {
            let output = try model.forwardForSpeculation(
                inputIds: inputIds,
                pixelValues: imageBatch.pixelValues,
                imagePositionIds: imageBatch.imagePositionIds,
                mmTokenTypeIds: mmTokenTypeIds,
                cache: cache
            )
            MLX.eval(output.logits, output.hidden)
            result = Gemma4PrefillResult(
                logits: output.logits,
                hidden: output.hidden,
                sharedKVStates: output.sharedKVStates
            )
        } else {
            let logits = try model.forward(
                inputIds: inputIds,
                pixelValues: imageBatch.pixelValues,
                imagePositionIds: imageBatch.imagePositionIds,
                mmTokenTypeIds: mmTokenTypeIds,
                cache: cache
            )
            MLX.eval(logits)
            result = Gemma4PrefillResult(logits: logits, hidden: nil, sharedKVStates: [:])
        }
        await Task.yield()
        return result
    }

    func makeLayerCaches(
        model: any Gemma4CausalModel,
        quantization: Gemma4KVCacheQuantization
    ) throws -> [Gemma4AttentionCache] {
        model.makeAttentionCache(quantization: quantization)
    }

    func collectSharedKVStatesForMTP(
        config: Gemma4TextConfig,
        caches: [Gemma4AttentionCache]
    ) -> [String: Gemma4SharedKVState] {
        guard !caches.isEmpty else { return [:] }
        let firstSharedLayerIndex = max(0, config.numHiddenLayers - config.numKVSharedLayers)
        var cacheMap: [Int] = Array(0..<firstSharedLayerIndex)
        if firstSharedLayerIndex < config.numHiddenLayers {
            let concreteLayerTypes = Array(config.layerTypes.prefix(firstSharedLayerIndex))
            let sharedFullIndex = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIndex = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for index in firstSharedLayerIndex..<config.numHiddenLayers {
                if config.layerTypes[index] == "full_attention" {
                    cacheMap.append(sharedFullIndex)
                } else {
                    cacheMap.append(sharedSlidingIndex)
                }
            }
        }

        var states: [String: Gemma4SharedKVState] = [:]
        for index in 0..<min(config.numHiddenLayers, cacheMap.count) {
            let cacheIndex = cacheMap[index]
            guard cacheIndex < caches.count,
                  index < config.layerTypes.count,
                  let state = caches[cacheIndex].currentState() else {
                continue
            }
            let maxSize = (caches[cacheIndex] as? Gemma4SlidingKVCache)?.configuredMaxSize
            states[config.layerTypes[index]] = Gemma4SharedKVState(
                keys: state.0,
                values: state.1,
                offset: caches[cacheIndex].offset,
                maxSize: maxSize
            )
        }
        return states
    }
}
