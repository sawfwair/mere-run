import Foundation
import MLX

extension LagunaGenerator {
    func prefill(
        model: LagunaCausalLM,
        promptTokens: [Int],
        cache: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> LagunaPrefillResult {
        var processed = 0
        var logits: MLXArray?
        let dflashContextStart = dflash.map {
            max(0, promptTokens.count - $0.config.slidingWindow)
        }
        let dflashCache: [Gemma4AttentionCache]?
        if let dflash, let dflashContextStart {
            dflashCache = dflash.makeCache(initialOffset: dflashContextStart)
        } else {
            dflashCache = nil
        }
        let prefillChunkSize = promptTokens.count > Self.prefillChunkingThreshold
            ? Self.prefillChunkSize
            : promptTokens.count
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let end = min(processed + prefillChunkSize, promptTokens.count)
            if promptTokens.count > Self.prefillChunkingThreshold {
                progressHandler?(ChatProgress(
                    stage: .encoding,
                    message: "Prefilling \(end)/\(promptTokens.count) Laguna tokens"
                ))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            let chunkLogits: MLXArray
            if let dflash, let dflashCache, let dflashContextStart {
                let capturesDraftContext = end > dflashContextStart
                let output = model.forward(
                    chunk,
                    cache: cache,
                    captureLayerIndices: capturesDraftContext
                        ? Set(dflash.config.dflash.targetLayerIDs)
                        : [],
                    lastPositionOnly: true
                )
                if capturesDraftContext {
                    let retainedStart = max(processed, dflashContextStart) - processed
                    let retainedHiddenStates = output.capturedHiddenStates.mapValues {
                        $0[0..., retainedStart..., 0...]
                    }
                    let combined = dflash.combineTargetHiddenStates(
                        retainedHiddenStates
                    )
                    dflash.appendTargetContext(combined, cache: dflashCache)
                    dflashCache.forEach { $0.evaluateStorage() }
                }
                chunkLogits = output.logits
            } else {
                chunkLogits = model.lastPositionLogits(chunk, cache: cache)
            }
            MLX.eval(chunkLogits)
            logits = chunkLogits
            processed = end
        }
        guard let logits else {
            throw LagunaError.generationFailed("Laguna prefill produced no logits.")
        }
        return LagunaPrefillResult(logits: logits, dflashCache: dflashCache)
    }

    func accumulateDFlashStats(_ stats: LagunaDFlashStats) {
        dflashRounds += stats.rounds
        dflashDraftedTokens += stats.draftedTokens
        dflashAcceptedDraftTokens += stats.acceptedDraftTokens
        dflashRejectedDraftTokens += stats.rejectedDraftTokens
        dflashFullAcceptanceRounds += stats.fullAcceptanceRounds
        dflashTargetVerificationForwards += stats.targetVerificationForwards
        dflashTargetRecoveryForwards += stats.targetRecoveryForwards
        dflashTargetFallbackForwards += stats.targetFallbackForwards
        dflashAdaptiveFallbacks += stats.adaptiveFallbacks
    }

}
