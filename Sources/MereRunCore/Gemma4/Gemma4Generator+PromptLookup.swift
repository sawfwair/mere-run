import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    enum PromptLookupBurstOutcome {
        /// No n-gram match; the caller proceeds with the normal lagged step.
        case notAttempted
        /// Burst ran; pendingToken/repetitionHistory/layerCaches rebuilt — caller continues the loop.
        case resumed
        /// EOS or token budget reached inside the burst — caller ends decode.
        case finished
    }

    /// One prompt-lookup speculation round inside the pipelined loop. Fires
    /// only when the trailing n-gram of the confirmed context recurs earlier
    /// (host-side scan, no GPU cost when it doesn't). The already-scheduled
    /// token is confirmed early — the burst's only pipeline cost — and
    /// anchors a serial-style batched verify of the lookup draft, after
    /// which the lagged loop resumes with rebuilt state. Verify sampling is
    /// the same sampler with the same prospective histories the plain loop
    /// would have used, so greedy outputs are token-identical.
    func promptLookupBurst(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        scheduledToken: MLXArray,
        promptTokens: [Int],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        generated: inout [Int],
        layerCaches: inout [Gemma4AttentionCache],
        pendingToken: inout MLXArray,
        repetitionHistory: inout MLXArray?,
        mtpStats: inout Gemma4MTPStats,
        promptLookupMisses: inout Int,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) -> PromptLookupBurstOutcome {
        let probe = Self.promptLookupDraft(
            context: promptTokens + generated,
            blockSize: Self.promptLookupBlockSize
        )
        guard !probe.isEmpty else { return .notAttempted }

        func emit(_ token: Int) {
            if let progressHandler {
                let piece = tokenizerAndTemplate.decode(token: token)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }
        }

        // Confirm the scheduled token; it becomes the verify anchor (the
        // serial loop's `next`): sampled and appended, forward not yet run.
        let anchor = scheduledToken.item(Int.self)
        if eosSet.contains(anchor) {
            return .finished
        }
        generated.append(anchor)
        emit(anchor)
        if generated.count >= tokenBudget {
            return .finished
        }

        var hostContext = promptTokens + generated
        let draftTokens = Self.promptLookupDraft(
            context: hostContext,
            blockSize: Self.promptLookupBlockSize
        )

        func resume(samplingFrom logits: MLXArray, contextBeforePending: [Int]) {
            let history = repetitionHistoryArray(promptTokens: contextBeforePending, config: generationConfig)
            let banMask = tokenBanMask(
                vocabularySize: logits.dim(-1),
                dtype: logits.dtype,
                tokens: generationConfig.bannedTokens
            )
            let sampled = sampledTokenArray(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokenIndices: history,
                banMask: banMask
            )
            MLX.asyncEval(sampled)
            pendingToken = sampled
            repetitionHistory = appendingRepetitionHistory(history, token: sampled, config: generationConfig)
        }

        guard !draftTokens.isEmpty else {
            // The anchor broke the match. Its forward has not run yet, so run
            // it and sample the next token — one synchronous step, then the
            // lag re-establishes. Counted as a miss: repeated anchor-breaks
            // mean the text repeats n-grams without repeating continuations.
            promptLookupMisses += 1
            let logits = model.forward(
                inputIds: MLXArray([Int32(anchor)]).reshaped(1, 1),
                cache: layerCaches
            )
            resume(samplingFrom: logits, contextBeforePending: hostContext)
            return .resumed
        }

        mtpStats.rounds += 1
        mtpStats.draftedTokens += draftTokens.count
        mtpStats.reason = "prompt-lookup speculation"

        let candidateCaches = forkLayerCaches(layerCaches)
        let candidateInput = MLXArray(([anchor] + draftTokens).map(Int32.init))
            .reshaped(1, draftTokens.count + 1)
        let candidate = model.forwardForSpeculation(
            inputIds: candidateInput,
            cache: candidateCaches
        )

        // Batched verify sampling with prospective histories — one graph,
        // one readback, identical math to the serial speculation block.
        let verifyBanMask = tokenBanMask(
            vocabularySize: candidate.logits.dim(-1),
            dtype: candidate.logits.dtype,
            tokens: generationConfig.bannedTokens
        )
        var verifySampleArrays: [MLXArray] = []
        verifySampleArrays.reserveCapacity(draftTokens.count + 1)
        var prospectiveHistory = hostContext
        for index in 0...draftTokens.count {
            verifySampleArrays.append(sampledTokenArray(
                logits: candidate.logits[0, index, 0...],
                config: generationConfig,
                previousTokenIndices: repetitionHistoryArray(
                    promptTokens: prospectiveHistory,
                    config: generationConfig
                ),
                banMask: verifyBanMask
            ))
            if index < draftTokens.count {
                prospectiveHistory.append(draftTokens[index])
            }
        }
        let stackedVerify = MLX.stacked(verifySampleArrays)
        MLX.eval(stackedVerify)
        let verifySamples = stackedVerify.asArray(Int32.self).map(Int.init)

        var accepted = 0
        var replacement: Int?
        for (index, draftToken) in draftTokens.enumerated() {
            guard verifySamples[index] == draftToken else {
                replacement = verifySamples[index]
                mtpStats.rejectedTokens += 1
                break
            }
            accepted += 1
        }
        promptLookupMisses = accepted == 0 ? promptLookupMisses + 1 : 0

        if accepted == draftTokens.count {
            for token in draftTokens {
                if eosSet.contains(token) {
                    return .finished
                }
                generated.append(token)
                hostContext.append(token)
                mtpStats.acceptedTokens += 1
                emit(token)
            }
            layerCaches = candidateCaches
            if generated.count >= tokenBudget {
                return .finished
            }
            // The bonus token was already sampled in the batched verify pass
            // (last position, full-draft history) — it becomes pendingToken.
            let bonus = verifySamples[draftTokens.count]
            pendingToken = MLXArray([Int32(bonus)])
            repetitionHistory = appendingRepetitionHistory(
                repetitionHistoryArray(promptTokens: hostContext, config: generationConfig),
                token: pendingToken,
                config: generationConfig
            )
            return .resumed
        }

        let acceptedPrefix = Array(draftTokens.prefix(accepted))
        for token in acceptedPrefix {
            if eosSet.contains(token) {
                return .finished
            }
            generated.append(token)
            hostContext.append(token)
            mtpStats.acceptedTokens += 1
            emit(token)
        }
        if generated.count >= tokenBudget {
            return .finished
        }
        guard let replacement else {
            return .finished
        }
        if eosSet.contains(replacement) {
            return .finished
        }
        generated.append(replacement)
        hostContext.append(replacement)
        emit(replacement)
        if generated.count >= tokenBudget {
            return .finished
        }

        // Rebuild real cache state through the replacement (the candidate
        // fork carries rejected draft positions) — same replay the serial
        // loop uses on partial accepts.
        let replacementCaches = forkLayerCaches(layerCaches)
        let replacementInput = MLXArray(([anchor] + acceptedPrefix + [replacement]).map(Int32.init))
            .reshaped(1, acceptedPrefix.count + 2)
        let replacementLogits = model.forward(
            inputIds: replacementInput,
            cache: replacementCaches
        )
        layerCaches = replacementCaches
        resume(samplingFrom: replacementLogits, contextBeforePending: hostContext)
        return .resumed
    }
}
