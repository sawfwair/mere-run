import MLX
import MLXNN
import MereRunGemmaModel
import Foundation

package struct SOVDecodeResult {
    package let tokens: [Int]
    package let verificationRounds: Int
    package let acceptedDraftTokens: Int
    package let rejectedRounds: Int
    package let reusedVerificationRounds: Int
    package let acceptedLengths: [Int]
    package let emittedDraftTokens: Int
    package let proposedLengths: [Int]
}

/// Greedy verification commits accepted KV rows from an isolated target trial.
package enum SOVTranscribeDecoder {
    package static func decode(
        target: Gemma4TextCausalLM, embeddings: MLXArray, inputIDs: MLXArray,
        draft: SOVDFlashModel? = nil, maxTokens: Int, eosTokens: Set<Int>, prefillChunkSize: Int = 1024,
        prefillCompleted: (() -> Void)? = nil, progress: ((Int) -> Void)? = nil,
        pipelined: Bool = false, draftHead: Linear? = nil, reuseAttentionStorage: Bool = false,
        rollbackCache: Bool = false, draftTokenLimit: Int = 15, adaptiveConfidence: Float? = nil,
        draftTokenMapping: MLXArray? = nil,
        adaptiveAfterFirst: Bool = false,
        roundTiming: ((SOVDecodeRoundTiming) -> Void)? = nil,
        diagnostics: SOVDecodeDiagnostics? = nil
    ) throws -> SOVDecodeResult {
        guard maxTokens > 0, prefillChunkSize > 0, draftTokenLimit > 0, inputIDs.ndim == 2, inputIDs.dim(0) == 1,
              inputIDs.dim(1) > 0, embeddings.shape == [1, inputIDs.dim(1), target.config.hiddenSize],
              inputIDs.dim(1) + maxTokens <= target.config.maxPositionEmbeddings else {
            throw SOVTranscribeError.invalid("Audio, instruction, and output budget exceed the decoder context or have invalid shapes.")
        }
        let captured = Set(draft?.config.dflash.targetLayerIDs ?? [])
        var cache = target.makeAttentionCache()
        if rollbackCache {
            cache = cache.map { SOVRollbackKVCache(window: ($0 as? Gemma4SlidingKVCache)?.configuredMaxSize) }
        }
        for case let entry as Gemma4SlidingKVCache in cache { entry.reuseAttentionStorage = reuseAttentionStorage }
        let draftCache = draft?.makeCache() ?? []
        var logits: MLXArray = MLXArray.zeros([1, 1, target.config.vocabSize])
        for start in stride(from: 0, to: inputIDs.dim(1), by: prefillChunkSize) {
            try Task.checkCancellation()
            let end = min(start + prefillChunkSize, inputIDs.dim(1))
            let output = target.forwardEmbeddings(
                embeddings[0..., start..<end, 0...], inputIDs: inputIDs[0..., start..<end],
                cache: cache, captureLayerIndices: captured, lastLogitsOnly: true
            )
            logits = output.logits
            eval([logits] + Array(output.states.values))
            draft?.appendContext(output.states, cache: draftCache)
            evaluateGemma4CacheStorage(cache)
        }
        prefillCompleted?()
        diagnostics?.beginGeneration()
        var tokens: [Int] = []
        var rounds = 0
        var acceptedTotal = 0
        var rejected = 0
        var reused = 0
        var acceptedLengths: [Int] = []
        var proposedLengths: [Int] = []
        var emittedDraftTokens = 0
        var nextAnchor: Int?
        while tokens.count < maxTokens {
            try Task.checkCancellation()
            let anchor = nextAnchor ?? argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            nextAnchor = nil
            if eosTokens.contains(anchor) { break }
            tokens.append(anchor)
            progress?(tokens.count)
            if tokens.count == maxTokens { break }
            var count = min(draft.map { min(draftTokenLimit, $0.config.blockSize - 1) } ?? 0, maxTokens - tokens.count)
            var timing = SOVDecodeRoundTiming()
            func now() -> Double { roundTiming == nil ? 0 : ProcessInfo.processInfo.systemUptime }
            timing.round = rounds + 1
            timing.proposed = count
            timing.startUptime = now()
            var stageStarted = now()
            if count > 0 { try diagnostics?.beginRound(rounds + 1) }
            defer {
                diagnostics?.endRound()
                timing.endUptime = now()
                timing.wallSeconds = timing.endUptime - timing.startUptime
                if count > 0 { roundTiming?(timing) }
            }
            if let draft, count > 0 {
                diagnostics?.stage("SOVDraftGraph")
                let threshold = adaptiveAfterFirst && rounds == 0 ? nil : adaptiveConfidence
                let proposed = draft.proposal(anchor: anchor, count: count, target: target, cache: draftCache,
                    outputHead: draftHead, tokenMapping: draftTokenMapping, confidence: threshold != nil)
                var proposalIDs = proposed.ids
                if let threshold, let confidence = proposed.confidence {
                    eval(proposalIDs, confidence)
                    let scores = confidence.asArray(Float.self)
                    count = max(1, scores.prefix { $0 >= threshold }.count)
                    proposalIDs = proposalIDs[..<count]
                }
                timing.proposed = count
                timing.draftGraphSeconds = now() - stageStarted
                stageStarted = now()
                diagnostics?.stage("SOVDraftSubmit")
                // Start draft execution while the host constructs the target graph.
                // Both graphs stay on the request's stream; dependencies retain their arrays.
                if pipelined { asyncEval(proposalIDs) }
                let hostProposals = pipelined ? nil : proposalIDs.asArray(Int.self)
                timing.draftMaterializeSeconds = now() - stageStarted
                try Task.checkCancellation()
                stageStarted = now()
                diagnostics?.stage("SOVVerificationGraph")
                let candidateIDs = pipelined
                    ? concatenated([MLXArray([Int32(anchor)]), proposalIDs]).reshaped(1, count + 1)
                    : MLXArray(([anchor] + hostProposals!).map(Int32.init)).reshaped(1, count + 1)
                let trialCache: [Gemma4AttentionCache]
                if rollbackCache {
                    for case let entry as SOVRollbackKVCache in cache { entry.beginTrial(count: count + 1) }
                    trialCache = cache
                } else { trialCache = cache.map { $0.fork() } }
                let verification = target.forwardEmbeddings(
                    target.inputEmbeddings(for: candidateIDs), inputIDs: candidateIDs,
                    cache: trialCache, captureLayerIndices: captured
                )
                let predictionIDs = argMax(verification.logits, axis: -1).asType(.int32).reshaped(count + 1)
                timing.verificationGraphSeconds = now() - stageStarted
                stageStarted = now()
                diagnostics?.stage("SOVPredictionWait")
                let proposals: [Int]
                let predictions: [Int]
                if pipelined {
                    let both = concatenated([proposalIDs, predictionIDs]).asArray(Int.self)
                    proposals = Array(both.prefix(count))
                    predictions = Array(both.dropFirst(count))
                } else {
                    proposals = hostProposals!
                    predictions = predictionIDs.asArray(Int.self)
                }
                timing.predictionMaterializeSeconds = now() - stageStarted
                stageStarted = now()
                diagnostics?.stage("SOVCommit")
                try Task.checkCancellation()
                var accepted = 0
                while accepted < proposals.count && proposals[accepted] == predictions[accepted] { accepted += 1 }
                rounds += 1
                proposedLengths.append(count)
                acceptedTotal += accepted
                acceptedLengths.append(accepted)
                timing.accepted = accepted
                var stopped = false
                for token in proposals.prefix(accepted) {
                    if eosTokens.contains(token) { stopped = true; break }
                    tokens.append(token)
                    emittedDraftTokens += 1
                    progress?(tokens.count)
                }
                if stopped || tokens.count == maxTokens {
                    timing.commitSeconds = now() - stageStarted
                    break
                }
                if accepted == proposals.count {
                    if rollbackCache { for case let entry as SOVRollbackKVCache in cache { entry.commit(count: count + 1) } }
                    cache = trialCache
                    logits = verification.logits[0..., (-1)..., 0...]
                    draft.appendContext(verification.states, cache: draftCache, evaluate: !pipelined)
                    if pipelined { nextAnchor = predictions[count] }
                } else {
                    rejected += 1
                    let committedCount = accepted + 1
                    let committed: Bool
                    if rollbackCache {
                        for case let entry as SOVRollbackKVCache in cache { entry.commit(count: committedCount) }
                        committed = true
                    } else { committed = commitVerifiedPrefix(cache, from: trialCache, proposedCount: count + 1, committedCount: committedCount) }
                    if committed {
                        reused += 1
                        logits = verification.logits[0..., accepted..<(accepted + 1), 0...]
                        draft.appendContext(verification.states.mapValues { $0[0..., ..<committedCount, 0...] }, cache: draftCache, evaluate: !pipelined)
                        if pipelined { nextAnchor = predictions[accepted] }
                    } else {
                        // A window smaller than the proposal can discard new rows. Replay
                        // only in that case, from the untouched base cache.
                        let committed = Array(([anchor] + proposals).prefix(committedCount))
                        let committedIDs = MLXArray(committed.map(Int32.init)).reshaped(1, committed.count)
                        let recovery = target.forwardEmbeddings(
                            target.inputEmbeddings(for: committedIDs), inputIDs: committedIDs,
                            cache: cache, captureLayerIndices: captured, lastLogitsOnly: true
                        )
                        logits = recovery.logits
                        draft.appendContext(recovery.states, cache: draftCache)
                    }
                }
                timing.commitSeconds = now() - stageStarted
            } else {
                let ids = MLXArray([Int32(anchor)]).reshaped(1, 1)
                logits = target.forwardEmbeddings(target.inputEmbeddings(for: ids), inputIDs: ids, cache: cache,
                                                  lastLogitsOnly: true).logits
            }
            stageStarted = now()
            diagnostics?.stage("SOVTailWait")
            if !pipelined || nextAnchor == nil {
                eval(logits)
                evaluateGemma4CacheStorage(cache)
            }
            timing.tailEvaluateSeconds = now() - stageStarted
        }
        try Task.checkCancellation()
        try diagnostics?.finish()
        return SOVDecodeResult(tokens: tokens, verificationRounds: rounds, acceptedDraftTokens: acceptedTotal,
                               rejectedRounds: rejected, reusedVerificationRounds: reused,
                               acceptedLengths: acceptedLengths, emittedDraftTokens: emittedDraftTokens, proposedLengths: proposedLengths)
    }
}
