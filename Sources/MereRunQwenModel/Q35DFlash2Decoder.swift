import MLX
import MereRunTensor

/// Optional phase boundaries for synchronized diagnostics. The ordinary decode
/// path has no observer and retains its asynchronous graph scheduling.
package enum Q35DFlash2Event {
    case beginRound(Int)
    case drafted(MLXArray)
    case verificationSubmitted
    case verificationCompleted
    case accepted(Int)
    case finished([MLXArray])
}

package struct Q35DFlash2Stats {
    package var rounds = 0
    package var draftedTokens = 0
    package var acceptedTokens = 0
    package var recoveryForwards = 0
}

package enum Q35DFlash2Decoder {
    /// Target verification owns every emitted token. Proposals execute
    /// against disposable target state, and only confirmed captures enter the draft.
    package static func decode(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        sampling: Q35DFlash2Sampling? = nil,
        checkCancellation: () throws -> Void = {}, observer: ((Q35DFlash2Event) -> Void)? = nil,
        emit: (Int) -> Void
    ) throws -> Q35DFlash2Stats {
        var caches = layerCaches
        var generated = 0
        var stats = Q35DFlash2Stats()
        guard tokenBudget > 0 else { return stats }
        try checkCancellation()
        var history = sampling?.history.map { MLXArray($0.map(Int32.init)) }
        var anchor = sampling.map {
            Q35DFlash2Sampling.sample($0.distribution(initialLogits[0, -1, 0...], history)).item(Int.self)
        } ?? argMax(initialLogits[0, -1, 0...]).item(Int.self)
        let capture = Set(draft.config.dflash.targetLayerIDs)
        while generated < tokenBudget {
            try checkCancellation()
            if eosSet.contains(anchor) { break }
            emit(anchor)
            if let previous = history {
                history = concatenated([previous, MLXArray([Int32(anchor)])])
            }
            generated += 1
            if generated == tokenBudget { break }
            let count = min(draft.config.dflash.blockSize - 1, tokenBudget - generated)
            observer?(.beginRound(count))
            defer { observer?(.finished(draftCache.flatMap(\.state))) }
            let selection = draft.selection(
                anchor: anchor, count: count, cache: draftCache, target: model,
                temperature: sampling?.temperature ?? 0
            )
            let proposalIDs = selection.tokens
            observer?(.drafted(proposalIDs))
            stats.rounds += 1
            stats.draftedTokens += count
            let input = concatenated([MLXArray([Int32(anchor)]).reshaped(1, 1), proposalIDs], axis: 1)
            let inputCount = count + 1
            let candidateCaches = caches.map { $0?.fork() }
            let candidate = model.forward(
                input,
                cache: candidateCaches, targetVerify: true, captureLayerIndices: capture
            )
            let targetIDs: MLXArray
            let acceptance: MLXArray
            if let sampling {
                let verification = sampling.verify(logits: candidate.logits, selection: selection, history: history)
                targetIDs = verification.replacements
                acceptance = verification.accepted
            } else {
                targetIDs = argMax(candidate.logits, axis: -1).asType(.int32)
                acceptance = MLXArray.ones([count], dtype: .int32)
            }
            observer?(.verificationSubmitted)
            // Feed proposals directly into verification. One small readback
            // completes both graphs and carries the next anchor as a bonus.
            let tokens = concatenated([proposalIDs.reshaped(-1), acceptance, targetIDs.reshaped(-1)])
                .asArray(Int32.self).map(Int.init)
            observer?(.verificationCompleted)
            let proposals = Array(tokens.prefix(count))
            let acceptanceFlags = Array(tokens[count..<(2 * count)])
            let targetTokens = Array(tokens.dropFirst(2 * count))
            var accepted = 0
            for (position, proposal) in proposals.enumerated() {
                if sampling == nil ? proposal != targetTokens[position] : acceptanceFlags[position] == 0 { break }
                if eosSet.contains(proposal) {
                    stats.acceptedTokens += accepted
                    observer?(.accepted(accepted))
                    return stats
                }
                emit(proposal)
                generated += 1
                accepted += 1
            }
            stats.acceptedTokens += accepted
            if let previous = history, accepted > 0 {
                history = concatenated([previous, proposalIDs[0, 0..<accepted]])
            }
            observer?(.accepted(accepted))
            if generated == tokenBudget { break }
            let committed = accepted + 1
            let present = candidateCaches.compactMap { $0 }
            let canRestore = present.allSatisfy {
                $0.canRestoreVerificationPrefix(totalTokens: inputCount, tokenCount: committed)
            }
            let confirmed: Q35ForwardOutput
            if committed == inputCount || canRestore {
                for cache in present {
                    if committed < inputCount {
                        precondition(cache.restoreVerificationPrefix(totalTokens: inputCount, tokenCount: committed))
                    }
                    cache.commitVerification()
                }
                caches = candidateCaches
                confirmed = candidate
                anchor = targetTokens[committed - 1]
            } else {
                // Quantized KV implementations without prefix rollback replay
                // only the anchor and accepted prefix on the untouched base.
                let replay = [anchor] + Array(proposals.prefix(accepted))
                confirmed = model.forward(
                    MLXArray(replay.map(Int32.init)).reshaped(1, committed),
                    cache: caches, captureLayerIndices: capture
                )
                stats.recoveryForwards += 1
                anchor = sampling == nil
                    ? argMax(confirmed.logits[0, -1, 0...]).item(Int.self)
                    : targetTokens[committed - 1]
            }
            draft.appendTargetContext(
                confirmed.capturedHiddenStates.mapValues { $0[0..., 0..<committed, 0...] }, cache: draftCache
            )
        }
        return stats
    }
}
