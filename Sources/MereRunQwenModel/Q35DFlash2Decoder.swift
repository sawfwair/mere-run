import Foundation
import MereRunEnvironment
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
    case targetOnlyCompleted(Double)
}

package struct Q35DFlash2Stats {
    package var rounds = 0
    package var draftedTokens = 0
    package var acceptedTokens = 0
    package var recoveryForwards = 0
    package var targetOnlyRounds = 0
    package var roundsByDepth: [Int] = []
    package var treeRounds = 0
    package var treeRoundsByNodeCount: [Int] = []
    package var treeRoundSecondsByNodeCount: [Double?] = []
}

package enum Q35DFlash2Decoder {
    /// Target verification owns every emitted token. Proposals execute
    /// against disposable target state, and only confirmed captures enter the draft.
    package static func decode(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        sampling: Q35DFlash2Sampling? = nil, positionSampler: Q35PositionSampler? = nil,
        costs: Q35DFlash2Costs? = nil,
        initialHistory: [Int] = [],
        checkCancellation: () throws -> Void = {}, observer: ((Q35DFlash2Event) -> Void)? = nil,
        stopAfterEmission: (() -> Bool)? = nil,
        emit: (Int) -> Void
    ) throws -> Q35DFlash2Stats {
        #if os(macOS)
        if #available(macOS 26.0, *), stopAfterEmission == nil, let positionSampler, let sampling, costs == nil,
           MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_SAMPLED_CHAIN", default: false),
           Q35DFlash2TreePolicy.eligible(model: model, caches: layerCaches, sampled: false),
           let stats = try decodeNativeSampledChain(model: model, draft: draft, draftCache: draftCache,
               initialLogits: initialLogits, layerCaches: layerCaches, eosSet: eosSet, tokenBudget: tokenBudget,
               rule: positionSampler, sampling: sampling, checkCancellation: checkCancellation, emit: emit) {
            return stats
        }
        if #available(macOS 26.0, *), stopAfterEmission == nil, let positionSampler, sampling != nil,
           positionSampler.treeMode != nil,
           Q35DFlash2TreePolicy.eligible(model: model, caches: layerCaches, sampled: false),
           let stats = try decodePositionTree(model: model, draft: draft, draftCache: draftCache,
               initialLogits: initialLogits, layerCaches: layerCaches, eosSet: eosSet, tokenBudget: tokenBudget,
               sampler: positionSampler, checkCancellation: checkCancellation, observer: observer, emit: emit) {
            return stats
        }
        #endif
        if stopAfterEmission == nil,
           Q35DFlash2TreePolicy.eligible(model: model, caches: layerCaches, sampled: sampling != nil) {
            return try decodeTree(model: model, draft: draft, draftCache: draftCache,
                initialLogits: initialLogits, layerCaches: layerCaches, eosSet: eosSet, tokenBudget: tokenBudget,
                initialHistory: initialHistory,
                checkCancellation: checkCancellation, observer: observer, emit: emit)
        }
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
        let requestedProposals = Q35DFlash2VerificationPolicy.proposalLimit(
            model: model, trainedBlockSize: draft.config.dflash.blockSize, sampled: sampling != nil
        )
        let maximumProposals = Q35DFlash2VerificationPolicy.limitForCaches(
            requestedProposals, trainedBlockSize: draft.config.dflash.blockSize, caches: layerCaches
        )
        let effectiveCosts = costs?.limited(to: maximumProposals)
        var policy = effectiveCosts.map { Q35DFlash2Policy(costs: $0) }
        stats.roundsByDepth = Array(repeating: 0, count: maximumProposals + 1)
        let maintainContext = policy?.supportsTargetOnlyProbes ?? false
        var targetOnly = false
        while generated < tokenBudget {
            try checkCancellation()
            if eosSet.contains(anchor) { break }
            emit(anchor)
            if stopAfterEmission?() == true { break }
            if let previous = history {
                history = concatenated([previous, MLXArray([Int32(anchor)])])
            }
            generated += 1
            if generated == tokenBudget { break }
            let offered = min(maximumProposals, tokenBudget - generated)
            let count = targetOnly ? 0 : (policy?.draftDepth(offeredDepth: offered) ?? offered)
            if count == 0 {
                // Keep the chosen target backend. Switching to the generator's
                // reference route here would change its numerical contract.
                targetOnly = !maintainContext
                let started = observer.map { _ in DispatchTime.now().uptimeNanoseconds }
                let output = model.forward(
                    MLXArray([Int32(anchor)]).reshaped(1, 1), cache: caches,
                    captureLayerIndices: maintainContext ? capture : []
                )
                anchor = sampling.map {
                    Q35DFlash2Sampling.sample($0.distribution(output.logits[0, -1, 0...], history)).item(Int.self)
                } ?? argMax(output.logits[0, -1, 0...]).item(Int.self)
                if maintainContext {
                    draft.appendTargetContext(output.capturedHiddenStates, cache: draftCache)
                    policy?.recordTargetOnly()
                }
                stats.targetOnlyRounds += 1
                stats.roundsByDepth[0] += 1
                if let started {
                    observer?(.targetOnlyCompleted(Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000))
                }
                continue
            }
            observer?(.beginRound(count))
            defer { observer?(.finished(draftCache.flatMap(\.state))) }
            let selection = draft.selection(
                anchor: anchor, count: count, cache: draftCache, target: model,
                temperature: sampling?.temperature ?? 0
            )
            let proposalIDs = selection.tokens
            observer?(.drafted(proposalIDs))
            stats.rounds += 1
            stats.roundsByDepth[count] += 1
            stats.draftedTokens += count
            let input = concatenated([MLXArray([Int32(anchor)]).reshaped(1, 1), proposalIDs], axis: 1)
            let inputCount = count + 1
            let candidateCaches = caches.map { $0?.fork() }
            let candidate = model.forward(
                input, cache: candidateCaches, targetVerify: true, captureLayerIndices: capture
            )
            let targetIDs: MLXArray
            let acceptance: MLXArray
            if let sampling {
                let verification = sampling.verify(logits: candidate.logits, selection: selection, history: history)
                targetIDs = verification.nextToken
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
                if stopAfterEmission?() == true {
                    stats.acceptedTokens += accepted
                    observer?(.accepted(accepted))
                    return stats
                }
            }
            stats.acceptedTokens += accepted
            if let previous = history, accepted > 0 {
                history = concatenated([previous, proposalIDs[0, 0..<accepted]])
            }
            observer?(.accepted(accepted))
            policy?.record(proposed: count, accepted: accepted)
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
                anchor = sampling == nil ? targetTokens[accepted] : targetTokens[0]
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
                    : targetTokens[0]
            }
            draft.appendTargetContext(
                confirmed.capturedHiddenStates.mapValues { $0[0..., 0..<committed, 0...] }, cache: draftCache
            )
        }
        return stats
    }
}
