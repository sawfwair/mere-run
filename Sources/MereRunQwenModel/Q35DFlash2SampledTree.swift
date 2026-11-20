#if os(macOS)
import Foundation
import MLX

@available(macOS 26.0, *)
extension Q35DFlash2Decoder {
    static func decodePositionTree(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        sampler: Q35PositionSampler, checkCancellation: () throws -> Void, observer: ((Q35DFlash2Event) -> Void)?,
        emit: (Int) -> Void
    ) throws -> Q35DFlash2Stats? {
        guard tokenBudget > 1, initialLogits.dtype == .bfloat16, let head = model.lmHead,
              let prefix = layerCaches.compactMap({ entry -> Int? in
                  if case .full(let cache)? = entry { return cache.offset }
                  return nil
              }).first, prefix > 0,
              let proposer = try draft.makeNativeDraftSession(target: model, cache: draftCache) as? Q35NativeDraftProgram.Session,
              let target = try model.model.nativeExecutionTarget(head: head, capacity: prefix + tokenBudget + 16,
                  capture: Set(draft.config.dflash.targetLayerIDs), importingPrompt: true) else { return nil }
        try target.importPrompt(from: layerCaches, count: prefix)
        var published = false
        defer {
            if !published {
                try? target.publish(to: layerCaches)
                try? proposer.finish(draftCache)
            }
        }
        var thinking = sampler.thinkingLimit
        var sampler = sampler
        var order: Q35SerialTreeKeys.Order? = sampler.treeMode == .vocabulary ? .vocabulary : nil
        #if DEBUG || MERERUN_RELEASE_REPLAY
        switch ProcessInfo.processInfo.environment["MERERUN_Q38_SERIAL_TREE_DRAWS"] {
        case "1": order = .sorted
        case "vocabulary": order = .vocabulary
        default: break
        }
        #endif
        if let order {
            sampler.serialKeys = Q35SerialTreeKeys(seed: sampler.seed, position: prefix, order: order)
            sampler.serialKeys?.prepare(position: prefix, count: 1)
        }
        let initial = initialLogits[0, -1, 0...].view(dtype: .uint16).asArray(UInt16.self)
        var anchor = initial.withUnsafeBufferPointer { sampler.choose($0, position: prefix) }
        var generated = 0, stats = Q35DFlash2Stats()
        let serial = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_TREE_SERIAL"] == "1"
        let blockDepth = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_DRAFT_DEPTH"] == "15"
            ? 15 : draft.config.dflash.blockSize - 1
        stats.treeRoundsByNodeCount = Array(repeating: 0, count: 16)
        while generated < tokenBudget && !eosSet.contains(anchor) {
            try checkCancellation()
            #if MERERUN_RELEASE_REPLAY
            if serial, let trace = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_DRAFT_TRACE"] {
                let file = URL(fileURLWithPath: trace).appendingPathComponent(String(prefix + generated) + ".token")
                try String(anchor).write(to: file, atomically: false, encoding: .utf8)
            }
            #endif
            emit(anchor)
            if thinking?.finished(after: anchor, at: generated) == true { thinking = nil }
            generated += 1
            if generated == tokenBudget { break }
            let remaining = tokenBudget - generated
            let shape = Q35DFlash2TreePolicy.proposalDimensions(remaining: remaining)
            observer?(.beginRound(shape.nodes))
            sampler.serialKeys?.prepare(position: prefix + generated, count: max(shape.depth, shape.nodes) + 1)
            let offered = try proposer.propose(anchor: anchor, depth: min(blockDepth, shape.depth), nodes: shape.nodes,
                                            sampling: sampler, position: prefix + generated)
            let tree = serial ? offered.prefix(nodes: 0) : offered
            observer?(.drafted(MLXArray(tree.tokens.dropFirst().map(Int32.init))))
            observer?(.verificationSubmitted)
            let children = thinking?.childBudget(remaining: remaining, generated: generated) ?? remaining
            let round = try target.verifySampledTree(tree, budget: 1 + children, eos: eosSet,
                                                     sampler: sampler, position: prefix + generated)
            observer?(.verificationCompleted)
            observer?(.accepted(round.count - 1))
            for row in round.path.dropFirst() {
                let token = tree.tokens[row]
                emit(token)
                if thinking?.finished(after: token, at: generated) == true { thinking = nil }
                generated += 1
            }
            stats.rounds += 1
            stats.treeRounds += 1
            stats.treeRoundsByNodeCount[tree.tokens.count - 1] += 1
            stats.draftedTokens += tree.tokens.count - 1
            stats.acceptedTokens += round.count - 1
            try proposer.stageNative(round.taps, count: round.count)
            observer?(.finished([]))
            // Each forced token becomes a verified root, keeping target and draft state in sync.
            anchor = thinking?.forcedToken(at: generated) ?? round.following!
        }
        try target.publish(to: layerCaches)
        try proposer.finish(draftCache)
        published = true
        return stats
    }
}
#endif
