import Foundation
import MereRunKVCache
import MereRunTensor
import MLX

private protocol Q35ResidentTreeRound {
    func verify(_ proposal: DFlash2TreeProposal, budget: Int, eos: Set<Int>) throws -> (path: [Int], following: Int)
    func publish(to caches: [Q35LayerCache?]) throws
}

#if os(macOS)
@available(macOS 26.0, *)
private struct Q35ResidentTreeDecode: Q35ResidentTreeRound {
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_RESIDENT_ROUNDS")
    static var enabled: Bool { enabledFlag.enabled }
    let target: Q35NativeVerification.Session
    let draft: Q35NativeDraftProgram.Session

    func verify(_ proposal: DFlash2TreeProposal, budget: Int, eos: Set<Int>) throws -> (path: [Int], following: Int) {
        let round = try target.verify(proposal, budget: budget, eos: eos)
        try draft.stageNative(round.taps, count: round.count)
        return (round.path, round.following!)
    }

    func publish(to caches: [Q35LayerCache?]) throws { try target.publish(to: caches) }
}
#endif

protocol Q35NativeDraftSession {
    func propose(anchor: Int, depth: Int, nodes: Int) throws -> DFlash2TreeProposal
    func stage(_ output: Q35GreedyTreeOutput) throws
    func finish(_ caches: [Q35DFlash2Cache]) throws
}

extension Q35Model {
    package func forwardTree(_ tokens: [Int], tree: Q35VerificationTree, cache: [Q35LayerCache?],
                             captureLayerIndices: Set<Int> = []) -> Q35ForwardOutput {
        precondition(tokens.count == tree.count && !config.textConfig.isQwen4Exp && !config.textConfig.usesMoE)
        return Q35TreeVerificationContext.$current.withValue(tree) {
            forward(MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count), cache: cache,
                    targetVerify: true, captureLayerIndices: captureLayerIndices, verificationTokens: tokens)
        }
    }

    package func forwardGreedyTree(_ tokens: [Int], tree: Q35VerificationTree, cache: [Q35LayerCache?],
                                   captureLayerIndices: Set<Int>, budget: Int, eos: Set<Int>,
                                   submitted: () -> Void = {}, completed: () -> Void = {}) -> Q35GreedyTreeOutput {
        precondition(tokens.count == tree.count && budget > 0 && !config.textConfig.isQwen4Exp && !config.textConfig.usesMoE)
        let rows = Q35DFlash2VerificationPolicy.rowLimit(config: config,
            backend: (lmHead as? PortableQuantizedLinear)?.effectiveVerificationBackend)
        #if os(macOS)
        let request = Q35NativeVerificationPolicy.greedyTail
            ? lmHead.map { Q35GreedyTreeRequest(head: $0, tokens: tokens, budget: budget, eos: eos) } : nil
        #else
        let request: Q35GreedyTreeRequest? = nil
        #endif
        return Q35TreeVerificationContext.$current.withValue(tree) {
            QuantizedVerificationContext.$maximumM5Rows.withValue(rows) {
                let output = model.forward(MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count),
                    cache: cache, targetVerify: true, captureLayerIndices: captureLayerIndices, greedy: request,
                    verificationTokens: tokens)
                submitted()
                if let selection = output.greedySelection {
                    completed()
                    var result = Q35GreedyTreeOutput(selection: selection, taps: output.capturedHiddenStates,
                        attention: output.acceptedAttention)
                    #if os(macOS)
                    result.nativeDraftTaps = output.nativeDraftTaps
                    #endif
                    return result
                }
                let picks = argMax(logits(from: output.hidden), axis: -1).asArray(Int32.self).map(Int.init)
                completed()
                let path = tree.matchingPath(tokens: tokens, targetPicks: picks, budget: budget, eos: eos)
                return Q35GreedyTreeOutput(selection: Q35GreedyTreeSelection(path: path, following: picks[path.last!], picks: picks),
                    taps: output.capturedHiddenStates.mapValues { Q35VerificationTree.take($0, rows: path) }, attention: [:])
            }
        }
    }
}

package enum Q35DFlash2TreePolicy {
    @TaskLocal package static var pricedNodesEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_TREE_PRICED_NODES"] == "1"
    private static let calibratedScoresFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_TREE_CALIBRATED_SCORES")
    @TaskLocal package static var calibratedScoresEnabledOverride: Bool?
    package static var calibratedScoresEnabled: Bool { calibratedScoresEnabledOverride ?? calibratedScoresFlag.enabled }
    @TaskLocal package static var contextCopiesEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_TREE_CONTEXT_COPIES"] == "1"
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_TREE")
    @TaskLocal package static var enabledOverride: Bool?
    package static var enabled: Bool { enabledOverride ?? enabledFlag.enabled }
    private static let adaptiveWidthEnabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_TREE_ADAPTIVE")
    @TaskLocal package static var adaptiveWidthEnabledOverride: Bool?
    package static var adaptiveWidthEnabled: Bool { adaptiveWidthEnabledOverride ?? adaptiveWidthEnabledFlag.enabled }
    @TaskLocal package static var maximumNodes: Int = {
        switch ProcessInfo.processInfo.environment["MERERUN_Q38_DFLASH2_TREE_NODES"] {
        case "3": return 3
        case "7": return 7
        default: return 15
        }
    }()

    static func calibratedScoresDefault(environment: [String: String]) -> Bool {
        environment["MERERUN_Q38_TREE_CALIBRATED_SCORES"] == "1"
    }

    static func proposalDimensions(remaining: Int, nodeLimit: Int? = nil) -> (depth: Int, nodes: Int) {
        let limit = nodeLimit ?? maximumNodes
        precondition(remaining > 0 && [3, 7, 15].contains(limit))
        let depth = min(limit, remaining)
        let nodes = depth < 4 ? min(limit, (1 << (depth + 1)) - 2) : limit
        return (depth, nodes)
    }

    static func eligible(model: Q35Model, caches: [Q35LayerCache?], sampled: Bool) -> Bool {
        guard enabled, !sampled, !model.config.textConfig.isQwen4Exp, !model.config.textConfig.usesMoE,
              Q35DFlash2VerificationPolicy.rowLimit(config: model.config,
                backend: (model.lmHead as? PortableQuantizedLinear)?.effectiveVerificationBackend) == 16 else { return false }
        return caches.allSatisfy { entry in
            switch entry {
            case .linear(let cache): return cache.pleConvState == nil && cache.pleTokenContext == nil
            case .full(let cache): return cache is KVCacheSimple
            case nil: return false
            }
        }
    }
}

extension Q35DFlash2Decoder {
    package static func decodeTree(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        initialHistory: [Int] = [],
        checkCancellation: () throws -> Void = {}, observer: ((Q35DFlash2Event) -> Void)? = nil,
        emit: (Int) -> Void
    ) throws -> Q35DFlash2Stats {
        var stats = Q35DFlash2Stats()
        stats.treeRoundsByNodeCount = Array(repeating: 0, count: 16)
        guard tokenBudget > 0 else { return stats }
        var anchor = argMax(initialLogits[0, -1, 0...]).item(Int.self)
        let pipeline = Q35TreePipeline.enabled && Device.defaultDevice().deviceType == .gpu
            && model.config.textConfig.linearKeyHeadDim.isMultiple(of: 32) ? Q35TreePipeline.Frame() : nil
        defer { pipeline?.finish() }
        let nativeDraft = pipeline == nil && observer == nil
            ? try draft.makeNativeDraftSession(target: model, cache: draftCache) : nil
        var generated = 0
        let capture = Set(draft.config.dflash.targetLayerIDs)
        var resident: (any Q35ResidentTreeRound)?
        #if os(macOS)
        if #available(macOS 26.0, *), Q35ResidentTreeDecode.enabled, Q35NativeVerificationPolicy.directEmbedding,
           let draftSession = nativeDraft as? Q35NativeDraftProgram.Session, let head = model.lmHead,
           eosSet.count <= 128, tokenBudget > 1,
           let prefix = layerCaches.compactMap({ entry -> Int? in
               guard case .full(let cache)? = entry else { return nil }
               return cache.offset
           }).first, prefix > 0,
           let target = try model.model.nativeExecutionTarget(head: head, capacity: prefix + tokenBudget + 16,
                                                              capture: capture, importingPrompt: true) {
            try target.importPrompt(from: layerCaches, count: prefix)
            resident = Q35ResidentTreeDecode(target: target, draft: draftSession)
        }
        #endif
        var completed = false
        defer {
            if !completed {
                try? resident?.publish(to: layerCaches)
                try? nativeDraft?.finish(draftCache)
            }
        }
        var pricing = Q35DFlash2TreePolicy.pricedNodesEnabled && Q35DFlash2TreePolicy.maximumNodes == 15
            ? Q35DFlash2TreePricing() : nil
        var copyIndex = Q35DFlash2TreePolicy.contextCopiesEnabled ? Q35ContextCopyIndex(tokens: initialHistory) : nil
        var widthPolicy = pricing == nil && Q35DFlash2TreePolicy.adaptiveWidthEnabled && Q35DFlash2TreePolicy.maximumNodes == 15
            ? Q35DFlash2TreeCostPolicy() : nil
        while generated < tokenBudget && !eosSet.contains(anchor) {
            try checkCancellation()
            let started = widthPolicy == nil && pricing == nil ? 0 : DispatchTime.now().uptimeNanoseconds
            emit(anchor)
            copyIndex?.append([anchor])
            generated += 1
            if generated == tokenBudget { break }
            let limit = widthPolicy?.nodeLimit(remaining: tokenBudget - generated)
            let (depth, nodes) = Q35DFlash2TreePolicy.proposalDimensions(
                remaining: tokenBudget - generated, nodeLimit: limit
            )
            observer?(.beginRound(nodes))
            defer { observer?(.finished(draftCache.flatMap(\.state))) }
            var proposal = try nativeDraft?.propose(anchor: anchor, depth: depth, nodes: nodes)
                ?? draft.tree(anchor: anchor, depth: depth, nodeBudget: nodes, cache: draftCache, target: model)
            if let copyIndex {
                proposal = proposal.mergingCopy(copyIndex.continuation(limit: nodes), maximumNodes: nodes)
            }
            let fullProposal = depth == nodes && proposal.tokens.count - 1 == nodes
            if pricing != nil { proposal = pricing!.select(proposal, remaining: tokenBudget - generated) }
            let tree = try Q35VerificationTree(parents: proposal.parents)
            observer?(.drafted(MLXArray(proposal.tokens.dropFirst().map(Int32.init))))
            let candidates = resident == nil ? layerCaches.map { $0?.fork() } : []
            let budget = tokenBudget - generated + 1
            #if os(macOS)
            let greedyTail = Q35NativeVerificationPolicy.enabled && Q35NativeVerificationPolicy.greedyTail
            let deferDraftContext = greedyTail && Q35NativeVerificationPolicy.deferDraftContext
            #else
            let greedyTail = false
            let deferDraftContext = false
            #endif
            let path: [Int], following: Int
            if let resident {
                (path, following) = try resident.verify(proposal, budget: budget, eos: eosSet)
            } else if let pipeline {
                let output = model.forwardTree(proposal.tokens, tree: tree, cache: candidates, captureLayerIndices: capture)
                observer?(.verificationSubmitted)
                let picks = argMax(output.logits, axis: -1)
                let prefix = layerCaches.compactMap { cache -> Int? in
                    if case .full(let kv)? = cache { return kv.offset }
                    return nil
                }.first ?? 0
                let match = Q35TreePipeline.match(picks: picks, proposal: proposal,
                    budget: budget, eos: eosSet, prefix: prefix)
                let gathered = output.capturedHiddenStates.mapValues { MLX.take($0, match.path, axis: 1) }
                MLX.asyncEval([match.host] + Array(gathered.values))
                let prepared = pipeline.prepare(caches: candidates, tree: tree, match: match, prefix: prefix)
                MLX.asyncEval(prepared.flatMap(\.state))
                let metadata = match.host.asArray(Int32.self)
                let count = Int(metadata[0])
                path = metadata.dropFirst(2).prefix(count).map(Int.init)
                following = Int(metadata[1])
                observer?(.verificationCompleted)
                for (base, state) in zip(layerCaches, prepared) { state.publish(to: base!, path: path) }
                draft.appendTargetContext(gathered.mapValues { $0[0..., 0..<count, 0...] }, cache: draftCache)
                // Target-only state lives on the keep stream. The next draft
                // reads confirmed taps, so it need not join those state writes.
                MLX.asyncEval(draftCache.flatMap(\.state))
            } else {
                let result: Q35GreedyTreeOutput
                if greedyTail {
                    result = model.forwardGreedyTree(proposal.tokens, tree: tree, cache: candidates,
                        captureLayerIndices: capture, budget: budget, eos: eosSet,
                        submitted: { observer?(.verificationSubmitted) }, completed: { observer?(.verificationCompleted) })
                } else {
                    let output = model.forwardTree(proposal.tokens, tree: tree, cache: candidates, captureLayerIndices: capture)
                    observer?(.verificationSubmitted)
                    let targetPicks = argMax(output.logits, axis: -1).asArray(Int32.self).map(Int.init)
                    observer?(.verificationCompleted)
                    let accepted = tree.matchingPath(tokens: proposal.tokens, targetPicks: targetPicks, budget: budget, eos: eosSet)
                    result = Q35GreedyTreeOutput(selection: Q35GreedyTreeSelection(path: accepted,
                        following: targetPicks[accepted.last!], picks: targetPicks),
                        taps: output.capturedHiddenStates.mapValues { Q35VerificationTree.take($0, rows: accepted) }, attention: [:])
                }
                path = result.selection.path
                following = result.selection.following
                result.commit(candidates: candidates, to: layerCaches)
                let targetState = layerCaches.compactMap { $0 }.flatMap(\.pendingTreeState)
                if let nativeDraft {
                    try nativeDraft.stage(result)
                    MLX.asyncEval(targetState)
                } else {
                    draft.appendTargetContext(result.taps, cache: draftCache)
                    // The next proposal's attention can schedule the lazy append.
                    MLX.asyncEval(deferDraftContext ? targetState : targetState + draftCache.flatMap(\.state))
                }
            }
            stats.rounds += 1
            stats.treeRounds += 1
            stats.draftedTokens += tree.count - 1
            stats.acceptedTokens += path.count - 1
            stats.treeRoundsByNodeCount[tree.count - 1] += 1
            observer?(.accepted(path.count - 1))
            copyIndex?.append(path.dropFirst().map { proposal.tokens[$0] })
            for row in path.dropFirst() { emit(proposal.tokens[row]); generated += 1 }
            anchor = following
            if pricing != nil {
                pricing?.observe(proposal, acceptedPath: path)
                if fullProposal {
                    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
                    pricing?.record(nodes: tree.count - 1, seconds: seconds)
                    stats.treeRoundSecondsByNodeCount = pricing!.prices
                }
            } else if widthPolicy != nil, depth == nodes, tree.count - 1 == nodes {
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
                widthPolicy?.record(nodes: nodes, confirmedTokens: path.count, seconds: seconds)
                stats.treeRoundSecondsByNodeCount = widthPolicy!.secondsByNodeCount
            }
        }
        try resident?.publish(to: layerCaches)
        completed = true
        try nativeDraft?.finish(draftCache)
        return stats
    }
}
