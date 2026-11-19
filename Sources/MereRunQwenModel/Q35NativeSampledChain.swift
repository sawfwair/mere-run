#if os(macOS)
import Foundation
import MLX
import MereRunTensor

struct Q35SampledProposal {
    let tokens: [Int]
    let candidates: [Int32]
    let probabilities: [Float]

    init(_ selection: DFlash2Selection) {
        tokens = selection.tokens.asArray(Int32.self).map(Int.init)
        candidates = selection.candidateTokenIds.asArray(Int32.self)
        probabilities = selection.candidateProbabilities!.asArray(Float.self)
    }

    init(candidates: [Int32], scores: [Float], depth: Int, noise: [Float]) {
        self.candidates = candidates
        let width = candidates.count / depth
        var tokens: [Int] = [], probabilities: [Float] = []
        var parent = 0
        for row in 0..<depth {
            let first = (row * width + parent) * width
            let weights = scores[first..<(first + width)].map { exp($0) }
            let total = weights.reduce(0, +)
            let p = weights.map { $0 / total }
            probabilities.append(contentsOf: p)
            var largest = -Float.infinity
            for rank in 0..<width {
                let choice = log(p[rank]) + noise[row * width + rank]
                if choice > largest { largest = choice; parent = rank }
            }
            tokens.append(Int(candidates[row * width + parent]))
        }
        self.tokens = tokens
        self.probabilities = probabilities
    }
}

#if MERERUN_RELEASE_REPLAY
private struct Q35ChainCoverageRecord: Encodable {
    let anchor: Int
    let tokens: [Int]
    let candidates: [Int32]
    let probabilities: [Float]
    let uniforms: [Float]
    let keys: [UInt32]
    let accepted: Int
    let following: Int
}
#endif

@available(macOS 26.0, *)
extension Q35DFlash2Decoder {
    static func decodeNativeSampledChain(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        rule: Q35PositionSampler, sampling: Q35DFlash2Sampling,
        checkCancellation: () throws -> Void, emit: (Int) -> Void
    ) throws -> Q35DFlash2Stats? {
        guard tokenBudget > 1, let head = model.lmHead, sampling.history == nil,
              let prefix = layerCaches.compactMap({ cache -> Int? in
                  if case .full(let entry)? = cache { return entry.offset }
                  return nil
              }).first, prefix > 0,
              let target = try model.model.nativeExecutionTarget(head: head, capacity: prefix + tokenBudget + 16,
                  capture: Set(draft.config.dflash.targetLayerIDs), importingPrompt: true) else { return nil }
        let native = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_CHAIN_DRAFT"] == "1"
            ? try draft.makeNativeDraftSession(target: model, cache: draftCache) as? Q35NativeDraftProgram.Session : nil
        try target.importPrompt(from: layerCaches, count: prefix)
        var published = false
        defer {
            if !published {
                try? target.publish(to: layerCaches)
                try? native?.finish(draftCache)
            }
        }
        var anchor = Q35DFlash2Sampling.sample(sampling.distribution(initialLogits[0, -1, 0...], nil)).item(Int.self)
        let deviceRejection = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_GPU_HEAD"] == "1"
        let depth = draft.config.dflash.blockSize - 1
        var stats = Q35DFlash2Stats(), generated = 0
        stats.roundsByDepth = Array(repeating: 0, count: depth + 1)
        while generated < tokenBudget && !eosSet.contains(anchor) {
            try checkCancellation()
            emit(anchor); generated += 1
            if generated == tokenBudget { break }
            let count = min(depth, tokenBudget - generated)
            // Build the draft first: its categorical draws precede the acceptance
            // and replacement keys. Submit their independent graphs together.
            let selection: DFlash2Selection?
            let nativeNoise: MLXArray?
            if native != nil {
                selection = nil
                nativeNoise = stacked((0..<count).map { _ in MLXRandom.gumbel([16]) })
            } else {
                selection = draft.selection(anchor: anchor, count: count, cache: draftCache,
                    target: model, temperature: sampling.temperature)
                nativeNoise = nil
            }
            var uniforms: [MLXArray] = [], keys: [MLXArray] = []
            for _ in 0..<count { uniforms.append(MLXRandom.uniform()); keys.append(resolve()) }
            keys.append(resolve())
            let acceptance = stacked(uniforms), replacementKeys = stacked(keys)
            var pending = [acceptance, replacementKeys]
            if let selection {
                pending += [selection.tokens, selection.candidateTokenIds, selection.candidateProbabilities!]
            }
            if let nativeNoise { pending.append(nativeNoise) }
            let proposal: Q35SampledProposal
            if let native, let nativeNoise {
                MLX.asyncEval(pending)
                proposal = try native.sampledChain(anchor: anchor, depth: count, noise: nativeNoise)
            } else {
                MLX.eval(pending)
                proposal = Q35SampledProposal(selection!)
            }
            let draws = acceptance.asArray(Float.self)
            let gpuDraws = deviceRejection ? Q35RejectionDraws(proposal: proposal, uniforms: draws,
                keys: replacementKeys.asArray(UInt32.self)) : nil
            let round = try target.verifyReferenceChain([anchor] + proposal.tokens, eos: eosSet, rule: rule, rejection: gpuDraws) { words in
                let width = proposal.candidates.count / count
                for row in 0...count {
                    let p = rule.support(.init(start: words + row * 248_320, count: 248_320))
                    var masses = p.map(\.probability)
                    if row < count {
                        let start = row * width
                        let ids = proposal.candidates[start..<(start + width)]
                        let token = proposal.tokens[row]
                        let q = proposal.probabilities[ids.firstIndex(of: Int32(token))!]
                        let mass = p.first(where: { $0.id == token })?.probability ?? 0
                        if draws[row] < min(1, mass / max(q, .leastNonzeroMagnitude)) {
                            if eosSet.contains(token) { return (Array(0...row), token) }
                            continue
                        }
                        for (i, item) in p.enumerated() {
                            if let rank = ids.firstIndex(of: Int32(item.id)) {
                                masses[i] = max(0, masses[i] - proposal.probabilities[rank])
                            }
                        }
                    }
                    let total = masses.reduce(0, +)
                    let probabilities = total > 0 ? masses.map { $0 / total } : p.map(\.probability)
                    let ids = MLXArray(p.map { Int32($0.id) })
                    let noise = MLXRandom.gumbel([248_320], key: keys[row])
                    let pick = argMax(MLX.log(MLXArray(probabilities)) + noise[ids]).item(Int.self)
                    return (Array(0...row), p[pick].id)
                }
                preconditionFailure("The bonus row always supplies a target token")
            }
            #if MERERUN_RELEASE_REPLAY
            if let trace = ProcessInfo.processInfo.environment["MERERUN_Q38_CHAIN_COVERAGE_TRACE"] {
                let row = Q35ChainCoverageRecord(anchor: anchor, tokens: proposal.tokens,
                    candidates: proposal.candidates, probabilities: proposal.probabilities, uniforms: draws,
                    keys: gpuDraws?.keys ?? [], accepted: round.count - 1, following: round.following!)
                let url = URL(fileURLWithPath: trace, isDirectory: true)
                try JSONEncoder().encode(row).write(to: url.appendingPathComponent("\(generated).json"))
            }
            #endif
            let accepted = proposal.tokens.prefix(round.count - 1).prefix { !eosSet.contains($0) }
            for token in accepted { emit(token); generated += 1 }
            stats.rounds += 1
            stats.roundsByDepth[count] += 1
            stats.draftedTokens += count
            stats.acceptedTokens += accepted.count
            if accepted.count < round.count - 1 || generated == tokenBudget { break }
            if let native {
                try native.stageNative(round.taps, count: round.count)
            } else {
                draft.appendTargetContext(target.selectedArrays(count: round.count), cache: draftCache)
                MLX.asyncEval(draftCache.flatMap(\.state))
            }
            anchor = round.following!
        }
        try target.publish(to: layerCaches)
        try native?.finish(draftCache)
        published = true
        return stats
    }
}
#endif
