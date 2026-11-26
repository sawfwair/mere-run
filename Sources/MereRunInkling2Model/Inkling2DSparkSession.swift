import Foundation
import MLX

package struct Inkling2DSparkDecodeResult {
    package let tokens: [Int]
    package let seconds: Double
    package let firstTokenSeconds: Double?
    package let rounds: Int
    package let drafted: Int
    package let accepted: Int
    package let depthCounts: [Int: Int]
}

/// The draft's state after a prompt prefix, kept with the target cache for reuse.
package struct Inkling2DSparkPrefix {
    fileprivate let caches: [Inkling2DSparkLayerCache]
    fileprivate let lastFeatures: MLXArray
}

package final class Inkling2DSparkSession {
    private struct TargetVerification {
        let logits: MLXArray
        let commit: (Int) throws -> MLXArray
    }

    private let draft: Inkling2DSpark
    private let targetCache: Inkling2Cache
    private var draftCaches: [Inkling2DSparkLayerCache]
    private var lastFeatures: MLXArray?

    package init(draft: Inkling2DSpark, cache: Inkling2Cache, prefix: Inkling2DSparkPrefix? = nil) {
        self.draft = draft
        targetCache = cache
        draftCaches = prefix?.caches ?? Array(repeating: Inkling2DSparkLayerCache(), count: draft.config.numHiddenLayers)
        lastFeatures = prefix?.lastFeatures
    }

    /// The draft state after the tokens prefilled so far.
    package var prefix: Inkling2DSparkPrefix? {
        lastFeatures.map { Inkling2DSparkPrefix(caches: draftCaches, lastFeatures: $0) }
    }

    package func prefill(_ embeddings: MLXArray) throws -> MLXArray {
        let offset = targetCache.offset
        let capture = Inkling2DSparkFeatures(draft.config.targetLayerIds)
        let logits = try draft.target.forward(embeddings: embeddings, cache: targetCache, lastPositionOnly: true, capture: capture)
        let features = try capture.joined()
        draft.append(features, offset: offset, caches: &draftCaches)
        lastFeatures = features[0..., (embeddings.dim(1) - 1)..., 0..., 0...]
        eval([logits, features] + draftCaches.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        return logits
    }

    package func decode(initial: MLXArray, budget: Int, eos: Set<Int>, fixedDepth: Int?,
                        distribution: ((MLXArray) -> MLXArray)? = nil) throws -> Inkling2DSparkDecodeResult {
        guard var lastFeatures else { throw Inkling2Error.invalid("DSpark requires prefilled target features") }
        let start = Date()
        #if os(macOS)
        let speedPack = draft.target.usesSpeedPack && draft.target.usePackedExperts
        if speedPack { Inkling2SpeedPackAllocator.enter() }
        defer { if speedPack { Inkling2SpeedPackAllocator.leave() } }
        if speedPack, budget <= Inkling2CompiledVerifier.capacityLimit - targetCache.offset {
            draft.prepareChronologicalBank(offset: targetCache.offset, budget: budget, caches: &draftCaches)
        }
        var compiled = distribution == nil ? nil : Inkling2CompiledVerifier(draft: draft, cache: targetCache, budget: budget)
        defer { compiled?.restore() }
        let overlapsDraft = compiled != nil && GPU.deviceInfo().architecture == "applegpu_g17s"
        #endif
        let initialRow = initial[0, -1]
        var next = distribution.map { MLXRandom.categorical(log($0(initialRow))) }
            .map { $0.item(Int.self) } ?? argMax(initialRow, axis: -1).item(Int.self)
        var firstTokenSeconds: Double?
        var tokens: [Int] = []
        var rounds = 0, drafted = 0, accepted = 0
        // The fused packed verifier makes each additional proposed position cheaper.
        let verifierBaseCost: Double = draft.target.usePackedExperts ? 12 : 8
        var controller = Inkling2DSparkDepth(maximum: draft.maxProposalCount, fixedDepth: fixedDepth,
                                            baseCost: verifierBaseCost)
        var depthCounts: [Int: Int] = [:]
        let lookupEnabled = fixedDepth == nil && draft.target.usePackedExperts
        var fullyAcceptedRounds = 0
        while tokens.count < budget {
            try Task.checkCancellation()
            if eos.contains(next) { break }
            tokens.append(next)
            if firstTokenSeconds == nil { firstTokenSeconds = Date().timeIntervalSince(start) }
            let remaining = budget - tokens.count
            let replay = lookupEnabled && fullyAcceptedRounds >= 4
                ? Inkling2DSparkLookup.match(in: tokens, limit: min(16, remaining))
                : nil
            let count = replay?.count ?? min(controller.depth, remaining)
            if count == 0 { break }

            #if os(macOS)
            let offset = compiled?.offset ?? targetCache.offset
            #else
            let offset = targetCache.offset
            #endif
            let anchor = MLXArray(Int32(next))
            let proposed: (tokens: MLXArray, probabilities: [MLXArray])
            if let replay {
                let sequence = [Int32(next)] + tokens[replay].map(Int32.init)
                proposed = (MLXArray(sequence).reshaped(1, -1), [])
            } else {
                proposed = try draft.propose(anchor: anchor, count: count, features: lastFeatures,
                                              offset: offset, caches: &draftCaches, distribution: distribution)
            }
            var target: TargetVerification?
            #if os(macOS)
            if let verifier = compiled {
                if overlapsDraft, replay == nil, Inkling2CompiledVerifier.supportedWidths.contains(proposed.tokens.dim(1)) {
                    // Let the draft execute while the CPU builds the target graph.
                    asyncEval(proposed.tokens)
                }
                if let panel = try verifier.verify(proposed.tokens) {
                    target = TargetVerification(logits: panel.logits) { count in
                        verifier.commit(panel, count: count)
                        return panel.features[0..., ..<count, 0..., 0...]
                    }
                } else {
                    verifier.restore()
                    compiled = nil
                    if proposed.tokens.dim(1) < Inkling2CompiledVerifier.supportedWidths.lowerBound {
                        let direct = try verifyDirect(proposed.tokens)
                        target = TargetVerification(logits: direct.logits) { count in
                            let features = try direct.commit(count)
                            if verifier.resumeAfterDirectCommit() {
                                compiled = verifier
                            }
                            return features
                        }
                    }
                }
            }
            #endif
            let targetVerification = try target ?? verifyDirect(proposed.tokens)
            let verification = Inkling2DSparkVerification.check(
                proposals: proposed.tokens[0, 1...], targetLogits: targetVerification.logits[0],
                proposalProbabilities: proposed.probabilities, distribution: distribution, deterministic: replay != nil
            )
            let prefix = verification.proposals.prefix(verification.accepted).prefix { !eos.contains($0) }
            let acceptedThisRound = prefix.count
            rounds += 1
            depthCounts[count, default: 0] += 1
            drafted += count
            accepted += acceptedThisRound
            fullyAcceptedRounds = acceptedThisRound == count ? fullyAcceptedRounds + 1 : 0
            if replay == nil {
                controller.observe(proposed: count, accepted: acceptedThisRound)
            }
            tokens.append(contentsOf: prefix)
            next = verification.next
            if acceptedThisRound < verification.accepted || tokens.count == budget || eos.contains(next) { break }

            let commit = 1 + acceptedThisRound
            let features = try targetVerification.commit(commit)
            draft.append(features, offset: offset, caches: &draftCaches)
            if replay != nil {
                // A lookup round has no neural draft to evaluate these cache updates.
                asyncEval(draftCaches.flatMap { [$0.keys, $0.values].compactMap { $0 } })
            }
            lastFeatures = features[0..., (commit - 1)..<commit, 0..., 0...]
        }
        let seconds = Date().timeIntervalSince(start)
        return Inkling2DSparkDecodeResult(tokens: tokens, seconds: seconds,
                                         firstTokenSeconds: firstTokenSeconds, rounds: rounds, drafted: drafted, accepted: accepted,
                                         depthCounts: depthCounts)
    }

    private func verifyDirect(_ tokens: MLXArray) throws -> TargetVerification {
        let embeddings = try draft.target.embeddings(tokens)
        let capture = Inkling2DSparkFeatures(draft.config.targetLayerIds)
        targetCache.beginVerification()
        let logits = try draft.target.forward(embeddings: embeddings, cache: targetCache,
                                              lastPositionOnly: false, capture: capture)
        return TargetVerification(logits: logits) { [targetCache] count in
            targetCache.commitVerification(count)
            return try capture.joined()[0..., ..<count, 0..., 0...]
        }
    }
}

enum Inkling2DSparkLookup {
    static func match(in tokens: [Int], limit: Int) -> Range<Int>? {
        let suffixLength = 8, minimumProposal = 8
        guard limit >= minimumProposal, tokens.count >= suffixLength + minimumProposal else { return nil }
        let tail = tokens.suffix(suffixLength)
        let newest = tokens.count - suffixLength - minimumProposal
        let oldest = max(0, tokens.count - 2048)
        var later: Range<Int>?
        for start in stride(from: newest, through: oldest, by: -1) where tokens[start] == tail.first {
            let continuation = start + suffixLength
            if tokens[start..<continuation].elementsEqual(tail) {
                let earlier = continuation..<(continuation + min(limit, tokens.count - continuation))
                if let later {
                    let agreed = zip(tokens[earlier], tokens[later]).prefix { $0.0 == $0.1 }.count
                    return agreed >= minimumProposal ? later.lowerBound..<(later.lowerBound + agreed) : nil
                }
                later = earlier
            }
        }
        return nil
    }
}

enum Inkling2DSparkVerification {
    static func check(
        proposals: MLXArray, targetLogits: MLXArray, proposalProbabilities: [MLXArray],
        distribution: ((MLXArray) -> MLXArray)?, deterministic: Bool = false
    ) -> (proposals: [Int], accepted: Int, next: Int) {
        let count = proposals.size
        if let distribution {
            let target = distribution(targetLogits)
            let uniforms = MLXRandom.uniform(0..<Float(1), [count])
            let (accepted, mass) = deterministic
                ? checkDeterministic(proposals: proposals, target: target, uniforms: uniforms)
                : checkSampled(proposals: proposals, target: target, draft: stacked(proposalProbabilities, axis: 0), uniforms: uniforms)
            let next = MLXRandom.categorical(log(mass)).asType(.int32)
            let result = concatenated([stacked([accepted, next]), proposals], axis: 0).asArray(Int32.self)
            return (result.dropFirst(2).map(Int.init), Int(result[0]), Int(result[1]))
        }
        let ids = argMax(targetLogits, axis: -1).asType(.int32)
        let result = concatenated([proposals, ids], axis: 0).asArray(Int32.self)
        let accepted = (0..<count).prefix { result[$0] == result[count + $0] }.count
        return (result.prefix(count).map(Int.init), accepted, Int(result[count + accepted]))
    }

    static func checkDeterministic(proposals: MLXArray, target: MLXArray, uniforms: MLXArray) -> (MLXArray, MLXArray) {
        let count = proposals.size
        let p = takeAlong(target[..<count], proposals.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let rejected = concatenated([(uniforms .>= p), MLXArray([true])])
        let accepted = argMax(rejected.asType(.int32)).asType(.int32)
        let rejectedID = concatenated([proposals, MLXArray([Int32(-1)])])[accepted]
        let vocabulary = MLXArray(Int32(0)..<Int32(target.dim(-1)))
        return (accepted, MLX.where(vocabulary .== rejectedID, MLXArray(Float(0)), target[accepted]))
    }

    static func checkSampled(proposals: MLXArray, target: MLXArray, draft: MLXArray, uniforms: MLXArray) -> (MLXArray, MLXArray) {
        let count = proposals.size
        let indices = proposals.expandedDimensions(axis: -1)
        let p = takeAlong(target[..<count], indices, axis: -1).squeezed(axis: -1)
        let q = takeAlong(draft, indices, axis: -1).squeezed(axis: -1)
        // Stop at the first rejection, or use the target's bonus row if all drafts pass.
        let rejected = concatenated([(uniforms * q .>= p), MLXArray([true])])
        let accepted = argMax(rejected.asType(.int32)).asType(.int32)
        let paddedDraft = concatenated([draft, MLXArray.zeros([1, draft.dim(-1)])], axis: 0)
        return (accepted, maximum(target[accepted] - paddedDraft[accepted], 0))
    }
}

#if os(macOS)
/// MLX's allocator is shared by sessions on the same device.
private enum Inkling2SpeedPackAllocator {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var users = 0
    nonisolated(unsafe) private static var originalLimit = 0

    static func enter() {
        lock.withLock {
            if users == 0 {
                originalLimit = Memory.cacheLimit
                Memory.cacheLimit = min(originalLimit, 4 * 1_024 * 1_024 * 1_024)
            }
            users += 1
        }
    }

    static func leave() {
        lock.withLock {
            users -= 1
            if users == 0 { Memory.cacheLimit = originalLimit }
        }
    }
}
#endif
