import Foundation
import MLX

public enum LagunaDFlashRouting {
    public static let defaultSpeculativeTokens = 12
    public static let defaultMinimumOutputTokens = 32
    public static let immediateFallbackAcceptanceRate = 0.25
    public static let defaultMinimumAcceptanceRate = 0.6
    public static let defaultAcceptanceEvaluationRounds = 2

    public static func shouldUseDFlash(
        tokenBudget: Int,
        minimumOutputTokens: Int
    ) -> Bool {
        tokenBudget >= minimumOutputTokens
    }
}

public enum LagunaDFlashRoutingMode: String, Codable, Sendable {
    case automatic
    case targetOnly = "target-only"
    case dflash
}

public struct LagunaDFlashStats: Codable, Equatable, Sendable {
    public let enabled: Bool
    public let speculativeTokens: Int
    public let minimumOutputTokens: Int
    public let routedRequests: Int
    public let bypassedRequests: Int
    public let rounds: Int
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let rejectedDraftTokens: Int
    public let fullAcceptanceRounds: Int
    public let targetVerificationForwards: Int
    public let targetRecoveryForwards: Int
    public let targetFallbackForwards: Int
    public let adaptiveFallbacks: Int

    public var acceptanceRate: Double {
        guard draftedTokens > 0 else { return 0 }
        return Double(acceptedDraftTokens) / Double(draftedTokens)
    }

    public init(
        enabled: Bool,
        speculativeTokens: Int,
        minimumOutputTokens: Int = LagunaDFlashRouting.defaultMinimumOutputTokens,
        routedRequests: Int = 0,
        bypassedRequests: Int = 0,
        rounds: Int = 0,
        draftedTokens: Int = 0,
        acceptedDraftTokens: Int = 0,
        rejectedDraftTokens: Int = 0,
        fullAcceptanceRounds: Int = 0,
        targetVerificationForwards: Int = 0,
        targetRecoveryForwards: Int = 0,
        targetFallbackForwards: Int = 0,
        adaptiveFallbacks: Int = 0
    ) {
        self.enabled = enabled
        self.speculativeTokens = speculativeTokens
        self.minimumOutputTokens = minimumOutputTokens
        self.routedRequests = routedRequests
        self.bypassedRequests = bypassedRequests
        self.rounds = rounds
        self.draftedTokens = draftedTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.rejectedDraftTokens = rejectedDraftTokens
        self.fullAcceptanceRounds = fullAcceptanceRounds
        self.targetVerificationForwards = targetVerificationForwards
        self.targetRecoveryForwards = targetRecoveryForwards
        self.targetFallbackForwards = targetFallbackForwards
        self.adaptiveFallbacks = adaptiveFallbacks
    }
}

struct LagunaDFlashDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let stats: LagunaDFlashStats
    let targetCache: [Gemma4AttentionCache]
    let draftCache: [Gemma4AttentionCache]
}

private struct LagunaDFlashPreparedGreedyRound {
    let anchor: Int
    let proposals: [Int]
    let targetTokens: [Int]
    let candidateTargetCache: [Gemma4AttentionCache]
    let candidate: LagunaForwardOutput
}

enum LagunaDFlashDecoder {
    static func decode(
        initialLogits: MLXArray,
        target: LagunaCausalLM,
        targetCache initialTargetCache: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel,
        draftCache initialDraftCache: [Gemma4AttentionCache],
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int],
        speculativeTokens: Int,
        adaptiveMinimumAcceptanceRate: Double? = nil,
        adaptiveAcceptanceEvaluationRounds: Int =
            LagunaDFlashRouting.defaultAcceptanceEvaluationRounds,
        decodeToken: ((Int) -> String)? = nil,
        emitPiece: ((Int, String) -> Void)? = nil,
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> LagunaDFlashDecodeResult {
        guard tokenBudget > 0 else {
            return LagunaDFlashDecodeResult(
                generatedTokens: [],
                decodeSeconds: 0,
                firstTokenSeconds: nil,
                stats: LagunaDFlashStats(
                    enabled: true,
                    speculativeTokens: speculativeTokens
                ),
                targetCache: initialTargetCache,
                draftCache: initialDraftCache
            )
        }

        let startedAt = Date()
        let captureLayerIndices = Set(dflash.config.dflash.targetLayerIDs)
        var targetCache = initialTargetCache
        let draftCache = initialDraftCache
        var logits = initialLogits
        var generatedTokens: [Int] = []
        generatedTokens.reserveCapacity(tokenBudget)
        var repetitionHistory = historySeedTokens
        var firstTokenSeconds: Double?
        var pendingProgressWhitespace = ""
        var rounds = 0
        var draftedTokens = 0
        var acceptedDraftTokens = 0
        var rejectedDraftTokens = 0
        var fullAcceptanceRounds = 0
        var targetVerificationForwards = 0
        var targetRecoveryForwards = 0
        var targetFallbackForwards = 0
        var adaptiveFallbacks = 0
        var dflashActive = true

        func emit(_ token: Int) {
            generatedTokens.append(token)
            repetitionHistory.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(startedAt)
            }
            guard let decodeToken, let emitPiece else { return }
            let piece = decodeToken(token)
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingProgressWhitespace += piece
            } else if !piece.isEmpty {
                emitPiece(token, pendingProgressWhitespace + piece)
                pendingProgressWhitespace = ""
            }
        }

        func result() -> LagunaDFlashDecodeResult {
            LagunaDFlashDecodeResult(
                generatedTokens: generatedTokens,
                decodeSeconds: Date().timeIntervalSince(startedAt),
                firstTokenSeconds: firstTokenSeconds,
                stats: LagunaDFlashStats(
                    enabled: true,
                    speculativeTokens: speculativeTokens,
                    rounds: rounds,
                    draftedTokens: draftedTokens,
                    acceptedDraftTokens: acceptedDraftTokens,
                    rejectedDraftTokens: rejectedDraftTokens,
                    fullAcceptanceRounds: fullAcceptanceRounds,
                    targetVerificationForwards: targetVerificationForwards,
                    targetRecoveryForwards: targetRecoveryForwards,
                    targetFallbackForwards: targetFallbackForwards,
                    adaptiveFallbacks: adaptiveFallbacks
                ),
                targetCache: targetCache,
                draftCache: draftCache
            )
        }

        while generatedTokens.count < tokenBudget {
            try checkCancellation?()
            let remainingAfterAnchor = tokenBudget - generatedTokens.count - 1
            let draftCount = min(
                max(1, speculativeTokens),
                min(
                    dflash.config.dflash.blockSize - 1,
                    max(0, remainingAfterAnchor)
                )
            )
            let preparedGreedyRound: LagunaDFlashPreparedGreedyRound?
            if dflashActive,
               remainingAfterAnchor > 0,
               generationConfig.temperature == 0,
               generationConfig.repetitionPenalty == nil,
               generationConfig.bannedTokens.isEmpty {
                preparedGreedyRound = prepareGreedyRound(
                    logits: logits,
                    target: target,
                    targetCache: targetCache,
                    dflash: dflash,
                    draftCache: draftCache,
                    draftCount: draftCount,
                    captureLayerIndices: captureLayerIndices
                )
            } else {
                preparedGreedyRound = nil
            }
            let anchor = preparedGreedyRound?.anchor ?? sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )
            guard !eosTokens.contains(anchor) else { break }
            emit(anchor)
            guard generatedTokens.count < tokenBudget else { break }

            if !dflashActive {
                logits = target.lastPositionLogits(
                    MLXArray([Int32(anchor)]).reshaped(1, 1),
                    cache: targetCache
                )
                MLX.eval(logits)
                targetFallbackForwards += 1
                continue
            }

            let proposals: [Int]
            var proposalProbabilities: [MLXArray] = []
            let candidateTargetCache: [Gemma4AttentionCache]
            let candidate: LagunaForwardOutput
            if let preparedGreedyRound {
                proposals = preparedGreedyRound.proposals
                candidateTargetCache = preparedGreedyRound.candidateTargetCache
                candidate = preparedGreedyRound.candidate
            } else {
                let candidateDraftCache = draftCache.map { $0.fork() }
                let draftLogits = dflash.draftLogits(
                    anchorTokens: MLXArray([Int32(anchor)]).reshaped(1, 1),
                    speculativeTokenCount: draftCount,
                    cache: candidateDraftCache,
                    target: target
                )
                MLX.eval(draftLogits)

                var sampledProposals: [Int] = []
                var proposalHistory = repetitionHistory
                sampledProposals.reserveCapacity(draftCount)
                proposalProbabilities.reserveCapacity(draftCount)
                for index in 0..<draftCount {
                    let proposalLogits = draftLogits[0, index, 0...]
                    if generationConfig.temperature == 0 {
                        let token = sampleToken(
                            logits: proposalLogits,
                            config: generationConfig,
                            previousTokens: proposalHistory
                        )
                        sampledProposals.append(token)
                        proposalHistory.append(token)
                    } else {
                        let probabilities = samplingProbabilities(
                            logits: proposalLogits,
                            config: generationConfig,
                            previousTokens: proposalHistory
                        )
                        let token = sampleToken(probabilities: probabilities)
                        sampledProposals.append(token)
                        proposalProbabilities.append(probabilities)
                        proposalHistory.append(token)
                    }
                }
                proposals = sampledProposals

                candidateTargetCache = targetCache.map { $0.fork() }
                let candidateInput = MLXArray(
                    ([anchor] + proposals).map(Int32.init)
                ).reshaped(1, proposals.count + 1)
                candidate = target.forward(
                    candidateInput,
                    cache: candidateTargetCache,
                    captureLayerIndices: captureLayerIndices
                )
                MLX.eval(
                    [candidate.logits]
                        + Array(candidate.capturedHiddenStates.values)
                )
            }

            rounds += 1
            draftedTokens += proposals.count
            targetVerificationForwards += 1

            var accepted = 0
            var replacement: Int?
            var verificationHistory = repetitionHistory
            for (index, proposal) in proposals.enumerated() {
                let targetLogits = candidate.logits[0, index, 0...]
                if generationConfig.temperature == 0 {
                    let targetToken = preparedGreedyRound?.targetTokens[index]
                        ?? sampleToken(
                            logits: targetLogits,
                            config: generationConfig,
                            previousTokens: verificationHistory
                        )
                    guard targetToken == proposal else {
                        replacement = targetToken
                        break
                    }
                } else {
                    let targetProbabilities = samplingProbabilities(
                        logits: targetLogits,
                        config: generationConfig,
                        previousTokens: verificationHistory
                    )
                    let draftProbabilities = proposalProbabilities[index]
                    let draftProbability = max(
                        draftProbabilities[proposal].item(Float.self),
                        Float.leastNonzeroMagnitude
                    )
                    let targetProbability = targetProbabilities[proposal].item(Float.self)
                    let acceptanceProbability = min(
                        1,
                        targetProbability / draftProbability
                    )
                    guard Float.random(in: 0..<1) <= acceptanceProbability else {
                        replacement = sampleToken(probabilities: rejectionDistribution(
                            target: targetProbabilities,
                            draft: draftProbabilities
                        ))
                        break
                    }
                }
                accepted += 1
                verificationHistory.append(proposal)
            }
            acceptedDraftTokens += accepted

            if accepted == proposals.count {
                fullAcceptanceRounds += 1
                for proposal in proposals {
                    guard !eosTokens.contains(proposal) else { return result() }
                    emit(proposal)
                    guard generatedTokens.count < tokenBudget else { return result() }
                }
                targetCache = candidateTargetCache
                logits = candidate.logits[
                    0...,
                    (candidate.logits.dim(1) - 1)...,
                    0...
                ]
                let combined = dflash.combineTargetHiddenStates(
                    candidate.capturedHiddenStates
                )
                dflash.appendTargetContext(combined, cache: draftCache)
                evaluateGemma4CacheStorage(draftCache)
                if shouldFallBack(
                    minimumAcceptanceRate: adaptiveMinimumAcceptanceRate,
                    evaluationRounds: adaptiveAcceptanceEvaluationRounds,
                    rounds: rounds,
                    draftedTokens: draftedTokens,
                    acceptedDraftTokens: acceptedDraftTokens
                ) {
                    dflashActive = false
                    adaptiveFallbacks = 1
                }
                continue
            }

            rejectedDraftTokens += 1
            for proposal in proposals.prefix(accepted) {
                guard !eosTokens.contains(proposal) else { return result() }
                emit(proposal)
                guard generatedTokens.count < tokenBudget else { return result() }
            }
            guard let replacement, !eosTokens.contains(replacement) else {
                break
            }

            let committedCandidateTokenCount = accepted + 1
            let recoveryCache = commitCandidatePrefix(
                base: targetCache,
                candidate: candidateTargetCache,
                tokenCount: committedCandidateTokenCount
            )
            let recovery = target.forward(
                MLXArray([Int32(replacement)]).reshaped(1, 1),
                cache: recoveryCache,
                captureLayerIndices: captureLayerIndices
            )
            MLX.eval([recovery.logits] + Array(recovery.capturedHiddenStates.values))
            targetRecoveryForwards += 1
            emit(replacement)
            targetCache = recoveryCache
            logits = recovery.logits[
                0...,
                (recovery.logits.dim(1) - 1)...,
                0...
            ]
            let committedHiddenStates = Dictionary(
                uniqueKeysWithValues: dflash.config.dflash.targetLayerIDs.map { layerID in
                    let candidatePrefix = candidate.capturedHiddenStates[layerID]![
                        0...,
                        ..<committedCandidateTokenCount,
                        0...
                    ]
                    return (
                        layerID,
                        concatenated(
                            [
                                candidatePrefix,
                                recovery.capturedHiddenStates[layerID]!,
                            ],
                            axis: 1
                        )
                    )
                }
            )
            let combined = dflash.combineTargetHiddenStates(committedHiddenStates)
            dflash.appendTargetContext(combined, cache: draftCache)
            evaluateGemma4CacheStorage(draftCache)
            if shouldFallBack(
                minimumAcceptanceRate: adaptiveMinimumAcceptanceRate,
                evaluationRounds: adaptiveAcceptanceEvaluationRounds,
                rounds: rounds,
                draftedTokens: draftedTokens,
                acceptedDraftTokens: acceptedDraftTokens
            ) {
                dflashActive = false
                adaptiveFallbacks = 1
            }
        }

        return result()
    }

    private static func prepareGreedyRound(
        logits: MLXArray,
        target: LagunaCausalLM,
        targetCache: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel,
        draftCache: [Gemma4AttentionCache],
        draftCount: Int,
        captureLayerIndices: Set<Int>
    ) -> LagunaDFlashPreparedGreedyRound {
        let anchorToken = MLX.argMax(
            logits[0, -1, 0...],
            axis: -1
        ).asType(.int32).reshaped(1, 1)
        let candidateDraftCache = draftCache.map { $0.fork() }
        let draftLogits = dflash.draftLogits(
            anchorTokens: anchorToken,
            speculativeTokenCount: draftCount,
            cache: candidateDraftCache,
            target: target
        )
        let proposalTokens = MLX.argMax(
            draftLogits,
            axis: -1
        ).asType(.int32)
        let candidateTargetCache = targetCache.map { $0.fork() }
        let candidate = target.forward(
            concatenated([anchorToken, proposalTokens], axis: 1),
            cache: candidateTargetCache,
            captureLayerIndices: captureLayerIndices
        )
        let targetTokens = MLX.argMax(
            candidate.logits[0, 0..<draftCount, 0...],
            axis: -1
        ).asType(.int32)
        MLX.eval(
            [anchorToken, proposalTokens, targetTokens, candidate.logits]
                + Array(candidate.capturedHiddenStates.values)
        )
        return LagunaDFlashPreparedGreedyRound(
            anchor: Int(anchorToken.item(Int32.self)),
            proposals: proposalTokens.asArray(Int32.self).map(Int.init),
            targetTokens: targetTokens.asArray(Int32.self).map(Int.init),
            candidateTargetCache: candidateTargetCache,
            candidate: candidate
        )
    }

    static func commitCandidatePrefix(
        base: [Gemma4AttentionCache],
        candidate: [Gemma4AttentionCache],
        tokenCount: Int
    ) -> [Gemma4AttentionCache] {
        precondition(base.count == candidate.count && tokenCount > 0)
        return zip(base, candidate).map { baseCache, candidateCache in
            let appendedCount = candidateCache.offset - baseCache.offset
            precondition(appendedCount >= tokenCount)
            let state = candidateCache.currentState()!
            let appendedStart = state.0.dim(2) - appendedCount
            let committedEnd = appendedStart + tokenCount
            let committed = baseCache.fork()
            committed.append(
                keys: state.0[0..., 0..., appendedStart..<committedEnd, 0...],
                values: state.1[0..., 0..., appendedStart..<committedEnd, 0...]
            )
            return committed
        }
    }

    static func rejectionDistribution(
        target: MLXArray,
        draft: MLXArray
    ) -> MLXArray {
        let residual = MLX.maximum(target - draft, MLXArray(0))
        let residualMass = residual.sum().item(Float.self)
        return residualMass > 1e-6
            ? residual / residual.sum()
            : target
    }

    private static func shouldFallBack(
        minimumAcceptanceRate: Double?,
        evaluationRounds: Int,
        rounds: Int,
        draftedTokens: Int,
        acceptedDraftTokens: Int
    ) -> Bool {
        guard let minimumAcceptanceRate,
              draftedTokens > 0 else {
            return false
        }
        let acceptanceRate = Double(acceptedDraftTokens) / Double(draftedTokens)
        if rounds == 1,
           acceptanceRate < min(
               LagunaDFlashRouting.immediateFallbackAcceptanceRate,
               minimumAcceptanceRate
           ) {
            return true
        }
        return rounds >= evaluationRounds && acceptanceRate < minimumAcceptanceRate
    }
}
