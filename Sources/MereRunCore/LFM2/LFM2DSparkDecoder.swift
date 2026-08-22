import Foundation
import MLX

public enum LFM2DSparkPolicy {
    public static let defaultMinimumOutputTokens = 16
    public static let defaultMinimumAcceptanceRate = 0.2
    public static let defaultAcceptanceEvaluationRounds = 3

    static func enabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = environment["MERERUN_LFM25_DSPARK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return !["0", "false", "no", "off"].contains(value)
    }

    static func minimumOutputTokens(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        max(
            1,
            environment["MERERUN_LFM25_DSPARK_MIN_OUTPUT"].flatMap(Int.init)
                ?? defaultMinimumOutputTokens
        )
    }

    static func minimumAcceptanceRate(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Double {
        let configured = environment["MERERUN_LFM25_DSPARK_MIN_ACCEPTANCE"]
            .flatMap(Double.init)
        return min(max(configured ?? defaultMinimumAcceptanceRate, 0), 1)
    }
}

public struct LFM2DSparkStats: Sendable, Hashable {
    public let enabled: Bool
    public let active: Bool
    public let speculativeTokens: Int
    public let rounds: Int
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let rejectedDraftTokens: Int
    public let targetVerificationForwards: Int
    public let targetRecoveryForwards: Int
    public let targetFallbackForwards: Int
    public let adaptiveFallbacks: Int
    public let reason: String?

    public var acceptanceRate: Double {
        draftedTokens == 0 ? 0 : Double(acceptedDraftTokens) / Double(draftedTokens)
    }

    public init(
        enabled: Bool = false,
        active: Bool = false,
        speculativeTokens: Int = 0,
        rounds: Int = 0,
        draftedTokens: Int = 0,
        acceptedDraftTokens: Int = 0,
        rejectedDraftTokens: Int = 0,
        targetVerificationForwards: Int = 0,
        targetRecoveryForwards: Int = 0,
        targetFallbackForwards: Int = 0,
        adaptiveFallbacks: Int = 0,
        reason: String? = nil
    ) {
        self.enabled = enabled
        self.active = active
        self.speculativeTokens = speculativeTokens
        self.rounds = rounds
        self.draftedTokens = draftedTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.rejectedDraftTokens = rejectedDraftTokens
        self.targetVerificationForwards = targetVerificationForwards
        self.targetRecoveryForwards = targetRecoveryForwards
        self.targetFallbackForwards = targetFallbackForwards
        self.adaptiveFallbacks = adaptiveFallbacks
        self.reason = reason
    }
}

struct LFM2DSparkDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let stats: LFM2DSparkStats
}

private struct LFM2PreparedGreedyRound {
    let anchor: Int
    let proposals: [Int]
    let targetTokens: [Int]
    let candidateCache: [LFM2LayerCache?]
    let candidate: LFM2ForwardOutput
}

enum LFM2DSparkDecoder {
    static func decode(
        initialLogits: MLXArray,
        target: LFM2Model,
        targetCache initialTargetCache: [LFM2LayerCache?],
        dspark: LFM2DSparkModel,
        draftCache: [Gemma4AttentionCache],
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int],
        decodeToken: ((Int) -> String)?,
        emitPiece: ((Int, String) -> Void)?,
        checkCancellation: (() throws -> Void)?
    ) throws -> LFM2DSparkDecodeResult {
        let startedAt = Date()
        var logits = initialLogits
        var targetCache = initialTargetCache
        var generated: [Int] = []
        var history = historySeedTokens
        var firstTokenSeconds: Double?
        var pendingWhitespace = ""
        var rounds = 0
        var drafted = 0
        var acceptedTotal = 0
        var rejected = 0
        var verificationForwards = 0
        let recoveryForwards = 0
        var fallbackForwards = 0
        var adaptiveFallbacks = 0
        var dsparkActive = true
        var fallbackReason: String?
        var pendingSampledAnchor: Int?
        let captures = Set(dspark.config.features.targetLayerIDs)
        let speculativeTokens = dspark.config.blockSize

        func emit(_ token: Int) {
            generated.append(token)
            history.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(startedAt)
            }
            guard let decodeToken, let emitPiece else { return }
            let piece = decodeToken(token)
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingWhitespace += piece
            } else if !piece.isEmpty {
                emitPiece(token, pendingWhitespace + piece)
                pendingWhitespace = ""
            }
        }

        func result() -> LFM2DSparkDecodeResult {
            LFM2DSparkDecodeResult(
                generatedTokens: generated,
                decodeSeconds: Date().timeIntervalSince(startedAt),
                firstTokenSeconds: firstTokenSeconds,
                stats: LFM2DSparkStats(
                    enabled: true,
                    active: dsparkActive,
                    speculativeTokens: speculativeTokens,
                    rounds: rounds,
                    draftedTokens: drafted,
                    acceptedDraftTokens: acceptedTotal,
                    rejectedDraftTokens: rejected,
                    targetVerificationForwards: verificationForwards,
                    targetRecoveryForwards: recoveryForwards,
                    targetFallbackForwards: fallbackForwards,
                    adaptiveFallbacks: adaptiveFallbacks,
                    reason: fallbackReason
                )
            )
        }

        func updateAdaptiveRouting() {
            let minimumAcceptanceRate = LFM2DSparkPolicy.minimumAcceptanceRate()
            guard rounds >= LFM2DSparkPolicy.defaultAcceptanceEvaluationRounds,
                  drafted > 0,
                  Double(acceptedTotal) / Double(drafted) < minimumAcceptanceRate else {
                return
            }
            dsparkActive = false
            adaptiveFallbacks = 1
            fallbackReason = String(
                format: "draft acceptance fell below %.0f%%",
                minimumAcceptanceRate * 100
            )
        }

        while generated.count < tokenBudget {
            try checkCancellation?()
            let remainingAfterAnchor = tokenBudget - generated.count - 1
            let proposalCount = min(speculativeTokens, max(0, remainingAfterAnchor))
            let prepared = dsparkActive && proposalCount > 0 && generationConfig.temperature == 0
                ? prepareGreedyRound(
                    logits: logits,
                    history: history,
                    target: target,
                    targetCache: targetCache,
                    dspark: dspark,
                    draftCache: draftCache,
                    generationConfig: generationConfig,
                    proposalCount: proposalCount,
                    captureLayerIndices: captures
                )
                : nil
            let anchor = pendingSampledAnchor
                ?? prepared?.anchor
                ?? sampleToken(
                    logits: logits[0, -1, 0...],
                    config: generationConfig,
                    previousTokens: history
                )
            pendingSampledAnchor = nil
            guard !eosTokens.contains(anchor) else { break }
            emit(anchor)
            guard generated.count < tokenBudget else { break }

            if !dsparkActive || proposalCount == 0 {
                let serial = target.forward(
                    MLXArray([Int32(anchor)]).reshaped(1, 1),
                    cache: targetCache
                )
                MLX.eval(serial.logits)
                logits = serial.logits
                fallbackForwards += 1
                let remaining = tokenBudget - generated.count
                guard remaining > 0 else { return result() }
                let fallback = try AutoregressiveDecodeEngine.decode(
                    AutoregressiveDecodeRequest(
                        initialLogits: logits,
                        generationConfig: generationConfig,
                        eosTokens: eosTokens,
                        tokenBudget: remaining,
                        historySeedTokens: history
                    ),
                    stepForward: { token in target(token, cache: targetCache) },
                    decodeToken: decodeToken,
                    emitPiece: { token, piece in
                        emitPiece?(token, pendingWhitespace + piece)
                        pendingWhitespace = ""
                    },
                    checkCancellation: checkCancellation
                )
                generated.append(contentsOf: fallback.generatedTokens)
                history.append(contentsOf: fallback.generatedTokens)
                fallbackForwards += fallback.generatedTokens.count
                return result()
            }

            var proposalProbabilities: [MLXArray] = []
            let proposals: [Int]
            let candidateCache: [LFM2LayerCache?]
            let candidate: LFM2ForwardOutput
            if let prepared {
                proposals = prepared.proposals
                candidateCache = prepared.candidateCache
                candidate = prepared.candidate
            } else {
                let candidateDraftCache = draftCache.map { $0.fork() }
                let baseHidden = dspark.draftBaseHidden(
                    anchorToken: anchor,
                    proposalCount: proposalCount,
                    target: target,
                    cache: candidateDraftCache
                )
                var proposalHistory = history
                proposals = dspark.proposalTokens(
                    baseHidden: baseHidden,
                    anchorToken: anchor,
                    target: target
                ) { proposalLogits, _ in
                    let probabilities = samplingProbabilities(
                        logits: proposalLogits,
                        config: generationConfig,
                        previousTokens: proposalHistory
                    )
                    MLX.eval(probabilities)
                    let token = sampleToken(probabilities: probabilities)
                    proposalProbabilities.append(probabilities)
                    proposalHistory.append(token)
                    return token
                }

                candidateCache = target.forkCache(targetCache)
                let candidateInput = MLXArray(
                    ([anchor] + proposals).map(Int32.init)
                ).reshaped(1, proposals.count + 1)
                candidate = target.forward(
                    candidateInput,
                    cache: candidateCache,
                    captureLayerIndices: captures,
                    captureSpeculativeState: true
                )
                MLX.eval(
                    [candidate.logits]
                        + Array(candidate.capturedHiddenStates.values)
                        + target.speculativeCacheStorageArrays(candidateCache)
                )
            }
            rounds += 1
            drafted += proposals.count
            verificationForwards += 1

            var accepted = 0
            var replacement: Int?
            var verificationHistory = history
            for (index, proposal) in proposals.enumerated() {
                let targetLogits = candidate.logits[0, index, 0...]
                if generationConfig.temperature == 0 {
                    let targetToken = prepared!.targetTokens[index]
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
                    MLX.eval(targetProbabilities)
                    let q = max(
                        draftProbabilities[proposal].item(Float.self),
                        Float.leastNonzeroMagnitude
                    )
                    let p = targetProbabilities[proposal].item(Float.self)
                    guard Float.random(in: 0..<1) <= min(1, p / q) else {
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
            acceptedTotal += accepted

            if accepted == proposals.count {
                for proposal in proposals {
                    guard !eosTokens.contains(proposal) else { return result() }
                    emit(proposal)
                    guard generated.count < tokenBudget else { return result() }
                }
                targetCache = candidateCache
                let finalPosition = candidate.logits.dim(1) - 1
                logits = candidate.logits[0..., finalPosition..., 0...]
                dspark.appendTargetContext(
                    dspark.combineTargetHiddenStates(candidate.capturedHiddenStates),
                    cache: draftCache
                )
                evaluateDraftCache(draftCache)
                updateAdaptiveRouting()
                continue
            }

            rejected += 1
            for proposal in proposals.prefix(accepted) {
                guard !eosTokens.contains(proposal) else { return result() }
                emit(proposal)
                guard generated.count < tokenBudget else { return result() }
            }
            guard let replacement, !eosTokens.contains(replacement) else { break }

            let committedCandidateTokens = accepted + 1
            target.rollbackSpeculativeCache(
                candidateCache,
                candidateTokenCount: proposals.count + 1,
                committedTokenCount: committedCandidateTokens
            )
            targetCache = candidateCache
            logits = candidate.logits[0..., accepted..<(accepted + 1), 0...]
            let committedHiddenStates = Dictionary(
                uniqueKeysWithValues: captures.map { layerIndex in
                    (
                        layerIndex,
                        candidate.capturedHiddenStates[layerIndex]![
                            0...,
                            ..<committedCandidateTokens,
                            0...
                        ]
                    )
                }
            )
            dspark.appendTargetContext(
                dspark.combineTargetHiddenStates(committedHiddenStates),
                cache: draftCache
            )
            evaluateDraftCache(draftCache)
            if generationConfig.temperature != 0 {
                pendingSampledAnchor = replacement
            }
            updateAdaptiveRouting()
        }
        return result()
    }

    static func rejectionDistribution(target: MLXArray, draft: MLXArray) -> MLXArray {
        let residual = MLX.maximum(target - draft, MLXArray(0))
        let mass = residual.sum().item(Float.self)
        return mass > 1e-6 ? residual / residual.sum() : target
    }

    private static func prepareGreedyRound(
        logits: MLXArray,
        history: [Int],
        target: LFM2Model,
        targetCache: [LFM2LayerCache?],
        dspark: LFM2DSparkModel,
        draftCache: [Gemma4AttentionCache],
        generationConfig: GenerationConfig,
        proposalCount: Int,
        captureLayerIndices: Set<Int>
    ) -> LFM2PreparedGreedyRound {
        var context: [MLXArray] = []
        if !history.isEmpty {
            context.append(MLXArray(history.map(Int32.init)))
        }
        let anchorToken = greedyTokenArray(
            logits: logits[0, -1, 0...],
            config: generationConfig,
            context: context
        )
        context.append(anchorToken)

        let candidateDraftCache = draftCache.map { $0.fork() }
        let baseHidden = dspark.draftBaseHidden(
            anchorToken: anchorToken,
            proposalCount: proposalCount,
            target: target,
            cache: candidateDraftCache
        )
        let proposalTokens = dspark.proposalTokenArray(
            baseHidden: baseHidden,
            anchorToken: anchorToken,
            target: target
        ) { proposalLogits, _ in
            let token = greedyTokenArray(
                logits: proposalLogits,
                config: generationConfig,
                context: context
            )
            context.append(token)
            return token
        }

        let candidateCache = target.forkCache(targetCache)
        let candidate = target.forward(
            MLX.concatenated([anchorToken.reshaped(1, 1), proposalTokens], axis: 1),
            cache: candidateCache,
            captureLayerIndices: captureLayerIndices,
            captureSpeculativeState: true
        )

        var targetContext: [MLXArray] = []
        if !history.isEmpty {
            targetContext.append(MLXArray(history.map(Int32.init)))
        }
        targetContext.append(anchorToken)
        var targetTokenArrays: [MLXArray] = []
        targetTokenArrays.reserveCapacity(proposalCount)
        for index in 0..<proposalCount {
            targetTokenArrays.append(greedyTokenArray(
                logits: candidate.logits[0, index, 0...],
                config: generationConfig,
                context: targetContext
            ))
            targetContext.append(proposalTokens[0, index].reshaped(1))
        }
        let targetTokens = MLX.concatenated(targetTokenArrays, axis: 0)
        MLX.eval(
            [anchorToken, proposalTokens, targetTokens, candidate.logits]
                + Array(candidate.capturedHiddenStates.values)
                + target.speculativeCacheStorageArrays(candidateCache)
        )
        return LFM2PreparedGreedyRound(
            anchor: Int(anchorToken.item(Int32.self)),
            proposals: proposalTokens.asArray(Int32.self).map(Int.init),
            targetTokens: targetTokens.asArray(Int32.self).map(Int.init),
            candidateCache: candidateCache,
            candidate: candidate
        )
    }

    private static func greedyTokenArray(
        logits: MLXArray,
        config: GenerationConfig,
        context: [MLXArray]
    ) -> MLXArray {
        var adjusted = applyTokenBan(logits: logits, tokens: config.bannedTokens)
        if let penalty = config.repetitionPenalty,
           config.repetitionContextSize > 0,
           !context.isEmpty {
            let allTokens = MLX.concatenated(context, axis: 0)
            let start = max(0, allTokens.dim(0) - config.repetitionContextSize)
            adjusted = applyRepetitionPenalty(
                logits: adjusted,
                tokenIndices: allTokens[start...],
                penalty: penalty
            )
        }
        return MLX.argMax(adjusted, axis: -1).asType(.int32).reshaped(1)
    }

    private static func evaluateDraftCache(_ cache: [Gemma4AttentionCache]) {
        let arrays = cache.flatMap { $0.storageArraysForEvaluation() }
        if !arrays.isEmpty { MLX.eval(arrays) }
    }
}
