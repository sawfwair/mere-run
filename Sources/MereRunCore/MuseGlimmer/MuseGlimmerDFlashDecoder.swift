import Foundation
import MLX

public enum MuseGlimmerDFlashPolicy {
    /// MLX affine-Q4 target verification is fastest with three proposals.
    /// The checkpoint supports up to 15, but longer blocks become compute-bound
    /// on Apple GPU quantized matmuls before their extra matches pay back.
    public static let defaultSpeculativeTokens = 3
    public static let defaultMinimumOutputTokens = 32
    public static let defaultMinimumAcceptanceRate = 0.4
    public static let defaultAcceptanceEvaluationRounds = 2

    static func enabled(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        let value = environment["MERERUN_MUSE_GLIMMER_DFLASH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return !["0", "false", "no", "off"].contains(value)
    }

    static func speculativeTokens(
        maximum: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let configured = environment["MERERUN_MUSE_GLIMMER_DFLASH_TOKENS"].flatMap(Int.init)
            ?? defaultSpeculativeTokens
        return min(max(1, configured), maximum)
    }

    static func minimumOutputTokens(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        max(
            1,
            environment["MERERUN_MUSE_GLIMMER_DFLASH_MIN_OUTPUT"].flatMap(Int.init)
                ?? defaultMinimumOutputTokens
        )
    }

    static func minimumAcceptanceRate(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Double {
        let configured = environment["MERERUN_MUSE_GLIMMER_DFLASH_MIN_ACCEPTANCE"]
            .flatMap(Double.init)
        return min(max(configured ?? defaultMinimumAcceptanceRate, 0), 1)
    }
}

public struct MuseGlimmerDFlashStats: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var active: Bool
    public var assistantModelPath: String?
    public var speculativeTokens: Int
    public var rounds: Int
    public var draftedTokens: Int
    public var acceptedTokens: Int
    public var rejectedTokens: Int
    public var fullAcceptanceRounds: Int
    public var targetVerificationForwards: Int
    public var targetRecoveryForwards: Int
    public var targetFallbackForwards: Int
    public var adaptiveFallbacks: Int
    public var reason: String?

    public var acceptanceRate: Double {
        guard draftedTokens > 0 else { return 0 }
        return Double(acceptedTokens) / Double(draftedTokens)
    }

    public init(
        enabled: Bool = false,
        active: Bool = false,
        assistantModelPath: String? = nil,
        speculativeTokens: Int = 0,
        rounds: Int = 0,
        draftedTokens: Int = 0,
        acceptedTokens: Int = 0,
        rejectedTokens: Int = 0,
        fullAcceptanceRounds: Int = 0,
        targetVerificationForwards: Int = 0,
        targetRecoveryForwards: Int = 0,
        targetFallbackForwards: Int = 0,
        adaptiveFallbacks: Int = 0,
        reason: String? = nil
    ) {
        self.enabled = enabled
        self.active = active
        self.assistantModelPath = assistantModelPath
        self.speculativeTokens = speculativeTokens
        self.rounds = rounds
        self.draftedTokens = draftedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
        self.fullAcceptanceRounds = fullAcceptanceRounds
        self.targetVerificationForwards = targetVerificationForwards
        self.targetRecoveryForwards = targetRecoveryForwards
        self.targetFallbackForwards = targetFallbackForwards
        self.adaptiveFallbacks = adaptiveFallbacks
        self.reason = reason
    }
}

struct MuseGlimmerDFlashDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let stats: MuseGlimmerDFlashStats
}

private struct MuseGlimmerPreparedGreedyRound {
    let anchor: Int
    let proposals: [Int]
    let targetTokens: [Int]
    let candidateTargetCache: [Gemma4AttentionCache]
    let candidate: MuseGlimmerForwardOutput
}

enum MuseGlimmerDFlashDecoder {
    static func decode(
        initialLogits: MLXArray,
        target: MuseGlimmerModel,
        targetCache initialTargetCache: [Gemma4AttentionCache],
        assistant: MuseGlimmerAssistantModel,
        assistantCache: [Gemma4AttentionCache],
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int],
        speculativeTokens: Int,
        assistantModelPath: String?,
        decodeToken: ((Int) -> String)? = nil,
        emitPiece: ((Int, String) -> Void)? = nil,
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> MuseGlimmerDFlashDecodeResult {
        let startedAt = Date()
        var targetCache = initialTargetCache
        var logits = initialLogits
        var generatedTokens: [Int] = []
        var repetitionHistory = historySeedTokens
        var firstTokenSeconds: Double?
        var pendingWhitespace = ""
        var stats = MuseGlimmerDFlashStats(
            enabled: true,
            active: true,
            assistantModelPath: assistantModelPath,
            speculativeTokens: speculativeTokens
        )
        var assistantActive = true
        var pendingSampledAnchor: Int?

        func emit(_ token: Int) {
            generatedTokens.append(token)
            repetitionHistory.append(token)
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

        func result() -> MuseGlimmerDFlashDecodeResult {
            MuseGlimmerDFlashDecodeResult(
                generatedTokens: generatedTokens,
                decodeSeconds: Date().timeIntervalSince(startedAt),
                firstTokenSeconds: firstTokenSeconds,
                stats: stats
            )
        }

        guard tokenBudget > 0 else { return result() }
        let captureLayers = Set(assistant.config.targetLayerIds)

        while generatedTokens.count < tokenBudget {
            try checkCancellation?()
            let remainingAfterAnchor = tokenBudget - generatedTokens.count - 1
            let draftCount = min(
                speculativeTokens,
                min(assistant.config.blockSize - 1, max(0, remainingAfterAnchor))
            )
            let prepared: MuseGlimmerPreparedGreedyRound?
            if assistantActive,
               draftCount > 0,
               generationConfig.temperature == 0,
               generationConfig.repetitionPenalty == nil,
               generationConfig.bannedTokens.isEmpty {
                prepared = prepareGreedyRound(
                    logits: logits,
                    target: target,
                    targetCache: targetCache,
                    assistant: assistant,
                    assistantCache: assistantCache,
                    draftCount: draftCount,
                    captureLayers: captureLayers
                )
            } else {
                prepared = nil
            }
            let anchor = prepared?.anchor
                ?? pendingSampledAnchor
                ?? sampleToken(
                    logits: logits[0, -1, 0...],
                    config: generationConfig,
                    previousTokens: repetitionHistory
                )
            pendingSampledAnchor = nil
            guard !eosTokens.contains(anchor) else { break }
            emit(anchor)
            guard generatedTokens.count < tokenBudget else { break }

            guard assistantActive, draftCount > 0 else {
                logits = target(
                    MLXArray([Int32(anchor)]).reshaped(1, 1),
                    cache: targetCache
                )
                MLX.eval(logits)
                evaluateGemma4CacheStorage(targetCache)
                stats.targetFallbackForwards += 1
                continue
            }

            let proposals: [Int]
            var proposalProbabilities: [MLXArray] = []
            let candidateTargetCache: [Gemma4AttentionCache]
            let candidate: MuseGlimmerForwardOutput
            if let prepared {
                proposals = prepared.proposals
                candidateTargetCache = prepared.candidateTargetCache
                candidate = prepared.candidate
            } else {
                let candidateAssistantCache = assistantCache.map { $0.fork() }
                let draftLogits = assistant.draftLogits(
                    anchorTokens: MLXArray([Int32(anchor)]).reshaped(1, 1),
                    speculativeTokenCount: draftCount,
                    cache: candidateAssistantCache,
                    target: target
                )
                MLX.eval(draftLogits)
                var sampled: [Int] = []
                var proposalHistory = repetitionHistory
                sampled.reserveCapacity(draftCount)
                proposalProbabilities.reserveCapacity(draftCount)
                for index in 0..<draftCount {
                    let probabilities = samplingProbabilities(
                        logits: draftLogits[0, index, 0...],
                        config: generationConfig,
                        previousTokens: proposalHistory
                    )
                    let token = sampleToken(probabilities: probabilities)
                    sampled.append(token)
                    proposalProbabilities.append(probabilities)
                    proposalHistory.append(token)
                }
                proposals = sampled
                candidateTargetCache = targetCache.map { $0.fork() }
                candidate = target.forward(
                    MLXArray(([anchor] + proposals).map(Int32.init))
                        .reshaped(1, proposals.count + 1),
                    cache: candidateTargetCache,
                    captureLayerIndices: captureLayers
                )
                MLX.eval(
                    [candidate.logits]
                        + Array(candidate.capturedHiddenStates.values)
                )
                evaluateGemma4CacheStorage(candidateTargetCache)
            }

            stats.rounds += 1
            stats.draftedTokens += proposals.count
            stats.targetVerificationForwards += 1
            var accepted = 0
            var replacement: Int?
            var verificationHistory = repetitionHistory
            for (index, proposal) in proposals.enumerated() {
                let targetLogits = candidate.logits[0, index, 0...]
                if generationConfig.temperature == 0 {
                    let targetToken = prepared?.targetTokens[index] ?? sampleToken(
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
                    let acceptanceProbability = min(
                        1,
                        targetProbabilities[proposal].item(Float.self) / draftProbability
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
            stats.acceptedTokens += accepted

            if accepted == proposals.count {
                stats.fullAcceptanceRounds += 1
                for proposal in proposals {
                    guard !eosTokens.contains(proposal) else { return result() }
                    emit(proposal)
                    guard generatedTokens.count < tokenBudget else { return result() }
                }
                targetCache = candidateTargetCache
                logits = candidate.logits[0..., (candidate.logits.dim(1) - 1)..., 0...]
                assistant.appendTargetContext(
                    candidate.capturedHiddenStates,
                    cache: assistantCache
                )
                evaluateGemma4CacheStorage(assistantCache)
            } else {
                stats.rejectedTokens += 1
                for proposal in proposals.prefix(accepted) {
                    guard !eosTokens.contains(proposal) else { return result() }
                    emit(proposal)
                    guard generatedTokens.count < tokenBudget else { return result() }
                }
                guard let replacement, !eosTokens.contains(replacement) else { break }
                let committedCount = accepted + 1
                targetCache = commitCandidatePrefix(
                    base: targetCache,
                    candidate: candidateTargetCache,
                    tokenCount: committedCount
                )
                evaluateGemma4CacheStorage(targetCache)
                let committedStates = Dictionary(
                    uniqueKeysWithValues: assistant.config.targetLayerIds.map { layerId in
                        return (
                            layerId,
                            candidate.capturedHiddenStates[layerId]![
                                0...,
                                ..<committedCount,
                                0...
                            ]
                        )
                    }
                )
                assistant.appendTargetContext(committedStates, cache: assistantCache)
                evaluateGemma4CacheStorage(assistantCache)
                logits = candidate.logits[0..., accepted..<(accepted + 1), 0...]
                if generationConfig.temperature != 0 {
                    pendingSampledAnchor = replacement
                }
            }

            let minimumAcceptanceRate = MuseGlimmerDFlashPolicy.minimumAcceptanceRate()
            if stats.rounds >= MuseGlimmerDFlashPolicy.defaultAcceptanceEvaluationRounds,
               stats.acceptanceRate < minimumAcceptanceRate {
                assistantActive = false
                stats.active = false
                stats.adaptiveFallbacks = 1
                stats.reason = String(
                    format: "draft acceptance fell below %.0f%%",
                    minimumAcceptanceRate * 100
                )
            }
        }

        return result()
    }

    private static func prepareGreedyRound(
        logits: MLXArray,
        target: MuseGlimmerModel,
        targetCache: [Gemma4AttentionCache],
        assistant: MuseGlimmerAssistantModel,
        assistantCache: [Gemma4AttentionCache],
        draftCount: Int,
        captureLayers: Set<Int>
    ) -> MuseGlimmerPreparedGreedyRound {
        let anchor = MLX.argMax(logits[0, -1, 0...], axis: -1)
            .asType(.int32)
            .reshaped(1, 1)
        let draftLogits = assistant.draftLogits(
            anchorTokens: anchor,
            speculativeTokenCount: draftCount,
            cache: assistantCache.map { $0.fork() },
            target: target
        )
        let proposalTokens = MLX.argMax(draftLogits, axis: -1).asType(.int32)
        let candidateTargetCache = targetCache.map { $0.fork() }
        let candidate = target.forward(
            concatenated([anchor, proposalTokens], axis: 1),
            cache: candidateTargetCache,
            captureLayerIndices: captureLayers
        )
        let targetTokens = MLX.argMax(
            candidate.logits[0, 0..<draftCount, 0...],
            axis: -1
        ).asType(.int32)
        MLX.eval(
            [anchor, proposalTokens, targetTokens, candidate.logits]
                + Array(candidate.capturedHiddenStates.values)
        )
        evaluateGemma4CacheStorage(candidateTargetCache)
        return MuseGlimmerPreparedGreedyRound(
            anchor: Int(anchor.item(Int32.self)),
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
            let committed = baseCache.fork()
            committed.append(
                keys: state.0[
                    0...,
                    0...,
                    appendedStart..<(appendedStart + tokenCount),
                    0...
                ],
                values: state.1[
                    0...,
                    0...,
                    appendedStart..<(appendedStart + tokenCount),
                    0...
                ]
            )
            return committed
        }
    }

    static func rejectionDistribution(target: MLXArray, draft: MLXArray) -> MLXArray {
        let residual = MLX.maximum(target - draft, MLXArray(0))
        let mass = residual.sum().item(Float.self)
        return mass > 1e-6 ? residual / residual.sum() : target
    }
}
