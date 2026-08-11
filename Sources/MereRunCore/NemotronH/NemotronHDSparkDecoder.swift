import Foundation
import MLX

public enum NemotronHDSparkPolicy {
    public static let defaultMinimumOutputTokens = 16
    public static let defaultMinimumAcceptanceRate = 2.0 / 3.0
    public static let defaultAcceptanceEvaluationRounds = 2

    static func enabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = environment["MERERUN_NEMOTRON35_DSPARK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return !["0", "false", "no", "off"].contains(value)
    }

    static func minimumOutputTokens(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        max(
            1,
            environment["MERERUN_NEMOTRON35_DSPARK_MIN_OUTPUT"].flatMap(Int.init)
                ?? defaultMinimumOutputTokens
        )
    }

    static func minimumAcceptanceRate(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Double {
        let configured = environment["MERERUN_NEMOTRON35_DSPARK_MIN_ACCEPTANCE"]
            .flatMap(Double.init)
        return min(max(configured ?? defaultMinimumAcceptanceRate, 0), 1)
    }
}

public struct NemotronHDSparkStats: Sendable, Hashable {
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

struct NemotronHDSparkDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let stats: NemotronHDSparkStats
}

enum NemotronHDSparkDecoder {
    static func decode(
        initialLogits: MLXArray,
        target: NemotronHCausalLM,
        targetCache initialTargetCache: [NemotronHLayerCache?],
        dspark: NemotronHDSparkModel,
        draftCache: [Gemma4AttentionCache],
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int],
        speculativeTokens: Int,
        decodeToken: ((Int) -> String)?,
        emitPiece: ((Int, String) -> Void)?,
        checkCancellation: (() throws -> Void)?
    ) throws -> NemotronHDSparkDecodeResult {
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
        var recoveryForwards = 0
        var fallbackForwards = 0
        var adaptiveFallbacks = 0
        var dsparkActive = true
        var fallbackReason: String?
        let captures = Set(dspark.config.speculation.targetLayerIDs)

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

        func result() -> NemotronHDSparkDecodeResult {
            NemotronHDSparkDecodeResult(
                generatedTokens: generated,
                decodeSeconds: Date().timeIntervalSince(startedAt),
                firstTokenSeconds: firstTokenSeconds,
                stats: NemotronHDSparkStats(
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
            let minimumAcceptanceRate = NemotronHDSparkPolicy.minimumAcceptanceRate()
            guard rounds >= NemotronHDSparkPolicy.defaultAcceptanceEvaluationRounds,
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
            let anchor = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: history
            )
            guard !eosTokens.contains(anchor) else { break }
            emit(anchor)
            guard generated.count < tokenBudget else { break }

            if !dsparkActive {
                let serial = target.forward(
                    MLXArray([Int32(anchor)]).reshaped(1, 1),
                    cache: targetCache,
                    captureLayerIndices: []
                )
                MLX.eval(serial.logits)
                logits = serial.logits
                fallbackForwards += 1
                continue
            }

            let draftCount = min(
                speculativeTokens,
                dspark.config.blockSize - 1,
                tokenBudget - generated.count
            )
            let candidateDraftCache = draftCache.map { $0.fork() }
            let baseHidden = dspark.draftBaseHidden(
                anchorToken: anchor,
                speculativeTokenCount: draftCount,
                cache: candidateDraftCache
            )
            var proposalProbabilities: [MLXArray] = []
            var proposalHistory = history
            let proposals = dspark.proposalTokens(
                baseHidden: baseHidden,
                anchorToken: anchor,
                target: target
            ) { proposalLogits, _ in
                if generationConfig.temperature == 0 {
                    let token = sampleToken(
                        logits: proposalLogits,
                        config: generationConfig,
                        previousTokens: proposalHistory
                    )
                    proposalHistory.append(token)
                    return token
                }
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

            let candidateCache = target.forkCache(targetCache)
            let candidateInput = MLXArray(
                ([anchor] + proposals).map(Int32.init)
            ).reshaped(1, proposals.count + 1)
            let candidate = target.forward(
                candidateInput,
                cache: candidateCache,
                captureLayerIndices: captures
            )
            MLX.eval([candidate.logits] + Array(candidate.capturedHiddenStates.values))
            rounds += 1
            drafted += proposals.count
            verificationForwards += 1

            var accepted = 0
            var replacement: Int?
            var verificationHistory = history
            for (index, proposal) in proposals.enumerated() {
                let targetLogits = candidate.logits[0, index, 0...]
                if generationConfig.temperature == 0 {
                    let targetToken = sampleToken(
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

            // Mamba recurrent state cannot be sliced from the candidate's final
            // state. Replay only the committed prefix from the untouched cache;
            // this is exact and bounded to at most one eight-token DSpark block.
            let recoveryCache = target.forkCache(targetCache)
            let committed = [anchor] + Array(proposals.prefix(accepted)) + [replacement]
            let recovery = target.forward(
                MLXArray(committed.map(Int32.init)).reshaped(1, committed.count),
                cache: recoveryCache,
                captureLayerIndices: captures
            )
            MLX.eval([recovery.logits] + Array(recovery.capturedHiddenStates.values))
            recoveryForwards += 1
            emit(replacement)
            targetCache = recoveryCache
            let finalPosition = recovery.logits.dim(1) - 1
            logits = recovery.logits[0..., finalPosition..., 0...]
            dspark.appendTargetContext(
                dspark.combineTargetHiddenStates(recovery.capturedHiddenStates),
                cache: draftCache
            )
            evaluateDraftCache(draftCache)
            updateAdaptiveRouting()
        }
        return result()
    }

    static func rejectionDistribution(target: MLXArray, draft: MLXArray) -> MLXArray {
        let residual = MLX.maximum(target - draft, MLXArray(0))
        let mass = residual.sum().item(Float.self)
        return mass > 1e-6 ? residual / residual.sum() : target
    }

    private static func evaluateDraftCache(_ cache: [Gemma4AttentionCache]) {
        let arrays = cache.flatMap { $0.storageArraysForEvaluation() }
        if !arrays.isEmpty { MLX.eval(arrays) }
    }
}
