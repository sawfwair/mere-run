import MereRunTensor
import MereRunGemmaModel
import MereRunDecode
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

package struct LagunaDFlashDecodeResult {
    package let generatedTokens: [Int]
    package let decodeSeconds: Double
    package let firstTokenSeconds: Double?
    package let stats: LagunaDFlashStats
    package let targetCache: [Gemma4AttentionCache]
    package let draftCache: [Gemma4AttentionCache]
}

struct LagunaDFlashPreparedGreedyRound {
    package let anchor: Int
    package let proposals: [Int]
    package let targetTokens: [Int]
    package let candidateTargetCache: [Gemma4AttentionCache]
    package let candidate: LagunaForwardOutput
}
