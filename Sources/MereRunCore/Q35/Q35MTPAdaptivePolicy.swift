import Foundation

/// Chooses a greedy MTP draft depth from the acceptance observed in this request.
///
/// The MTP head only proposes. A deeper round is useful while the probability of
/// reaching its next draft outweighs the marginal head work. The target still
/// verifies every proposed token, while rejected suffixes rewind in-place.
struct Q35MTPAdaptivePolicy {
    private static let acceptanceAlpha = 0.15
    private static let inferredAcceptanceCeiling = 0.95

    /// Fit for the rollback-only path: a rejected suffix replays linear state
    /// from verification intermediates instead of paying another target pass.
    private let headStepCostRatio: Double

    private var positionAcceptance: [Double]

    init(maxDraftDepth: Int, headStepCostRatio: Double = 0.18) {
        self.headStepCostRatio = headStepCostRatio
        positionAcceptance = (0..<max(0, maxDraftDepth)).map { index in
            0.85 * pow(0.98, Double(index))
        }
    }

    static func configuredCostRatio(environment: [String: String] = ProcessInfo.processInfo.environment) -> Double {
        guard let raw = environment["MERERUN_Q35_MTP_HEAD_COST_RATIO"],
              let value = Double(raw), value.isFinite, value > 0, value <= 5 else {
            return 0.18
        }
        return value
    }

    func draftDepth(offeredDepth: Int) -> Int {
        let cap = min(max(0, offeredDepth), positionAcceptance.count)
        guard cap > 0 else { return 0 }

        let cost = headStepCostRatio
        var reach = 1.0
        var expectedAccepted = 0.0
        var depth = 0
        while depth < cap {
            reach *= positionAcceptance[depth]
            let threshold = cost * (1.0 + expectedAccepted) / (1.0 + Double(depth) * cost)
            guard reach > threshold else { break }
            expectedAccepted += reach
            depth += 1
        }
        return depth
    }

    mutating func record(acceptedDrafts: Int, drafted: Int) {
        let observedDrafts = min(max(0, drafted), positionAcceptance.count)
        guard observedDrafts > 0 else { return }

        let accepted = min(max(0, acceptedDrafts), observedDrafts)
        let alpha = Self.acceptanceAlpha
        for index in 0..<accepted {
            positionAcceptance[index] += alpha * (1.0 - positionAcceptance[index])
        }

        if accepted < observedDrafts {
            positionAcceptance[accepted] += alpha * (0.0 - positionAcceptance[accepted])
        } else if accepted < positionAcceptance.count,
                  positionAcceptance[accepted] < Self.inferredAcceptanceCeiling {
            // A fully accepted round is weak evidence that the next position is
            // worth trying. Cap transferred optimism because it was not observed.
            positionAcceptance[accepted] += alpha * (
                Self.inferredAcceptanceCeiling - positionAcceptance[accepted]
            )
        }
    }
}
