/// Estimates the value of each proposal from observed, position-specific
/// acceptance. The first draft has to pass before the deeper drafts can help.
struct Inkling2DSparkDepth {
    private static let decisionInterval = 4
    // On short-block verification, the anchor, draft setup, and target weight
    // reads cost much more than adding one more proposed position.
    private let baseCost: Double

    private let maximum: Int
    private let fixed: Bool
    private var attempts: [Double]
    private var successes: [Double]
    private var sinceDecision = 0
    private(set) var depth: Int

    init(maximum: Int, fixedDepth: Int?, baseCost: Double = 8) {
        self.baseCost = baseCost
        self.maximum = maximum
        fixed = fixedDepth != nil
        depth = fixedDepth ?? min(3, maximum)
        attempts = Array(repeating: 0, count: maximum)
        successes = attempts
    }

    mutating func observe(proposed: Int, accepted: Int) {
        guard !fixed, proposed == depth else { return }
        for position in 0..<maximum {
            attempts[position] *= 0.9
            successes[position] *= 0.9
        }
        // A rejection censors the remaining positions. Their conditional
        // acceptance is unknown because the verifier never reaches them.
        for position in 0..<min(proposed, accepted + 1) {
            attempts[position] += 1
            if position < accepted { successes[position] += 1 }
        }
        sinceDecision += 1
        guard sinceDecision == Self.decisionInterval else { return }
        sinceDecision = 0
        let scores = expectedRates(through: min(maximum, depth + 1))
        let candidates = max(1, depth - 1)...scores.count
        if let best = candidates.max(by: { scores[$0 - 1] < scores[$1 - 1] }),
           scores[best - 1] > scores[depth - 1] * 1.025 {
            depth = best
        }
    }

    private func expectedRates(through limit: Int) -> [Double] {
        var conditional = 0.75
        var survival = 1.0
        var tokens = 1.0
        return (0..<limit).map { position in
            let prior = position == 0 ? conditional : min(conditional, 0.65)
            conditional = (successes[position] + prior) / (attempts[position] + 1)
            survival *= conditional
            tokens += survival
            return tokens / (baseCost + Double(position + 1))
        }
    }
}
