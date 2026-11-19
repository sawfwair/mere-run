/// Learns width-specific work and prices from useful, target-confirmed rounds.
/// The first round after a width change is not a price sample: it may still
/// wait for the preceding width's asynchronous state commit.
struct Q35DFlash2TreeCostPolicy {
    private var counts = Array(repeating: 0, count: 16)
    private var weights = Array(repeating: 0.0, count: 16)
    private var tokens = Array(repeating: 0.0, count: 16)
    private var prices: [Double?] = Array(repeating: nil, count: 16)
    private var previous: Int?
    private var consecutive = 0
    private var rounds = 0

    var secondsByNodeCount: [Double?] { prices }

    func nodeLimit(remaining: Int) -> Int {
        guard remaining >= 15 else { return 7 }
        if counts[7] < 2 { return 7 }
        if counts[15] < 2 { return 15 }
        if consecutive < 2 { return previous! }
        let narrowRate = tokens[7] / weights[7] / prices[7]!
        let wideRate = tokens[15] / weights[15] / prices[15]!
        let best = wideRate > 1.05 * narrowRate ? 15 : 7
        let alternate = best == 7 ? 15 : 7
        let bestRate = best == 7 ? narrowRate : wideRate
        // Sparse probes can recover after a change in the continuation, but
        // cannot pay when even perfect acceptance loses to the current rate.
        if rounds >= 64, rounds.isMultiple(of: 64),
           Double(alternate + 1) / prices[alternate]! > 1.05 * bestRate {
            return alternate
        }
        return best
    }

    mutating func record(nodes: Int, confirmedTokens: Int, seconds: Double) {
        counts[nodes] += 1
        weights[nodes] = 0.75 * weights[nodes] + 1
        tokens[nodes] = 0.75 * tokens[nodes] + Double(confirmedTokens)
        if previous == nodes {
            prices[nodes] = prices[nodes].map { 0.75 * $0 + 0.25 * seconds } ?? seconds
            consecutive += 1
        } else {
            consecutive = 1
        }
        previous = nodes
        rounds += 1
    }
}
