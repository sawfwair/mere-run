import Foundation

/// Width-specific acceptance estimates avoid treating a prefix of a longer
/// masked draft as an observation of a different masked width. Costs remain
/// frozen; only target-confirmed counts update the request's estimates.
struct Q35DFlash2WidthPolicy {
    private let full: [Double]
    private let rejected: [Double]
    private var weight: [Double]
    private var accepted: [Double]
    private var complete: [Double]
    private var lastUsed: [Int]
    private var rounds = 0

    init(full: [Double], rejected: [Double]) {
        self.full = full
        self.rejected = rejected
        weight = Array(repeating: 0, count: full.count)
        accepted = weight
        complete = weight
        lastUsed = Array(repeating: -1, count: full.count)
    }

    func draftDepth(offeredDepth: Int) -> Int {
        let cap = min(offeredDepth, full.count - 1)
        guard cap > 0 else { return 0 }
        let candidates = [1, 3, 7, 15].filter { $0 <= cap }
        // Bootstrap on the trained width (or a shorter available band). Further
        // probes must have a price-model ceiling above the current best rate.
        if rounds == 0 {
            for depth in [7, 3, 1] where candidates.contains(depth) { return depth }
        }
        var best = 0
        var rate = 1.05 / full[0]
        for depth in candidates {
            let priorAccepted = (1...depth).reduce(0.0) { $0 + pow(0.8, Double($1)) }
            let expected = 1 + (0.1 * priorAccepted + accepted[depth]) / (0.1 + weight[depth])
            let fullProbability = (0.1 * pow(0.8, Double(depth)) + complete[depth]) / (0.1 + weight[depth])
            let seconds = full[depth] + (1 - fullProbability) * (rejected[depth] - full[depth])
            let candidateRate = expected / seconds
            if candidateRate > rate {
                rate = candidateRate
                best = depth
            }
        }
        let viable = candidates.filter { Double($0 + 1) / min(full[$0], rejected[$0]) > rate }
        for depth in [7, 15, 3, 1] where viable.contains(depth) && weight[depth] == 0 {
            return depth
        }
        // Do not spend a recovery round on a width that cannot beat the best
        // estimate even with perfect acceptance under the frozen price model.
        if rounds >= 4, rounds.isMultiple(of: 8), let probe = viable.min(by: { lastUsed[$0] < lastUsed[$1] }) {
            return probe
        }
        return best
    }

    mutating func recordTargetOnly() { rounds += 1 }

    mutating func record(proposed: Int, accepted count: Int) {
        weight[proposed] = 0.9 * weight[proposed] + 1
        accepted[proposed] = 0.9 * accepted[proposed] + Double(count)
        complete[proposed] = 0.9 * complete[proposed] + (count == proposed ? 1 : 0)
        lastUsed[proposed] = rounds
        rounds += 1
    }
}
