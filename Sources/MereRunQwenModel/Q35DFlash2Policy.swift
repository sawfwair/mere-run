import Foundation

/// Full-accept forward costs, indexed by proposed token count. Zero is target-only
/// execution on the same backend. A loaded model keeps this profile fixed so
/// request scheduling does not make seeded sampling depend on live timers.
package struct Q35DFlash2Costs {
    package let secondsByDepth: [Double]
    package let rejectedSecondsByDepth: [Double]?

    package init(secondsByDepth: [Double], rejectedSecondsByDepth: [Double]? = nil) {
        precondition(secondsByDepth.count > 1 && secondsByDepth.allSatisfy { $0.isFinite && $0 > 0 })
        self.secondsByDepth = secondsByDepth
        if let rejectedSecondsByDepth {
            precondition(rejectedSecondsByDepth.count == secondsByDepth.count
                         && rejectedSecondsByDepth.allSatisfy { $0.isFinite && $0 > 0 })
        }
        self.rejectedSecondsByDepth = rejectedSecondsByDepth
    }

    package func limited(to depth: Int) -> Q35DFlash2Costs {
        guard depth + 1 < secondsByDepth.count else { return self }
        return Q35DFlash2Costs(secondsByDepth: Array(secondsByDepth.prefix(depth + 1)))
    }
}

/// Estimates the probability that each proposed prefix lands. Only proposal
/// work changes: the target backend and the sampling distribution stay fixed.
package struct Q35DFlash2Policy {
    private let costs: Q35DFlash2Costs
    private var landings: [Double]
    private var observations: [Int]
    private var rounds = 0
    private var widths: Q35DFlash2WidthPolicy?

    package init(costs: Q35DFlash2Costs) {
        self.costs = costs
        landings = (1..<costs.secondsByDepth.count).map { pow(0.8, Double($0)) }
        observations = Array(repeating: 1, count: landings.count)
        widths = costs.rejectedSecondsByDepth.map { Q35DFlash2WidthPolicy(full: costs.secondsByDepth, rejected: $0) }
    }

    package var supportsTargetOnlyProbes: Bool { widths != nil }

    package mutating func recordTargetOnly() { widths?.recordTargetOnly() }

    package func draftDepth(offeredDepth: Int) -> Int {
        if let widths { return widths.draftDepth(offeredDepth: offeredDepth) }
        let cap = min(offeredDepth, landings.count)
        guard cap > 0 else { return 0 }
        // Observe several full rounds before acting on the acceptance prior.
        guard rounds >= 4 else { return cap }
        var best = 0
        var rate = 1.05 / costs.secondsByDepth[0]
        var expected = 1.0
        var reach = 1.0
        for index in 0..<cap {
            reach = min(reach, landings[index] / Double(observations[index]))
            expected += reach
            let candidate = expected / costs.secondsByDepth[index + 1]
            if candidate > rate {
                rate = candidate
                best = index + 1
            }
        }
        return best
    }

    package mutating func record(proposed: Int, accepted: Int) {
        precondition(proposed > 0 && proposed <= landings.count && (0...proposed).contains(accepted))
        if widths != nil {
            widths?.record(proposed: proposed, accepted: accepted)
            return
        }
        rounds += 1
        for index in 0..<proposed {
            observations[index] += 1
            if index < accepted { landings[index] += 1 }
        }
    }
}
