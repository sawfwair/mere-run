import Foundation
import MereRunTensor

/// Prices parent-closed prefixes using observed costs and conditional landing
/// counts. Unmeasured costs interpolate between useful 1-, 7-, and 15-node rounds.
struct Q35DFlash2TreePricing {
    private struct Bucket { var offered = 0.0; var landed = 0.0 }
    private var buckets = Array(repeating: Array(repeating: Bucket(), count: 16), count: 2)
    private var samples = Array(repeating: 0, count: 16)
    private(set) var prices: [Double?] = Array(repeating: nil, count: 16)
    private var previous: Int?

    mutating func select(_ proposal: DFlash2TreeProposal, remaining: Int) -> DFlash2TreeProposal {
        let available = proposal.tokens.count - 1
        guard remaining >= 15 && available == 15 else { return proposal }
        if let probe = [1, 7, 15].first(where: { samples[$0] < 2 }) { return proposal.prefix(nodes: probe) }
        var chance = [1.0], expected = 1.0, best = 1, bestRate = 0.0
        for row in 1...available {
            let own = proposal.logProbabilities[row] - proposal.logProbabilities[proposal.parents[row]]
            let bucket = buckets[proposal.copyRows.contains(row) ? 1 : 0][Self.bucket(own)]
            let conditional = (bucket.landed + 2 * exp(own)) / (bucket.offered + 2)
            chance.append(chance[proposal.parents[row]] * conditional)
            expected += chance[row]
            let rate = expected / estimate(row)
            if rate > bestRate { best = row; bestRate = rate }
        }
        return proposal.prefix(nodes: best)
    }

    mutating func observe(_ proposal: DFlash2TreeProposal, acceptedPath: [Int]) {
        let kept = Set(acceptedPath)
        for row in proposal.parents.indices.dropFirst() where kept.contains(proposal.parents[row]) {
            let own = proposal.logProbabilities[row] - proposal.logProbabilities[proposal.parents[row]]
            let source = proposal.copyRows.contains(row) ? 1 : 0, index = Self.bucket(own)
            buckets[source][index].offered += 1
            if kept.contains(row) { buckets[source][index].landed += 1 }
        }
    }

    mutating func record(nodes: Int, seconds: Double) {
        precondition((1...15).contains(nodes) && seconds > 0 && seconds.isFinite)
        samples[nodes] += 1
        // The first round at a changed size may join the previous size's keep.
        if previous == nodes {
            prices[nodes] = prices[nodes].map { 0.75 * $0 + 0.25 * seconds } ?? seconds
        }
        previous = nodes
    }

    private func estimate(_ nodes: Int) -> Double {
        if let price = prices[nodes] { return price }
        let lower = (1..<nodes).last { prices[$0] != nil }
        let upper = ((nodes + 1)...15).first { prices[$0] != nil }
        if let lower, let upper {
            let fraction = Double(nodes - lower) / Double(upper - lower)
            return prices[lower]! * (1 - fraction) + prices[upper]! * fraction
        }
        return prices[lower ?? upper!]!
    }

    private static func bucket(_ own: Double) -> Int { min(15, max(0, Int(-own * 2))) }
}

/// An incremental four-token index over confirmed token IDs. It never appends
/// unverified proposals; a returned continuation is verified like every draft.
struct Q35ContextCopyIndex {
    private struct Key: Hashable { let a: Int; let b: Int; let c: Int; let d: Int }
    private var tokens: [Int] = []
    private var previous: [Key: Int] = [:]
    private var match: Int?

    init(tokens: [Int] = []) { append(tokens) }

    mutating func append(_ confirmed: [Int]) {
        for token in confirmed {
            tokens.append(token)
            guard tokens.count >= 4 else { continue }
            let end = tokens.count
            let key = Key(a: tokens[end - 4], b: tokens[end - 3], c: tokens[end - 2], d: tokens[end - 1])
            match = previous.updateValue(end, forKey: key)
        }
    }

    func continuation(limit: Int) -> [Int] {
        guard let match else { return [] }
        return Array(tokens[match..<min(tokens.count, match + limit)])
    }
}
