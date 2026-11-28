import Foundation
import MereRunEnvironment

package enum Q35SampledTreeMode: String, Sendable {
    case position = "1"
    case vocabulary

    package static func resolve(automatic: Bool, environment: [String: String] = ProcessInfo.processInfo.environment) -> Self? {
        if let value = environment["MERERUN_Q38_SAMPLED_TREE"] { return Self(rawValue: value) }
        return automatic && !MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_SAMPLED_CHAIN", default: false, environment: environment) ? .vocabulary : nil
    }

    static let gpuHeadOverride = MereRunEnvironmentSwitch.value("MERERUN_Q38_SAMPLED_GPU_HEAD")
}

/// The target and draft share one Gumbel variate for each output position and vocabulary ID.
/// Only the target's filtered distribution decides which token is emitted.
package struct Q35PositionSampler {
    package init(seed: UInt64, temperature: Double, topP: Double, topK: Int) {
        self.seed = seed
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    package var treeMode: Q35SampledTreeMode?
    package var thinkingLimit: Q35TreeThinkingLimit?
    var serialKeys: Q35SerialTreeKeys?
    let seed: UInt64
    let temperature: Double
    let topP: Double
    let topK: Int

    private func mix(_ input: UInt64) -> UInt64 {
        var x = input
        x = (x ^ (x >> 30)) &* 0xbf58476d1ce4e5b9
        x = (x ^ (x >> 27)) &* 0x94d049bb133111eb
        return x ^ (x >> 31)
    }

    func noise(position: Int, id: Int) -> Double {
        let base = mix(mix(seed &+ 0x9e3779b97f4a7c15) ^ (UInt64(position) &* 0xd1b54a32d192ed03))
        let u = min(Double(Float(UInt32(mix(base ^ UInt64(id)) >> 40)) + 0.5) * 0x1p-24, Double(Float(1).nextDown))
        return -log(-log(u))
    }

    /// Native projections return BF16 scores. Two passes keep the entire kth-score tie.
    private func ranked(_ row: UnsafeBufferPointer<UInt16>, descending: Bool = false) -> [(id: Int, score: Double)] {
        var histogram = [Int32](repeating: 0, count: 65_536)
        @inline(__always) func key(_ word: UInt16) -> Int {
            let bits = word == 0x8000 ? UInt16(0) : word
            return Int(bits & 0x8000 == 0 ? bits ^ 0x8000 : ~bits)
        }
        for word in row { histogram[key(word)] += 1 }
        var cutoff = histogram.count
        var total = 0
        repeat { cutoff -= 1; total += Int(histogram[cutoff]) } while total < topK
        var support: [(id: Int, score: Double)] = []
        support.reserveCapacity(total)
        for (id, word) in row.enumerated() where key(word) >= cutoff {
            let score = Double(Float(bitPattern: UInt32(word) << 16)) / temperature
            support.append((id, score))
        }
        support.sort {
            if $0.score == $1.score { return $0.id < $1.id }
            return descending ? $0.score > $1.score : $0.score < $1.score
        }
        return support
    }

    func support(_ row: UnsafeBufferPointer<UInt16>) -> [(id: Int, probability: Float)] {
        let order = ranked(row)
        let peak = Float(order.last!.score)
        let weights = order.map { exp(Float($0.score) - peak) }
        let total = weights.reduce(0, +)
        var cumulative: Float = 0, kept: [(id: Int, probability: Float)] = []
        for (i, item) in order.enumerated() {
            let p = weights[i] / total
            cumulative += p
            if cumulative > 1 - Float(topP) { kept.append((item.id, p)) }
        }
        let normalizer = kept.reduce(Float(0)) { $0 + $1.probability }
        return kept.map { ($0.id, $0.probability / normalizer) }.sorted { $0.id < $1.id }
    }

    func choose(_ row: UnsafeBufferPointer<UInt16>, position: Int) -> Int {
        if let serialKeys {
            let scores = ranked(row)
            let weights = scores.map { exp(Float($0.score - scores.last!.score)) }
            let total = weights.reduce(Float(0), +)
            var cumulative: Float = 0, best = -Float.infinity
            var choice = scores.last!.id
            for (index, item) in scores.enumerated() {
                let probability = weights[index] / total
                cumulative += probability
                if cumulative > 1 - Float(topP) {
                    let rank = serialKeys.order == .vocabulary ? item.id : row.count - scores.count + index
                    let race = log(probability) + Q35SerialTreeKeys.gumbel(
                        key: serialKeys[position], rank: rank, vocabulary: row.count)
                    if race > best || (serialKeys.order == .vocabulary && race == best && item.id < choice) {
                        best = race
                        choice = item.id
                    }
                }
            }
            return choice
        }
        // The keyed tree orders tied logits by token ID before either rank cutoff.
        let support = ranked(row, descending: true).prefix(topK)
        let peak = support.first!.score
        let weights = support.map { exp($0.score - peak) }
        let threshold = topP * weights.reduce(0, +)
        var sum = 0.0, best = -Double.infinity, token = support.first!.id
        for (i, item) in support.enumerated() {
            let value = item.score + noise(position: position, id: item.id)
            if value > best { best = value; token = item.id }
            sum += weights[i]
            if sum >= threshold { break }
        }
        return token
    }

    private static let draftNoise = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_DRAFT_NOISE"].flatMap(Double.init) ?? 0.7
    private static let draftScale = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_DRAFT_SCALE"].flatMap(Float.init) ?? 1

    func draftScores(_ scores: [Float], tokens: [Int32], depth: Int, width: Int, position: Int) -> [Float] {
        #if MERERUN_RELEASE_REPLAY
        if let trace = ProcessInfo.processInfo.environment["MERERUN_Q38_SAMPLED_DRAFT_TRACE"] {
            let file = URL(fileURLWithPath: trace).appendingPathComponent(String(position))
            try? tokens.withUnsafeBytes { try Data($0).write(to: file.appendingPathExtension("ids")) }
            try? scores.withUnsafeBytes { try Data($0).write(to: file.appendingPathExtension("scores")) }
        }
        #endif
        var result = scores
        for row in 0..<depth {
            let perturbation = (0..<width).map { rank -> Float in
                if let serialKeys {
                    return Float(Self.draftNoise) * Q35SerialTreeKeys.gumbel(key: serialKeys[position + row],
                        rank: serialKeys.order == .vocabulary ? Int(tokens[row * width + rank]) : 248_319 - rank,
                        vocabulary: 248_320)
                }
                return Float(Self.draftNoise * noise(position: position + row, id: Int(tokens[row * width + rank])))
            }
            for previous in 0..<width {
                let start = (row * width + previous) * width
                var largest = -Float.infinity
                for choice in 0..<width {
                    result[start + choice] = (scores[start + choice] / Float(temperature) + perturbation[choice]) / Self.draftScale
                    largest = max(largest, result[start + choice])
                }
                let denominator = log(result[start..<(start + width)].reduce(Float(0)) { $0 + exp($1 - largest) }) + largest
                for choice in 0..<width { result[start + choice] -= denominator }
            }
        }
        return result
    }
}
