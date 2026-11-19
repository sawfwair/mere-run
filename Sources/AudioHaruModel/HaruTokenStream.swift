import Foundation
import MLX
import MLXRandom

struct HaruTokenStream {
    private let silent: Set<Int>
    let exempt: Set<Int>
    private var history: [Int] = []
    private var pendingText: [Int] = []
    private var inGap = false

    init(pad: Int, eop: Int, eos: [Int]) {
        silent = [pad, eop]
        exempt = silent.union(eos)
    }

    func penalizeRepeats(in logits: MLXArray) {
        guard let last = history.last, history.count > 1 else { return }
        let end = history.count - 1
        var penalties: [Int: Float] = [:]
        for index in max(0, end - 1_024)..<end where history[index] == last {
            var length = 1
            while length < 32, index >= length, history[index - length] == history[end - length] {
                length += 1
            }
            let token = history[index + 1]
            let penalty = min(Float(32), 0.8 * pow(Float(1.75), Float(length - 1)))
            penalties[token] = max(penalties[token, default: 0], penalty)
        }
        guard !penalties.isEmpty else { return }
        let keys = penalties.keys.sorted()
        let ids = MLXArray(keys.map(Int32.init))
        logits[ids] = logits[ids] - MLXArray(keys.map { penalties[$0, default: 0] })
    }

    mutating func append(_ token: Int, decode: ([Int]) -> String) -> String {
        if silent.contains(token) {
            defer { inGap = true }
            return inGap ? "" : " "
        }
        inGap = false
        if !exempt.contains(token) {
            history.append(token)
            if history.count > 1_057 { history.removeFirst() }
        }
        pendingText.append(token)
        let text = decode(pendingText)
        if text.hasSuffix("\u{FFFD}") { return "" }
        pendingText.removeAll(keepingCapacity: true)
        return text
    }

    static func sample(_ logits: MLXArray, temperature: Float, key: MLXArray) -> MLXArray {
        if temperature == 0 { return argMax(logits) }
        let topK = min(25, logits.dim(0))
        let candidates = argPartition(-logits, kth: topK - 1)[..<topK]
        let logitsForCandidates = logits[candidates]
        let values = (logitsForCandidates - logitsForCandidates.max()) / temperature
        return candidates[MLXRandom.categorical(values, key: key)]
    }
}
