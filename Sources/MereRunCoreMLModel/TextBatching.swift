import Foundation

/// Token-isolated packing for the fixed 256-token, eight-slot ANE graph.
public enum CoreMLTextBatching {
    public static let tokenCapacity = 256
    public static let slotCapacity = 8

    public static func groups(for sequences: [MultimodalSequence]) -> [[Int]] {
        var result: [[Int]] = [], current: [Int] = []
        var used = 0
        func flush() {
            if !current.isEmpty { result.append(current) }
            current = []; used = 0
        }
        for (index, sequence) in sequences.enumerated() {
            let count = sequence.ids.count
            guard count > 0, count <= tokenCapacity, sequence.slots.isEmpty, sequence.softTokens.isEmpty else {
                flush(); result.append([index]); continue
            }
            if used + count > tokenCapacity || current.count == slotCapacity { flush() }
            current.append(index); used += count
        }
        flush()
        return result
    }
}
