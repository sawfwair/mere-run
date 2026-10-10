import Foundation
import AudioCore
import AudioWhistleModel

/// Cross-attention DTW on the 80 ms encoder grid. These are acoustic estimates, not forced alignment.
enum WhistleAlignment {
    static func frames(attention: [[Float]], frames: Int) -> [Int] {
        let count = attention.count
        guard count > 0, frames > 0 else { return [] }
        let heads = attention[0].count / frames
        precondition(heads > 0 && attention.allSatisfy { $0.count == heads * frames })
        var cost = Array(repeating: Float(0), count: count * frames)
        for head in 0..<heads {
            for frame in 0..<frames {
                let index = head * frames + frame
                let mean = attention.reduce(Float(0)) { $0 + $1[index] } / Float(count)
                let variance = attention.reduce(Float(0)) { $0 + pow($1[index] - mean, 2) } / Float(count)
                let scale = 1 / (sqrt(variance) + 1e-9) / Float(heads)
                for token in 0..<count { cost[token * frames + frame] -= (attention[token][index] - mean) * scale }
            }
        }
        var previous = Array(repeating: Float.infinity, count: frames + 1)
        previous[0] = 0
        var directions = Array(repeating: UInt8(0), count: count * frames)
        for token in 0..<count {
            var current = Array(repeating: Float.infinity, count: frames + 1)
            for frame in 0..<frames {
                let diagonal = previous[frame], up = previous[frame + 1], left = current[frame]
                let direction: UInt8 = diagonal <= up && diagonal <= left ? 0 : (up <= left ? 1 : 2)
                directions[token * frames + frame] = direction
                current[frame + 1] = cost[token * frames + frame] + min(diagonal, min(up, left))
            }
            previous = current
        }
        var aligned = Array(repeating: 0, count: count)
        var token = count - 1, frame = frames - 1
        while token >= 0, frame >= 0 {
            aligned[token] = frame
            let direction = directions[token * frames + frame]
            if direction != 2 { token -= 1 }
            if direction != 1 { frame -= 1 }
        }
        return aligned
    }

    static func words(tokens: [Int], attention: [[Float]], frames: Int, vocabulary: WhistleVocabulary,
                      offset: Double, duration: Double) throws -> [ASRTokenAlignment] {
        let positions = Self.frames(attention: attention, frames: frames)
        guard positions.count >= tokens.count else { return [] }
        var groups: [(start: Int, tokens: [Int])] = []
        for (index, token) in tokens.enumerated() {
            if groups.isEmpty || vocabulary.startsWord(token) { groups.append((index, [token])) }
            else { groups[groups.count - 1].tokens.append(token) }
        }
        return try groups.enumerated().compactMap { index, group in
            let text = try vocabulary.decode(group.tokens).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            let start = min(duration, Double(positions[group.start]) * 0.08)
            let next = index + 1 < groups.count ? groups[index + 1].start : min(tokens.count, positions.count - 1)
            let end = max(start, min(duration, Double(positions[next]) * 0.08))
            return ASRTokenAlignment(text: text, startSeconds: offset + start, durationSeconds: end - start)
        }
    }
}
