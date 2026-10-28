// Adapted from Gradium xn-ptts 49b9033e44ab0fbb935973eca3213757644d1bbb.
// MIT license; see THIRD_PARTY_NOTICES.md.
import Foundation

struct PhononTextChunk {
    let text: String
    let tokens: [Int]
    let tailFrames: Int
}

extension PhononTokenizer {
    /// Upstream's token-aligned sentence grouping, followed by its 200-token fit policy.
    func chunks(_ input: String) throws -> [PhononTextChunk] {
        let prepared = Self.prepare(input).text
        let tokens = try encode(prepared)
        let markers = try encode(".!...?")
        let ends = Set(markers.count > 1 ? Array(markers.dropFirst()) : markers)
        var boundaries = [0], previousWasEnd = false
        for (index, token) in tokens.enumerated() {
            if ends.contains(token) { previousWasEnd = true }
            else {
                if previousWasEnd { boundaries.append(index) }
                previousWasEnd = false
            }
        }
        boundaries.append(tokens.count)
        var grouped: [String] = [], current = "", count = 0
        for (start, end) in zip(boundaries, boundaries.dropFirst()) {
            let sentence = try decode(Array(tokens[start..<end]))
            let length = end - start
            if current.isEmpty { current = sentence; count = length }
            else if count + length > 50 {
                grouped.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = sentence; count = length
            } else { current += " " + sentence; count += length }
        }
        if !current.isEmpty { grouped.append(current.trimmingCharacters(in: .whitespacesAndNewlines)) }
        var pending = try grouped.reversed().map(makeChunk), fitted: [PhononTextChunk] = []
        while let chunk = pending.popLast() {
            if chunk.tokens.count <= 200 { fitted.append(chunk); continue }
            let words = chunk.text.split(whereSeparator: \.isWhitespace).map(String.init)
            let halves: [String]
            if words.count >= 2 {
                let cut = try cutPoint(words)
                halves = [words[..<cut].joined(separator: " "), words[cut...].joined(separator: " ")]
            } else {
                let scalars = Array(chunk.text.trimmingCharacters(in: .whitespacesAndNewlines).unicodeScalars)
                guard scalars.count > 1 else { throw PhononError.invalidRequest("A character exceeds Phonon's 200-token chunk limit.") }
                let middle = scalars.count / 2
                halves = [String(String.UnicodeScalarView(scalars[..<middle])), String(String.UnicodeScalarView(scalars[middle...]))]
            }
            for half in halves.reversed() { pending.append(try makeChunk(half)) }
        }
        return fitted
    }

    private func makeChunk(_ text: String) throws -> PhononTextChunk {
        let prepared = Self.prepare(text)
        return PhononTextChunk(text: text, tokens: try encode(prepared.text), tailFrames: prepared.tailFrames)
    }

    private func cutPoint(_ words: [String]) throws -> Int {
        var ends: [Int] = [], total = 0
        for word in words { total += try encode(word).count; ends.append(total) }
        let closers = Set("\"') ]”’»".filter { $0 != " " })
        func best(_ marks: Set<Character>?) -> Int? {
            var answer: Int?, imbalance = Int.max
            for index in 0..<(words.count - 1) {
                let end = ends[index]
                if let marks {
                    var word = words[index]
                    while let last = word.last, closers.contains(last) { word.removeLast() }
                    guard let last = word.last, marks.contains(last), end >= 50, total - end >= 50 else { continue }
                }
                let distance = abs(end - (total - end))
                if distance < imbalance { answer = index + 1; imbalance = distance }
            }
            return answer
        }
        guard let cut = best(Set(".!?")) ?? best(Set(",;:")) ?? best(nil) else {
            throw PhononError.invalidRequest("A text chunk cannot be split.")
        }
        return cut
    }
}
