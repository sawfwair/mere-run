import Foundation
import AudioCore

public struct ParakeetAlignedToken: Sendable, Hashable {
    public let id: Int
    public let text: String
    public var start: TimeInterval
    public var duration: TimeInterval

    public init(id: Int, text: String, start: TimeInterval, duration: TimeInterval) {
        self.id = id
        self.text = text
        self.start = start
        self.duration = duration
    }

    public var end: TimeInterval {
        start + duration
    }
}

public struct ParakeetAlignedSentence: Sendable, Hashable {
    public let text: String
    public let tokens: [ParakeetAlignedToken]

    public init(text: String, tokens: [ParakeetAlignedToken]) {
        self.text = text
        self.tokens = tokens.sorted { $0.start < $1.start }
    }

    public var start: TimeInterval {
        tokens.first?.start ?? 0
    }

    public var end: TimeInterval {
        tokens.last?.end ?? 0
    }

    public var duration: TimeInterval {
        max(0, end - start)
    }
}

public struct ParakeetAlignedResult: Sendable, Hashable {
    public let text: String
    public let sentences: [ParakeetAlignedSentence]

    public init(text: String, sentences: [ParakeetAlignedSentence]) {
        self.text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sentences = sentences
    }
}

public enum ParakeetAlignment {
    public static func tokensToSentences(_ tokens: [ParakeetAlignedToken]) -> [ParakeetAlignedSentence] {
        guard !tokens.isEmpty else { return [] }

        var sentences: [ParakeetAlignedSentence] = []
        var current: [ParakeetAlignedToken] = []

        for (index, token) in tokens.enumerated() {
            current.append(token)

            let next = index + 1 < tokens.count ? tokens[index + 1] : nil
            if isSentenceBoundary(tokenText: token.text, nextTokenText: next?.text) {
                let sentenceText = current.map(\.text).joined()
                sentences.append(ParakeetAlignedSentence(text: sentenceText, tokens: current))
                current.removeAll(keepingCapacity: true)
            }
        }

        if !current.isEmpty {
            let sentenceText = current.map(\.text).joined()
            sentences.append(ParakeetAlignedSentence(text: sentenceText, tokens: current))
        }

        return sentences
    }

    public static func sentencesToResult(_ sentences: [ParakeetAlignedSentence]) -> ParakeetAlignedResult {
        ParakeetAlignedResult(text: sentences.map(\.text).joined(), sentences: sentences)
    }

    public static func mergeLongestContiguous(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        overlapDuration: TimeInterval
    ) throws -> [ParakeetAlignedToken] {
        guard !a.isEmpty else { return b }
        guard !b.isEmpty else { return a }

        let aEnd = a[a.count - 1].end
        let bStart = b[0].start
        if aEnd <= bStart {
            return a + b
        }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }

        let enoughPairs = max(1, overlapA.count / 2)

        if overlapA.count < 2 || overlapB.count < 2 {
            return midpointMerge(a, b, aEnd: aEnd, bStart: bStart)
        }

        var best: [(Int, Int)] = []
        for i in overlapA.indices {
            for j in overlapB.indices {
                guard overlapA[i].id == overlapB[j].id,
                      Swift.abs(overlapA[i].start - overlapB[j].start) < overlapDuration / 2 else {
                    continue
                }

                var chain: [(Int, Int)] = []
                var k = i
                var l = j
                while k < overlapA.count,
                      l < overlapB.count,
                      overlapA[k].id == overlapB[l].id,
                      Swift.abs(overlapA[k].start - overlapB[l].start) < overlapDuration / 2 {
                    chain.append((k, l))
                    k += 1
                    l += 1
                }

                if chain.count > best.count {
                    best = chain
                }
            }
        }

        guard best.count >= enoughPairs else {
            throw ParakeetAlignmentError.noContiguousOverlap
        }

        return stitchMerge(a, b, overlapA: overlapA, overlapPairs: best)
    }

    public static func mergeLongestCommonSubsequence(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        overlapDuration: TimeInterval
    ) -> [ParakeetAlignedToken] {
        guard !a.isEmpty else { return b }
        guard !b.isEmpty else { return a }

        let aEnd = a[a.count - 1].end
        let bStart = b[0].start
        if aEnd <= bStart {
            return a + b
        }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }

        if overlapA.count < 2 || overlapB.count < 2 {
            return midpointMerge(a, b, aEnd: aEnd, bStart: bStart)
        }

        var dp = Array(
            repeating: Array(repeating: 0, count: overlapB.count + 1),
            count: overlapA.count + 1
        )

        for i in 1...overlapA.count {
            for j in 1...overlapB.count {
                if overlapA[i - 1].id == overlapB[j - 1].id,
                   Swift.abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2 {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = overlapA.count
        var j = overlapB.count
        while i > 0, j > 0 {
            if overlapA[i - 1].id == overlapB[j - 1].id,
               Swift.abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2 {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        pairs.reverse()
        guard !pairs.isEmpty else {
            return midpointMerge(a, b, aEnd: aEnd, bStart: bStart)
        }

        return stitchMerge(a, b, overlapA: overlapA, overlapPairs: pairs)
    }

    public static func toASRTokenAlignments(_ tokens: [ParakeetAlignedToken]) -> [ASRTokenAlignment] {
        tokens.map {
            ASRTokenAlignment(
                id: $0.id,
                text: $0.text,
                startSeconds: $0.start,
                durationSeconds: $0.duration,
                endSeconds: $0.end
            )
        }
    }

    public static func toASRSentenceAlignments(_ sentences: [ParakeetAlignedSentence]) -> [ASRSentenceAlignment] {
        sentences.map { sentence in
            ASRSentenceAlignment(
                text: sentence.text,
                startSeconds: sentence.start,
                durationSeconds: sentence.duration,
                endSeconds: sentence.end,
                tokens: toASRTokenAlignments(sentence.tokens)
            )
        }
    }

    private static func stitchMerge(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        overlapA: [ParakeetAlignedToken],
        overlapPairs: [(Int, Int)]
    ) -> [ParakeetAlignedToken] {
        let aStartIndex = a.count - overlapA.count
        let lcsA = overlapPairs.map { aStartIndex + $0.0 }
        let lcsB = overlapPairs.map { $0.1 }

        var result: [ParakeetAlignedToken] = []
        result.reserveCapacity(a.count + b.count)

        if let firstA = lcsA.first {
            result.append(contentsOf: a[..<firstA])
        }

        for index in overlapPairs.indices {
            let idxA = lcsA[index]
            let idxB = lcsB[index]
            result.append(a[idxA])

            if index < overlapPairs.count - 1 {
                let nextA = lcsA[index + 1]
                let nextB = lcsB[index + 1]
                let gapA = Array(a[(idxA + 1)..<nextA])
                let gapB = Array(b[(idxB + 1)..<nextB])
                if gapB.count > gapA.count {
                    result.append(contentsOf: gapB)
                } else {
                    result.append(contentsOf: gapA)
                }
            }
        }

        if let lastB = lcsB.last {
            result.append(contentsOf: b[(lastB + 1)...])
        }

        return result
    }

    private static func midpointMerge(
        _ a: [ParakeetAlignedToken],
        _ b: [ParakeetAlignedToken],
        aEnd: TimeInterval,
        bStart: TimeInterval
    ) -> [ParakeetAlignedToken] {
        let cutoff = (aEnd + bStart) / 2
        return a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }
    }

    private static func isSentenceBoundary(tokenText: String, nextTokenText: String?) -> Bool {
        if tokenText.contains("!") || tokenText.contains("?") || tokenText.contains("。") || tokenText.contains("？") || tokenText.contains("！") {
            return true
        }

        if tokenText.contains(".") {
            guard let nextTokenText else { return true }
            return nextTokenText.contains(" ")
        }

        return false
    }
}

public enum ParakeetAlignmentError: LocalizedError {
    case noContiguousOverlap

    public var errorDescription: String? {
        switch self {
        case .noContiguousOverlap:
            return "No sufficiently contiguous overlap was found between token windows."
        }
    }
}
