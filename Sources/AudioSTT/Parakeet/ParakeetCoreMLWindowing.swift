import Foundation

enum ParakeetCoreMLWindowing {
    static let windowSeconds: TimeInterval = 15
    static let overlapSeconds: TimeInterval = 2

    static func sampleRanges(sampleCount: Int, sampleRate: Int) -> [Range<Int>] {
        guard sampleCount > 0, sampleRate > 0 else { return [] }
        let windowSamples = Int(windowSeconds * TimeInterval(sampleRate))
        let overlapSamples = Int(overlapSeconds * TimeInterval(sampleRate))
        precondition(windowSamples > overlapSamples)

        var ranges: [Range<Int>] = []
        var start = 0
        while start < sampleCount {
            let end = min(start + windowSamples, sampleCount)
            ranges.append(start..<end)
            guard end < sampleCount else { break }
            start = end - overlapSamples
        }
        return ranges
    }

    static func shiftedTokens(
        from result: ParakeetAlignedResult,
        by offset: TimeInterval
    ) -> [ParakeetAlignedToken] {
        result.sentences.flatMap(\.tokens).map { token in
            ParakeetAlignedToken(
                id: token.id,
                text: token.text,
                start: token.start + offset,
                duration: token.duration
            )
        }
    }
}
