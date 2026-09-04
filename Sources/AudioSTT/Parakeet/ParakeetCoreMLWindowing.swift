import Foundation

enum ParakeetCoreMLWindowing {
    static let windowSeconds: TimeInterval = 15
    static let overlapSeconds: TimeInterval = 2
    static let maximumOverlapSeconds: TimeInterval = 8

    static func sampleRanges(samples: [Float], sampleRate: Int) -> [Range<Int>] {
        guard !samples.isEmpty, sampleRate > 0 else { return [] }
        let windowSamples = Int(windowSeconds * TimeInterval(sampleRate))
        let overlapSamples = Int(overlapSeconds * TimeInterval(sampleRate))
        precondition(windowSamples > overlapSamples)

        var ranges: [Range<Int>] = []
        var start = 0
        while start < samples.count {
            let end = min(start + windowSamples, samples.count)
            ranges.append(start..<end)
            guard end < samples.count else { break }
            start = quietBoundary(samples: samples, windowEnd: end, sampleRate: sampleRate)
        }
        return ranges
    }

    private static func quietBoundary(samples: [Float], windowEnd: Int, sampleRate: Int) -> Int {
        let earliest = windowEnd - Int(maximumOverlapSeconds * TimeInterval(sampleRate))
        let latest = windowEnd - Int(overlapSeconds * TimeInterval(sampleRate))
        let interval = max(1, sampleRate / 5)
        let step = max(1, sampleRate / 100)
        var selected = latest
        // Starting inside a word can make Parakeet skip an entire window.
        // Prefer the center of a quiet 200 ms interval below -40 dBFS RMS;
        // preserve the two-second overlap when no such interval is available.
        var lowestMeanSquare = 0.0001
        var energy = samples[earliest..<(earliest + interval)].reduce(0.0) {
            $0 + Double($1) * Double($1)
        }
        var candidate = earliest
        while candidate <= latest {
            let meanSquare = energy / Double(interval)
            if meanSquare < lowestMeanSquare {
                lowestMeanSquare = meanSquare
                selected = min(candidate + interval / 2, latest)
            }
            guard candidate + step <= latest else { break }
            for index in candidate..<(candidate + step) {
                let leaving = Double(samples[index])
                let entering = Double(samples[index + interval])
                energy += entering * entering - leaving * leaving
            }
            candidate += step
        }
        return selected
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
