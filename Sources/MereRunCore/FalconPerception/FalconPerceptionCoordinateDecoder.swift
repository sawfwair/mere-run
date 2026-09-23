/// Falcon's coordinate-token policy, applied before feeding coordinates into the next token.
/// History includes every coordinate token in one query, even without a completed detection.
struct FalconPerceptionCoordinateDecoder {
    struct Coordinate: Equatable {
        let x: Double
        let y: Double
    }

    private(set) var history: [Coordinate] = []

    mutating func decode(logits: [Float]) -> Coordinate {
        precondition(logits.count >= 4 && logits.count.isMultiple(of: 2))
        let binCount = logits.count / 2
        var scores = logits
        var coordinate = Coordinate(x: 0, y: 0)
        // Match the pinned reference's strict 1% threshold and bounded retry loop.
        // The final candidate is retained even if all 100 attempts repeat.
        for _ in 0..<100 {
            let xBin = Self.argmax(scores, start: 0, count: binCount)
            let yBin = Self.argmax(scores, start: binCount, count: binCount)
            coordinate = Coordinate(
                x: Double(xBin) / Double(binCount - 1),
                y: Double(yBin) / Double(binCount - 1)
            )
            let repeated = history.contains {
                abs($0.x - coordinate.x) < 0.01 && abs($0.y - coordinate.y) < 0.01
            }
            if !repeated { break }
            scores[xBin] = -.infinity
            scores[binCount + yBin] = -.infinity
        }
        // Keep Double history: the reference stores Python bin ratios before tensor casting.
        history.append(coordinate)
        return coordinate
    }

    private static func argmax(_ values: [Float], start: Int, count: Int) -> Int {
        var best = 0
        for index in 1..<count {
            let candidate = values[start + index]
            let current = values[start + best]
            // Reference argmax chooses the first maximum, including the first NaN.
            if (candidate.isNaN && !current.isNaN) || candidate > current {
                best = index
            }
        }
        return best
    }
}
