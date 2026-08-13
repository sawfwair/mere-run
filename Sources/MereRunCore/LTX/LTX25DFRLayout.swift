import Foundation
import MLX

public struct LTX25DFROptions: Sendable, Equatable {
    public let temporalUpsampleRounds: Int
    public let detailingLoRAs: [LTXLoRAConfiguration]
    public let detailingReferenceDownscaleFactor: Int?

    public init(
        temporalUpsampleRounds: Int = 0,
        detailingLoRAs: [LTXLoRAConfiguration] = [],
        detailingReferenceDownscaleFactor: Int? = nil
    ) {
        precondition((0...2).contains(temporalUpsampleRounds), "temporalUpsampleRounds must be 0, 1, or 2")
        precondition(
            detailingReferenceDownscaleFactor.map { $0 > 0 } ?? true,
            "detailingReferenceDownscaleFactor must be positive"
        )
        self.temporalUpsampleRounds = temporalUpsampleRounds
        self.detailingLoRAs = detailingLoRAs
        self.detailingReferenceDownscaleFactor = detailingReferenceDownscaleFactor
    }

    public var resolvedDetailingReferenceDownscaleFactor: Int {
        if let detailingReferenceDownscaleFactor {
            return detailingReferenceDownscaleFactor
        }
        let metadataFactor = detailingLoRAs.first.map(ltxLoRAReferenceDownscaleFactor) ?? 1
        return metadataFactor == 1 ? 2 : metadataFactor
    }

    public var playbackRateMultiplier: Int {
        1 << temporalUpsampleRounds
    }
}

public enum LTX25DFRLayoutError: LocalizedError, Equatable {
    case invalidFrameCount(Int)
    case invalidTemporalScale(Int)
    case emptySeams
    case invalidSeams([Int])
    case invalidTileCount(Int)
    case tileCountMismatch(expected: Int, actual: Int)
    case invalidLatentShape([Int])
    case unexpectedTileFrames(expected: Int, actual: Int)
    case invalidDropPrefix(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidFrameCount(let value):
            return "DFR frame count must be 8n+1 and at least 9 (got \(value))."
        case .invalidTemporalScale(let value):
            return "DFR temporal scale must be positive (got \(value))."
        case .emptySeams:
            return "DFR requires at least one temporal seam."
        case .invalidSeams(let values):
            return "DFR seams must be increasing x8 borders ending on the final frame (got \(values))."
        case .invalidTileCount(let value):
            return "DFR tile count must be positive (got \(value))."
        case .tileCountMismatch(let expected, let actual):
            return "DFR expected \(expected) tile latents but received \(actual)."
        case .invalidLatentShape(let shape):
            return "DFR expected a five-dimensional NCTHW latent (got \(shape))."
        case .unexpectedTileFrames(let expected, let actual):
            return "DFR tile has \(actual) latent frames instead of \(expected)."
        case .invalidDropPrefix(let value):
            return "DFR tile drop prefix is outside its latent window (got \(value))."
        }
    }
}

public struct LTX25DFRCanvas: Sendable, Equatable {
    public let frameCount: Int
    public let segmentLength: Int
    public let keyframePositions: [Int]

    public init(frameCount: Int, segmentLength: Int, keyframePositions: [Int]) {
        self.frameCount = frameCount
        self.segmentLength = segmentLength
        self.keyframePositions = keyframePositions
    }
}

public struct LTX25DFRTileRange: Sendable, Equatable {
    public let pixelStart: Int
    public let pixelEnd: Int
    public let latentStart: Int
    public let latentEndExclusive: Int
    public let anchorKeyframes: [Int]
    public let slotKeyframes: [Int]
    public let dropLatentPrefix: Int

    public init(
        pixelStart: Int,
        pixelEnd: Int,
        latentStart: Int,
        latentEndExclusive: Int,
        anchorKeyframes: [Int],
        slotKeyframes: [Int],
        dropLatentPrefix: Int
    ) {
        self.pixelStart = pixelStart
        self.pixelEnd = pixelEnd
        self.latentStart = latentStart
        self.latentEndExclusive = latentEndExclusive
        self.anchorKeyframes = anchorKeyframes
        self.slotKeyframes = slotKeyframes
        self.dropLatentPrefix = dropLatentPrefix
    }
}

public enum LTX25DFRLayout {
    public static let segmentCandidates = [24, 32]
    public static let tileLeadSegments = 1

    public static func chooseSegmentLength(contentFrames: Int) throws -> Int {
        guard contentFrames > 0 else {
            throw LTX25DFRLayoutError.invalidFrameCount(contentFrames + 1)
        }
        return segmentCandidates.min { lhs, rhs in
            let lhsPad = (lhs - contentFrames % lhs) % lhs
            let rhsPad = (rhs - contentFrames % rhs) % rhs
            if lhsPad == rhsPad {
                return lhs > rhs
            }
            return lhsPad < rhsPad
        }!
    }

    public static func resolveCanvas(
        frameCount: Int,
        temporalScale: Int = 8
    ) throws -> LTX25DFRCanvas {
        guard temporalScale > 0 else {
            throw LTX25DFRLayoutError.invalidTemporalScale(temporalScale)
        }
        guard frameCount >= temporalScale + 1, (frameCount - 1).isMultiple(of: temporalScale) else {
            throw LTX25DFRLayoutError.invalidFrameCount(frameCount)
        }
        let content = frameCount - 1
        let segment = try chooseSegmentLength(contentFrames: content)
        let paddedContent = content + (segment - content % segment) % segment
        let positions = stride(from: segment, through: paddedContent, by: segment).map { $0 }
        return LTX25DFRCanvas(
            frameCount: paddedContent + 1,
            segmentLength: segment,
            keyframePositions: positions
        )
    }

    public static func pixelToLatentIndex(
        _ pixelFrame: Int,
        temporalScale: Int = 8
    ) throws -> Int {
        guard temporalScale > 0 else {
            throw LTX25DFRLayoutError.invalidTemporalScale(temporalScale)
        }
        guard pixelFrame >= 0,
              pixelFrame == 0 || pixelFrame.isMultiple(of: temporalScale) else {
            throw LTX25DFRLayoutError.invalidSeams([pixelFrame])
        }
        return pixelFrame / temporalScale
    }

    public static func tileRanges(
        seamPositions: [Int],
        frameCount: Int,
        tileCount: Int,
        temporalScale: Int = 8,
        leadSegments: Int = tileLeadSegments
    ) throws -> [LTX25DFRTileRange] {
        guard temporalScale > 0 else {
            throw LTX25DFRLayoutError.invalidTemporalScale(temporalScale)
        }
        guard tileCount > 0, leadSegments > 0 else {
            throw LTX25DFRLayoutError.invalidTileCount(tileCount)
        }
        guard !seamPositions.isEmpty else {
            throw LTX25DFRLayoutError.emptySeams
        }
        let boundaries = [0] + seamPositions
        guard frameCount >= 2,
              seamPositions.last == frameCount - 1,
              zip(boundaries, boundaries.dropFirst()).allSatisfy({ lower, upper in
                  let span = upper - lower
                  return span >= temporalScale * 2 && span.isMultiple(of: temporalScale)
              }) else {
            throw LTX25DFRLayoutError.invalidSeams(seamPositions)
        }

        let segmentCount = boundaries.count - 1
        let activeTileCount = min(tileCount, segmentCount)
        let base = segmentCount / activeTileCount
        let remainder = segmentCount % activeTileCount
        var ownedStart = 0
        var result: [LTX25DFRTileRange] = []
        result.reserveCapacity(activeTileCount)
        for tileIndex in 0..<activeTileCount {
            let ownedCount = base + (tileIndex < remainder ? 1 : 0)
            let ownedEnd = ownedStart + ownedCount
            let windowStart = max(0, ownedStart - (tileIndex > 0 ? leadSegments : 0))
            let pixelStart = boundaries[windowStart]
            let pixelEnd = boundaries[ownedEnd]
            let latentStart = try pixelToLatentIndex(pixelStart, temporalScale: temporalScale)
            var dropPrefix = try pixelToLatentIndex(
                boundaries[ownedStart],
                temporalScale: temporalScale
            ) - latentStart
            if ownedStart > 0 {
                dropPrefix += 1
            }
            result.append(
                LTX25DFRTileRange(
                    pixelStart: pixelStart,
                    pixelEnd: pixelEnd,
                    latentStart: latentStart,
                    latentEndExclusive: try pixelToLatentIndex(
                        pixelEnd,
                        temporalScale: temporalScale
                    ) + 1,
                    anchorKeyframes: Array(boundaries[windowStart...ownedEnd]).filter { $0 != 0 },
                    slotKeyframes: (windowStart..<ownedEnd).map {
                        (boundaries[$0] + boundaries[$0 + 1]) / 2
                    },
                    dropLatentPrefix: dropPrefix
                )
            )
            ownedStart = ownedEnd
        }
        return result
    }

    public static func remapPositionsToLocal(_ positions: [Int], pixelStart: Int) -> [Int] {
        positions.map { $0 - pixelStart }
    }

    public static func stitchTileLatents(
        _ tileLatents: [MLXArray],
        ranges: [LTX25DFRTileRange]
    ) throws -> MLXArray {
        guard tileLatents.count == ranges.count else {
            throw LTX25DFRLayoutError.tileCountMismatch(
                expected: ranges.count,
                actual: tileLatents.count
            )
        }
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(tileLatents.count)
        for (latent, range) in zip(tileLatents, ranges) {
            guard latent.ndim == 5 else {
                throw LTX25DFRLayoutError.invalidLatentShape(latent.shape)
            }
            let expectedFrames = range.latentEndExclusive - range.latentStart
            guard latent.dim(2) == expectedFrames else {
                throw LTX25DFRLayoutError.unexpectedTileFrames(
                    expected: expectedFrames,
                    actual: latent.dim(2)
                )
            }
            guard range.dropLatentPrefix >= 0,
                  range.dropLatentPrefix < latent.dim(2) else {
                throw LTX25DFRLayoutError.invalidDropPrefix(range.dropLatentPrefix)
            }
            pieces.append(latent[0..., 0..., range.dropLatentPrefix..., 0..., 0...])
        }
        return MLX.concatenated(pieces, axis: 2)
    }
}
