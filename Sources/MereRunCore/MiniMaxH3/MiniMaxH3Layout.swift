import Foundation
import MLX

public enum MiniMaxH3LayoutError: LocalizedError, Sendable {
    case invalidGeometry(String)

    public var errorDescription: String? {
        switch self {
        case .invalidGeometry(let reason): return "Invalid MiniMax-H3 geometry: \(reason)"
        }
    }
}

public enum MiniMaxH3Modality: Int32, Sendable {
    case video = 0
    case text = 1
    case audio = 2
}

public enum MiniMaxH3KeyframeAnchor: RawRepresentable, Codable, Sendable, Equatable {
    public typealias RawValue = String

    case history(latentFrameCount: Int)
    case first
    case frame(Int)
    case last

    public init?(rawValue: String) {
        switch rawValue {
        case "first": self = .first
        case "last": self = .last
        default:
            let components = rawValue.split(separator: ":", maxSplits: 1)
            guard components.count == 2,
                  let value = Int(components[1]) else {
                return nil
            }
            switch components[0] {
            case "history": self = .history(latentFrameCount: value)
            case "frame": self = .frame(value)
            default: return nil
            }
        }
    }

    public var rawValue: String {
        switch self {
        case .history(let count): "history:\(count)"
        case .first: "first"
        case .frame(let index): "frame:\(index)"
        case .last: "last"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        guard let value = Self(rawValue: rawValue) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "invalid MiniMax-H3 keyframe anchor: \(rawValue)"
            )
        }
        self = value
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    var latentFrameCount: Int {
        switch self {
        case .history(let count): count
        case .first, .frame, .last: 1
        }
    }
}

public enum MiniMaxH3AudioConditionAnchor: Sendable, Equatable {
    case history(latentFrameCount: Int)
    case first(latentFrameCount: Int)

    var latentFrameCount: Int {
        switch self {
        case .history(let count), .first(let count): count
        }
    }
}

public struct MiniMaxH3ConditionSegment: Sendable {
    public let modality: MiniMaxH3Modality
    public let packedRows: Range<Int>
    public let sourceRows: Range<Int>
}

public enum MiniMaxH3ReferenceKind: String, Codable, Sendable {
    case image
    case video
    case audio
}

public struct MiniMaxH3PreparedReferenceGeometry: Sendable {
    public let kind: MiniMaxH3ReferenceKind
    public let videoLatentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let audioLatentFrames: Int

    public init(
        kind: MiniMaxH3ReferenceKind,
        videoLatentFrames: Int = 0,
        latentHeight: Int = 0,
        latentWidth: Int = 0,
        audioLatentFrames: Int = 0
    ) {
        self.kind = kind
        self.videoLatentFrames = videoLatentFrames
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.audioLatentFrames = audioLatentFrames
    }

    var videoRowCount: Int {
        kind == .audio ? 0 : videoLatentFrames * (latentHeight / 2) * (latentWidth / 2)
    }

    var audioRowCount: Int { audioLatentFrames * 2 }
}

public struct MiniMaxH3PackedLayout: @unchecked Sendable {
    public let positions: MLXArray
    public let tokenTags: [Int32]
    public let textRows: Range<Int>
    public let conditionRows: Range<Int>
    public let conditionSegments: [MiniMaxH3ConditionSegment]
    public let conditionVideoRowCount: Int
    public let conditionAudioRowCount: Int
    public let targetAudioRows: Range<Int>
    public let targetVideoRows: Range<Int>
    public let videoLatentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let audioLatentFrames: Int

    public var sequenceLength: Int { tokenTags.count }
}

public enum MiniMaxH3Geometry {
    public static let framesPerSecond = 24
    public static let audioLatentsPerSecond = 40
    public static let videoFramesPerChunk = 17
    public static let videoLatentsPerChunk = 5
    public static let frameSpanPattern: [Double] = [1, 4, 4, 4, 4]
    public static let frameSpanScale = 5.0 / 3.0

    public static func alignFrameCount(_ frameCount: Int) throws -> Int {
        guard frameCount > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("frame count must be positive")
        }
        var aligned = frameCount
        while aligned % videoFramesPerChunk != videoLatentsPerChunk {
            aligned += 1
        }
        return aligned
    }

    public static func videoLatentFrameCount(for frameCount: Int) throws -> Int {
        guard frameCount % videoFramesPerChunk == videoLatentsPerChunk else {
            throw MiniMaxH3LayoutError.invalidGeometry("frame count must have form 17*n+5")
        }
        return ((frameCount - videoLatentsPerChunk) / videoFramesPerChunk) * videoLatentsPerChunk + 2
    }

    public static func audioLatentFrameCount(for frameCount: Int) -> Int {
        Int((Double(frameCount) / Double(framesPerSecond) * Double(audioLatentsPerSecond)).rounded())
    }

    public static func shiftedSigma(_ sigma: Float, from sourceShift: Float, to targetShift: Float) -> Float {
        let base = sigma / (sourceShift + sigma * (1 - sourceShift))
        return targetShift * base / (1 + (targetShift - 1) * base)
    }

    public static func shiftedSigmaSlope(_ sigma: Float, from sourceShift: Float, to targetShift: Float) -> Float {
        let base = sigma / (sourceShift + sigma * (1 - sourceShift))
        let numerator = targetShift * pow(1 + (sourceShift - 1) * base, 2)
        let denominator = sourceShift * pow(1 + (targetShift - 1) * base, 2)
        return numerator / denominator
    }

    public static func patchifyVideo(_ latent: MLXArray, patchSize: [Int] = [1, 2, 2]) -> MLXArray {
        precondition(latent.ndim == 5)
        let batch = latent.dim(0)
        let channels = latent.dim(1)
        let frames = latent.dim(2)
        let height = latent.dim(3)
        let width = latent.dim(4)
        let temporalPatch = patchSize[0]
        let heightPatch = patchSize[1]
        let widthPatch = patchSize[2]
        return latent
            .reshaped(
                batch, channels, frames / temporalPatch, temporalPatch,
                height / heightPatch, heightPatch, width / widthPatch, widthPatch
            )
            .transposed(0, 2, 4, 6, 1, 3, 5, 7)
            .reshaped(
                batch,
                (frames / temporalPatch) * (height / heightPatch) * (width / widthPatch),
                channels * temporalPatch * heightPatch * widthPatch
            )
    }

    public static func unpatchifyVideo(
        _ rows: MLXArray,
        frames: Int,
        height: Int,
        width: Int,
        channels: Int = 24,
        patchSize: [Int] = [1, 2, 2]
    ) -> MLXArray {
        let temporalPatch = patchSize[0]
        let heightPatch = patchSize[1]
        let widthPatch = patchSize[2]
        return rows
            .reshaped(
                rows.dim(0), frames / temporalPatch, height / heightPatch, width / widthPatch,
                channels, temporalPatch, heightPatch, widthPatch
            )
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(rows.dim(0), channels, frames, height, width)
    }

    public static func packAudio(_ latent: MLXArray) -> MLXArray {
        precondition(latent.ndim == 4 && latent.dim(0) == 1 && latent.dim(2) == 2)
        return latent[0].transposed(1, 2, 0).reshaped(2 * latent.dim(3), latent.dim(1))
    }

    public static func unpackAudio(_ rows: MLXArray) -> MLXArray {
        let frames = rows.dim(0) / 2
        return rows.reshaped(2, frames, rows.dim(1)).transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    public static func buildFL2VA(
        textTokenTags: [Int32],
        videoLatentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        audioLatentFrames: Int,
        keyframeAnchors: [MiniMaxH3KeyframeAnchor],
        audioConditionAnchors: [MiniMaxH3AudioConditionAnchor] = []
    ) throws -> MiniMaxH3PackedLayout {
        guard videoLatentFrames > 0, latentHeight > 0, latentWidth > 0, audioLatentFrames > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("all latent dimensions must be positive")
        }
        guard latentHeight.isMultiple(of: 2), latentWidth.isMultiple(of: 2) else {
            throw MiniMaxH3LayoutError.invalidGeometry("latent spatial dimensions must be divisible by 2")
        }
        let textCount = textTokenTags.count
        let rowsPerFrame = (latentHeight / 2) * (latentWidth / 2)
        let conditionVideoCount = keyframeAnchors.reduce(0) {
            $0 + $1.latentFrameCount * rowsPerFrame
        }
        let conditionAudioCount = audioConditionAnchors.reduce(0) {
            $0 + $1.latentFrameCount * 2
        }
        let audioCount = audioLatentFrames * 2
        let videoCount = videoLatentFrames * rowsPerFrame
        let textRows = 0..<textCount
        let conditionVideoRows = textCount..<(textCount + conditionVideoCount)
        let conditionAudioStart = conditionVideoRows.upperBound
        let conditionAudioRows = conditionAudioStart..<(conditionAudioStart + conditionAudioCount)
        let conditionRows = textCount..<conditionAudioRows.upperBound
        let audioRows = conditionRows.upperBound..<(conditionRows.upperBound + audioCount)
        let videoRows = audioRows.upperBound..<(audioRows.upperBound + videoCount)

        let squareRootArea = sqrt(Double(latentHeight * latentWidth))
        let heightGrid = spatialGrid(dimension: latentHeight, patch: 2, squareRootArea: squareRootArea)
        let widthGrid = spatialGrid(dimension: latentWidth, patch: 2, squareRootArea: squareRootArea)
        let frameGrid = heightGrid.flatMap { height in widthGrid.map { width in (height, width) } }
        var positions = Array(repeating: Float(0), count: videoRows.upperBound * 3)
        for row in textRows {
            positions[row * 3] = Float(row)
        }
        let historyLatentFrames = keyframeAnchors.reduce(0) { total, anchor in
            if case .history(let count) = anchor { return total + count }
            return total
        }
        let targetOrigin = Double(textCount) + temporalSpan(historyLatentFrames)
        let targetTemporal = temporalGrid(videoLatentFrames, origin: targetOrigin)
        var conditionVideoCursor = conditionVideoRows.lowerBound
        var historyOrigin = Double(textCount)
        for anchor in keyframeAnchors {
            let latentFrameCount = anchor.latentFrameCount
            let times: [Double] = switch anchor {
            case .history:
                temporalGrid(latentFrameCount, origin: historyOrigin)
            case .first:
                Array(targetTemporal.prefix(latentFrameCount))
            case .frame(let frameIndex):
                [targetOrigin + Double(frameIndex) * frameSpanScale]
            case .last:
                [targetOrigin + temporalSpan(videoLatentFrames) - frameSpanScale]
            }
            if case .history = anchor {
                historyOrigin += temporalSpan(latentFrameCount)
            }
            for latentFrame in 0..<latentFrameCount {
                for (offset, spatial) in frameGrid.enumerated() {
                    let row = conditionVideoCursor + latentFrame * rowsPerFrame + offset
                    positions[row * 3] = Float(times[latentFrame])
                    positions[row * 3 + 1] = Float(spatial.0)
                    positions[row * 3 + 2] = Float(spatial.1)
                }
            }
            conditionVideoCursor += latentFrameCount * rowsPerFrame
        }
        var conditionAudioCursor = conditionAudioRows.lowerBound
        var audioHistoryOrigin = Double(textCount)
        for anchor in audioConditionAnchors {
            let origin: Double = switch anchor {
            case .history: audioHistoryOrigin
            case .first: targetOrigin
            }
            let rows = conditionAudioCursor..<(conditionAudioCursor + anchor.latentFrameCount * 2)
            fillAudioPositions(
                &positions,
                rows: rows,
                frames: anchor.latentFrameCount,
                origin: origin,
                widthGrid: widthGrid
            )
            if case .history = anchor {
                audioHistoryOrigin += Double(anchor.latentFrameCount)
            }
            conditionAudioCursor = rows.upperBound
        }
        for channel in 0..<2 {
            for frame in 0..<audioLatentFrames {
                let row = audioRows.lowerBound + channel * audioLatentFrames + frame
                positions[row * 3] = Float(targetOrigin + Double(frame))
                positions[row * 3 + 2] = Float(channel == 0 ? widthGrid[0] : widthGrid[widthGrid.count - 1])
            }
        }
        for frame in 0..<videoLatentFrames {
            for (offset, spatial) in frameGrid.enumerated() {
                let row = videoRows.lowerBound + frame * rowsPerFrame + offset
                positions[row * 3] = Float(targetTemporal[frame])
                positions[row * 3 + 1] = Float(spatial.0)
                positions[row * 3 + 2] = Float(spatial.1)
            }
        }

        var tags = textTokenTags
        tags.append(contentsOf: repeatElement(
            MiniMaxH3Modality.video.rawValue,
            count: conditionVideoCount
        ))
        tags.append(contentsOf: repeatElement(
            MiniMaxH3Modality.audio.rawValue,
            count: conditionAudioCount
        ))
        tags.append(contentsOf: repeatElement(MiniMaxH3Modality.audio.rawValue, count: audioCount))
        tags.append(contentsOf: repeatElement(MiniMaxH3Modality.video.rawValue, count: videoCount))
        var segments: [MiniMaxH3ConditionSegment] = []
        if !conditionVideoRows.isEmpty {
            segments.append(.init(
                modality: .video,
                packedRows: conditionVideoRows,
                sourceRows: 0..<conditionVideoRows.count
            ))
        }
        if !conditionAudioRows.isEmpty {
            segments.append(.init(
                modality: .audio,
                packedRows: conditionAudioRows,
                sourceRows: 0..<conditionAudioRows.count
            ))
        }
        return MiniMaxH3PackedLayout(
            positions: MLXArray(positions, [videoRows.upperBound, 3]),
            tokenTags: tags,
            textRows: textRows,
            conditionRows: conditionRows,
            conditionSegments: segments,
            conditionVideoRowCount: conditionVideoCount,
            conditionAudioRowCount: conditionAudioCount,
            targetAudioRows: audioRows,
            targetVideoRows: videoRows,
            videoLatentFrames: videoLatentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioLatentFrames
        )
    }

    public static func buildRef2VA(
        textTokenTags: [Int32],
        references: [MiniMaxH3PreparedReferenceGeometry],
        videoLatentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        audioLatentFrames: Int,
        keyframeAnchors: [MiniMaxH3KeyframeAnchor] = [],
        audioConditionAnchors: [MiniMaxH3AudioConditionAnchor] = []
    ) throws -> MiniMaxH3PackedLayout {
        guard videoLatentFrames > 0, latentHeight > 0, latentWidth > 0, audioLatentFrames > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("all target latent dimensions must be positive")
        }
        guard latentHeight.isMultiple(of: 2), latentWidth.isMultiple(of: 2) else {
            throw MiniMaxH3LayoutError.invalidGeometry("target latent spatial dimensions must be divisible by 2")
        }
        for reference in references {
            switch reference.kind {
            case .image, .video:
                guard reference.videoLatentFrames > 0,
                      reference.latentHeight > 0,
                      reference.latentWidth > 0,
                      reference.latentHeight.isMultiple(of: 2),
                      reference.latentWidth.isMultiple(of: 2) else {
                    throw MiniMaxH3LayoutError.invalidGeometry("visual reference latent geometry is invalid")
                }
            case .audio:
                guard reference.audioLatentFrames > 0 else {
                    throw MiniMaxH3LayoutError.invalidGeometry("audio reference must contain latent frames")
                }
            }
        }

        let textRows = 0..<textTokenTags.count
        let targetGrid = framePositionGrid(latentHeight: latentHeight, latentWidth: latentWidth)
        let targetRowsPerFrame = targetGrid.frame.count
        let targetVideoCount = videoLatentFrames * targetRowsPerFrame
        let targetAudioCount = audioLatentFrames * 2
        let keyframeVideoCount = keyframeAnchors.reduce(0) {
            $0 + $1.latentFrameCount * targetRowsPerFrame
        }
        let keyframeAudioCount = audioConditionAnchors.reduce(0) {
            $0 + $1.latentFrameCount * 2
        }
        let referenceVideoCount = references.reduce(0) { $0 + $1.videoRowCount }
        let referenceAudioCount = references.reduce(0) { $0 + $1.audioRowCount }
        let conditionVideoCount = keyframeVideoCount + referenceVideoCount
        let conditionAudioCount = keyframeAudioCount + referenceAudioCount
        let keyframeVideoRows = textRows.upperBound..<(textRows.upperBound + keyframeVideoCount)
        let keyframeAudioRows = keyframeVideoRows.upperBound..<(
            keyframeVideoRows.upperBound + keyframeAudioCount
        )
        let conditionRows = textRows.upperBound..<(
            textRows.upperBound + conditionVideoCount + conditionAudioCount
        )
        let targetAudioRows = conditionRows.upperBound..<(conditionRows.upperBound + targetAudioCount)
        let targetVideoRows = targetAudioRows.upperBound..<(targetAudioRows.upperBound + targetVideoCount)
        var positions = Array(repeating: Float(0), count: targetVideoRows.upperBound * 3)
        for row in textRows { positions[row * 3] = Float(row) }
        var tags = textTokenTags + Array(
            repeating: MiniMaxH3Modality.text.rawValue,
            count: targetVideoRows.upperBound - textRows.count
        )
        func mark(_ rows: Range<Int>, as modality: MiniMaxH3Modality) {
            for row in rows { tags[row] = modality.rawValue }
        }

        var segments: [MiniMaxH3ConditionSegment] = []
        if !keyframeVideoRows.isEmpty {
            segments.append(.init(
                modality: .video,
                packedRows: keyframeVideoRows,
                sourceRows: 0..<keyframeVideoCount
            ))
            mark(keyframeVideoRows, as: .video)
        }
        if !keyframeAudioRows.isEmpty {
            segments.append(.init(
                modality: .audio,
                packedRows: keyframeAudioRows,
                sourceRows: 0..<keyframeAudioCount
            ))
            mark(keyframeAudioRows, as: .audio)
        }
        var packedCursor = keyframeAudioRows.upperBound
        var videoCursor = keyframeVideoCount
        var audioCursor = keyframeAudioCount
        var rotaryTime = Double(textRows.count)
        for reference in references {
            switch reference.kind {
            case .image:
                let packed = packedCursor..<(packedCursor + reference.videoRowCount)
                let source = videoCursor..<(videoCursor + reference.videoRowCount)
                segments.append(.init(modality: .video, packedRows: packed, sourceRows: source))
                fillVideoPositions(
                    &positions,
                    rows: packed,
                    latentFrames: 1,
                    latentHeight: reference.latentHeight,
                    latentWidth: reference.latentWidth,
                    origin: rotaryTime,
                    singleImage: true
                )
                mark(packed, as: .video)
                packedCursor = packed.upperBound
                videoCursor = source.upperBound
                rotaryTime += 1
            case .audio:
                let packed = packedCursor..<(packedCursor + reference.audioRowCount)
                let source = audioCursor..<(audioCursor + reference.audioRowCount)
                segments.append(.init(modality: .audio, packedRows: packed, sourceRows: source))
                fillAudioPositions(
                    &positions,
                    rows: packed,
                    frames: reference.audioLatentFrames,
                    origin: rotaryTime,
                    widthGrid: targetGrid.width
                )
                mark(packed, as: .audio)
                packedCursor = packed.upperBound
                audioCursor = source.upperBound
                rotaryTime += Double(reference.audioLatentFrames)
            case .video:
                if reference.audioRowCount > 0 {
                    let packed = packedCursor..<(packedCursor + reference.audioRowCount)
                    let source = audioCursor..<(audioCursor + reference.audioRowCount)
                    segments.append(.init(modality: .audio, packedRows: packed, sourceRows: source))
                    let grid = framePositionGrid(
                        latentHeight: reference.latentHeight,
                        latentWidth: reference.latentWidth
                    )
                    fillAudioPositions(
                        &positions,
                        rows: packed,
                        frames: reference.audioLatentFrames,
                        origin: rotaryTime,
                        widthGrid: grid.width
                    )
                    mark(packed, as: .audio)
                    packedCursor = packed.upperBound
                    audioCursor = source.upperBound
                }
                let packed = packedCursor..<(packedCursor + reference.videoRowCount)
                let source = videoCursor..<(videoCursor + reference.videoRowCount)
                segments.append(.init(modality: .video, packedRows: packed, sourceRows: source))
                fillVideoPositions(
                    &positions,
                    rows: packed,
                    latentFrames: reference.videoLatentFrames,
                    latentHeight: reference.latentHeight,
                    latentWidth: reference.latentWidth,
                    origin: rotaryTime,
                    singleImage: false
                )
                mark(packed, as: .video)
                packedCursor = packed.upperBound
                videoCursor = source.upperBound
                rotaryTime += max(
                    Double(reference.audioLatentFrames),
                    temporalSpan(reference.videoLatentFrames)
                )
            }
        }
        let historyLatentFrames = keyframeAnchors.reduce(0) { total, anchor in
            if case .history(let count) = anchor { return total + count }
            return total
        }
        let targetOrigin = rotaryTime + temporalSpan(historyLatentFrames)
        let targetTemporal = temporalGrid(videoLatentFrames, origin: targetOrigin)
        var keyframeVideoCursor = keyframeVideoRows.lowerBound
        var historyOrigin = rotaryTime
        for anchor in keyframeAnchors {
            let latentFrameCount = anchor.latentFrameCount
            let times: [Double] = switch anchor {
            case .history:
                temporalGrid(latentFrameCount, origin: historyOrigin)
            case .first:
                Array(targetTemporal.prefix(latentFrameCount))
            case .frame(let frameIndex):
                [targetOrigin + Double(frameIndex) * frameSpanScale]
            case .last:
                [targetOrigin + temporalSpan(videoLatentFrames) - frameSpanScale]
            }
            if case .history = anchor {
                historyOrigin += temporalSpan(latentFrameCount)
            }
            for latentFrame in 0..<latentFrameCount {
                for (offset, spatial) in targetGrid.frame.enumerated() {
                    let row = keyframeVideoCursor + latentFrame * targetRowsPerFrame + offset
                    positions[row * 3] = Float(times[latentFrame])
                    positions[row * 3 + 1] = Float(spatial.0)
                    positions[row * 3 + 2] = Float(spatial.1)
                }
            }
            keyframeVideoCursor += latentFrameCount * targetRowsPerFrame
        }
        var keyframeAudioCursor = keyframeAudioRows.lowerBound
        var audioHistoryOrigin = rotaryTime
        for anchor in audioConditionAnchors {
            let origin: Double = switch anchor {
            case .history: audioHistoryOrigin
            case .first: targetOrigin
            }
            let rows = keyframeAudioCursor..<(
                keyframeAudioCursor + anchor.latentFrameCount * 2
            )
            fillAudioPositions(
                &positions,
                rows: rows,
                frames: anchor.latentFrameCount,
                origin: origin,
                widthGrid: targetGrid.width
            )
            if case .history = anchor {
                audioHistoryOrigin += Double(anchor.latentFrameCount)
            }
            keyframeAudioCursor = rows.upperBound
        }
        fillAudioPositions(
            &positions,
            rows: targetAudioRows,
            frames: audioLatentFrames,
            origin: targetOrigin,
            widthGrid: targetGrid.width
        )
        fillVideoPositions(
            &positions,
            rows: targetVideoRows,
            latentFrames: videoLatentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            origin: targetOrigin,
            singleImage: false
        )
        mark(targetAudioRows, as: .audio)
        mark(targetVideoRows, as: .video)
        return MiniMaxH3PackedLayout(
            positions: MLXArray(positions, [targetVideoRows.upperBound, 3]),
            tokenTags: tags,
            textRows: textRows,
            conditionRows: conditionRows,
            conditionSegments: segments,
            conditionVideoRowCount: conditionVideoCount,
            conditionAudioRowCount: conditionAudioCount,
            targetAudioRows: targetAudioRows,
            targetVideoRows: targetVideoRows,
            videoLatentFrames: videoLatentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioLatentFrames
        )
    }

    private static func framePositionGrid(
        latentHeight: Int,
        latentWidth: Int
    ) -> (frame: [(Double, Double)], width: [Double]) {
        let squareRootArea = sqrt(Double(latentHeight * latentWidth))
        let height = spatialGrid(dimension: latentHeight, patch: 2, squareRootArea: squareRootArea)
        let width = spatialGrid(dimension: latentWidth, patch: 2, squareRootArea: squareRootArea)
        return (height.flatMap { y in width.map { x in (y, x) } }, width)
    }

    private static func fillAudioPositions(
        _ positions: inout [Float],
        rows: Range<Int>,
        frames: Int,
        origin: Double,
        widthGrid: [Double]
    ) {
        guard frames > 0 else { return }
        for channel in 0..<2 {
            for frame in 0..<frames {
                let row = rows.lowerBound + channel * frames + frame
                positions[row * 3] = Float(origin + Double(frame))
                positions[row * 3 + 2] = Float(channel == 0 ? widthGrid[0] : widthGrid[widthGrid.count - 1])
            }
        }
    }

    private static func fillVideoPositions(
        _ positions: inout [Float],
        rows: Range<Int>,
        latentFrames: Int,
        latentHeight: Int,
        latentWidth: Int,
        origin: Double,
        singleImage: Bool
    ) {
        let grid = framePositionGrid(latentHeight: latentHeight, latentWidth: latentWidth).frame
        let temporal = singleImage ? [origin] : temporalGrid(latentFrames, origin: origin)
        for frame in 0..<latentFrames {
            for (offset, spatial) in grid.enumerated() {
                let row = rows.lowerBound + frame * grid.count + offset
                positions[row * 3] = Float(temporal[frame])
                positions[row * 3 + 1] = Float(spatial.0)
                positions[row * 3 + 2] = Float(spatial.1)
            }
        }
    }

    private static func spatialGrid(dimension: Int, patch: Int, squareRootArea: Double) -> [Double] {
        let ratio = Double(dimension) / squareRootArea
        let count = dimension / patch
        let left = (1 - ratio) / 2
        return (0..<count).map { (left + Double($0) * ratio / Double(count)) * 32 }
    }

    private static func temporalGrid(_ count: Int, origin: Double) -> [Double] {
        var values: [Double] = []
        values.reserveCapacity(count)
        var current = origin
        for index in 0..<count {
            values.append(current)
            current += frameSpanScale * frameSpanPattern[index % frameSpanPattern.count]
        }
        return values
    }

    private static func temporalSpan(_ count: Int) -> Double {
        (0..<count).reduce(0) { partial, index in
            partial + frameSpanScale * frameSpanPattern[index % frameSpanPattern.count]
        }
    }
}
