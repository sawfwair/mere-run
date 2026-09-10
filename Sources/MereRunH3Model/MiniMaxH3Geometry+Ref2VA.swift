import Foundation
import MLX

extension MiniMaxH3Geometry {
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

}
