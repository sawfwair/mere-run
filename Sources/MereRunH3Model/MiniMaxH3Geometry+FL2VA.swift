import Foundation
import MLX

extension MiniMaxH3Geometry {
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

}
