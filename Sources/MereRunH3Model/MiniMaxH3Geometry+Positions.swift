import Foundation
import MLX

extension MiniMaxH3Geometry {
    static func framePositionGrid(
        latentHeight: Int,
        latentWidth: Int
    ) -> (frame: [(Double, Double)], width: [Double]) {
        let squareRootArea = sqrt(Double(latentHeight * latentWidth))
        let height = spatialGrid(dimension: latentHeight, patch: 2, squareRootArea: squareRootArea)
        let width = spatialGrid(dimension: latentWidth, patch: 2, squareRootArea: squareRootArea)
        return (height.flatMap { y in width.map { x in (y, x) } }, width)
    }

    static func fillAudioPositions(
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

    static func fillVideoPositions(
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

    static func spatialGrid(dimension: Int, patch: Int, squareRootArea: Double) -> [Double] {
        let ratio = Double(dimension) / squareRootArea
        let count = dimension / patch
        let left = (1 - ratio) / 2
        return (0..<count).map { (left + Double($0) * ratio / Double(count)) * 32 }
    }

    static func temporalGrid(_ count: Int, origin: Double) -> [Double] {
        var values: [Double] = []
        values.reserveCapacity(count)
        var current = origin
        for index in 0..<count {
            values.append(current)
            current += frameSpanScale * frameSpanPattern[index % frameSpanPattern.count]
        }
        return values
    }

    static func temporalSpan(_ count: Int) -> Double {
        (0..<count).reduce(0) { partial, index in
            partial + frameSpanScale * frameSpanPattern[index % frameSpanPattern.count]
        }
    }
}
