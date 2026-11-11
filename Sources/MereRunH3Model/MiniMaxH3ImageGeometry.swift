import Foundation
import MLX

public struct MiniMaxH3ImageLayout {
    public let positions: MLXArray
    public let tokenTags: MLXArray
    public let referenceRowCount: Int
    public let textRowCount: Int
    public let imageRowCount: Int
    public var targetRows: Range<Int> { (textRowCount + referenceRowCount)..<(textRowCount + imageRowCount) }

    public init(textTags: [Int32], referenceGrids: [(height: Int, width: Int)], height: Int, width: Int) {
        precondition(!textTags.isEmpty && textTags.allSatisfy { $0 == 0 || $0 == 1 })
        precondition(height > 0 && width > 0 && height.isMultiple(of: 2) && width.isMultiple(of: 2))
        self.textRowCount = textTags.count
        self.referenceRowCount = referenceGrids.reduce(0) { $0 + ($1.height / 2) * ($1.width / 2) }
        self.imageRowCount = referenceRowCount + (height / 2) * (width / 2)
        var coordinates: [Float] = []
        for index in textTags.indices { coordinates += [Float(index), 0, 0] }
        for (index, grid) in (referenceGrids + [(height, width)]).enumerated() {
            precondition(grid.height > 0 && grid.width > 0 && grid.height.isMultiple(of: 2) && grid.width.isMultiple(of: 2))
            let spatial = MiniMaxH3Geometry.framePositionGrid(latentHeight: grid.height, latentWidth: grid.width).frame
            for (y, x) in spatial { coordinates += [Float(textTags.count + index), Float(y), Float(x)] }
        }
        self.positions = MLXArray(coordinates).reshaped(textRowCount + imageRowCount, 3)
        self.tokenTags = MLXArray(textTags + Array(repeating: 0, count: imageRowCount))
    }

    /// Text (including vision tokens) and target use the current timestep; VAE references stay at 0.999.
    package func adaLNIndices() -> MLXArray {
        let timeIndices = Array(repeating: Int32(0), count: textRowCount)
            + Array(repeating: Int32(1), count: referenceRowCount)
            + Array(repeating: Int32(0), count: imageRowCount - referenceRowCount)
        return MLXArray(timeIndices) * 3 + tokenTags
    }
}

public enum MiniMaxH3ImageGeometry {
    public static func canvas(width: Int, height: Int) -> (width: Int, height: Int) {
        precondition(width > 0 && height > 0)
        let scale = max(1, sqrt(4_194_304 / (Double(width) * Double(height))))
        func aligned(_ size: Int) -> Int { max(32, Int((Double(size) * scale / 32).rounded(.toNearestOrEven)) * 32) }
        return (aligned(width), aligned(height))
    }

    public static func referenceSize(width: Int, height: Int, vision: Bool) -> (width: Int, height: Int) {
        let edge = Double(min(width, height))
        if (1_024...(vision ? 1_024 : 2_048)).contains(min(width, height)) { return (width, height) }
        let bounded = min(max(edge, 1_024), vision ? 1_024 : 2_048)
        func aligned(_ size: Int) -> Int { max(32, Int((Double(size) * bounded / edge / 32).rounded(.toNearestOrEven)) * 32) }
        return (aligned(width), aligned(height))
    }

    /// Qwen3-VL processor alignment and its released 16 MP vision budget, after reference resizing.
    public static func visionProcessorSize(width: Int, height: Int) -> (width: Int, height: Int) {
        var w = max(32, Int((Double(width) / 32).rounded(.toNearestOrEven)) * 32)
        var h = max(32, Int((Double(height) / 32).rounded(.toNearestOrEven)) * 32)
        if Double(w) * Double(h) > 16_777_216 {
            let beta = sqrt(Double(width) * Double(height) / 16_777_216)
            w = max(32, Int(floor(Double(width) / beta / 32)) * 32)
            h = max(32, Int(floor(Double(height) / beta / 32)) * 32)
        }
        return (w, h)
    }
}
