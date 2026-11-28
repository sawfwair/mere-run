import Foundation
import MereRunCore
@preconcurrency import MLX

extension OlmoEarthResources {
    public static func loadModel(from checkpoint: OlmoEarthCheckpoint) throws -> OlmoEarthModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.source.tensorCount else {
            throw OlmoEarthResourceError.tensorInventory(
                expected: checkpoint.source.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.source.scalarCount else {
            throw OlmoEarthResourceError.scalarInventory(
                expected: checkpoint.source.scalarCount,
                actual: scalarCount
            )
        }
        return try OlmoEarthModel(weights: arrays, architecture: checkpoint.source.architecture)
    }
}
