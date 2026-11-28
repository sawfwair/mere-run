import Foundation
import MereRunCore
@preconcurrency import MLX

extension TerraMindFloodResources {
    public static func loadModel(from checkpoint: TerraMindFloodCheckpoint) throws -> TerraMindFloodModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.configuration.tensorCount else {
            throw TerraMindFloodResourceError.tensorInventory(
                expected: checkpoint.configuration.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.configuration.scalarCount else {
            throw TerraMindFloodResourceError.scalarInventory(
                expected: checkpoint.configuration.scalarCount,
                actual: scalarCount
            )
        }
        return try TerraMindFloodModel(weights: arrays)
    }
}
