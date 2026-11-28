import Foundation
import MereRunCore
@preconcurrency import MLX

extension TerraMindFireResources {
    public static func loadModel(from checkpoint: TerraMindFireCheckpoint) throws -> TerraMindFloodModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == tensorCount else {
            throw TerraMindFireResourceError.tensorInventory(expected: tensorCount, actual: arrays.count)
        }
        let actualScalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard actualScalarCount == scalarCount else {
            throw TerraMindFireResourceError.scalarInventory(
                expected: scalarCount,
                actual: actualScalarCount
            )
        }
        return try TerraMindFloodModel(weights: arrays)
    }
}
