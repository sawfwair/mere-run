import Foundation
import MereRunCore
@preconcurrency import MLX

extension TESSERAResources {
    public static func loadModel(from checkpoint: TESSERACheckpoint) throws -> TESSERAModel {
        let arrays = try MLX.loadArrays(url: checkpoint.weightsURL)
        guard arrays.count == checkpoint.source.tensorCount else {
            throw TESSERAResourceError.tensorInventory(
                expected: checkpoint.source.tensorCount,
                actual: arrays.count
            )
        }
        let scalarCount = arrays.values.reduce(0) { partial, value in
            partial + value.shape.reduce(1, *)
        }
        guard scalarCount == checkpoint.source.scalarCount else {
            throw TESSERAResourceError.scalarInventory(
                expected: checkpoint.source.scalarCount,
                actual: scalarCount
            )
        }
        return try TESSERAModel(weights: arrays, architecture: checkpoint.source.architecture)
    }
}
