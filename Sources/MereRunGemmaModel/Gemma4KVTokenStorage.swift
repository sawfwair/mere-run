import Foundation
import MLX
import MLXFast

enum Gemma4KVTokenStorage {
    static let allocationStep = 256

    // Tensor states returned by `appending` share these capacity-padded buffers with their
    // ancestors: new rows are written in place at indices >= the writer's previous tokenCount,
    // so a snapshot must only ever read rows below its own tokenCount. Growth reallocates for
    // the writer, leaving older snapshots on the previous buffer.
    static func appended(_ array: MLXArray, rows: MLXArray, validCount: Int) -> MLXArray {
        let newCount = validCount + rows.dim(2)
        var target = array
        if newCount > array.dim(2) {
            let steps = max(1, (newCount - validCount + allocationStep - 1) / allocationStep)
            let valid = validCount < array.dim(2) ? array[0..., 0..., 0..<validCount, 0...] : array
            let padding = MLXArray.zeros(
                [array.dim(0), array.dim(1), steps * allocationStep, array.dim(3)],
                dtype: array.dtype
            )
            target = concatenated([valid, padding], axis: 2)
        }
        target[0..., 0..., validCount..<newCount, 0...] = rows
        return target
    }
}
