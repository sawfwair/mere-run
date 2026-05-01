import Foundation
import MLX

enum ACEStepSequencePacker {
    /// Pack two sequences by concatenating and stably partitioning tokens so that mask values of 1 appear before 0.
    ///
    /// - Returns: Packed hidden states `[B, L1+L2, D]` and a new attention mask `[B, L1+L2]`.
    static func pack(
        hidden1: MLXArray,
        hidden2: MLXArray,
        mask1: MLXArray,
        mask2: MLXArray
    ) -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let hiddenCat = MLX.concatenated([hidden1, hidden2], axis: 1)
        let maskCat = MLX.concatenated([mask1, mask2], axis: 1).asType(.int32)

        let B = maskCat.dim(0)
        let L = maskCat.dim(1)

        MLX.eval(maskCat)
        let maskValues = maskCat.asArray(Int32.self)

        var sortIndices: [Int32] = []
        sortIndices.reserveCapacity(B * L)

        var newMaskValues: [Int32] = []
        newMaskValues.reserveCapacity(B * L)

        for b in 0..<B {
            let rowStart = b * L
            let rowEnd = rowStart + L
            let row = maskValues[rowStart..<rowEnd]

            var keep: [Int32] = []
            var drop: [Int32] = []
            keep.reserveCapacity(L)
            drop.reserveCapacity(L)

            var keepCount = 0
            for i in 0..<L {
                if row[row.index(row.startIndex, offsetBy: i)] != 0 {
                    keep.append(Int32(i))
                    keepCount += 1
                } else {
                    drop.append(Int32(i))
                }
            }

            sortIndices.append(contentsOf: keep)
            sortIndices.append(contentsOf: drop)

            newMaskValues.append(contentsOf: Array(repeating: 1, count: keepCount))
            newMaskValues.append(contentsOf: Array(repeating: 0, count: L - keepCount))
        }

        let idx = MLXArray(sortIndices.map(Float32.init), [B, L]).asType(.int32)
        let idxExpanded = idx.reshaped(B, L, 1)
        let idxBroadcast = broadcast(idxExpanded, to: [B, L, hiddenCat.dim(2)])
        let packedHidden = takeAlong(hiddenCat, idxBroadcast, axis: 1)

        let newMask = MLXArray(newMaskValues.map(Float32.init), [B, L]).asType(.int32)
        return (packedHidden, newMask)
    }
}

