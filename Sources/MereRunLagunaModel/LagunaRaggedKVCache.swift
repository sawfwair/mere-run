import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

/// A transient decode-time cache view that packs independently positioned
/// request rows into one MLX batch. The underlying row caches remain the
/// source of truth, so splitting after the forward is zero-copy at the cache
/// object level and each row keeps its own absolute position.
package final class LagunaRaggedKVCache: Gemma4AttentionCache {
    let rows: [Gemma4AttentionCache]
    var lastAttentionKeyLengths: [Int] = []

    package init?(rows: [Gemma4AttentionCache]) {
        guard !rows.isEmpty else { return nil }
        let cacheType = String(describing: type(of: rows[0]))
        guard rows.allSatisfy({ String(describing: type(of: $0)) == cacheType }) else {
            return nil
        }
        self.rows = rows
    }

    package var offset: Int {
        positionOffsets.min() ?? 0
    }

    package var positionOffsets: [Int] {
        rows.map(\.offset)
    }

    package func currentState() -> (MLXArray, MLXArray)? {
        paddedState(rows.compactMap { $0.currentState() })
    }

    package func decodeState() -> (MLXArray, MLXArray)? {
        paddedState(rows.compactMap { $0.decodeState() })
    }

    package func append(keys: MLXArray, values: MLXArray) {
        precondition(keys.dim(0) == rows.count && values.dim(0) == rows.count)
        for (index, row) in rows.enumerated() {
            row.append(
                keys: keys[index..<(index + 1), 0..., 0..., 0...],
                values: values[index..<(index + 1), 0..., 0..., 0...]
            )
        }
    }

    package func attentionState(
        appending keys: MLXArray,
        values: MLXArray
    ) -> (MLXArray, MLXArray)? {
        precondition(keys.dim(0) == rows.count && values.dim(0) == rows.count)
        var states: [(MLXArray, MLXArray)] = []
        states.reserveCapacity(rows.count)
        for (index, row) in rows.enumerated() {
            guard let state = row.attentionState(
                appending: keys[index..<(index + 1), 0..., 0..., 0...],
                values: values[index..<(index + 1), 0..., 0..., 0...]
            ) else {
                return nil
            }
            states.append(state)
        }
        lastAttentionKeyLengths = states.map { $0.0.dim(2) }
        return paddedState(states)
    }

    package func fork() -> Gemma4AttentionCache {
        LagunaRaggedKVCache(rows: rows.map { $0.fork() })!
    }

    package func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        let nestedRows = caches.compactMap { ($0 as? LagunaRaggedKVCache)?.rows }
        guard nestedRows.count == caches.count else { return nil }
        return LagunaRaggedKVCache(rows: nestedRows.flatMap { $0 })
    }

    package func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count == rows.count else { return nil }
        return rows
    }

    package func specializedAttention(
        queries: MLXArray,
        repeats: Int,
        scale: Float
    ) -> MLXArray? {
        nil
    }

    package func reencoded(
        quantization: Gemma4KVCacheQuantization
    ) -> Gemma4AttentionCache? {
        let converted = rows.compactMap { $0.reencoded(quantization: quantization) }
        guard converted.count == rows.count else { return nil }
        return LagunaRaggedKVCache(rows: converted)
    }

    package func evaluateStorage() {
        rows.forEach { $0.evaluateStorage() }
    }

    package func storageArraysForEvaluation() -> [MLXArray] {
        rows.flatMap { $0.storageArraysForEvaluation() }
    }

    func paddedState(
        _ states: [(MLXArray, MLXArray)]
    ) -> (MLXArray, MLXArray)? {
        guard states.count == rows.count, let first = states.first else {
            return nil
        }
        let maximumLength = states.map { $0.0.dim(2) }.max() ?? 0
        guard maximumLength > 0 else { return nil }

        func pad(_ array: MLXArray, to length: Int) -> MLXArray {
            let missing = length - array.dim(2)
            guard missing > 0 else { return array }
            return concatenated(
                [
                    array,
                    MLXArray.zeros(
                        [1, array.dim(1), missing, array.dim(3)],
                        dtype: array.dtype
                    ),
                ],
                axis: 2
            )
        }

        let keys = concatenated(states.map { pad($0.0, to: maximumLength) }, axis: 0)
        let values = concatenated(states.map { pad($0.1, to: maximumLength) }, axis: 0)
        precondition(keys.dim(1) == first.0.dim(1))
        return (keys, values)
    }
}
