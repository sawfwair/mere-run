import Foundation
import MLX
import MLXFast
import MLXNN

package final class Gemma4FullKVCache: Gemma4AttentionCache {
    package init() {}

    private static let allocationStep = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    package private(set) var offset: Int = 0

    package func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        guard offset < keys.dim(2) else {
            return (keys, values)
        }
        return (keys[0..., 0..., 0..<offset, 0...], values[0..., 0..., 0..<offset, 0...])
    }

    package func append(keys: MLXArray, values: MLXArray) {
        let previousOffset = offset
        let newOffset = previousOffset + keys.dim(2)

        if self.keys == nil || newOffset > (self.keys?.dim(2) ?? 0) {
            let batch = keys.dim(0)
            let heads = keys.dim(1)
            let keyDim = keys.dim(3)
            let valueDim = values.dim(3)
            // Growth retains only the valid prefix below, discarding spare
            // capacity. Allocate enough new rows for the entire incoming chunk.
            let steps = max(1, (keys.dim(2) + Self.allocationStep - 1) / Self.allocationStep)
            let growth = steps * Self.allocationStep
            let newKeys = MLXArray.zeros([batch, heads, growth, keyDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, heads, growth, valueDim], dtype: values.dtype)
            if let existingKeys = self.keys, let existingValues = self.values {
                let trimmedKeys = previousOffset < existingKeys.dim(2)
                    ? existingKeys[0..., 0..., 0..<previousOffset, 0...]
                    : existingKeys
                let trimmedValues = previousOffset < existingValues.dim(2)
                    ? existingValues[0..., 0..., 0..<previousOffset, 0...]
                    : existingValues
                self.keys = concatenated([trimmedKeys, newKeys], axis: 2)
                self.values = concatenated([trimmedValues, newValues], axis: 2)
            } else {
                self.keys = newKeys
                self.values = newValues
            }
        }

        guard let storedKeys = self.keys, let storedValues = self.values else {
            preconditionFailure("Gemma4FullKVCache should allocate before writing.")
        }
        storedKeys[0..., 0..., previousOffset..<newOffset, 0...] = keys
        storedValues[0..., 0..., previousOffset..<newOffset, 0...] = values
        self.offset = newOffset
    }

    package func fork() -> Gemma4AttentionCache {
        let copy = Gemma4FullKVCache()
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    package static func reencoded(keys: MLXArray, values: MLXArray, offset: Int) -> Gemma4FullKVCache {
        let cache = Gemma4FullKVCache()
        cache.keys = keys
        cache.values = values
        cache.offset = offset
        return cache
    }

    package func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: nil,
            quantization: quantization
        )
    }

    package func evaluateStorage() {
        if let keys, let values {
            MLX.eval(keys, values)
        }
    }

    package func storageArraysForEvaluation() -> [MLXArray] {
        guard let keys, let values else { return [] }
        return [keys, values]
    }

    package func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        guard let typed = caches as? [Gemma4FullKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4FullKVCache()
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        return copy
    }

    package func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4FullKVCache()
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}
