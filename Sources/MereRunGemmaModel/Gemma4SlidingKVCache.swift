import Foundation
import MLX
import MLXFast
import MLXNN

package final class Gemma4SlidingKVCache: Gemma4AttentionCache {
    private static let allocationStep = 256

    private let maxSize: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    package private(set) var offset: Int = 0
    private var writeIndex: Int = 0

    package init(maxSize: Int, initialOffset: Int = 0) {
        self.maxSize = max(1, maxSize)
        self.offset = max(0, initialOffset)
    }

    package var configuredMaxSize: Int {
        maxSize
    }

    package func currentState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (temporalOrder(keys), temporalOrder(values))
    }

    package func decodeState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        // Storage order is fine for q-len 1: every slot is a valid in-window
        // token once the ring is full, and attention over an unmasked key set
        // is permutation-invariant. Before the ring fills, the valid prefix is
        // already in temporal order.
        if offset < keys.dim(2) {
            return (
                keys[0..., 0..., 0..<offset, 0...],
                values[0..., 0..., 0..<offset, 0...]
            )
        }
        return (keys, values)
    }

    package func append(keys: MLXArray, values: MLXArray) {
        if keys.dim(2) == 1 {
            updateInPlace(keys: keys, values: values)
        } else {
            updateByConcatenating(keys: keys, values: values)
        }
    }

    package func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        guard keys.dim(2) > 1 else {
            append(keys: keys, values: values)
            return decodeState()
        }

        let previous = currentState()
        append(keys: keys, values: values)
        guard let previous else {
            return currentState()
        }
        return (
            concatenated([previous.0, keys], axis: 2),
            concatenated([previous.1, values], axis: 2)
        )
    }

    private func updateByConcatenating(keys: MLXArray, values: MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            let orderedKeys = temporalOrder(existingKeys)
            let orderedValues = temporalOrder(existingValues)
            writeIndex = orderedKeys.dim(2)
            let trimSize = writeIndex - maxSize + keys.dim(2)
            self.keys = trim(orderedKeys, trimSize: trimSize, appending: keys)
            self.values = trim(orderedValues, trimSize: trimSize, appending: values)
        } else {
            self.keys = keys
            self.values = values
        }
        offset += keys.dim(2)
        writeIndex = self.keys?.dim(2) ?? 0
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) {
        let previousOffset = offset
        if self.keys == nil || (previousOffset >= (self.keys?.dim(2) ?? 0) && (self.keys?.dim(2) ?? 0) < maxSize) {
            let batch = keys.dim(0)
            let heads = keys.dim(1)
            let keyDim = keys.dim(3)
            let valueDim = values.dim(3)
            let existingSize = self.keys?.dim(2) ?? 0
            let growth = min(Self.allocationStep, maxSize - existingSize)
            if growth > 0 {
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
                writeIndex = min(previousOffset, self.keys?.dim(2) ?? previousOffset)
            }
        }

        if let existingKeys = self.keys, existingKeys.dim(2) > maxSize {
            self.keys = trim(existingKeys, trimSize: existingKeys.dim(2) - maxSize)
            writeIndex = maxSize
        }
        if let existingValues = self.values, existingValues.dim(2) > maxSize {
            self.values = trim(existingValues, trimSize: existingValues.dim(2) - maxSize)
        }
        if writeIndex >= maxSize {
            writeIndex = 0
        }

        let newWriteIndex = writeIndex + keys.dim(2)
        guard let storedKeys = self.keys, let storedValues = self.values else {
            preconditionFailure("Gemma4SlidingKVCache should allocate before writing.")
        }
        storedKeys[0..., 0..., writeIndex..<newWriteIndex, 0...] = keys
        storedValues[0..., 0..., writeIndex..<newWriteIndex, 0...] = values
        offset += keys.dim(2)
        writeIndex = newWriteIndex
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        let length = array.dim(2)
        if offset < length {
            return array[0..., 0..., 0..<offset, 0...]
        }
        guard writeIndex < length, writeIndex < offset else {
            return array
        }
        return concatenated([
            array[0..., 0..., writeIndex..., 0...],
            array[0..., 0..., 0..<writeIndex, 0...],
        ], axis: 2)
    }

    private func trim(_ array: MLXArray, trimSize: Int, appending append: MLXArray? = nil) -> MLXArray {
        var parts: [MLXArray]
        if trimSize > 0 {
            parts = [array[0..., 0..., trimSize..., 0...]]
        } else {
            parts = [array]
        }
        if let append {
            parts.append(append)
        }
        return concatenated(parts, axis: 2)
    }

    package func fork() -> Gemma4AttentionCache {
        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        copy.writeIndex = writeIndex
        return copy
    }

    package static func reencoded(
        keys: MLXArray,
        values: MLXArray,
        offset: Int,
        maxSize: Int
    ) -> Gemma4SlidingKVCache {
        let cache = Gemma4SlidingKVCache(maxSize: maxSize)
        let totalLength = keys.dim(2)
        if totalLength > cache.maxSize {
            let start = totalLength - cache.maxSize
            cache.keys = keys[0..., 0..., start..., 0...]
            cache.values = values[0..., 0..., start..., 0...]
        } else {
            cache.keys = keys
            cache.values = values
        }
        cache.offset = offset
        cache.writeIndex = cache.keys?.dim(2) ?? 0
        return cache
    }

    package func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        guard let state = currentState() else { return nil }
        return makeGemma4AttentionCache(
            keys: state.0,
            values: state.1,
            offset: offset,
            maxSize: maxSize,
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
        guard let typed = caches as? [Gemma4SlidingKVCache],
              !typed.isEmpty,
              typed.allSatisfy({ $0.offset == offset && $0.maxSize == maxSize }) else {
            return nil
        }

        let states = typed.compactMap { $0.currentState() }
        guard states.count == typed.count else {
            return nil
        }

        let copy = Gemma4SlidingKVCache(maxSize: maxSize)
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        copy.writeIndex = copy.keys?.dim(2) ?? 0
        return copy
    }

    package func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        guard count > 0, let keys, let values, keys.dim(0) == count, values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Gemma4SlidingKVCache(maxSize: maxSize)
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            copy.writeIndex = writeIndex
            return copy
        }
    }
}
