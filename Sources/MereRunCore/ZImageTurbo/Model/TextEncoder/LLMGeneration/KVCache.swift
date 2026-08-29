// Adapted from mlx-swift-lm MLXLMCommon/KVCache.swift

import Foundation
import MLX
import MLXFast
import MLXNN

public protocol KVCache: AnyObject {
    var offset: Int { get }
    var rowOffsets: [Int]? { get }
    var supportsVariablePositionBatching: Bool { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode
    func fork() -> KVCache
    func batched(with caches: [KVCache]) -> KVCache?
    func unbatchedRows(count: Int) -> [KVCache]?
}

public extension KVCache {
    var rowOffsets: [Int]? {
        nil
    }

    var supportsVariablePositionBatching: Bool {
        false
    }

    func batched(with caches: [KVCache]) -> KVCache? {
        nil
    }

    func unbatchedRows(count: Int) -> [KVCache]? {
        nil
    }
}

public class KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0
    private var step: Int = 256

    public init(step: Int = 256) {
        self.step = step
    }

    public var supportsVariablePositionBatching: Bool {
        true
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset: Bool
        if let currentKeys = self.keys {
            reset = (previous + keys.dim(2)) > currentKeys.dim(2)
        } else {
            reset = true
        }

        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous..<self.offset, 0...] = keys
        self.values?[.ellipsis, previous..<self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    private func populatedState() -> (MLXArray, MLXArray)? {
        guard let keys, let values else { return nil }
        return (
            keys[.ellipsis, ..<offset, 0...],
            values[.ellipsis, ..<offset, 0...]
        )
    }

    public func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        return .causal
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }

    /// Moves the logical write head back without reallocating storage.
    /// The next update overwrites the discarded suffix.
    public func rollback(toOffset newOffset: Int) {
        precondition(newOffset >= 0 && newOffset <= offset)
        offset = newOffset
    }

    public func fork() -> KVCache {
        let copy = KVCacheSimple(step: step)
        // `update` writes new tokens with subscript assignment, which rebinds
        // the SAME MLXArray wrapper in place (`_updateInternal`). Sharing the
        // wrapper objects with a fork means every later write on the parent
        // mutates the fork too — prefix-KV snapshots stored mid-request were
        // silently corrupted by the request's remaining prefill and decode.
        // Fresh wrappers over the current (immutable) arrays isolate the fork;
        // the parent's rebinds can no longer reach it.
        // No-op dtype casts return self in MLX; a same-shape reshape creates
        // a fresh wrapper without copying the underlying tensor storage.
        copy.keys = keys.map { $0.reshaped($0.shape) }
        copy.values = values.map { $0.reshaped($0.shape) }
        copy.offset = offset
        return copy
    }

    public func batched(with caches: [KVCache]) -> KVCache? {
        guard let typed = caches as? [KVCacheSimple],
              !typed.isEmpty,
              typed.allSatisfy({ $0.step == step }) else {
            return nil
        }

        let states = typed.compactMap { $0.populatedState() }
        guard states.count == typed.count else {
            return nil
        }

        if !typed.allSatisfy({ $0.offset == offset }) {
            return KVRaggedBatchCache(
                states: zip(typed, states).map { cache, state in
                    KVRaggedBatchCache.RowState(
                        keys: state.0,
                        values: state.1,
                        offset: cache.offset
                    )
                },
                step: step
            )
        }

        let copy = KVCacheSimple(step: step)
        copy.keys = concatenated(states.map(\.0), axis: 0)
        copy.values = concatenated(states.map(\.1), axis: 0)
        copy.offset = offset
        return copy
    }

    public func unbatchedRows(count: Int) -> [KVCache]? {
        guard count > 0,
              let state = populatedState(),
              state.0.dim(0) == count,
              state.1.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = KVCacheSimple(step: step)
            copy.keys = state.0[index..<(index + 1), 0..., 0..., 0...]
            copy.values = state.1[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}

/// A fixed-capacity cache for generation loops whose maximum sequence length
/// is known before prefill. Unlike `KVCacheSimple`, it never concatenates a new
/// backing allocation while decoding.
public final class KVCacheStatic: KVCache {
    public let capacity: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset = 0

    public init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    public var supportsVariablePositionBatching: Bool {
        true
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let nextOffset = offset + newKeys.dim(2)
        precondition(
            nextOffset <= capacity,
            "KV cache capacity \(capacity) is smaller than required length \(nextOffset)"
        )
        if keys == nil {
            keys = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)],
                dtype: newKeys.dtype
            )
            values = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)],
                dtype: newValues.dtype
            )
        }

        keys?[.ellipsis, offset..<nextOffset, 0...] = newKeys
        values?[.ellipsis, offset..<nextOffset, 0...] = newValues
        offset = nextOffset
        return (
            keys![.ellipsis, ..<offset, 0...],
            values![.ellipsis, ..<offset, 0...]
        )
    }

    public func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n == 1 ? .none : .causal
    }

    public func fork() -> KVCache {
        let copy = KVCacheStatic(capacity: capacity)
        copy.keys = keys.map { $0.reshaped($0.shape) }
        copy.values = values.map { $0.reshaped($0.shape) }
        copy.offset = offset
        return copy
    }

    public func unbatchedRows(count: Int) -> [KVCache]? {
        guard count > 0,
              let keys,
              let values,
              keys.dim(0) == count,
              values.dim(0) == count else {
            return nil
        }
        return (0..<count).map { index in
            let copy = KVCacheStatic(capacity: capacity)
            copy.keys = keys[index..<(index + 1), 0..., 0..., 0...]
            copy.values = values[index..<(index + 1), 0..., 0..., 0...]
            copy.offset = offset
            return copy
        }
    }
}

public final class KVRaggedBatchCache: KVCache {
    struct RowState {
        let keys: MLXArray
        let values: MLXArray
        let offset: Int
    }

    private let step: Int
    private var keys: MLXArray
    private var values: MLXArray
    private var offsets: [Int]

    public var offset: Int {
        offsets.min() ?? 0
    }

    public var rowOffsets: [Int]? {
        offsets
    }

    public var supportsVariablePositionBatching: Bool {
        true
    }

    init?(states: [RowState], step: Int) {
        guard !states.isEmpty else { return nil }
        self.step = step
        self.offsets = states.map(\.offset)

        let maxOffset = offsets.max() ?? 0
        guard let padded = Self.padded(states: states, maxOffset: maxOffset) else {
            return nil
        }
        self.keys = padded.keys
        self.values = padded.values
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let tokenCount = newKeys.dim(2)
        let newOffsets = offsets.map { $0 + tokenCount }
        let maxOffset = newOffsets.max() ?? tokenCount
        let states = offsets.indices.map { index in
            let existingKeys = keys[index..<(index + 1), 0..., ..<offsets[index], 0...]
            let existingValues = values[index..<(index + 1), 0..., ..<offsets[index], 0...]
            return RowState(
                keys: concatenated([existingKeys, newKeys[index..<(index + 1), 0..., 0..., 0...]], axis: 2),
                values: concatenated([existingValues, newValues[index..<(index + 1), 0..., 0..., 0...]], axis: 2),
                offset: newOffsets[index]
            )
        }

        if let padded = Self.padded(states: states, maxOffset: maxOffset) {
            self.keys = padded.keys
            self.values = padded.values
            self.offsets = newOffsets
        }
        return (keys, values)
    }

    public func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 0 else { return .none }
        let maxLength = (offsets.max() ?? 0) + n
        guard maxLength > 0 else { return .none }

        var mask: [Float] = []
        mask.reserveCapacity(offsets.count * n * maxLength)
        for rowOffset in offsets {
            for queryOffset in rowOffset..<(rowOffset + n) {
                for keyPosition in 0..<maxLength {
                    mask.append(keyPosition <= queryOffset ? 0 : -1e9)
                }
            }
        }
        let array = MLXArray(mask).reshaped(offsets.count, 1, n, maxLength)
        return .array(array)
    }

    public func fork() -> KVCache {
        KVRaggedBatchCache(
            states: offsets.indices.map { index in
                RowState(
                    keys: keys[index..<(index + 1), 0..., ..<offsets[index], 0...],
                    values: values[index..<(index + 1), 0..., ..<offsets[index], 0...],
                    offset: offsets[index]
                )
            },
            step: step
        )!
    }

    public func unbatchedRows(count: Int) -> [KVCache]? {
        guard count == offsets.count else { return nil }
        return offsets.indices.map { index in
            let copy = KVCacheSimple(step: step)
            let validKeys = keys[index..<(index + 1), 0..., ..<offsets[index], 0...]
            let validValues = values[index..<(index + 1), 0..., ..<offsets[index], 0...]
            _ = copy.update(keys: validKeys, values: validValues)
            return copy
        }
    }

    private static func padded(states: [RowState], maxOffset: Int) -> (keys: MLXArray, values: MLXArray)? {
        guard let first = states.first else { return nil }
        let keyPaddingShape = [
            1,
            first.keys.dim(1),
            0,
            first.keys.dim(3),
        ]
        let valuePaddingShape = [
            1,
            first.values.dim(1),
            0,
            first.values.dim(3),
        ]

        var paddedKeys: [MLXArray] = []
        var paddedValues: [MLXArray] = []
        paddedKeys.reserveCapacity(states.count)
        paddedValues.reserveCapacity(states.count)
        for state in states {
            let keyPadCount = max(0, maxOffset - state.keys.dim(2))
            let valuePadCount = max(0, maxOffset - state.values.dim(2))
            let rowKeys = keyPadCount > 0
                ? concatenated(
                    [
                        state.keys,
                        MLXArray.zeros(
                            [keyPaddingShape[0], keyPaddingShape[1], keyPadCount, keyPaddingShape[3]],
                            dtype: state.keys.dtype
                        ),
                    ],
                    axis: 2
                )
                : state.keys
            let rowValues = valuePadCount > 0
                ? concatenated(
                    [
                        state.values,
                        MLXArray.zeros(
                            [valuePaddingShape[0], valuePaddingShape[1], valuePadCount, valuePaddingShape[3]],
                            dtype: state.values.dtype
                        ),
                    ],
                    axis: 2
                )
                : state.values
            paddedKeys.append(rowKeys)
            paddedValues.append(rowValues)
        }
        return (concatenated(paddedKeys, axis: 0), concatenated(paddedValues, axis: 0))
    }
}

public func createCausalMask(n: Int, offset: Int) -> MLXArray {
    let indices = MLXArray(0..<Int32(n + offset))
    let rows = MLXArray(Int32(offset)..<Int32(n + offset)).reshaped(-1, 1)
    return rows .>= indices
}

public func createAttentionMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)
    if let cache = cache {
        return cache.makeMask(n: n)
    }
    if n == 1 {
        return .none
    }
    return .causal
}
