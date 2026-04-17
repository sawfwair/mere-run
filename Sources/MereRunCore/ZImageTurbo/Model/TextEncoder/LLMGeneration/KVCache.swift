// Adapted from mlx-swift-lm MLXLMCommon/KVCache.swift

import Foundation
import MLX
import MLXFast
import MLXNN

public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode
}

public class KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0
    private var step: Int = 256

    public init(step: Int = 256) {
        self.step = step
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
