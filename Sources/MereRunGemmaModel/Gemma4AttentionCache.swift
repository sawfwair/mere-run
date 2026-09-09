import Foundation
import MLX
import MLXFast
import MLXNN

package protocol Gemma4AttentionCache: AnyObject {
    var offset: Int { get }
    func currentState() -> (MLXArray, MLXArray)?
    /// State for single-token decode, where softmax attention is invariant to
    /// key/value ordering (positions are already baked in via RoPE at append
    /// time and the decode mask is `.none`). Ring-buffer caches may return
    /// storage order here to skip the temporal-order copy. Multi-token queries
    /// must keep using `currentState()` — their causal masks assume temporal
    /// order.
    func decodeState() -> (MLXArray, MLXArray)?
    func append(keys: MLXArray, values: MLXArray)
    /// Appends the new keys and values and returns the state that a query of
    /// the same length must attend over. Sliding caches override this for
    /// multi-token chunked prefill so early queries in the chunk can still see
    /// the preceding window even though the resident cache is trimmed for the
    /// next step.
    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)?
    func fork() -> Gemma4AttentionCache
    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache?
    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]?
    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray?
    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache?
    func storageArraysForEvaluation() -> [MLXArray]
    func evaluateStorage()
}

package extension Gemma4AttentionCache {
    func attentionState(appending keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)? {
        append(keys: keys, values: values)
        return keys.dim(2) == 1 ? decodeState() : currentState()
    }

    func decodeState() -> (MLXArray, MLXArray)? {
        currentState()
    }

    func batched(with caches: [Gemma4AttentionCache]) -> Gemma4AttentionCache? {
        nil
    }

    func unbatchedRows(count: Int) -> [Gemma4AttentionCache]? {
        nil
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        nil
    }

    func reencoded(quantization: Gemma4KVCacheQuantization) -> Gemma4AttentionCache? {
        nil
    }

    func storageArraysForEvaluation() -> [MLXArray] {
        guard let state = currentState() else { return [] }
        return [state.0, state.1]
    }

    func evaluateStorage() {}
}

package func evaluateGemma4CacheStorage(_ caches: [Gemma4AttentionCache]) {
    let arrays = caches.flatMap { $0.storageArraysForEvaluation() }
    guard !arrays.isEmpty else { return }
    MLX.eval(arrays)
}

package func makeGemma4AttentionCache(
    keys: MLXArray,
    values: MLXArray,
    offset: Int,
    maxSize: Int?,
    quantization: Gemma4KVCacheQuantization? = nil
) -> Gemma4AttentionCache {
    if let quantization, quantization.isEnabled {
        if quantization.scheme == .polar {
            return Gemma4PolarKVCache.reencoded(
                keys: keys,
                values: values,
                configuration: quantization,
                maxSize: maxSize,
                offset: offset
            )
        }
        return Gemma4QuantizedKVCache.reencoded(
            keys: keys,
            values: values,
            configuration: quantization,
            maxSize: maxSize,
            offset: offset
        )
    }

    if let maxSize {
        return Gemma4SlidingKVCache.reencoded(keys: keys, values: values, offset: offset, maxSize: maxSize)
    }
    return Gemma4FullKVCache.reencoded(keys: keys, values: values, offset: offset)
}
