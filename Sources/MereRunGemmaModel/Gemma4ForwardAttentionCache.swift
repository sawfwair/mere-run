import MLX

/// Retains query positions and attention context for one model forward call.
/// Shared layers reuse the producer's context after its resident cache advances.
final class Gemma4ForwardAttentionCache: Gemma4AttentionCache {
    private let cache: Gemma4AttentionCache
    private var attentionContext: (MLXArray, MLXArray)?
    let offset: Int

    init(_ cache: Gemma4AttentionCache) {
        self.cache = cache
        self.offset = cache.offset
    }

    private init(cache: Gemma4AttentionCache, offset: Int, context: (MLXArray, MLXArray)?) {
        self.cache = cache
        self.offset = offset
        self.attentionContext = context
    }

    func currentState() -> (MLXArray, MLXArray)? {
        attentionContext ?? cache.currentState()
    }

    func decodeState() -> (MLXArray, MLXArray)? {
        attentionContext ?? cache.decodeState()
    }

    func append(keys: MLXArray, values: MLXArray) {
        if keys.dim(2) == 1 {
            // Specialized quantized decode must not materialize dense KV state.
            cache.append(keys: keys, values: values)
        } else {
            attentionContext = cache.attentionState(appending: keys, values: values)
        }
    }

    func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
        cache.specializedAttention(queries: queries, repeats: repeats, scale: scale)
    }

    func fork() -> Gemma4AttentionCache {
        Gemma4ForwardAttentionCache(cache: cache.fork(), offset: offset, context: attentionContext)
    }
}
