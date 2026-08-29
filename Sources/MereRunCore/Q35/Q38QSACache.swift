import MLX
import MLXFast

/// The learned indexer's unpooled keys and rotary positions belong to the same
/// request snapshot as the target KV. Never keep them on the shared model.
final class Q38QSACache: KVCache {
    let attention: KVCache
    private let indexer: KVCache
    private var pooledKeys: MLXArray?
    private var poolingRatio: Int?

    var pooledBlockCount: Int { pooledKeys?.dim(2) ?? 0 }

    init(attention: KVCache = KVCacheSimple(), indexer: KVCache = KVCacheSimple()) {
        self.attention = attention
        self.indexer = indexer
    }

    var offset: Int { attention.offset }
    var rowOffsets: [Int]? { attention.rowOffsets }
    var supportsVariablePositionBatching: Bool { attention.supportsVariablePositionBatching }
    var canRollback: Bool { attention is KVCacheSimple && indexer is KVCacheSimple }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        attention.update(keys: keys, values: values)
    }

    func updateIndexer(keys: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
        indexer.update(keys: keys, values: positions)
    }

    /// Cache only complete blocks, as in mlx-serve's QSA path at 09970f9b
    /// (MIT, David Dalcu; see THIRD_PARTY_NOTICES.md). Ragged batches retain
    /// the full calculation because padding is replaced as each row advances.
    func compressedKeys(
        _ keys: MLXArray, positions: MLXArray, ratio: Int,
        compress: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        guard rowOffsets == nil else { return compress(keys, positions) }
        precondition(poolingRatio == nil || poolingRatio == ratio)
        poolingRatio = ratio
        let completed = keys.dim(2) / ratio
        let cached = pooledBlockCount
        precondition(cached <= completed, "QSA pooled cache must rewind with raw index keys")
        if completed > cached {
            let range = (cached * ratio)..<(completed * ratio)
            let added = compress(keys[0..., 0..., range, 0...], positions[0..., 0..., range, 0...])
            pooledKeys = pooledKeys.map { MLX.concatenated([$0, added], axis: 2) } ?? added
        }
        return pooledKeys ?? compress(keys, positions)
    }

    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        attention.makeMask(n: n)
    }

    func fork() -> KVCache {
        let copy = Q38QSACache(attention: attention.fork(), indexer: indexer.fork())
        copy.pooledKeys = pooledKeys.map { $0.reshaped($0.shape) }
        copy.poolingRatio = poolingRatio
        return copy
    }

    func rollback(toOffset offset: Int) {
        precondition(canRollback)
        (attention as? KVCacheSimple)?.rollback(toOffset: offset)
        (indexer as? KVCacheSimple)?.rollback(toOffset: offset)
        if let ratio = poolingRatio, let pooledKeys, offset / ratio < pooledBlockCount {
            self.pooledKeys = offset < ratio ? nil : pooledKeys[0..., 0..., 0..<(offset / ratio), 0...]
        }
    }

    func batched(with caches: [KVCache]) -> KVCache? {
        guard let caches = caches as? [Q38QSACache], !caches.isEmpty,
              let main = attention.batched(with: caches.map(\.attention)),
              let side = indexer.batched(with: caches.map(\.indexer)) else { return nil }
        return Q38QSACache(attention: main, indexer: side)
    }

    func unbatchedRows(count: Int) -> [KVCache]? {
        guard let main = attention.unbatchedRows(count: count),
              let side = indexer.unbatchedRows(count: count) else { return nil }
        return zip(main, side).map { Q38QSACache(attention: $0, indexer: $1) }
    }
}
