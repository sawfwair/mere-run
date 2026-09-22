import MLX

extension LightOnOCRGenerator {
    /// Growing decode contexts leave temporary buffers that later tokens cannot reuse.
    /// Reclaim only unused allocations; keep weights, KV state, and allocator limits intact.
    static func reclaimUnusedDecodeBuffers(cacheThresholdBytes: Int = 1_024 * 1_024 * 1_024) {
        if Memory.cacheMemory >= cacheThresholdBytes {
            Memory.clearCache()
        }
    }
}
