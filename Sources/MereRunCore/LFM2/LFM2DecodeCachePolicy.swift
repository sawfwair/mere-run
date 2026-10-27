import MLX

/// Reclaim disposable allocations during long FP32 A2B decode without touching
/// live weights or prefix/KV state. Growing attention shapes cannot reuse every
/// buffer left by earlier steps.
enum LFM2DecodeCachePolicy {
    private static let minimumReclaimableBytes = 256 * 1_024 * 1_024
    private static let maximumReusableBytes = 1_024 * 1_024 * 1_024

    static func shouldClear(
        config: LFM2Config, activeMemory: Int, cacheMemory: Int, memoryLimit: Int
    ) -> Bool {
        guard config.isA2B24B, config.useFP32Activations,
              activeMemory >= 0, cacheMemory >= minimumReclaimableBytes,
              memoryLimit > 0 else { return false }
        if cacheMemory >= maximumReusableBytes { return true }
        let total = activeMemory.addingReportingOverflow(cacheMemory)
        let highWatermark = (memoryLimit / 10) * 9 + (memoryLimit % 10) * 9 / 10
        return total.overflow || total.partialValue >= highWatermark
    }

    static func clearIfNeeded(config: LFM2Config) {
        guard config.isA2B24B, config.useFP32Activations else { return }
        let snapshot = MLX.Memory.snapshot()
        if shouldClear(
            config: config, activeMemory: snapshot.activeMemory,
            cacheMemory: snapshot.cacheMemory, memoryLimit: MLX.Memory.memoryLimit
        ) {
            MLX.Memory.clearCache()
        }
    }
}
