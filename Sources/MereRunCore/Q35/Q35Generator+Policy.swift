import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    /// Select a prefill width from the model, live machine headroom, and current
    /// scheduler contention. Q38's dense BF16 path must not infer activation
    /// headroom from total physical memory: model residency and concurrent work
    /// can consume most unified memory before prefill starts. The environment
    /// override remains an upper bound, not permission to ignore live pressure.
    static func prefillChunkSize(
        modelId: String,
        availableMemory: UInt64? = currentHostAvailableMemoryBytes(),
        activeRequestCount: Int = 1,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let configured = environment["MERERUN_Q35_PREFILL_CHUNK_TOKENS"]
            .flatMap(Int.init)
            .flatMap { (64...8192).contains($0) ? $0 : nil }
        let base = configured ?? 1_024
        if activeRequestCount > 1 {
            return min(base, contendedPrefillChunkSize)
        }
        if Q35Resources.isQ38ModelId(modelId),
           let availableMemory,
           availableMemory < q38LowPrefillHeadroomBytes {
            return min(base, contendedPrefillChunkSize)
        }
        return base
    }

    static func currentHostAvailableMemoryBytes() -> UInt64? {
        #if canImport(Darwin)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size) / 4
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pageSize = UInt64(getpagesize())
        let availablePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
        let availableBytes = availablePages.multipliedReportingOverflow(by: pageSize)
        return availableBytes.overflow ? UInt64.max : availableBytes.partialValue
        #else
        return nil
        #endif
    }

    static func shouldClearMLXCache(
        activeMemory: Int,
        cacheMemory: Int,
        memoryLimit: Int
    ) -> Bool {
        guard activeMemory >= 0,
              cacheMemory >= minimumReclaimableCacheBytes,
              memoryLimit > 0 else {
            return false
        }
        // Growing prompts leave buffers that later chunks cannot reuse.
        // The device-wide limit does not reserve headroom for other processes;
        // reclaim disposable buffers without evicting live prefix/KV state.
        if cacheMemory >= maximumReusableCacheBytes {
            return true
        }
        let total = activeMemory.addingReportingOverflow(cacheMemory)
        guard !total.overflow else { return true }
        return total.partialValue >= memoryLimit * 9 / 10
    }

    func clearMLXCacheUnderPressureIfNeeded() {
        let snapshot = Memory.snapshot()
        if Self.shouldClearMLXCache(
            activeMemory: snapshot.activeMemory,
            cacheMemory: snapshot.cacheMemory,
            memoryLimit: Memory.memoryLimit
        ) {
            Memory.clearCache()
        }
    }

    static func qwen3VLTargetSize(
        originalWidth width: Int,
        originalHeight height: Int,
        patchSize: Int,
        spatialMergeSize: Int,
        minPixels: Int = Q35Generator.qwen3VLMinPixels,
        maxPixels: Int = Q35Generator.qwen3VLMaxPixels
    ) throws -> (width: Int, height: Int) {
        let aspectRatio = Double(max(width, height)) / Double(min(width, height))
        guard aspectRatio <= 200 else {
            throw Q35Error.generationFailed(
                "Qwen-family image aspect ratio must not exceed 200; received \(aspectRatio)."
            )
        }
        let factor = max(1, patchSize * max(1, spatialMergeSize))

        func roundedToFactor(_ value: Int) -> Int {
            max(factor, Int((Double(value) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        }

        var targetHeight = roundedToFactor(height)
        var targetWidth = roundedToFactor(width)

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            targetHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (targetWidth, targetHeight)
    }

    static func visionTokenLimitPerImage(
        contextLength: Int,
        generationTokenCount: Int,
        nonVisionPromptTokenCount: Int,
        imageCount: Int
    ) -> Int? {
        guard contextLength > 0, imageCount > 0 else { return nil }
        let reservedGenerationTokens = min(contextLength, max(0, generationTokenCount))
        let availableVisionTokens = contextLength
            - reservedGenerationTokens
            - max(0, nonVisionPromptTokenCount)
        guard availableVisionTokens >= imageCount else { return nil }
        return availableVisionTokens / imageCount
    }

    static func visionPixelLimit(
        tokenLimit: Int,
        patchSize: Int,
        spatialMergeSize: Int,
        configuredMaximum: Int
    ) -> Int {
        let factor = max(1, patchSize * max(1, spatialMergeSize))
        let factorArea = factor * factor
        let requestedArea = max(1, tokenLimit).multipliedReportingOverflow(by: factorArea)
        let contextMaximum = requestedArea.overflow ? Int.max : requestedArea.partialValue
        return min(configuredMaximum, contextMaximum)
    }
}
