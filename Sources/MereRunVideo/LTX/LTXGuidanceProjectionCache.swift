import Foundation
import MereRunCore

struct LTXGuidanceProjectionCacheDecision: Equatable {
    let shouldCache: Bool
    let estimatedBytes: UInt64
    let reserveBytes: UInt64
}

func ltxGuidanceProjectionCacheDecision(
    mode: LTXGuidanceProjectionCacheMode,
    positivePredictionCount: Int,
    batchSize: Int,
    videoTextTokens: Int,
    audioTextTokens: Int,
    blockCount: Int,
    bytesPerElement: Int,
    activeMemoryBytes: UInt64,
    physicalMemoryBytes: UInt64
) -> LTXGuidanceProjectionCacheDecision {
    let videoElements = ltxSaturatingProduct([
        UInt64(max(0, batchSize)),
        UInt64(max(0, videoTextTokens)),
        4_096,
        2,
    ])
    let audioElements = ltxSaturatingProduct([
        UInt64(max(0, batchSize)),
        UInt64(max(0, audioTextTokens)),
        2_048,
        2,
    ])
    let (elementCount, elementOverflow) = videoElements.addingReportingOverflow(audioElements)
    let estimatedBytes = elementOverflow ? UInt64.max : ltxSaturatingProduct([
        elementCount,
        UInt64(max(0, blockCount)),
        UInt64(max(1, bytesPerElement)),
    ])
    let reserveBytes = max(8 * 1_073_741_824, physicalMemoryBytes / 10)
    let (activeAndCache, cacheOverflow) = activeMemoryBytes.addingReportingOverflow(estimatedBytes)
    let (requiredBytes, reserveOverflow) = activeAndCache.addingReportingOverflow(reserveBytes)
    let hasSafeHeadroom = !cacheOverflow
        && !reserveOverflow
        && requiredBytes <= physicalMemoryBytes
    let shouldCache = mode != .disabled
        && positivePredictionCount > 1
        && estimatedBytes > 0
        && hasSafeHeadroom
    return LTXGuidanceProjectionCacheDecision(
        shouldCache: shouldCache,
        estimatedBytes: estimatedBytes,
        reserveBytes: reserveBytes
    )
}

private func ltxSaturatingProduct(_ values: [UInt64]) -> UInt64 {
    var result: UInt64 = 1
    for value in values {
        let (product, overflow) = result.multipliedReportingOverflow(by: value)
        if overflow { return .max }
        result = product
    }
    return result
}
