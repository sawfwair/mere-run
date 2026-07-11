import Foundation

enum MuScriptorBatchPolicy {
    static let gibibyte = UInt64(1_073_741_824)

    private static let minimumReserveBytes = 4 * gibibyte
    private static let bytesPerLayerDimension = UInt64(65_536)
    private static let maximumLiveBeamLanes = UInt64(32)
    // The large checkpoint stopped gaining throughput above eight live lanes
    // in the matched beam-search sweep documented in docs/runtime/music.md.
    private static let largeLiveBeamLanes = UInt64(8)
    private static let conservativeBytesPerParameter = UInt64(4)

    static func effectiveChunkBatchSize(
        requested: Int,
        configuration: MuScriptorConfiguration,
        beamSize: Int,
        physicalMemoryBytes: UInt64?,
        activeMemoryBytes: Int?,
        cacheMemoryBytes: Int?,
        memoryScale: Int = 1
    ) -> Int {
        let requested = max(1, requested)
        if requested == 1 { return 1 }

        let beamSize = max(1, beamSize)
        let liveLaneCapacity = maximumLiveLanes(configuration: configuration)
        let performanceCapacity = max(1, liveLaneCapacity / UInt64(beamSize))
        let performanceLimited = min(requested, Int(performanceCapacity))
        guard performanceLimited > 1,
              let physicalMemoryBytes,
              physicalMemoryBytes > 0 else {
            return performanceLimited
        }

        let reserveBytes = max(minimumReserveBytes, physicalMemoryBytes / 8)
        guard physicalMemoryBytes > reserveBytes else { return 1 }

        let currentAllocationBytes = observedMLXBytes(
            activeMemoryBytes: activeMemoryBytes,
            cacheMemoryBytes: cacheMemoryBytes
        ) ?? estimatedModelBytes(configuration: configuration)
        let usableBytes = physicalMemoryBytes - reserveBytes
        guard usableBytes > currentAllocationBytes else { return 1 }

        let bytesPerChunk = estimatedBytesPerChunk(
            configuration: configuration,
            beamSize: beamSize,
            memoryScale: memoryScale
        )
        guard bytesPerChunk > 0, bytesPerChunk < UInt64.max else { return 1 }

        let memoryCapacity = (usableBytes - currentAllocationBytes) / bytesPerChunk
        guard memoryCapacity > 0 else { return 1 }
        if memoryCapacity >= UInt64(performanceLimited) {
            return performanceLimited
        }
        return max(1, Int(memoryCapacity))
    }

    static func maximumLiveLanes(configuration: MuScriptorConfiguration) -> UInt64 {
        guard let complexity = product([
            positiveUInt64(configuration.numLayers),
            positiveUInt64(configuration.dim),
            positiveUInt64(configuration.dim),
        ]), complexity > 0 else {
            return 1
        }

        let largeComplexity = UInt64(48) * UInt64(1_536) * UInt64(1_536)
        let calibratedBudget = largeComplexity * largeLiveBeamLanes
        return min(maximumLiveBeamLanes, max(1, calibratedBudget / complexity))
    }

    static func estimatedBytesPerChunk(
        configuration: MuScriptorConfiguration,
        beamSize: Int,
        memoryScale: Int = 1
    ) -> UInt64 {
        product([
            positiveUInt64(configuration.numLayers),
            positiveUInt64(configuration.dim),
            bytesPerLayerDimension,
            positiveUInt64(beamSize),
            positiveUInt64(memoryScale),
        ]) ?? UInt64.max
    }

    static func estimatedModelBytes(configuration: MuScriptorConfiguration) -> UInt64 {
        // Each transformer layer has twelve dim-by-dim projection equivalents.
        guard let transformerParameters = product([
            positiveUInt64(configuration.numLayers),
            positiveUInt64(configuration.dim),
            positiveUInt64(configuration.dim),
            12,
        ]) else {
            return UInt64.max
        }

        // The remaining row equivalents cover layer norms, mel/class/token
        // embeddings, the output norm, and the untied output projection.
        guard let layerNormRows = product([
            positiveUInt64(configuration.numLayers),
            4,
        ]),
        let vocabularyRows = product([
            positiveUInt64(configuration.card),
            2,
        ]),
        let nonTransformerRows = sum([layerNormRows, vocabularyRows, 1_522]),
        let nonTransformerParameters = product([
            nonTransformerRows,
            positiveUInt64(configuration.dim),
        ]),
        let parameters = sum([transformerParameters, nonTransformerParameters]),
        let bytes = product([parameters, conservativeBytesPerParameter]) else {
            return UInt64.max
        }
        return bytes
    }

    private static func observedMLXBytes(
        activeMemoryBytes: Int?,
        cacheMemoryBytes: Int?
    ) -> UInt64? {
        let active = UInt64(max(0, activeMemoryBytes ?? 0))
        let cache = UInt64(max(0, cacheMemoryBytes ?? 0))
        guard let observed = sum([active, cache]), observed > 0 else { return nil }
        return observed
    }

    private static func positiveUInt64(_ value: Int) -> UInt64 {
        UInt64(max(1, value))
    }

    private static func product(_ factors: [UInt64]) -> UInt64? {
        var result = UInt64(1)
        for factor in factors {
            let multiplied = result.multipliedReportingOverflow(by: factor)
            guard !multiplied.overflow else { return nil }
            result = multiplied.partialValue
        }
        return result
    }

    private static func sum(_ values: [UInt64]) -> UInt64? {
        var result = UInt64(0)
        for value in values {
            let added = result.addingReportingOverflow(value)
            guard !added.overflow else { return nil }
            result = added.partialValue
        }
        return result
    }
}
