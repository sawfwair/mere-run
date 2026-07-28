import Foundation
#if canImport(Darwin)
import Darwin
#elseif os(Linux)
import Glibc
#endif
#if canImport(Metal)
import Metal
#endif

/// Honest process-level telemetry for the resident server. Values are optional where the host
/// does not expose a stable public API; in particular, this reports Metal allocation rather than
/// inventing a GPU-utilization percentage.
struct RuntimeProcessTelemetry: Codable, Equatable, Sendable {
    let processID: Int32
    let startedAt: Date
    let uptimeSeconds: Double
    let cpuPercent: Double?
    let thermalState: String?
    let lowPowerModeEnabled: Bool?
    let metalDeviceName: String?
    let metalCurrentAllocatedBytes: UInt64?
    let metalRecommendedMaxWorkingSetBytes: UInt64?
    let metalHasUnifiedMemory: Bool?
}

struct RuntimeProcessTelemetrySampler {
    private let startedAt: Date
    private let startedUptime: TimeInterval
    private var previousUptime: TimeInterval?
    private var previousCPUSeconds: Double?

    init(
        startedAt: Date = Date(),
        startedUptime: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) {
        self.startedAt = startedAt
        self.startedUptime = startedUptime
    }

    mutating func snapshot(
        processInfo: ProcessInfo = .processInfo
    ) -> RuntimeProcessTelemetry {
        let nowUptime = processInfo.systemUptime
        let cpuSeconds = Self.processCPUSeconds()
        let cpuPercent: Double?
        if let previousUptime,
           let previousCPUSeconds,
           let cpuSeconds,
           nowUptime > previousUptime {
            cpuPercent = max(0, (cpuSeconds - previousCPUSeconds) / (nowUptime - previousUptime) * 100)
        } else {
            cpuPercent = nil
        }
        previousUptime = nowUptime
        previousCPUSeconds = cpuSeconds

        let metal = Self.metalSnapshot()
        return RuntimeProcessTelemetry(
            processID: processInfo.processIdentifier,
            startedAt: startedAt,
            uptimeSeconds: max(0, nowUptime - startedUptime),
            cpuPercent: cpuPercent,
            thermalState: Self.thermalState(processInfo),
            lowPowerModeEnabled: Self.lowPowerModeEnabled(processInfo),
            metalDeviceName: metal.name,
            metalCurrentAllocatedBytes: metal.currentAllocatedBytes,
            metalRecommendedMaxWorkingSetBytes: metal.recommendedMaxWorkingSetBytes,
            metalHasUnifiedMemory: metal.hasUnifiedMemory
        )
    }

    private static func processCPUSeconds() -> Double? {
        #if canImport(Darwin)
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return nil }
        return Double(usage.ru_utime.tv_sec)
            + Double(usage.ru_utime.tv_usec) / 1_000_000
            + Double(usage.ru_stime.tv_sec)
            + Double(usage.ru_stime.tv_usec) / 1_000_000
        #elseif os(Linux)
        guard let stat = try? String(contentsOfFile: "/proc/self/stat"),
              let closingParenthesis = stat.lastIndex(of: ")") else {
            return nil
        }
        let fields = stat[stat.index(after: closingParenthesis)...]
            .split(whereSeparator: \.isWhitespace)
        guard fields.count > 12,
              let userTicks = Double(fields[11]),
              let systemTicks = Double(fields[12]) else {
            return nil
        }
        let ticksPerSecond = Double(sysconf(Int32(_SC_CLK_TCK)))
        guard ticksPerSecond > 0 else { return nil }
        return (userTicks + systemTicks) / ticksPerSecond
        #else
        return nil
        #endif
    }

    private static func thermalState(_ processInfo: ProcessInfo) -> String? {
        #if os(macOS)
        switch processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
        #else
        return nil
        #endif
    }

    private static func lowPowerModeEnabled(_ processInfo: ProcessInfo) -> Bool? {
        #if os(macOS)
        return processInfo.isLowPowerModeEnabled
        #else
        return nil
        #endif
    }

    private static func metalSnapshot() -> (
        name: String?,
        currentAllocatedBytes: UInt64?,
        recommendedMaxWorkingSetBytes: UInt64?,
        hasUnifiedMemory: Bool?
    ) {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return (nil, nil, nil, nil)
        }
        return (
            device.name,
            UInt64(device.currentAllocatedSize),
            UInt64(device.recommendedMaxWorkingSetSize),
            device.hasUnifiedMemory
        )
        #else
        return (nil, nil, nil, nil)
        #endif
    }
}
