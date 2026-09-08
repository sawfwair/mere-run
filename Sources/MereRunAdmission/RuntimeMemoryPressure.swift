import Foundation
#if canImport(Darwin)
import Darwin
#endif

public enum RuntimeMemoryPressureLevel: String, Codable, Equatable, Sendable {
    case disabled
    case unknown
    case nominal
    case elevated
    case critical
}

public enum RuntimeMemoryGuardTier: String, Codable, CaseIterable, Equatable, Sendable {
    case off
    case safe
    case balanced
    case aggressive
    case custom

    public static let `default` = RuntimeMemoryGuardTier.balanced
}

public struct RuntimeMemorySample: Equatable, Sendable {
    public let physicalBytes: UInt64
    public let residentBytes: UInt64?
    public let physicalFootprintBytes: UInt64?
    public let availableBytes: UInt64?
    public let freeBytes: UInt64?
    public let activeBytes: UInt64?
    public let inactiveBytes: UInt64?

    public init(
        physicalBytes: UInt64,
        residentBytes: UInt64?,
        physicalFootprintBytes: UInt64? = nil,
        availableBytes: UInt64? = nil,
        freeBytes: UInt64? = nil,
        activeBytes: UInt64? = nil,
        inactiveBytes: UInt64? = nil
    ) {
        self.physicalBytes = physicalBytes
        self.residentBytes = residentBytes
        self.physicalFootprintBytes = physicalFootprintBytes
        self.availableBytes = availableBytes
        self.freeBytes = freeBytes
        self.activeBytes = activeBytes
        self.inactiveBytes = inactiveBytes
    }

    public static func current() -> RuntimeMemorySample {
        RuntimeProcessMemory.currentSample()
    }
}

public struct RuntimeMemoryPressurePolicy: Equatable, Sendable {
    public let tier: RuntimeMemoryGuardTier
    public let customCeilingBytes: UInt64?
    public let softLimitFraction: Double
    public let hardLimitFraction: Double

    public static let `default` = RuntimeMemoryPressurePolicy(
        tier: .default
    )

    public init(
        tier: RuntimeMemoryGuardTier = .default,
        customCeilingBytes: UInt64? = nil,
        softLimitFraction: Double = 0.90,
        hardLimitFraction: Double = 0.95
    ) {
        self.tier = tier
        self.customCeilingBytes = customCeilingBytes
        self.softLimitFraction = softLimitFraction
        self.hardLimitFraction = hardLimitFraction
    }

    public func pressure(for sample: RuntimeMemorySample) -> RuntimeMemoryPressureLevel {
        guard tier != .off else {
            return .disabled
        }
        guard let currentBytes = currentBytes(for: sample),
              let limits = limits(for: sample) else {
            return .unknown
        }
        if currentBytes >= limits.hard {
            return .critical
        }
        if currentBytes >= limits.soft {
            return .elevated
        }
        return .nominal
    }

    public func projectedPressure(
        for sample: RuntimeMemorySample,
        additionalBytes: UInt64
    ) -> RuntimeMemoryPressureLevel {
        guard tier != .off else {
            return .disabled
        }
        guard let currentBytes = currentBytes(for: sample),
              let limits = limits(for: sample) else {
            return .unknown
        }
        let (projected, overflow) = currentBytes.addingReportingOverflow(additionalBytes)
        guard !overflow else {
            return .critical
        }
        if projected >= limits.hard {
            return .critical
        }
        if projected >= limits.soft {
            return .elevated
        }
        return .nominal
    }

    public func currentBytes(for sample: RuntimeMemorySample) -> UInt64? {
        sample.physicalFootprintBytes ?? sample.residentBytes
    }

    public func limits(for sample: RuntimeMemorySample) -> (ceiling: UInt64, soft: UInt64, hard: UInt64)? {
        guard tier != .off, sample.physicalBytes > 0 else {
            return nil
        }
        let ceiling: UInt64
        switch tier {
        case .off:
            return nil
        case .custom:
            let custom = customCeilingBytes ?? 0
            guard custom > 0 else {
                return nil
            }
            ceiling = min(custom, staticCeiling(for: sample))
        case .safe, .balanced, .aggressive:
            ceiling = min(staticCeiling(for: sample), dynamicCeiling(for: sample))
        }
        guard ceiling > 0 else {
            return nil
        }
        let soft = UInt64((Double(ceiling) * softLimitFraction).rounded(.down))
        let hard = UInt64((Double(ceiling) * hardLimitFraction).rounded(.down))
        return (ceiling, soft, hard)
    }

    private func staticCeiling(for sample: RuntimeMemorySample) -> UInt64 {
        let reserve = staticReserveBytes(forPhysicalBytes: sample.physicalBytes)
        guard sample.physicalBytes > reserve else {
            return 0
        }
        return sample.physicalBytes - reserve
    }

    private func dynamicCeiling(for sample: RuntimeMemorySample) -> UInt64 {
        guard let currentBytes = currentBytes(for: sample) else {
            return staticCeiling(for: sample)
        }
        if let freeBytes = sample.freeBytes,
           let inactiveBytes = sample.inactiveBytes,
           let activeBytes = sample.activeBytes {
            return currentBytes
                + freeBytes
                + inactiveBytes
                + UInt64((Double(activeBytes) * activeReclaimRatio).rounded(.down))
        }
        if let availableBytes = sample.availableBytes {
            return currentBytes + availableBytes
        }
        return sample.physicalBytes
    }

    private var activeReclaimRatio: Double {
        switch tier {
        case .off, .custom:
            return 0
        case .safe:
            return 0.20
        case .balanced:
            return 0.50
        case .aggressive:
            return 0.80
        }
    }

    private func staticReserveBytes(forPhysicalBytes physicalBytes: UInt64) -> UInt64 {
        let gib = UInt64(1024 * 1024 * 1024)
        let smallSystemThreshold = 24 * gib
        if physicalBytes < smallSystemThreshold {
            return 4 * gib
        }
        switch tier {
        case .off:
            return physicalBytes
        case .safe:
            return 8 * gib
        case .balanced:
            return 6 * gib
        case .aggressive:
            return 4 * gib
        case .custom:
            return 2 * gib
        }
    }
}

private enum RuntimeProcessMemory {
    static func currentSample() -> RuntimeMemorySample {
        let physicalBytes = ProcessInfo.processInfo.physicalMemory
        let residentBytes = currentResidentBytes()
        let physicalFootprintBytes = currentPhysicalFootprintBytes()
        let host = currentHostMemory()
        return RuntimeMemorySample(
            physicalBytes: physicalBytes,
            residentBytes: residentBytes,
            physicalFootprintBytes: physicalFootprintBytes,
            availableBytes: host.availableBytes,
            freeBytes: host.freeBytes,
            activeBytes: host.activeBytes,
            inactiveBytes: host.inactiveBytes
        )
    }

    private static func currentPhysicalFootprintBytes() -> UInt64? {
        #if os(macOS)
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            // The Darwin module imports `rusage_info_t *` as a pointer to an
            // optional raw pointer. Rebinding the struct storage mirrors the
            // C API's required `(rusage_info_t *)&info` cast; passing `&raw`
            // would instead let the kernel overwrite the pointer variable.
            pointer.withMemoryRebound(
                to: rusage_info_t?.self,
                capacity: 1
            ) { rebound in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, rebound)
            }
        }
        guard result == 0 else {
            return nil
        }
        return info.ri_phys_footprint
        #else
        return nil
        #endif
    }

    private static func currentResidentBytes() -> UInt64? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return UInt64(info.resident_size)
        #elseif os(Linux)
        guard let status = try? String(contentsOfFile: "/proc/self/status") else {
            return nil
        }
        for line in status.split(separator: "\n") where line.hasPrefix("VmRSS:") {
            let parts = line.split(separator: " ").compactMap { UInt64($0) }
            guard let kilobytes = parts.first else {
                return nil
            }
            return kilobytes * 1024
        }
        return nil
        #else
        return nil
        #endif
    }

    private static func currentHostMemory() -> (
        availableBytes: UInt64?,
        freeBytes: UInt64?,
        activeBytes: UInt64?,
        inactiveBytes: UInt64?
    ) {
        #if canImport(Darwin)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size) / 4
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return (nil, nil, nil, nil)
        }
        let pageSize = UInt64(getpagesize())
        let free = UInt64(stats.free_count) * pageSize
        let active = UInt64(stats.active_count) * pageSize
        let inactive = UInt64(stats.inactive_count) * pageSize
        return (free + inactive, free, active, inactive)
        #elseif os(Linux)
        guard let meminfo = try? String(contentsOfFile: "/proc/meminfo") else {
            return (nil, nil, nil, nil)
        }
        var values: [String: UInt64] = [:]
        for line in meminfo.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard let key = parts.first?.dropLast(),
                  let kilobytes = parts.dropFirst().compactMap({ UInt64($0) }).first else {
                continue
            }
            values[String(key)] = kilobytes * 1024
        }
        return (
            values["MemAvailable"],
            values["MemFree"],
            values["Active"],
            values["Inactive"]
        )
        #else
        return (nil, nil, nil, nil)
        #endif
    }
}
