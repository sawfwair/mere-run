import MLX
#if os(macOS)
import Darwin
#endif

struct SOVBenchmarkProcessMemory: Encodable {
    let physicalFootprintBytes: UInt64
    let peakPhysicalFootprintBytes: UInt64
    let peakResidentBytes: Int

    static func snapshot() -> Self? {
        #if os(macOS)
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard status == 0 else { return nil }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return Self(physicalFootprintBytes: info.ri_phys_footprint,
            peakPhysicalFootprintBytes: info.ri_lifetime_max_phys_footprint, peakResidentBytes: usage.ru_maxrss)
        #else
        return nil
        #endif
    }
}

enum SOVBenchmarkResidency {
    static func budget(activeBytes: Int, recommended: Int?) -> Int {
        guard let recommended else { return 0 }
        return min(activeBytes + 4 * 1_073_741_824, max(0, recommended - 1_048_576))
    }

    static var recommendedBytes: Int? {
        #if os(macOS)
        GPU.maxRecommendedWorkingSetBytes()
        #else
        nil
        #endif
    }
}

struct SOVBenchmarkWiredPolicy: WiredMemoryPolicy, Hashable {
    func limit(baseline: Int, activeSizes: [Int]) -> Int {
        max(baseline, activeSizes.max() ?? baseline)
    }
}
